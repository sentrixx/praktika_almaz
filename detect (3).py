"""Поиск малых ярких объектов на ИК-снимках (упрощённый метод).

Идея (по методу RUMFR из статьи, но без точных формул оттуда - здесь
упрощённая версия той же идеи, чтобы было проще разобраться и объяснить
в отчёте):
    1. Для каждой точки снимка смотрим, насколько она ярче фона вокруг
       себя. Фон - это средняя яркость в окне побольше (BG_WINDOW).
       Если точка заметно ярче - она кандидат в цель.
    2. Соседние точки-кандидаты объединяем в один объект (одна цель
       почти всегда состоит не из одного пикселя, а из нескольких).
    3. Если объект получился больше 9x9 пикселей - это не цель (скорее
       облако, край крыши, засветка), и мы его отбрасываем.

Запуск:
    pip install numpy opencv-python
    python detect.py снимки/            # все снимки в папке
    python detect.py кадр.tif           # один снимок
    python detect.py снимки/ -o итог/   # своя папка для результатов

Вход: 16-битные PNG/TIFF (основной случай) или обычные 8-битные картинки.
Выход (по умолчанию в папке results/):
    <имя>_boxes.png   копия снимка с пронумерованными рамками
    detections.csv    таблица: файл, номер, координаты, размер, яркость
"""
import argparse
import csv
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np


BG_WINDOW = 15      # размер окна для оценки фона, пикс. Должен быть заметно
                     # больше цели (цель до 9x9), иначе фон будет "заражён"
                     # яркостью самой цели.
MIN_CONTRAST = 15    # на сколько точка должна быть ярче фона (в шкале 0..255),
                      # чтобы считаться кандидатом. Больше -> меньше находок
                      # (и меньше ложных), меньше -> больше находок.
MAX_TARGET_SIZE = 9   # цель не больше MAX_TARGET_SIZE x MAX_TARGET_SIZE
                       # пикселей - крупнее отбрасываем.
MIN_PIXELS = 2         # у настоящей цели почти всегда больше одного яркого
                       # пикселя; объекты из 1 пикселя слишком часто
                       # оказываются случайным шумом - отбрасываем и их.

NORM_LOW, NORM_HIGH = 1.0, 99.9  # процентили, которые станут 0 и 255 при нормализации

IMAGE_EXT = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}
# ====================================================================


def read_gray(path):
    """Читает снимок без потери разрядности; цветной переводит в серый."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("не удалось прочитать файл")
    if img.ndim == 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    return img


def normalize(raw):
    """Переводит яркость в диапазон 0..255 (без потери точности 16 бит).
    Берём не минимум и максимум, а процентили 1% и 99.9% - так один
    случайный супер-яркий пиксель-шум не "сплющит" всю шкалу."""
    lo, hi = np.percentile(raw, (NORM_LOW, NORM_HIGH))
    scale = 255.0 / max(hi - lo, 1e-6)
    return (raw.astype(np.float32) - lo) * scale


def to_view(raw):
    """8-битная копия только для просмотра (рисования рамок)."""
    return np.clip(normalize(raw), 0, 255).astype(np.uint8)


def find_candidate_pixels(img):
    """Шаг 1: находит пиксели, которые заметно ярче фона вокруг них.

    cv2.blur(img, (N, N)) для каждой точки считает среднюю яркость в
    окне N x N вокруг неё - это и есть "фон". Если сама точка ярче
    фона больше, чем на MIN_CONTRAST - она кандидат.
    Возвращает булеву маску (True/False) того же размера, что img."""
    background = cv2.blur(img, (BG_WINDOW, BG_WINDOW))
    contrast = img - background
    return contrast > MIN_CONTRAST


def find_blobs(mask):
    """Шаг 2: группирует соседние точки-кандидаты в отдельные объекты.

    Это обход в ширину (BFS), он же "разлив краски": берём любую
    точку-кандидата, которую ещё не посещали, и находим всех её
    соседей-кандидатов, потом соседей соседей и так далее - все они
    становятся одним объектом. Соседними считаем 8 точек вокруг
    (включая диагонали).
    Возвращает список объектов; каждый объект - список координат
    (y, x) своих пикселей."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    blobs = []

    for start_y in range(h):
        for start_x in range(w):
            if not mask[start_y, start_x] or visited[start_y, start_x]:
                continue

            blob = []
            queue = deque([(start_y, start_x)])
            visited[start_y, start_x] = True

            while queue:
                y, x = queue.popleft()
                blob.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))

            blobs.append(blob)

    return blobs


def detect(raw):
    """Полный метод для одного снимка. Возвращает список найденных объектов."""
    img = normalize(raw)
    mask = find_candidate_pixels(img)
    blobs = find_blobs(mask)

    results = []
    for blob in blobs:
        ys = [p[0] for p in blob]
        xs = [p[1] for p in blob]
        y1, y2 = min(ys), max(ys)
        x1, x2 = min(xs), max(xs)
        width = x2 - x1 + 1
        height = y2 - y1 + 1

        # Шаг 3: слишком большое пятно - не цель, пропускаем.
        # Слишком маленькое (один случайный пиксель) - тоже пропускаем.
        if width > MAX_TARGET_SIZE or height > MAX_TARGET_SIZE:
            continue
        if len(blob) < MIN_PIXELS:
            continue

        peak = int(raw[ys, xs].max())
        mean = float(raw[ys, xs].mean())

        # фон вокруг объекта (кольцо шириной 5 пикселей) - для отчёта
        y1b, y2b = max(0, y1 - 5), min(raw.shape[0], y2 + 6)
        x1b, x2b = max(0, x1 - 5), min(raw.shape[1], x2 + 6)
        background = float(np.median(raw[y1b:y2b, x1b:x2b]))

        results.append({
            "x": xs[0], "y": ys[0],
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "width": width, "height": height, "pixels": len(blob),
            "peak": peak, "mean": round(mean, 1),
            "background": round(background, 1),
            "contrast": round(mean - background, 1),
        })

    # сортируем по контрасту - самые заметные объекты будут первыми
    results.sort(key=lambda d: -d["contrast"])
    return results


def draw(raw, dets):
    vis = cv2.cvtColor(to_view(raw), cv2.COLOR_GRAY2BGR)
    for i, d in enumerate(dets, 1):
        cv2.rectangle(vis, (d["x1"] - 4, d["y1"] - 4), (d["x2"] + 4, d["y2"] + 4), (0, 0, 255), 1)
        cv2.putText(vis, str(i), (d["x2"] + 6, d["y1"]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return vis


def main():
    ap = argparse.ArgumentParser(description="Поиск малых объектов на ИК-снимках (упрощённый метод)")
    ap.add_argument("input", help="файл снимка или папка со снимками")
    ap.add_argument("-o", "--out", default="results", help="папка для результатов (по умолчанию results)")
    args = ap.parse_args()

    src = Path(args.input)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXT) if src.is_dir() else [src]
    if not files:
        sys.exit(f"В {src} нет снимков ({', '.join(sorted(IMAGE_EXT))})")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fields = ["file", "n", "x", "y", "x1", "y1", "x2", "y2", "width", "height",
              "pixels", "peak", "mean", "background", "contrast"]
    with open(out / "detections.csv", "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fields)
        writer.writeheader()
        for path in files:
            try:
                raw = read_gray(path)
            except ValueError as e:
                print(f"{path.name}: {e}")
                continue
            dets = detect(raw)
            cv2.imwrite(str(out / f"{path.stem}_boxes.png"), draw(raw, dets))
            for i, d in enumerate(dets, 1):
                writer.writerow({"file": path.name, "n": i, **d})
            print(f"{path.name}: {raw.dtype}, {raw.shape[1]}x{raw.shape[0]}, найдено объектов: {len(dets)}")
    print(f"Готово. Результаты в папке {out.resolve()}")


if __name__ == "__main__":
    main()
