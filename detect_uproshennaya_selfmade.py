import argparse
import csv
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np


BG_WINDOW = 15  # размер окна для фона
MIN_CONTRAST = 15  # насколько точка ярче фона
MAX_TARGET_SIZE = 9  # максимальный размер цели, пикс.
MIN_PIXELS = 2  # минимум пикселей в цели

NORM_LOW, NORM_HIGH = 1.0, 99.9  # границы нормализации, проценты

IMAGE_EXT = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}  # допустимые форматы снимков


def read_gray(path):
    # чтение снимка в оттенках серого
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # чтение без потери разрядности
    if img is None:
        raise ValueError("не удалось прочитать файл")
    if img.ndim == 3:  # если снимок цветной
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)  # перевод в серый
    return img


def normalize(raw):
    # яркость в шкале 0-255
    lo, hi = np.percentile(raw, (NORM_LOW, NORM_HIGH))  # границы
    scale = 255.0 / max(hi - lo, 1e-6)  # коэффициент растяжения
    return (raw.astype(np.float32) - lo) * scale  # растяжение яркости


def to_view(raw):
    # 8-битная копия
    return np.clip(normalize(raw), 0, 255).astype(np.uint8)  # обрезка до 0-255


def find_candidate_pixels(img):
    background = cv2.blur(img, (BG_WINDOW, BG_WINDOW))  # средняя яркость вокруг
    contrast = img - background  # насколько точка ярче фона
    return contrast > MIN_CONTRAST  # маска ярких точек


def find_blobs(mask):
    # склейка соседних точек в объекты
    h, w = mask.shape  # высота и ширина
    visited = np.zeros_like(mask, dtype=bool)  # отметки посещённых точек
    blobs = []  # список всех объектов

    for start_y in range(h):  # перебор всех строк
        for start_x in range(w):  # перебор всех столбцов
            if not mask[start_y, start_x] or visited[start_y, start_x]:
                continue  # пропуск тусклых и пройденных

            blob = []  # новый объект
            queue = deque([(start_y, start_x)])  # очередь точек для обхода
            visited[start_y, start_x] = True  # отметка старта

            while queue:  # пока в очереди есть точки
                y, x = queue.popleft()  # взятие следующей точки
                blob.append((y, x))  # добавление в объект
                for dy in (-1, 0, 1):  # просмотр соседей по высоте и ширине
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue  # пропуск самой точки
                        ny, nx = y + dy, x + dx  # координаты соседа
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:  # сосед яркий и новый
                            visited[ny, nx] = True  # отметка как посещённого
                            queue.append((ny, nx))  # добавление в очередь

            blobs.append(blob)  # сохранение готового объекта

    return blobs


def detect(raw):
    # поиск целей на одном снимке
    img = normalize(raw)
    mask = find_candidate_pixels(img)  # маска
    blobs = find_blobs(mask)  # группы точек

    results = []  # итоговый список целей
    for blob in blobs:
        ys = [p[0] for p in blob]
        xs = [p[1] for p in blob]
        y1, y2 = min(ys), max(ys)
        x1, x2 = min(xs), max(xs)
        width = x2 - x1 + 1
        height = y2 - y1 + 1

        if width > MAX_TARGET_SIZE or height > MAX_TARGET_SIZE:
            continue
        if len(blob) < MIN_PIXELS:
            continue

        peak = int(raw[ys, xs].max())  # максимальная яркость объекта
        mean = float(raw[ys, xs].mean())  # средняя яркость объекта

        # область вокруг объекта для фона
        y1b, y2b = max(0, y1 - 5), min(raw.shape[0], y2 + 6)  # границы по y
        x1b, x2b = max(0, x1 - 5), min(raw.shape[1], x2 + 6)  # границы по x
        background = float(np.median(raw[y1b:y2b, x1b:x2b]))  # яркость фона

        results.append({  # сохранение данных цели
            "x": xs[0], "y": ys[0],
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "width": width, "height": height, "pixels": len(blob),
            "peak": peak, "mean": round(mean, 1),
            "background": round(background, 1),
            "contrast": round(mean - background, 1),
        })

    results.sort(key=lambda d: -d["contrast"])  # сначала самые контрастные
    return results


def draw(raw, dets):
    # отрисовка рамок
    vis = cv2.cvtColor(to_view(raw), cv2.COLOR_GRAY2BGR)  # серый снимок в цветной
    for i, d in enumerate(dets, 1):  # перебор целей
        cv2.rectangle(vis, (d["x1"] - 4, d["y1"] - 4), (d["x2"] + 4, d["y2"] + 4), (0, 0, 255), 1)  # красная рамка
        cv2.putText(vis, str(i), (d["x2"] + 6, d["y1"]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)  # номер цели
    return vis


def main():
    ap = argparse.ArgumentParser(description="Поиск малых объектов на ИК-снимках (упрощённый метод)")  # настройка аргументов
    ap.add_argument("input", help="файл снимка или папка со снимками")  # путь ко входу
    ap.add_argument("-o", "--out", default="results", help="папка для результатов (по умолчанию results)")  # папка результатов
    args = ap.parse_args()  # чтение аргументов

    src = Path(args.input)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXT) if src.is_dir() else [src]  # список снимков
    if not files:  # если снимков нет
        sys.exit(f"В {src} нет снимков ({', '.join(sorted(IMAGE_EXT))})")
    out = Path(args.out)  # папка результатов
    out.mkdir(parents=True, exist_ok=True)  # создание папки

    fields = ["file", "n", "x", "y", "x1", "y1", "x2", "y2", "width", "height",
              "pixels", "peak", "mean", "background", "contrast"]  # столбцы таблицы
    with open(out / "detections.csv", "w", newline="", encoding="utf-8") as fcsv:  # открытие CSV
        writer = csv.DictWriter(fcsv, fieldnames=fields)  # писатель таблицы
        writer.writeheader()  # запись заголовка
        for path in files:  # обработка каждого снимка
            try:
                raw = read_gray(path)  # чтение снимка
            except ValueError as e:
                print(f"{path.name}: {e}")  # сообщение об ошибке
                continue
            dets = detect(raw)  # поиск целей
            cv2.imwrite(str(out / f"{path.stem}_boxes.png"), draw(raw, dets))  # сохранение снимка с рамками
            for i, d in enumerate(dets, 1):  # перебор найденных целей
                writer.writerow({"file": path.name, "n": i, **d})  # запись строки в таблицу
            print(f"{path.name}: {raw.dtype}, {raw.shape[1]}x{raw.shape[0]}, найдено объектов: {len(dets)}")  # итог по снимку
    print(f"Готово. Результаты в папке {out.resolve()}")


if __name__ == "__main__":
    main()