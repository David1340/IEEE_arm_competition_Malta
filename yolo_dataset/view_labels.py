"""View YOLO segmentation labels over the images.

Keys (click the window first):
    d / RIGHT = next image       a / LEFT = previous image
    0-9 = toggle class           v = show all classes    h = hide all classes
    q / ESC = quit.
Usage:
    python view_labels.py [--images images] [--labels labels]
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png"}
COLORS = [(0, 255, 0), (0, 128, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]


def draw(img, label_path, classes, visible):
    h, w = img.shape[:2]
    overlay = img.copy()
    n = 0
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 7:
                continue
            c = int(parts[0])
            if c not in visible:
                continue
            xy = np.array(parts[1:], dtype=float).reshape(-1, 2) * [w, h]
            pts = xy.astype(np.int32)
            color = COLORS[c % len(COLORS)]
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(img, [pts], True, color, 3)
            name = classes[c] if c < len(classes) else str(c)
            cv2.putText(img, name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            n += 1
    return cv2.addWeighted(overlay, 0.35, img, 0.65, 0), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(HERE / "images"))
    ap.add_argument("--labels", default=str(HERE / "labels"))
    ap.add_argument("--classes-file", default=str(HERE / "classes.txt"))
    ap.add_argument("--max-width", type=int, default=1280)
    args = ap.parse_args()

    classes = [c.strip() for c in Path(args.classes_file).read_text().splitlines() if c.strip()]
    files = sorted(f for f in Path(args.images).iterdir() if f.suffix.lower() in IMG_EXT)
    if not files:
        raise SystemExit("no images found")

    win = "labels"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    i = 0
    visible = set(range(len(classes)))
    while True:
        f = files[i]
        vis, n = draw(cv2.imread(str(f)), Path(args.labels) / f"{f.stem}.txt", classes, visible)
        scale = min(1.0, args.max_width / vis.shape[1])
        vis = cv2.resize(vis, None, fx=scale, fy=scale)
        cv2.putText(vis, f"[{i + 1}/{len(files)}] {f.name} visible objects={n}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        for c, name in enumerate(classes):
            state = "ON" if c in visible else "OFF"
            color = COLORS[c % len(COLORS)] if c in visible else (120, 120, 120)
            cv2.putText(vis, f"[{c}] {name}: {state}", (10, 60 + c * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.imshow(win, vis)

        key = cv2.waitKeyEx(0)
        if key in (ord("q"), 27):
            break
        if key in (ord("d"), 65363):
            i = min(i + 1, len(files) - 1)
        elif key in (ord("a"), 65361):
            i = max(i - 1, 0)
        elif ord("0") <= key <= ord("9") and key - ord("0") < len(classes):
            c = key - ord("0")
            if c in visible:
                visible.remove(c)
            else:
                visible.add(c)
        elif key == ord("v"):
            visible = set(range(len(classes)))
        elif key == ord("h"):
            visible.clear()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
