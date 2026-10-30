"""Browse images and inspect the YOLO segmentation, toggling each class on/off.

Keys (click the window first):
    d / RIGHT   next image          a / LEFT   previous image
    0-9         toggle that class   v          show all classes     h  hide all classes
    + / -       confidence threshold +/- 0.05   t  toggle text (name + confidence)
    q / ESC     quit
Usage:
    python segview.py [--weights best.pt] [--source images] [--no-roi]
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np
from ultralytics import YOLO

from roi import ROI_FILE, crop, load_roi

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs" / "segment"
IMG_EXT = {".jpg", ".jpeg", ".png"}
COLORS = [(0, 255, 0), (0, 128, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
          (128, 0, 255), (0, 200, 120), (200, 120, 0), (120, 120, 255)]


def latest_weights() -> Path:
    cands = sorted(RUNS.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit(f"no best.pt found under {RUNS}")
    return cands[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="default: most recent run")
    ap.add_argument("--split", choices=["val", "train"], default="val", help="split from dataset/ (already cropped)")
    ap.add_argument("--source", default=None, help="full-resolution folder; overrides --split and applies the ROI")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=None, help="default: longest side of the ROI")
    ap.add_argument("--no-roi", action="store_true", help="images in --source are already cropped")
    ap.add_argument("--max-width", type=int, default=1400)
    args = ap.parse_args()

    # The dataset/ split already holds cropped images, so the ROI is applied only to --source.
    if args.source is None:
        args.source = str(HERE / "dataset" / "images" / args.split)
        args.no_roi = True
    if not ROI_FILE.exists():
        raise SystemExit(f"{ROI_FILE} not found: run 'python roi.py select'")
    roi = None if args.no_roi else load_roi()
    imgsz = args.imgsz or max(load_roi()[2:])

    weights = Path(args.weights) if args.weights else latest_weights()
    print(f"weights={weights} imgsz={imgsz}")
    model = YOLO(str(weights))
    names = model.names
    n_classes = len(names)

    files = sorted(f for f in Path(args.source).iterdir() if f.suffix.lower() in IMG_EXT)
    if not files:
        raise SystemExit(f"no images found in {args.source}")
    print(f"{len(files)} images from {args.source}")

    cache = {}  # filename -> (img, [(cls, conf, polygon_px)])  predicted once with a low threshold

    def predict(f):
        if f not in cache:
            img = cv2.imread(str(f))
            if roi:
                img = crop(img, roi)
            r = model.predict(img, imgsz=imgsz, conf=0.05, verbose=False)[0]
            dets = []
            if r.masks is not None:
                for poly, c, s in zip(r.masks.xy, r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                    if len(poly) >= 3:
                        dets.append((c, s, poly.astype(np.int32)))
            cache[f] = (img, dets)
        return cache[f]

    enabled = set(range(n_classes))
    conf = args.conf
    show_text = True
    i = 0
    win = "segmentation"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        img, dets = predict(files[i])
        vis = img.copy()
        overlay = img.copy()
        shown = {}
        for c, s, poly in dets:
            if c not in enabled or s < conf:
                continue
            color = COLORS[c % len(COLORS)]
            cv2.fillPoly(overlay, [poly], color)
            cv2.polylines(vis, [poly], True, color, 2)
            shown[c] = shown.get(c, 0) + 1
            if show_text:
                x, y = poly.min(axis=0)
                cv2.putText(vis, f"{names[c]} {s:.2f}", (int(x), max(int(y) - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

        # Legend: one line per class with on/off state and visible count.
        for c in range(n_classes):
            on = c in enabled
            color = COLORS[c % len(COLORS)] if on else (110, 110, 110)
            cv2.putText(vis, f"[{c}] {names[c]}: {shown.get(c, 0) if on else 'off'}", (10, 25 + 24 * c),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(vis, f"[{i + 1}/{len(files)}] {files[i].name}  conf>={conf:.2f}  objects={sum(shown.values())}",
                    (10, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        scale = min(1.0, args.max_width / vis.shape[1])
        cv2.imshow(win, cv2.resize(vis, None, fx=scale, fy=scale) if scale < 1 else vis)

        key = cv2.waitKeyEx(0)
        if key in (ord("q"), 27):
            break
        if key in (ord("d"), 65363):
            i = min(i + 1, len(files) - 1)
        elif key in (ord("a"), 65361):
            i = max(i - 1, 0)
        elif ord("0") <= key <= ord("9") and key - ord("0") < n_classes:
            enabled ^= {key - ord("0")}
        elif key == ord("v"):
            enabled = set(range(n_classes))
        elif key == ord("h"):
            enabled = set()
        elif key in (ord("+"), ord("=")):
            conf = min(0.95, conf + 0.05)
        elif key in (ord("-"), ord("_")):
            conf = max(0.05, conf - 0.05)
        elif key == ord("t"):
            show_text = not show_text
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
