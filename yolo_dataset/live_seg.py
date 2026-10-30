"""Real-time YOLO segmentation on a ZED camera, using the ROI and the latest trained model.

Keys (click the window first):
    0-9     toggle that class       v / h   show / hide all classes
    + / -   confidence +/- 0.05    t       toggle text
    c       toggle crop-only view   s       save the current frame (with detections) to live_captures/
    q / ESC quit
Usage:
    python live_seg.py [--serial N] [--weights best.pt] [--conf 0.25]
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dataset_generation import open_camera  # noqa: E402
from roi import ROI_FILE, crop, load_roi  # noqa: E402

RUNS = HERE.parent / "runs" / "segment"
COLORS = [(0, 255, 0), (0, 128, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
          (128, 0, 255), (0, 200, 120), (200, 120, 0), (120, 120, 255)]


def latest_weights() -> Path:
    cands = sorted(RUNS.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit(f"no best.pt found under {RUNS}")
    return cands[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", type=int, default=None)
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--weights", default=None, help="default: most recent run")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=None, help="default: longest side of the ROI")
    ap.add_argument("--resolution", choices=["HD2K", "HD1080", "HD720", "VGA"], default="HD1080")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--max-width", type=int, default=1400)
    ap.add_argument("--save-dir", default=str(HERE / "live_captures"))
    args = ap.parse_args()

    if not ROI_FILE.exists():
        raise SystemExit(f"{ROI_FILE} not found: run 'python roi.py select' first")
    roi = load_roi()
    x0, y0, rw, rh = roi
    imgsz = args.imgsz or max(rw, rh)

    weights = Path(args.weights) if args.weights else latest_weights()
    print(f"weights={weights} imgsz={imgsz}")
    model = YOLO(str(weights))
    names = model.names
    n_classes = len(names)
    model.predict(np.zeros((rh, rw, 3), np.uint8), imgsz=imgsz, verbose=False)  # warm-up

    zed = open_camera(args.serial, args.id, getattr(sl.RESOLUTION, args.resolution), args.fps)
    mat = sl.Mat()
    runtime = sl.RuntimeParameters()

    enabled = set(range(n_classes))
    conf, show_text, crop_only = args.conf, True, False
    fps, n, t_fps = 0.0, 0, time.time()
    win = "live segmentation"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                cv2.waitKey(1)
                continue
            zed.retrieve_image(mat, sl.VIEW.LEFT)
            frame = cv2.cvtColor(mat.get_data(), cv2.COLOR_BGRA2BGR)

            r = model.predict(crop(frame, roi), imgsz=imgsz, conf=conf, verbose=False)[0]

            # Draw on the crop or on the full frame (shifted by the ROI offset).
            canvas = crop(frame, roi).copy() if crop_only else frame.copy()
            off = np.array([0, 0] if crop_only else [x0, y0])
            overlay = canvas.copy()
            shown = {}
            if r.masks is not None:
                for poly, c, s in zip(r.masks.xy, r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                    if c not in enabled or len(poly) < 3:
                        continue
                    pts = (poly + off).astype(np.int32)
                    color = COLORS[c % len(COLORS)]
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.polylines(canvas, [pts], True, color, 2)
                    shown[c] = shown.get(c, 0) + 1
                    if show_text:
                        px, py = pts.min(axis=0)
                        cv2.putText(canvas, f"{names[c]} {s:.2f}", (int(px), max(int(py) - 4, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
            if not crop_only:
                cv2.rectangle(canvas, (x0, y0), (x0 + rw, y0 + rh), (255, 255, 255), 2)

            for c in range(n_classes):
                on = c in enabled
                color = COLORS[c % len(COLORS)] if on else (110, 110, 110)
                cv2.putText(canvas, f"[{c}] {names[c]}: {shown.get(c, 0) if on else 'off'}", (10, 25 + 24 * c),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(canvas, f"{fps:.1f} FPS  conf>={conf:.2f}", (10, canvas.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            n += 1
            if time.time() - t_fps >= 1.0:
                fps, n, t_fps = n / (time.time() - t_fps), 0, time.time()
                print(f"{fps:.1f} FPS, {sum(shown.values())} objects")

            scale = min(1.0, args.max_width / canvas.shape[1])
            cv2.imshow(win, cv2.resize(canvas, None, fx=scale, fy=scale) if scale < 1 else canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if ord("0") <= key <= ord("9") and key - ord("0") < n_classes:
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
            elif key == ord("c"):
                crop_only = not crop_only
            elif key == ord("s"):
                out = Path(args.save_dir)
                out.mkdir(parents=True, exist_ok=True)
                path = out / f"live_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(path), canvas)
                print(f"saved {path}")
    finally:
        zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
