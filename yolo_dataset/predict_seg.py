"""Segment every image in a folder using the ROI crop and the trained model.

Usage:
    python predict_seg.py                              # latest run, images/ -> predictions/
    python predict_seg.py --weights path/to/best.pt --source some_folder --conf 0.25
    python predict_seg.py --save-txt                   # also write YOLO label files
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from roi import ROI_FILE, crop, load_roi

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs" / "segment"
IMG_EXT = {".jpg", ".jpeg", ".png"}


def latest_weights() -> Path:
    cands = sorted(RUNS.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit(f"no best.pt found under {RUNS}")
    return cands[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="default: most recent run in runs/segment")
    ap.add_argument("--source", default=str(HERE / "images"), help="folder of full-resolution images")
    ap.add_argument("--out", default=str(HERE / "predictions"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=None, help="default: longest side of the ROI")
    ap.add_argument("--no-roi", action="store_true", help="images are already cropped")
    ap.add_argument("--save-txt", action="store_true")
    args = ap.parse_args()

    roi = None
    if not args.no_roi:
        if not ROI_FILE.exists():
            raise SystemExit(f"{ROI_FILE} not found: run 'python roi.py select' or pass --no-roi")
        roi = load_roi()
    imgsz = args.imgsz or (max(roi[2], roi[3]) if roi else 1024)

    weights = Path(args.weights) if args.weights else latest_weights()
    print(f"weights={weights} imgsz={imgsz} conf={args.conf}")
    model = YOLO(str(weights))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in Path(args.source).iterdir() if f.suffix.lower() in IMG_EXT)

    total = 0
    for f in files:
        img = cv2.imread(str(f))
        if roi:
            img = crop(img, roi)
        r = model.predict(img, imgsz=imgsz, conf=args.conf, verbose=False)[0]
        cv2.imwrite(str(out / f.name), r.plot())
        names = [model.names[int(c)] for c in r.boxes.cls]
        total += len(names)
        if args.save_txt and r.masks is not None:
            lines = [
                f"{int(c)} " + " ".join(f"{v:.6f}" for v in poly.flatten())
                for c, poly in zip(r.boxes.cls, r.masks.xyn)
                if len(poly) >= 3
            ]
            (out / f"{f.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"{f.name}: {len(names)} objects")
    print(f"{len(files)} images, {total} objects -> {out}")


if __name__ == "__main__":
    main()
