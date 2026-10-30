"""Prepare the ROI dataset and train YOLO segmentation at the ROI's native resolution.

imgsz is taken from roi.json (longest side of the crop), so the crop is not downscaled.

Usage:
    python train_seg.py                       # roi apply -> split -> train
    python train_seg.py --skip-prepare        # train on the existing dataset/ only
    python train_seg.py --max-imgsz 1280      # cap the size if the GPU runs out of memory
"""

import argparse
import subprocess
import sys
from pathlib import Path

from roi import ROI_FILE, load_roi

HERE = Path(__file__).resolve().parent


def run(cmd):
    print("+", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n-seg.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = automatic from GPU memory")
    ap.add_argument("--max-imgsz", type=int, default=None)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--skip-prepare", action="store_true")
    ap.add_argument("--project", default=str(HERE.parent / "runs" / "segment"))
    args = ap.parse_args()

    if not ROI_FILE.exists():
        raise SystemExit(f"{ROI_FILE} not found: run 'python roi.py select' first")
    _, _, w, h = load_roi()
    imgsz = max(w, h)
    if args.max_imgsz:
        imgsz = min(imgsz, args.max_imgsz // 32 * 32)
    print(f"ROI {w}x{h} -> imgsz={imgsz}")

    if not args.skip_prepare:
        py = sys.executable
        run([py, str(HERE / "roi.py"), "apply"])
        run([py, str(HERE / "prepare_dataset.py"), "--val", str(args.val),
             "--images", str(HERE / "images_roi"), "--labels", str(HERE / "labels_roi")])

    from ultralytics import YOLO

    YOLO(args.model).train(
        data=str(HERE / "data.yaml"),
        epochs=args.epochs,
        imgsz=imgsz,
        batch=args.batch,
        project=args.project,
    )


if __name__ == "__main__":
    main()
