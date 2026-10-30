"""Split yolo_dataset into train/val and write data.yaml for YOLO segmentation.

Usage:
    python prepare_dataset.py --val 0.2
Then train:
    yolo segment train data=yolo_dataset/data.yaml model=yolo11n-seg.pt epochs=100 imgsz=640
"""

import argparse
import random
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=float, default=0.2, help="validation fraction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "dataset"))
    ap.add_argument("--images", default=str(HERE / "images"), help="use images_roi after 'roi.py apply'")
    ap.add_argument("--labels", default=str(HERE / "labels"), help="use labels_roi after 'roi.py apply'")
    args = ap.parse_args()

    classes = [c.strip() for c in (HERE / "classes.txt").read_text().splitlines() if c.strip()]
    lbl_dir = Path(args.labels)
    imgs = sorted(f for f in Path(args.images).iterdir() if f.suffix.lower() in IMG_EXT)
    imgs = [f for f in imgs if (lbl_dir / f"{f.stem}.txt").exists()]
    random.Random(args.seed).shuffle(imgs)

    n_val = max(1, round(len(imgs) * args.val))
    splits = {"val": imgs[:n_val], "train": imgs[n_val:]}

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for split, files in splits.items():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        for f in files:
            shutil.copy2(f, out / "images" / split / f.name)
            shutil.copy2(lbl_dir / f"{f.stem}.txt", out / "labels" / split / f"{f.stem}.txt")
        print(f"{split}: {len(files)} images")

    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    (HERE / "data.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    print(f"wrote {HERE / 'data.yaml'}")


if __name__ == "__main__":
    main()
