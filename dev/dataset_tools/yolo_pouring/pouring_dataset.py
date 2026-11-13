#!/usr/bin/env python3
"""Bowl and cup (Game 2, pouring): collect pictures, label them, train and try a YOLO segmentation model.

The same tools as the blocks (dev/dataset_tools/yolo_dataset/*.py), run on datasets/yolo_pouring/ (images, labels,
classes.txt (bowl, cup), roi.json; not in git: copy it between computers); the trained models go to runs/pouring/
(not runs/segment/, which the blocks pipeline takes its newest model from, so the two models never mix).

    python dev/dataset_tools/yolo_pouring/pouring_dataset.py status        # pictures / labels / models so far
    python dev/dataset_tools/yolo_pouring/pouring_dataset.py capture       # ZED pictures: s = save one, a = auto, q = quit
    python dev/dataset_tools/yolo_pouring/pouring_dataset.py capture --interval 0.5
    python dev/dataset_tools/yolo_pouring/pouring_dataset.py roi select    # drag the area the bowl and cup are in (once)
    python dev/dataset_tools/yolo_pouring/pouring_dataset.py label         # SAM labelling (0 = bowl, 1 = cup)
    python dev/dataset_tools/yolo_pouring/pouring_dataset.py view          # check the labels
    python dev/dataset_tools/yolo_pouring/pouring_dataset.py train         # crop to the ROI, split, train
    python dev/dataset_tools/yolo_pouring/pouring_dataset.py live          # the newest model on the live camera

The dataset folder is made the first time, with this folder's classes.txt and roi.json in it.

Anything after the step is passed on to that tool (e.g. `label --redo`, `train --epochs 150`).
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]            # the repo
HERE = ROOT / "datasets" / "yolo_pouring"              # the images and labels (not in git)
TOOLS = ROOT / "dev" / "dataset_tools" / "yolo_dataset"
RUNS = ROOT / "runs" / "pouring"
STEPS = {
    "capture": "dataset_generation.py",
    "roi": "roi.py",
    "label": "dataset_segmentation.py",
    "view": "view_labels.py",
    "train": "train_seg.py",
    "live": "live_seg.py",
    "predict": "predict_seg.py",
}
IMG_EXT = {".jpg", ".jpeg", ".png"}


def status():
    images = [f for f in (HERE / "images").glob("*") if f.suffix.lower() in IMG_EXT]
    labelled = [f for f in images if (HERE / "labels" / f"{f.stem}.txt").exists()]
    counts = {}
    names = [c.strip() for c in (HERE / "classes.txt").read_text().splitlines() if c.strip()]
    for f in labelled:
        for line in (HERE / "labels" / f"{f.stem}.txt").read_text().splitlines():
            if line.split():
                c = int(line.split()[0])
                counts[names[c] if c < len(names) else c] = counts.get(names[c] if c < len(names) else c, 0) + 1
    models = sorted(RUNS.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    print(f"pictures: {len(images)} | labelled: {len(labelled)} | objects: {counts or 'none yet'}")
    print(f"ROI: {'set' if (HERE / 'roi.json').exists() else 'not set (python yolo_pouring/pouring_dataset.py roi select)'}")
    print(f"models: {len(models)}" + (f", newest {models[-1].relative_to(ROOT)}" if models else ""))


def prepare():
    """datasets/yolo_pouring/ with its images/ and labels/, and this folder's classes.txt and roi.json in it (the
    tools read them from the dataset folder) unless it has its own."""
    import shutil

    for sub in ("images", "labels"):
        (HERE / sub).mkdir(parents=True, exist_ok=True)
    for name in ("classes.txt", "roi.json"):
        own = Path(__file__).resolve().parent / name
        if own.is_file() and not (HERE / name).is_file():
            shutil.copy2(own, HERE / name)
            print(f"(made {(HERE / name).relative_to(ROOT)} from {own.relative_to(ROOT)})")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") or sys.argv[1] not in [*STEPS, "status"]:
        print(__doc__)
        raise SystemExit(0 if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help") else 1)
    step, rest = sys.argv[1], sys.argv[2:]
    prepare()
    if step == "status":
        status()
        return
    env = dict(os.environ, YOLO_DATASET=str(HERE), YOLO_RUNS=str(RUNS))
    if step == "capture" and "--prefix" not in rest:
        rest = ["--prefix", "pour", *rest]
    command = [sys.executable, str(TOOLS / STEPS[step]), *rest]
    print("+", " ".join(command), f"  (dataset {HERE.relative_to(ROOT)}, models {RUNS.relative_to(ROOT)})")
    raise SystemExit(subprocess.run(command, env=env).returncode)


if __name__ == "__main__":
    main()
