# Step 02 — Train YOLO and compare masks

Run from `Step_02_Training` in your `AI_ARM` Conda environment:

```bat
python -m pip install -r requirements.txt
python main.py
```

This folder sits **beside Step 01**. It reads `../Step_01_Labeling/data/images` and `../Step_01_Labeling/data/labels` directly. It does not copy anything back to Step 1 unless you explicitly select **Update Step 1 model**.

```text
Code/
  Step_01_Labeling/
    main.py
    data/
      images/       your labelled photographs
      labels/       your corrected YOLO polygons
      best.pt       your existing trained model
      sam2_t.pt     SAM weights (optional)
  Step_02_Training/
    main.py
    settings.py
    requirements.txt
    README.md
    data/
      test_images/  NEW images for checking model generalisation
    results/      generated automatically
```

## Four menu options

**1 — Prepare/check dataset**

Reads Step 1 saved polygons. By default, creates **YOLO segmentation** training labels (not boxes), clips polygons when tiling, splits by **original photograph before tiling**, and generates `results/dataset/preview.jpg`. Empty/unsaved labels are excluded by default. Original Step 1 photos and annotations are unchanged. A separate test split is reserved only when there are enough independently labelled photos.

**2 — Train YOLO segmentation**

Starts from `yolo26n-seg.pt` by default. You can also fine-tune your existing `Step_01_Labeling/data/best.pt` (when it is a segmentation checkpoint), or continue training from an earlier Step 2 segmentation checkpoint. Weights and training plots are saved in `results/training/`. Your Step 1 `best.pt` is untouched until option 4.

**3 — Compare the model on new photos**

Put **unseen** JPG/PNG images in `data/test_images/`. The supplied Shanghai photograph is an example. Run option 3 (press Enter for the default folder) or:

```bat
python main.py test
```

For **each image**, the program writes:

```text
results/predictions/comparison_.../001_photo/
  comparison.jpg           3 views side by side
  yolo_boxes.jpg           YOLO bounding boxes
  yolo_segmentation.jpg    YOLO's own instance segmentation masks
  yolo_boxes_plus_sam.jpg  SAM masks prompted by YOLO boxes
  predictions.json         per-hold boxes, masks and confidence for Step 3
  summary.json
```

If Step 2 has no checkpoint yet, it tests your existing Step 1 `best.pt`. You may explicitly select weights with:

```bat
python main.py test --weights "../Step_01_Labeling/data/best.pt"
```

If the chosen model is detection-only, its YOLO segmentation panel is empty and the program warns you. **YOLO + SAM does not run unless Ultralytics and SAM weights can be loaded.** SAM 2 tiny weights may download once automatically.

Optional quantitative evaluation of a separately held-out labelled test split:

```bat
python main.py test --metrics
```

**4 — Update Step 1 model**

Backs up Step 1's existing `best.pt` to `results/backups/`, then installs the latest trained checkpoint after confirmation. Restart Step 1 to load it. Do not update unless you prefer the new results.

## Settings

The small `settings.py` controls epochs, batch size, tile size, image size and model task. **Default: `TRAIN_TASK='segment'` and `BASE_MODEL='yolo26n-seg.pt'`.** If you intentionally want detection-only training, change both to `detect` and `yolo26n.pt` and re-run Prepare before training.

## Important interpretation

The side-by-side pictures are a **qualitative** comparison, not a proof that SAM is more accurate. Quantitative performance needs manually verified masks on held-out wall photos. Avoid putting the same photo, crops of it, or near-duplicate photos from the same camera session in both training and independent testing.

## Tests

```bat
python -m unittest discover -s tests -v
```

Dataset, model-backup and mock-comparison tests passed locally. Actual Ultralytics YOLO/SAM inference and training must still be tested on your PC.
