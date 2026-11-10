# Step 01 — Label the speed-climbing holds

**Run:** `python main.py` from the `Step_01_Labeling` folder in `AI_ARM`.

Only one-class red speed handholds. Images and model weights stay here:

```text
Step_01_Labeling/
  main.py
  data/
    images/       your original JPG/PNG photos
    labels/       saved YOLO segmentation polygons (class 0)
    best.pt       your YOLO26 segmentation model or detector
    sam2_t.pt     SAM 2 tiny weights (optional; auto-downloaded)
```

## Manual or model-assisted

- **Manual:** draw or freehand a complete closed hold. You can edit it afterwards.
- **Model-assisted:** choose **YOLO segmentation** or **YOLO boxes + SAM** from the `Mask source` dropdown.
  - YOLO segmentation: the original `best.pt` generates the mask directly; requires `best.pt` with `task='segment'`.
  - YOLO boxes + SAM: `best.pt` supplies hold boxes and SAM 2 estimates a detailed mask inside each box. Works with detection-only or segmentation weights.

Both methods show orange **proposals**, not automatically saved labels. Click an orange hold to accept and edit it, or choose **Use all new suggestions**. Changing method regenerates proposals, **never your previously accepted labels**. The new trained model can be installed from Step 2; restart Step 1 after an update.

## Editing

- `E` — click a hold to select and magnify it; drag a vertex; double-click an edge to add a vertex.
- `R` — drag **from one part of the existing boundary to another** to replace just that section; repeat as often as necessary. The hold stays one closed polygon.
- `B` — drag a rectangle to delete several holds (centres inside rectangle).
- `T` — freehand a new closed hold; `D` — polygon by clicking points.
- `X` — remove selected vertex; arrows — nudge vertex; `S` — gently smooth selected polygon.
- `Z` — focus selected hold; `F` — show whole photo; mouse wheel — zoom; `Ctrl+Z` — undo; `Ctrl+S` — save.

Labels are saved automatically in `data/labels/` as YOLO segmentation polygons. Original image orientation is normalised and backed up before modification.

## What changed in this update

A dropdown selects the **source of initial mask proposals**. Existing redraw, polygon-editing, box-delete and undo logic remains as in the last working Step 1 patch. No change to original photographs, labels or `best.pt` is required.

**Testing:** model-choice logic and YOLO box/mask extraction passed mocked tests. Actual GPU/SAM results need to be run with your checkpoints on your PC.
