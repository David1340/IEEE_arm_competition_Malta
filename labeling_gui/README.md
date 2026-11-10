# Labelling GUI (multi-class, model-assisted)

Polygon labelling for the YOLO segmentation datasets (`yolo_dataset/`, `yolo_pouring/`).
Reads and writes `<data>/labels/<image>.txt` in the same full-frame YOLO format the rest of the
pipeline (`roi.py apply`, `train_seg.py`) already uses, so existing labels open as they are.

## Run

From the repo root, in the `lerobot` env (needs `ultralytics`, `pillow`, `opencv-python`, Tk):

```bash
python labeling_gui/label_gui.py
```

Defaults:

| What | Default | Change with |
|---|---|---|
| Dataset | `yolo_dataset/` (`images/`, `labels/`, `classes.txt`, `roi.json`) | `--data yolo_pouring` |
| YOLO model | newest `runs/segment/*/weights/best.pt` | `--weights runs/segment/train-3/weights/best.pt` |
| SAM model | `labeling_gui/sam2_t.pt` (downloaded on first use) | `--sam sam_b.pt` |
| Confidence | 0.25 | `--conf 0.4` |
| Model input | the `roi.json` crop at its native size, as in training | `--no-roi` |

The paths it uses are printed at start-up.

## Workflow for the unlabelled images

1. Tick **Model-assisted**. Mask source **YOLO segmentation** uses the trained model's own masks;
   **YOLO boxes + SAM** keeps YOLO's boxes and classes and lets SAM draw the outline.
2. Suggestions appear dashed, coloured by class, with `? class` above them. Click one to accept it,
   or **Use all new suggestions**.
3. Fix what is wrong: `E` + click an outline, then `1`-`5` changes its class; drag points, `R` redraws
   part of the edge, `Delete` removes it. Draw anything the model missed with `D` (polygon) or `T` (freehand)
   after choosing the class with `1`-`5` or the dropdown.
4. **Next unlabelled** jumps to the next image without a label file. Everything saves automatically.

## Keys

`1`-`9` class (relabels the selected outline in Edit mode) · `D` polygon · `T` freehand · `E` edit ·
`R` redraw edge · `B` box delete · `P` pan · `A`/`X` add/remove point · `S` smooth · `Z` focus ·
`F` fit · wheel zoom · `Ctrl+Z` undo · `Ctrl+S` save · `Enter` finish polygon · `Esc` cancel draft

## Behaviour to know

- Outlines of **different** classes may overlap (objects inside the bin). Two outlines of the **same**
  class that overlap almost entirely are treated as a duplicate: new ones are refused, and on load an
  exact duplicate is removed after backing the file up as `labels/<name>.before_polygon_cleanup.bak`.
- Classes the model knows but `classes.txt` does not list (e.g. `gripper`) are ignored, with a message.
  Model classes are matched to `classes.txt` by name, not by id.
- An image whose EXIF says it is rotated is saved upright once, with the original kept in `<data>/originals/`.
  Camera captures from the ZED have no EXIF rotation, so this does not normally happen.
