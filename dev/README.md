# dev/

Development and research scripts. Nothing outside `dev/` imports from here; the competition code is in the main
folders. These are kept working on a best-effort basis: their imports follow the new layout, but they are not
part of the tested path.

| folder | contents |
|---|---|
| `experiments/` | earlier pipelines and tools: `click_pick_place.py` (clicked positions instead of YOLO), `click_to_go.py`, `go_to_point.py`, `Full_pipeline.py`, `object_3d_live.py` (digital twin), `pointcloud_view.py`, `match_gui.py`, `silhouette_gui.py`, `capture_pointcloud_case.py`, ... |
| `calibration/` | the older calibration GUIs: `fine_joint_calibration_gui.py` (replaced by `02_calibrate.py offsets`), `gripper_current_gui.py` (plots the gripper current, to choose the grip threshold) |
| `dataset_tools/yolo_dataset/` | capture, ROI, labelling, training and live testing of the YOLO model |
| `dataset_tools/yolo_pouring/` | the same tools for the Game 2 pouring dataset |
| `tools/` | `extract_3mf.py`: the object STLs from the organisers' 3MF |

Run them from the repository root, e.g. `python dev/experiments/click_pick_place.py --no-robot`.

The datasets (images and labels) are not in git. Put them in `datasets/yolo_dataset/` and `datasets/yolo_pouring/`,
each with `images/`, `labels/`, `classes.txt` (the class lists are in `dataset_tools/*/classes.txt`) and its
`roi.json`; or point `YOLO_DATASET` at another folder. Training writes to `runs/segment/` (not in git); copy the
`best.pt` you keep to `models/yolo/`.
