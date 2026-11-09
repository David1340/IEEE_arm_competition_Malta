# SO-101 pick and place

Run everything from the repository root.

| folder | contents | how to use |
|---|---|---|
| `vision/` | libraries: YOLO/ZED helpers (`detection.py`), STL models, silhouette / top-face fitting | `from vision.detection import ...` |
| `control/` | libraries: M-FRPSO IK, RRT-Connect, arm connection (`IK_control.py`), fine joint offsets | `from control.modified_frpso_ik import ...` |
| `calibration/` | calibration tools (`calibrate.py`: ArUco camera calibration; fine joint offsets GUI, gripper current GUI) and the saved offsets in `robots/` | `python calibration/calibrate.py` |
| `experiments/` | runnable scripts: `Full_pipeline.py`, `sim_plan.py` (simulation only), `object_3d_live.py`, `match_gui.py`, `silhouette_gui.py`, ... | `python experiments/sim_plan.py` |
| `yolo_dataset/` | dataset, labels, ROI and YOLO training / capture tools | `python yolo_dataset/train_seg.py` |
| `arms.json`, `pick_and_place.json` | config: the arms' ports and positions; the sorting modes (colour / shape) and which bin each object goes in | edit the file; `python experiments/pick_and_place.py --round shape` |
| `stl/`, `robot_urdfs/`, `docs/` | object models, robot model, documents | |

`vision/` and `control/` modules with a command line run with `python -m`, e.g.
`python -m vision.silhouette_live --replay point_cloud_mathing/case_...`.
Scripts in `experiments/` and `calibration/` add the repository root to `sys.path`, so they work from any directory.
