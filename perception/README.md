# perception/

What the ZED camera sees: YOLO segmentation of the objects and bins, each object's centre on the table, and its
colour. Settings in `config/perception.yaml`, the camera calibration in `config/calibration/camera.yaml`.

| file | what it does |
|---|---|
| `live.py` | `Vision`: grab a ZED frame, run YOLO, give every object's and bin's centre (the games use this) |
| `detection.py` | shared helpers: settings, ROI, YOLO weights, camera transform, object STLs and sizes |
| `colour.py` | an object's colour (red / green / blue / yellow) from its pixels' hue; `--calibrate` fits the hues |
| `white_balance.py` | optional colour-cast correction for `colour.py` |
| `silhouette.py`, `silhouette_live.py`, `top_face.py` | fit the object's STL silhouette / top face to the mask (`--centre fit`) |
| `object_models.py`, `matching.py`, `cases.py` | STL point models, ICP matching, recorded point-cloud cases (research tools) |

`python -m perception.object_models` prints every object model with its height and centre.

## Point-cloud matching (research)

Fits the object STLs (`models/objects/`) to the YOLO-segmented ZED point cloud. Record a case with
`python dev/experiments/capture_pointcloud_case.py` (saved under `datasets/point_cloud_cases/`, not in git), then
compare fits with `python dev/experiments/match_gui.py`.

Known issue, ZED depth on dark objects: dark blue and green objects come out much flatter than they are (blue cube:
about 10 mm measured, 30 mm real). That is why the games default to `--centre parallax`, which needs no depth.
