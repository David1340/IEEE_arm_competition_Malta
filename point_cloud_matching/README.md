# Point-cloud matching (STL model vs ZED + YOLO)

Fits the official object STLs (`stl/`) to the YOLO-segmented ZED point cloud, to get each
object's true centre (and later its top face), instead of the mean of the visible points.

## Run

1. Capture a case (ZED + YOLO -> `point_cloud_matching/cases/case_*/`):

       python yolo_dataset/capture_pointcloud_case.py

2. Open the comparison GUI (newest case by default):

       python point_cloud_matching/match_gui.py
       python point_cloud_matching/match_gui.py point_cloud_matching/cases/case_20261101T165958_770293Z

Needs numpy, scipy, matplotlib, opencv-python (all already pulled in by ultralytics).

## Files

| file | what it does |
|---|---|
| `models.py` | loads the STLs, finds each shape's stable resting poses (e.g. cylinder upright / on side), samples surface points + normals, in metres |
| `cases.py` | loads a capture case, converts it to the table frame, and fits the local table plane from a ring of pixels around each mask |
| `matching.py` | 4-DoF ICP (X, Y, yaw, optional Z) of every size x pose candidate against the data, comparing only the model faces the camera can see |
| `match_gui.py` | the GUI |

`python point_cloud_matching/models.py` prints every candidate with its height and centre.

## How the fit works

* The object rests on the table, so the model's base is locked to the **local** table plane
  (the calibrated Z = 0 is about 1-2 cm off the real table in the first case).
* Only model points whose normal faces the camera are compared (all shapes are convex, so
  this is exact). Matching is symmetric, data -> model and model -> data, which is what
  separates small from large.
* Each candidate starts from 12 yaw angles; the best result by RMS x coverage wins.
* **3D mode** uses XYZ distances. **Top-down mode** uses XY only, for objects whose depth is
  unreliable.

## Known issue: ZED depth on dark objects

In the first case, dark blue and green objects come out much flatter than they really are
(blue cube: about 10 mm measured, 30 mm real; blue cylinders: about 16 mm, 50 mm real). The
yellow ones are close to correct. With that data the 3D fit cannot match well, and the GUI
shows a warning. Things to try: `--depth-mode NEURAL` when capturing, more light, or a
lower camera.
