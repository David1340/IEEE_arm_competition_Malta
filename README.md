# IEEE Robot Championship 2026 (Malta): SO-101 pick and place

A ZED stereo camera looks down at the table; a YOLO segmentation model finds the game objects; the ZED
point cloud gives each one's 3D centre; an SO-101 (LeRobot follower) arm goes to it.

The repo lives inside LeRobot's data folder (`~/.cache/huggingface/lerobot`), next to `calibration/` and
`runs/`, so `.gitignore` ignores everything and lists the project's files back in. **A new file or folder
must be added to `.gitignore` to be committed.**

## Layout

| folder | what |
|---|---|
| `robot/` | the arm. `go_to_object.py`: detect, then move the gripper to an object. `detect.py`: object positions in the robot's base frame. `gripper_current_gui.py`: live gripper current plot and grip detection |
| `yolo_dataset/` | the vision. Capturing images (`capture_images.py`), labelling, training (`train_seg.py`), live views (`live_seg.py`, `object_3d_live.py`), the ROI (`roi.py`, `roi.json`), HSV colour of the blocks (`block_color.py`, `color_flow_viewer.py`, `white_balance.py`), point-cloud cases (`capture_pointcloud_case.py`) |
| `point_cloud_matching/` | fitting the official STL models to the point cloud (true centre and pose); saved cases in `cases/` |
| `stl/` | the game objects (cube, cylinder, dodecahedron, pentagonal prism; small and large) |
| `docs/` | competition rules |

## Install

```sh
pip install -r requirements.txt     # pyzed comes with the ZED SDK installer
```

`so101ik` is the inverse kinematics and servo driver (C, with a Python binding). It builds on install, so it
needs a C compiler (`sudo apt install build-essential`). It finds the follower's LeRobot calibration in
`calibration/robots/so_follower/` and the serial port itself; set `SO101_PORT` / `SO101_CALIBRATION` if it
cannot tell.

## Go to an object

Start with the arm **folded in its rest pose** and nothing else in its reach.

```sh
python robot/detect.py                                  # where the objects are (robot base frame); nothing moves
python robot/go_to_object.py --dry                      # + the plan for the arm; nothing moves
python robot/go_to_object.py                            # the most confident reachable object (asks first)
python robot/go_to_object.py --target cube
```

What it does:

1. **Detect**: 15 ZED frames through YOLO; each object's centre is computed as `object_3d_live.py` does
   (mean of the mask's point-cloud samples, in the ArUco TABLE frame) and the median over the frames is taken.
   The camera is closed before the arm moves.
2. **Plan** (`so101ik`): the tip goes 1.5 cm below the object's measured top, never lower than 1.2 cm above
   the table. A vertical descent from 7 cm above is solved at the steepest gripper pitch the arm can hold all
   the way down (straight down near the base, sloping further out). Every pose is checked against the joint
   ranges, the table and the arm's own parts.
3. **Move**: unfold, go above the object, open the jaws, then go straight down. The descent stops early if the
   tip is held up by something (a contact, not a stall), or if a joint stalls. It waits for Enter, goes
   straight up, then folds home.

Ctrl-C stops and folds home. A second Ctrl-C switches the torque off (hold the arm).

The settings are at the top of `robot/go_to_object.py` (`HOVER`, `GRASP_DEPTH`, `MIN_TIP_Z`, `PRESS`, `OPEN`,
speeds).

### Frames: check this once

`detect.py` assumes the ArUco TABLE frame **is** the robot's base frame: metres, x forward, y to the arm's
left, z up, origin under the shoulder pan axis. The saved case supports this: the detected gripper sits at
y = +0.016 m, in front of the base. If the markers were measured from somewhere else, set `TABLE_TO_BASE` in
`robot/detect.py` (or pass `--base-offset X Y Z YAW`).

To check it:
1. Switch the torque off.
2. Run so101ik's `examples/where.py`. It prints the tip's x, y, z from the servos.
3. Put the gripper tip on a block that `detect.py` lists.
4. The two printouts should agree to about 1 cm.

### Without the camera or the arm

```sh
python robot/go_to_object.py --case point_cloud_matching/cases/case_20261101T165958_770293Z --dry
python robot/go_to_object.py --case point_cloud_matching/cases/case_20261101T165958_770293Z --sim --yes
python robot/go_to_object.py --case ... --sim --yes --sim-floor 0.03     # tries the contact stop
```

`--case` takes the positions from a saved capture. `--sim` drives so101ik's simulated servos (no gravity, so
the gravity-sag compensation is switched off there). On the saved case, the tip ended within 1 mm of the
goal for every reachable object. With a surface at 3 cm, the descent stopped on it at 30 mm.

### Reach

At table height the arm reaches about 15–45 cm from its base. Pointing straight down, it reaches only about
15–30 cm. Of the 8 objects in the saved case, 3 cannot be reached at grasp height: the cylinders at 47 and 55 cm, and
the cube at 46 cm. The dodecahedron at 46 cm can, at a shallow 20° pitch. The program skips what it cannot
reach and says so.
