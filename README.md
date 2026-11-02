# IEEE Robot Championship 2026 (Malta): SO-101 pick and place

A ZED stereo camera looks down at the table; a YOLO segmentation model finds the game objects; the ZED
point cloud gives each one's 3D centre; an SO-101 (LeRobot follower) arm goes to it.

The repo lives inside LeRobot's data folder (`~/.cache/huggingface/lerobot`), next to `calibration/` and
`runs/`, so `.gitignore` ignores everything and lists the project's files back in. **A new file or folder
must be added to `.gitignore` to be committed.**

## Layout

| folder | what |
|---|---|
| top level | David's pipeline: `Full_pipeline.py` (RRT-Connect + FRPSO IK in PyBullet), `IK_control.py`, `fine_joint_offsets.py` (per-joint offsets, read by the pipeline), `set_so101_joint_positions.py` |
| `robot/` | the arm. `go_to_object.py`: detect, then move the gripper to an object. `detect.py`: object positions in the robot's base frame. `gripper_current_gui.py`: live gripper current plot and grip detection |
| `yolo_dataset/` | the vision. Capturing images (`capture_images.py`), labelling, training (`train_seg.py`), live views (`live_seg.py`, `object_3d_live.py`), the ROI (`roi.py`, `roi.json`), HSV colour of the blocks (`block_color.py`, `color_flow_viewer.py`, `white_balance.py`), point-cloud cases (`capture_pointcloud_case.py`) |
| `point_cloud_matching/` | fitting the official STL models to the point cloud (true centre and pose); saved cases in `cases/` |
| `stl/` | the game objects (cube, cylinder, dodecahedron, hexagonal (really pentagonal) prism; small and large) |
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
   (x, y and height from the top face of the mask's point-cloud samples, in the ArUco TABLE frame; a tall object's side wall would pull a plain mean towards the camera) and the median over the frames is taken.
   The camera is closed before the arm moves.
2. **Plan** (`so101ik`): the **middle of the open jaws** goes to the object: 1.5 cm below its measured top,
   never lower than 3 cm above the table,
   2.5 cm in from the finger ends (nearer the palm). The jaws open sideways (wrist roll fixed at -90°, the camera is on one
   side), so the tip (the fixed jaw) passes 3 cm to one side of the object and the moving jaw to the other.
   A vertical descent from 12 cm above is solved at the steepest gripper pitch the arm can hold all the way
   down. Every pose is checked against the joint ranges, the table and the arm's own parts.
3. **Move**: unfold, go above the object, open the jaws to 50 % (6.8 cm between the jaw ends; the blocks are
   3 to 5.5 cm), then go straight down. The descent stops early if the tip is held up by something (a contact,
   not a stall), or if a joint stalls. Then it closes the jaws until they stop on the object (holding it
   2.5 % past that), lifts it 12 cm watching the grip (it stops if the object slips), waits for Enter, puts it
   back down where it was, lets go, goes straight up and folds home. `--no-grasp` only goes to the object.

**Pick and place by clicking**: in the picture window, click an object to pick it (red ring) and click an empty
spot on the table to put it there (purple cross; its position comes from the ZED point cloud under the click).
Click a **bin** to drop the object in it: the jaws' middle hovers over the bin's centre 15 cm above the table
(`DROP_HEIGHT`; higher if the carried object would not clear the rim by 1.5 cm) and the jaws open. YOLO's masks
are drawn on the picture. `g` or Enter goes, `c` clears the place (the object is then put back where it was), `q` quits. Without a
window: `--place X Y` (robot base frame, m) or `--drop-in N` (bin N of the printed list). The object is carried from 12 cm above the pick spot to 12 cm above
the place, every step checked and the grip watched, then lowered until it touches the table and let go.

**Size, height and opening**: every object's official STL is fitted to its YOLO mask
(`robot/shape_fit.py`, the render-and-compare fit from `point_cloud_matching/silhouette.py`): small or large,
yaw, and its true height (the ZED reads dark objects too flat). The jaws open just wider than the object is
*across the jaws* (`JAW_MARGIN`, 8 mm each side; from the gripper's meshes, `robot/gripper.py`), not to a fixed
50 %. Near the base the gripper comes down steeply, so the object then sits less deep in the jaws, keeping the
fixed finger's tip `TIP_MIN` (1.2 cm) above the table: it used to touch the table there and stop.

Ctrl-C stops and folds home. A second Ctrl-C switches the torque off (hold the arm).

**Calibration**: the LeRobot file `~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower_arm.json`
(`--calibration` to change it). **Fine joint offsets**: `calibration/robots/so_follower/fine_joint_offsets.json`,
set with `python robot/calibrate.py offsets` and read with `fine_joint_offsets.py`, as the rest of the repo does
(true angle = LeRobot reading + offset; motor target = wanted angle - offset). `--offsets none` runs without them.

The settings are at the top of `robot/go_to_object.py` (`HOVER`, `GRASP_DEPTH`, `MIN_GRASP_Z`, `OPEN`, `ROLL`,
`JAW_CENTRE`, `PRESS`, speeds). `python robot/calibrate.py check` (torque off) shows where the model puts the tip while
the arm is moved by hand.

## Game 1: sort the objects into the bins

```sh
python robot/sort_objects.py                  # the round in robot/sort_rules.json; shows the plan, g starts
python robot/sort_objects.py --round shape
python robot/sort_objects.py --dry            # detect and show the plan only
```

`robot/sort_rules.json` says, for each round, what goes into each bin colour. An entry is a colour, a YOLO class
or both (`"yellow cube"`):

| round | red bin | blue bin | green bin | yellow bin |
|---|---|---|---|---|
| `colour` (Round 1) | red objects | blue | green | yellow |
| `shape` (Rounds 2 and 3) | cube | cylinder | dodecahedron | hexagonal |

Every detection gets its colour from its YOLO mask with `yolo_dataset/block_color.py` (the HSV vote; blocks lying
in a bin are left out of the bin's vote). Each cycle the arm is folded, the camera detects again (so bins or
objects moved by a judge are found where they are now), the most confident object that the arm can reach and
whose bin is seen and reachable is picked and dropped into its bin (15 cm over its centre; it goes up to that height first and then straight across, so it
does not swing into the bin's side), and the arm folds.
Objects already inside a bin are left; one that fails twice (slipped, jaws closed on nothing) is left too.
The plan picture (masks in their detected colour, an arrow from each object to its bin) is saved in
`outputs/sort_objects/`.

### Calibration: one script, one file

Every calibration is a mode of one script, `robot/calibrate.py`, and every value lives in one file,
`calibration.json` at the repo root: the camera's pose in the robot's base frame, the ArUco markers' positions,
the table height, the arm's LeRobot calibration file and the fine joint offsets. `camera_transform.npy` is
written from it for the scripts that read that.

```sh
python robot/calibrate.py ids                      # live view: which marker has which ID
python robot/calibrate.py --marker-ids 0,1,2,3     # first time: IDs in the order TOP_RIGHT, BOTTOM_RIGHT,
                                                   # BOTTOM_LEFT, TOP_LEFT (the measured marker positions)
python robot/calibrate.py arm                      # once, the most accurate: touch 4 blocks with the arm
python robot/calibrate.py                          # after the camera is bumped: markers only, a few seconds
python robot/calibrate.py check                    # torque off: the model's tip and jaws while you move the arm
python robot/calibrate.py offsets                  # fine joint offsets: PyBullet sliders over the live arm
python robot/calibrate.py show                     # what calibration.json holds
```

The camera fit: the table plane from the point cloud gives the camera's tilt and height; two or more markers (or
the blocks touched with the arm) give its turn and position. `arm` also measures where the markers really are in
the arm's frame (hand-measured positions are often a few cm off) and stores them, so later marker-only
calibrations land where the arm does. Each run saves a picture in `outputs/calibrate/`.

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
