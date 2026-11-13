# SO-101 Robot Arm AI Challenge

Two SO-101 follower arms (pink, blue), each teleoperated by a leader arm (blue, green), a ZED camera above the
table, and one script per competition game. Rules: `docs/Robot-Championship-2026-V0.5.pdf`, section 5.8.

Run everything from the repository root.

**For Claude / new readers:** `CLAUDE.md` is a short map of where everything is, and `docs/knowledge_graph.md` is
a generated graph of every module (purpose, definitions, imports, users, config). Rebuild it after changing files:
`python dev/tools/knowledge_graph.py` (`--check` tells whether it is current).

## Setup, in order

| step | command | result |
|---|---|---|
| 0. install | `pip install -r requirements.txt` (plus LeRobot and the ZED SDK, see the file) | |
| 1. USB ports | `python 01_setup_usb.py --list`, then `python 01_setup_usb.py` | each arm's USB ID in `config/arms.yaml`, fixed names `/dev/so101_<arm>` (udev rule) |
| 2. calibrate | `python 02_calibrate.py motors --arm all` | LeRobot motor calibration of the 4 arms, copied to `config/calibration/<arm>/motors.yaml` |
| | `python 02_calibrate.py offsets --arm both` | fine joint offsets of the followers |
| | `python 02_calibrate.py camera tape` (or `camera aruco`) | camera -> table, `config/calibration/camera.yaml` |
| | `python 02_calibrate.py camera touch --arm both` | each follower's base position in `config/arms.yaml` |
| | `python 02_calibrate.py status` | what is calibrated |
| 3. teleop | `python 03_teleop.py` (`--gui` for live telemetry, `net send/receive` over the network) | |
| 4. play | `python run_game.py pick_and_place --round colour` | |
| test without hardware | `python 04_simulate.py run example` | the round in PyBullet (see Simulation below) |

**On another computer:** `git pull`, then `python 01_setup_usb.py --install-saved` (the udev rule from the saved
IDs) and `python 02_calibrate.py sync push` (the repo's motor calibration into LeRobot's folder). After
re-calibrating motors anywhere: `python 02_calibrate.py sync pull` and commit `config/`.

**Coming from the old layout:** `python 02_calibrate.py import-old path/to/calibration.json` moves the old camera
calibration into `config/calibration/camera.yaml`. The fine joint offsets and `arms.json` are already converted.

## Grasp check (real arms)

After every grasp the game prints a `grasp check` line and appends a row to `outputs/grasp_log.csv`: where the grasp
point was planned and where the arm really put it (forward kinematics of the joint readings), both in mm from the
object along the reach (+ beyond, - short), across it and in height, and how far each joint stopped from its target.
Planned close to 0 but reached short: the motors do not reach the planned angles (sag). Both close to 0 but the
grasp still misses: the error is outside the model (the camera's position of the object, or the calibration).

## Gripper: holding or missed

The gripper closes until it is holding something: either the motor current stays at `--grip-current-ma` (80 mA), or
the gripper has stopped moving (`--grip-stall-readings`) while still above `--grip-empty` (8/100); closing to
`--grip-empty` or fully means a miss. Every reading of every close (time, position, current) is appended to
`outputs/gripper_trace.csv`, with the object and the result, to see how the real gripper behaves on each object.
`--miss-margin 10` brings back the earlier model-based early miss check.

## Carrying and dropping into a bin

- The held object is part of the plan from the grasp on (a cylinder of its widest width and its height, moving with
  the gripper): no carry path or drop pose may put it through a bin wall or into the other arm
  (`--no-hold-check` turns this off).
- It is dropped in the posture it was picked with: an object picked top-down hangs straight down (carried level, a
  tall one swings up sideways and sticks out past the jaws, over the rim).
- The drop height is set by the object's bottom, not the jaws: it hangs as far below the grasp point as it was
  grasped above the table, so a high grasp is carried higher (`--drop-clearance`, `--drop-above`).

## Objects close together

- **Other objects are obstacles** (default `--neighbour-radius -1`): each one a column of its own size (its widest
  half width, as its orientation is not known, + `--neighbour-margin` 5 mm). `--neighbour-radius 0` ignores them (the
  gripper may push them; the old default).
- **The wrist turns for top-down grasps** (`--down-roll auto`): the jaws close along the line where the fingers have
  room (`planning/pick_planner.py` `down_rolls`), preferring along the reach (the fixed finger on the base's side, so an
  arm that stops short still closes on the object); up to `--down-rolls` (3) rolls are tried. The wrist turns at the
  start pose, before the arm moves. `--down-roll across` keeps the jaws across the reach (the old behaviour).
- **Blocked objects wait**: an object with no plan and others close around it is tried again after another object
  has been placed; when nothing else can move, the other arm tries it from its side if it reaches it and its bin,
  else it is left.

`--down-roll across --neighbour-radius 0` gives exactly the old plans.

**Cubes are gripped square to their faces.** The camera measures each object's angle seen from above
(`perception/live.py` `top_face_yaw`: the rectangle around its mask outline cast onto the plane of its top; shown as
`..deg` in the live view). For a shape with parallel faces (`faces: 90` for cubes in `config/grasping.yaml`) a top-down
grasp turns the wrist square to a pair of faces; across a corner only if the cube has room to turn into the jaws. Without
an angle (e.g. `--centre fit`) any wrist roll, as before.

**Short objects far out are grasped level.** Top-down reaches ~35 cm from a base, so objects in the middle of the table
(~37-42 cm from both arms) need a level grasp; an object under 5 cm tall is grasped level with the grasp point over its
top, as the last resort (`level_over_top` in `config/grasping.yaml`).

## Planning speed

Each arm plans in a process of its own (`--planner process`, the default): both arms plan at once, and each plans its
next object while the current one is carried; that plan is used only if nothing it depended on has changed (the
object, the bins, the other arm, the arm's start pose within `--prefetch-tolerance`). `--planner thread
--no-prefetch` is the old behaviour. Each arm's planning world is built once and reused (same plans, ~0.85 s less
per plan).

## Simulation: test without the arms or the camera

`04_simulate.py` plays a pick-and-place round in a PyBullet window with the same code as the real game: the same
rules, coordinator, behaviour trees, planner (IK, RRT-Connect, the other arm, the bins and the table as obstacles),
gripper openings and arm bases. You place the objects and bins yourself instead of the camera finding them. Every
animation step is checked for contact: a `COLLISION` line names the arm, the link and what it hit, and the round
ends with a summary (placed / failed / in which bin / collisions).

Simulated, so not tested here: YOLO and the camera (positions are exact, no orientation), grasp physics (a grasp
always holds), motor tracking (the arm follows the path exactly).

**Get it on your computer** (conda; Python 3.12, as pybullet has no ready build for 3.13):

```bash
git clone https://github.com/David1340/IEEE_arm_competition_Malta.git
cd IEEE_arm_competition_Malta
git checkout clean
conda env create -f environment-sim.yml
conda activate so101-sim
```

Without conda: a Python 3.10-3.12 venv and `pip install -r requirements-sim.txt`. On a computer with ROS 2, open a
terminal where ROS's `setup.bash` is not sourced (or ignore pip's warnings about ROS packages such as launch-ros:
they come from ROS's Python path and do not affect this project). The lab PC's `lerobot` conda environment can run
the simulator too: `conda activate lerobot` and `pip install -r requirements-sim.txt`.

Already cloned: `git fetch origin`, `git checkout clean`, `git pull`.

**Run it:**

| command | what it does |
|---|---|
| `python 04_simulate.py edit example` | top-down editor: click to place, drag to move, `1-4` shape, `Tab` bin, `r g b y` colour, `s l` size, `, .` turn, `x` delete, `Enter` save and run |
| `python 04_simulate.py edit my_test --random 8` | a new scene with 8 random objects |
| `python 04_simulate.py run example` | play the round in the PyBullet window (mouse: Ctrl + left drag turns the view, wheel zooms) |
| `python 04_simulate.py run example --round shape --speed 2` | any game option (`run -h`), faster playback |
| `python 04_simulate.py run example --arm follower_pink` | one arm only |
| `python 04_simulate.py run example --headless` | no window: plan, move and check (quick tests, CI) |
| `python 04_simulate.py test` | every regression scene in `simulation/scenes/tests/` (touching objects, clusters, against a bin), one line each |
| `python 04_simulate.py test --gui` | the same, each scene played in the PyBullet window (`test --gui -- --speed 3` faster) |

Scenes are YAML files in `simulation/scenes/` (TABLE frame, metres), so a test layout can be committed and shared.
Joint limits: the planner uses each arm's motor calibration from `config/calibration/<arm>/motors.yaml`; until
that is in the repo (`python 02_calibrate.py sync pull` on the lab PC, then commit), the simulator warns and uses the
URDF's limits, which can be wider than the real arm's.

## Layout

```
01_setup_usb.py  02_calibrate.py  03_teleop.py  run_game.py  04_simulate.py   the entry points

config/          every setting and calibration value, YAML (see config/README.md)
  arms.yaml        the 4 arms: ports, USB IDs, pairs, bases, home pose, handoff points
  perception.yaml  camera ROI, YOLO weights and classes, colours
  teleop.yaml      network teleop
  games/           one rules file per game
  calibration/     camera.yaml, and per arm: motors.yaml (LeRobot), joint_offsets.yaml

core/            config loading (config.py), every path (paths.py), the arms' setup and TABLE frame (arms.py)
hardware/        USB ports and udev (usb_ports.py), connect and move the arms (arm.py), gripper, joint offsets
calibration/     camera and base calibration, offsets GUI (calibrate.py), LeRobot calibration sync (lerobot_sync.py)
teleop/          leader -> follower teleop: lerobot-teleoperate launcher, telemetry GUI, over the network
perception/      ZED + YOLO: object and bin positions (live.py), colours (colour.py), silhouette fitting
planning/        IK (ik.py, M-FRPSO), paths (rrt_connect.py), the simulation world (sim.py), pick plans (pick_planner.py)
state_machine/   behaviour trees (behaviour.py), the two-arm coordinator (coordinator.py), each arm's worker
games/           one script per game: pick_and_place.py (Game 1), two_arm_coordination.py (Game 2, stub); rules.py
simulation/      PyBullet simulation of a round: scenes/ (YAML layouts), editor.py, viewer.py, run.py (04_simulate.py)
models/          objects/ (STLs), robot/ (SO-101 URDF), yolo/ (trained weights, not in git)
dev/             experiments, older GUIs and dataset tools (see dev/README.md)
docs/            competition rules
```

How a game is put together, e.g. `games/pick_and_place.py`: `perception.live.Vision` finds the objects and bins;
`games/rules.py` says which bin each goes in (`config/games/pick_and_place.yaml`); `state_machine.coordinator`
gives each arm its tasks and keeps the arms apart; each arm runs `state_machine.arm_worker` (a behaviour tree:
locate, plan with `planning.pick_planner`, grasp with `hardware.gripper`, carry, release, verify); `hardware.arm`
moves the real arms.

## Frames

TABLE frame: origin on the table halfway between the two followers' shoulder pan axes, x forward, y towards
follower_pink, z up, metres. Each follower's base pose in it is `config/arms.yaml` `base`.

## Games

| game | script | rounds | state |
|---|---|---|---|
| 1. Pick-and-place | `games/pick_and_place.py` | colour, shape, dynamic | working (dynamic: re-detects before each pick and drop) |
| 2. Two-arm coordination | `games/two_arm_coordination.py` | pouring, folding, wrap_and_pack | stub |

`python run_game.py pick_and_place -h` lists every option. `--no-robot` plans without arms.
