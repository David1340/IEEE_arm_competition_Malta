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

**On another computer:** `git pull`, then `python 01_setup_usb.py --install-saved` (the udev rule from the saved
IDs) and `python 02_calibrate.py sync push` (the repo's motor calibration into LeRobot's folder). After
re-calibrating motors anywhere: `python 02_calibrate.py sync pull` and commit `config/`.

**Coming from the old layout:** `python 02_calibrate.py import-old path/to/calibration.json` moves the old camera
calibration into `config/calibration/camera.yaml`. The fine joint offsets and `arms.json` are already converted.

## Layout

```
01_setup_usb.py  02_calibrate.py  03_teleop.py  run_game.py      the entry points, in the order they are used

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
