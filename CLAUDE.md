# CLAUDE.md

SO-101 robot arms for the IEEE Robot Arm AI Challenge: two followers (`follower_pink`, `follower_blue`) driven by
two leaders (`leader_blue`, `leader_green`), a ZED camera over the table, one script per game.

**Find code with `docs/knowledge_graph.md` first** (every file: purpose, definitions, imports, who imports it,
which config it uses). Grep inside it for a function or module name instead of searching the tree. It is
generated from the code: rebuild it with `python dev/tools/knowledge_graph.py` after adding, moving or renaming
files or functions (`--check` tells whether it is current).

## Where things are

| need | go to |
|---|---|
| a setting or calibration value | `config/` (YAML; see `config/README.md`), read only through the modules below |
| arms: ports, USB IDs, pairs, bases, home pose, handoff points | `config/arms.yaml` via `core/arms.py` |
| a path | `core/paths.py` (never build paths from `__file__` in new code) |
| YAML load/save (keeps header comments) | `core/config.py` |
| TABLE frame -> an arm's base frame | `core/arms.py` `arm_from_table()` |
| USB detection, udev rule | `hardware/usb_ports.py` (`01_setup_usb.py`) |
| connect, move, home, park an arm; `send_joints`, `follow_path` | `hardware/arm.py` |
| gripper opening, grasp with current feedback | `hardware/gripper.py` |
| fine joint offsets | `hardware/joint_offsets.py`, `config/calibration/<arm>/joint_offsets.yaml` |
| LeRobot motor calibration copy (pull/push) | `calibration/lerobot_sync.py`, `config/calibration/<arm>/motors.yaml` |
| camera calibration, base touch, offsets GUI | `calibration/calibrate.py` (`02_calibrate.py camera/offsets`) |
| camera + YOLO -> object/bin positions | `perception/live.py` (`Vision`); ROI/weights/STLs in `perception/detection.py` |
| object colour | `perception/colour.py` |
| IK | `planning/ik.py` (M-FRPSO); paths `planning/rrt_connect.py` |
| PyBullet planning world | `planning/sim.py` (`World`) |
| planning a pick, grasp heights, postures | `planning/pick_planner.py` (`plan_task`, `plan_sequence`) |
| wrist roll for top-down grasps, neighbours as obstacles | `planning/pick_planner.py` (`down_rolls`, `finger_points`, `neighbour_radius`, `add_column`) |
| held object in the plan (carry, drop): `World.hold`, checker `attach`; drop height and posture | `planning/sim.py`, `planning/ik.py`, `planning/pick_planner.py` `plan_task` |
| gripper close: holding by current or stall, `outputs/gripper_trace.csv` | `hardware/gripper.py` `close_gripper_with_current_feedback`, `state_machine/arm_worker.py` `_log_gripper` |
| blocked objects: retry later, give to the other arm | `state_machine/coordinator.py` (`blocked`, `_resolve_blocked`), `state_machine/arm_worker.py` |
| planning world reuse (one per arm, `World.reset`) | `planning/pick_planner.py` `planning_world()`, `planning/sim.py` |
| planner process per arm, plan-ahead of the next object | `planning/planner_service.py` (`ArmPlanner`); used in `state_machine/arm_worker.py` (`_prefetch_next`, `_take_prefetch`) |
| grasp check: planned vs reached grasp point (real arm), `outputs/grasp_log.csv` | `state_machine/arm_worker.py` `_check_grasp()`, FK in `planning/kinematics.py` |
| which arm does what, regions, handoffs | `state_machine/coordinator.py` |
| one arm's behaviour tree (locate, plan, grasp, carry, verify) | `state_machine/arm_worker.py`; nodes in `state_machine/behaviour.py` |
| game rules (which bin) | `config/games/<game>.yaml` via `games/rules.py` |
| Game 1 main loop, live view | `games/pick_and_place.py` |
| Game 2 (stub) | `games/two_arm_coordination.py` |
| simulate a round in PyBullet (no hardware): scenes, editor, viewer, runner | `simulation/` (`04_simulate.py`); scenes in `simulation/scenes/*.yaml` |
| game options shared by the game and the simulator | `games/pick_and_place.py` `add_game_args()` |
| teleop | `teleop/` (`03_teleop.py`) |
| meshes, URDF, YOLO weights | `models/objects/`, `models/robot/so101/robot.urdf`, `models/yolo/` |
| experiments, old GUIs, dataset tools | `dev/` (nothing outside `dev/` imports it; ignore it unless asked) |

Dependency direction: `games -> state_machine -> planning -> hardware -> core`, with `perception` and
`calibration` beside them; `core` imports nothing from the project.

## Commands

```
python 01_setup_usb.py --list | --check | --install-saved
python 02_calibrate.py status | motors --arm all | sync status|pull|push | offsets --arm both | camera tape|aruco|touch|show
python 03_teleop.py [--gui | net send|receive]
python run_game.py pick_and_place [--round colour|shape|dynamic] [--arm follower_pink] [--no-robot]
python -m planning.sim --direct --no-pause          # planner smoke test, no hardware or camera
conda env create -f environment-sim.yml && conda activate so101-sim   # the simulator's environment (team uses conda)
python 04_simulate.py run example --headless        # a full simulated round: plans, moves, collision check
python 04_simulate.py edit example                   # place objects and bins (OpenCV window)
python 04_simulate.py test                           # regression: every scene in simulation/scenes/tests/
```

## Rules for changes

- Settings belong in `config/*.yaml`, not in code; add a key there and read it through the owning module.
- Keep USB IDs quoted in YAML. Tools that write config use `core.config.save`, which keeps the header comment.
- New game: `games/<game>.py` with `main(argv=None)`, register it in `games/__init__.py` `GAMES`, rules in
  `config/games/<game>.yaml`. Build from the shared packages; do not import from `dev/`.
- A reused planning world must give the same plans as a new one: after changing `World` or `plan_task`, compare
  plan digests with a fresh world (same seeds) and run `04_simulate.py test` (no collisions, nothing in a wrong bin).
- The planner leaves fine joint offsets out of its joint limits on purpose (`planning/sim.py`
  `fine_offsets_in_limits=False`, matching earlier behaviour).
- Hardware (arms, ZED, LeRobot) is not available in a cloud session: test with imports, `--help`,
  `planning.sim --direct` and `04_simulate.py run example --headless` (prints COLLISION lines and a summary).
- The simulator reuses the game's code: change behaviour in `state_machine/`, `planning/` or `games/`, not in
  `simulation/`, which only replaces the camera (scene file) and the motors (`SimActions`, `simulation/viewer.py`). pybullet needs Python <= 3.12 to install from a wheel.
- After adding, moving or renaming files or functions, run `python dev/tools/knowledge_graph.py` and commit
  `docs/knowledge_graph.md` with the change.
