"""Play a pick-and-place round in PyBullet on a scene: no arms, no camera, the same code as the real game.

    python 04_simulate.py run example                       # simulation/scenes/example.yaml, both arms
    python 04_simulate.py run example --round shape         # any game option works (python 04_simulate.py run -h)
    python 04_simulate.py run example --speed 3             # faster playback
    python 04_simulate.py run example --headless            # no window: plan and check only (e.g. in CI)

What is the same as on the real arms: the game's options and rules (games/pick_and_place.py add_game_args,
config/games/pick_and_place.yaml), which arm does what and when (state_machine/coordinator.py), each arm's behaviour
tree (state_machine/arm_worker.py), the planner (planning/pick_planner.py: IK, RRT-Connect, the other arm, bins and
table as obstacles), the gripper openings (hardware/gripper.py) and the arms' bases (config/arms.yaml).
What is simulated: the camera is the scene file (exact positions; no YOLO, no orientation), the motors follow the
planned paths exactly, a grasp always holds, and a released object lands upright where it is. So this tests the
planning, the collision avoidance and the two-arm coordination, not perception or grasp physics.

Every animation step is checked for contact (simulation/viewer.py): any COLLISION is printed, and counted at the end.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import arms as arm_config  # noqa: E402
from games import rules  # noqa: E402
from hardware import gripper as jaw  # noqa: E402
from hardware.gripper import GRIPPER_OPEN  # noqa: E402
from simulation import scene as scenes  # noqa: E402
from state_machine.arm_worker import TaskActions, short  # noqa: E402


JOINT_SPEED = 90.0      # deg/s the viewer plays an ordinary move at (simulation/viewer.py JOINT_SPEED_DEG)


class SimArm:
    """Stands in for a connected follower: its joint angles are wherever the viewer last put them."""

    def __init__(self, name, viewer, q):
        self.name, self.viewer, self.q = name, viewer, list(q)


def sim_read_joints(robot):
    return list(robot.viewer.q[robot.name])


def sim_send_joints(robot, q, gripper=None):
    robot.viewer.run(robot.name, "move", [q], robot.viewer.grip[robot.name] if gripper is None else gripper)


class SimActions(TaskActions):
    """The arm's behaviour-tree actions, moving the arm in the viewer instead of the motors. Seeing, claiming
    regions and planning are the real ones (TaskActions)."""

    def _source(self, bb):
        shape = bb["task"].shape
        return shape.get("origin", shape)["source"]

    def _follow(self, bb, step, gripper):
        self.robot.viewer.run(self.arm, "move", dict(bb["plans"])[step], gripper)

    def grasp(self, bb):
        task = bb["task"]
        bb["opening"] = task.shape.get("open_cmd", GRIPPER_OPEN)
        viewer = self.robot.viewer
        viewer.run(self.arm, "handle", self._source(bb))
        self._follow(bb, "above shape", bb["opening"])
        self._follow(bb, "grasp", bb["opening"])
        o = viewer.scene_objects[self._source(bb)]
        contact = jaw.open_command(jaw.object_footprint(scenes.stl(o["shape"], o["size"]))[0], margin=0.0, min_open=0.0)
        viewer.run(self.arm, "gripper", contact)            # close until the jaws meet the object
        viewer.run(self.arm, "attach", self._source(bb))
        bb["hold"] = contact
        print(f"  {self.arm}: holding it (simulated)")
        return True

    def release(self, bb):
        bb["released"] = True
        viewer = self.robot.viewer
        viewer.run(self.arm, "gripper", bb.get("opening", GRIPPER_OPEN))     # as TaskActions.release
        viewer.run(self.arm, "release")
        return True

    def home(self, bb):
        from planning.pick_planner import HOME_Q

        # quick after a release (or a miss), as on the real arms: on screen at the real ratio of --return-speed
        # (60 deg/s) to --speed (20 deg/s)
        quick = bb.get("released") or bb.get("missed")
        scale = JOINT_SPEED * 60.0 / 20.0 if quick else None
        self.robot.viewer.run(self.arm, "move", [list(HOME_Q)], self.robot.viewer.grip[self.arm], scale)
        self.last_q[self.arm] = sim_read_joints(self.robot)
        self.home_at = time.monotonic()
        return True


def build_round(scene, args, bases, regions):
    """The round's tasks from the scene, as games/pick_and_place.py go() builds them from the camera's detections:
    (tasks, skipped, marks, handoff_points, can_grasp, close_checker)."""
    from planning.pick_planner import handoff_checker
    from state_machine.coordinator import build_tasks, handoff_area

    objects, bins = scenes.marks(scene)
    boxes = [{"kind": "box", "colour": b["colour"], "xyz": b["xyz"], "pixel": b["pixel"], "state": "todo",
              "conf": b["conf"]} for b in bins]
    shapes = []
    for o in objects:
        wanted = rules.bin_colour(o["name"], o["colour"], args.round)
        label = f"{o['colour']} {o['size']} {o['name']} #{o['source']}"
        if wanted is None or not any(b["colour"] == wanted for b in boxes):
            print(f"{label}: left, {'no rule' if wanted is None else f'no {wanted} bin'}")
            continue
        opening = (args.grip_open if args.grip_open is not None
                   else jaw.open_command(jaw.object_width(scenes.stl(o["name"], o["size"])), args.grip_margin))
        shapes.append({"kind": "shape", "colour": wanted, "label": label, "name": o["name"],
                       "object_colour": o["colour"], "size": o["size"], "xyz": o["xyz"], "pixel": o["pixel"],
                       "state": "todo", "open_cmd": opening, "yaw": o.get("yaw"), "source": o["source"]})
    fixed = [list(p) for p in arm_config.handoff_points()] if len(bases) > 1 else []
    area = handoff_area(bases, args.max_reach, args.min_reach, regions, step=args.drop_step) if len(bases) > 1 else []
    taken = [o["xyz"][:2] for o in objects] + [b["xyz"][:2] for b in bins]
    handoff_points = [p for p in fixed + area if all(np.linalg.norm(np.asarray(p) - t) >= 0.07 for t in taken)]
    can_grasp, close_checker = handoff_checker(handoff_points, args)
    tasks, skipped = build_tasks(shapes, boxes, bases, handoff_points, args.max_reach, regions, can_grasp,
                                 max_usable=3)
    return tasks, skipped, shapes + boxes, handoff_points, close_checker


def main(argv=None):
    from games.pick_and_place import add_game_args

    ap = argparse.ArgumentParser(prog="04_simulate.py run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", nargs="?", default="example",
                    help=f"a scene file, or a name in simulation/scenes/ ({', '.join(scenes.available())})")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (2 = twice as fast)")
    ap.add_argument("--headless", action="store_true", help="no window: plan, move and check as fast as possible")
    ap.add_argument("--keep-open", action="store_true", help="leave the window open at the end until Ctrl+C")
    ap.add_argument("--realtime", action="store_true",
                    help="with --headless: the moves still take their time (at --speed), to compare round times")
    add_game_args(ap)
    args = ap.parse_args(argv)
    args.gui = False                                   # the planners' own worlds stay headless
    args.camera = None

    import planning.ik
    from planning.pick_planner import HOME_Q
    from simulation.viewer import Viewer
    from state_machine import arm_worker as workers_module
    from state_machine.coordinator import BOX_SLACK, DIRECT, Coordinator, Regions

    planning.ik.ALLOW_URDF_LIMITS = True               # a laptop without the arms' motor calibration still plans
    BOX_SLACK["value"] = max(0.0, args.box_size / 2 - args.drop_inset)
    scene_path = scenes.resolve(args.scene)
    scene = scenes.load(scene_path)
    all_arms = arm_config.followers()
    names = list(all_arms) if args.arm == "both" else [args.arm]
    if any(n not in all_arms for n in names):
        raise SystemExit(f"unknown arm {args.arm!r}; known: {', '.join(all_arms)} or both")
    print(f"Scene {scene_path}: {len(scene['objects'])} objects, {len(scene['bins'])} bins | "
          f"sorting {rules.describe(args.round)}")

    viewer = Viewer(scene, args, gui=not args.headless, speed=args.speed, realtime=args.realtime)
    viewer.scene_objects = scene["objects"]
    bases = {n: np.array([arm_config.base(n)["x"], arm_config.base(n)["y"]]) for n in names}
    regions = Regions(bases, arm_config.middle_half_width())
    robots = {n: SimArm(n, viewer, HOME_Q) for n in names}
    for name in viewer.arms:                           # every arm at home (an unused one stays there)
        viewer._apply(name, list(HOME_Q), arm_config.HOME_GRIPPER)
    last_q = {n: list(HOME_Q) for n in names}

    from planning.planner_service import ArmPlanner

    planners = {n: ArmPlanner(n, args, args.planner, HOME_Q) for n in names}   # build each arm's world meanwhile
    planning_started = time.monotonic()
    tasks, skipped, marks, handoff_points, close_checker = build_round(scene, args, bases, regions)
    for shape, why in skipped:
        print(f"{shape['label']}: left, {why}")
    print(f"\nPlan ({time.monotonic() - planning_started:.1f} s to check the drop-off points):")
    for task in tasks:
        then = f", then {task.then_arm} to the {task.box['colour']} bin" if task.then_arm else ""
        print(f"  {task.label()}{then}")
    if not tasks:
        print("Nothing to do.")
    coordinator = Coordinator(tasks, regions, handoff_points, not args.one_at_a_time,
                              arm_config.handoff_min_spacing(), max_reach=args.max_reach)
    workers_module.CURRENT["coordinator"] = coordinator
    workers = [threading.Thread(target=workers_module.arm_worker, daemon=True, name=name,
                                args=(name, robot, coordinator, marks, last_q, args,
                                      (sim_send_joints, sim_read_joints), handoff_points, None, SimActions,
                                      planners[name]))
               for name, robot in robots.items()]
    started = time.monotonic()
    placed = failed = wrong = 0
    for worker in workers:
        worker.start()
    try:
        last_status = 0.0
        while any(w.is_alive() for w in workers) or viewer.busy():
            status = None
            if time.monotonic() - last_status > 0.3:
                last_status = time.monotonic()
                status = [f"{short(a)}: {s}" for a, s in sorted(coordinator.arm_state.items())]
            wait = viewer.step(status)
            time.sleep(wait if wait else 0.002)
        took = time.monotonic() - started
        placed = sum(m["state"] == "placed" for m in marks if m["kind"] == "shape")
        failed = sum(m["state"] == "failed" for m in marks if m["kind"] == "shape")
        print(f"\nRound done in {took:.1f} s (planning and playback): {placed} placed, {failed} failed, "
              f"{len(skipped)} left before the start")
        final = viewer.object_final()
        bin_colours = [b["colour"] for b in scene["bins"]]
        wrong = 0
        for i, (o, (x, y, _, inside)) in enumerate(zip(scene["objects"], final)):
            wanted = rules.bin_colour(o["shape"], o["colour"], args.round)
            where = f"{bin_colours[inside]} bin" if inside is not None else f"table at x {x:.2f} y {y:.2f}"
            ok = inside is not None and bin_colours[inside] == wanted
            wrong += inside is not None and not ok
            print(f"  #{i} {o['colour']} {o['size']} {o['shape']}: {where}" + ("" if ok or wanted is None
                                                                            else f"  (belongs in {wanted})"))
        hits = sorted({(a, w) for a, w, _ in viewer.collisions})
        print(f"Collisions: {len(viewer.collisions)} contact steps"
              + (": " + ", ".join(f"{a} with {w}" for a, w in hits) if hits else " (none)")
              + (f" | {wrong} object(s) in the wrong bin" if wrong else ""))
        if args.keep_open and not args.headless:
            print("Window left open: Ctrl+C to quit")
            while True:
                viewer.step()
                time.sleep(0.05)
    except KeyboardInterrupt:
        coordinator.stop.set()
    finally:
        close_checker()
        for planner in planners.values():
            planner.close()
        viewer.close()
    return {"placed": placed, "failed": failed, "left": len(skipped), "collisions": len(viewer.collisions),
            "wrong_bin": wrong, "seconds": round(time.monotonic() - started, 1),
            "objects": len(scene["objects"])}


def test(argv=None):
    """Run scenes headless one after another and print one line each: the regression check after a change to the
    planner, the coordinator or the behaviour tree.

        python 04_simulate.py test                       # every scene in simulation/scenes/tests/, and example
        python 04_simulate.py test tests/side_by_side    # only these scenes
        python 04_simulate.py test -- --down-roll across # game options after --, for every scene
        python 04_simulate.py test --gui                 # watch each scene in the PyBullet window, one after another
        python 04_simulate.py test --gui -- --speed 3    # ... faster

    Each scene's full printout goes to outputs/sim_tests/<scene>.log."""
    import contextlib
    import io

    from core.paths import OUTPUTS

    argv = list(argv or [])
    options = argv[argv.index("--") + 1:] if "--" in argv else []
    names = argv[:argv.index("--")] if "--" in argv else argv
    gui = "--gui" in names
    names = [n for n in names if n != "--gui"]
    names = names or ["example", *[n for n in scenes.available() if n.startswith("tests/")]]
    logs = OUTPUTS / "sim_tests"
    logs.mkdir(parents=True, exist_ok=True)
    rows, bad = [], 0
    for name in names:
        if gui:
            print(f"{name:28s} playing in the window ...", flush=True)
        out = io.StringIO()
        log = logs / f"{name.replace('/', '_')}.log"
        log.write_text("")
        try:
            # PyBullet prints from C straight to file descriptor 1: send that to the log as well
            sys.stdout.flush()
            saved = os.dup(1)
            with open(log, "ab") as native:
                os.dup2(native.fileno(), 1)
            try:
                with contextlib.redirect_stdout(out):
                    result = main([name, *([] if gui else ["--headless"]), *options])
            finally:
                sys.stdout.flush()
                os.dup2(saved, 1)
                os.close(saved)
        except BaseException as error:              # noqa: BLE001 - one broken scene must not stop the others
            result = {"error": f"{type(error).__name__}: {error}"}
        with open(log, "a") as handle:
            handle.write(out.getvalue())
        rows.append((name, result))
        ok = "error" not in result and result["collisions"] == 0 and result["wrong_bin"] == 0
        bad += not ok
        print(f"{name:28s} " + (result["error"] if "error" in result else
              f"placed {result['placed']}/{result['objects']}  failed {result['failed']}  left {result['left']}  "
              f"collisions {result['collisions']}  wrong bin {result['wrong_bin']}  {result['seconds']:.0f} s")
              + ("" if ok else "   <-- check"), flush=True)
    print(f"\n{len(rows)} scenes, {bad} with collisions, wrong bins or errors; logs in {logs}")
    return rows


if __name__ == "__main__":
    main()
