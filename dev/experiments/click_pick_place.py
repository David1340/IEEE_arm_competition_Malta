"""Pick and place with clicked positions instead of YOLO: mark the shapes and the boxes in the camera image, then
the arms pick every shape and drop it into the box of its colour.

    python dev/experiments/click_pick_place.py                       # both arms, one task at a time
    python dev/experiments/click_pick_place.py --parallel            # both arms at the same time where it is safe
    python dev/experiments/click_pick_place.py --arm follower_blue   # one arm only
    python dev/experiments/click_pick_place.py --no-robot --gui      # no arm: plan and show the moves in PyBullet

In the camera window:
    left click        mark a point, then say what it is:
      r g b y           a SHAPE of that colour (click its centre)
      R G B Y / 1-4     the BOX of that colour (1 red, 2 green, 3 blue, 4 yellow; click where to drop)
    u                 undo the last mark          c   clear all marks
    Space / Enter     go: sort every shape into its box
    h                 send the arms home          q / Esc   quit: after the tasks running now, each arm goes back
                                                            to the pose it started in and holds it (torque on)

Who does what (control/coordinator.py): a shape goes to the arm nearest to it if that arm also reaches its box
(DIRECT). If only the other arm reaches the box, the near arm puts the shape down on a handoff point in the middle
(arms.json "handoff") and the other arm takes it from there to the box (HANDOFF). Each mark is labelled with its
plan. The field is split into each arm's side and a middle strip; a task reserves every region it touches before
it starts, and frees them when its arm is home again, so the arms are never in the same region. With --parallel
both arms work at once (e.g. each on its own side); without it, one task runs at a time.

Each arm runs in its own thread; each attempt at a task is one run of a small behaviour tree (task_tree: locate the
object and its bin, plan, grasp or back off, carry, release, home, verify the drop; control/behaviour.py); the planner (sim_plan's World: the table, the other marked shapes, the boxes and the OTHER ARM at its
last known joint angles as obstacles) is used by one arm at a time. The click's 3D point comes from the ZED point
cloud and the ArUco calibration; for a shape it is its top, so the grasp height follows the shape.
"""
import argparse
import math
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from calibration.calibrate import FILE as CALIBRATION_FILE, arm_from_table, arms, camera_to_table, open_zed  # noqa: E402
from core import arms as arm_config  # noqa: E402
from state_machine.coordinator import BOX_SLACK, DIRECT, TO_HANDOFF, Coordinator, Regions, build_tasks, drop_point  # noqa: E402
from dev.experiments.click_to_go import click_to_camera_xyz  # noqa: E402
from hardware.gripper import GRIP_CURRENT_MA, GRIPPER_OPEN  # noqa: E402,F401
from planning.pick_planner import HOME_Q, SEED_TRIES, plan_sequence  # noqa: E402,F401
from planning.sim import JOINTS, TABLE_CLEARANCE, URDF_FILE, World  # noqa: E402

WINDOW = "click pick and place"
SHAPE_KEYS = {ord("r"): "red", ord("g"): "green", ord("b"): "blue", ord("y"): "yellow"}
BOX_KEYS = {ord("R"): "red", ord("G"): "green", ord("B"): "blue", ord("Y"): "yellow",
            ord("1"): "red", ord("2"): "green", ord("3"): "blue", ord("4"): "yellow"}
BGR = {"red": (40, 40, 230), "green": (40, 180, 40), "blue": (230, 90, 30), "yellow": (30, 220, 240)}


# The planning and the arms' workers live in planning/pick_planner.py and state_machine/arm_worker.py (shared with
# games/pick_and_place.py).
from planning.pick_planner import (PLAN_LOCK, add_obstacle, add_other_arm, grasp_height, grasp_heights,  # noqa: E402,F401
                                   handoff_checker, miss_position, plan_task, postures_for)
from state_machine.arm_worker import CURRENT, TaskActions, arm_worker, follow, go_home, short, task_tree  # noqa: E402,F401


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="both", help="both (default), follower_pink or follower_blue")
    ap.add_argument("--parallel", action="store_true",
                    help="both arms at the same time when their tasks are in different regions")
    ap.add_argument("--max-reach", type=float, default=0.45,
                    help="an arm only takes shapes, boxes and handoff points within this distance of its base [m]")
    ap.add_argument("--no-robot", action="store_true", help="no arm connected: plan only")
    ap.add_argument("--gui", action="store_true", help="show each planned move in PyBullet (one task at a time)")
    ap.add_argument("--box-height", type=float, default=0.08, help="height of the boxes [m]")
    ap.add_argument("--box-size", type=float, default=0.15, help="width of the boxes, for the planner [m]")
    ap.add_argument("--box-margin", type=float, default=0.03,
                    help="the arm keeps this much above the boxes' rim (room for the shape it holds) [m]")
    ap.add_argument("--drop-above", type=float, default=0.08, help="jaws this high over the box's rim at release [m]")
    ap.add_argument("--drop-inset", type=float, default=0.04,
                    help="the shape is released this far inside the box's rim nearest the arm (so a box whose centre "
                         "is a little out of reach still works) [m]")
    ap.add_argument("--approach", type=float, default=0.06, help="height above the shape's top to come down from [m]")
    ap.add_argument("--grasp-at", choices=["middle", "top"], default="middle",
                    help="grasp at half the shape's height, or --grasp-depth below its top")
    ap.add_argument("--level-below-top", type=float, default=0.01,
                    help="a level (horizontal) grasp goes at least this far below the shape's top [m]; higher, the "
                         "jaws close over it, so a shorter shape is grasped top-down")
    ap.add_argument("--grasp-depth", type=float, default=0.015, help="--grasp-at top: this far below the top [m]")
    ap.add_argument("--min-grasp-z", type=float, default=0.02,
                    help="lowest grasp point for a level grasp [m] (the table clearance check still applies)")
    ap.add_argument("--lift", type=float, default=0.10, help="lift height above the shape's top [m]")
    ap.add_argument("--close-radius", type=float, default=0.15,
                    help="--pick-posture auto: closer than this to an arm's base, the gripper picks top-down first [m]")
    ap.add_argument("--pick-posture", choices=["level", "auto"], default="auto",
                    help="auto (default): level first, top-down near the base (--close-radius) or when level has no "
                         "pose; level: the fingers always parallel to the table (never top-down; objects too close to "
                         "the base for that are left)")
    ap.add_argument("--level-tolerance", type=float, default=15.0,
                    help="how far a level gripper may tilt from level [deg] (was 30)")
    ap.add_argument("--ik-stall", type=int, default=30,
                    help="the IK search gives up on a pose after this many iterations without progress (0 = run all "
                         "160); unreachable grasp heights then cost ~0.6 s instead of ~1.5 s")
    ap.add_argument("--level-min-contact", type=float, default=0.015,
                    help="--pick-posture level: a short object may be grasped with the grasp point above its top while "
                         "the fingers cover at least this much of it [m]")
    ap.add_argument("--attempts", type=int, default=2, help="grasp attempts per task")
    ap.add_argument("--miss-margin", type=float, default=10.0,
                    help="a grasp is a miss once the gripper closes this many units (0-100) past where the object "
                         "would touch the jaws with no current rise, instead of closing fully (-1 = close fully)")
    ap.add_argument("--neighbour-radius", type=float, default=0.0,
                    help="the other objects as obstacles for the planner: columns this many m either side of their "
                         "centre up to their top. 0 (default): ignored, so objects close together or touching are "
                         "still picked (the gripper may push a neighbour); 0.03 = the old 6 cm columns")
    ap.add_argument("--clearance", type=float, default=0.005,
                    help="the gripper keeps this far above the table in the plans [m] (was 0.01; a level grasp point "
                         "then goes down to 4.0 cm instead of 4.5 cm; raise it if the fingers touch the table)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grip-current-ma", type=float, default=GRIP_CURRENT_MA)
    ap.add_argument("--serial", type=int, help="the ZED's serial number (default: the first camera)")
    ap.add_argument("--width", type=int, default=1280, help="window width in pixels")
    from hardware.arm import add_motion_args, apply_motion_args

    add_motion_args(ap)
    args = ap.parse_args()
    apply_motion_args(args)
    BOX_SLACK["value"] = max(0.0, args.box_size / 2 - args.drop_inset)
    if args.parallel and args.gui:
        raise SystemExit("--gui shows one task at a time: use it without --parallel")

    T_table = camera_to_table()
    if T_table is None:
        raise SystemExit(f"No camera_to_table in {CALIBRATION_FILE}: run `python 02_calibrate.py camera` first")
    all_arms = arms()
    names = list(all_arms) if args.arm == "both" else [args.arm]
    if any(name not in all_arms for name in names):
        raise SystemExit(f"unknown arm {args.arm!r}; known: both, {', '.join(all_arms)}")
    handoff_points = arm_config.handoff_points()

    import pyzed.sl as sl
    from hardware.arm import connect_robot, on_every_arm, park_and_disconnect, read_joints_rad, send_joints

    robots = {}
    for name in names:
        if args.no_robot:
            robots[name] = None
            continue
        try:
            robots[name] = connect_robot(name)
        except (ConnectionError, OSError, RuntimeError) as error:
            if len(names) == 1:
                raise
            print(f"{name}: could not connect ({str(error).strip().splitlines()[0]}); the other arm does the whole "
                  f"field. Check `python 01_setup_usb.py --check` and the arm's power.")
    if not robots:
        raise SystemExit("no arm connected")
    bases = {name: np.array([all_arms[name]["x"], all_arms[name]["y"]]) for name in robots}
    regions = Regions(bases, arm_config.middle_half_width())
    missing = [name for name in names if name not in robots]
    warning = (f"{', '.join(missing)} NOT CONNECTED: no handoffs, {next(iter(robots))} does what it reaches alone"
               if missing else "")
    if warning:
        print("\n*** " + warning + " ***\n")
    print("Arms: " + ", ".join(f"{name} at y {bases[name][1]:+.2f}" for name in bases)
          + f" | handoff points: {handoff_points} | {'parallel' if args.parallel else 'one task at a time'}")
    last_q = {name: list(HOME_Q) for name in robots}
    workers = []
    can_grasp, close_checker = handoff_checker(handoff_points, args)

    def disconnect_all():
        """Let running tasks finish, then park every arm (back to its start pose, torque kept) and disconnect."""
        if CURRENT["coordinator"] is not None:
            CURRENT["coordinator"].stop.set()
        for worker in workers:
            worker.join()
        on_every_arm(robots, lambda name, robot: park_and_disconnect(robot))    # both arms at the same time

    def home_all():
        def home(name, robot):
            go_home(robot, send_joints, read_joints_rad)
            last_q[name] = read_joints_rad(robot)
        on_every_arm(robots, home)                     # both arms at the same time

    def running():
        return any(w.is_alive() for w in workers)

    marks = []                     # {"kind": "shape"/"box", "colour", "xyz" (table frame), "pixel", "state"}
    state = {"pending": None, "status": "click a shape's centre or a box", "plan": {}}
    frame = cloud = None
    try:
        home_all()                           # start at the home pose (control/arm_config.py), out of the camera's view
        zed = open_zed(args.serial)
    except BaseException:
        disconnect_all()
        raise
    image, cloud_mat, runtime = sl.Mat(), sl.Mat(), sl.RuntimeParameters()
    camera_from_table = np.linalg.inv(T_table)
    try:
        from perception.detection import camera_intrinsics

        intrinsics = camera_intrinsics(zed)
    except Exception:
        intrinsics = None              # handoff points are then not drawn

    def to_pixel(xyz):
        """TABLE-frame point -> image pixel (None without the camera's intrinsics)."""
        if intrinsics is None:
            return None
        fx, fy, cx, cy = intrinsics
        x, y, z = camera_from_table[:3, :3] @ np.asarray(xyz, float) + camera_from_table[:3, 3]
        return (int(round(fx * x / z + cx)), int(round(fy * y / z + cy))) if z > 0 else None

    def plan_labels():
        """{id(mark): text} of what will happen to each todo shape."""
        shapes = [m for m in marks if m["kind"] == "shape" and m["state"] == "todo"]
        tasks, skipped = build_tasks(shapes, [m for m in marks if m["kind"] == "box"], bases, handoff_points,
                                     args.max_reach, regions, can_grasp)
        labels = {id(t.shape): (f"-> {short(t.arm)}" if t.kind == DIRECT else
                                f"-> {short(t.arm)} -> handoff -> {short(t.then_arm)}") for t in tasks}
        labels.update({id(s): "-> no arm" for s, _ in skipped})
        return labels

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or cloud is None or running():
            return
        scale = frame.shape[1] / args.width if args.width < frame.shape[1] else 1.0
        u, v = int(round(x * scale)), int(round(y * scale))
        xyz_cam = click_to_camera_xyz(cloud, u, v)
        if xyz_cam is None:
            state.update(pending=None, status="no depth at that pixel: click again")
            return
        xyz = T_table[:3, :3] @ xyz_cam + T_table[:3, 3]
        state.update(pending={"pixel": (u, v), "xyz": xyz},
                     status="shape: r g b y | box: R G B Y (or 1 2 3 4)")

    def draw(view):
        for i, (hx, hy) in enumerate(handoff_points):
            pixel = to_pixel([hx, hy, 0.0])
            if pixel:
                cv2.drawMarker(view, pixel, (255, 255, 255), cv2.MARKER_DIAMOND, 22, 2)
                cv2.putText(view, f"H{i}", (pixel[0] + 12, pixel[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2)
        for m in marks:
            colour = BGR[m["colour"]]
            u, v = m["pixel"]
            if m["kind"] == "box":
                cv2.rectangle(view, (u - 22, v - 22), (u + 22, v + 22), colour, 3)
                text = f"{m['colour']} box"
            else:
                cv2.circle(view, (u, v), 12, colour, -1 if m["state"] == "todo" else 2)
                if m["state"] == "placed":
                    cv2.line(view, (u - 12, v), (u + 12, v), colour, 2)
                elif m["state"] == "failed":
                    cv2.drawMarker(view, (u, v), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 26, 3)
                elif m["state"] == "handoff" and m.get("handoff_xyz") is not None:
                    pixel = to_pixel(m["handoff_xyz"])
                    if pixel:
                        cv2.circle(view, pixel, 12, colour, -1)
                        cv2.arrowedLine(view, (u, v), pixel, colour, 2, tipLength=0.08)
                text = f"{m['colour']} {m['state']} {state['plan'].get(id(m), '')}"
            cv2.putText(view, text, (u + 16, v - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
            cv2.putText(view, text, (u + 16, v - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
        if state["pending"] is not None:
            cv2.drawMarker(view, state["pending"]["pixel"], (255, 255, 255), cv2.MARKER_CROSS, 30, 3)
        coordinator = CURRENT["coordinator"]
        lines = [state["status"], "u undo | c clear | Space go | h home | q quit"] + ([warning] if warning else [])
        if coordinator is not None:
            lines += [f"{short(a)}: {s}" for a, s in sorted(coordinator.arm_state.items())]
        for i, text in enumerate(lines):
            cv2.putText(view, text, (15, 35 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5)
            cv2.putText(view, text, (15, 35 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def show():
        view = frame.copy()
        draw(view)
        if args.width < view.shape[1]:
            view = cv2.resize(view, (args.width, round(view.shape[0] * args.width / view.shape[1])))
        cv2.imshow(WINDOW, view)
        return view

    def go():
        shapes = [m for m in marks if m["kind"] == "shape" and m["state"] == "todo"]
        boxes = [m for m in marks if m["kind"] == "box"]
        if not shapes:
            state["status"] = "no shapes left to pick: mark some first"
            return
        tasks, skipped = build_tasks(shapes, boxes, bases, handoff_points, args.max_reach, regions, can_grasp)
        for shape, why in skipped:
            print(f"{shape['colour']} shape at x {shape['xyz'][0]:.2f} y {shape['xyz'][1]:.2f}: skipped, {why}")
        if not tasks:
            state["status"] = "nothing the arms can do: see the terminal"
            if warning:
                print(f"\n*** {warning} ***")
            return
        print("\nPlan:")
        for task in tasks:
            then = f", then {task.then_arm} to the {task.box['colour']} box" if task.kind == TO_HANDOFF else ""
            print(f"  {task.label()}{then}")
        home_all()                           # every arm starts at home (its pose is an obstacle for the other)
        coordinator = Coordinator(tasks, regions, handoff_points, args.parallel, arm_config.handoff_min_spacing())
        CURRENT["coordinator"] = coordinator
        workers.clear()
        for name, robot in robots.items():
            worker = threading.Thread(target=arm_worker, daemon=True, name=name,
                                      args=(name, robot, coordinator, marks, last_q, args,
                                            (send_joints, read_joints_rad), handoff_points))
            workers.append(worker)
            worker.start()
        state["status"] = "working ... (q stops after the tasks running now)"

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(WINDOW, on_mouse)
    sized, was_running, last_report = False, False, time.monotonic()
    print(__doc__.split("In the camera window:")[1].split("Who does what")[0])
    try:
        while True:
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                zed.retrieve_measure(cloud_mat, sl.MEASURE.XYZRGBA)
                frame = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)
                cloud = np.array(cloud_mat.get_data()[:, :, :3], np.float32)
            if frame is None:
                cv2.waitKey(1)
                continue
            busy = running()
            if busy and CURRENT["coordinator"] is not None and time.monotonic() - last_report > 15:
                last_report = time.monotonic()
                print("  status: " + " | ".join(f"{short(a)}: {s}"
                                                 for a, s in sorted(CURRENT["coordinator"].arm_state.items())))
            if was_running and not busy:
                done = sum(m["state"] == "placed" for m in marks if m["kind"] == "shape")
                state["status"] = f"done: {done} placed; mark more or q to quit"
            was_running = busy
            if not busy:
                state["plan"] = plan_labels()
            view = show()
            if not sized:
                cv2.resizeWindow(WINDOW, view.shape[1], view.shape[0])
                sized = True

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                if busy:
                    print("Stopping after the tasks running now ...")
                    CURRENT["coordinator"].stop.set()
                    for worker in workers:
                        worker.join()
                break
            if busy:
                continue                    # no marking while the arms work
            pending = state["pending"]
            if pending is not None and (key in SHAPE_KEYS or key in BOX_KEYS):
                kind, colour = ("shape", SHAPE_KEYS[key]) if key in SHAPE_KEYS else ("box", BOX_KEYS[key])
                if kind == "box":       # one box per colour: a new one replaces the old
                    marks[:] = [m for m in marks if not (m["kind"] == "box" and m["colour"] == colour)]
                marks.append({"kind": kind, "colour": colour, "state": "todo", **pending})
                x, y, z = pending["xyz"]
                print(f"{colour} {kind} at x {x:.3f} y {y:.3f} z {z:.3f} m (table frame)")
                state.update(pending=None, status=f"{colour} {kind} marked; click the next one or Space to go")
            elif key == ord("u") and marks:
                removed = marks.pop()
                state["status"] = f"removed the {removed['colour']} {removed['kind']}"
            elif key == ord("c"):
                marks.clear()
                state.update(pending=None, status="all marks cleared")
            elif key in (32, 13, 10):           # Space or Enter
                state["pending"] = None
                go()
            elif key == ord("h"):
                home_all()
                state["status"] = "home"
    finally:
        zed.close()
        cv2.destroyAllWindows()
        close_checker()
        disconnect_all()


if __name__ == "__main__":
    main()
