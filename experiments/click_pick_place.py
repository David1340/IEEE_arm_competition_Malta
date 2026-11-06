"""Pick and place with clicked positions instead of YOLO: mark the shapes and the boxes in the camera image, then
the arms pick every shape and drop it into the box of its colour.

    python experiments/click_pick_place.py                       # both arms share the field
    python experiments/click_pick_place.py --arm follower_blue   # one arm only
    python experiments/click_pick_place.py --no-robot --gui      # no arm: plan and show the moves in PyBullet

Dividing the field: each shape goes to the arm whose base is nearest to it (the arms are parallel, so the left half
of the field is follower_pink's and the right half follower_blue's), provided that arm reaches both the shape and its
box within --max-reach; otherwise the other arm takes it if it reaches both, else the shape is skipped. Each mark is
labelled with its arm and distance in the window. The arms move one at a time; the waiting arm is an obstacle.

In the camera window:
    left click        mark a point, then say what it is:
      r g b y           a SHAPE of that colour (click its centre)
      R G B Y / 1-4     the BOX of that colour (1 red, 2 green, 3 blue, 4 yellow; click where to drop)
    u                 undo the last mark          c   clear all marks
    Space / Enter     go: pick every shape (nearest first) and drop it in its box
    h                 send the arms home          q / Esc   quit

The click's 3D point comes from the ZED point cloud and the ArUco calibration (camera_to_table): for a shape it is
the top of the shape, so the grasp height follows the shape. Each pick uses pick_and_place.py's sequence (above
the shape -> down -> close on motor current -> lift) and then goes over the box and opens there. The boxes (8 cm
by default) are obstacles for the planner, and the shape is dropped from --drop-above over the box's rim.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from calibration.calibrate import FILE as CALIBRATION_FILE, arm_from_table, arms, camera_to_table, open_zed  # noqa: E402
from experiments.click_to_go import click_to_camera_xyz  # noqa: E402
from experiments.Full_pipeline import (GRIP_CONFIRM_READINGS, GRIP_CURRENT_MA, GRIP_SQUEEZE_UNITS,  # noqa: E402
                                       GRIPPER_OPEN, HOME_Q, close_gripper_with_current_feedback)
from experiments.pick_and_place import follow, go_home, plan_sequence  # noqa: E402
from experiments.sim_plan import TABLE_CLEARANCE, World  # noqa: E402

WINDOW = "click pick and place"
COLOURS = ("red", "green", "blue", "yellow")
SHAPE_KEYS = {ord("r"): "red", ord("g"): "green", ord("b"): "blue", ord("y"): "yellow"}
BOX_KEYS = {ord("R"): "red", ord("G"): "green", ord("B"): "blue", ord("Y"): "yellow",
            ord("1"): "red", ord("2"): "green", ord("3"): "blue", ord("4"): "yellow"}
BGR = {"red": (40, 40, 230), "green": (40, 180, 40), "blue": (230, 90, 30), "yellow": (30, 220, 240)}
SHAPE_RADIUS = 0.03      # other marked shapes are obstacles: a column of this radius up to their top [m]
# The waiting arm at its home pose (0 0 0 0 -90 deg: upper arm up, forearm forward ~38 cm out at ~24 cm height),
# as an obstacle box in its own base frame: centre x, half length x, half width y, height [m].
ARM_HOME_BOX = (0.17, 0.25, 0.07, 0.32)


def short(arm):
    return arm.replace("follower_", "")


def assign(shape, boxes, bases, max_reach):
    """(arm, box, distance to that arm's base, why not) for a shape: the arm nearest the shape that reaches both the
    shape and the box of its colour; (None, box, None, reason) when there is none."""
    matching = [m for m in boxes if m["colour"] == shape["colour"]]
    if not matching:
        return None, None, None, f"no {shape['colour']} box marked"
    box = matching[-1]
    by_distance = sorted(bases, key=lambda arm: np.linalg.norm(shape["xyz"][:2] - bases[arm]))
    for arm in by_distance:
        to_shape = float(np.linalg.norm(shape["xyz"][:2] - bases[arm]))
        to_box = float(np.linalg.norm(box["xyz"][:2] - bases[arm]))
        if to_shape <= max_reach and to_box <= max_reach:
            return arm, box, to_shape, None
    nearest = by_distance[0]
    return None, box, None, (f"out of reach: {short(nearest)} is {100 * np.linalg.norm(shape['xyz'][:2] - bases[nearest]):.0f} cm "
                             f"from it and {100 * np.linalg.norm(box['xyz'][:2] - bases[nearest]):.0f} cm from its box "
                             f"(max {100 * max_reach:.0f})")


def add_obstacle(world, x, y, half_x, half_y, height):
    """A box from the table up to `height` at (x, y) in the arm frame, as an obstacle for the planners."""
    import pybullet as p

    c = world.client
    half = [half_x, half_y, height / 2]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=c)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.6, 0.4, 0.2, 0.5], physicsClientId=c)
    body = p.createMultiBody(0, col, vis, basePosition=[x, y, height / 2], physicsClientId=c)
    world.checker.obstacle_body_ids.append(body)
    return body


def pick_one(arm, robot, shape, box, others, boxes, args, io, other_bases=()):
    """Plan and (with a robot) do one pick: returns 'placed', 'missed' or 'no plan'. other_bases: the TABLE-frame
    xy of the arms waiting at home, kept clear as obstacles."""
    send_joints, read_joints_rad, arm_files = io
    to_arm = arm_from_table(arm)
    o = (to_arm @ np.append(shape["xyz"], 1.0))[:3]
    b = (to_arm @ np.append(box["xyz"], 1.0))[:3]
    top = max(o[2], 0.0)
    drop_z = max(args.box_height + args.drop_above, top + args.lift)
    print(f"{arm}: {shape['colour']} shape at x {o[0]:.3f} y {o[1]:.3f} (top {100 * top:.1f} cm) "
          f"-> {box['colour']} box at x {b[0]:.3f} y {b[1]:.3f} (drop from {100 * drop_z:.0f} cm)")

    start_q = HOME_Q if robot is None else read_joints_rad(robot)
    world = World([], start_q, gui=args.gui, clearance=args.clearance, arm=arm, calibration_path=arm_files(arm)[1])
    try:
        for other in others:
            xy = (to_arm @ np.append(other["xyz"], 1.0))[:2]
            add_obstacle(world, *xy, SHAPE_RADIUS, SHAPE_RADIUS, max(other["xyz"][2], 0.01))
        for other in boxes:
            xy = (to_arm @ np.append(other["xyz"], 1.0))[:2]
            # the box walls, plus a margin for the shape hanging below the jaws
            add_obstacle(world, *xy, args.box_size / 2, args.box_size / 2, args.box_height + args.box_margin)
        for xy_table in other_bases:        # the arms are parallel, so the box keeps its axes in this frame
            cx, hx, hy, height = ARM_HOME_BOX
            xy = (to_arm @ np.array([xy_table[0] + cx, xy_table[1], 0.0, 1.0]))[:2]
            add_obstacle(world, *xy, hx, hy, height)

        grasp_z = max(top - args.grasp_depth, args.min_grasp_z)
        plans, error = None, None
        for z in np.arange(grasp_z, max(top + 0.015, grasp_z) + 1e-6, 0.005):   # as in pick_and_place.py
            try:
                plans = plan_sequence(world, start_q, [
                    ("above shape", [o[0], o[1], top + args.approach]),
                    ("grasp", [o[0], o[1], z]),
                    ("lift", [o[0], o[1], top + args.lift]),
                    ("over box", [b[0], b[1], drop_z]),
                ], args.seed)
                print(f"  grasp at {100 * z:.1f} cm above the table")
                break
            except RuntimeError as exc:
                error = exc
                if "grasp" not in str(exc):
                    break
        if plans is None:
            print(f"  no plan: {error}")
            return "no plan"
        if args.gui:
            for _, path in plans:
                world.start_q = list(path[0])
                world.play(path)
        if robot is None:
            print("  planned (no arm connected)")
            return "placed"

        (_, to_above), (_, to_grasp), (_, to_lift), (_, to_box) = plans
        follow(robot, to_above, GRIPPER_OPEN, send_joints)
        follow(robot, to_grasp, GRIPPER_OPEN, send_joints)
        grasped, stop, hold, current_ma = close_gripper_with_current_feedback(
            robot, 0.0, args.grip_current_ma, GRIP_CONFIRM_READINGS, GRIP_SQUEEZE_UNITS)
        if not grasped:
            print(f"  missed it (gripper closed to {stop:.0f}/100, {current_ma:.0f} mA)")
            follow(robot, to_lift, GRIPPER_OPEN, send_joints)
            return "missed"
        print(f"  holding it ({current_ma:.0f} mA)")
        follow(robot, to_lift, hold, send_joints)
        follow(robot, to_box, hold, send_joints)
        send_joints(robot, to_box[-1], gripper=GRIPPER_OPEN)
        print("  dropped")
        return "placed"
    finally:
        world.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="both", help="both (default), follower_pink or follower_blue")
    ap.add_argument("--max-reach", type=float, default=0.45,
                    help="an arm only takes shapes and boxes within this distance of its base [m]")
    ap.add_argument("--no-robot", action="store_true", help="no arm connected: plan only")
    ap.add_argument("--gui", action="store_true", help="show each planned move in PyBullet")
    ap.add_argument("--box-height", type=float, default=0.08, help="height of the boxes [m]")
    ap.add_argument("--box-size", type=float, default=0.15, help="width of the boxes, for the planner [m]")
    ap.add_argument("--box-margin", type=float, default=0.03,
                    help="the arm keeps this much above the boxes' rim (room for the shape it holds) [m]")
    ap.add_argument("--drop-above", type=float, default=0.08, help="jaws this high over the box's rim at release [m]")
    ap.add_argument("--approach", type=float, default=0.06, help="height above the shape's top to come down from [m]")
    ap.add_argument("--grasp-depth", type=float, default=0.015, help="jaws this far below the shape's top [m]")
    ap.add_argument("--min-grasp-z", type=float, default=0.04, help="lowest point between the jaws [m]")
    ap.add_argument("--lift", type=float, default=0.10, help="lift height above the shape's top [m]")
    ap.add_argument("--attempts", type=int, default=2, help="grasp attempts per shape")
    ap.add_argument("--clearance", type=float, default=TABLE_CLEARANCE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grip-current-ma", type=float, default=GRIP_CURRENT_MA)
    ap.add_argument("--serial", type=int, help="the ZED's serial number (default: the first camera)")
    ap.add_argument("--width", type=int, default=1280, help="window width in pixels")
    from control.IK_control import add_motion_args, apply_motion_args

    add_motion_args(ap)
    args = ap.parse_args()
    apply_motion_args(args)

    T_table = camera_to_table()
    if T_table is None:
        raise SystemExit(f"No camera_to_table in {CALIBRATION_FILE}: run `python calibration/calibrate.py` first")
    all_arms = arms()
    names = list(all_arms) if args.arm == "both" else [args.arm]
    if any(name not in all_arms for name in names):
        raise SystemExit(f"unknown arm {args.arm!r}; known: both, {', '.join(all_arms)}")

    import pyzed.sl as sl
    from control.IK_control import arm_files, connect_robot, read_joints_rad, send_joints

    io = (send_joints, read_joints_rad, arm_files)
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
                  f"field. Check ports/{name}.txt and the arm's power.")
    if not robots:
        raise SystemExit("no arm connected")
    bases = {name: np.array([all_arms[name]["x"], all_arms[name]["y"]]) for name in robots}
    print("Arms: " + ", ".join(f"{name} at y {bases[name][1]:+.2f}" for name in bases))

    def disconnect_all():
        for robot in robots.values():
            if robot:
                robot.disconnect()

    def home_all():
        for robot in robots.values():
            if robot:
                go_home(robot, send_joints, read_joints_rad)
    marks = []                     # {"kind": "shape"/"box", "colour", "xyz" (table frame), "pixel", "state"}
    state = {"pending": None, "status": "click a shape's centre or a box"}
    frame = cloud = None
    try:
        zed = open_zed(args.serial)
    except BaseException:
        disconnect_all()
        raise
    image, cloud_mat, runtime = sl.Mat(), sl.Mat(), sl.RuntimeParameters()

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or cloud is None:
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
        boxes = [m for m in marks if m["kind"] == "box"]
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
                text = f"{m['colour']} {m['state']}"
                if m["state"] == "todo":
                    arm, _, distance, why = assign(m, boxes, bases, args.max_reach)
                    text += f" -> {short(arm)} ({100 * distance:.0f} cm)" if arm else " -> no arm"
            cv2.putText(view, text, (u + 16, v - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
            cv2.putText(view, text, (u + 16, v - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
        if state["pending"] is not None:
            cv2.drawMarker(view, state["pending"]["pixel"], (255, 255, 255), cv2.MARKER_CROSS, 30, 3)
        for i, text in enumerate([state["status"], "u undo | c clear | Space go | h home | q quit"]):
            cv2.putText(view, text, (15, 35 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
            cv2.putText(view, text, (15, 35 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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
        plan = []
        for shape in shapes:
            arm, box, distance, why = assign(shape, boxes, bases, args.max_reach)
            if arm is None:
                print(f"{shape['colour']} shape at x {shape['xyz'][0]:.2f} y {shape['xyz'][1]:.2f}: skipped, {why}")
                continue
            plan.append((distance, arm, shape, box))
        counts = {name: sum(a == name for _, a, _, _ in plan) for name in bases}
        print("Field split: " + ", ".join(f"{short(n)} {c} shape(s)" for n, c in counts.items()))
        if plan and len(bases) > 1:
            home_all()                       # the waiting arm must really be at home (it is an obstacle there)
        for distance, arm, shape, box in sorted(plan, key=lambda item: (item[1], item[0])):   # per arm, nearest first
            robot = robots[arm]
            state["status"] = f"{short(arm)}: picking the {shape['colour']} shape ..."
            show()
            cv2.waitKey(1)
            others = [m for m in marks if m["kind"] == "shape" and m["state"] == "todo" and m is not shape]
            waiting = [bases[name] for name in bases if name != arm]
            for attempt in range(1, args.attempts + 1):
                result = pick_one(arm, robot, shape, box, others, boxes, args, io, other_bases=waiting)
                if result != "missed":
                    break
                if attempt < args.attempts:
                    print(f"  trying again ({attempt + 1}/{args.attempts})")
            shape["state"] = "placed" if result == "placed" else "failed"
            if robot:
                go_home(robot, send_joints, read_joints_rad)
        done = sum(m["state"] == "placed" for m in marks)
        state["status"] = f"done: {done} placed; mark more or q to quit"

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(WINDOW, on_mouse)
    sized = False
    print(__doc__.split("In the camera window:")[1].split("The click's")[0])
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
            view = show()
            if not sized:
                cv2.resizeWindow(WINDOW, view.shape[1], view.shape[0])
                sized = True

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
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
        disconnect_all()


if __name__ == "__main__":
    main()
