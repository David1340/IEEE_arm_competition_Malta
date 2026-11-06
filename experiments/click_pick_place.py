"""Pick and place with clicked positions instead of YOLO: mark the shapes and the boxes in the camera image, then
the arms pick every shape and drop it into the box of its colour.

    python experiments/click_pick_place.py                       # both arms, one task at a time
    python experiments/click_pick_place.py --parallel            # both arms at the same time where it is safe
    python experiments/click_pick_place.py --arm follower_blue   # one arm only
    python experiments/click_pick_place.py --no-robot --gui      # no arm: plan and show the moves in PyBullet

In the camera window:
    left click        mark a point, then say what it is:
      r g b y           a SHAPE of that colour (click its centre)
      R G B Y / 1-4     the BOX of that colour (1 red, 2 green, 3 blue, 4 yellow; click where to drop)
    u                 undo the last mark          c   clear all marks
    Space / Enter     go: sort every shape into its box
    h                 send the arms home          q / Esc   quit (after the tasks running now)

Who does what (control/coordinator.py): a shape goes to the arm nearest to it if that arm also reaches its box
(DIRECT). If only the other arm reaches the box, the near arm puts the shape down on a handoff point in the middle
(arms.json "handoff") and the other arm takes it from there to the box (HANDOFF). Each mark is labelled with its
plan. The field is split into each arm's side and a middle strip; a task reserves every region it touches before
it starts, and frees them when its arm is home again, so the arms are never in the same region. With --parallel
both arms work at once (e.g. each on its own side); without it, one task runs at a time.

Each arm runs a small state machine (waiting -> planning -> picking -> carrying -> placing/dropping -> homing) in its
own thread; the planner (sim_plan's World: the table, the other marked shapes, the boxes and the OTHER ARM at its
last known joint angles as obstacles) is used by one arm at a time. The click's 3D point comes from the ZED point
cloud and the ArUco calibration; for a shape it is its top, so the grasp height follows the shape.
"""
import argparse
import math
import sys
import threading
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from calibration.calibrate import FILE as CALIBRATION_FILE, arm_from_table, arms, camera_to_table, open_zed  # noqa: E402
from control import arm_config  # noqa: E402
from control.coordinator import DIRECT, TO_HANDOFF, Coordinator, Regions, build_tasks  # noqa: E402
from experiments.click_to_go import click_to_camera_xyz  # noqa: E402
from experiments.Full_pipeline import (GRIP_CONFIRM_READINGS, GRIP_CURRENT_MA, GRIP_SQUEEZE_UNITS,  # noqa: E402
                                       GRIPPER_OPEN, HOME_Q, close_gripper_with_current_feedback)
from experiments.pick_and_place import follow, go_home, plan_sequence  # noqa: E402
from experiments.sim_plan import JOINTS, TABLE_CLEARANCE, URDF_FILE, World  # noqa: E402

WINDOW = "click pick and place"
SHAPE_KEYS = {ord("r"): "red", ord("g"): "green", ord("b"): "blue", ord("y"): "yellow"}
BOX_KEYS = {ord("R"): "red", ord("G"): "green", ord("B"): "blue", ord("Y"): "yellow",
            ord("1"): "red", ord("2"): "green", ord("3"): "blue", ord("4"): "yellow"}
BGR = {"red": (40, 40, 230), "green": (40, 180, 40), "blue": (230, 90, 30), "yellow": (30, 220, 240)}
SHAPE_RADIUS = 0.03      # other marked shapes are obstacles: a column of this radius up to their top [m]
PLAN_LOCK = threading.Lock()   # PyBullet is used by one thread at a time
CURRENT = {"coordinator": None}  # the running coordinator (for tests and the window)


def short(arm):
    return arm.replace("follower_", "")


# ---------------------------------------------------------------------------------------------- the planning world
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


def add_other_arm(world, to_arm, other, q):
    """The other arm's robot model at its base and joint angles q, as an obstacle (it is not planned)."""
    import pybullet as p

    c = world.client
    b = arm_config.base(other)
    xy = (to_arm @ np.array([b["x"], b["y"], 0.0, 1.0]))[:2]
    yaw = math.atan2(to_arm[1, 0], to_arm[0, 0]) + math.radians(b["yaw_deg"])
    body = p.loadURDF(str(URDF_FILE), basePosition=[*xy, 0.0], baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]),
                      useFixedBase=True, physicsClientId=c)
    index = {p.getJointInfo(body, i, physicsClientId=c)[1].decode(): i
             for i in range(p.getNumJoints(body, physicsClientId=c))}
    for name, value in zip(JOINTS, q):
        p.resetJointState(body, index[name], float(value), physicsClientId=c)
    world.checker.obstacle_body_ids.append(body)
    return body


def plan_task(task, start_q, obstacles, args, handoff_xyz=None):
    """Plan the whole task in the arm's frame. Returns ([(label, path)], grasp z) or raises RuntimeError."""
    arm = task.arm
    to_arm = arm_from_table(arm)
    o = (to_arm @ np.append(task.shape["xyz"], 1.0))[:3]
    b = (to_arm @ np.append(task.box["xyz"], 1.0))[:3]
    top = max(o[2], 0.0)
    world = World([], start_q, gui=args.gui, clearance=args.clearance, arm=arm,
                  calibration_path=arm_config.follower_calibration(arm))
    try:
        for other in obstacles["shapes"]:
            if other is task.shape or other is task.shape.get("origin"):
                continue
            xyz = other.get("handoff_xyz") if other["state"] == "handoff" else other["xyz"]
            xy = (to_arm @ np.append(xyz, 1.0))[:2]
            add_obstacle(world, *xy, SHAPE_RADIUS, SHAPE_RADIUS, max(xyz[2], 0.01))
        for box in obstacles["boxes"]:
            xy = (to_arm @ np.append(box["xyz"], 1.0))[:2]
            add_obstacle(world, *xy, args.box_size / 2, args.box_size / 2, args.box_height + args.box_margin)
        for other, q in obstacles["arms"].items():
            if other != arm:
                add_other_arm(world, to_arm, other, q)

        if task.kind == TO_HANDOFF:
            h = (to_arm @ np.array([*handoff_xyz[:2], 0.0, 1.0]))[:3]
            rest = lambda z: [("over handoff", [h[0], h[1], top + args.lift]), ("place", [h[0], h[1], z]),
                              ("leave handoff", [h[0], h[1], top + args.approach])]
        else:
            drop_z = max(args.box_height + args.drop_above, top + args.lift)
            rest = lambda z: [("over box", [b[0], b[1], drop_z])]
        grasp_z = max(top - args.grasp_depth, args.min_grasp_z)
        error = None
        for z in np.arange(grasp_z, max(top + 0.015, grasp_z) + 1e-6, 0.005):   # as in pick_and_place.py
            try:
                plans = plan_sequence(world, start_q, [
                    ("above shape", [o[0], o[1], top + args.approach]),
                    ("grasp", [o[0], o[1], z]),
                    ("lift", [o[0], o[1], top + args.lift]),
                    *rest(z),
                ], args.seed)
                if args.gui:
                    for _, path in plans:
                        world.start_q = list(path[0])
                        world.play(path)
                return plans, z
            except RuntimeError as exc:
                error = exc
                if "grasp" not in str(exc) and "place" not in str(exc):
                    break
        raise RuntimeError(str(error))
    finally:
        world.close()


def execute(task, robot, plans, args, send_joints):
    """Do the planned task. Returns 'done' or 'missed'."""
    steps = dict(plans)
    follow(robot, steps["above shape"], GRIPPER_OPEN, send_joints)
    follow(robot, steps["grasp"], GRIPPER_OPEN, send_joints)
    grasped, stop, hold, current_ma = close_gripper_with_current_feedback(
        robot, 0.0, args.grip_current_ma, GRIP_CONFIRM_READINGS, GRIP_SQUEEZE_UNITS)
    if not grasped:
        print(f"  {task.arm}: missed it (gripper closed to {stop:.0f}/100, {current_ma:.0f} mA)")
        follow(robot, steps["lift"], GRIPPER_OPEN, send_joints)
        return "missed"
    print(f"  {task.arm}: holding it ({current_ma:.0f} mA)")
    follow(robot, steps["lift"], hold, send_joints)
    if task.kind == TO_HANDOFF:
        follow(robot, steps["over handoff"], hold, send_joints)
        follow(robot, steps["place"], hold, send_joints)
        send_joints(robot, steps["place"][-1], gripper=GRIPPER_OPEN)        # let go on the table
        follow(robot, steps["leave handoff"], GRIPPER_OPEN, send_joints)
    else:
        follow(robot, steps["over box"], hold, send_joints)
        send_joints(robot, steps["over box"][-1], gripper=GRIPPER_OPEN)
    return "done"


# ---------------------------------------------------------------------------------------------- an arm's worker
def arm_worker(arm, robot, coordinator, marks, last_q, args, io, handoff_points):
    """The arm's state machine: waiting -> planning -> picking/carrying/placing -> homing, task after task."""
    send_joints, read_joints_rad = io
    say = lambda text: coordinator.set_state(arm, text)
    try:
        while True:
            say("waiting")
            task = coordinator.next_task(arm)
            if task is None:
                break
            say(f"planning {task.label()}")
            print(f"\n{task.label()}  (regions: {', '.join(sorted(task.regions))})")
            handoff_xyz = None
            if task.kind == TO_HANDOFF:
                hx, hy = handoff_points[task.slot]
                handoff_xyz = np.array([hx, hy, task.shape["xyz"][2]])     # same top height: it stands on the table
            obstacles = {"shapes": [m for m in marks if m["kind"] == "shape" and m["state"] in ("todo", "handoff")],
                         "boxes": [m for m in marks if m["kind"] == "box"], "arms": dict(last_q)}
            start_q = last_q[arm] if robot is None else read_joints_rad(robot)
            origin = task.shape.get("origin", task.shape)
            try:
                with PLAN_LOCK:
                    plans, z = plan_task(task, start_q, obstacles, args, handoff_xyz)
                print(f"  {arm}: grasp at {100 * z:.1f} cm above the table")
            except RuntimeError as error:
                print(f"  {arm}: no plan: {error}")
                origin["state"] = "failed"
                coordinator.finish(task, "failed")
                continue
            if robot is None:
                result = "done"                       # planned only
            else:
                say(f"moving: {task.label()}")
                result = execute(task, robot, plans, args, send_joints)
                say("homing")
                go_home(robot, send_joints, read_joints_rad)
                last_q[arm] = read_joints_rad(robot)
            if result == "missed":
                if task.attempts + 1 < args.attempts:
                    print(f"  {arm}: trying again ({task.attempts + 2}/{args.attempts})")
                    coordinator.requeue(task)
                else:
                    origin["state"] = "failed"
                    coordinator.finish(task, "failed")
                continue
            if task.kind == TO_HANDOFF:
                origin["state"] = "handoff"
                origin["handoff_xyz"] = handoff_xyz
                print(f"  {arm}: left it on handoff point {task.slot} for {task.then_arm}")
            else:
                origin["state"] = "placed"
                print(f"  {arm}: dropped it in the {task.box['colour']} box")
            coordinator.finish(task, "done", handoff_xyz)
    except Exception:
        traceback.print_exc()
        print(f"{arm}: stopped after an error; the other arm carries on with its own tasks")
        say("ERROR (see the terminal)")
        return
    say("done")


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
    ap.add_argument("--approach", type=float, default=0.06, help="height above the shape's top to come down from [m]")
    ap.add_argument("--grasp-depth", type=float, default=0.015, help="jaws this far below the shape's top [m]")
    ap.add_argument("--min-grasp-z", type=float, default=0.04, help="lowest point between the jaws [m]")
    ap.add_argument("--lift", type=float, default=0.10, help="lift height above the shape's top [m]")
    ap.add_argument("--attempts", type=int, default=2, help="grasp attempts per task")
    ap.add_argument("--clearance", type=float, default=TABLE_CLEARANCE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grip-current-ma", type=float, default=GRIP_CURRENT_MA)
    ap.add_argument("--serial", type=int, help="the ZED's serial number (default: the first camera)")
    ap.add_argument("--width", type=int, default=1280, help="window width in pixels")
    from control.IK_control import add_motion_args, apply_motion_args

    add_motion_args(ap)
    args = ap.parse_args()
    apply_motion_args(args)
    if args.parallel and args.gui:
        raise SystemExit("--gui shows one task at a time: use it without --parallel")

    T_table = camera_to_table()
    if T_table is None:
        raise SystemExit(f"No camera_to_table in {CALIBRATION_FILE}: run `python calibration/calibrate.py` first")
    all_arms = arms()
    names = list(all_arms) if args.arm == "both" else [args.arm]
    if any(name not in all_arms for name in names):
        raise SystemExit(f"unknown arm {args.arm!r}; known: both, {', '.join(all_arms)}")
    handoff_points = arm_config.handoff_points()

    import pyzed.sl as sl
    from control.IK_control import connect_robot, read_joints_rad, send_joints

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
                  f"field. Check `python ports/setup_usb_ports.py --check` and the arm's power.")
    if not robots:
        raise SystemExit("no arm connected")
    bases = {name: np.array([all_arms[name]["x"], all_arms[name]["y"]]) for name in robots}
    regions = Regions(bases, arm_config.middle_half_width())
    print("Arms: " + ", ".join(f"{name} at y {bases[name][1]:+.2f}" for name in bases)
          + f" | handoff points: {handoff_points} | {'parallel' if args.parallel else 'one task at a time'}")
    last_q = {name: list(HOME_Q) for name in robots}
    workers = []

    def disconnect_all():
        for robot in robots.values():
            if robot:
                robot.disconnect()

    def home_all():
        for name, robot in robots.items():
            if robot:
                go_home(robot, send_joints, read_joints_rad)
                last_q[name] = read_joints_rad(robot)

    def running():
        return any(w.is_alive() for w in workers)

    marks = []                     # {"kind": "shape"/"box", "colour", "xyz" (table frame), "pixel", "state"}
    state = {"pending": None, "status": "click a shape's centre or a box", "plan": {}}
    frame = cloud = None
    try:
        zed = open_zed(args.serial)
    except BaseException:
        disconnect_all()
        raise
    image, cloud_mat, runtime = sl.Mat(), sl.Mat(), sl.RuntimeParameters()
    camera_from_table = np.linalg.inv(T_table)
    try:
        from vision.detection import camera_intrinsics

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
                                     args.max_reach, regions)
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
        lines = [state["status"], "u undo | c clear | Space go | h home | q quit"]
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
        tasks, skipped = build_tasks(shapes, boxes, bases, handoff_points, args.max_reach, regions)
        for shape, why in skipped:
            print(f"{shape['colour']} shape at x {shape['xyz'][0]:.2f} y {shape['xyz'][1]:.2f}: skipped, {why}")
        if not tasks:
            state["status"] = "nothing the arms can do: see the terminal"
            return
        print("\nPlan:")
        for task in tasks:
            then = f", then {task.then_arm} to the {task.box['colour']} box" if task.kind == TO_HANDOFF else ""
            print(f"  {task.label()}{then}")
        home_all()                           # every arm starts at home (its pose is an obstacle for the other)
        coordinator = Coordinator(tasks, regions, handoff_points, args.parallel)
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
    sized, was_running = False, False
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
        disconnect_all()


if __name__ == "__main__":
    main()
