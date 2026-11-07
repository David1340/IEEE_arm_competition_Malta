"""Game 1 pick-and-place: sort every object on the field into its bin, with both arms.

Each cycle: the arms go home (out of the camera's view), the camera (YOLO + ZED) finds the objects and bins and
their colours, the next object is chosen with its bin, the arm that reaches both best is chosen, and the whole
move is planned (sim_plan's World: table and the other objects as obstacles) and shown in PyBullet:
    above the object -> down to it -> close the gripper (motor current) -> lift -> over the bin -> open
Positions come from the shared TABLE frame of the ArUco calibration (calibration/calibrate.py).

    python experiments/pick_and_place.py --round colour              # plan + preview each pick, arms do not move
    python experiments/pick_and_place.py --round colour --execute    # move the arms ('yes' before each object)
    python experiments/pick_and_place.py --round shape --execute --auto   # no confirmations (watch the arms!)
    python experiments/pick_and_place.py --arm follower_pink         # only one arm
    python experiments/pick_and_place.py --no-robot                  # no arms connected: vision + planning only

Rounds (Robot Championship 2026, Game 1): colour = object colour -> same-colour bin; shape = cube -> red,
cylinder -> blue, dodecahedron -> green, hexagonal prism -> yellow. Detecting again every cycle also follows
objects and bins that are moved (round 3: use --round shape).

Needs a trained YOLO segmentation model (runs/segment/*/weights/best.pt, or --weights) whose classes include the
objects and "bin".
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "yolo_dataset"))

from block_color import classify, load_colors, object_pixels, to_hsv  # noqa: E402
from calibration.calibrate import arm_from_table, arms, default_arm  # noqa: E402
from experiments.Full_pipeline import (GRIP_CONFIRM_READINGS, GRIP_CURRENT_MA, GRIP_SQUEEZE_UNITS,  # noqa: E402
                                       GRIPPER_OPEN, HOME_Q,
                                       close_gripper_with_current_feedback)
from experiments.sim_plan import TABLE_CLEARANCE, World  # noqa: E402
from vision.detection import OBJECT_STL_FILES  # noqa: E402
from vision.live import Vision, add_vision_args  # noqa: E402

SHAPE_TO_BIN = {"cube": "red", "cylinder": "blue", "dodecahedron": "green",
                "hexagonal": "yellow", "hexagonal_prism": "yellow"}
BIN_CLASS = "bin"


# ---------------------------------------------------------------------------------------------- perception
def colour_of(frame, polygon, colors):
    """(colour name, confidence) of the pixels inside a detection's outline (yolo_dataset/block_color.py)."""
    poly = np.asarray(polygon, np.int32).reshape(-1, 2)
    h, w = frame.shape[:2]
    x1, y1 = np.clip(poly.min(axis=0), 0, [w - 1, h - 1])
    x2, y2 = np.clip(poly.max(axis=0) + 1, 1, [w, h])
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None, 0.0
    hue, sat = object_pixels(to_hsv(frame), poly, (x1, y1, x2, y2), 0.35, 0.15, 1.01)
    if len(hue) < 20:
        return None, 0.0
    name, confidence, _ = classify(hue, sat, colors)
    return name, confidence


def look(vision, colors, table_from_default):
    """One camera frame -> (canvas, objects, bins). Each is a dict: name, colour, xyz (TABLE frame, top-face
    centre for objects, visible-surface centre for bins)."""
    frame, cloud = vision.grab()
    canvas, detections = vision.detect(frame, cloud)
    objects, bins = [], []
    for (name, conf, centre, _), polygon in zip(detections, vision.last_polygons):
        if centre is None:
            continue
        colour, colour_conf = colour_of(frame, polygon, colors)
        xyz = (table_from_default @ np.append(np.asarray(centre, float), 1.0))[:3]
        item = {"name": name.casefold(), "conf": conf, "colour": colour, "colour_conf": colour_conf, "xyz": xyz,
                "pixel": np.asarray(polygon).reshape(-1, 2).mean(axis=0)}
        if item["name"] == BIN_CLASS:
            bins.append(item)
        elif item["name"] in OBJECT_STL_FILES:
            objects.append(item)
    for item in objects + bins:
        label = f"{item['colour'] or '?'} {item['name']}"
        org = tuple(int(v) for v in item["pixel"])
        cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return canvas, objects, bins


def bin_for(obj, bins, round_name):
    """The bin this object belongs in (None when that bin is not in view)."""
    wanted = obj["colour"] if round_name == "colour" else SHAPE_TO_BIN.get(obj["name"])
    matches = [b for b in bins if b["colour"] == wanted]
    return min(matches, key=lambda b: np.linalg.norm(b["xyz"][:2] - obj["xyz"][:2])) if matches else None


def choose_arm(obj, target_bin, bases, max_reach):
    """The arm whose base is closest to both the object and its bin (the farther of the two counts), or None
    when neither reaches both."""
    def need(name):
        return max(np.linalg.norm(obj["xyz"][:2] - bases[name]), np.linalg.norm(target_bin["xyz"][:2] - bases[name]))
    name = min(bases, key=need)
    return name if need(name) <= max_reach else None


# ---------------------------------------------------------------------------------------------- motion
PRECISE = 0.012        # m: how close the grasp must be
ROUGH = 0.03           # m: the in-between points (above, lift, over the box) only need to be about right
SEED_TRIES = 3         # a failed point is tried again from other random starts


def plan_sequence(world, start_q, waypoints, seed, postures=None):
    """Plan start -> each waypoint in turn. waypoints: [(label, xyz in the arm frame)] or
    [(label, xyz, tolerance m)]; without a tolerance the grasp is PRECISE and every other point ROUGH.
    postures: {label: [gripper postures to try, in order]} ("horizontal", "down", "free"; sim_plan.World.find_pose),
    default horizontal. Returns [(label, path)] or raises RuntimeError naming the step that failed."""
    q = np.array(start_q, float)
    q[4] = world.start_q[4]          # wrist roll is fixed at its value when the world was built
    plans = []
    for waypoint in waypoints:
        label, xyz = waypoint[:2]
        if len(waypoint) > 2:
            tolerances = [waypoint[2]]
        else:   # aim precisely; only the in-between points fall back to ROUGH when that fails
            tolerances = [PRECISE] if label == "grasp" else [PRECISE, ROUGH]
        world.start_q = q.tolist()
        result = None
        print(f"    planning '{label}' ...", flush=True)
        for posture in (postures or {}).get(label, ["horizontal"]):
            for tolerance in tolerances:
                for attempt in range(SEED_TRIES):
                    result = world.find_pose(xyz, seed + attempt, tolerance=tolerance, posture=posture)
                    if result is not None and result.success:
                        break
                if result is not None and result.success:
                    if tolerance != tolerances[0]:
                        print(f"  '{label}': {1000 * result.position_error_m:.0f} mm off (allowed for this point)")
                    break
            if result is not None and result.success:
                if posture != "horizontal":
                    print(f"    '{label}': gripper {posture}")
                break
        if result is None or not result.success:
            err = "" if result is None else f" (best {1000 * result.position_error_m:.0f} mm off)"
            raise RuntimeError(f"no pose for '{label}' at {np.round(xyz, 3)}{err}")
        route = world.find_path(result.joint_positions, seed)
        if route is None:
            raise RuntimeError(f"no collision-free path to '{label}'")
        plans.append((label, route.path))
        q = np.asarray(route.path[-1], float)
    return plans


def follow(robot, path, gripper, send_joints=None):
    """Move along a joint path in one smooth motion (control/IK_control.py follow_path), the gripper going to
    `gripper` on the way. send_joints is not used any more (kept for older callers)."""
    from control.IK_control import follow_path

    return follow_path(robot, path, gripper)


def go_home(robot, send_joints, read_joints_rad):
    follow(robot, [read_joints_rad(robot), HOME_Q], GRIPPER_OPEN, send_joints)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_vision_args(ap)
    ap.add_argument("--round", choices=["colour", "shape"], default="colour", help="which bin each object goes in")
    ap.add_argument("--arm", default="both", help="follower_pink, follower_blue, or both (default)")
    ap.add_argument("--execute", action="store_true", help="move the real arms")
    ap.add_argument("--auto", action="store_true", help="with --execute: no 'yes' before each object")
    ap.add_argument("--no-robot", action="store_true", help="no arms connected: vision and planning only")
    ap.add_argument("--no-gui", action="store_true", help="no PyBullet window")
    ap.add_argument("--max-objects", type=int, default=100, help="stop after this many objects")
    ap.add_argument("--attempts", type=int, default=2, help="grasp attempts per object before skipping it")
    ap.add_argument("--approach", type=float, default=0.06, help="height above the object's top to come down from [m]")
    ap.add_argument("--grasp-depth", type=float, default=0.015,
                    help="the point between the jaws goes this far below the object's top [m]")
    ap.add_argument("--min-grasp-z", type=float, default=0.04,
                    help="lowest point between the jaws [m] (the jaws reach ~4 cm at the lowest in the model)")
    ap.add_argument("--lift", type=float, default=0.10, help="lift height above the object's top [m]")
    ap.add_argument("--drop-z", type=float, default=0.16, help="height of the jaws over a bin when releasing [m]")
    ap.add_argument("--max-reach", type=float, default=0.45, help="farthest point an arm is asked to reach [m]")
    ap.add_argument("--clearance", type=float, default=TABLE_CLEARANCE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grip-current-ma", type=float, default=GRIP_CURRENT_MA)
    from control.IK_control import add_motion_args, apply_motion_args

    add_motion_args(ap)
    args = ap.parse_args()
    apply_motion_args(args)
    if args.execute and args.no_robot:
        raise SystemExit("--execute needs the arms: drop --no-robot")

    all_arms = arms()
    names = list(all_arms) if args.arm == "both" else [args.arm]
    if any(n not in all_arms for n in names):
        raise SystemExit(f"unknown arm {args.arm!r}; known: {', '.join(all_arms)} or both")
    table_from_default = np.linalg.inv(arm_from_table(default_arm()))   # camera_transform.npy is in its frame
    colors = load_colors()

    from control.IK_control import arm_files, connect_robot, read_joints_rad, send_joints

    robots = {}
    try:
        for name in names:
            if args.no_robot:
                robots[name] = None
                continue
            try:
                robots[name] = connect_robot(name)
            except (ConnectionError, OSError) as error:
                if len(names) == 1:
                    raise
                print(f"{name}: could not connect ({str(error).strip().splitlines()[0]}); continuing without it")
        if not robots:
            raise SystemExit("no arm connected")
        bases = {n: np.array([all_arms[n]["x"], all_arms[n]["y"]]) for n in robots}
        if args.execute:
            print(f"Home pose [deg]: {np.round(np.degrees(HOME_Q), 1)}")
            if input("Type 'yes' to send the arms home and start: ").strip().lower() != "yes":
                raise SystemExit("Not executed.")
            for robot in robots.values():
                go_home(robot, send_joints, read_joints_rad)

        vision = Vision(args)
        done, skipped, misses = 0, set(), {}
        try:
            while done < args.max_objects:
                canvas, objects, bins = look(vision, colors, table_from_default)
                print(f"\n{len(objects)} object(s), bins: "
                      + (", ".join(f"{b['colour']}" for b in bins) or "none"))
                todo = []
                for obj in objects:
                    key = (obj["name"], obj["colour"], *np.round(obj["xyz"][:2], 2))
                    label = f"{obj['colour'] or '?'} {obj['name']} at x {obj['xyz'][0]:.2f} y {obj['xyz'][1]:.2f}"
                    if key in skipped:
                        continue
                    target_bin = bin_for(obj, bins, args.round)
                    if target_bin is None:
                        wanted = obj["colour"] if args.round == "colour" else SHAPE_TO_BIN.get(obj["name"])
                        print(f"  {label}: its {wanted} bin is not in view")
                        continue
                    if np.linalg.norm(obj["xyz"][:2] - target_bin["xyz"][:2]) < 0.06:
                        continue                                  # already in (or on) its bin
                    arm = choose_arm(obj, target_bin, bases, args.max_reach)
                    if arm is None:
                        print(f"  {label}: no arm reaches both it and its bin within {100 * args.max_reach:.0f} cm")
                        continue
                    todo.append((np.linalg.norm(obj["xyz"][:2] - bases[arm]), obj, target_bin, arm, key))
                cv2.imshow("pick and place (any key = continue, q = stop)", canvas)
                if cv2.waitKey(0 if not args.auto else 500) & 0xFF in (ord("q"), 27):
                    break
                if not todo:
                    print("Nothing left that can be sorted (no object with a visible bin within reach).")
                    break
                _, obj, target_bin, arm, key = min(todo, key=lambda item: item[0])   # nearest first
                to_arm = arm_from_table(arm)
                o = (to_arm @ np.append(obj["xyz"], 1.0))[:3]
                b = (to_arm @ np.append(target_bin["xyz"], 1.0))[:3]
                top = max(o[2], 0.0)
                grasp_z = max(top - args.grasp_depth, args.min_grasp_z)
                print(f"{arm}: {obj['colour']} {obj['name']} at x {o[0]:.3f} y {o[1]:.3f} (top {100 * top:.1f} cm) "
                      f"-> {target_bin['colour']} bin at x {b[0]:.3f} y {b[1]:.3f}")

                robot = robots[arm]
                start_q = HOME_Q if robot is None or not args.execute else read_joints_rad(robot)
                others = [(OBJECT_STL_FILES[x["name"]], *(to_arm @ np.append(x["xyz"], 1.0))[:2], 0.0)
                          for x in objects if x is not obj]
                world = World(others, start_q, gui=not args.no_gui, clearance=args.clearance,
                              arm=arm, calibration_path=arm_files(arm)[1])
                try:
                    # The point between the jaws reaches about 4 cm above the table at the lowest; if the grasp
                    # height has no pose, try 5 mm higher each time, up to 1.5 cm over the object's top (the jaw
                    # tips reach ~2 cm below that point, so they still close on its upper part).
                    plans, error = None, None
                    for z in np.arange(grasp_z, max(top + 0.015, grasp_z) + 1e-6, 0.005):
                        try:
                            plans = plan_sequence(world, start_q, [
                                ("above object", [o[0], o[1], top + args.approach]),
                                ("grasp", [o[0], o[1], z]),
                                ("lift", [o[0], o[1], top + args.lift]),
                                ("over bin", [b[0], b[1], args.drop_z]),
                            ], args.seed)
                            print(f"  grasp at {100 * z:.1f} cm above the table")
                            break
                        except RuntimeError as exc:
                            error = exc
                            if "grasp" not in str(exc):
                                break
                    if plans is None:
                        print(f"  skipped: {error}")
                        skipped.add(key)
                        continue
                    for label, path in plans:
                        world.start_q = list(path[0])      # play() starts from start_q
                        world.play(path, realtime=not args.no_gui)
                    if not args.execute:
                        print("  planned (add --execute to move the arm)")
                        skipped.add(key)
                        continue
                    if not args.auto and input("  Type 'yes' to do it: ").strip().lower() != "yes":
                        skipped.add(key)
                        continue
                    (_, to_above), (_, to_grasp), (_, to_lift), (_, to_bin) = plans
                    follow(robot, to_above, GRIPPER_OPEN, send_joints)
                    follow(robot, to_grasp, GRIPPER_OPEN, send_joints)
                    grasped, stop, hold, current_ma = close_gripper_with_current_feedback(
                        robot, 0.0, args.grip_current_ma, GRIP_CONFIRM_READINGS, GRIP_SQUEEZE_UNITS)
                    if not grasped:
                        misses[key] = misses.get(key, 0) + 1
                        again = misses[key] < args.attempts
                        print(f"  missed it (gripper closed to {stop:.0f}/100, {current_ma:.0f} mA); "
                              + ("trying again" if again else f"giving up after {args.attempts} attempts"))
                        if not again:
                            skipped.add(key)
                        follow(robot, to_lift, GRIPPER_OPEN, send_joints)
                        go_home(robot, send_joints, read_joints_rad)
                        continue
                    print(f"  holding it ({current_ma:.0f} mA)")
                    follow(robot, to_lift, hold, send_joints)
                    follow(robot, to_bin, hold, send_joints)
                    send_joints(robot, to_bin[-1], gripper=GRIPPER_OPEN)
                    go_home(robot, send_joints, read_joints_rad)
                    done += 1
                    print(f"  placed ({done} so far)")
                    # Seen again in the same place next time means it did not leave: counts as an attempt.
                    misses[key] = misses.get(key, 0) + 1
                    if misses[key] >= args.attempts:
                        skipped.add(key)
                finally:
                    world.close()
        finally:
            vision.close()
            cv2.destroyAllWindows()
    finally:
        for robot in robots.values():
            if robot is not None:
                robot.disconnect()


if __name__ == "__main__":
    main()
