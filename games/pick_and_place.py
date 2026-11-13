"""Game 1, pick and place: the live view of every object and bin, Enter, and the arms sort them straight away.

    python run_game.py pick_and_place                          # both arms, working at the same time
    python run_game.py pick_and_place --arm follower_pink      # one arm
    python run_game.py pick_and_place --round shape            # sort by shape (config/games/pick_and_place.yaml)
    python run_game.py pick_and_place --one-at-a-time          # both arms, but one task at a time
    python run_game.py pick_and_place --no-robot               # no arms: detection and planning only
    python run_game.py pick_and_place --headless               # no window: sorts whatever appears, then waits
    python run_game.py pick_and_place --auto                   # the window, but rounds start by themselves
    (or: python -m games.pick_and_place ...)

Built from the shared parts: perception/ (what the camera sees), planning/ (IK and paths), state_machine/ (who does
what, and each arm's behaviour tree), hardware/ (the arms and the gripper), config/ (every setting).

The arms go to their home pose at once (config/arms.yaml home) and the camera window shows, live:
  every object and bin YOLO finds, with its colour and, under it, what will happen to it:
      -> pink                 follower_pink takes it to its bin
      -> pink > drop > blue   follower_pink puts it down in the drop-off area, follower_blue takes it to its bin
      no arm / no red bin     why it will be left
  each arm's reach on the table: its far limit (--max-reach, solid ring) and near limit (--min-reach, thin ring),
  the area both arms reach (light) and, inside it, the drop-off area (brighter): the middle strip between the arms
  where one arm leaves an object for the other. Any free point of it is used, not one fixed point.
Enter: the objects and bins in view now become the round's tasks; each is planned and done at once (no PyBullet
preview, no confirmation, no pauses), both arms at the same time where they do not get in each other's way
(state_machine/coordinator.py: each arm's side and the middle strip are reserved by one arm at a time; each arm takes its
closest object first, skipping ones whose region the other arm holds). Each attempt at an object is a behaviour tree
(state_machine/arm_worker.py task_tree): look where the object and its bin are now (from home, out of the view),
plan, grasp (missed: back off, home, try again from where it is seen next), carry, release, home, and check it is not
seen outside the bin (else try again from there); up to --attempts times. A handed-off object is looked for again by
the arm that takes it over, so one that tipped or slid is picked where it is. When the round is
done its time is printed; Enter again for what is left. h = arms home, q / Esc = quit (after the moves running now;
the arms park at home).

Which bin each object goes in comes from config/games/pick_and_place.yaml (games/rules.py): colour = the object's colour,
shape = cube -> red, cylinder -> blue, dodecahedron -> green, hexagonal prism -> yellow. Object centres come from the
perception code (perception/live.py, --centre); objects already inside a bin are left out. The gripper opens only as wide as
each object needs (hardware/gripper.py, --grip-margin).

Needs a trained YOLO segmentation model (config/perception.yaml yolo.weights, or --weights) whose classes include the
objects and "bin", and the camera calibration (python 02_calibrate.py camera ...).
"""
import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration.calibrate import arm_from_table, arms, default_arm  # noqa: E402
from games import rules as task_config  # noqa: E402
from hardware import gripper as jaw  # noqa: E402
from hardware.gripper import GRIP_CURRENT_MA, GRIPPER_OPEN  # noqa: E402
from perception.colour import classify, load_colors, object_pixels, to_hsv  # noqa: E402
from perception.detection import OBJECT_SIZES, OBJECT_STL_FILES  # noqa: E402
from perception.live import Vision, add_vision_args  # noqa: E402
from planning.pick_planner import HOME_Q  # noqa: E402
from state_machine.arm_worker import go_home, short  # noqa: E402

BIN_CLASS = "bin"
WINDOW = "pick and place (Enter = go, h = home, q = quit)"
ARM_BGR = {"follower_pink": (180, 105, 255), "follower_blue": (255, 140, 0)}
BIN_BGR = {"red": (40, 40, 230), "green": (40, 180, 40), "blue": (230, 90, 30), "yellow": (30, 220, 240)}


# ---------------------------------------------------------------------------------------------- perception
def colour_of(frame, polygon, colors):
    """(colour name, confidence) of the pixels inside a detection's outline (perception/colour.py)."""
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


def object_stl(name, size=""):
    """The STL of an object of that size ("small" / "large", as vision/live.py measured it); the large one when the
    size is unknown (--centre rect / fit do not measure it)."""
    return OBJECT_SIZES.get(name, {}).get(size) or OBJECT_STL_FILES[name]


def same_size(seen, shape):
    """Whether a detection can be that object by size: the same size, or either size unknown."""
    return not seen.get("size") or not shape.get("size") or seen["size"] == shape["size"]


def detect(vision, colors, table_from_default):
    """One live frame -> (canvas, objects, bins). Each is a dict: name, colour, size ("small" / "large", or "" when the
    vision did not measure it), xyz (TABLE frame; an object's top centre, at its size's height), yaw (its angle seen
    from above, TABLE frame, degrees 0-180, or None), pixel. The centres
    are the median over the last frames (perception/live.py --smooth-frames)."""
    frame, cloud = vision.grab()
    canvas, detections = vision.detect(frame, cloud)
    objects, bins = [], []
    sizes = getattr(vision, "last_sizes", None) or [""] * len(detections)
    yaws = getattr(vision, "last_yaws", None) or {}
    turn = np.degrees(np.arctan2(table_from_default[1, 0], table_from_default[0, 0]))   # vision frame -> TABLE
    for index, ((name, conf, centre, _), polygon, size) in enumerate(zip(detections, vision.last_polygons, sizes)):
        if centre is None:
            continue
        colour, _ = colour_of(frame, polygon, colors)
        xyz = (table_from_default @ np.append(np.asarray(centre, float), 1.0))[:3]
        item = {"name": name.casefold(), "conf": conf, "colour": colour, "xyz": xyz,
                "size": size if size in ("small", "large") else "",   # rect mode puts its rectangle here
                "yaw": None if yaws.get(index) is None else float((yaws[index] + turn) % 180.0),   # deg, TABLE
                "pixel": np.asarray(polygon).reshape(-1, 2).mean(axis=0)}
        if item["name"] == BIN_CLASS:
            bins.append(item)
        elif item["name"] in OBJECT_STL_FILES:
            objects.append(item)
    return canvas, objects, bins


def bin_for(obj, bins, round_name):
    """The bin this object belongs in under config/games/pick_and_place.yaml's rules (None when that bin is not in view)."""
    wanted = task_config.bin_colour(obj["name"], obj["colour"], round_name)
    matches = [b for b in bins if b["colour"] == wanted]
    return min(matches, key=lambda b: np.linalg.norm(b["xyz"][:2] - obj["xyz"][:2])) if matches else None


# ---------------------------------------------------------------------------------------------- the view
class TableView:
    """Draws TABLE-frame things on the camera image: each arm's far and near reach rings, the area both arms reach
    and the drop-off area inside it (worked out once: the camera does not move)."""

    def __init__(self, cam, bases, regions, max_reach, min_reach, image_shape, step=0.01):
        self.cam = cam
        self.default_from_table = arm_from_table(default_arm())
        self.rings = []
        angles = np.linspace(0, 2 * np.pi, 181)
        for name, b in bases.items():
            for radius, thickness in ((max_reach, 2), (min_reach, 1)):
                ring = np.column_stack([b[0] + radius * np.cos(angles), b[1] + radius * np.sin(angles),
                                        np.zeros_like(angles)])
                self.rings.append((self.pixels(ring), ARM_BGR.get(name, (200, 200, 200)), thickness))
        gx, gy = np.meshgrid(np.arange(-0.10, 0.80, step), np.arange(-0.70, 0.70, step))
        cells = np.column_stack([gx.ravel(), gy.ravel()])
        within = np.stack([(np.linalg.norm(cells - b, axis=1) >= min_reach) &
                           (np.linalg.norm(cells - b, axis=1) <= max_reach) for b in bases.values()])
        both = within.all(axis=0) if len(bases) > 1 else np.zeros(len(cells), bool)
        drop = both & np.array([regions.of(c) == "middle" for c in cells])
        h, w = image_shape[:2]
        self.shared = self.area(cells[both], step, (h, w))
        self.drop = self.area(cells[drop], step, (h, w))
        self.drop_label = self.pixels(np.append(cells[drop].mean(axis=0), 0.0)[None])[0] if drop.any() else None

    def pixels(self, table_xyz):
        """TABLE-frame points (n x 3) -> image pixels (n x 2, int)."""
        p = np.column_stack([table_xyz, np.ones(len(table_xyz))]) @ self.default_from_table.T
        return np.rint(self.cam.project(p[:, :3])).astype(np.int32)

    def area(self, cells, step, shape):
        mask = np.zeros(shape, np.uint8)
        if len(cells):
            half = step / 2
            corners = np.concatenate([np.column_stack([cells + [dx, dy], np.zeros(len(cells))])
                                      for dx, dy in ((-half, -half), (half, -half), (half, half), (-half, half))])
            px = np.ascontiguousarray(self.pixels(corners).reshape(4, -1, 2).transpose(1, 0, 2))
            for quad in px:
                cv2.fillConvexPoly(mask, quad, 1)
        return mask.astype(bool)

    def draw(self, canvas):
        tint = canvas.copy()
        tint[self.shared] = (235, 235, 235)
        tint[self.drop] = (255, 255, 0)
        canvas = cv2.addWeighted(tint, 0.25, canvas, 0.75, 0)
        for px, colour, thickness in self.rings:
            cv2.polylines(canvas, [px], True, colour, thickness)
        if self.drop_label is not None:
            put(canvas, "drop-off area", tuple(self.drop_label - [50, 0]), (255, 255, 0), 0.6)
        return canvas


def put(canvas, text, org, colour, scale=0.55):
    org = (int(org[0]), int(org[1]))
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2)


def add_game_args(ap):
    """The game's options (rules, arms, reach, grasp, bins, planner tuning); shared with the simulator
    (simulation/run.py), so a simulated round plans with exactly the same settings."""
    ap.add_argument("--round", choices=task_config.modes(), default=task_config.default_mode(),
                    help="which bin each object goes in: a mode of config/games/pick_and_place.yaml (default: its 'mode')")
    ap.add_argument("--arm", default="both", help="follower_pink, follower_blue, or both (default)")
    ap.add_argument("--one-at-a-time", action="store_true",
                    help="with both arms: one task at a time (default: both arms work at once where it is safe)")
    ap.add_argument("--no-robot", action="store_true", help="no arms connected: detection and planning only")
    ap.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)     # old flag: Enter now moves the arms
    ap.add_argument("--max-reach", type=float, default=0.45, help="far limit of an arm, from its base [m]")
    ap.add_argument("--min-reach", type=float, default=0.12, help="near limit of an arm, from its base [m]")
    ap.add_argument("--drop-step", type=float, default=0.04, help="spacing of the drop-off area's points [m]")
    ap.add_argument("--attempts", type=int, default=3,
                    help="attempts per object (a missed grasp or a drop outside the bin uses one)")
    ap.add_argument("--redetect-wait", type=float, default=0.6,
                    help="after a miss, use detections from this long after the arm is home [s] (the centres are a "
                         "median over --smooth-frames frames, ~0.5 s at 15 fps)")
    ap.add_argument("--redetect-radius", type=float, default=0.12,
                    help="before each pick, the object is looked for this far from where it was [m]")
    ap.add_argument("--approach", type=float, default=0.06, help="height above the object's top to come down from [m]")
    ap.add_argument("--grasp-at", choices=["middle", "top"], default="middle",
                    help="grasp at half the object's height (default), or --grasp-depth below its top")
    ap.add_argument("--level-below-top", type=float, default=0.01,
                    help="a level (horizontal) grasp goes at least this far below the object's top [m]; higher, the "
                         "jaws close over it, so a shorter object is grasped top-down")
    ap.add_argument("--grasp-depth", type=float, default=0.015, help="--grasp-at top: this far below the top [m]")
    ap.add_argument("--min-grasp-z", type=float, default=0.02,
                    help="lowest grasp point for a level grasp [m] (the table clearance check still applies)")
    ap.add_argument("--lift", type=float, default=0.10, help="lift height above the object's top [m]")
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
    ap.add_argument("--box-height", type=float, default=0.08, help="height of the bins [m]")
    ap.add_argument("--box-size", type=float, default=0.15, help="width of the bins, for the planner [m]")
    ap.add_argument("--box-margin", type=float, default=0.03, help="the arm keeps this much above a bin's rim [m]")
    ap.add_argument("--drop-above", type=float, default=0.08,
                    help="the jaws at least this high over a bin's rim at release [m]")
    ap.add_argument("--drop-clearance", type=float, default=0.01,
                    help="the held object's bottom this far above the bin as the planner sees it (rim + --box-margin) "
                         "at release [m]: an object grasped high hangs lower, so it is carried higher")
    ap.add_argument("--no-hold-check", dest="hold_check", action="store_false",
                    help="do not check the held object against the bins and the other arm while it is carried")
    ap.add_argument("--drop-inset", type=float, default=0.04,
                    help="the object is released this far inside the bin's rim nearest the arm [m]")
    ap.add_argument("--bin-conf", type=float, default=0.6,
                    help="bins detected with less confidence than this are left out (e.g. a half-visible bin at the "
                         "edge of the view)")
    ap.add_argument("--miss-margin", type=float, default=-1.0,
                    help="-1 (default): the gripper closes until it holds something or is fully closed. >= 0: a grasp "
                         "is also a miss once the gripper closes this many units (0-100) past where the robot model says "
                         "it squeezes the object (faster misses, but the real gripper's scale may not match the model)")
    ap.add_argument("--grip-stall-readings", type=int, default=5,
                    help="holding = the gripper stops moving while closing (this many readings, ~40 ms each, with "
                         "less than 0.5 units of movement) above --grip-empty; 0 = only the current tells")
    ap.add_argument("--grip-empty", type=float, default=8.0,
                    help="a gripper that stops at or below this (0-100) has closed on nothing: a miss")
    ap.add_argument("--neighbour-radius", type=float, default=-1.0,
                    help="the other objects as obstacles for the planner. < 0 (default): each object's own size (its "
                         "widest half width, any orientation) + --neighbour-margin; 0: ignored, the gripper may push a "
                         "neighbour (the old default); > 0: a column this many m either side of every object")
    ap.add_argument("--neighbour-margin", type=float, default=0.005,
                    help="room kept around each other object, on top of its size [m]")
    ap.add_argument("--down-roll", choices=["auto", "across"], default="auto",
                    help="top-down grasps: auto (default) turns the wrist so the fingers come down in free space, "
                         "preferring jaws that close along the reach; across: the jaws always close across the reach "
                         "(the old behaviour)")
    ap.add_argument("--down-rolls", type=int, default=3,
                    help="how many wrist rolls a top-down grasp tries, best first")
    ap.add_argument("--finger-clearance", type=float, default=0.006,
                    help="room a finger needs from a neighbour or a bin wall for a wrist roll to count as free [m]")
    ap.add_argument("--clearance", type=float, default=0.005,
                    help="the gripper keeps this far above the table in the plans [m] (was 0.01; a level grasp point "
                         "then goes down to 4.0 cm instead of 4.5 cm; raise it if the fingers touch the table)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--planner", choices=["process", "thread"], default="process",
                    help="process (default): each arm plans in a process of its own, both at the same time and the "
                         "next object ahead while the current one is carried; thread: one plan at a time, in this "
                         "process (the old way)")
    ap.add_argument("--no-prefetch", dest="prefetch", action="store_false",
                    help="do not plan the next object ahead of time (with --planner process)")
    ap.add_argument("--prefetch-tolerance", type=float, default=3.0,
                    help="a plan made ahead is used only when the arm starts within this many degrees (every "
                         "joint) of the home pose it assumed; else the object is planned again")
    ap.add_argument("--grip-current-ma", type=float, default=GRIP_CURRENT_MA)
    ap.add_argument("--grip-margin", type=float, default=0.02,
                    help="the jaws open to the object's width (its STL, widest direction) plus this much [m]")
    ap.add_argument("--grip-open", type=float, default=None,
                    help=f"open the jaws to this fixed command (0-100) instead (the old: {GRIPPER_OPEN:.0f})")
    ap.add_argument("--width", type=int, default=1280, help="window width in pixels")
    ap.add_argument("--auto", action="store_true",
                    help="start a round by itself whenever objects with their bin are in view and the arms are home "
                         "(instead of Enter); the arms go home after each round and wait for more")
    ap.add_argument("--headless", action="store_true",
                    help="no window (implies --auto): watch the camera, sort whatever appears, wait; Ctrl+C stops "
                         "(after the moves running now; the arms park)")
    ap.add_argument("--auto-settle", type=float, default=1.0,
                    help="--auto: the same objects must be seen this long before a round starts [s] (not a hand "
                         "passing through, not a detection that flickers)")
    ap.add_argument("--auto-retries", type=int, default=1,
                    help="--auto: an object that failed is tried again in at most this many later rounds while it "
                         "stays where it is (it is tried afresh once it is moved)")


# ---------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_vision_args(ap)
    add_game_args(ap)
    from hardware.arm import add_motion_args, apply_motion_args

    add_motion_args(ap)
    args = ap.parse_args(argv)
    args.auto = args.auto or args.headless
    apply_motion_args(args)
    args.gui = False                       # plan in the background: no PyBullet window, no playback

    from core import arms as arm_config
    from planning.pick_planner import handoff_checker
    from state_machine import arm_worker as workers_module    # each arm's behaviour tree (planning, moving, handoffs)
    from state_machine.coordinator import BOX_SLACK, DIRECT, Coordinator, Regions, build_tasks, handoff_area

    BOX_SLACK["value"] = max(0.0, args.box_size / 2 - args.drop_inset)
    all_arms = arms()
    names = list(all_arms) if args.arm == "both" else [args.arm]
    if any(n not in all_arms for n in names):
        raise SystemExit(f"unknown arm {args.arm!r}; known: {', '.join(all_arms)} or both")
    table_from_default = np.linalg.inv(arm_from_table(default_arm()))   # the vision code works in this frame
    colors = load_colors()
    print("Sorting " + task_config.describe(args.round))

    from hardware.arm import connect_robot, on_every_arm, park_and_disconnect, read_joints_rad, send_joints

    robots, workers, vision, planners = {}, [], None, {}
    run = {"coordinator": None, "start": 0.0, "reported": True, "status": "Enter: sort what is in view"}
    close_checker = lambda: None
    try:
        for name in names:
            if args.no_robot:
                robots[name] = None
                continue
            try:
                robots[name] = connect_robot(name)
            except (ConnectionError, OSError, RuntimeError) as error:
                if len(names) == 1:
                    raise
                print(f"{name}: could not connect ({str(error).strip().splitlines()[0]}); continuing without it")
        if not robots:
            raise SystemExit("no arm connected")
        bases = {n: np.array([all_arms[n]["x"], all_arms[n]["y"]]) for n in robots}
        regions = Regions(bases, arm_config.middle_half_width())
        last_q = {n: list(HOME_Q) for n in robots}

        def home_all():
            def home(name, robot):
                go_home(robot, send_joints, read_joints_rad)
                last_q[name] = read_joints_rad(robot)
            on_every_arm(robots, home)                 # both arms at the same time

        print("Arms going home ...")
        from planning.planner_service import ArmPlanner

        planners = {n: ArmPlanner(n, args, args.planner, HOME_Q) for n in robots}   # build each arm's world now
        home_all()
        vision = Vision(args)
        area = handoff_area(bases, args.max_reach, args.min_reach, regions, step=args.drop_step)
        fixed = [list(p) for p in arm_config.handoff_points()] if len(bases) > 1 else []
        handoff_points = []                   # this round's free drop-off points (filled on Enter)
        can_grasp, close_checker = handoff_checker(handoff_points, args)
        marks, view = [], None
        seen = {"objects": [], "bins": [], "time": 0.0, "lock": threading.Lock()}   # the live view's latest

        class Camera:
            """What the live view sees now, for the arms' behaviour trees (state_machine/arm_worker.py). Every
            answer comes from detections made --redetect-wait s after `since` (when the asking arm got home, out of
            the view): the centres are a median over the last frames, so older frames must have left it."""

            @staticmethod
            def _fresh(since):
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline:
                    with seen["lock"]:
                        if seen["time"] >= since + args.redetect_wait:
                            return list(seen["objects"]), list(seen["bins"])
                    time.sleep(0.05)
                return None, None

            def find_object(self, shape, since):
                """The same object (shape and colour) nearest to where it was, within --redetect-radius:
                (xyz, pixel), or None."""
                objects, _ = self._fresh(since)
                old = np.asarray(shape["xyz"][:2], float)
                same = [o for o in objects or [] if o["name"] == shape.get("name")
                        and o["colour"] == shape.get("object_colour") and same_size(o, shape)]
                if not same:
                    return None
                best = min(same, key=lambda o: np.linalg.norm(o["xyz"][:2] - old))
                if np.linalg.norm(best["xyz"][:2] - old) > args.redetect_radius:
                    return None
                return best["xyz"], best["pixel"], best.get("yaw")

            def find_bin(self, box, since):
                """The bin of that colour nearest to where it was (within 15 cm): xyz, or None."""
                _, bins_now = self._fresh(since)
                same = [b for b in bins_now or [] if b["colour"] == box["colour"]]
                if not same:
                    return None
                best = min(same, key=lambda b: np.linalg.norm(b["xyz"][:2] - np.asarray(box["xyz"][:2], float)))
                return best["xyz"] if np.linalg.norm(best["xyz"][:2] - box["xyz"][:2]) <= 0.15 else None

            def find_stray(self, shape, since, near, marks):
                """After a drop: the same object seen OUTSIDE a bin (objects in a bin are left out of the
                detections) within 15 cm of one of `near` (the bin, where it was picked), and not another object
                still to do: (xyz, pixel), or None (it is in the bin)."""
                objects, _ = self._fresh(since)
                others = [m["xyz"][:2] for m in marks if m["kind"] == "shape" and m["state"] == "todo"
                          and m is not shape and m is not shape.get("origin")
                          and m.get("name") == shape.get("name")]
                for o in objects or []:
                    if (o["name"] != shape.get("name") or o["colour"] != shape.get("object_colour")
                            or not same_size(o, shape)):
                        continue
                    if any(np.linalg.norm(o["xyz"][:2] - np.asarray(t, float)[:2]) < 0.03 for t in others):
                        continue
                    if any(np.linalg.norm(o["xyz"][:2] - np.asarray(p, float)[:2]) <= 0.15 for p in near):
                        return o["xyz"], o["pixel"]
                return None

        camera = Camera()
        print(f"Arms: {', '.join(bases)} | drop-off area: {len(area)} points"
              + ("" if len(bases) > 1 else " (one arm: no drop-off)")
              + ("" if len(bases) < 2 else
                 f" | {'one task at a time' if args.one_at_a_time else 'both arms at once where safe'}"))

        def free_points(objects, bins):
            """The drop-off points not taken by an object or a bin (7 cm clear of them)."""
            taken = [o["xyz"][:2] for o in objects] + [b["xyz"][:2] for b in bins]
            return [p for p in fixed + area if all(np.linalg.norm(np.asarray(p) - t) >= 0.07 for t in taken)]

        def to_marks(objects, bins):
            """The detections as the coordinator's marks: shapes (colour = the colour of the bin they go in) and
            boxes; and {id(object): why it is left} for the ones without a rule or a bin in view."""
            boxes = [{"kind": "box", "colour": b["colour"], "xyz": b["xyz"], "pixel": b["pixel"], "state": "todo",
                      "conf": b["conf"]}
                     for b in bins if b["colour"] and b["conf"] >= args.bin_conf]     # weak detections left out
            shapes, notes = [], {}
            for o in objects:
                wanted = task_config.bin_colour(o["name"], o["colour"], args.round)
                if wanted is None:
                    notes[id(o)] = f"no rule ({o['colour'] or 'colour?'})"
                    continue
                if not any(b["colour"] == wanted for b in boxes):
                    notes[id(o)] = f"no {wanted} bin"
                    continue
                opening = (args.grip_open if args.grip_open is not None
                           else jaw.open_command(jaw.object_width(object_stl(o["name"], o["size"])),
                                                 args.grip_margin))   # only as wide as an object of ITS size needs
                label = " ".join(part for part in (o["colour"] or "?", o["size"], o["name"]) if part)
                shapes.append({"kind": "shape", "colour": wanted, "label": label,
                               "name": o["name"], "object_colour": o["colour"], "size": o["size"],
                               "xyz": o["xyz"], "pixel": o["pixel"], "state": "todo", "open_cmd": opening,
                               "yaw": o.get("yaw"), "source": id(o)})
            return shapes, boxes, notes

        def plan_labels(objects, bins):
            """{id(object): what will happen to it} (distances only: quick enough for every frame)."""
            shapes, boxes, labels = to_marks(objects, bins)
            tasks, skipped = build_tasks(shapes, boxes, bases, free_points(objects, bins), args.max_reach, regions,
                                         None, max_usable=1)
            for t in tasks:
                labels[t.shape["source"]] = (f"-> {short(t.arm)}" if t.kind == DIRECT
                                             else f"-> {short(t.arm)} > drop > {short(t.then_arm)}")
            for shape, _ in skipped:
                labels[shape["source"]] = "no arm reaches"
            return labels

        def go(objects, bins, skip=()):
            shapes, boxes, notes = to_marks(objects, bins)
            shapes = [sh for sh in shapes if sh["source"] not in skip]     # --auto: failed too often where it is
            for o in objects:
                if id(o) in notes:
                    print(f"{o['colour'] or '?'} {o['name']}: left, {notes[id(o)]}")
            if not shapes:
                run["status"] = "nothing to sort: no object with its bin in view"
                return
            handoff_points[:] = free_points(objects, bins)
            checked = time.monotonic()
            tasks, skipped = build_tasks(shapes, boxes, bases, handoff_points, args.max_reach, regions, can_grasp,
                                         max_usable=3)
            for shape, why in skipped:
                print(f"{shape['label']} at x {shape['xyz'][0]:.2f} y {shape['xyz'][1]:.2f}: left, {why}")
            run["skipped"] = [shape for shape, _ in skipped]          # --auto: not tried again until moved
            if not tasks:
                run["status"] = "nothing the arms can do (see the terminal)"
                return
            print(f"\nPlan ({time.monotonic() - checked:.1f} s to check the drop-off points):")
            for task in tasks:
                then = f", then {task.then_arm} to the {task.box['colour']} bin" if task.then_arm else ""
                print(f"  {task.label()}{then}")
            marks[:] = shapes + boxes
            coordinator = Coordinator(tasks, regions, handoff_points, not args.one_at_a_time,
                                      arm_config.handoff_min_spacing(), max_reach=args.max_reach)
            workers_module.CURRENT["coordinator"] = coordinator
            run.update(coordinator=coordinator, start=time.monotonic(), reported=False, status="working ...")
            workers.clear()
            for name, robot in robots.items():
                worker = threading.Thread(target=workers_module.arm_worker, daemon=True, name=name,
                                          args=(name, robot, coordinator, marks, last_q, args,
                                                (send_joints, read_joints_rad), handoff_points, camera, None,
                                                planners[name]))
                workers.append(worker)
                worker.start()

        def draw(canvas, objects, labels, busy):
            canvas = view.draw(canvas)
            if not busy:
                for o in objects:
                    text = labels.get(id(o))
                    if text:
                        wanted = task_config.bin_colour(o["name"], o["colour"], args.round)
                        put(canvas, text, o["pixel"] + [-30, 34], BIN_BGR.get(wanted, (255, 255, 255)))
            for m in marks:                    # the round's objects and what happened to them
                if m["kind"] != "shape" or not run["coordinator"]:
                    continue
                u, v = (int(c) for c in m["pixel"])
                if m["state"] == "placed":
                    cv2.circle(canvas, (u, v), 14, (0, 220, 0), 3)
                elif m["state"] == "failed":
                    cv2.drawMarker(canvas, (u, v), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 28, 3)
                elif m["state"] == "handoff" and m.get("handoff_xyz") is not None:
                    target = tuple(int(c) for c in view.pixels(np.asarray(m["handoff_xyz"], float)[None])[0])
                    cv2.arrowedLine(canvas, (u, v), target, (255, 255, 0), 2, tipLength=0.1)
                    cv2.circle(canvas, target, 12, (255, 255, 0), 2)
            lines = [f"{args.round} round | " + ("auto | " if args.auto else "") + "Enter go | h home | q quit",
                     run["status"]]
            if run["coordinator"] is not None and busy:
                lines += [f"{short(a)}: {s}" for a, s in sorted(run["coordinator"].arm_state.items())]
            for i, text in enumerate(lines):
                put(canvas, text, (15, 32 + 28 * i), (0, 255, 0), 0.65)
            return canvas

        failed_here = []                       # --auto: [{name, colour, xy, count}] objects that failed, where they were
        settle = {"key": None, "since": 0.0, "said": ""}

        def given_up(o):
            """Whether this detection is an object that already failed --auto-retries + 1 times where it still is."""
            return any(f["name"] == o["name"] and f["colour"] == o["colour"] and f["count"] > args.auto_retries
                       and np.linalg.norm(np.asarray(o["xyz"][:2]) - f["xy"]) < 0.03 for f in failed_here)

        def remember_failures():
            for m in marks:
                if m["kind"] != "shape" or m["state"] != "failed":
                    continue
                xy = np.asarray(m["xyz"][:2], float)
                same = next((f for f in failed_here if f["name"] == m["name"] and f["colour"] == m["object_colour"]
                             and np.linalg.norm(xy - f["xy"]) < 0.03), None)
                if same is None:
                    failed_here.append({"name": m["name"], "colour": m["object_colour"], "xy": xy, "count": 1})
                else:
                    same["count"] += 1

        def say(text):
            """--auto status: printed when it changes (the terminal is the only view with --headless)."""
            run["status"] = text
            if text != settle["said"]:
                settle["said"] = text
                print(f"[auto] {text}", flush=True)

        def auto_step(objects, bins):
            """Start a round when the same sortable objects have been in view for --auto-settle seconds."""
            shapes, _, _ = to_marks(objects, bins)
            sortable = {sh["source"] for sh in shapes}
            todo = [o for o in objects if id(o) in sortable and not given_up(o)]
            if not todo:
                settle["key"] = None
                left = sum(given_up(o) for o in objects)
                say("waiting for objects" + (f" ({left} left where they failed: move them to try again)" if left
                                             else ""))
                return
            key = sorted((o["name"], o["colour"] or "", round(o["xyz"][0] / 0.02), round(o["xyz"][1] / 0.02))
                         for o in todo)
            now = time.monotonic()
            if key != settle["key"]:
                settle["key"], settle["since"] = key, now
            if now - settle["since"] < args.auto_settle:
                say(f"{len(todo)} object(s) in view, starting when they stay put")
                return
            settle["key"] = None
            say(f"sorting {len(todo)} object(s)")
            run["skipped"] = []
            go(objects, bins, skip={id(o) for o in objects if given_up(o)})
            for shape in run.get("skipped", []):        # no arm reaches it: leave it until it is moved
                failed_here.append({"name": shape["name"], "colour": shape["object_colour"],
                                    "xy": np.asarray(shape["xyz"][:2], float), "count": args.auto_retries + 1})
            if run["reported"]:                       # nothing could be done: wait for the scene to change
                settle["key"], settle["since"] = key, now + 5.0

        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        else:
            print("Headless: sorting whatever appears in view; Ctrl+C stops (the arms park).", flush=True)
        sized = False
        while True:
            canvas, objects, bins = detect(vision, colors, table_from_default)
            with seen["lock"]:
                seen["objects"], seen["bins"], seen["time"] = objects, bins, time.monotonic()
            if view is None and not args.headless:
                view = TableView(vision.cam, bases, regions, args.max_reach, args.min_reach, canvas.shape)
            busy = any(w.is_alive() for w in workers)
            if not busy and not run["reported"]:
                took = time.monotonic() - run["start"]
                placed = sum(m["state"] == "placed" for m in marks if m["kind"] == "shape")
                failed = sum(m["state"] == "failed" for m in marks if m["kind"] == "shape")
                run.update(reported=True, status=f"done in {took:.1f} s: {placed} placed, {failed} failed; "
                                                 + ("waiting for more" if args.auto else "Enter for the next round"))
                print(f"\nRound done in {took:.1f} s: {placed} placed, {failed} failed"
                      + (f" ({took / placed:.1f} s per object)" if placed else ""))
                if args.auto:
                    remember_failures()
                    settle["key"], settle["said"] = None, ""
            if args.headless:
                if not busy:
                    auto_step(objects, bins)
                continue
            labels = {} if busy else plan_labels(objects, bins)
            out = draw(canvas, objects, labels, busy)
            if args.width < out.shape[1]:
                out = cv2.resize(out, (args.width, round(out.shape[0] * args.width / out.shape[1])))
            cv2.imshow(WINDOW, out)
            if not sized:
                cv2.resizeWindow(WINDOW, out.shape[1], out.shape[0])
                sized = True
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                if busy:
                    print("Stopping after the moves running now ...")
                break
            if busy:
                continue
            if args.auto:
                auto_step(objects, bins)
            if key in (13, 10, 32):              # Enter or Space
                go(objects, bins)
            elif key == ord("h"):
                home_all()
                run["status"] = "home"
    except KeyboardInterrupt:
        print("\nStopping after the moves running now ...")
    finally:
        if run["coordinator"] is not None:
            run["coordinator"].stop.set()
        for worker in workers:
            worker.join()
        # whatever goes wrong closing the rest, the arms still park
        for step in (lambda: vision is not None and vision.close(),
                     lambda: None if args.headless else cv2.destroyAllWindows(),
                     close_checker, *[planner.close for planner in planners.values()]):
            try:
                step()
            except Exception as error:                      # noqa: BLE001
                print(f"(while stopping: {error})")
        on_every_arm(robots, lambda name, robot: park_and_disconnect(robot))    # both arms at the same time


if __name__ == "__main__":
    main()
