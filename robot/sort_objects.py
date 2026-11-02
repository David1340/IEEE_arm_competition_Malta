"""Game 1: detect the objects and the bins, and drop every object into its bin by the rules in sort_rules.json.

Each cycle, with the arm folded out of the way:
    1. detect      ZED + YOLO: objects and bins, each with its colour (yolo_dataset/block_color.py, HSV)
    2. assign      every object gets a bin colour from sort_rules.json (round "colour": its own colour; round
                   "shape": cube -> red, cylinder -> blue, dodecahedron -> green, hexagonal -> yellow)
    3. pick one    the most confident object whose bin is there and which the arm can reach (and its bin);
                   objects already inside a bin are left alone
    4. move        go_to_object's pick and drop: down to it, close, lift, carry over the bin at 11 cm, open,
                   fold home
and again, until nothing is left to sort. Detecting again every cycle finds bins and objects where they are
now (round 3 moves them). An object that fails twice (it slipped, the jaws closed on nothing) is left.

Usage:
    python robot/sort_objects.py                       # the round in sort_rules.json; asks before starting
    python robot/sort_objects.py --round shape
    python robot/sort_objects.py --dry                 # detect and print the plan; the arm does not move
    python robot/sort_objects.py --yes                 # no question (the picture is still saved)
    python robot/sort_objects.py --case point_cloud_matching/cases/case_... --sim --yes   # no camera, no arm

Ctrl-C stops the move and folds the arm home; a second Ctrl-C switches the torque off (hold the arm).
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import detect  # noqa: E402
import go_to_object as g  # noqa: E402

RULES_FILE = HERE / "sort_rules.json"
MIN_COLOUR_CONF = 0.6      # a colour vote under this is too unsure to sort by
MAX_TRIES = 2              # attempts at one object before it is left
DRAW = {"red": (0, 0, 255), "yellow": (0, 220, 255), "green": (0, 170, 0), "blue": (255, 80, 0)}


def load_rules(path=RULES_FILE, round_name=None):
    """(round name, {bin colour: [terms]}) from the JSON file."""
    data = json.loads(Path(path).read_text())
    name = round_name or data["round"]
    if name not in data["rounds"]:
        raise SystemExit(f"round {name!r} is not in {path}; there are: {', '.join(data['rounds'])}")
    return name, {colour.lower(): [str(t).lower() for t in terms] for colour, terms in data["rounds"][name].items()}


def bin_colour_for(d, rules):
    """The bin colour an object goes to: the first rule whose term matches it. A term is a colour, a class or
    both ("yellow cube"): every word must be the object's colour or its class."""
    for colour, terms in rules.items():
        for term in terms:
            if all(word in (d.colour, d.name.lower()) for word in term.split()):
                return colour
    return None


@dataclass
class Task:
    obj: object
    bin_colour: str = None
    bin: object = None
    pick: tuple = None         # (goal, pitch, poses)
    drop: object = None        # go_to_object.Place
    status: str = ""

    @property
    def ok(self):
        return self.status == "ok"


def inside(d, outer):
    """d's centre pixel inside outer's mask outline."""
    import cv2

    if d.pixel is None or outer.polygon is None or len(outer.polygon) < 3:
        return False
    return cv2.pointPolygonTest(np.asarray(outer.polygon, np.float32), tuple(map(float, d.pixel)), False) >= 0


def find_bins(dets):
    """{colour: bin Detection}, the most confident bin of each colour."""
    bins = {}
    for d in dets:
        if d.name == "bin" and d.colour_conf >= MIN_COLOUR_CONF:
            if d.colour not in bins or d.conf > bins[d.colour].conf:
                bins[d.colour] = d
    return bins


def plan_tasks(dets, model, rules):
    """A Task for every object (not bins or the gripper), ok ones first, the most confident first."""
    bins = find_bins(dets)
    all_bins = [d for d in dets if d.name == "bin"]
    tasks = []
    for d in dets:
        if d.name in detect.SKIP_CLASSES:
            continue
        t = Task(d)
        tasks.append(t)
        if any(inside(d, b) for b in all_bins):
            t.status = "already in a bin"
            continue
        if d.colour_conf < MIN_COLOUR_CONF:
            t.status = f"colour unsure ({d.colour} {d.colour_conf:.2f})"
            continue
        t.bin_colour = bin_colour_for(d, rules)
        if t.bin_colour is None:
            t.status = "no rule for it"
            continue
        t.bin = bins.get(t.bin_colour)
        if t.bin is None:
            t.status = f"no {t.bin_colour} bin seen"
            continue
        d.plan = None
        t.pick = g.plan_pick(d, model)
        if t.pick is None:
            t.status = "out of reach"
            continue
        t.drop = g.plan_drop(model, t.bin, t.pick[0][2], like=t.pick[2])
        if t.drop is None:
            t.status = f"the {t.bin_colour} bin is out of reach"
            continue
        t.status = "ok"
    tasks.sort(key=lambda t: (not t.ok, -t.obj.conf * t.obj.colour_conf))
    return tasks


def draw_plan(scene, dets, tasks, path, head, model=None):
    """The picture: masks shaded in their detected colour, an arrow from each object to its bin, saved."""
    import cv2

    img = scene.frame.copy()
    shade = img.copy()
    for d in dets:
        if d.polygon is not None and len(d.polygon) >= 3:
            cv2.fillPoly(shade, [np.rint(d.polygon).astype(np.int32)], DRAW.get(d.colour, (160, 160, 160)))
    img = cv2.addWeighted(shade, 0.35, img, 0.65, 0)
    if model is not None:
        g.draw_reach(img, g.reach_outline(scene, model))

    def label(text, org, c, scale=0.6):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, c, 2)

    for d in dets:
        if d.polygon is not None and len(d.polygon) >= 3:
            cv2.polylines(img, [np.rint(d.polygon).astype(np.int32)], True, DRAW.get(d.colour, (160, 160, 160)),
                          4 if d.name == "bin" else 2)
        if d.name == "bin" and d.pixel is not None:
            label(f"{d.colour.upper()} BIN", (int(d.pixel[0]) - 50, int(d.pixel[1])), (255, 255, 255), 0.8)
            label(f"yolo {d.conf:.2f} colour {d.colour_conf:.2f}", (int(d.pixel[0]) - 50, int(d.pixel[1]) + 24),
                  (255, 255, 255), 0.5)
    for n, t in enumerate(tasks):
        d = t.obj
        if d.pixel is None:
            continue
        px = tuple(map(int, d.pixel))
        label(f"yolo {d.conf:.2f} colour {d.colour_conf:.2f}", (px[0] + 12, px[1] + 16), (255, 255, 255), 0.5)
        if t.ok:
            c = (0, 0, 255) if n == 0 else (255, 255, 255)
            cv2.arrowedLine(img, px, tuple(map(int, t.bin.pixel)), c, 4 if n == 0 else 2, tipLength=0.03)
            size = f"{d.fit.size} " if d.fit is not None else ""
            label(f"{d.colour} {size}{d.name} -> {t.bin_colour} bin" + ("  (NEXT)" if n == 0 else ""),
                  (px[0] + 12, px[1] - 8), c)
        else:
            label(f"{d.colour} {d.name}: {t.status}", (px[0] + 12, px[1] - 8), (0, 140, 255))
    label(head, (20, 45), (255, 255, 255), 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return img


def confirm(img, look_only=False):
    """Show the plan; True on g/Enter, False on q/Esc or a closed window. Without a display, ask in the terminal.
    look_only: any key (or closing the window) closes it."""
    import cv2

    win = "sort_objects"
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, img)
    except cv2.error:
        return False if look_only else input("Start sorting? [y/N] ").strip().lower() == "y"
    print("Picture window: click on it, then " + ("any key closes it" if look_only
                                                  else "g or Enter = start sorting, q or Esc = quit"))
    try:
        while True:
            k = cv2.waitKey(100)
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                return False
            if k != -1 and look_only:
                return False
            if k != -1 and (k & 0xFF) in (ord("g"), 13, 10):
                return True
            if k != -1 and (k & 0xFF) in (ord("q"), 27):
                return False
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def key(d):
    """An object's identity across detections: its class and where it is, to 2 cm."""
    return d.name, round(d.base_xyz[0] / 0.02), round(d.base_xyz[1] / 0.02)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", help="rules to use from sort_rules.json (default: its 'round')")
    ap.add_argument("--rules", default=str(RULES_FILE))
    ap.add_argument("--case", help="a saved point-cloud case instead of the camera (sorted once, no re-detection)")
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--weights")
    ap.add_argument("--max", type=int, default=50, help="stop after this many objects")
    ap.add_argument("--dry", action="store_true", help="detect and print the plan; the arm does not move")
    ap.add_argument("--yes", action="store_true", help="do not ask before starting")
    ap.add_argument("--sim", action="store_true", help="so101ik's simulated servos instead of the arm")
    ap.add_argument("--calibration", default=str(g.CALIBRATION))
    ap.add_argument("--offsets", default=str(g.FINE_OFFSETS))
    args = ap.parse_args()

    from so101ik import Arm, Model

    round_name, rules = load_rules(args.rules, args.round)
    print(f"Round {round_name!r}: " + ", ".join(f"{c} bin <- {' / '.join(ts)}" for c, ts in rules.items()))
    cal = Path(args.calibration).expanduser()
    if not (args.sim or args.dry) and not cal.is_file():
        raise SystemExit(f"Calibration file not found: {cal}; pass --calibration")
    offsets = {j: 0.0 for j in g.ARM} if args.offsets == "none" else g.fine_offsets(args.offsets)
    model = Model(calibration_file=cal if cal.is_file() else None, offsets=offsets)
    out = g.ROOT / "outputs" / "sort_objects"
    tries, done, sorted_n, asked = {}, set(), 0, args.yes

    while sorted_n < args.max:
        # 1. detect (the arm is folded)
        if args.case:
            dets, scene = detect.detect_from_case(args.case)
        else:
            dets, scene = detect.detect_live(args.frames, args.weights)
        dets = [d for d in dets if key(d) not in done]          # (a case does not change: drop what was sorted)

        # 2-3. assign and pick one
        tasks = plan_tasks(dets, model, rules)
        for t in tasks:
            if t.ok and tries.get(key(t.obj), 0) >= MAX_TRIES:
                t.status = f"failed {MAX_TRIES} times"
        tasks.sort(key=lambda t: not t.ok)
        print(f"\n{'-' * 100}\n{len(tasks)} objects, bins: "
              + (", ".join(f"{c} {np.round(b.base_xyz[:2], 3)}" for c, b in find_bins(dets).items()) or "none"))
        for t in tasks:
            x, y, _ = t.obj.base_xyz
            print(f"  {t.obj.colour:>7} {t.obj.name:<13} x {x:+.3f} y {y:+.3f}  "
                  + (f"-> {t.bin_colour} bin" if t.ok else t.status))
        todo = [t for t in tasks if t.ok]
        head = (f"round {round_name}: {len(todo)} to sort" + (f", next: {todo[0].obj.colour} {todo[0].obj.name} -> "
                f"{todo[0].bin_colour} bin" if todo else "") + "   (g = start, q = quit)" * (not asked))
        img = None
        if scene is not None and scene.frame is not None:
            img = draw_plan(scene, dets, tasks, out / time.strftime("plan_%Y%m%d_%H%M%S.jpg"), head, model)
            print(f"Picture: {sorted(out.glob('plan_*.jpg'))[-1]}")
        if args.dry or not todo:
            if not todo:
                print(f"\nNothing (more) to sort. {sorted_n} sorted.")
            if img is not None and not args.yes:
                confirm(img, look_only=True)
            return
        if not asked:
            if not (confirm(img) if img is not None else input("Start sorting? [y/N] ").strip().lower() == "y"):
                return
            asked = True

        # 4. pick it and drop it in its bin
        t = todo[0]
        print(f"\n>>> {t.obj.colour} {t.obj.name} -> {t.bin_colour} bin")
        with Arm(port="sim" if args.sim else None, calibration_file=cal if cal.is_file() else None, offsets=offsets,
                 name="sort_objects") as arm:
            arm._sim = args.sim
            if args.sim:
                arm.raw.sag = 0.0
            goal = t.pick[0]
            pitch, poses = g.plan(arm.model, goal, width=t.obj.width)
            drop = g.replan(arm.model, t.drop, goal[2], like=poses)
            status = g.go(arm, goal, pitch, poses, hold=0.0, place=drop)
        print(f"<<< {status}")
        if status == "placed":
            sorted_n += 1
            if args.case:
                done.add(key(t.obj))
        else:
            tries[key(t.obj)] = tries.get(key(t.obj), 0) + 1
    print(f"\nStopped after {sorted_n} objects (--max).")


if __name__ == "__main__":
    main()
