"""Place objects and bins on the table from above, then save the scene (and run it).

    python 04_simulate.py edit                    # a new scene (saved as simulation/scenes/new.yaml)
    python 04_simulate.py edit example            # edit simulation/scenes/example.yaml
    python 04_simulate.py edit my_test --random 8 # start from 8 random objects and 4 bins

The view is the TABLE frame from above: x forward (up the screen), y towards follower_pink (left). It shows each
follower's base and reach (outer ring: --max-reach, inner: --min-reach), the middle strip between the arms, the
handoff points (+) and the taped working area (dashed).

    left click        place the current item on empty table; on an item: select it and drag it
    1 2 3 4           item: cube, cylinder, dodecahedron, hexagonal prism        Tab   item: bin
    r g b y           colour (of the current item, and of the selected one)
    s / l             small / large (current item and selected object)
    , .               turn the selected object 15 deg (shift: < > 5 deg)
    x / Delete        remove the selected item          c   remove everything
    w                 save                              Enter   save and run the simulation
    q                 save and quit                     Esc     quit without saving
"""
from __future__ import annotations

import math
import random

import cv2
import numpy as np

from core import arms as arm_config
from hardware.gripper import _stl_vertices
from simulation import scene as scenes

X_RANGE = (-0.08, 0.76)            # TABLE x shown (forward) [m]
Y_RANGE = (-0.66, 0.66)            # TABLE y shown [m]
PX_PER_M = 900
BGR = {"red": (40, 40, 220), "green": (40, 170, 40), "blue": (220, 90, 30), "yellow": (30, 210, 235)}
ARM_BGR = {"follower_pink": (180, 105, 255), "follower_blue": (255, 140, 0)}
SHAPE_KEYS = {ord("1"): "cube", ord("2"): "cylinder", ord("3"): "dodecahedron", ord("4"): "hexagonal"}
COLOUR_KEYS = {ord("r"): "red", ord("g"): "green", ord("b"): "blue", ord("y"): "yellow"}
WINDOW = "scene editor (Enter = run, w = save, q = save + quit, Esc = quit)"
AREA = {"near": 0.06, "forward": 0.50, "across": 0.80}   # the taped working area (02_calibrate.py camera offsets)


def to_px(x, y):
    return int(round((Y_RANGE[1] - y) * PX_PER_M)), int(round((X_RANGE[1] - x) * PX_PER_M))


def to_table(u, v):
    return X_RANGE[1] - v / PX_PER_M, Y_RANGE[1] - u / PX_PER_M


_OUTLINES = {}


def outline(shape, size):
    """The object's outline seen from above (convex hull of its STL), metres, centred."""
    key = (shape, size)
    if key not in _OUTLINES:
        v = _stl_vertices(scenes.stl(shape, size))[:, :2].astype(np.float32)
        v = v / 1000.0 if np.ptp(v) > 1.0 else v
        _OUTLINES[key] = cv2.convexHull(v).reshape(-1, 2).astype(float)
    return _OUTLINES[key]


class Editor:
    def __init__(self, scene, args):
        self.scene, self.args = scene, args
        self.item = {"kind": "object", "shape": "cube", "size": "large", "colour": "red"}
        self.selected = None            # ("objects" | "bins", index)
        self.dragging = False
        self.message = ""
        self.dirty = False

    # ---------------------------------------------------------------------------------------------- drawing
    def footprint_px(self, o):
        yaw = math.radians(o["yaw_deg"])
        R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        pts = outline(o["shape"], o["size"]) @ R.T + [o["x"], o["y"]]
        return np.array([to_px(x, y) for x, y in pts], np.int32)

    def bin_px(self, b):
        h = self.args.box_size / 2
        return np.array([to_px(b["x"] + dx, b["y"] + dy) for dx, dy in ((h, h), (h, -h), (-h, -h), (-h, h))], np.int32)

    def draw(self):
        w = int((Y_RANGE[1] - Y_RANGE[0]) * PX_PER_M)
        h = int((X_RANGE[1] - X_RANGE[0]) * PX_PER_M)
        img = np.full((h, w, 3), 236, np.uint8)
        for x in np.arange(-0.0, X_RANGE[1], 0.1):                          # 10 cm grid
            cv2.line(img, to_px(x, Y_RANGE[0]), to_px(x, Y_RANGE[1]), (222, 222, 222), 1)
        for y in np.arange(-0.6, 0.61, 0.1):
            cv2.line(img, to_px(X_RANGE[0], y), to_px(X_RANGE[1], y), (222, 222, 222), 1)
        half = arm_config.middle_half_width()                               # the middle strip
        overlay = img.copy()
        cv2.rectangle(overlay, to_px(X_RANGE[1], half), to_px(0.0, -half), (200, 235, 235), -1)
        img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
        a0, a1 = AREA["near"], AREA["near"] + AREA["forward"]               # the taped area, dashed
        corners = [(a1, AREA["across"] / 2), (a1, -AREA["across"] / 2), (a0, -AREA["across"] / 2), (a0, AREA["across"] / 2)]
        for (xa, ya), (xb, yb) in zip(corners, corners[1:] + corners[:1]):
            for t in np.arange(0, 1, 0.04):
                cv2.line(img, to_px(xa + (xb - xa) * t, ya + (yb - ya) * t),
                         to_px(xa + (xb - xa) * (t + 0.02), ya + (yb - ya) * (t + 0.02)), (60, 60, 60), 2)
        for name, b in arm_config.bases().items():                          # bases and reach
            colour = ARM_BGR.get(name, (120, 120, 120))
            centre = to_px(b["x"], b["y"])
            cv2.circle(img, centre, int(self.args.max_reach * PX_PER_M), colour, 2)
            cv2.circle(img, centre, int(self.args.min_reach * PX_PER_M), colour, 1)
            cv2.circle(img, centre, 14, colour, -1)
            yaw = math.radians(b["yaw_deg"])
            cv2.arrowedLine(img, centre, to_px(b["x"] + 0.07 * math.cos(yaw), b["y"] + 0.07 * math.sin(yaw)),
                            (40, 40, 40), 2, tipLength=0.3)
            cv2.putText(img, name.replace("follower_", ""), (centre[0] - 18, centre[1] + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
        for x, y in arm_config.handoff_points():
            cv2.drawMarker(img, to_px(x, y), (30, 30, 30), cv2.MARKER_CROSS, 16, 2)
        for i, b in enumerate(self.scene["bins"]):
            poly = self.bin_px(b)
            cv2.polylines(img, [poly], True, BGR[b["colour"]], 6)
            if self.selected == ("bins", i):
                cv2.polylines(img, [poly], True, (0, 0, 0), 2)
        for i, o in enumerate(self.scene["objects"]):
            poly = self.footprint_px(o)
            cv2.fillPoly(img, [poly], BGR[o["colour"]])
            cv2.polylines(img, [poly], True, (0, 0, 0) if self.selected == ("objects", i) else (90, 90, 90),
                          3 if self.selected == ("objects", i) else 1)
            u, v = to_px(o["x"], o["y"])
            cv2.putText(img, f"{o['shape'][0].upper()}{o['size'][0]}", (u - 10, v + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1)
        item = ("bin" if self.item["kind"] == "bin" else f"{self.item['size']} {self.item['shape']}")
        lines = [f"placing: {self.item['colour']} {item}   |   {len(self.scene['objects'])} objects, "
                 f"{len(self.scene['bins'])} bins" + ("   (unsaved)" if self.dirty else ""),
                 "1-4 shape  Tab bin  r g b y colour  s l size  , . turn  x delete  w save  Enter run  q quit",
                 self.message]
        for i, text in enumerate(lines):
            cv2.putText(img, text, (12, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
        return img

    # ---------------------------------------------------------------------------------------------- editing
    def hit(self, x, y):
        for i in reversed(range(len(self.scene["objects"]))):
            o = self.scene["objects"][i]
            if math.hypot(o["x"] - x, o["y"] - y) < 0.03:
                return ("objects", i)
        for i in reversed(range(len(self.scene["bins"]))):
            b = self.scene["bins"][i]
            if abs(b["x"] - x) < self.args.box_size / 2 and abs(b["y"] - y) < self.args.box_size / 2:
                return ("bins", i)
        return None

    def on_mouse(self, event, u, v, flags, _):
        x, y = to_table(u, v)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected = self.hit(x, y)
            if self.selected is None:
                if self.item["kind"] == "bin":
                    self.scene["bins"].append({"colour": self.item["colour"], "x": x, "y": y})
                    self.selected = ("bins", len(self.scene["bins"]) - 1)
                else:
                    self.scene["objects"].append({"shape": self.item["shape"], "size": self.item["size"],
                                                  "colour": self.item["colour"], "x": x, "y": y, "yaw_deg": 0.0})
                    self.selected = ("objects", len(self.scene["objects"]) - 1)
                self.dirty = True
            self.dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging and self.selected:
            entry = self.scene[self.selected[0]][self.selected[1]]
            entry["x"], entry["y"] = x, y
            self.dirty = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def on_key(self, key):
        sel = self.scene[self.selected[0]][self.selected[1]] if self.selected else None
        if key in SHAPE_KEYS:
            self.item.update(kind="object", shape=SHAPE_KEYS[key])
        elif key == 9:                                                       # Tab
            self.item["kind"] = "object" if self.item["kind"] == "bin" else "bin"
        elif key in COLOUR_KEYS:
            self.item["colour"] = COLOUR_KEYS[key]
            if sel is not None:
                sel["colour"] = COLOUR_KEYS[key]
                self.dirty = True
        elif key in (ord("s"), ord("l")):
            self.item["size"] = "small" if key == ord("s") else "large"
            if sel is not None and self.selected[0] == "objects":
                sel["size"] = self.item["size"]
                self.dirty = True
        elif key in (ord(","), ord("."), ord("<"), ord(">")) and sel is not None and self.selected[0] == "objects":
            sel["yaw_deg"] = (sel["yaw_deg"] + {ord(","): 15, ord("."): -15, ord("<"): 5, ord(">"): -5}[key]) % 360
            self.dirty = True
        elif key in (ord("x"), 255, 127, 8) and self.selected:
            del self.scene[self.selected[0]][self.selected[1]]
            self.selected = None
            self.dirty = True
        elif key == ord("c"):
            self.scene["objects"].clear()
            self.scene["bins"].clear()
            self.selected = None
            self.dirty = True


def random_scene(n_objects, args, seed=None):
    """n random objects in the working area clear of each other and of the bases, and the four bins at the sides."""
    rng = random.Random(seed)
    scene = {"objects": [], "bins": [{"colour": "red", "x": 0.14, "y": 0.46}, {"colour": "green", "x": 0.42, "y": 0.44},
                                     {"colour": "blue", "x": 0.14, "y": -0.44}, {"colour": "yellow", "x": 0.42, "y": -0.40}]}
    bases = list(arm_config.bases().values())
    tries = 0
    while len(scene["objects"]) < n_objects and tries < 5000:
        tries += 1
        x, y = rng.uniform(0.15, 0.48), rng.uniform(-0.30, 0.30)
        if any(math.hypot(x - o["x"], y - o["y"]) < 0.08 for o in scene["objects"]):
            continue
        if any(math.hypot(x - b["x"], y - b["y"]) < args.min_reach + 0.03 for b in bases):
            continue
        scene["objects"].append({"shape": rng.choice(scenes.SHAPES), "size": rng.choice(scenes.SIZES),
                                 "colour": rng.choice(scenes.COLOURS), "x": round(x, 3), "y": round(y, 3),
                                 "yaw_deg": float(rng.randrange(0, 360, 15))})
    return scene


def edit(path, args):
    """Open the editor; returns True when the scene should be run next (Enter)."""
    path = scenes.resolve(path)
    scene = scenes.load(path) if path.is_file() else {"objects": [], "bins": []}
    if getattr(args, "random", None):
        scene = random_scene(args.random, args, args.seed)
    editor = Editor(scene, args)
    editor.dirty = not path.is_file() or bool(getattr(args, "random", None))
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, editor.on_mouse)
    run_after = False
    while True:
        cv2.imshow(WINDOW, editor.draw())
        key = cv2.waitKey(30)
        if key == -1:
            continue
        key &= 0xFF                                    # Delete arrives as 255 on Linux; x works everywhere
        if key == 27:                                  # Esc: quit without saving
            if editor.dirty:
                print("Quit without saving.")
            break
        if key in (ord("w"), ord("q"), 13, 10):
            scenes.save(path, scenes.check(editor.scene))
            editor.dirty = False
            editor.message = f"saved {path}"
            print(f"Saved {path}")
            if key in (13, 10):
                run_after = True
            if key != ord("w"):
                break
            continue
        editor.on_key(key)
    cv2.destroyAllWindows()
    return run_after, path
