"""A scene: the objects and bins on the table, in the TABLE frame, saved as YAML in simulation/scenes/.

    scene = load("simulation/scenes/example.yaml")
    scene["objects"]   # [{shape: cube, size: large, colour: red, x: 0.30, y: 0.10, yaw_deg: 0}, ...]
    scene["bins"]      # [{colour: red, x: 0.12, y: 0.47}, ...]
    save(path, scene)

The simulator turns a scene into exactly what the camera would give the game: each object's top centre (x, y and
its STL height) with its shape, size and colour, and each bin's centre (marks()).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from core import config
from core.paths import ROOT

SCENES = ROOT / "simulation" / "scenes"
TESTS = SCENES / "tests"          # regression scenes: python 04_simulate.py test
SHAPES = ("cube", "cylinder", "dodecahedron", "hexagonal")
SIZES = ("small", "large")
COLOURS = ("red", "green", "blue", "yellow")
COMMENT = """A simulated table: objects and bins in the TABLE frame (metres, degrees; x forward, y towards follower_pink).
objects: shape (cube, cylinder, dodecahedron, hexagonal), size (small, large), colour (red, green, blue, yellow),
x, y, yaw_deg. bins: colour, x, y (the bin's centre).
Edit with `python 04_simulate.py edit <this file>`, run with `python 04_simulate.py run <this file>`."""


def resolve(path) -> Path:
    """A scene path as given, or a name in simulation/scenes/ ("example" -> simulation/scenes/example.yaml)."""
    path = Path(path)
    if path.suffix not in (".yaml", ".yml"):
        path = path.with_suffix(".yaml")
    if not path.is_absolute() and not path.exists():
        for folder in (SCENES, TESTS):
            if (folder / path).exists() or (folder is SCENES and len(path.parts) == 1 and not (TESTS / path).exists()):
                return folder / path
    return path


def available() -> list[str]:
    """Saved scenes: simulation/scenes/*.yaml, and the regression tests as tests/<name>."""
    return sorted(p.stem for p in SCENES.glob("*.yaml")) + sorted(f"tests/{p.stem}" for p in TESTS.glob("*.yaml"))


def stl(shape: str, size: str) -> Path:
    from perception.detection import OBJECT_SIZES, OBJECT_STL_FILES

    return OBJECT_SIZES.get(shape, {}).get(size) or OBJECT_STL_FILES[shape]


def check(scene: dict, where: str = "scene") -> dict:
    scene = {"objects": list(scene.get("objects") or []), "bins": list(scene.get("bins") or [])}
    for i, o in enumerate(scene["objects"]):
        if o.get("shape") not in SHAPES or o.get("size", "large") not in SIZES or o.get("colour") not in COLOURS:
            raise SystemExit(f"{where}: object {i} {o}: shape one of {SHAPES}, size {SIZES}, colour {COLOURS}")
        o.setdefault("size", "large")
        o.setdefault("yaw_deg", 0.0)
        o["x"], o["y"], o["yaw_deg"] = float(o["x"]), float(o["y"]), float(o["yaw_deg"])
    for i, b in enumerate(scene["bins"]):
        if b.get("colour") not in COLOURS:
            raise SystemExit(f"{where}: bin {i} {b}: colour one of {COLOURS}")
        b["x"], b["y"] = float(b["x"]), float(b["y"])
    return scene


def load(path) -> dict:
    path = resolve(path)
    return check(config.load_copy(path, {}), str(path))


def save(path, scene: dict) -> Path:
    path = resolve(path)
    clean = {"objects": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()}
                         for o in scene["objects"]],
             "bins": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()} for b in scene["bins"]]}
    config.save(path, clean, comment=COMMENT)
    return path


def marks(scene: dict):
    """(objects, bins) as games/pick_and_place.py detect() gives them: dicts with name, colour, size, xyz (TABLE
    frame; an object's top centre at its STL height, a bin's centre on the table), conf, and `source` = the index in
    the scene (the simulator uses it to find the body)."""
    from hardware.gripper import object_height

    objects = [{"name": o["shape"], "colour": o["colour"], "size": o["size"], "conf": 1.0, "source": i,
                "yaw": float(o.get("yaw_deg", 0.0)) % 180.0,
                "xyz": np.array([o["x"], o["y"], object_height(stl(o["shape"], o["size"]))]), "pixel": np.zeros(2)}
               for i, o in enumerate(scene["objects"])]
    bins = [{"name": "bin", "colour": b["colour"], "size": "", "conf": 1.0, "source": i,
             "xyz": np.array([b["x"], b["y"], 0.0]), "pixel": np.zeros(2)} for i, b in enumerate(scene["bins"])]
    return objects, bins
