"""The pick-and-place rules from pick_and_place.json (repo root): the sorting modes and which bin each object
goes in.

    from control import task_config
    task_config.modes()                                  # ["colour", "shape"]
    task_config.default_mode()                           # "colour"
    task_config.bin_colour("cube", "blue", "shape")      # "red"   (sorted by shape)
    task_config.bin_colour("cube", "blue", "colour")     # "blue"  (sorted by colour)

Edit pick_and_place.json to change a rule or add a mode; experiments/pick_and_place.py reads it from here.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "pick_and_place.json"
SORT_BY = ("colour", "shape")


def load() -> dict:
    """pick_and_place.json as a dict, checked: every mode sorts by colour or shape and has a bins table."""
    if not FILE.is_file():
        raise SystemExit(f"{FILE} not found: it holds the sorting modes (see control/task_config.py)")
    data = json.loads(FILE.read_text())
    modes = data.get("modes") or {}
    if not modes:
        raise SystemExit(f"{FILE.name}: no modes")
    for name, mode in modes.items():
        if mode.get("by") not in SORT_BY:
            raise SystemExit(f"{FILE.name}: mode {name!r} has by={mode.get('by')!r}; use one of {SORT_BY}")
        if not isinstance(mode.get("bins"), dict) or not mode["bins"]:
            raise SystemExit(f"{FILE.name}: mode {name!r} has no 'bins' table")
    if data.get("mode", next(iter(modes))) not in modes:
        raise SystemExit(f"{FILE.name}: mode {data['mode']!r} is not one of {list(modes)}")
    return data


def modes() -> list[str]:
    return list(load()["modes"])


def default_mode() -> str:
    data = load()
    return data.get("mode", next(iter(data["modes"])))


def mode(name: str) -> dict:
    all_modes = load()["modes"]
    if name not in all_modes:
        raise SystemExit(f"unknown mode {name!r} in {FILE.name}; known: {', '.join(all_modes)}")
    return all_modes[name]


def bin_colour(shape: str | None, colour: str | None, mode_name: str) -> str | None:
    """The colour of the bin an object goes in under that mode, or None when the rules do not say (unknown
    colour or shape)."""
    m = mode(mode_name)
    key = colour if m["by"] == "colour" else shape
    if key is None:
        return None
    bins = {k.casefold(): v for k, v in m["bins"].items() if not k.startswith("_")}
    return bins.get(key.casefold())


def describe(mode_name: str) -> str:
    """One line with the mode's rules, e.g. 'shape: cube -> red, cylinder -> blue, ...'."""
    m = mode(mode_name)
    return f"{mode_name} (by {m['by']}): " + ", ".join(f"{k} -> {v}" for k, v in m["bins"].items()
                                                         if not k.startswith("_"))
