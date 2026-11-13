"""Game definitions: each game's rules are in config/games/<game>.yaml.

    from games import rules
    rules.load("pick_and_place")                                        # the whole file, checked
    rules.modes()                                                       # ["colour", "shape"]
    rules.default_mode()                                                # "colour"
    rules.bin_colour("cube", "blue", "shape")                           # "red"   (sorted by shape)
    rules.bin_colour("cube", "blue", "colour")                          # "blue"  (sorted by colour)

The sorting functions (modes, bin_colour, describe) are for the pick-and-place game, the default `game`.
Edit config/games/pick_and_place.yaml to change a rule or add a mode.
"""
from __future__ import annotations

from core import config
from core.paths import GAMES_CONFIG

PICK_AND_PLACE = "pick_and_place"
SORT_BY = ("colour", "shape")


def file(game: str):
    return GAMES_CONFIG / f"{game}.yaml"


def available() -> list[str]:
    """Every game with a config file."""
    return sorted(path.stem for path in GAMES_CONFIG.glob("*.yaml"))


def load(game: str = PICK_AND_PLACE) -> dict:
    """config/games/<game>.yaml as a dict. For pick and place it is checked: every mode sorts by colour or shape and
    has a bins table, and the default mode exists."""
    path = file(game)
    data = config.load(path, {})
    if game != PICK_AND_PLACE:
        return data
    modes_ = data.get("modes") or {}
    if not modes_:
        raise config.ConfigError(f"{path.name}: no modes")
    for name, mode_ in modes_.items():
        if mode_.get("by") not in SORT_BY:
            raise config.ConfigError(f"{path.name}: mode {name!r} has by={mode_.get('by')!r}; use one of {SORT_BY}")
        if not isinstance(mode_.get("bins"), dict) or not mode_["bins"]:
            raise config.ConfigError(f"{path.name}: mode {name!r} has no 'bins' table")
    if data.get("mode", next(iter(modes_))) not in modes_:
        raise config.ConfigError(f"{path.name}: mode {data['mode']!r} is not one of {list(modes_)}")
    return data


def modes(game: str = PICK_AND_PLACE) -> list[str]:
    return list(load(game)["modes"])


def default_mode(game: str = PICK_AND_PLACE) -> str:
    data = load(game)
    return data.get("mode", next(iter(data["modes"])))


def mode(name: str, game: str = PICK_AND_PLACE) -> dict:
    all_modes = load(game)["modes"]
    if name not in all_modes:
        raise SystemExit(f"unknown mode {name!r} in {file(game).name}; known: {', '.join(all_modes)}")
    return all_modes[name]


def bin_colour(shape: str | None, colour: str | None, mode_name: str, game: str = PICK_AND_PLACE) -> str | None:
    """The colour of the bin an object goes in under that mode, or None when the rules do not say (unknown
    colour or shape)."""
    m = mode(mode_name, game)
    key = colour if m["by"] == "colour" else shape
    if key is None:
        return None
    bins = {str(k).casefold(): v for k, v in m["bins"].items()}
    return bins.get(key.casefold())


def describe(mode_name: str, game: str = PICK_AND_PLACE) -> str:
    """One line with the mode's rules, e.g. 'shape (by shape): cube -> red, cylinder -> blue, ...'."""
    m = mode(mode_name, game)
    return f"{mode_name} (by {m['by']}): " + ", ".join(f"{k} -> {v}" for k, v in m["bins"].items())
