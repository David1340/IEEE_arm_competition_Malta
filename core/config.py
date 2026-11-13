"""Read and write the YAML files in config/.

    from core import config
    arms = config.load(ARMS_FILE)            # dict; re-read only when the file changed
    config.save(ARMS_FILE, arms)             # atomic; keeps the file's header comment

Every file starts with a comment block that explains it. save() keeps that block, so a tool writing a value back
(the USB setup, the calibration) never loses the explanation. Comments inside the data are not kept: put
explanations in the header.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml


class ConfigError(SystemExit):
    """A config file is missing or wrong; the message says which and how to fix it."""


@lru_cache(maxsize=32)
def _read(path: str, mtime_ns: int, size: int):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load(path: Path, default=None, *, required: bool = True):
    """The YAML file as Python data. A missing file raises ConfigError when required, else returns default.
    The result is cached until the file changes: copy it before changing it in place."""
    path = Path(path)
    if not path.is_file():
        if required:
            raise ConfigError(f"{path} not found")
        return default
    stat = path.stat()
    data = _read(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    return default if data is None else data


def load_copy(path: Path, default=None, *, required: bool = True):
    """load(), as a deep copy that is safe to change and save()."""
    import copy

    return copy.deepcopy(load(path, default, required=required))


def header(path: Path) -> str:
    """The comment block at the top of a YAML file ('' when there is none)."""
    path = Path(path)
    if not path.is_file():
        return ""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or (not line.strip() and lines):
            lines.append(line)
        else:
            break
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines) + "\n" if lines else ""


def dump(data) -> str:
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=None, allow_unicode=True, width=110)


def save(path: Path, data, comment: str | None = None) -> None:
    """Write data to path atomically. The file's existing header comment is kept; `comment` (lines without '#')
    is used when the file has none yet."""
    path = Path(path)
    top = header(path)
    if not top and comment:
        top = "".join(f"# {line}".rstrip() + "\n" for line in comment.strip().splitlines())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(top + ("\n" if top else "") + dump(data), encoding="utf-8")
    tmp.replace(path)
    _read.cache_clear()
