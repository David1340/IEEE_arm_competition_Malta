"""What the arms are doing, as events, for the live views (monitor/): a small thread-safe publish / subscribe.

    from core import telemetry
    telemetry.subscribe(fn)                       # fn(event: dict) is called for every event, from any thread
    telemetry.publish("node", arm="follower_pink", path="0/3", status="running")
    telemetry.active()                            # whether anyone listens (skip expensive work when not)

Every event is a dict with "kind" and "t" (time.monotonic()) plus its own fields. The kinds:
    tree      arm, task, attempt, structure     a new attempt: its behaviour tree (state_machine/behaviour.py structure)
    node      arm, path, name, status, why      a tree node started ("running") or ended ("success" / "failure")
    outcome   arm, task, result, why            how an attempt ended (done, retry, later, other arm, left, ...)
    worker    arm, state                        the arm's outer state machine: waiting, attempt, finished, error
    plan      arm, task, plans, target, goal    the planned joint paths of an attempt [(step, [q, ...]), ...]
    move      arm, step, path, gripper          the arm starts along a joint path
    moved     arm, step, ok                     ... and has finished it
    joints    arm, q, gripper                   measured joint angles [rad] while moving (about 25 per second)

Nothing listens unless a game subscribes: publishing is then a no-op. A subscriber that raises is dropped (a view
must never stop an arm). core imports nothing from the project, so every package may publish.
"""
from __future__ import annotations

import threading
import time

_LOCK = threading.Lock()
_SUBSCRIBERS: list = []


def subscribe(fn) -> None:
    with _LOCK:
        if fn not in _SUBSCRIBERS:
            _SUBSCRIBERS.append(fn)


def unsubscribe(fn) -> None:
    with _LOCK:
        if fn in _SUBSCRIBERS:
            _SUBSCRIBERS.remove(fn)


def active() -> bool:
    return bool(_SUBSCRIBERS)


def publish(kind: str, **fields) -> None:
    if not _SUBSCRIBERS:
        return
    event = {"kind": kind, "t": time.monotonic(), **fields}
    with _LOCK:
        listeners = list(_SUBSCRIBERS)
    for fn in listeners:
        try:
            fn(event)
        except Exception as error:                          # noqa: BLE001 - a view must never stop an arm
            print(f"(live view {getattr(fn, '__qualname__', fn)} stopped: {error})")
            unsubscribe(fn)
