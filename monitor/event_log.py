"""Every behaviour-tree step, plan and move of a round, one JSON line each, to look back at a mistake afterwards.

    log = EventLog()            # outputs/behaviour_log.jsonl (appended); subscribes to core/telemetry.py
    log.close()

Each line: {"time": "2026-11-14 10:03:04.21", "kind": "node", "arm": ..., ...} (the event's fields; see
core/telemetry.py). The measured joints (many per second) are left out; plans keep only their step names and end
poses.
"""
from __future__ import annotations

import json
import threading
import time

from core import telemetry


class EventLog:
    def __init__(self, path=None):
        from core.paths import OUTPUTS

        self.path = path or OUTPUTS / "behaviour_log.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a")
        self.lock = threading.Lock()
        self.handle.write(json.dumps({"time": self._now(), "kind": "start"}) + "\n")
        telemetry.subscribe(self.on_event)

    @staticmethod
    def _now():
        now = time.time()
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)) + f".{int(now * 100) % 100:02d}"

    def on_event(self, e):
        if e["kind"] == "joints":
            return
        row = {"time": self._now(), **{k: v for k, v in e.items() if k != "t"}}
        if e["kind"] == "plan":
            row["plans"] = {step: [round(v, 4) for v in path[-1]] for step, path in e["plans"]}
        elif e["kind"] == "move":
            row["path"] = [round(v, 4) for v in e["path"][-1]] if e["path"] else []
        elif e["kind"] == "tree":
            row.pop("structure", None)
        with self.lock:
            self.handle.write(json.dumps(row, default=str) + "\n")
            self.handle.flush()

    def close(self):
        telemetry.unsubscribe(self.on_event)
        with self.lock:
            self.handle.close()
