"""The behaviour-tree window: for each arm, its outer state machine, the behaviour tree of the attempt running now
with every node's state, and a history of what each arm did.

    view = TreeView(["follower_pink", "follower_blue"])     # subscribes to core/telemetry.py
    view.show()                                             # in the main loop (OpenCV window), after cv2.waitKey
    view.toggle_freeze()                                    # f in the game: hold the picture to look at a mistake
    image = view.render()                                   # the picture alone (BGR), e.g. for a screenshot
    view.close()

Per arm, top to bottom:
  - the arm, the task it works on and the attempt (2/3), and what it does now (the running action, the move).
  - the outer state machine of state_machine/arm_worker.py arm_worker: waiting for a task -> attempt (the tree)
    -> done / retry / later / other arm / left -> waiting ...; finished or error at the end. The state it is in is
    filled; the last outcome stays lit.
  - the attempt's behaviour tree (state_machine/arm_worker.py task_tree): "->" a Sequence (all in order), "?" a
    Fallback (the first that works), boxes the actions. Grey: not run yet; amber: running (with its time so far);
    green: succeeded; red: failed, with why; pale: not run because an earlier node failed.
  - the last failure, kept after the next attempt has started.
Below both: the history, newest first (every action's end, every attempt's outcome), with the time.

Nothing here talks to the arms: it only listens (a view must never stop an arm).
"""
from __future__ import annotations

import collections
import threading
import time

import cv2
import numpy as np

from core import telemetry

WINDOW = "behaviour tree (f = freeze / unfreeze)"
FONT = cv2.FONT_HERSHEY_SIMPLEX
ARM_BGR = {"follower_pink": (180, 105, 255), "follower_blue": (255, 140, 0)}

BG = (250, 250, 248)
INK = (40, 40, 40)
MUTED = (140, 140, 140)
LINE = (190, 190, 190)
FILL = {"pending": (228, 228, 228), "running": (60, 200, 255), "success": (120, 205, 120),
        "failure": (90, 90, 235), "skipped": (242, 242, 242)}
TEXT = {"pending": INK, "running": INK, "success": INK, "failure": (255, 255, 255), "skipped": MUTED}
OUTCOME_BGR = {"done": (120, 205, 120), "retry": (60, 170, 255), "later": (60, 170, 255),
               "other arm": (60, 170, 255), "left": (90, 90, 235), "error": (90, 90, 235)}
OUTER = ["waiting", "attempt", "outcome", "finished"]           # the outer state machine, left to right
KIND = {"Sequence": "->", "Fallback": "?", "Fail": "x"}

ROW, INDENT, MARGIN = 26, 26, 14


def short(arm):
    return arm.replace("follower_", "")


def text(img, s, org, colour=INK, scale=0.5, thick=1):
    cv2.putText(img, s, (int(org[0]), int(org[1])), FONT, scale, colour, thick, cv2.LINE_AA)


def text_width(s, scale=0.5, thick=1):
    return cv2.getTextSize(s, FONT, scale, thick)[0][0]


def pill(img, x, y, label, fill, ink=INK, scale=0.45, border=None, pad=7, h=22):
    w = text_width(label, scale) + 2 * pad
    cv2.rectangle(img, (x, y), (x + w, y + h), fill, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + w, y + h), border or tuple(max(0, c - 60) for c in fill), 1, cv2.LINE_AA)
    text(img, label, (x + pad, y + h - 7), ink, scale)
    return x + w


def ago(seconds):
    return f"{seconds:.1f} s" if seconds < 60 else f"{seconds / 60:.0f} min"


class ArmTrack:
    """What the view knows about one arm."""

    def __init__(self):
        self.structure = None
        self.rows = []                    # [(path, name, type, depth, parent path)]
        self.nodes = {}                   # path -> {"status", "start", "end", "why"}
        self.task, self.attempt, self.attempts = "", 0, 0
        self.worker = "waiting"
        self.outcome = None               # (result, why, t) of the last attempt
        self.running = None               # (name, start) of the running action
        self.move = None                  # (step, start) of the running move
        self.last_failure = None          # (task, node, why, t): the first failure of the last attempt that had one
        self.failed_now = False           # this attempt has failed somewhere already
        self.finished_tree = False
        self.counts = collections.Counter()


class TreeView:
    def __init__(self, arms, width=1200, history=12):
        self.arms = {arm: ArmTrack() for arm in arms}
        self.width, self.history_rows = width, history
        self.history = collections.deque(maxlen=300)    # (wall time, arm, text, colour)
        self.lock = threading.Lock()
        self.frozen = None                # the picture held by toggle_freeze()
        self.window = False
        self.last_shown = 0.0
        telemetry.subscribe(self.on_event)

    # ------------------------------------------------------------------------------------------ events
    def on_event(self, e):
        arm = e.get("arm")
        if arm is None or e["kind"] in ("joints", "plan"):
            return
        with self.lock:
            track = self.arms.setdefault(arm, ArmTrack())
            kind = e["kind"]
            if kind == "tree":
                track.structure = e["structure"]
                track.rows = list(self._flatten(e["structure"]))
                track.nodes = {}
                track.task, track.attempt, track.attempts = e["task"], e["attempt"], e.get("attempts", 0)
                track.running = track.move = None
                track.finished_tree = track.failed_now = False
                self._log(arm, f"attempt {track.attempt}: {track.task}", INK)
            elif kind == "node":
                node = track.nodes.setdefault(e["path"], {})
                node["status"] = e["status"]
                if e["status"] == "running":
                    node["start"], node["end"], node["why"] = e["t"], None, ""
                    if e.get("type", "Action") == "Action":
                        track.running = (e["name"], e["t"])
                else:
                    node["end"] = e["t"]
                    node["why"] = e.get("why", "") if e["status"] == "failure" else ""
                    leaf = e.get("type", "Action") in ("Action", "Fail")
                    if leaf:
                        took = e["t"] - node.get("start", e["t"])
                        if e["status"] == "success":
                            self._log(arm, f"{e['name']}  ok  {took:.1f} s", (60, 140, 60))
                        else:
                            why = node["why"] or "failed"
                            self._log(arm, f"{e['name']}  FAILED: {why}", (60, 60, 210))
                            if not track.failed_now:      # the cause, not the clean-up branch's "fail: missed"
                                track.last_failure = (track.task, e["name"], why, time.time())
                                track.failed_now = True
                    if track.running and track.running[0] == e["name"]:
                        track.running = None
                    if e["path"] == "0":
                        track.finished_tree = True
            elif kind == "move":
                track.move = (e.get("step") or "move", e["t"])
            elif kind == "moved":
                track.move = None
            elif kind == "outcome":
                result, why = e["result"], e.get("why", "")
                track.outcome = (result, why, e["t"])
                track.counts[result] += 1
                self._log(arm, f"=> {result}" + (f" ({why})" if why else ""), OUTCOME_BGR.get(result, INK))
            elif kind == "worker":
                track.worker = e["state"]
                if e["state"] == "error":
                    track.outcome = ("error", "see the terminal", e["t"])
                    self._log(arm, "ERROR: stopped (see the terminal)", (60, 60, 210))
                elif e["state"] == "finished":
                    self._log(arm, "no more tasks", MUTED)

    def _log(self, arm, s, colour):
        self.history.appendleft((time.time(), arm, s, colour))

    @staticmethod
    def _flatten(node, depth=0, parent=None):
        yield node["path"], node["name"], node["type"], depth, parent
        for child in node["children"]:
            yield from TreeView._flatten(child, depth + 1, node["path"])

    # ------------------------------------------------------------------------------------------ drawing
    def toggle_freeze(self):
        self.frozen = None if self.frozen is not None else self.render()
        print("behaviour tree view " + ("frozen (f again to follow live)" if self.frozen is not None else "live"))

    def render(self):
        with self.lock:
            arms = list(self.arms.items())
            history = list(self.history)[:self.history_rows]
            heights = [self._arm_height(t) for _, t in arms]
            panel_h = max(heights + [300])
            h = 40 + panel_h + 30 + 22 * self.history_rows + 16
            img = np.full((h, self.width, 3), BG, np.uint8)
            now = time.monotonic()
            text(img, "Behaviour trees", (MARGIN, 27), INK, 0.7, 2)
            text(img, time.strftime("%H:%M:%S"), (self.width - 90, 27), MUTED, 0.55)
            col_w = self.width // max(1, len(arms))
            for i, (arm, track) in enumerate(arms):
                x0 = i * col_w
                if i:
                    cv2.line(img, (x0, 40), (x0, 40 + panel_h), LINE, 1)
                self._draw_arm(img, arm, track, x0 + MARGIN, 44, col_w - 2 * MARGIN, now)
            y = 40 + panel_h + 8
            cv2.line(img, (0, y), (self.width, y), LINE, 1)
            text(img, "History (newest first)", (MARGIN, y + 20), INK, 0.55, 1)
            for j, (stamp, arm, s, colour) in enumerate(history):
                yy = y + 44 + 22 * j
                text(img, time.strftime("%H:%M:%S", time.localtime(stamp)) + f".{int(stamp * 10) % 10}",
                     (MARGIN, yy), MUTED, 0.45)
                cv2.circle(img, (MARGIN + 92, yy - 5), 5, ARM_BGR.get(arm, MUTED), -1, cv2.LINE_AA)
                text(img, short(arm), (MARGIN + 102, yy), INK, 0.45)
                text(img, s, (MARGIN + 160, yy), colour, 0.45)
        return img

    def _arm_height(self, track):
        return 150 + ROW * max(len(track.rows), 1) + 40

    def _draw_arm(self, img, arm, track, x, y, w, now):
        colour = ARM_BGR.get(arm, MUTED)
        cv2.rectangle(img, (x, y), (x + 14, y + 22), colour, -1)
        text(img, arm, (x + 22, y + 17), INK, 0.6, 2)
        counts = " ".join(f"{k} {v}" for k, v in sorted(track.counts.items()))
        if counts:
            text(img, counts, (x + w - text_width(counts, 0.45), y + 16), MUTED, 0.45)
        task = track.task or "(no task yet)"
        if track.attempts:
            task += f"   attempt {track.attempt}/{track.attempts}"
        text(img, task[:90], (x, y + 42), INK, 0.48)
        doing = []
        if track.running:
            doing.append(f"now: {track.running[0]} ({now - track.running[1]:.1f} s)")
        if track.move:
            doing.append(f"moving: {track.move[0]}")
        text(img, "   ".join(doing) or ("idle" if not track.finished_tree else "between attempts"),
             (x, y + 62), (0, 120, 200) if doing else MUTED, 0.48)

        # the outer state machine
        yy = y + 76
        xx = x
        state = {"waiting": "waiting", "attempt": "attempt", "finished": "finished", "error": "finished"}.get(
            track.worker, "waiting")
        outcome = track.outcome[0] if track.outcome else None
        for k, name in enumerate(OUTER):
            if name == "outcome":
                label = outcome or "outcome"
                fill = OUTCOME_BGR.get(outcome, FILL["skipped"]) if outcome else FILL["skipped"]
                ink = (255, 255, 255) if outcome in ("left", "error") else (INK if outcome else MUTED)
                if outcome and (track.worker == "attempt" or now - track.outcome[2] > 3.0):
                    fill = tuple(int(0.45 * c + 0.55 * 255) for c in fill)   # an older outcome fades
            elif name == "finished" and track.worker == "error":
                label, fill, ink = "error", OUTCOME_BGR["error"], (255, 255, 255)
            else:
                label = {"waiting": "waiting for a task", "attempt": "attempt"}.get(name, name)
                active = state == name
                fill, ink = (FILL["running"], INK) if active else (FILL["skipped"], MUTED)
            xx = pill(img, xx, yy, label, fill, ink, border=(30, 30, 30) if name == state else None)
            if k < len(OUTER) - 1:
                cv2.arrowedLine(img, (xx + 3, yy + 11), (xx + 19, yy + 11), MUTED, 1, cv2.LINE_AA, tipLength=0.4)
                xx += 22
        if track.worker == "waiting" and outcome:
            text(img, "(back to waiting)", (x, yy + 40), MUTED, 0.4)

        # the tree
        top = y + 120
        if not track.rows:
            text(img, "no attempt yet", (x, top + 20), MUTED, 0.5)
            return
        place = {}
        for r, (path, name, kind, depth, parent) in enumerate(track.rows):
            cy = top + r * ROW + ROW // 2
            nx = x + depth * INDENT
            place[path] = (nx, cy)
            if parent is not None:
                px, py = place[parent]
                cv2.line(img, (px + 8, py + 9), (px + 8, cy), LINE, 1)
                cv2.line(img, (px + 8, cy), (nx, cy), LINE, 1)
            node = track.nodes.get(path, {})
            status = node.get("status", "skipped" if track.finished_tree else "pending")
            if kind in ("Sequence", "Fallback"):
                fill = FILL[status] if status != "pending" else FILL["skipped"]
                end = pill(img, nx, cy - 10, f"{KIND[kind]}  {name}", fill, TEXT[status], 0.45, h=20, pad=5)
            else:
                label = name if kind == "Action" else f"x {name}"
                end = pill(img, nx, cy - 10, label, FILL[status], TEXT[status], 0.45, h=20, pad=6)
            note = ""
            if status == "running" and node.get("start") is not None:
                note, ink = f"{now - node['start']:.1f} s ...", (0, 120, 200)
            elif status in ("success", "failure") and node.get("start") is not None and kind == "Action":
                note, ink = f"{node['end'] - node['start']:.1f} s", MUTED
            if status == "failure" and node.get("why") and kind in ("Action", "Fail"):
                note, ink = (note + "  " if note else "") + node["why"], (60, 60, 210)
            if note:
                text(img, note[:60], (end + 8, cy + 5), ink, 0.42)
        if track.last_failure:
            task, node, why, stamp = track.last_failure
            yy = top + len(track.rows) * ROW + 22
            text(img, f"last failure ({ago(time.time() - stamp)} ago): {node}: {why}"[:95], (x, yy), (60, 60, 210),
                 0.45)
            text(img, f"  on {task}"[:95], (x, yy + 18), MUTED, 0.42)

    def show(self, every=0.15):
        """Draw the window (at most every `every` s; the main loop's cv2.waitKey shows it)."""
        now = time.monotonic()
        if now - self.last_shown < every:
            return
        self.last_shown = now
        if not self.window:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            self.window = True
        cv2.imshow(WINDOW, self.frozen if self.frozen is not None else self.render())

    def close(self):
        telemetry.unsubscribe(self.on_event)
        if self.window:
            try:
                cv2.destroyWindow(WINDOW)
            except cv2.error:
                pass
