"""Two-arm task coordination: which arm does what, handoffs through the middle, and who may be where.

The motion layer (IK + RRT-Connect + follow_path) moves one arm from A to B. This layer decides the tasks:

    build_tasks(shapes, boxes, ...)   every shape becomes a DIRECT task (one arm reaches the shape and its box) or a
                                      HANDOFF (the arm nearest the shape puts it on a handoff point in the middle,
                                      then the other arm takes it from there to the box), or is skipped with a reason
    Regions                           the field is split along the line between the two bases into
                                      <arm A's side> | middle | <arm B's side>; a task needs every region it touches
                                      (always its own arm's side, the middle for a handoff point near the centre ...)
    Coordinator                       hands each arm (a worker thread running the arm's state machine) the next
                                      task it may do: its own, ready (a handoff's second half waits for the first),
                                      and whose regions are all free; it reserves them all at once (no deadlock)
                                      and frees them when the arm is back home

With parallel=False only one task runs at a time (the safe default); with parallel=True the arms work at the same
time whenever their tasks touch different regions, e.g. both working on their own sides.
"""
from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field

import numpy as np

DIRECT, TO_HANDOFF, FROM_HANDOFF = "direct", "to_handoff", "from_handoff"


@dataclass
class Task:
    id: int
    kind: str                     # DIRECT, TO_HANDOFF or FROM_HANDOFF
    arm: str
    shape: dict                   # the shape's mark: colour, xyz (TABLE frame; its top), state ...
    box: dict                     # the box it ends in (for TO_HANDOFF: where the second half takes it)
    slot: int | None = None       # handoff point index (TO_HANDOFF: where to put it; FROM_HANDOFF: where it is)
    then_arm: str | None = None   # TO_HANDOFF: the arm that does the second half
    depends_on: int | None = None
    state: str = "waiting"        # waiting, running, done, failed, cancelled
    attempts: int = 0
    regions: frozenset = field(default_factory=frozenset)
    note: str = ""

    def label(self):
        point = "" if self.slot is None else f" H{self.slot}"
        where = {DIRECT: f"-> {self.box['colour']} box", TO_HANDOFF: f"-> handoff{point}",
                 FROM_HANDOFF: f"handoff{point} -> {self.box['colour']} box"}[self.kind]
        return f"#{self.id} {self.arm}: {self.shape.get('label', self.shape['colour'] + ' shape')} {where}"


class Regions:
    """The field split along the line between the two arms' bases: each arm's side and the middle strip."""

    def __init__(self, bases: dict, middle_half_width: float):
        self.bases = {name: np.asarray(xy, float) for name, xy in bases.items()}
        self.half = float(middle_half_width)
        names = list(self.bases)
        self.pair = names[:2] if len(names) >= 2 else None
        if self.pair:
            a, b = (self.bases[n] for n in self.pair)
            self.mid = (a + b) / 2
            self.axis = (a - b) / np.linalg.norm(a - b)       # points towards the first arm

    def of(self, xy) -> str:
        """'middle', or the name of the arm whose side the point is on."""
        if not self.pair:
            return next(iter(self.bases))
        s = float(np.dot(np.asarray(xy[:2], float) - self.mid, self.axis))
        if abs(s) < self.half:
            return "middle"
        return self.pair[0] if s > 0 else self.pair[1]

    def needed(self, arm: str, points) -> frozenset:
        return frozenset({arm, *(self.of(p) for p in points)})


def reach(xy, base) -> float:
    return float(np.linalg.norm(np.asarray(xy[:2], float) - np.asarray(base, float)))


BOX_SLACK = {"value": 0.0}   # how much nearer than its centre a box's drop point may be (set by the caller) [m]


def drop_point(box_xy, base, slack=None):
    """Where to release over a box for the arm at `base`: the box's centre moved towards the arm by `slack` (a
    shape may land anywhere inside the box, and its near side is nearer than its centre)."""
    slack = BOX_SLACK["value"] if slack is None else slack
    box_xy, base = np.asarray(box_xy[:2], float), np.asarray(base, float)
    d = np.linalg.norm(box_xy - base)
    return box_xy if d <= 1e-9 else box_xy + (base - box_xy) / d * min(slack, d)


def box_reach(box, base) -> float:
    """Distance from the arm's base to its drop point in the box."""
    return reach(drop_point(box["xyz"], base), base)


def why_skipped(shape, box, bases, handoff_points, max_reach):
    """The plain reason no arm (directly or by a handoff) can take this shape to its box."""
    cm = lambda d: f"{100 * d:.0f} cm"
    to_shape = {a: reach(shape["xyz"], b) for a, b in bases.items()}
    to_box = {a: box_reach(box, b) for a, b in bases.items()}
    near_shape, near_box = min(to_shape, key=to_shape.get), min(to_box, key=to_box.get)
    limit = f"(max {cm(max_reach)}, --max-reach)" + (f"; only {near_shape} is connected" if len(bases) == 1 else "")
    if to_shape[near_shape] > max_reach:
        return f"no arm reaches the shape: {near_shape} is {cm(to_shape[near_shape])} from it {limit}"
    if to_box[near_box] > max_reach:
        return (f"no arm reaches its {box['colour']} box: the nearest drop point is {cm(to_box[near_box])} from "
                f"{near_box} {limit}")
    if len(bases) == 1:
        return (f"only {near_shape} is connected: it reaches the shape but not its {box['colour']} box "
                f"({cm(to_box[near_shape])}); a handoff needs the other arm")
    return (f"{near_shape} reaches the shape and {near_box} its {box['colour']} box, but no handoff point is "
            f"one both arms can grasp at (within {cm(max_reach)} and an IK check; arms.json \"handoff\" points)")


def handoff_area(bases, max_reach, min_reach, regions, step=0.04, margin=0.03, x_range=(0.0, 0.70)):
    """Handoff (drop-off) points spread over the area both arms reach: a grid, every `step` m, of the points at
    least min_reach + margin and at most max_reach - margin from every base, in the middle strip between the arms
    (TABLE frame). The arms' coordinator uses any of them, so a handoff is not tied to one or two fixed points."""
    if len(bases) < 2:
        return []
    ys = [float(b[1]) for b in bases.values()]
    points = []
    for x in np.arange(x_range[0], x_range[1] + 1e-9, step):
        for y in np.arange(min(ys), max(ys) + 1e-9, step / 2):
            p = np.array([x, y])
            d = [np.linalg.norm(p - np.asarray(b, float)) for b in bases.values()]
            if min(d) >= min_reach + margin and max(d) <= max_reach - margin and regions.of(p) == "middle":
                points.append([round(float(x), 3), round(float(y), 3)])
    return points


def build_tasks(shapes, boxes, bases, handoff_points, max_reach, regions, can_grasp=None, max_usable=None):
    """Tasks for the shapes (todo marks), and [(shape, reason)] for the ones no arm or handoff can do.
    can_grasp(arm, point index, top z): whether that arm really reaches a shape of that height on that handoff
    point (an IK check); without it only the distance is checked. Handoff points are tried shortest detour first
    (shape -> point -> box); max_usable: stop after this many usable ones (an area has many points, each IK-checked)."""
    ids = itertools.count(1)
    tasks, skipped = [], []
    for shape in shapes:
        matching = [m for m in boxes if m["colour"] == shape["colour"]]
        if not matching:
            skipped.append((shape, f"no {shape['colour']} box marked"))
            continue
        box = matching[-1]
        by_distance = sorted(bases, key=lambda arm: reach(shape["xyz"], bases[arm]))
        direct = [arm for arm in by_distance
                  if reach(shape["xyz"], bases[arm]) <= max_reach and box_reach(box, bases[arm]) <= max_reach]
        if direct:
            arm = direct[0]
            tasks.append(Task(next(ids), DIRECT, arm, shape, box,
                              regions=regions.needed(arm, [shape["xyz"], box["xyz"]])))
            continue
        plan = None
        for first in [a for a in by_distance if reach(shape["xyz"], bases[a]) <= max_reach]:
            for second in [a for a in bases if a != first and box_reach(box, bases[a]) <= max_reach]:
                top = float(shape["xyz"][2])
                order = sorted(range(len(handoff_points)),
                               key=lambda i: reach(shape["xyz"], handoff_points[i][:2])
                               + reach(box["xyz"], handoff_points[i][:2]))
                usable = []
                for i in order:
                    p = handoff_points[i]
                    if (reach(p, bases[first]) <= max_reach and reach(p, bases[second]) <= max_reach
                            and (can_grasp is None or (can_grasp(first, i, top) and can_grasp(second, i, top)))):
                        usable.append(i)
                        if max_usable is not None and len(usable) >= max_usable:
                            break
                if usable:
                    plan = (first, second, usable)
                    break
            if plan:
                break
        if plan is None:
            skipped.append((shape, why_skipped(shape, box, bases, handoff_points, max_reach)))
            continue
        first, second, usable = plan
        task = Task(next(ids), TO_HANDOFF, first, shape, box, then_arm=second,
                    regions=regions.needed(first, [shape["xyz"], *(handoff_points[i] for i in usable)]))
        task.note = ",".join(map(str, usable))     # the handoff points both arms reach
        tasks.append(task)
    return tasks, skipped


class Coordinator:
    """Gives tasks to the arms' workers and keeps the regions and handoff points exclusive."""

    def __init__(self, tasks, regions: Regions, handoff_points, parallel: bool, min_spacing: float = 0.10):
        self.tasks = list(tasks)
        self.regions = regions
        self.handoff_points = [np.asarray(p, float) for p in handoff_points]
        self.min_spacing = float(min_spacing)        # a shape on a point blocks the gripper at points this close
        self.parallel = parallel
        self.held: dict[str, str] = {}               # region -> arm
        self.slots: dict[int, int] = {}              # handoff point -> task id holding it (shape on it or coming)
        self.running: set[str] = set()               # arms with a task
        self.stop = threading.Event()
        self.cond = threading.Condition()
        self.ids = itertools.count(max([t.id for t in self.tasks], default=0) + 1)
        self.arm_state: dict[str, str] = {}
        self.dead: set[str] = set()                  # arms whose worker stopped after an error

    # ---- state shown in the window
    def set_state(self, arm, text):
        with self.cond:
            self.arm_state[arm] = text

    def busy(self) -> bool:
        with self.cond:
            return any(t.state in ("waiting", "running") for t in self.tasks)

    # ---- workers
    def _free_slot(self, task) -> int | None:
        candidates = [int(i) for i in task.note.split(",") if i != ""] if task.note else range(len(self.handoff_points))
        for i in candidates:
            if i in self.slots:
                continue
            if any(np.linalg.norm(self.handoff_points[i] - self.handoff_points[j]) < self.min_spacing
                   for j in self.slots):
                continue                                  # a shape lies (or will lie) right next to it
            return i
        return None

    def _blocker(self, task) -> str | None:
        """Why the task cannot start now (None: it can)."""
        if task.depends_on is not None:
            parent = next(t for t in self.tasks if t.id == task.depends_on)
            if parent.state != "done":
                return f"for #{parent.id} ({parent.arm}) to put the shape on the handoff point"
        others = sorted(self.running - self.dead - {task.arm})
        if not self.parallel and others:
            return f"its turn ({others[0]} is working; one task at a time)"
        for region in sorted(task.regions):
            holder = self.held.get(region, task.arm)
            if holder != task.arm:
                return f"the {region.replace('follower_', '')} region ({holder} is working there)"
        if task.kind == TO_HANDOFF and self._free_slot(task) is None:
            return "a free handoff point"
        return None

    def _fail(self, task, why):
        task.state = "failed"
        task.note = why
        origin = task.shape.get("origin", task.shape)
        origin["state"] = "failed"
        if task.slot is not None and self.slots.get(task.slot) == task.id:
            self.slots.pop(task.slot)
        print(f"  {task.label()}: given up, {why}")

    def abort(self, arm, task=None):
        """The arm's worker stopped after an error: its tasks fail; it keeps the regions it holds (the arm may
        still be out in the field), so the other arm's tasks that need those regions fail too instead of waiting."""
        with self.cond:
            self.dead.add(arm)
            if task is not None and task.state == "running":
                self._fail(task, f"{arm} stopped after an error")
            for t in self.tasks:
                if t.state == "waiting" and (t.arm == arm or (t.kind == TO_HANDOFF and t.then_arm == arm)):
                    self._fail(t, f"{arm} stopped after an error")
            self.cond.notify_all()

    def next_task(self, arm) -> Task | None:
        """Block until a task for this arm can start (then reserve its regions), or None when nothing is left."""
        with self.cond:
            while not self.stop.is_set():
                mine = [t for t in self.tasks if t.arm == arm and t.state == "waiting"]
                coming = any(t.kind == TO_HANDOFF and t.then_arm == arm and t.state in ("waiting", "running")
                             for t in self.tasks)            # a handoff whose second half will be this arm's
                if not mine and not coming:
                    return None
                reasons = []
                for task in mine:
                    blocker = self._blocker(task)
                    dead_region = [r for r in task.regions if self.held.get(r) in self.dead and self.held.get(r) != arm]
                    if dead_region:
                        self._fail(task, f"it needs the region {self.held[dead_region[0]]} stopped in")
                        continue
                    if blocker is not None:
                        reasons.append(f"#{task.id} waits for {blocker}")
                    else:
                        if task.kind == TO_HANDOFF:
                            task.slot = self._free_slot(task)
                            self.slots[task.slot] = task.id
                            # the handoff point decides the regions (it was a guess over all usable points)
                            task.regions = self.regions.needed(arm, [task.shape["xyz"], self.handoff_points[task.slot]])
                            if any(self.held.get(r, arm) != arm for r in task.regions):
                                self.slots.pop(task.slot)
                                task.slot = None
                                continue
                        for region in task.regions:
                            self.held[region] = arm
                        task.state = "running"
                        self.running.add(arm)
                        return task
                if not mine and coming:
                    reasons.append("waiting for the other arm's handoff")
                self.arm_state[arm] = reasons[0] if reasons else "waiting"
                self.cond.wait(timeout=0.5)
            return None

    def finish(self, task, result: str, handoff_xyz=None):
        """result: 'done' or 'failed'; frees the task's regions (the arm is home again)."""
        with self.cond:
            task.state = result
            for region in task.regions:
                if self.held.get(region) == task.arm:
                    self.held.pop(region)
            self.running.discard(task.arm)
            if task.kind == FROM_HANDOFF and task.slot is not None:
                if result == "done":
                    self.slots.pop(task.slot, None)       # picked up: the point is free again
                else:
                    self.slots[task.slot] = -1            # the shape is still lying there: keep the point blocked
            if task.kind == TO_HANDOFF:
                if result == "done":
                    moved = dict(task.shape, xyz=np.asarray(handoff_xyz, float), state="todo", handoff=task.slot,
                                 origin=task.shape.get("origin", task.shape))
                    follow = Task(next(self.ids), FROM_HANDOFF, task.then_arm, moved, task.box, slot=task.slot,
                                  depends_on=task.id,
                                  regions=self.regions.needed(task.then_arm, [handoff_xyz, task.box["xyz"]]))
                    self.slots[task.slot] = follow.id
                    self.tasks.append(follow)
                else:
                    self.slots.pop(task.slot, None)
            self.cond.notify_all()

    def requeue(self, task):
        """Try a task again later (a missed grasp)."""
        with self.cond:
            task.state = "waiting"
            task.attempts += 1
            for region in task.regions:
                if self.held.get(region) == task.arm:
                    self.held.pop(region)
            self.running.discard(task.arm)
            if task.kind == TO_HANDOFF and task.slot is not None:
                self.slots.pop(task.slot, None)
                task.slot = None
            self.cond.notify_all()
