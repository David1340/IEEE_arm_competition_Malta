"""A small behaviour tree: Sequence, Fallback and Action nodes, run (ticked) to the end in one call.

Each node returns True (success) or False (failure); the actions block until their move is done, so there is no
RUNNING state. A shared dict (the blackboard) carries the data between the actions, and a failing action writes why
it failed into blackboard["why"] for the caller.

    tree = Sequence("attempt", [
        Action("locate object", locate),
        Action("plan", plan),
        Fallback("grasp", [Action("close on it", grasp),
                           Sequence("missed", [Action("back off", back_off), Action("home", home), Fail("missed")])]),
        ...
    ])
    ok = tree.tick(blackboard)       # blackboard["trace"]: the actions that ran, with their results
    print(tree.describe())           # the tree, one node per line

Used by the arms' workers (experiments/click_pick_place.py arm_worker, which pick_and_place.py runs too).
"""
from __future__ import annotations


class Node:
    def __init__(self, name: str):
        self.name = name

    def tick(self, bb: dict) -> bool:
        raise NotImplementedError

    def describe(self, depth: int = 0) -> str:
        return "  " * depth + f"{type(self).__name__}: {self.name}"


class Action(Node):
    """A leaf: fn(bb) -> bool. on_start(name) is called before it runs (e.g. to show the arm's state)."""

    def __init__(self, name: str, fn, on_start=None):
        super().__init__(name)
        self.fn, self.on_start = fn, on_start

    def tick(self, bb):
        if self.on_start is not None:
            self.on_start(self.name)
        ok = bool(self.fn(bb))
        bb.setdefault("trace", []).append((self.name, ok))
        return ok


class Fail(Node):
    """Always fails, recording why (e.g. at the end of a fallback branch that only cleans up)."""

    def __init__(self, why: str):
        super().__init__(f"fail: {why}")
        self.why = why

    def tick(self, bb):
        bb["why"] = self.why
        bb.setdefault("trace", []).append((self.name, False))
        return False


class Sequence(Node):
    """Runs its children in order; fails at the first child that fails, succeeds when all do."""

    def __init__(self, name: str, children):
        super().__init__(name)
        self.children = [c for c in children if c is not None]

    def tick(self, bb):
        return all(child.tick(bb) for child in self.children)

    def describe(self, depth=0):
        return "\n".join([super().describe(depth)] + [c.describe(depth + 1) for c in self.children])


class Fallback(Node):
    """Tries its children in order; succeeds at the first child that succeeds, fails when all do."""

    def __init__(self, name: str, children):
        super().__init__(name)
        self.children = [c for c in children if c is not None]

    def tick(self, bb):
        return any(child.tick(bb) for child in self.children)

    def describe(self, depth=0):
        return "\n".join([super().describe(depth)] + [c.describe(depth + 1) for c in self.children])
