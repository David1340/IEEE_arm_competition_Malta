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
    tree.structure()                 # the tree as nested dicts (path, name, type, children), for the live view

blackboard["observer"] (optional): observer(node, status) is called when any node starts ("running") and ends
("success" / "failure"); node.path names it in structure() (the tree view, monitor/tree_view.py, follows it).

Used by the arms' workers (state_machine/arm_worker.py task_tree).
"""
from __future__ import annotations


class Node:
    children: list = []

    def __init__(self, name: str):
        self.name = name
        self.path = name

    def tick(self, bb: dict) -> bool:
        observer = bb.get("observer")
        if observer is not None:
            observer(self, "running")
        ok = bool(self._tick(bb))
        if observer is not None:
            observer(self, "success" if ok else "failure")
        return ok

    def _tick(self, bb: dict) -> bool:
        raise NotImplementedError

    def structure(self, path: str = "0") -> dict:
        """The tree as nested dicts {path, name, type, children}; sets each node's path ("0", "0/3", "0/3/1", ...:
        the child indices from the root, unique even where two nodes share a name)."""
        self.path = path
        return {"path": path, "name": self.name, "type": type(self).__name__,
                "children": [c.structure(f"{path}/{i}") for i, c in enumerate(self.children)]}

    def describe(self, depth: int = 0) -> str:
        return "  " * depth + f"{type(self).__name__}: {self.name}"


class Action(Node):
    """A leaf: fn(bb) -> bool. on_start(name) is called before it runs (e.g. to show the arm's state)."""

    def __init__(self, name: str, fn, on_start=None):
        super().__init__(name)
        self.fn, self.on_start = fn, on_start

    def _tick(self, bb):
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

    def _tick(self, bb):
        bb["why"] = self.why
        bb.setdefault("trace", []).append((self.name, False))
        return False


class Sequence(Node):
    """Runs its children in order; fails at the first child that fails, succeeds when all do."""

    def __init__(self, name: str, children):
        super().__init__(name)
        self.children = [c for c in children if c is not None]

    def _tick(self, bb):
        return all(child.tick(bb) for child in self.children)

    def describe(self, depth=0):
        return "\n".join([super().describe(depth)] + [c.describe(depth + 1) for c in self.children])


class Fallback(Node):
    """Tries its children in order; succeeds at the first child that succeeds, fails when all do."""

    def __init__(self, name: str, children):
        super().__init__(name)
        self.children = [c for c in children if c is not None]

    def _tick(self, bb):
        return any(child.tick(bb) for child in self.children)

    def describe(self, depth=0):
        return "\n".join([super().describe(depth)] + [c.describe(depth + 1) for c in self.children])
