"""A planner per arm, so planning runs in parallel with the other arm's planning and with the arm's own moves.

    planner = ArmPlanner("follower_pink", args)            # mode "process" (default) or "thread"
    future = planner.submit(task, start_q, obstacles, args) # returns at once
    plans, grasp_z, log = future.result()                   # (raises RuntimeError: no plan)
    planner.close()

mode "process": each arm plans in a process of its own (its own PyBullet and its own reused planning world,
planning/pick_planner.py planning_world), so both arms plan at the same time and planning does not hold up the
moving threads. mode "thread": in this process, one plan at a time for both arms (PLAN_LOCK), as before.

The planner's printout (planning 'grasp' ..., planned grasp ...) is returned as `log` instead of printed, so a plan
made ahead of time (state_machine/arm_worker.py prefetch) only prints when it is used.
"""
from __future__ import annotations

import contextlib
import io
import multiprocessing
import sys
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor

MODES = ("process", "thread")


class PlanError(RuntimeError):
    """No plan: the reason, and the planner's printout up to then (log)."""

    def __init__(self, message, log=""):
        super().__init__(message)
        self.log = log

    def __reduce__(self):                      # so it crosses from the planner's process with its log
        return PlanError, (str(self), self.log)


def _init_child(paths, allow_urdf_limits):
    for path in reversed(paths):
        if path not in sys.path:
            sys.path.insert(0, path)
    import planning.ik

    planning.ik.ALLOW_URDF_LIMITS = allow_urdf_limits


def _plan(task, start_q, obstacles, args, handoff_xyz, capture):
    from planning.pick_planner import PLAN_LOCK, plan_task

    buffer = io.StringIO()
    with PLAN_LOCK, (contextlib.redirect_stdout(buffer) if capture else contextlib.nullcontext()):
        try:
            plans, z = plan_task(task, start_q, obstacles, args, handoff_xyz)
        except RuntimeError as error:
            raise PlanError(str(error), buffer.getvalue()) from None
    plans = [(label, [list(map(float, q)) for q in path]) for label, path in plans]
    return plans, float(z), buffer.getvalue()


def _warm(arm, start_q, args):
    """Build the arm's planning world now (~1 s), before the first real plan needs it."""
    from planning.pick_planner import PLAN_LOCK, planning_world

    with PLAN_LOCK:
        planning_world(arm, start_q, args)
    return True


class ArmPlanner:
    def __init__(self, arm, args, mode="process", start_q=None):
        if mode not in MODES:
            raise ValueError(f"planner mode {mode!r}: one of {MODES}")
        self.arm, self.mode = arm, mode
        if mode == "process":
            import planning.ik

            self.pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"), initializer=_init_child,
                                            initargs=(list(sys.path), planning.ik.ALLOW_URDF_LIMITS))
        else:
            self.pool = ThreadPoolExecutor(1, thread_name_prefix=f"plan-{arm}")
        if start_q is not None and not getattr(args, "gui", False):
            self.pool.submit(_warm, arm, list(start_q), args)

    def submit(self, task, start_q, obstacles, args, handoff_xyz=None) -> Future:
        """Plan the task (planning/pick_planner.py plan_task) without waiting: a Future of (plans, grasp z, log)."""
        capture = self.mode == "process"       # threads share stdout: capturing one would swallow the others' output
        return self.pool.submit(_plan, task, list(map(float, start_q)), obstacles, args, handoff_xyz, capture)

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
