"""The PyBullet window of a real round: the table, both arms as they really are (their measured joints), every object
and bin the camera sees, and the paths: what each arm planned, the move it is on, and where its gripper really went.

    view = PyBulletView(["follower_pink", "follower_blue"], args)   # its own process; subscribes to core/telemetry.py
    view.scene(objects, bins, marks)                                # from the main loop (at most 4 times a second)
    view.close()

In the window:
  arms           solid, in their colours: where the motors really are (the joints measured while moving);
                 a stick figure of each arm (pale lines through its joints): the pose its current move ends at (after
                 a plan: the planned grasp pose)
  objects, bins  what the camera sees now (perception), at their TABLE positions; the round's objects carry a label
                 with their state (todo / placed / failed / handoff) at the place the arm planned for
  thin lines     the latest plan of each arm: where the grasp point goes along each step (colour per step), a cross
                 on the object it goes for and a line to its bin (or drop-off point)
  thick line     the move running now
  dark trail     where the grasp point really went (forward kinematics of the measured joints), in this attempt
                 and the one before
  text           over each arm: the action it is doing; red after a failure, with why
Mouse: drag to turn, wheel to zoom, Ctrl + drag to move (PyBullet's own).

The window runs in a process of its own, fed by a queue: the camera loop and the arms never wait for it, and closing
the window stops only the view. Planning and the arms work exactly as without it.
"""
from __future__ import annotations

import math
import multiprocessing
import queue
import sys
import time

import numpy as np

from core import telemetry

STEP_RGB = {"above shape": (0.25, 0.55, 0.95), "grasp": (0.9, 0.2, 0.2), "lift": (0.95, 0.6, 0.1),
            "over box": (0.1, 0.65, 0.3), "over handoff": (0.1, 0.65, 0.3), "place": (0.55, 0.25, 0.75),
            "leave handoff": (0.4, 0.4, 0.4), "home": (0.45, 0.45, 0.45), "back off": (0.8, 0.1, 0.5),
            "release": (0.1, 0.65, 0.3)}
COLOUR_RGB = {"red": (0.85, 0.15, 0.12), "green": (0.15, 0.65, 0.2), "blue": (0.12, 0.35, 0.85),
              "yellow": (0.95, 0.8, 0.1), "orange": (0.95, 0.5, 0.1), "purple": (0.55, 0.25, 0.7),
              "pink": (1.0, 0.5, 0.7), "black": (0.15, 0.15, 0.15), "white": (0.95, 0.95, 0.95)}
STATE_RGB = {"todo": (0.2, 0.2, 0.2), "placed": (0.1, 0.6, 0.2), "failed": (0.85, 0.1, 0.1),
             "handoff": (0.0, 0.55, 0.65)}
ARM_RGB = {"follower_pink": (1.0, 0.55, 0.75), "follower_blue": (0.45, 0.7, 1.0)}
SCENE_EVERY = 0.25            # s between scene updates sent from the main loop
TRAIL_POINTS = 1500           # segments of an attempt's trail kept
FORWARD = ("tree", "node", "outcome", "worker", "plan", "move", "moved", "joints")


class PyBulletView:
    """The parent's side: starts the window's process and sends it the telemetry and the scene."""

    def __init__(self, arms, args, gui=True):
        ctx = multiprocessing.get_context("spawn")
        self.queue = ctx.Queue(maxsize=4000)
        self.queue.cancel_join_thread()                    # never wait at exit for messages the view will not read
        settings = {"arms": list(arms), "box_size": float(getattr(args, "box_size", 0.15)),
                    "box_height": float(getattr(args, "box_height", 0.08)),
                    "max_reach": float(getattr(args, "max_reach", 0.45)),
                    "min_reach": float(getattr(args, "min_reach", 0.0) or 0.0), "gui": gui}
        self.process = ctx.Process(target=_run, args=(self.queue, list(sys.path), settings), daemon=True,
                                   name="pybullet view")
        self.process.start()
        self.last_scene = 0.0
        self.warned = False
        telemetry.subscribe(self.on_event)

    def alive(self):
        if self.process.is_alive():
            return True
        if not self.warned:
            self.warned = True
            print("(the PyBullet view has closed; the round carries on without it)")
            telemetry.unsubscribe(self.on_event)
        return False

    def _send(self, message, wait=0.0):
        if not self.alive():
            return
        try:
            if wait:
                self.queue.put(message, timeout=wait)
            else:
                self.queue.put_nowait(message)
        except queue.Full:
            pass                                           # the view is behind: drop it, never hold up an arm

    def on_event(self, e):
        if e["kind"] in FORWARD:
            # joints come often and the next one replaces it; the rest must arrive
            self._send(e, wait=0.0 if e["kind"] == "joints" else 0.2)

    def scene(self, objects, bins, marks=(), force=False):
        """The camera's objects and bins (games/pick_and_place.py detect) and the round's marks, sent at most every
        SCENE_EVERY s."""
        now = time.monotonic()
        if not force and now - self.last_scene < SCENE_EVERY:
            return
        self.last_scene = now
        from perception.detection import OBJECT_SIZES, OBJECT_STL_FILES

        def stl(o):
            path = OBJECT_SIZES.get(o["name"], {}).get(o.get("size") or "") or OBJECT_STL_FILES.get(o["name"])
            return None if path is None else str(path)

        message = {"kind": "scene",
                   "objects": [{"stl": stl(o), "colour": o.get("colour"), "xyz": [float(v) for v in o["xyz"]],
                                "yaw": o.get("yaw")} for o in objects if stl(o)],
                   "bins": [{"colour": b.get("colour"), "xyz": [float(v) for v in b["xyz"]]} for b in bins],
                   "marks": [{"label": m.get("label", m.get("name", "")), "state": m["state"],
                              "xyz": [float(v) for v in (m.get("handoff_xyz") if m["state"] == "handoff"
                                                         and m.get("handoff_xyz") is not None else m["xyz"])]}
                             for m in marks if m.get("kind") == "shape"]}
        self._send(message)

    def close(self):
        telemetry.unsubscribe(self.on_event)
        if self.process.is_alive():
            try:
                self.queue.put(None, timeout=0.5)
            except queue.Full:
                pass
            self.process.join(2.0)
            if self.process.is_alive():
                self.process.terminate()


# ================================================================================================ the window process
def _run(q, paths, settings):
    for path in reversed(paths):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        window = _Window(settings)
    except Exception as error:                              # noqa: BLE001 - no display, no pybullet, ...
        print(f"(PyBullet view could not open: {error})")
        return
    window.loop(q)


class _Window:
    def __init__(self, settings):
        import pybullet as p

        from core import arms as arm_config
        from core.paths import URDF

        self.p, self.settings = p, settings
        self.c = c = p.connect(p.GUI if settings["gui"] else p.DIRECT)
        if c < 0:
            raise RuntimeError("no PyBullet GUI")
        self.fk = p.connect(p.DIRECT)                       # forward kinematics of plans, out of sight
        if settings["gui"]:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=c)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=c)
            p.resetDebugVisualizerCamera(1.0, -90.0, -50.0, [0.32, 0.0, 0.0], physicsClientId=c)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=c)   # build it, then draw once
        half = [0.55, 0.85, 0.01]
        p.createMultiBody(0, -1, p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.86, 0.86, 0.84, 1],
                                                     physicsClientId=c),
                          basePosition=[0.30, 0.0, -half[2]], physicsClientId=c)
        # only the two arms are loaded in the window: PyBullet's window has room for about three arm meshes, and
        # anything loaded after that is not drawn; the target poses are stick figures of lines instead
        self.body, self.fk_body, self.joint, self.link, self.grip_range = {}, {}, {}, {}, {}
        used = set(settings["arms"])
        for name in arm_config.followers():
            b = arm_config.base(name)
            pose = dict(basePosition=[b["x"], b["y"], 0.0],
                        baseOrientation=p.getQuaternionFromEuler([0, 0, math.radians(b["yaw_deg"])]),
                        useFixedBase=True)
            body = p.loadURDF(str(URDF), physicsClientId=c, **pose)
            infos = [p.getJointInfo(body, i, physicsClientId=c) for i in range(p.getNumJoints(body, physicsClientId=c))]
            self.joint[name] = {i[1].decode(): i[0] for i in infos}
            self.link[name] = {i[12].decode(): i[0] for i in infos}
            self.grip_range[name] = infos[self.joint[name]["gripper"]][8:10]
            rgb = ARM_RGB.get(name, (0.8, 0.8, 0.8)) if name in used else (0.7, 0.7, 0.7)
            self._tint(body, [*rgb, 1.0 if name in used else 0.5])
            self.body[name] = body
            if name in used:
                self.fk_body[name] = p.loadURDF(str(URDF), physicsClientId=self.fk, **pose)
        from planning.pick_planner import HOME_Q

        for name in self.body:
            self._pose(self.body[name], name, HOME_Q, arm_config.HOME_GRIPPER)
        self.plan_lines = {n: [] for n in self.body}        # debug item ids
        self.active_lines = {n: [] for n in self.body}
        self.ghost_lines = {n: [] for n in self.body}
        self.pending_say = {}
        self.trail = {n: [[]] for n in self.body}           # per attempt: the previous one and this one
        self.last_tip = {n: None for n in self.body}
        self.label = {}
        self.objects = {}                                   # (stl, colour) -> [body ids]
        self.bins = []
        self.mark_text = []
        self.shapes = {}                                    # visual shape cache
        self._floor(arm_config)
        self._legend()
        for name in settings["arms"]:
            self._say(name, "starting", (0.3, 0.3, 0.3))
        self.flush()
        if settings["gui"]:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=c)

    # ------------------------------------------------------------------------------------------ building
    def _tint(self, body, rgba):
        p = self.p
        for link in range(-1, p.getNumJoints(body, physicsClientId=self.c)):
            p.changeVisualShape(body, link, rgbaColor=rgba, physicsClientId=self.c)

    def _floor(self, arm_config):
        p, c = self.p, self.c
        line = lambda a, b, rgb, w=1.5: p.addUserDebugLine([*a, 0.0008], [*b, 0.0008], rgb, w, physicsClientId=c)
        angles = np.linspace(0, 2 * np.pi, 73)
        for name in self.settings["arms"]:
            b = arm_config.base(name)
            for radius in (self.settings["max_reach"], self.settings["min_reach"]):
                if radius <= 0:
                    continue
                ring = np.column_stack([b["x"] + radius * np.cos(angles), b["y"] + radius * np.sin(angles)])
                for a, e in zip(ring[:-1], ring[1:]):
                    line(a, e, ARM_RGB.get(name, (0.5, 0.5, 0.5)), 1.0)
        if len(self.settings["arms"]) > 1:
            half = arm_config.middle_half_width()
            for y in (-half, half):
                line([0.0, y], [0.75, y], [0.6, 0.6, 0.0], 1.0)

    def _legend(self):
        p = self.p
        lines = [("thin: planned path (colour per step)", (0.2, 0.2, 0.2)),
                 ("thick: the move running now", (0.2, 0.2, 0.2)),
                 ("dark trail: where the gripper really went", (0.1, 0.1, 0.1)),
                 ("stick figure: the pose the move ends at", (0.2, 0.2, 0.2))]
        for i, (s, rgb) in enumerate(lines):
            p.addUserDebugText(s, [0.95, 0.55, 0.40 - 0.05 * i], rgb, 1.1, physicsClientId=self.c)

    def _pose(self, body, arm, q, grip=None, client=None):
        from planning.sim import JOINTS

        p, c = self.p, self.c if client is None else client
        joint = self.joint[arm]
        for name, value in zip(JOINTS, q):
            p.resetJointState(body, joint[name], float(value), physicsClientId=c)
        if grip is not None:
            low, high = self.grip_range[arm]
            p.resetJointState(body, joint["gripper"], low + float(grip) / 100.0 * (high - low), physicsClientId=c)

    def _tip(self, arm, q):
        """The grasp point (TABLE frame) for joints q, from the hidden model."""
        p = self.p
        body = self.fk_body[arm]
        self._pose(body, arm, q, client=self.fk)
        return np.array(p.getLinkState(body, self.link[arm]["grasp_center_link"], computeForwardKinematics=True,
                                       physicsClientId=self.fk)[4])

    def _ghost(self, arm, q):
        """The arm at joints q as a stick figure: lines through its joints, from the hidden model."""
        p = self.p
        self._clear(self.ghost_lines[arm])
        body = self.fk_body[arm]
        self._pose(body, arm, q, client=self.fk)
        points = [np.array(p.getBasePositionAndOrientation(body, physicsClientId=self.fk)[0])]
        for name in ("shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link", "gripper_link",
                     "grasp_center_link"):
            if name in self.link[arm]:
                points.append(np.array(p.getLinkState(body, self.link[arm][name], computeForwardKinematics=True,
                                                      physicsClientId=self.fk)[4]))
        rgb = tuple(0.55 * c for c in ARM_RGB.get(arm, (0.6, 0.6, 0.6)))
        self.ghost_lines[arm] = self._polyline(points, rgb, 4.0)
        tip, s = points[-1], 0.012
        for d in np.eye(3) * s:
            self.ghost_lines[arm].append(p.addUserDebugLine((tip - d).tolist(), (tip + d).tolist(), rgb, 4.0,
                                                            physicsClientId=self.c))

    def _curve(self, arm, path, step_deg=3.0, most=80):
        """The grasp point along a joint path (interpolated every step_deg of the largest joint change)."""
        qs = [np.asarray(q, float) for q in path]
        dense = [qs[0]]
        for a, b in zip(qs[:-1], qs[1:]):
            n = max(1, int(math.ceil(math.degrees(np.abs(b - a).max()) / step_deg)))
            dense += [a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:]]
        if len(dense) > most:
            dense = [dense[int(i)] for i in np.linspace(0, len(dense) - 1, most)]
        return [self._tip(arm, q) for q in dense]

    def _polyline(self, points, rgb, width):
        p = self.p
        return [p.addUserDebugLine(a.tolist(), b.tolist(), rgb, width, physicsClientId=self.c)
                for a, b in zip(points[:-1], points[1:]) if np.linalg.norm(b - a) > 1e-4]   # PyBullet refuses 0

    def _clear(self, items):
        for item in items:
            self.p.removeUserDebugItem(item, physicsClientId=self.c)
        items.clear()

    def _say(self, arm, s, rgb):
        self.pending_say[arm] = (s, rgb)                    # drawn once per batch (flush): only the latest counts

    def flush(self):
        from core import arms as arm_config

        for arm, (s, rgb) in self.pending_say.items():
            b = arm_config.base(arm)
            old = self.label.get(arm, -1)
            self.label[arm] = self.p.addUserDebugText(f"{arm.replace('follower_', '')}: {s}",
                                                      [b["x"], b["y"], 0.50], rgb, 1.3, replaceItemUniqueId=old,
                                                      physicsClientId=self.c)
        self.pending_say.clear()

    def _dot(self):
        """A 1 mm collision sphere: PyBullet only moves a body's picture (resetBasePositionAndOrientation) when the
        body has a collision shape. Nothing here is simulated, so it never touches anything."""
        p = self.p
        return self._visual(("dot",), lambda: p.createCollisionShape(p.GEOM_SPHERE, radius=0.001,
                                                                      physicsClientId=self.c))

    def _visual(self, key, make):
        if key not in self.shapes:
            self.shapes[key] = make()
        return self.shapes[key]

    # ------------------------------------------------------------------------------------------ messages
    def _trail(self, arm, q):
        """Where the grasp point really went: a dark line from the last reading, when it moved more than 3 mm."""
        tip = self._tip(arm, q)
        last = self.last_tip[arm]
        if last is not None and np.linalg.norm(tip - last) <= 0.003:
            return
        if last is not None:
            now = self.trail[arm][-1]
            now += self._polyline([last, tip], (0.12, 0.12, 0.12), 2.0)
            while len(now) > TRAIL_POINTS:
                self.p.removeUserDebugItem(now.pop(0), physicsClientId=self.c)
        self.last_tip[arm] = tip

    def handle(self, m):
        kind, arm = m["kind"], m.get("arm")
        if kind == "scene":
            self._scene(m)
            return
        if arm not in self.fk_body:
            return
        if kind == "joints":
            self._pose(self.body[arm], arm, m["q"], m.get("gripper"))
            self._trail(arm, m["q"])
        elif kind == "plan":
            self._clear(self.plan_lines[arm])
            for step, path in m["plans"]:
                if len(path) > 1:
                    self.plan_lines[arm] += self._polyline(self._curve(arm, path), STEP_RGB.get(step, (0.5, 0.5, 0.5)),
                                                           1.5)
            target = np.asarray(m["target"], float)
            target[2] = m.get("grasp_z", target[2])
            s = 0.025
            for d in ([s, 0, 0], [0, s, 0], [0, 0, s]):
                d = np.asarray(d)
                self.plan_lines[arm].append(self.p.addUserDebugLine((target - d).tolist(), (target + d).tolist(),
                                                                    (0.9, 0.1, 0.1), 3.0, physicsClientId=self.c))
            if m.get("goal") is not None:
                goal = np.asarray(m["goal"], float)
                goal[2] = self.settings["box_height"] + 0.02
                self.plan_lines[arm].append(self.p.addUserDebugLine(target.tolist(), goal.tolist(), (0.2, 0.6, 0.2),
                                                                    1.0, physicsClientId=self.c))
            grasp = dict(m["plans"]).get("grasp")
            if grasp and arm in self.fk_body:
                self._ghost(arm, grasp[-1])
        elif kind == "move":
            self._clear(self.active_lines[arm])
            path = m["path"]
            if len(path) > 1:
                self.active_lines[arm] = self._polyline(self._curve(arm, path),
                                                        STEP_RGB.get(m.get("step"), (0.3, 0.3, 0.3)), 5.0)
            if path and arm in self.fk_body:
                self._ghost(arm, path[-1])
            self._say(arm, f"moving: {m.get('step') or 'move'}", (0.0, 0.35, 0.7))
        elif kind == "tree":                                # a new attempt: keep only the last attempt's trail
            while len(self.trail[arm]) > 1:
                self._clear(self.trail[arm].pop(0))
            self.trail[arm].append([])
        elif kind == "moved":
            self._clear(self.active_lines[arm])
            if not m.get("ok", True):
                self._say(arm, f"{m.get('step')}: did not arrive", (0.8, 0.1, 0.1))
        elif kind == "node" and m.get("type", "Action") == "Action" and m["status"] == "running":
            self._say(arm, m["name"], (0.1, 0.1, 0.1))
        elif kind == "node" and m["status"] == "failure" and m.get("type") in ("Action", "Fail"):
            self._say(arm, f"FAILED {m['name']}: {m.get('why') or ''}", (0.85, 0.1, 0.1))
        elif kind == "outcome":
            rgb = (0.1, 0.55, 0.2) if m["result"] == "done" else (0.85, 0.1, 0.1) if m["result"] == "left" \
                else (0.85, 0.45, 0.0)
            self._say(arm, f"{m['result']}" + (f": {m['why']}" if m.get("why") else ""), rgb)
        elif kind == "worker" and m["state"] in ("finished", "error"):
            self._say(arm, "ERROR (see the terminal)" if m["state"] == "error" else "no more tasks",
                      (0.85, 0.1, 0.1) if m["state"] == "error" else (0.3, 0.3, 0.3))

    def _scene(self, m):
        p, c = self.p, self.c
        wanted = {}
        for o in m["objects"]:
            wanted.setdefault((o["stl"], o["colour"]), []).append(o)
        for key in set(self.objects) | set(wanted):
            bodies, items = self.objects.setdefault(key, []), wanted.get(key, [])
            stl, colour = key
            while len(bodies) < len(items):
                vis = self._visual(("mesh", stl, colour), lambda: p.createVisualShape(
                    p.GEOM_MESH, fileName=stl, meshScale=[0.001] * 3,
                    rgbaColor=[*COLOUR_RGB.get(colour, (0.6, 0.6, 0.6)), 1], physicsClientId=c))
                bodies.append(p.createMultiBody(0, self._dot(), vis, basePosition=[0, 0, -5], physicsClientId=c))
            for body, o in zip(bodies, items):
                yaw = math.radians(o["yaw"] or 0.0)
                p.resetBasePositionAndOrientation(body, [o["xyz"][0], o["xyz"][1], 0.0],
                                                  p.getQuaternionFromEuler([0, 0, yaw]), physicsClientId=c)
            for body in bodies[len(items):]:
                p.resetBasePositionAndOrientation(body, [0, 0, -5], [0, 0, 0, 1], physicsClientId=c)
        size, height = self.settings["box_size"], self.settings["box_height"]
        while len(self.bins) < len(m["bins"]):
            self.bins.append(None)
        for i, b in enumerate(m["bins"]):
            colour = b["colour"]
            if self.bins[i] is None or self.bins[i][1] != colour:
                if self.bins[i] is not None:
                    p.removeBody(self.bins[i][0], physicsClientId=c)
                self.bins[i] = (self._bin(colour, size, height), colour)
            p.resetBasePositionAndOrientation(self.bins[i][0], [b["xyz"][0], b["xyz"][1], 0.0], [0, 0, 0, 1],
                                              physicsClientId=c)
        for entry in self.bins[len(m["bins"]):]:
            if entry is not None:
                p.resetBasePositionAndOrientation(entry[0], [0, 0, -5], [0, 0, 0, 1], physicsClientId=c)
        while len(self.mark_text) < len(m["marks"]):
            self.mark_text.append(-1)
        for i in range(len(self.mark_text)):
            if i < len(m["marks"]):
                mark = m["marks"][i]
                s, xyz = f"{mark['label']}: {mark['state']}", [mark["xyz"][0], mark["xyz"][1], 0.09]
                rgb = STATE_RGB.get(mark["state"], (0.2, 0.2, 0.2))
            else:
                s, xyz, rgb = "", [0, 0, -5], (0, 0, 0)
            self.mark_text[i] = p.addUserDebugText(s, xyz, rgb, 1.0, replaceItemUniqueId=self.mark_text[i],
                                                   physicsClientId=c)

    def _bin(self, colour, size, height):
        p, c = self.p, self.c
        h, t = size / 2, 0.004
        wall_x, wall_y = [t / 2, h, height / 2], [h, t / 2, height / 2]
        parts = [([h, h, t / 2], [0, 0, t / 2]),
                 (wall_x, [h - t / 2, 0, height / 2]), (wall_x, [-h + t / 2, 0, height / 2]),
                 (wall_y, [0, h - t / 2, height / 2]), (wall_y, [0, -h + t / 2, height / 2])]
        rgba = [*COLOUR_RGB.get(colour, (0.6, 0.6, 0.6)), 0.55]
        vis = self._visual(("bin", colour, size, height), lambda: p.createVisualShapeArray(
            [p.GEOM_BOX] * 5, halfExtents=[e for e, _ in parts], visualFramePositions=[q for _, q in parts],
            rgbaColors=[rgba] * 5, physicsClientId=c))
        return p.createMultiBody(0, self._dot(), vis, basePosition=[0, 0, -5], physicsClientId=c)

    # ------------------------------------------------------------------------------------------ the loop
    def loop(self, q):
        """Take what has arrived, apply it with drawing paused (each line costs a frame otherwise), then let the
        window draw for about two frames (pausing waits for a frame, which measures how long one takes)."""
        p, gui = self.p, self.settings["gui"]
        frame = 0.02
        while p.isConnected(physicsClientId=self.c):
            try:
                batch = [q.get(timeout=0.05)]
            except queue.Empty:
                continue
            while len(batch) < 500:
                try:
                    batch.append(q.get_nowait())
                except queue.Empty:
                    break
            if gui:
                started = time.monotonic()
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.c)
                frame = 0.8 * frame + 0.2 * min(time.monotonic() - started, 0.5)
            try:
                last_joints = {}                            # the pose: only the newest reading of each arm
                for i, m in enumerate(batch):
                    if m is not None and m["kind"] == "joints":
                        last_joints[m["arm"]] = i
                for i, m in enumerate(batch):
                    if m is None:
                        p.disconnect(physicsClientId=self.c)
                        return
                    if m["kind"] == "joints" and last_joints.get(m["arm"]) != i:
                        self._trail(m["arm"], m["q"])       # older readings: only the trail
                    else:
                        self.handle(m)
                self.flush()
            finally:
                if gui and p.isConnected(physicsClientId=self.c):
                    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.c)
            time.sleep(max(0.04, 2.5 * frame))
