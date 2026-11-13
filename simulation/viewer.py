"""The simulator's PyBullet window: the table, both arms at their bases (config/arms.yaml), the objects and the bins,
all in the TABLE frame. The arms move along the planner's joint paths; a held object moves with the gripper.

Every animation step is checked for contact: an arm touching the other arm, a bin, the table or an object it is not
handling is printed as a COLLISION and counted, so a round shows whether the plans really avoid everything.

The arms' worker threads never call PyBullet themselves (one GUI, one thread): they queue commands with run() and
wait; the main thread calls step() in a loop, which moves both arms at the same time.
"""
from __future__ import annotations

import math
import queue
import threading
import time

import numpy as np
import pybullet as p

from core import arms as arm_config
from core.paths import URDF
from hardware.gripper import object_footprint
from planning.sim import JOINTS
from simulation.scene import stl

RGB = {"red": (0.85, 0.15, 0.12), "green": (0.15, 0.65, 0.2), "blue": (0.12, 0.35, 0.85),
       "yellow": (0.95, 0.8, 0.1)}
ARM_RGB = {"follower_pink": (1.0, 0.55, 0.75), "follower_blue": (0.45, 0.7, 1.0)}
STEP_DEG = 1.5               # joint interpolation step of the animation
JOINT_SPEED_DEG = 90.0       # nominal joint speed at --speed 1 [deg/s]
CONTACT_DEPTH = -0.001       # closer than this (penetrating 1 mm) counts as a collision [m]
BIN_WALL = 0.004             # bin wall / floor thickness [m]


def yaw_of(quaternion) -> float:
    return p.getEulerFromQuaternion(quaternion)[2]


class Viewer:
    def __init__(self, scene, args, gui=True, speed=1.0, realtime=False):
        self.gui, self.speed, self.args = gui, speed, args
        self.realtime = realtime        # headless, but moves take as long as on screen (timing tests)
        self.c = c = p.connect(p.GUI if gui else p.DIRECT)
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=c)
            p.resetDebugVisualizerCamera(1.0, -90.0, -50.0, [0.32, 0.0, 0.0], physicsClientId=c)   # from behind the arms
        p.setGravity(0, 0, 0, physicsClientId=c)
        half = [0.55, 0.85, 0.01]
        self.table = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=c),
                                       p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.82, 0.82, 0.8, 1],
                                                           physicsClientId=c),
                                       basePosition=[0.30, 0.0, -half[2]], physicsClientId=c)
        self.arms, self.joint, self.link, self.gripper_range = {}, {}, {}, {}
        for name in arm_config.followers():
            b = arm_config.base(name)
            body = p.loadURDF(str(URDF), basePosition=[b["x"], b["y"], 0.0],
                              baseOrientation=p.getQuaternionFromEuler([0, 0, math.radians(b["yaw_deg"])]),
                              useFixedBase=True, physicsClientId=c)
            infos = [p.getJointInfo(body, i, physicsClientId=c) for i in range(p.getNumJoints(body, physicsClientId=c))]
            self.joint[name] = {i[1].decode(): i[0] for i in infos}
            self.link[name] = {i[12].decode(): i[0] for i in infos}
            self.gripper_range[name] = infos[self.joint[name]["gripper"]][8:10]
            for link in [-1, *range(len(infos))]:
                p.changeVisualShape(body, link, rgbaColor=[*ARM_RGB.get(name, (0.8, 0.8, 0.8)), 1], physicsClientId=c)
            self.arms[name] = body
        self.objects = [self._add_object(o) for o in scene["objects"]]
        self.bins = [self._add_bin(b, args.box_size, args.box_height) for b in scene["bins"]]
        self.held = {}                  # arm -> (object body, transform object-in-gripper)
        self.handling = {}              # arm -> object body it is working on (its jaws may touch it)
        self.q = {name: [0.0] * 5 for name in self.arms}
        self.grip = {name: 50.0 for name in self.arms}
        self.collisions = []
        self.commands = {name: queue.Queue() for name in self.arms}
        self.active = {}                # arm -> (generator of steps, done event)
        self.text = {}
        self._draw_floor_marks()

    # ---------------------------------------------------------------------------------------------- building
    def _add_object(self, o):
        path = str(stl(o["shape"], o["size"]))
        col = p.createCollisionShape(p.GEOM_MESH, fileName=path, meshScale=[0.001] * 3, physicsClientId=self.c)
        vis = p.createVisualShape(p.GEOM_MESH, fileName=path, meshScale=[0.001] * 3,
                                  rgbaColor=[*RGB[o["colour"]], 1], physicsClientId=self.c)
        return p.createMultiBody(0, col, vis, basePosition=[o["x"], o["y"], 0.0],
                                 baseOrientation=p.getQuaternionFromEuler([0, 0, math.radians(o["yaw_deg"])]),
                                 physicsClientId=self.c)

    def _add_bin(self, b, size, height):
        h, t = size / 2, BIN_WALL
        parts = [([h, h, t / 2], [0, 0, t / 2]),                                  # floor
                 ([t / 2, h, height / 2], [h - t / 2, 0, height / 2]), ([t / 2, h, height / 2], [-h + t / 2, 0, height / 2]),
                 ([h, t / 2, height / 2], [0, h - t / 2, height / 2]), ([h, t / 2, height / 2], [0, -h + t / 2, height / 2])]
        rgba = [*RGB[b["colour"]], 0.55]
        col = p.createCollisionShapeArray([p.GEOM_BOX] * 5, halfExtents=[e for e, _ in parts],
                                          collisionFramePositions=[q for _, q in parts], physicsClientId=self.c)
        vis = p.createVisualShapeArray([p.GEOM_BOX] * 5, halfExtents=[e for e, _ in parts],
                                       visualFramePositions=[q for _, q in parts], rgbaColors=[rgba] * 5,
                                       physicsClientId=self.c)
        return p.createMultiBody(0, col, vis, basePosition=[b["x"], b["y"], 0.0], physicsClientId=self.c)

    def _draw_floor_marks(self):
        if not self.gui:
            return
        line = lambda a, b, rgb, w=1.5: p.addUserDebugLine([*a, 0.0005], [*b, 0.0005], rgb, w, physicsClientId=self.c)
        reach = getattr(self.args, "max_reach", 0.45)
        for name, b in arm_config.bases().items():
            angles = np.linspace(0, 2 * np.pi, 73)
            ring = np.column_stack([b["x"] + reach * np.cos(angles), b["y"] + reach * np.sin(angles)])
            for a, c in zip(ring[:-1], ring[1:]):
                line(a, c, ARM_RGB.get(name, (0.5, 0.5, 0.5)), 1.0)
        half = arm_config.middle_half_width()
        for y in (-half, half):
            line([0.0, y], [0.75, y], [0.6, 0.6, 0.0], 1.0)
        for x, y in arm_config.handoff_points():
            line([x - 0.015, y], [x + 0.015, y], [0, 0, 0], 2.5)
            line([x, y - 0.015], [x, y + 0.015], [0, 0, 0], 2.5)

    # ---------------------------------------------------------------------------------------------- state
    def _apply(self, arm, q, grip):
        body, joint = self.arms[arm], self.joint[arm]
        for name, value in zip(JOINTS, q):
            p.resetJointState(body, joint[name], float(value), physicsClientId=self.c)
        low, high = self.gripper_range[arm]
        p.resetJointState(body, joint["gripper"], low + float(grip) / 100.0 * (high - low), physicsClientId=self.c)
        self.q[arm], self.grip[arm] = list(map(float, q)), float(grip)
        if arm in self.held:
            obj, rel = self.held[arm]
            pos, orn = self._grasp_pose(arm)
            pos, orn = p.multiplyTransforms(pos, orn, *rel)
            p.resetBasePositionAndOrientation(obj, pos, orn, physicsClientId=self.c)

    def _grasp_pose(self, arm):
        state = p.getLinkState(self.arms[arm], self.link[arm]["grasp_center_link"], computeForwardKinematics=True,
                               physicsClientId=self.c)
        return state[4], state[5]

    def _check(self, arm):
        """Contacts of this arm (and what it holds) with anything it should not touch."""
        p.performCollisionDetection(physicsClientId=self.c)
        own = {self.arms[arm]}
        allowed = {self.handling.get(arm)}
        held = self.held.get(arm, (None,))[0]
        found = []
        for body in own | ({held} if held is not None else set()):
            for contact in p.getContactPoints(bodyA=body, physicsClientId=self.c):
                other, depth = contact[2], contact[8]
                if other in own or other == held or other in allowed or depth > CONTACT_DEPTH:
                    continue
                if other == self.table and (body == held or contact[3] == self.link[arm].get("base_link")):
                    continue                    # the arm's base stands on the table (the planner allows it too)
                part = "held object" if body == held else self._link_name(arm, contact[3])
                found.append((self._name(other), part, depth))
        for what, part, depth in found:
            record = (arm, what)
            if not self.collisions or self.collisions[-1][:2] != record:
                print(f"  COLLISION: {arm} ({part}) with {what} ({-1000 * depth:.0f} mm deep)")
            self.collisions.append((arm, what, depth))

    def _link_name(self, arm, index):
        return next((name for name, i in self.link[arm].items() if i == index), "base")

    def _name(self, body):
        if body == self.table:
            return "the table"
        for name, b in self.arms.items():
            if b == body:
                return name
        if body in self.bins:
            return f"bin {self.bins.index(body)}"
        if body in self.objects:
            return f"object {self.objects.index(body)}"
        return f"body {body}"

    # ---------------------------------------------------------------------------------------------- commands
    def run(self, arm, kind, *payload):
        """Queue a command for an arm and wait until the main thread has done it (from a worker thread)."""
        done = threading.Event()
        self.commands[arm].put((kind, payload, done))
        done.wait()

    def _steps(self, arm, kind, payload):
        if kind == "pose":
            q, grip = payload
            self._apply(arm, q, grip)
        elif kind == "move":                  # a joint path, the gripper going to `grip` on the way
            path, grip = payload
            path = [np.asarray(q, float) for q in path]
            start = np.asarray(self.q[arm], float)
            if np.linalg.norm(path[0] - start) > 1e-6:
                path.insert(0, start)
            g0 = self.grip[arm]
            total = sum(np.abs(b - a).max() for a, b in zip(path[:-1], path[1:])) or 1e-9
            done = 0.0
            for a, b in zip(path[:-1], path[1:]):
                span = np.abs(b - a).max()
                n = max(1, int(math.ceil(math.degrees(span) / STEP_DEG)))
                for t in np.linspace(0, 1, n + 1)[1:]:
                    share = (done + t * span) / total
                    self._apply(arm, (1 - t) * a + t * b, g0 + (grip - g0) * min(1.0, 2 * share))
                    self._check(arm)
                    yield math.radians(STEP_DEG) / math.radians(JOINT_SPEED_DEG)
                done += span
            if len(path) == 1:
                self._apply(arm, path[0], grip)
        elif kind == "gripper":
            grip = payload[0]
            g0 = self.grip[arm]
            for t in np.linspace(0, 1, 12)[1:]:
                self._apply(arm, self.q[arm], g0 + t * (grip - g0))
                yield 0.02
        elif kind == "handle":                # the object this arm works on now (its jaws may touch it)
            self.handling[arm] = None if payload[0] is None else self.objects[payload[0]]
        elif kind == "attach":
            obj = self.objects[payload[0]]
            pos, orn = self._grasp_pose(arm)
            inv = p.invertTransform(pos, orn)
            obj_pos, obj_orn = p.getBasePositionAndOrientation(obj, physicsClientId=self.c)
            self.held[arm] = (obj, p.multiplyTransforms(*inv, obj_pos, obj_orn))
        elif kind == "release":               # let go: the object lands upright where it is (in a bin: on its floor)
            obj, _ = self.held.pop(arm, (None, None))
            if obj is not None:
                pos, orn = p.getBasePositionAndOrientation(obj, physicsClientId=self.c)
                floor = 0.0
                for b in self.bins:
                    bx, by, _ = p.getBasePositionAndOrientation(b, physicsClientId=self.c)[0]
                    if abs(pos[0] - bx) < self.args.box_size / 2 and abs(pos[1] - by) < self.args.box_size / 2:
                        floor = BIN_WALL
                p.resetBasePositionAndOrientation(obj, [pos[0], pos[1], floor],
                                                  p.getQuaternionFromEuler([0, 0, yaw_of(orn)]), physicsClientId=self.c)
            self.handling[arm] = None
        if False:
            yield 0.0

    def step(self, status=None):
        """Advance every arm one animation step (main thread). Returns the time to wait before the next step."""
        wait = 0.0
        for arm in self.arms:
            if arm not in self.active:
                try:
                    kind, payload, done = self.commands[arm].get_nowait()
                except queue.Empty:
                    continue
                self.active[arm] = (self._steps(arm, kind, payload), done)
            steps, done = self.active[arm]
            try:
                wait = max(wait, next(steps))
            except StopIteration:
                del self.active[arm]
                done.set()
        if self.gui and status is not None:
            for i, line in enumerate(status):
                old = self.text.get(i)
                self.text[i] = p.addUserDebugText(line, [0.75, 0.60, 0.25 - 0.035 * i], [0.1, 0.1, 0.1], 1.2,
                                                  replaceItemUniqueId=old if old is not None else -1,
                                                  physicsClientId=self.c)
        return wait / self.speed if (self.gui or self.realtime) else 0.0

    def busy(self):
        return bool(self.active) or any(not q.empty() for q in self.commands.values())

    def object_final(self):
        """Where every object is now: [(x, y, z, in which bin index or None)]."""
        out = []
        for obj in self.objects:
            pos, _ = p.getBasePositionAndOrientation(obj, physicsClientId=self.c)
            inside = None
            for i, b in enumerate(self.bins):
                bx, by, _ = p.getBasePositionAndOrientation(b, physicsClientId=self.c)[0]
                if abs(pos[0] - bx) < self.args.box_size / 2 and abs(pos[1] - by) < self.args.box_size / 2:
                    inside = i
            out.append((*pos, inside))
        return out

    def close(self):
        p.disconnect(physicsClientId=self.c)


def object_size(o):
    """(short, long) footprint of a scene object [m]."""
    return object_footprint(stl(o["shape"], o["size"]))
