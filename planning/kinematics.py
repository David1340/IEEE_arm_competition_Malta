"""Forward kinematics of the repo's SO-101 model (models/robot/so101/robot.urdf): where the grasp point and the
gripper are for given joint angles, in the arm's base frame.

    from planning.kinematics import grasp_pose
    xyz, tilt_deg = grasp_pose([0.0, -1.2, 1.4, 0.9, -1.57])   # grasp point [m]; gripper tilt from straight down

One small PyBullet client of its own (DIRECT, no window), shared by the threads under a lock; nothing else in it.
"""
from __future__ import annotations

import math
import threading

import numpy as np

from core.paths import URDF

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
_LOCK = threading.Lock()
_MODEL = {}


def _model():
    import pybullet as p

    if not _MODEL:
        client = p.connect(p.DIRECT)
        body = p.loadURDF(str(URDF), useFixedBase=True, physicsClientId=client)
        infos = [p.getJointInfo(body, i, physicsClientId=client) for i in range(p.getNumJoints(body, physicsClientId=client))]
        _MODEL.update(p=p, client=client, body=body, joint={i[1].decode(): i[0] for i in infos},
                      link={i[12].decode(): i[0] for i in infos})
    return _MODEL


def grasp_pose(q):
    """(grasp point xyz in the arm's base frame [m], the gripper's tilt from pointing straight down [deg]) for the
    5 arm joint angles q [rad]. Tilt 0 = top-down, about 90 = level."""
    with _LOCK:
        m = _model()
        p, c, body = m["p"], m["client"], m["body"]
        for name, value in zip(JOINTS, q):
            p.resetJointState(body, m["joint"][name], float(value), physicsClientId=c)
        grasp = p.getLinkState(body, m["link"]["grasp_center_link"], computeForwardKinematics=True, physicsClientId=c)[4]
        orientation = p.getLinkState(body, m["link"]["gripper_link"], computeForwardKinematics=True,
                                     physicsClientId=c)[5]
    towards_tips = -np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)[:, 2]
    tilt = math.degrees(math.acos(float(np.clip(towards_tips @ [0.0, 0.0, -1.0], -1.0, 1.0))))
    return np.array(grasp), tilt


def reach_frame(xy):
    """Unit vectors (along the reach, across it to the left) at a point xy of the arm's base frame."""
    along = np.asarray(xy[:2], float) / max(float(np.linalg.norm(xy[:2])), 1e-9)
    return along, np.array([-along[1], along[0]])
