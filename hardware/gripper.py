"""The gripper: how far to open it for an object (its width from the STL, the jaw gap from the robot model), and
closing it on an object with motor-current feedback.

    from hardware import gripper
    width = gripper.object_width("models/objects/cube_large.stl")      # 0.0566 m: widest across, any direction
    cmd = gripper.open_command(width, margin=0.02)            # LeRobot 0-100 gripper command
    gripper.opening_width(cmd)                                # widest object that fits, centred on the grasp point

    python -m hardware.gripper           # print the jaw gap for each command and the command for every STL

The gripper does not turn to line up with the object (the wrist roll is locked), so the width is the widest the
object is across in any horizontal direction (a 40 mm cube: its 56.6 mm diagonal).

The jaw gap comes from the URDF's collision meshes (models/robot/so101): the moving jaw is turned to each gripper
angle and the part of it below the palm is measured against the grasp point (grasp_center_link), where the planner
puts the object's centre. A command c (0-100) is taken as the fraction c / 100 of the URDF gripper range, the same
mapping planning.sim.World uses for its collision model. Opening only as wide as needed matters: at the old fixed 80
the moving jaw swings ~9.6 cm out, and the planner then has to keep that whole sweep clear of the other objects.
"""
from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import time

import numpy as np

from core.paths import OBJECTS_DIR, URDF

# Closing on an object (close_gripper_with_current_feedback), LeRobot 0-100 scale
GRIPPER_OPEN = 80.0          # the fixed opening used when an object's width is not known
GRIP_CURRENT_MA = 80.0       # gripper motor current that means the object is held
GRIP_CONFIRM_READINGS = 3    # readings in a row above GRIP_CURRENT_MA
GRIP_SQUEEZE_UNITS = 2.0     # units held past the contact position
GRIPPER_MA_PER_UNIT = 6.5    # STS3215 Present_Current: ~6.5 mA per unit
FIXED_JAW_MESH = "wrist_roll_follower_so101_v1"   # the fixed finger is part of gripper_link
FINGER_BAND = 0.02      # m above the grasp point: the object's top (grasp depth 1.5 cm + 5 mm); the jaw below
                        # this height is what must clear the object
MAX_OPEN = 80.0         # never open further than this (the previous fixed opening)
MIN_OPEN = 20.0


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _origin(element):
    o = element.find("origin")
    xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
    rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
    return _rpy(*rpy), xyz


def _obj_vertices(path):
    return np.array([[float(v) for v in line.split()[1:4]] for line in path.read_text().splitlines()
                     if line.startswith("v ")])


def _collision_hulls(link, name_filter=None):
    """The collision hulls (vertex arrays) of a URDF link, in that link's frame."""
    hulls = []
    for collision in link.findall("collision"):
        mesh = collision.find("geometry/mesh")
        if mesh is None or (name_filter and name_filter not in mesh.get("filename")):
            continue
        R, t = _origin(collision)
        hulls.append(_obj_vertices(URDF.parent / mesh.get("filename")) @ R.T + t)
    return hulls


def _below(hull, z_cut):
    """The vertices of a convex hull cut off at z = z_cut (what is below): the vertices below, plus where every
    segment between a vertex below and one above crosses the cut (the hull is convex, so these lie in it)."""
    low, high = hull[hull[:, 2] < z_cut], hull[hull[:, 2] >= z_cut]
    if len(low) == 0 or len(high) == 0:
        return low
    a, b = low[:, None, :], high[None, :, :]
    s = (z_cut - a[..., 2]) / (b[..., 2] - a[..., 2])
    return np.vstack([low, (a + s[..., None] * (b - a)).reshape(-1, 3)])


@lru_cache(maxsize=1)
def jaw_table():
    """(commands 0-100, widest object centred on the grasp point [m]) from the URDF; NaN where the jaw has swung
    up above the finger band (wider than anything we pick)."""
    root = ET.parse(URDF).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    # the grasp point in gripper_link's frame
    R1, t1 = _origin(joints["gripper_frame_joint"])
    _, t2 = _origin(joints["grasp_center_joint"])
    grasp = R1 @ t2 + t1
    z_cut = grasp[2] + FINGER_BAND
    fixed = [_below(h, z_cut) for h in _collision_hulls(links["gripper_link"], FIXED_JAW_MESH)]
    fixed_face = max(h[:, 0].max() for h in fixed if len(h))
    moving_local = _collision_hulls(links["moving_jaw_so101_v1_link"])
    joint = joints["gripper"]
    Rj, tj = _origin(joint)
    limit = joint.find("limit")
    low, high = float(limit.get("lower")), float(limit.get("upper"))
    commands = np.arange(0.0, 100.01, 0.5)
    widths = []
    for c in commands:
        R = Rj @ _rpy(0, 0, low + c / 100 * (high - low))
        finger = [_below(h @ R.T + tj, z_cut) for h in moving_local]
        finger = [h for h in finger if len(h)]
        widths.append(2 * (min(h[:, 0].min() for h in finger) - grasp[0]) if finger else np.nan)
    return commands, np.array(widths), {"grasp_x": grasp[0], "fixed_face_x": fixed_face}


@lru_cache(maxsize=1)
def tip_below_grasp():
    """How far the fingertips are below the grasp point when the gripper points down [m] (from the URDF)."""
    root = ET.parse(URDF).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    R1, t1 = _origin(joints["gripper_frame_joint"])
    _, t2 = _origin(joints["grasp_center_joint"])
    grasp = R1 @ t2 + t1
    tip = min(h[:, 2].min() for h in _collision_hulls(links["gripper_link"], FIXED_JAW_MESH))
    return float(grasp[2] - tip)


def fixed_face_point():
    """The fixed finger's inner face at the grasp point's height, gripper_link's frame [m] (x, y, z). An object the
    gripper is closed on touches it, so its centre is half its width further along x."""
    root = ET.parse(URDF).getroot()
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    R1, t1 = _origin(joints["gripper_frame_joint"])
    _, t2 = _origin(joints["grasp_center_joint"])
    grasp = R1 @ t2 + t1
    _, _, info = jaw_table()
    return np.array([info["fixed_face_x"], grasp[1], grasp[2]])


def held_point(command):
    """Where the centre of an object held by the jaws is, in gripper_link's frame [m] (x, y, z): halfway between
    the fixed finger's inner face and the moving jaw at this gripper command (0-100), at the grasp point's height.
    With the gripper closed on an object, that is the object's centre."""
    root = ET.parse(URDF).getroot()
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    R1, t1 = _origin(joints["gripper_frame_joint"])
    _, t2 = _origin(joints["grasp_center_joint"])
    grasp = R1 @ t2 + t1
    _, _, info = jaw_table()
    moving_x = info["grasp_x"] + opening_width(command) / 2
    return np.array([(info["fixed_face_x"] + moving_x) / 2, grasp[1], grasp[2]])


def opening_width(command):
    """Widest object [m] that fits between the jaws at this command, centred on the grasp point."""
    commands, widths, _ = jaw_table()
    return float(np.interp(command, commands[np.isfinite(widths)], widths[np.isfinite(widths)]))


def open_command(width, margin=0.02, max_open=MAX_OPEN, min_open=MIN_OPEN):
    """The smallest command (0-100) whose opening clears an object `width` wide plus `margin` (total, m: half on
    each side), within [min_open, max_open]."""
    commands, widths, _ = jaw_table()
    ok = (commands >= min_open) & (commands <= max_open) & np.isfinite(widths) & (widths >= width + margin)
    return float(commands[ok][0]) if ok.any() else float(max_open)


def squeeze_command(width):
    """The command (0-100) at which the jaws close on an object `width` wide [m]: the moving jaw's face is then
    `width` from the fixed finger's face. Wherever the object sits between the jaws, the moving jaw pushes it against
    the fixed finger (30 mm from the grasp point) and the motor current only rises here, not where the jaw first
    touches an object centred on the grasp point (open_command(width, margin=0), up to ~25 units earlier for a 30 mm
    object)."""
    commands, widths, info = jaw_table()
    fixed_gap = info["grasp_x"] - info["fixed_face_x"]          # grasp point to the fixed finger's face [m]
    ok = np.isfinite(widths)
    order = np.argsort(widths[ok])
    # widths[i] / 2 = how far the moving jaw's face is from the grasp point at command i
    return float(np.interp(2 * (width - fixed_gap), widths[ok][order], commands[ok][order]))


def _stl_vertices(path):
    data = Path(path).read_bytes()
    if data[:5] == b"solid" and b"facet" in data[:400]:
        return np.array([[float(v) for v in line.split()[1:4]] for line in data.decode(errors="ignore").splitlines()
                         if line.strip().startswith("vertex")])
    count = struct.unpack("<I", data[80:84])[0]
    facets = np.frombuffer(data[84:84 + 50 * count],
                           dtype=np.dtype([("normal", "<3f4"), ("v", "<9f4"), ("attr", "<u2")]))
    return facets["v"].reshape(-1, 3).astype(float)


@lru_cache(maxsize=32)
def object_height(stl_path):
    """Height [m] of an object model as it stands in its STL (STL in mm or m)."""
    z = _stl_vertices(stl_path)[:, 2]
    height = float(z.max() - z.min())
    return height / 1000.0 if height > 1.0 else height


def _min_rect_size(xy):
    """(short, long) sides of the smallest rectangle, any angle, around 2-D points (1 degree steps)."""
    best = None
    for angle in np.radians(np.arange(0.0, 90.0, 1.0)):
        c, s_ = np.cos(angle), np.sin(angle)
        u, v = xy @ [c, s_], xy @ [-s_, c]
        size = (np.ptp(u), np.ptp(v))
        if best is None or size[0] * size[1] < best[0] * best[1]:
            best = size
    return tuple(sorted(best))


@lru_cache(maxsize=32)
def object_footprint(stl_path):
    """(short, long) sides [m] of the smallest rectangle around the object seen from above, standing as in its STL."""
    v = _stl_vertices(stl_path)
    scale = 1000.0 if np.ptp(v[:, 2]) > 1.0 else 1.0
    return tuple(x / scale for x in _min_rect_size(v[:, :2]))


@lru_cache(maxsize=32)
def object_top_face(stl_path):
    """(short, long) sides [m] of the smallest rectangle around the object's top face (its vertices within 0.5 mm
    of its highest point), standing as in its STL."""
    v = _stl_vertices(stl_path)
    scale = 1000.0 if np.ptp(v[:, 2]) > 1.0 else 1.0
    top = v[v[:, 2] >= v[:, 2].max() - 0.5 * (scale / 1000.0)]
    return tuple(x / scale for x in _min_rect_size(top[:, :2]))


@lru_cache(maxsize=32)
def object_width(stl_path):
    """Widest horizontal extent [m] of an object model, over every direction (STL in mm or m)."""
    xy = _stl_vertices(stl_path)[:, :2]
    xy = xy - xy.mean(axis=0)
    angles = np.linspace(0, np.pi, 180, endpoint=False)
    width = max(np.ptp(xy @ [np.cos(a), np.sin(a)]) for a in angles)
    return width / 1000.0 if width > 1.0 else width


def read_gripper_current_ma(robot):
    """STS3215 Present_Current in mA; bit 15 holds the direction."""
    raw = int(robot.bus.read("Present_Current", "gripper", normalize=False))
    return (raw & 0x7FFF) * GRIPPER_MA_PER_UNIT


def close_gripper_with_current_feedback(robot, closed_target, current_threshold_ma, confirm_readings,
                                        squeeze_units, *, timeout=10.0, miss_below=None, stall_readings=None,
                                        empty_below=None, trace=None, info=None, close_speed=None):
    """Close the gripper until it is holding something, then hold a light squeeze.

    Holding is recognised two ways, whichever comes first:
      current: the motor current stays at or above current_threshold_ma for confirm_readings readings;
      stall (stall_readings set): the gripper is told to close but has stopped moving (less than 0.5 units over the
          last stall_readings readings) while still above empty_below. Something is between the jaws then, whatever
          the current; a gripper that stops at or below empty_below has closed on nothing.
    A miss: the gripper reaches closed_target (or stops at or below empty_below) without either, or passes miss_below
    (a model-based shortcut; None = off) without a current rise.

    close_speed (units of the 0-100 scale per second): the target moves towards closed_target at this speed instead of
    being sent at once (the servo's full speed can hit an object hard and flick it out sideways); None = at once.

    Returns (grasped, stop_position, hold_position, current_ma). trace (a list): (seconds, position, current mA) of
    every reading is appended to it; info (a dict): info["reason"] is set to how it ended ("current", "stall",
    "closed empty", "closed fully", "passed miss_below" or "timeout").
    """
    info = {} if info is None else info
    started = time.monotonic()
    deadline = started + timeout
    position = start = float(robot.get_observation()["gripper.pos"])
    above = 0
    current_ma = 0.0
    history = []
    info["reason"] = "timeout"
    commanded, last = position, started
    while time.monotonic() < deadline:
        observation = robot.get_observation()
        hold = {key: value for key, value in observation.items()
                if key.endswith(".pos") and key != "gripper.pos"}
        now = time.monotonic()
        if close_speed:
            commanded = max(float(closed_target), commanded - close_speed * (now - last))
        else:
            commanded = float(closed_target)
        last = now
        robot.send_action({**hold, "gripper.pos": commanded})
        time.sleep(0.04)
        position = float(robot.get_observation()["gripper.pos"])
        current_ma = read_gripper_current_ma(robot)
        history.append(position)
        if trace is not None:
            trace.append((round(time.monotonic() - started, 3), round(position, 2), round(current_ma, 1)))
        above = above + 1 if current_ma >= current_threshold_ma else 0
        if above >= confirm_readings:
            info["reason"] = "current"
            hold_position = max(position - squeeze_units, closed_target)
            robot.send_action({**hold, "gripper.pos": hold_position})
            return True, position, hold_position, current_ma
        if stall_readings and len(history) > stall_readings and (start - position >= 2.0
                                                                 or time.monotonic() - started > 0.4):
            still = history[-1 - stall_readings] - position < 0.5            # not moving any more
            if still and empty_below is not None and position <= empty_below:
                info["reason"] = "closed empty"                              # stopped at the end: nothing there
                break
            if still and commanded < position - 2.0:                         # told to close further than it is
                info["reason"] = "stall"
                hold_position = max(position - squeeze_units, closed_target)
                robot.send_action({**hold, "gripper.pos": hold_position})
                return True, position, hold_position, current_ma
        if abs(position - closed_target) <= 1.0:
            info["reason"] = "closed fully"
            break
        if miss_below is not None and position < miss_below and above == 0:
            info["reason"] = "passed miss_below"
            break                                       # past where it would have been held: missed
    return False, position, position, current_ma


def main():
    commands, widths, info = jaw_table()
    print(f"grasp point {1000 * info['grasp_x']:.1f} mm from the gripper axis; objects wider than "
          f"{2000 * (info['grasp_x'] - info['fixed_face_x']):.0f} mm centred there touch the fixed finger")
    print("command  widest object centred on the grasp point")
    for c, w in zip(commands, widths):
        if c % 10 == 0:
            print(f"  {c:5.0f}  " + (f"{1000 * w:5.1f} mm" if np.isfinite(w) else "  (jaw above the finger band)"))
    print("\nobject                     width     command (margin 20 mm)")
    for stl in sorted(OBJECTS_DIR.glob("*.stl")):
        w = object_width(stl)
        c = open_command(w)
        print(f"  {stl.stem:24s} {1000 * w:5.1f} mm  {c:5.1f}  (opens {1000 * opening_width(c):.0f} mm)")


if __name__ == "__main__":
    main()
