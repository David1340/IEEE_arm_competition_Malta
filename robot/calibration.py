"""All the calibrations in one file: calibration.json at the repo root.

    camera_to_base     4 x 4: ZED left camera (IMAGE frame: x right, y down, z forward) -> robot base frame
                       (x forward, y to the arm's left, z up, origin under the shoulder pan axis), metres
    camera             when, how (aruco / arm points) and how well it was fitted; the ZED serial
    table_z            the table's height in the base frame (m)
    aruco              the marker dictionary and each marker's centre in the base frame, to re-fit the camera
                       after it is bumped (robot/calibrate.py)
    arm                the LeRobot calibration file of the follower, and the fine joint offsets (degrees; true
                       angle = LeRobot reading + offset; fine_joint_calibration_gui.py writes them here)

The camera fit: the table plane from the point cloud gives the camera's tilt (thousands of points: precise);
two or more known points on the table (ArUco markers, or blocks touched with the arm) give its turn about z and
its position. camera_transform.npy (read by yolo_dataset/object_3d_live.py and Full_pipeline.py) is written with
the same matrix, so their "table" frame is the robot's base frame.
"""

import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "calibration.json"
LEGACY_CAMERA = ROOT / "camera_transform.npy"
LEGACY_OFFSETS = ROOT / "calibration" / "robots" / "so_follower" / "fine_joint_offsets.json"
DEFAULT_LEROBOT = "~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower_arm.json"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def load():
    """calibration.json as a dict ({} when there is none)."""
    return json.loads(FILE.read_text()) if FILE.is_file() else {}


def save(data):
    """Write calibration.json (atomically), and camera_transform.npy from it for the scripts that read that. The
    arm's calibrations found elsewhere (the LeRobot file's path, the older fine_joint_offsets.json) are brought in
    the first time, so the file holds them all."""
    arm = data.setdefault("arm", {})
    arm.setdefault("lerobot_calibration", DEFAULT_LEROBOT)
    if "joint_offsets_deg" not in arm:
        arm["joint_offsets_deg"] = joint_offsets()
    data.setdefault("table_z", 0.0)
    tmp = FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(FILE)
    if data.get("camera_to_base") is not None:
        np.save(LEGACY_CAMERA, np.array(data["camera_to_base"], float))


def update(**sections):
    """Merge sections into calibration.json and save."""
    data = load()
    for k, v in sections.items():
        if isinstance(v, dict) and isinstance(data.get(k), dict):
            data[k].update(v)
        else:
            data[k] = v
    save(data)
    return data


def camera_to_base():
    """The 4 x 4 camera -> base matrix, or None."""
    m = load().get("camera_to_base")
    return None if m is None else np.array(m, float)


def lerobot_calibration():
    return Path(load().get("arm", {}).get("lerobot_calibration", DEFAULT_LEROBOT)).expanduser()


def joint_offsets():
    """The fine joint offsets (degrees) by joint: calibration.json's, else the old fine_joint_offsets.json's, else 0."""
    arm = load().get("arm", {})
    if "joint_offsets_deg" in arm:
        values = arm["joint_offsets_deg"]
    elif LEGACY_OFFSETS.is_file():
        values = json.loads(LEGACY_OFFSETS.read_text()).get("joint_offsets_deg", {})
    else:
        values = {}
    return {j: float(values.get(j, 0.0)) for j in JOINTS}


# ---------------------------------------------------------------------------------------------- the fit
def fit_plane(xyz, iters=300, tol=0.004, seed=0):
    """The dominant plane (the table) in camera-frame points: RANSAC, then least squares on its inliers.
    Returns (unit normal pointing towards the camera, offset d with n . p = d)."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        a, b, c = xyz[rng.choice(len(xyz), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs(xyz @ n - n @ a) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    p = xyz[best]
    centre = p.mean(axis=0)
    n = np.linalg.svd(p - centre)[2][2]
    if n @ (-centre) < 0:          # towards the camera (the camera is at the origin)
        n = -n
    return n, float(n @ centre), float(best.mean())


def rotation_to_z(n):
    """The rotation that turns unit vector n onto +z."""
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(n, z), float(n @ z)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * (1 / (1 + c))


def fit_camera(plane, cam_pts, base_xy, table_z=0.0):
    """camera -> base (4 x 4) from the table plane (normal, d in the camera frame) and >= 2 points on or above the
    table, known in both frames (camera xyz, base xy). The plane fixes the tilt and the height (the table at
    table_z); the points fix the turn about z (2D Procrustes) and x, y. Returns (T, residuals in mm)."""
    n, d = plane
    R1 = rotation_to_z(n)
    q = np.asarray(cam_pts, float) @ R1.T
    b = np.asarray(base_xy, float)[:, :2]
    if len(q) >= 2:
        qa, ba = q[:, :2] - q[:, :2].mean(0), b - b.mean(0)
        yaw = math.atan2((qa[:, 0] * ba[:, 1] - qa[:, 1] * ba[:, 0]).sum(), (qa * ba).sum())
    else:
        raise ValueError("need 2 or more points")
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    R = Rz @ R1
    t = np.zeros(3)
    t[:2] = (b - (np.asarray(cam_pts) @ R.T)[:, :2]).mean(axis=0)
    t[2] = table_z - d                      # the plane n . p = d lands at z = d after R1 (Rz keeps z)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    res = 1000 * np.linalg.norm((np.asarray(cam_pts) @ R.T + t)[:, :2] - b, axis=1)
    return T, res


def table_points(cloud, step=6, roi=None):
    """Camera-frame points sampled from the cloud (inside roi = (x, y, w, h) if given), for the plane fit."""
    h, w = cloud.shape[:2]
    x0, y0, rw, rh = roi if roi is not None else (0, 0, w, h)
    pts = np.asarray(cloud[y0:y0 + rh:step, x0:x0 + rw:step, :3], float).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0.2) & (pts[:, 2] < 2.0)]
    return pts


# ---------------------------------------------------------------------------------------------- ArUco
DICTS = ("DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_50", "DICT_6X6_50", "DICT_ARUCO_ORIGINAL", "DICT_APRILTAG_36h11")


def find_markers(frame, dictionary=None):
    """{id: 4 x 2 corner pixels} of the ArUco markers in the frame, and the dictionary used (the given one, else
    the one of DICTS that finds the most)."""
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    best = ({}, dictionary or DICTS[0])
    for name in ([dictionary] if dictionary else DICTS):
        det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name)), params)
        corners, ids, _ = det.detectMarkers(gray)
        found = {} if ids is None else {str(int(i)): c[0] for i, c in zip(ids.ravel(), corners)}
        if len(found) > len(best[0]):
            best = (found, name)
    return best


def marker_centres(cloud, markers):
    """{id: camera-frame xyz} of each marker's centre: the median of the point cloud inside its square (shrunk)."""
    import cv2

    out = {}
    for mid, c in markers.items():
        centre = c.mean(axis=0)
        inner = centre + 0.6 * (c - centre)
        x0, y0 = np.floor(inner.min(axis=0)).astype(int)
        x1, y1 = np.ceil(inner.max(axis=0)).astype(int) + 1
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(mask, [np.rint(inner - [x0, y0]).astype(np.int32)], 1)
        ys, xs = np.nonzero(mask)
        pts = np.asarray(cloud[y0 + ys, x0 + xs, :3], float)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) >= 5:
            out[mid] = np.median(pts, axis=0)
    return out


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")
