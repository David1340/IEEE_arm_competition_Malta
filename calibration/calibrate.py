"""Every calibration of the robot, in one script; every calibration value, in one file: calibration.json.

    python calibration/calibrate.py --marker-xy 0:0.45,0.30 1:0.15,0.30 2:0.15,-0.30 3:0.45,-0.30
                                                       # the first time: each marker ID's centre x,y in the TABLE
                                                       # frame (below), measured by hand (remembered afterwards)
    python calibration/calibrate.py                          # after the camera moved: the ArUco markers, a few s, no arm
    python calibration/calibrate.py --marker-ids 3,2,1,0     # the first time: which ID is TOP_RIGHT, BOTTOM_RIGHT,
                                                       # BOTTOM_LEFT, TOP_LEFT (remembered afterwards)
    python calibration/calibrate.py ids                      # live view: which marker has which ID
    python calibration/calibrate.py arm                      # the most accurate camera fit: touch 4 blocks with the arm
                                                       # (once); also re-measures the markers in the arm's frame
    python calibration/calibrate.py check                    # torque off: where the model puts the tip and the jaws while
                                                       # the arm is moved by hand
    python calibration/calibrate.py offsets                  # the fine joint offsets: PyBullet sliders over the live arm
    python calibration/calibrate.py offsets --arm follower_blue   # the same for one of the two arms (its own file,
                                                       # calibration/robots/so_follower/fine_joint_offsets_<arm>.json;
                                                       # plus sliders for its base origin x, y, yaw in arms.json)
    python calibration/calibrate.py offsets --arm follower_pink --objects   # ... with the camera's objects shown too
    python calibration/calibrate.py offsets --arm both --objects            # both arms in one window, objects too
    python calibration/calibrate.py show                     # print what calibration.json holds

The TABLE frame, shared by both arms: origin on the table halfway between the two arms' shoulder pan axes;
x forward (the way both arms face), y to the left (seen from behind the arms), z up; metres. Each arm's own base
frame has the same axes, with its origin under its shoulder pan axis: follower_pink (left) at y = +0.25,
follower_blue (right) at y = -0.25. These bases are set in arms.json ("base" of each follower).

calibration.json (at the repo root):
    camera_to_table    4 x 4: ZED left camera (IMAGE frame: x right, y down, z forward) -> TABLE frame, metres
    camera_to_base     (older files) camera -> one arm's base frame; camera_to_table replaces it
    camera             when, how (aruco / arm) and how well it was fitted; the ZED serial
    table_z            the table's height in the base frame (m)
    aruco              the marker dictionary (DICT_4X4_50) and each marker's centre in the TABLE frame
    arm                the follower's LeRobot calibration file, and the fine joint offsets (degrees; true angle =
                       LeRobot reading + offset; control/fine_joint_offsets.py still reads its own json; `offsets` keeps both in sync)
camera_transform.npy is written from it for the scripts that read that file (experiments/object_3d_live.py,
experiments/Full_pipeline.py, vision/live.py): camera -> the default arm's (follower_pink's) base frame.

The camera fit: the table plane from the ZED point cloud fixes the camera's tilt and height (thousands of
points); two or more known points on the table (markers, or blocks touched with the fixed finger's tip) fix its
turn about z and its position. The markers' positions are measured by hand in the TABLE frame (--marker-xy; the
older --marker-ids uses QR_POSITIONS below, which must then hold TABLE-frame values). Which way the camera looks
does not matter: the fit finds its rotation. Pictures of each run go to outputs/calibrate/.

Other modules import this one for calibration.json (load, camera_to_base, joint_offsets, ...): it runs nothing
on import.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
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
    if data.get("camera_to_table") is not None:
        np.save(LEGACY_CAMERA, arm_from_table(default_arm(data), data) @ np.array(data["camera_to_table"], float))
    elif data.get("camera_to_base") is not None:
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


def arms(data=None):
    """{arm id: {x, y, yaw_deg}}: each follower's base in the TABLE frame, from arms.json."""
    sys.path.insert(0, str(ROOT))
    from control import arm_config

    return {name: arm_config.base(name) for name in arm_config.followers()}


def default_arm(data=None):
    """The arm camera_transform.npy is written for (arms.json "default_arm")."""
    sys.path.insert(0, str(ROOT))
    from control import arm_config

    return arm_config.default_arm()


def arm_from_table(arm, data=None):
    """4 x 4: TABLE frame -> the arm's base frame."""
    all_arms = arms(data)
    if arm not in all_arms:
        raise KeyError(f"unknown arm {arm!r}; known: {', '.join(all_arms)}")
    b = all_arms[arm]
    yaw = math.radians(b["yaw_deg"])
    R = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ np.array([b["x"], b["y"], 0.0])
    return T


def camera_to_table():
    """The 4 x 4 camera -> TABLE matrix, or None."""
    m = load().get("camera_to_table")
    return None if m is None else np.array(m, float)


def camera_to_base(arm=None):
    """The 4 x 4 camera -> arm base matrix (the default arm if none is given), or None. An older file with only
    camera_to_base (one arm, no TABLE frame) gives that matrix, for the default arm only."""
    data = load()
    arm = arm or default_arm(data)
    if data.get("camera_to_table") is not None:
        return arm_from_table(arm, data) @ np.array(data["camera_to_table"], float)
    if data.get("camera_to_base") is not None and arm == default_arm(data):
        return np.array(data["camera_to_base"], float)
    return None


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


# ---------------------------------------------------------------------------------------------- the camera
# The hand-measured marker centres (from the old qr_calibrate.py) (robot base frame, m), by position on the table
QR_POSITIONS = {"TOP_RIGHT": [35.5, 12, 0.0], "BOTTOM_RIGHT": [35.5, 52, 0.0],
                "BOTTOM_LEFT": [-35.5, 52, 0.0], "TOP_LEFT": [-35.5, 52, 0.0]}
DICTIONARY = "DICT_4X4_50"


def open_zed(serial=None):
    import pyzed.sl as sl

    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    if serial is not None:
        init.set_from_serial_number(serial)
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"Could not open ZED camera: {status}")
    return zed


def grab(zed, frames=10):
    """(last BGR frame, its point cloud (camera frame, H x W x 3), {marker id: camera xyz} median over frames,
    the marker corners of the last frame, the dictionary)."""
    import cv2
    import pyzed.sl as sl

    image, cloud_mat, rt = sl.Mat(), sl.Mat(), sl.RuntimeParameters()
    for _ in range(5):
        zed.grab(rt)
    per, frame, cloud, corners = {}, None, None, {}
    dictionary = load().get("aruco", {}).get("dictionary", DICTIONARY)
    n = 0
    while n < frames:
        if zed.grab(rt) != sl.ERROR_CODE.SUCCESS:
            continue
        n += 1
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(cloud_mat, sl.MEASURE.XYZRGBA)
        frame = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)
        cloud = np.array(cloud_mat.get_data()[:, :, :3], np.float32)
        corners, dictionary = find_markers(frame, dictionary)
        for mid, xyz in marker_centres(cloud, corners).items():
            per.setdefault(mid, []).append(xyz)
    return frame, cloud, {k: np.median(v, axis=0) for k, v in per.items() if len(v) >= frames // 2}, corners, dictionary


def plane_of(cloud):
    """The table plane in the camera frame: (normal, d), from the cloud inside yolo_dataset/roi.json's region."""
    roi = None
    roi_file = ROOT / "yolo_dataset" / "roi.json"
    if roi_file.is_file():
        r = json.loads(roi_file.read_text())
        sx, sy = cloud.shape[1] / r.get("ref_size", [cloud.shape[1]])[0], cloud.shape[0] / r.get("ref_size", [0, cloud.shape[0]])[1]
        roi = (round(r["x"] * sx), round(r["y"] * sy), round(r["w"] * sx), round(r["h"] * sy))
    n, d, share = fit_plane(table_points(cloud, roi=roi))
    print(f"table plane: {100 * share:.0f} % of the points within 4 mm of it")
    return n, d


def draw(frame, corners, cam, T, markers, path):
    """The frame with each marker: its ID, and how far the fit puts it from its stored position (mm)."""
    import cv2

    img = frame.copy()
    for mid, c in corners.items():
        cv2.polylines(img, [np.rint(c).astype(np.int32)], True, (0, 255, 0), 3)
        text = f"ID {mid}"
        if mid in cam and mid in markers and T is not None:
            b = cam[mid] @ T[:3, :3].T + T[:3, 3]
            text += f"  off {1000 * np.hypot(*(b[:2] - np.array(markers[mid])[:2])):.0f} mm"
        elif mid not in markers:
            text += "  (position unknown)"
        org = tuple(np.rint(c.mean(axis=0) + [-40, -40]).astype(int))
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    print(f"Picture: {path}")


def aruco(args):
    data = load()
    markers = data.get("aruco", {}).get("markers", {})
    if args.marker_xy:
        markers = {}
        for item in args.marker_xy:
            try:
                mid, xy = item.split(":")
                x, y = (float(v) for v in xy.split(","))
            except ValueError:
                raise SystemExit(f"--marker-xy: {item!r} is not ID:x,y (e.g. 3:0.45,-0.30)")
            markers[mid.strip()] = [x, y, 0.0]
        if len(markers) < 2:
            raise SystemExit("--marker-xy: give 2 or more markers (4 is better)")
        print("Marker positions (TABLE frame, hand-measured): "
              + ", ".join(f"ID {m} = x {p[0]:.3f} y {p[1]:.3f}" for m, p in markers.items()))
    elif args.marker_ids:
        ids = [s.strip() for s in args.marker_ids.split(",")]
        if len(ids) != 4 or len(set(ids)) != 4:
            raise SystemExit("--marker-ids: four different IDs, in the order TOP_RIGHT,BOTTOM_RIGHT,BOTTOM_LEFT,TOP_LEFT")
        markers = {i: QR_POSITIONS[name] for i, name in zip(ids, QR_POSITIONS)}
        print("Marker positions (hand-measured): " + ", ".join(f"ID {i} = {n}" for i, n in zip(ids, QR_POSITIONS)))
    if not markers:
        raise SystemExit("No marker positions yet: run once with --marker-xy ID:x,y ... (see the mode ids for the IDs)")
    zed = open_zed(args.serial)
    try:
        frame, cloud, cam, corners, dictionary = grab(zed, args.frames)
    finally:
        zed.close()
    use = [m for m in markers if m in cam]
    print(f"markers seen: {sorted(cam)}; with a stored position: {use}")
    if len(use) < 2:
        draw(frame, corners, cam, None, markers, ROOT / "outputs" / "calibrate" / "aruco.jpg")
        raise SystemExit("Need 2 or more of the stored markers in view (see the picture)")
    plane = plane_of(cloud)
    T, res = fit_camera(plane, [cam[m] for m in use], [markers[m] for m in use], data.get("table_z", 0.0))
    for m, r in zip(use, res):
        print(f"  marker {m}: {r:.1f} mm from its stored position after the fit")
    draw(frame, corners, cam, T, markers, ROOT / "outputs" / "calibrate" / "aruco.jpg")
    old = camera_to_table()
    if old is not None:
        moved = 1000 * np.linalg.norm(old[:3, 3] - T[:3, 3])
        turned = np.degrees(np.arccos(np.clip((np.trace(old[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1)))
        print(f"compared with the last calibration: camera moved {moved:.0f} mm, turned {turned:.1f} deg")
    if args.yes or input("Save to calibration.json? [Y/n] ").strip().lower() in ("", "y", "yes"):
        data = load()
        data.pop("camera_to_base", None)          # the older one-arm matrix; camera_to_table replaces it
        save(data)
        update(camera_to_table=T.tolist(),
                   camera={"updated": stamp(), "method": f"aruco ({len(use)} markers + table plane)",
                           "residual_mm": float(np.max(res)), "serial": args.serial},
                   aruco={"dictionary": dictionary, "markers": {k: list(map(float, v)) for k, v in markers.items()}})
        print(f"Saved {FILE} (and {LEGACY_CAMERA.name}, for {default_arm()}).")
        for name in arms():
            Tb = camera_to_base(name)
            print(f"  {name}: camera at x {Tb[0, 3]:.3f} y {Tb[1, 3]:.3f} z {Tb[2, 3]:.3f} m in its base frame")


def need_so101ik():
    """The `arm` and `check` modes need the so101ik / detect / go_to_object helpers of the branch
    claude/busy-franklin-rgg1e9 (robot/), which are not in this layout."""
    raise SystemExit("This mode needs so101ik, detect and go_to_object (robot/ in the branch "
                     "claude/busy-franklin-rgg1e9), which are not in this repository layout.")


def arm(args):
    need_so101ik()
    import detect
    from so101ik import Arm

    data = load()
    offsets = joint_offsets()
    lr = lerobot_calibration()
    zed = open_zed(args.serial)
    cam_pts, arm_pts, frame, cloud, cam_markers, corners = [], [], None, None, {}, {}
    robot = Arm(calibration_file=lr if lr.is_file() else None, offsets=offsets, name="calibrate")
    try:
        robot.check()
        robot.torque_off()
        print("Torque OFF (hold the arm when you move it).")
        frame, cloud, cam_markers, corners, dictionary = grab(zed, args.frames)
        plane = plane_of(cloud)
        # A provisional pose from the plane alone (z up, the table at 0; x, y not yet known): the detections'
        # "top" is then the real top. Their camera-frame points are taken back out of it.
        T0 = np.eye(4)
        T0[:3, :3] = rotation_to_z(plane[0])
        T0[2, 3] = -plane[1]
        T0inv = np.linalg.inv(T0)
        while len(cam_pts) < args.points:
            n = len(cam_pts) + 1
            input(f"\n[{n}/{args.points}] Put ONE block somewhere new in reach (spread them: left, right, near, far), "
                  "move the arm and your hands out of view, Enter ... ")
            dets, _ = detect.detect_live(5, transform=detect.IDENTITY_T, camera_to_table=T0, zed=zed)
            dets = [d for d in dets if d.name not in detect.SKIP_CLASSES and d.name != "bin"]
            if not dets:
                print("  the camera found no block; try again")
                continue
            d = dets[0]
            print(f"  camera: {d.colour} {d.name} conf {d.conf:.2f}")
            input("  Now put the TIP OF THE FIXED FINGER on the centre of its top face, hold it still, Enter ... ")
            x, y, z, _ = robot.where()
            print(f"  arm: tip at x {100 * x:.1f} y {100 * y:.1f} z {100 * z:.1f} cm")
            cam_pts.append(T0inv[:3, :3] @ d.table_xyz + T0inv[:3, 3])   # back to the camera frame
            arm_pts.append([x, y, z])
    except KeyboardInterrupt:
        print()
    finally:
        robot.close()
        zed.close()
    if len(cam_pts) < 2:
        raise SystemExit("Need 2 or more positions")
    T, res = fit_camera(plane, cam_pts, arm_pts, data.get("table_z", 0.0))
    for a, r in zip(arm_pts, res):
        print(f"  block at x {100 * a[0]:.1f} y {100 * a[1]:.1f}: {r:.1f} mm off after the fit")
    markers = {m: (xyz @ T[:3, :3].T + T[:3, 3]).tolist() for m, xyz in cam_markers.items()}
    if markers:
        old = data.get("aruco", {}).get("markers", {})
        for m, p in sorted(markers.items()):
            was = f" (stored {np.round(np.array(old[m][:2]) * 100, 1)} cm)" if m in old else ""
            print(f"  marker {m} measured at x {100 * p[0]:.1f} y {100 * p[1]:.1f} cm{was}")
    if args.yes or input("Save to calibration.json (camera and marker positions)? [Y/n] ").strip().lower() in ("", "y", "yes"):
        sections = {"camera_to_base": T.tolist(),
                    "camera": {"updated": stamp(), "method": f"arm ({len(cam_pts)} blocks + table plane)",
                               "residual_mm": float(np.max(res)), "serial": args.serial}}
        if markers:
            sections["aruco"] = {"dictionary": dictionary, "markers": markers}
        update(**sections)
        print(f"Saved {FILE}: later camera bumps only need `python calibration/calibrate.py`.")


def ids(args):
    import cv2

    zed = open_zed(args.serial)
    print("Live view: each marker's ID. q or Esc quits.")
    try:
        while True:
            frame, _, _, corners, dictionary = grab(zed, 1)
            img = frame.copy()
            for mid, c in corners.items():
                cv2.polylines(img, [np.rint(c).astype(np.int32)], True, (0, 255, 0), 3)
                cv2.putText(img, f"ID {mid}", tuple(np.rint(c.mean(axis=0)).astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0), 3)
            cv2.putText(img, dictionary, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.namedWindow("marker ids", cv2.WINDOW_NORMAL)
            cv2.imshow("marker ids", img)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        zed.close()
        cv2.destroyAllWindows()


def show(args):
    data = load()
    if not data:
        print(f"No {FILE.name} yet.")
    print(json.dumps(data, indent=2))
    print(f"\nLeRobot calibration: {lerobot_calibration()}")
    print("Fine joint offsets: " + ", ".join(f"{j} {v:+.2f}" for j, v in joint_offsets().items()))




# ---------------------------------------------------------------------------------------------- check
SHORT = {"shoulder_pan": "pan", "shoulder_lift": "lift", "elbow_flex": "elbow", "wrist_flex": "wrist",
         "wrist_roll": "roll"}


def check(args):
    """Torque off; while the arm is moved by hand, print the tip and the jaws' middle (base frame, from the model)
    and every joint (LeRobot degrees > model degrees with the offset). With the tip on the table, z should be 0;
    on a block the camera found, x and y should be the camera's."""
    need_so101ik()
    from go_to_object import centre_of

    ARM = list(JOINTS)
    offsets = {j: 0.0 for j in ARM} if args.no_offsets else joint_offsets()
    cal = lerobot_calibration()
    from so101ik import Arm


    arm = Arm(port=args.port, calibration_file=cal if cal.is_file() else None, offsets=offsets,
              name="calibrate_check")
    try:
        arm.check()
        arm.torque_off()
        print("Torque OFF: move the arm by hand. Ctrl-C to stop.")
        print("fine offsets: " + "  ".join(f"{j} {offsets[j]:+.2f}" for j in ARM))
        print("joints: LeRobot deg -> model deg (LeRobot + offset)")
        print("Nothing moves by itself: move the arm BY HAND. A line is printed when a joint moves (and every 5 s).")
        last, shown = None, 0.0
        while True:
            deg = arm.read()[0]
            moved = last is None or max(abs(deg[j] - last[j]) for j in ARM) > 0.5
            if moved or time.time() - shown > 5:
                x, y, z, pitch = arm.where()
                c = centre_of(arm.model, arm.model.pose(deg))
                joints = "  ".join(f"{SHORT[j]} {deg[j]:+6.1f}>{deg[j] + offsets[j]:+6.1f}" for j in ARM)
                print(f"TIP x {100 * x:5.1f} y {100 * y:5.1f} z {100 * z:5.1f} | JAWS' MIDDLE x {100 * c[0]:5.1f} "
                      f"y {100 * c[1]:5.1f} z {100 * c[2]:5.1f} cm | pitch {pitch:+4.0f} grip {deg['gripper']:4.0f} | {joints}",
                      flush=True)
                last, shown = deg, time.time()
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        arm.close()




# ---------------------------------------------------------------------------------------------- offsets
class LiveObjects:
    """The camera's detected objects (vision/live.py, as pick and place sees them) in the PyBullet window: each object's
    STL of its detected size standing at its centre in the TABLE frame, a ball on its top centre and a label. The
    camera runs in a background thread; update() draws the latest detections (call it from the GUI loop)."""

    def __init__(self, p, client):
        import threading

        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(ROOT / "yolo_dataset"))
        from vision.live import Vision, add_vision_args

        parser = argparse.ArgumentParser()
        add_vision_args(parser)
        self.vision = Vision(parser.parse_args([]))       # the same defaults as pick_and_place.py
        # the vision code works in the default arm's base frame (camera_transform.npy): back to the TABLE frame
        self.table_from_default = np.linalg.inv(arm_from_table(default_arm()))
        self.p, self.client = p, client
        self.latest, self.lock, self.stop = [], threading.Lock(), threading.Event()
        self.bodies, self.shapes = {}, {}
        self.thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            try:
                frame, cloud = self.vision.grab()
                _, detections = self.vision.detect(frame, cloud)
            except Exception as error:                  # keep the sliders alive if the camera hiccups
                print(f"camera: {error}")
                time.sleep(0.5)
                continue
            sizes = getattr(self.vision, "last_sizes", None) or [""] * len(detections)
            found = []
            for (name, conf, centre, _), size in zip(detections, sizes):
                if centre is None:
                    continue
                xyz = (self.table_from_default @ np.append(np.asarray(centre, float), 1.0))[:3]
                found.append((name.casefold(), size if size in ("small", "large") else "", float(conf), xyz))
            with self.lock:
                self.latest = found

    def _stl(self, name, size):
        from vision.detection import OBJECT_SIZES, OBJECT_STL_FILES

        return OBJECT_SIZES.get(name, {}).get(size) or OBJECT_STL_FILES.get(name)

    def update(self):
        p, c = self.p, self.client
        with self.lock:
            found = list(self.latest)
        seen, count = set(), {}
        for name, size, conf, xyz in found:
            key = (name, size, count.setdefault((name, size), 0))
            count[(name, size)] += 1
            seen.add(key)
            label = f"{size + ' ' if size else ''}{name} {conf:.2f}  x {xyz[0]:+.3f} y {xyz[1]:+.3f} top {xyz[2]:.3f}"
            if key not in self.bodies:
                stl = self._stl(name, size) if name != "bin" else None
                if stl is not None and Path(stl).is_file():
                    if stl not in self.shapes:
                        self.shapes[stl] = p.createVisualShape(p.GEOM_MESH, fileName=str(stl), meshScale=[0.001] * 3,
                                                               rgbaColor=[0.2, 0.8, 0.3, 0.8], physicsClientId=c)
                    body = p.createMultiBody(0, -1, self.shapes[stl], basePosition=[xyz[0], xyz[1], 0.0],
                                             physicsClientId=c)
                else:                                   # a bin (or no STL): a flat square where it is
                    shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.075, 0.075, 0.002],
                                                rgbaColor=[0.9, 0.5, 0.1, 0.6], physicsClientId=c)
                    body = p.createMultiBody(0, -1, shape, basePosition=[xyz[0], xyz[1], 0.002], physicsClientId=c)
                top = p.createMultiBody(0, -1, p.createVisualShape(p.GEOM_SPHERE, radius=0.006,
                                                                   rgbaColor=[0.1, 0.9, 0.2, 1], physicsClientId=c),
                                        basePosition=list(xyz), physicsClientId=c)
                text = p.addUserDebugText(label, [xyz[0], xyz[1], xyz[2] + 0.03], [0.05, 0.4, 0.1], textSize=1.0,
                                          physicsClientId=c)
                self.bodies[key] = [body, top, text]
            else:
                body, top, text = self.bodies[key]
                p.resetBasePositionAndOrientation(body, [xyz[0], xyz[1], 0.0 if name != "bin" else 0.002],
                                                  [0, 0, 0, 1], physicsClientId=c)
                p.resetBasePositionAndOrientation(top, list(xyz), [0, 0, 0, 1], physicsClientId=c)
                self.bodies[key][2] = p.addUserDebugText(label, [xyz[0], xyz[1], xyz[2] + 0.03], [0.05, 0.4, 0.1],
                                                         textSize=1.0, replaceItemUniqueId=text, physicsClientId=c)
        for key in [k for k in self.bodies if k not in seen]:   # objects no longer seen
            body, top, text = self.bodies.pop(key)
            p.removeBody(body, physicsClientId=c)
            p.removeBody(top, physicsClientId=c)
            p.removeUserDebugItem(text, physicsClientId=c)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2.0)
        self.vision.close()


def reference_marks(p, client, shown=()):
    """Visual references in the TABLE frame for the sliders: a white ball at the TABLE origin, a ball at each
    other follower's base origin (arms.json) and a flat square at each ArUco marker of calibration.json."""
    sys.path.insert(0, str(ROOT))
    from control import arm_config

    def ball(xyz, rgba, radius=0.012):
        shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=client)
        p.createMultiBody(0, -1, shape, basePosition=list(xyz), physicsClientId=client)

    ball([0, 0, 0], [1, 1, 1, 1])
    p.addUserDebugText("TABLE origin", [0, 0, 0.05], [1, 1, 1], textSize=1.0, physicsClientId=client)
    for name in arm_config.followers():
        if name not in shown:
            b = arm_config.base(name)
            ball([b["x"], b["y"], 0], [0.6, 0.6, 0.6, 1])
            p.addUserDebugText(f"{name} origin", [b["x"], b["y"], 0.05], [0.3, 0.3, 0.3], textSize=1.0,
                               physicsClientId=client)
    for marker, xy in load().get("aruco", {}).get("markers", {}).items():
        square = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.0008], rgbaColor=[0.1, 0.1, 0.1, 1],
                                     physicsClientId=client)
        p.createMultiBody(0, -1, square, basePosition=[float(xy[0]), float(xy[1]), 0.0008], physicsClientId=client)
        ball([float(xy[0]), float(xy[1]), 0.002], [1, 1, 1, 1], radius=0.004)
        p.addUserDebugText(f"marker {marker}", [float(xy[0]), float(xy[1]), 0.03], [0.1, 0.1, 0.1], textSize=1.0,
                           physicsClientId=client)


def store_base(arm_id, base):
    """Write an arm's base origin (TABLE frame) into arms.json (the old file kept as arms.json.bak), and rewrite
    camera_transform.npy (camera -> the default arm's base) when the camera is calibrated."""
    sys.path.insert(0, str(ROOT))
    from control import arm_config

    data = arm_config.load()
    entry = data["followers"][arm_id].setdefault("base", {})
    old = {k: float(entry.get(k, 0.0)) for k in ("x", "y", "yaw_deg")}
    entry.update(x=round(base["x"], 4), y=round(base["y"], 4), yaw_deg=round(base["yaw_deg"], 2))
    backup = arm_config.FILE.with_suffix(".json.bak")
    backup.write_text(arm_config.FILE.read_text())
    arm_config.save(data)
    calibration = load()
    if calibration.get("camera_to_table") is not None:
        save(calibration)                     # camera_transform.npy follows the default arm's new base
    print(f"{arm_id}: base x {old['x']:+.4f} -> {entry['x']:+.4f}, y {old['y']:+.4f} -> {entry['y']:+.4f} m, "
          f"yaw {old['yaw_deg']:+.2f} -> {entry['yaw_deg']:+.2f} deg; saved {arm_config.FILE.name} "
          f"(old one: {backup.name})")


def offsets(args):
    """The fine joint offsets, by eye: the arm's torque off, PyBullet shows the URDF at the measured angles plus each
    joint's slider; move the arm by hand and set the sliders until the model matches it. s saves (into
    calibration.json), q or Esc leaves (unsaved changes dropped). --no-robot: sliders on the zero pose only;
    --reset: all offsets to 0.

    With --arm, three more sliders set where that arm's base frame origin is in the TABLE frame (arms.json "base":
    x, y [m], yaw [deg]; the model is drawn there, with the TABLE origin, the other arm's origin and the ArUco
    markers of calibration.json). To check them: put the real gripper tip on a marker (or another known point) by
    hand and move the base sliders until the model's tip (the yellow ball) is on it. s saves them into arms.json too
    (the old one is kept as arms.json.bak).

    --arm both: both followers in one window, each with its own model, offset and base sliders (prefixed with its
    name); s saves every arm. An arm that cannot be connected is shown at its zero pose with a warning.

    --objects: the camera's detected objects are shown too (their STL at the detected size, a green ball on the top
    centre, a label with x y and top), in the TABLE frame as pick and place sees them. With the real jaws closed on an
    object by hand, the model's jaws should then be around the shown object."""
    sys.path.insert(0, str(ROOT))
    from control import arm_config
    from control.fine_joint_offsets import ARM_JOINTS, load_fine_joint_offsets, own_offsets_file, save_fine_joint_offsets

    if args.arm in ("both", "all"):
        arm_ids = list(arm_config.followers())
    else:
        arm_ids = [args.arm]                       # None: the older single-arm files, no base sliders

    def store(arm_id, values):
        save_fine_joint_offsets(values, own_offsets_file(arm_id))
        if arm_id is None:
            update(arm={"joint_offsets_deg": {n: float(values[n]) for n in ARM_JOINTS}})

    if args.reset:
        names = ", ".join(a or "the arm" for a in arm_ids)
        if input(f"Type RESET to set every fine offset of {names} to 0 degrees: ").strip().upper() != "RESET":
            print("Cancelled; saved fine offsets were not changed.")
            return
        for arm_id in arm_ids:
            store(arm_id, {name: 0.0 for name in ARM_JOINTS})
        print("All saved fine offsets reset to 0 degrees.")
        return
    import pybullet as p
    from control.IK_control import connect_robot
    # The repo's model (robot_urdfs/so101/robot.urdf), the one pick and place plans with: its world_to_base z and
    # any other edit show here (the LeRobot copy that IK_control.py loads is not used).
    URDF = ROOT / "robot_urdfs" / "so101" / "robot.urdf"

    rgba = {"follower_pink": [1.0, 0.41, 0.71, 1], "follower_blue": [0.0, 0.55, 1.0, 1]}
    arms_state, objects = [], None
    client = p.connect(p.GUI)
    if client < 0:
        raise RuntimeError("Could not open the PyBullet calibration GUI")
    try:
        p.setAdditionalSearchPath(str(URDF.parent), physicsClientId=client)
        p.setGravity(0, 0, 0, physicsClientId=client)
        p.setRealTimeSimulation(0, physicsClientId=client)
        on_table = any(arm_ids)                    # models at their bases in the TABLE frame (arms.json)
        for arm_id in arm_ids:
            a = {"id": arm_id, "short": (arm_id or "").replace("follower_", ""), "robot": None,
                 "file": own_offsets_file(arm_id), "saved": load_fine_joint_offsets(arm=arm_id)}
            print(f"Fine offsets of {arm_id or 'the arm'}: saved to {a['file'].relative_to(ROOT)}")
            if not args.no_robot:
                try:
                    a["robot"] = connect_robot(arm_id)
                    a["robot"].bus.disable_torque()
                    print(f"{arm_id or 'SO-101'}: torque disabled. Move it by hand; no motor commands are sent.")
                except Exception as error:         # noqa: BLE001 - one arm missing should not stop the other
                    if len(arm_ids) == 1:
                        raise
                    print(f"WARNING: {arm_id} not connected ({error}); shown at its zero pose")
            base = arm_config.base(arm_id) if arm_id else {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
            a["base_saved"] = dict(base)
            a["model"] = p.loadURDF(str(URDF), basePosition=[base["x"], base["y"], 0.0],
                                    baseOrientation=p.getQuaternionFromEuler([0, 0, math.radians(base["yaw_deg"])]),
                                    useFixedBase=True, physicsClientId=client)
            prefix = f"{a['short']} " if len(arm_ids) > 1 else ""
            a["sliders"] = {}
            for name in ARM_JOINTS:
                span = max(args.offset_range_deg, abs(a["saved"][name]) + 0.5)
                a["sliders"][name] = p.addUserDebugParameter(f"{prefix}{name} fine offset (deg)", -span, span,
                                                             a["saved"][name], physicsClientId=client)
            a["base_sliders"] = {}
            if arm_id:
                r, ry = args.base_range, args.base_yaw_range
                a["base_sliders"] = {
                    "x": p.addUserDebugParameter(f"{prefix}base x in TABLE (m)", base["x"] - r, base["x"] + r,
                                                 base["x"], physicsClientId=client),
                    "y": p.addUserDebugParameter(f"{prefix}base y in TABLE (m)", base["y"] - r, base["y"] + r,
                                                 base["y"], physicsClientId=client),
                    "yaw_deg": p.addUserDebugParameter(f"{prefix}base yaw (deg)", base["yaw_deg"] - ry,
                                                       base["yaw_deg"] + ry, base["yaw_deg"], physicsClientId=client),
                }
            model = a["model"]
            a["index"] = {p.getJointInfo(model, i, physicsClientId=client)[1].decode(): i
                          for i in range(p.getNumJoints(model, physicsClientId=client))}
            a["tip_link"] = next(i for i in range(p.getNumJoints(model, physicsClientId=client))
                                 if p.getJointInfo(model, i, physicsClientId=client)[12].decode() == "gripper_frame_link")
            a["grip"] = p.getJointInfo(model, a["index"]["gripper"], physicsClientId=client)
            a["tip_ball"] = p.createMultiBody(0, -1, p.createVisualShape(
                p.GEOM_SPHERE, radius=0.007, rgbaColor=[1, 0.9, 0, 1], physicsClientId=client), physicsClientId=client)
            a["origin_ball"] = p.createMultiBody(0, -1, p.createVisualShape(
                p.GEOM_SPHERE, radius=0.012, rgbaColor=rgba.get(arm_id, [1, 0.8, 0, 1]), physicsClientId=client),
                physicsClientId=client)
            a["status"] = -1
            arms_state.append(a)
        if not args.no_robot and all(a["robot"] is None for a in arms_state):
            print("No arm connected: zero poses only.")
        if args.no_robot:
            print("Offline preview: using zero arm poses; no robot connected.")

        table = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.75, 0.9, 0.02], rgbaColor=[0.55, 0.57, 0.60, 1],
                                    physicsClientId=client)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=table,
                          basePosition=[0.25, 0.0, -0.02] if on_table else [0, 0.25, -0.02], physicsClientId=client)
        if on_table:
            reference_marks(p, client, shown=arm_ids)
        middle = np.mean([[a["base_saved"]["x"], a["base_saved"]["y"]] for a in arms_state], axis=0)
        p.resetDebugVisualizerCamera(1.65 if len(arm_ids) == 1 else 1.9, 45, -45, [middle[0] + 0.15, middle[1], 0.20],
                                     physicsClientId=client)
        if args.objects:
            print("Starting the camera: the detected objects are shown in the TABLE frame ...")
            objects = LiveObjects(p, client)
        print("Adjust the sliders: s saves, q or Esc leaves (unsaved changes dropped).")
        while p.isConnected(client):
            any_unsaved = False
            for row, a in enumerate(arms_state):
                model, robot = a["model"], a["robot"]
                if robot is None:
                    raw, grip_pos = {name: 0.0 for name in ARM_JOINTS}, 50.0
                else:
                    obs = robot.get_observation()
                    raw = {name: float(obs[f"{name}.pos"]) for name in ARM_JOINTS}
                    grip_pos = float(obs["gripper.pos"])
                a["now"] = now = {name: float(p.readUserDebugParameter(a["sliders"][name], physicsClientId=client))
                                  for name in ARM_JOINTS}
                a["base"] = base = ({k: float(p.readUserDebugParameter(v, physicsClientId=client))
                                     for k, v in a["base_sliders"].items()} if a["base_sliders"]
                                    else dict(a["base_saved"]))
                p.resetBasePositionAndOrientation(model, [base["x"], base["y"], 0.0],
                                                  p.getQuaternionFromEuler([0, 0, math.radians(base["yaw_deg"])]),
                                                  physicsClientId=client)
                p.resetBasePositionAndOrientation(a["origin_ball"], [base["x"], base["y"], 0.0], [0, 0, 0, 1],
                                                  physicsClientId=client)
                for name in ARM_JOINTS:
                    p.resetJointState(model, a["index"][name], math.radians(raw[name] + now[name]),
                                      physicsClientId=client)
                grip = a["grip"]
                p.resetJointState(model, a["index"]["gripper"],
                                  float(grip[8] + np.clip(grip_pos / 100.0, 0.0, 1.0) * (grip[9] - grip[8])),
                                  physicsClientId=client)
                tip = p.getLinkState(model, a["tip_link"], computeForwardKinematics=True, physicsClientId=client)[4]
                p.resetBasePositionAndOrientation(a["tip_ball"], tip, [0, 0, 0, 1], physicsClientId=client)
                a["base_unsaved"] = bool(a["base_sliders"]) and not np.allclose(
                    [base[k] for k in ("x", "y", "yaw_deg")], [a["base_saved"][k] for k in ("x", "y", "yaw_deg")],
                    atol=1e-4)
                a["unsaved"] = a["base_unsaved"] or not np.allclose(
                    [now[n] for n in ARM_JOINTS], [a["saved"][n] for n in ARM_JOINTS], atol=0.01)
                any_unsaved |= a["unsaved"]
                where = "TABLE" if on_table else "base"
                a["status"] = p.addUserDebugText(
                    (f"{a['id']}: " if a["id"] else "") + "offsets (deg) "
                    + ", ".join(f"{n}={now[n]:+.2f}" for n in ARM_JOINTS)
                    + (f" | base x {base['x']:+.4f} y {base['y']:+.4f} yaw {base['yaw_deg']:+.2f}"
                       if a["base_sliders"] else "")
                    + f" | tip ({where})={np.round(tip, 3)} m | " + ("UNSAVED" if a["unsaved"] else "SAVED")
                    + ("" if robot is not None or args.no_robot else " | NOT CONNECTED"),
                    [middle[0], middle[1], 0.75 - 0.06 * row], textColorRGB=[0.05, 0.05, 0.05], textSize=1.1,
                    lifeTime=0, replaceItemUniqueId=a["status"], physicsClientId=client)
            if objects is not None:
                objects.update()
            p.stepSimulation(physicsClientId=client)
            keys = p.getKeyboardEvents(physicsClientId=client)
            if ord("s") in keys and keys[ord("s")] & p.KEY_WAS_TRIGGERED:
                for a in arms_state:
                    store(a["id"], a["now"])
                    a["saved"] = dict(a["now"])
                    print(f"Saved fine offsets of {a['id'] or 'the arm'} (deg):",
                          {n: round(a["now"][n], 2) for n in ARM_JOINTS})
                    if a["base_unsaved"]:
                        store_base(a["id"], a["base"])
                        a["base_saved"] = dict(a["base"])
            if any(k in keys and keys[k] & p.KEY_WAS_TRIGGERED for k in (ord("q"), 27)):
                if any_unsaved:
                    print("Left; the unsaved slider changes were dropped.")
                break
            time.sleep(0.02)
    finally:
        if objects is not None:
            objects.close()
        for a in arms_state:
            if a["robot"] is not None:
                a["robot"].disconnect()
        if p.isConnected(client):
            p.disconnect(physicsClientId=client)


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", default="aruco", choices=["aruco", "ids", "arm", "check", "offsets", "show"])
    ap.add_argument("--marker-xy", nargs="+", metavar="ID:X,Y",
                    help="aruco: each marker's centre in the TABLE frame (m), e.g. 3:0.45,-0.30 (the first time)")
    ap.add_argument("--marker-ids", help="aruco: TOP_RIGHT,BOTTOM_RIGHT,BOTTOM_LEFT,TOP_LEFT IDs (the first time)")
    ap.add_argument("--show-ids", action="store_true", help="the same as the mode ids")
    ap.add_argument("--points", type=int, default=4, help="arm: how many block positions")
    ap.add_argument("--frames", type=int, default=10, help="aruco / arm: camera frames to average")
    ap.add_argument("--serial", type=int, help="the ZED's serial number (default: the first camera)")
    ap.add_argument("--yes", action="store_true", help="save without asking")
    ap.add_argument("--port", help="check: the arm's port (default: found by so101ik); 'sim' for no arm")
    ap.add_argument("--no-offsets", action="store_true", help="check: without the fine joint offsets")
    ap.add_argument("--no-robot", action="store_true", help="offsets: sliders on the zero pose, no arm")
    ap.add_argument("--arm", default=None,
                    help="offsets: which arm (follower_pink, follower_blue, or both: the two in one window); its own file")
    ap.add_argument("--reset", action="store_true", help="offsets: set every fine offset to 0")
    ap.add_argument("--offset-range-deg", type=float, default=10.0, help="offsets: the sliders' +- range")
    ap.add_argument("--objects", action="store_true",
                    help="offsets: also show the camera's detected objects (ZED + YOLO, as pick_and_place sees them)")
    ap.add_argument("--base-range", type=float, default=0.05,
                    help="offsets --arm: the base x / y sliders' +- range around arms.json [m]")
    ap.add_argument("--base-yaw-range", type=float, default=10.0,
                    help="offsets --arm: the base yaw slider's +- range around arms.json [deg]")
    args = ap.parse_args()
    if args.show_ids:
        args.what = "ids"
    {"aruco": aruco, "ids": ids, "arm": arm, "check": check, "offsets": offsets, "show": show}[args.what](args)


if __name__ == "__main__":
    main()
