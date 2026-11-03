"""Every calibration of the robot, in one script; every calibration value, in one file: calibration.json.

    python calibration/calibrate.py                          # after the camera moved: the ArUco markers, a few s, no arm
    python calibration/calibrate.py --marker-ids 0,1,2,3     # the first time: which ID is TOP_RIGHT, BOTTOM_RIGHT,
                                                       # BOTTOM_LEFT, TOP_LEFT (remembered afterwards)
    python calibration/calibrate.py ids                      # live view: which marker has which ID
    python calibration/calibrate.py arm                      # the most accurate camera fit: touch 4 blocks with the arm
                                                       # (once); also re-measures the markers in the arm's frame
    python calibration/calibrate.py check                    # torque off: where the model puts the tip and the jaws while
                                                       # the arm is moved by hand
    python calibration/calibrate.py offsets                  # the fine joint offsets: PyBullet sliders over the live arm
    python calibration/calibrate.py show                     # print what calibration.json holds

calibration.json (at the repo root):
    camera_to_base     4 x 4: ZED left camera (IMAGE frame: x right, y down, z forward) -> robot base frame
                       (x forward, y to the arm's left, z up, origin under the shoulder pan axis), metres
    camera             when, how (aruco / arm) and how well it was fitted; the ZED serial
    table_z            the table's height in the base frame (m)
    aruco              the marker dictionary (DICT_4X4_50) and each marker's centre in the base frame
    arm                the follower's LeRobot calibration file, and the fine joint offsets (degrees; true angle =
                       LeRobot reading + offset; control/fine_joint_offsets.py still reads its own json; `offsets` keeps both in sync)
camera_transform.npy is written from it for the scripts that read that file (experiments/object_3d_live.py, experiments/Full_pipeline.py).

The camera fit: the table plane from the ZED point cloud fixes the camera's tilt and height (thousands of
points); two or more known points on the table (markers, or blocks touched with the fixed finger's tip) fix its
turn about z and its position. The markers start at the hand-measured positions (TOP_RIGHT 0.51 -0.42,
BOTTOM_RIGHT 0.20 -0.42, BOTTOM_LEFT 0.20 0.97, TOP_LEFT 0.51 0.97 m); `arm` measures where they really are, so
every later marker-only calibration lands where the arm does. Pictures of each run go to outputs/calibrate/.

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


# ---------------------------------------------------------------------------------------------- the camera
# The hand-measured marker centres (from the old qr_calibrate.py) (robot base frame, m), by position on the table
QR_POSITIONS = {"TOP_RIGHT": [0.51, -0.42, 0.0], "BOTTOM_RIGHT": [0.20, -0.42, 0.0],
                "BOTTOM_LEFT": [0.20, 0.97, 0.0], "TOP_LEFT": [0.51, 0.97, 0.0]}
DICTIONARY = "DICT_4X4_50"


def open_zed(serial=None):
    import pyzed.sl as sl

    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD1080
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
    if args.marker_ids:
        ids = [s.strip() for s in args.marker_ids.split(",")]
        if len(ids) != 4 or len(set(ids)) != 4:
            raise SystemExit("--marker-ids: four different IDs, in the order TOP_RIGHT,BOTTOM_RIGHT,BOTTOM_LEFT,TOP_LEFT")
        markers = {i: QR_POSITIONS[name] for i, name in zip(ids, QR_POSITIONS)}
        print("Marker positions (hand-measured): " + ", ".join(f"ID {i} = {n}" for i, n in zip(ids, QR_POSITIONS)))
    if not markers:
        raise SystemExit("No marker positions yet: run once with --marker-ids (see --show-ids for the IDs)")
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
    old = camera_to_base()
    if old is not None:
        moved = 1000 * np.linalg.norm(old[:3, 3] - T[:3, 3])
        turned = np.degrees(np.arccos(np.clip((np.trace(old[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1)))
        print(f"compared with the last calibration: camera moved {moved:.0f} mm, turned {turned:.1f} deg")
    if args.yes or input("Save to calibration.json? [Y/n] ").strip().lower() in ("", "y", "yes"):
        update(camera_to_base=T.tolist(),
                   camera={"updated": stamp(), "method": f"aruco ({len(use)} markers + table plane)",
                           "residual_mm": float(np.max(res)), "serial": args.serial},
                   aruco={"dictionary": dictionary, "markers": {k: list(map(float, v)) for k, v in markers.items()}})
        print(f"Saved {FILE} (and {LEGACY_CAMERA.name}).")


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
def offsets(args):
    """The fine joint offsets, by eye: the arm's torque off, PyBullet shows the URDF at the measured angles plus each
    joint's slider; move the arm by hand and set the sliders until the model matches it. s saves (into
    calibration.json), q or Esc leaves (unsaved changes dropped). --no-robot: sliders on the zero pose only;
    --reset: all offsets to 0."""
    sys.path.insert(0, str(ROOT))
    from control.fine_joint_offsets import ARM_JOINTS, load_fine_joint_offsets, save_fine_joint_offsets

    offsets_now = load_fine_joint_offsets()
    if args.reset:
        if input("Type RESET to set every fine offset to 0 degrees: ").strip().upper() != "RESET":
            print("Cancelled; saved fine offsets were not changed.")
            return
        save_fine_joint_offsets({name: 0.0 for name in ARM_JOINTS})
        update(arm={"joint_offsets_deg": {name: 0.0 for name in ARM_JOINTS}})
        print("All saved fine offsets reset to 0 degrees.")
        return
    import pybullet as p
    from control.IK_control import URDF, connect_robot

    robot = None
    client = p.connect(p.GUI)
    if client < 0:
        raise RuntimeError("Could not open the PyBullet calibration GUI")
    try:
        if not args.no_robot:
            robot = connect_robot()
            robot.bus.disable_torque()
            print("SO-101 torque disabled. Move the arm by hand; no motor commands will be sent.")
        else:
            print("Offline preview: using zero arm pose; robot was not connected.")
        p.setAdditionalSearchPath(str(URDF.parent), physicsClientId=client)
        p.setGravity(0, 0, 0, physicsClientId=client)
        p.setRealTimeSimulation(0, physicsClientId=client)
        robot_id = p.loadURDF(str(URDF), useFixedBase=True, physicsClientId=client)
        sliders = {}
        for name in ARM_JOINTS:
            span = max(args.offset_range_deg, abs(offsets_now[name]) + 0.5)
            sliders[name] = p.addUserDebugParameter(f"{name} fine offset (deg)", -span, span, offsets_now[name],
                                                    physicsClientId=client)
        table = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.75, 0.9, 0.02], rgbaColor=[0.55, 0.57, 0.60, 1],
                                    physicsClientId=client)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=table, basePosition=[0, 0.25, -0.02], physicsClientId=client)
        p.resetDebugVisualizerCamera(1.65, 45, -45, [0.15, 0.25, 0.20], physicsClientId=client)
        index = {p.getJointInfo(robot_id, i, physicsClientId=client)[1].decode(): i
                 for i in range(p.getNumJoints(robot_id, physicsClientId=client))}
        tip_link = next(i for i in range(p.getNumJoints(robot_id, physicsClientId=client))
                        if p.getJointInfo(robot_id, i, physicsClientId=client)[12].decode() == "gripper_frame_link")
        grip = p.getJointInfo(robot_id, index["gripper"], physicsClientId=client)
        status_id, saved = -1, offsets_now.copy()
        print("Adjust the sliders: s saves, q or Esc leaves (unsaved changes dropped).")
        while p.isConnected(client):
            if robot is None:
                raw, grip_pos = {name: 0.0 for name in ARM_JOINTS}, 50.0
            else:
                obs = robot.get_observation()
                raw, grip_pos = {name: float(obs[f"{name}.pos"]) for name in ARM_JOINTS}, float(obs["gripper.pos"])
            now = {name: float(p.readUserDebugParameter(sliders[name], physicsClientId=client)) for name in ARM_JOINTS}
            for name in ARM_JOINTS:
                p.resetJointState(robot_id, index[name], math.radians(raw[name] + now[name]), physicsClientId=client)
            p.resetJointState(robot_id, index["gripper"],
                              float(grip[8] + np.clip(grip_pos / 100.0, 0.0, 1.0) * (grip[9] - grip[8])),
                              physicsClientId=client)
            tip = p.getLinkState(robot_id, tip_link, computeForwardKinematics=True, physicsClientId=client)[4]
            unsaved = not np.allclose([now[n] for n in ARM_JOINTS], [saved[n] for n in ARM_JOINTS], atol=0.01)
            status_id = p.addUserDebugText(
                "Fine offsets (deg): " + ", ".join(f"{n}={now[n]:+.2f}" for n in ARM_JOINTS)
                + f" | tip={np.round(tip, 3)} m | " + ("UNSAVED" if unsaved else "SAVED") + " | calibration.json",
                [0.0, 0.0, 0.75], textColorRGB=[0.05, 0.05, 0.05], textSize=1.1, lifeTime=0,
                replaceItemUniqueId=status_id, physicsClientId=client)
            p.stepSimulation(physicsClientId=client)
            keys = p.getKeyboardEvents(physicsClientId=client)
            if ord("s") in keys and keys[ord("s")] & p.KEY_WAS_TRIGGERED:
                save_fine_joint_offsets(now)
                update(arm={"joint_offsets_deg": {n: float(now[n]) for n in ARM_JOINTS}})
                saved = now.copy()
                print("Saved fine offsets (deg):", {n: round(now[n], 2) for n in ARM_JOINTS})
            if any(k in keys and keys[k] & p.KEY_WAS_TRIGGERED for k in (ord("q"), 27)):
                if unsaved:
                    print("Left; the unsaved slider changes were dropped.")
                break
            time.sleep(0.02)
    finally:
        if robot is not None:
            robot.disconnect()
        if p.isConnected(client):
            p.disconnect(physicsClientId=client)


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", default="aruco", choices=["aruco", "ids", "arm", "check", "offsets", "show"])
    ap.add_argument("--marker-ids", help="aruco: TOP_RIGHT,BOTTOM_RIGHT,BOTTOM_LEFT,TOP_LEFT IDs (the first time)")
    ap.add_argument("--show-ids", action="store_true", help="the same as the mode ids")
    ap.add_argument("--points", type=int, default=4, help="arm: how many block positions")
    ap.add_argument("--frames", type=int, default=10, help="aruco / arm: camera frames to average")
    ap.add_argument("--serial", type=int, help="the ZED's serial number (default: the first camera)")
    ap.add_argument("--yes", action="store_true", help="save without asking")
    ap.add_argument("--port", help="check: the arm's port (default: found by so101ik); 'sim' for no arm")
    ap.add_argument("--no-offsets", action="store_true", help="check: without the fine joint offsets")
    ap.add_argument("--no-robot", action="store_true", help="offsets: sliders on the zero pose, no arm")
    ap.add_argument("--reset", action="store_true", help="offsets: set every fine offset to 0")
    ap.add_argument("--offset-range-deg", type=float, default=10.0, help="offsets: the sliders' +- range")
    args = ap.parse_args()
    if args.show_ids:
        args.what = "ids"
    {"aruco": aruco, "ids": ids, "arm": arm, "check": check, "offsets": offsets, "show": show}[args.what](args)


if __name__ == "__main__":
    main()
