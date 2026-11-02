"""Calibrate the camera to the arm, and keep every calibration in one file, calibration.json (robot/calibration.py).

    python robot/calibrate.py                          # after the camera moved: the ArUco markers, ~5 s, no arm
    python robot/calibrate.py --marker-ids 0,1,2,3     # the first time: which ID is TOP_RIGHT, BOTTOM_RIGHT,
                                                       # BOTTOM_LEFT, TOP_LEFT (as qr_calibrate.py); remembered
    python robot/calibrate.py --show-ids               # which marker has which ID (live view)
    python robot/calibrate.py arm                      # the most accurate: touch blocks with the arm (once);
                                                       # also corrects the stored marker positions
    python robot/calibrate.py show                     # print what calibration.json holds

The ArUco fit: the table plane from the ZED point cloud fixes the camera's tilt and height, the markers (two or
more of the stored ones) its turn and position. The markers' positions start as qr_calibrate.py's hand-measured
ones; `arm` measures where they really are in the arm's frame (a few cm off is common), so every later ArUco
calibration lands where the arm does. `arm` asks for 4 positions: one block each time, its top face's centre
found by the camera, then the tip of the fixed finger put on it by hand (torque off).

calibration.json also holds the arm's LeRobot calibration file and the fine joint offsets
(fine_joint_calibration_gui.py writes them there), and camera_transform.npy is written from it for the scripts
that read that file.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import calibration as cal  # noqa: E402

# qr_calibrate.py's hand-measured marker centres (robot base frame, m), by position on the table
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
    dictionary = cal.load().get("aruco", {}).get("dictionary", DICTIONARY)
    n = 0
    while n < frames:
        if zed.grab(rt) != sl.ERROR_CODE.SUCCESS:
            continue
        n += 1
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(cloud_mat, sl.MEASURE.XYZRGBA)
        frame = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)
        cloud = np.array(cloud_mat.get_data()[:, :, :3], np.float32)
        corners, dictionary = cal.find_markers(frame, dictionary)
        for mid, xyz in cal.marker_centres(cloud, corners).items():
            per.setdefault(mid, []).append(xyz)
    return frame, cloud, {k: np.median(v, axis=0) for k, v in per.items() if len(v) >= frames // 2}, corners, dictionary


def plane_of(cloud):
    """The table plane in the camera frame: (normal, d), from the cloud inside yolo_dataset/roi.json's region."""
    roi = None
    roi_file = cal.ROOT / "yolo_dataset" / "roi.json"
    if roi_file.is_file():
        r = json.loads(roi_file.read_text())
        sx, sy = cloud.shape[1] / r.get("ref_size", [cloud.shape[1]])[0], cloud.shape[0] / r.get("ref_size", [0, cloud.shape[0]])[1]
        roi = (round(r["x"] * sx), round(r["y"] * sy), round(r["w"] * sx), round(r["h"] * sy))
    n, d, share = cal.fit_plane(cal.table_points(cloud, roi=roi))
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
    data = cal.load()
    markers = data.get("aruco", {}).get("markers", {})
    if args.marker_ids:
        ids = [s.strip() for s in args.marker_ids.split(",")]
        if len(ids) != 4 or len(set(ids)) != 4:
            raise SystemExit("--marker-ids: four different IDs, in the order TOP_RIGHT,BOTTOM_RIGHT,BOTTOM_LEFT,TOP_LEFT")
        markers = {i: QR_POSITIONS[name] for i, name in zip(ids, QR_POSITIONS)}
        print("Marker positions from qr_calibrate.py: " + ", ".join(f"ID {i} = {n}" for i, n in zip(ids, QR_POSITIONS)))
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
        draw(frame, corners, cam, None, markers, cal.ROOT / "outputs" / "calibrate" / "aruco.jpg")
        raise SystemExit("Need 2 or more of the stored markers in view (see the picture)")
    plane = plane_of(cloud)
    T, res = cal.fit_camera(plane, [cam[m] for m in use], [markers[m] for m in use], data.get("table_z", 0.0))
    for m, r in zip(use, res):
        print(f"  marker {m}: {r:.1f} mm from its stored position after the fit")
    draw(frame, corners, cam, T, markers, cal.ROOT / "outputs" / "calibrate" / "aruco.jpg")
    old = cal.camera_to_base()
    if old is not None:
        moved = 1000 * np.linalg.norm(old[:3, 3] - T[:3, 3])
        turned = np.degrees(np.arccos(np.clip((np.trace(old[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1)))
        print(f"compared with the last calibration: camera moved {moved:.0f} mm, turned {turned:.1f} deg")
    if args.yes or input("Save to calibration.json? [Y/n] ").strip().lower() in ("", "y", "yes"):
        cal.update(camera_to_base=T.tolist(),
                   camera={"updated": cal.stamp(), "method": f"aruco ({len(use)} markers + table plane)",
                           "residual_mm": float(np.max(res)), "serial": args.serial},
                   aruco={"dictionary": dictionary, "markers": {k: list(map(float, v)) for k, v in markers.items()}})
        print(f"Saved {cal.FILE} (and {cal.LEGACY_CAMERA.name}).")


def arm(args):
    import detect
    from so101ik import Arm

    data = cal.load()
    offsets = cal.joint_offsets()
    lr = cal.lerobot_calibration()
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
        T0[:3, :3] = cal.rotation_to_z(plane[0])
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
    T, res = cal.fit_camera(plane, cam_pts, arm_pts, data.get("table_z", 0.0))
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
                    "camera": {"updated": cal.stamp(), "method": f"arm ({len(cam_pts)} blocks + table plane)",
                               "residual_mm": float(np.max(res)), "serial": args.serial}}
        if markers:
            sections["aruco"] = {"dictionary": dictionary, "markers": markers}
        cal.update(**sections)
        print(f"Saved {cal.FILE}: later camera bumps only need `python robot/calibrate.py`.")


def show_ids(args):
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
    data = cal.load()
    if not data:
        print(f"No {cal.FILE.name} yet.")
    print(json.dumps(data, indent=2))
    print(f"\nLeRobot calibration: {cal.lerobot_calibration()}")
    print("Fine joint offsets: " + ", ".join(f"{j} {v:+.2f}" for j, v in cal.joint_offsets().items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", default="aruco", choices=["aruco", "arm", "show"])
    ap.add_argument("--marker-ids", help="TOP_RIGHT,BOTTOM_RIGHT,BOTTOM_LEFT,TOP_LEFT IDs (first time)")
    ap.add_argument("--show-ids", action="store_true", help="live view of the markers' IDs")
    ap.add_argument("--points", type=int, default=4, help="arm: how many block positions")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--serial", type=int)
    ap.add_argument("--yes", action="store_true", help="save without asking")
    args = ap.parse_args()
    if args.show_ids:
        return show_ids(args)
    {"aruco": aruco, "arm": arm, "show": show}[args.what](args)


if __name__ == "__main__":
    main()
