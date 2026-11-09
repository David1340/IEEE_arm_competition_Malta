"""Live ZED point cloud, seen from above and from the side, with every detection and its centre.

    python experiments/pointcloud_view.py                      # parallax centres (default)
    python experiments/pointcloud_view.py --centre rect        # or the point-cloud rectangle / the silhouette fit
    python experiments/pointcloud_view.py --3d                 # also a rotatable 3-D window (needs open3d)

Windows:
  camera     the ZED image with YOLO's masks and each object's centre and label (as pick_and_place sees it)
  top view   the point cloud in the TABLE frame seen from above (x forward = up, y left = left, 1 px = --scale
             mm), coloured by height (or by the camera's colours, key c). Over it:
               each object's centre (red cross) and name,
               its mask outline cast onto the table (grey) and onto its top (cyan): the gap between the two is the
               parallax the parallax method undoes,
               the ArUco markers' stored positions (white squares), the arms' bases (pink / blue), the point under
               the camera (yellow).
  side view  the same points from the side (y across, z up, z stretched x3): the table should lie on the z = 0 line;
             if it does not, the camera calibration is off (also printed: "table at z = ...").

Checks printed at the start: whether camera_transform.npy (what the vision code uses) matches calibration.json (what
calibrate.py saved) - if not, the calibration was not saved or is stale.

Keys (click a window first): c colours (height / camera), s save the windows as PNGs (outputs/screenshots/),
p save the point cloud as PLY (TABLE frame, with colours; opens in MeshLab / CloudCompare), q / Esc quit.
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from calibration.calibrate import arm_from_table, arms, camera_to_table, default_arm, load  # noqa: E402
from vision.detection import OBJECT_STL_FILES  # noqa: E402
from vision.live import BIN_CLASS, Vision, add_vision_args  # noqa: E402

ARM_BGR = {"follower_pink": (180, 105, 255), "follower_blue": (255, 140, 0)}
SHOTS = ROOT / "outputs" / "screenshots"


def to_px(xy, args):
    """TABLE-frame x, y [m] -> top-view pixel (col, row): x forward = up, y left = left."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    col = (args.y_max - xy[:, 1]) * 1000 / args.scale
    row = (args.x_max - xy[:, 0]) * 1000 / args.scale
    return np.column_stack([col, row])


def colour_by_height(z, z_max):
    t = np.clip(z / z_max, 0, 1)
    return cv2.applyColorMap((255 * t).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)


def check_calibration(vision):
    """Compare the transform the vision code uses with the one calibrate.py saved, both in the default arm's frame."""
    saved = camera_to_table()
    if saved is None:
        print("calibration.json has no camera_to_table: run python calibration/calibrate.py and save.")
        return
    saved_base = arm_from_table(default_arm()) @ saved
    used = vision.camera_to_robot
    moved = 1000 * np.linalg.norm(saved_base[:3, 3] - used[:3, 3])
    turned = np.degrees(np.arccos(np.clip((np.trace(saved_base[:3, :3].T @ used[:3, :3]) - 1) / 2, -1, 1)))
    state = "OK" if moved < 3 and turned < 0.5 else "DIFFERENT: re-run calibrate.py and save (y)"
    print(f"camera_transform.npy vs calibration.json: {moved:.1f} mm, {turned:.2f} deg apart -> {state}")
    print(f"camera at x {used[0, 3]:.3f} y {used[1, 3]:.3f} z {used[2, 3]:.3f} m ({default_arm()} base frame)")


def save_ply(path, xyz, bgr):
    header = (f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\n"
              f"property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    rows = np.column_stack([xyz, bgr[:, ::-1]])
    with open(path, "w") as f:
        f.write(header)
        np.savetxt(f, rows, fmt="%.4f %.4f %.4f %d %d %d")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_vision_args(ap)
    ap.add_argument("--scale", type=float, default=1.5, help="top view: mm per pixel")
    ap.add_argument("--x-min", type=float, default=-0.05)
    ap.add_argument("--x-max", type=float, default=0.70)
    ap.add_argument("--y-min", type=float, default=-0.55)
    ap.add_argument("--y-max", type=float, default=0.55)
    ap.add_argument("--z-max", type=float, default=0.08, help="height colour scale top [m]")
    ap.add_argument("--step", type=int, default=2, help="use every n-th pixel of the cloud")
    ap.add_argument("--3d", dest="three_d", action="store_true", help="also an open3d 3-D window")
    args = ap.parse_args()

    table_from_default = np.linalg.inv(arm_from_table(default_arm()))
    vision = Vision(args)
    check_calibration(vision)
    cam_to_table = table_from_default @ vision.camera_to_robot
    markers = load().get("aruco", {}).get("markers", {})
    bases = arms()
    nadir = cam_to_table[:2, 3]

    o3d = vis = pcd = None
    if args.three_d:
        try:
            import open3d as o3d
            vis = o3d.visualization.Visualizer()
            vis.create_window("point cloud (TABLE frame)", 1000, 700)
            pcd = o3d.geometry.PointCloud()
            vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(0.1))
        except ImportError:
            print("--3d needs open3d (pip install open3d); showing the top and side views only")
            args.three_d = False

    width = int((args.y_max - args.y_min) * 1000 / args.scale)
    height = int((args.x_max - args.x_min) * 1000 / args.scale)
    side_h = 300
    by_height, first, pcd_added = True, True, False
    try:
        while True:
            frame, cloud = vision.grab()
            canvas, objects = vision.detect(frame, cloud)
            # the cloud in the TABLE frame
            sub = cloud[::args.step, ::args.step, :3].reshape(-1, 3).astype(np.float64)
            rgb = frame[::args.step, ::args.step].reshape(-1, 3)
            ok = np.isfinite(sub).all(axis=1)
            xyz = sub[ok] @ cam_to_table[:3, :3].T + cam_to_table[:3, 3]
            rgb = rgb[ok]
            inside = ((xyz[:, 0] > args.x_min) & (xyz[:, 0] < args.x_max) & (xyz[:, 1] > args.y_min)
                      & (xyz[:, 1] < args.y_max) & (xyz[:, 2] > -0.15) & (xyz[:, 2] < 0.3))
            xyz, rgb = xyz[inside], rgb[inside]
            colours = colour_by_height(xyz[:, 2], args.z_max) if by_height else rgb
            table_z = float(np.median(xyz[:, 2])) if len(xyz) else float("nan")   # mostly table
            if first:
                print(f"table at z = {1000 * table_z:+.1f} mm (median of the cloud; should be ~0)")
                first = False

            # top view: far points first, so the higher ones are drawn over them
            top = np.full((height, width, 3), 30, np.uint8)
            order = np.argsort(xyz[:, 2])
            px = np.rint(to_px(xyz[order, :2], args)).astype(int)
            good = (px[:, 0] >= 0) & (px[:, 0] < width) & (px[:, 1] >= 0) & (px[:, 1] < height)
            top[px[good, 1], px[good, 0]] = colours[order][good]
            for mid, pos in markers.items():
                c = tuple(np.rint(to_px(pos[:2], args)[0]).astype(int))
                cv2.rectangle(top, (c[0] - 8, c[1] - 8), (c[0] + 8, c[1] + 8), (255, 255, 255), 1)
                cv2.putText(top, f"ID {mid}", (c[0] + 10, c[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            for name, base in bases.items():
                c = tuple(np.rint(to_px([base["x"], base["y"]], args)[0]).astype(int))
                cv2.circle(top, c, 10, ARM_BGR.get(name, (200, 200, 200)), 2)
                cv2.putText(top, name, (c[0] + 12, c[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            ARM_BGR.get(name, (200, 200, 200)), 1)
            c = tuple(np.rint(to_px(nadir, args)[0]).astype(int))
            cv2.drawMarker(top, c, (0, 255, 255), cv2.MARKER_DIAMOND, 14, 2)
            for (name, conf, centre, _), polygon in zip(objects, vision.last_polygons):
                key = name.casefold()
                if key in OBJECT_STL_FILES and key != BIN_CLASS:
                    from control.gripper import object_height

                    h = object_height(OBJECT_STL_FILES[key])
                    for z, colour in ((0.0, (150, 150, 150)), (h, (255, 255, 0))):   # outline on table / on top
                        outline = np.array([vision.cam.ray_to_plane(u, v, z) for u, v in
                                            np.asarray(polygon, float).reshape(-1, 2)])
                        outline = (table_from_default @ np.column_stack([outline, np.ones(len(outline))]).T).T
                        pts = np.rint(to_px(outline[:, :2], args)).astype(np.int32)
                        cv2.polylines(top, [pts], True, colour, 1)
                if centre is None:
                    continue
                ct = (table_from_default @ np.append(centre, 1.0))[:3]
                c = tuple(np.rint(to_px(ct[:2], args)[0]).astype(int))
                cv2.drawMarker(top, c, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
                cv2.putText(top, f"{name} {ct[0]:+.3f} {ct[1]:+.3f}", (c[0] + 10, c[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(top, f"table z {1000 * table_z:+.1f} mm | c colours  s save  p ply  q quit",
                        (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # side view: y across, z up (x3)
            side = np.full((side_h, width, 3), 30, np.uint8)
            zero_row = side_h - 40
            sx = np.rint((args.y_max - xyz[:, 1]) * 1000 / args.scale).astype(int)
            sy = np.rint(zero_row - xyz[:, 2] * 1000 / args.scale * 3).astype(int)
            good = (sx >= 0) & (sx < width) & (sy >= 0) & (sy < side_h)
            side[sy[good], sx[good]] = colours[good]
            cv2.line(side, (0, zero_row), (width, zero_row), (0, 255, 0), 1)
            cv2.putText(side, "side view (y across, z up x3); green = table z 0", (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("camera", canvas)
            cv2.imshow("top view (TABLE frame)", top)
            cv2.imshow("side view", side)
            if args.three_d:
                pcd.points = o3d.utility.Vector3dVector(xyz)
                pcd.colors = o3d.utility.Vector3dVector(colours[:, ::-1] / 255.0)
                if not pcd_added:
                    vis.add_geometry(pcd)            # the first time also sets the view around the points
                    pcd_added = True
                else:
                    vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                by_height = not by_height
            if key in (ord("s"), ord("p")):
                SHOTS.mkdir(parents=True, exist_ok=True)
                stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
                if key == ord("s"):
                    for name, img in (("camera", canvas), ("top", top), ("side", side)):
                        cv2.imwrite(str(SHOTS / f"pointcloud_{stamp}_{name}.png"), img)
                    print(f"saved {SHOTS}/pointcloud_{stamp}_camera/top/side.png")
                else:
                    path = SHOTS / f"pointcloud_{stamp}.ply"
                    save_ply(path, xyz, rgb)
                    print(f"saved {path} ({len(xyz)} points, TABLE frame)")
    finally:
        vision.close()
        cv2.destroyAllWindows()
        if vis is not None:
            vis.destroy_window()


if __name__ == "__main__":
    main()
