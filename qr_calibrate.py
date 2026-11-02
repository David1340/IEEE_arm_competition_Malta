"""Calibrate the ZED-to-robot transform using four ArUco markers.

Run with ``python qr_calibrate.py --marker-ids 0,1,2,3``. IDs must be listed in
the order TOP_RIGHT, BOTTOM_RIGHT, BOTTOM_LEFT, TOP_LEFT. Set them to the IDs
actually printed on your markers; the script requires this mapping explicitly.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path


# Measured positions rotated +90 degrees counterclockwise around the table origin.
KNOWN_MARKER_POSITIONS = {
    "TOP_RIGHT": np.array([0.51, -0.42, 0.0]),
    "BOTTOM_RIGHT": np.array([0.20, -0.42, 0.0]),
    "BOTTOM_LEFT": np.array([0.20, 0.97, 0.0]),
    "TOP_LEFT": np.array([0.51, 0.97, 0.0]),
}
POSITION_ORDER = tuple(KNOWN_MARKER_POSITIONS)
ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50


def solve_rigid_transform(A, B):
    """Find 4x4 transform T such that B ~ T @ A. A,B are Nx3."""
    assert A.shape == B.shape
    cA = np.mean(A, axis=0)
    cB = np.mean(B, axis=0)
    H = (A - cA).T @ (B - cB)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cB - R @ cA
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def calibrate_from_zed(zed_perception, marker_positions):
    """Detect the configured ArUco IDs and solve the camera-to-robot transform."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    print("Point the ZED at the workspace with all 4 mapped ArUco markers visible.")
    print("Press 'c' to capture and calibrate, 'q' to quit.")

    while True:
        bgr, pc = zed_perception.grab()
        if bgr is None:
            continue

        gray = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        display = bgr.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
            found = ", ".join(str(int(marker_id)) for marker_id in ids.flatten())
            cv2.putText(display, f"ArUco IDs: {found}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("ArUco Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if ids is None:
                print("No ArUco markers detected. Check the dictionary and lighting.")
                continue

            camera_by_id = {}
            for marker_corners, marker_id_value in zip(corners, ids.flatten()):
                marker_id = int(marker_id_value)
                if marker_id not in marker_positions:
                    print(f"Unmapped ArUco ID {marker_id}, skipping")
                    continue
                cx, cy = marker_corners.reshape(4, 2).mean(axis=0)
                cam_xyz = zed_perception.pixel_to_camera(int(cx), int(cy))
                if cam_xyz is None:
                    print(f"No depth for marker {marker_id}, skipping")
                    continue
                camera_by_id[marker_id] = cam_xyz
                position_name = marker_positions[marker_id]
                robot_xyz = KNOWN_MARKER_POSITIONS[position_name]
                print(f"  ID {marker_id} ({position_name}): pixel=({cx:.0f},{cy:.0f}) "
                      f"camera={cam_xyz} robot={robot_xyz}")

            if all(marker_id in camera_by_id for marker_id in marker_positions):
                marker_ids = list(marker_positions)
                cam_points = np.array([camera_by_id[marker_id] for marker_id in marker_ids])
                robot_points = np.array([
                    KNOWN_MARKER_POSITIONS[marker_positions[marker_id]] for marker_id in marker_ids
                ])
                T = solve_rigid_transform(cam_points, robot_points)
                output_path = Path(__file__).resolve().with_name("camera_transform.npy")
                np.save(output_path, T)
                print(f"\nCalibration saved to {output_path}")
                print(f"Transform:\n{T}")
                zed_perception.T_cam_to_robot = T
                break
            else:
                missing = [marker_id for marker_id in marker_positions if marker_id not in camera_by_id]
                print(f"Missing valid marker IDs: {missing}. Try again.")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()


def show_marker_ids(zed_perception):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    print("Move or identify each marker in the live view. Press q to quit.")
    try:
        while True:
            bgr, _ = zed_perception.grab()
            if bgr is None:
                continue
            gray = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            display = bgr.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(display, corners, ids)
                for marker_corners, marker_id in zip(corners, ids.flatten()):
                    cx, cy = marker_corners.reshape(4, 2).mean(axis=0).astype(int)
                    cv2.putText(display, f"ID {int(marker_id)}", (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("ArUco ID discovery", display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--marker-ids",
        default=None,
        help="four comma-separated IDs in order TOP_RIGHT,BOTTOM_RIGHT,BOTTOM_LEFT,TOP_LEFT; e.g. 0,1,2,3",
    )
    parser.add_argument("--show-ids", action="store_true", help="open the camera and display detected marker IDs")
    args = parser.parse_args()
    if args.show_ids and args.marker_ids is not None:
        parser.error("use --show-ids by itself; do not pass --marker-ids")
    if not args.show_ids and args.marker_ids is None:
        parser.error("pass --show-ids to inspect markers, or --marker-ids after you know their positions")
    marker_positions = None
    if not args.show_ids:
        try:
            marker_ids = [int(value.strip()) for value in args.marker_ids.split(",")]
        except ValueError as exc:
            parser.error(f"--marker-ids must contain four integers: {exc}")
        if len(marker_ids) != len(POSITION_ORDER) or len(set(marker_ids)) != len(marker_ids):
            parser.error("--marker-ids must contain four distinct IDs")
        marker_positions = dict(zip(marker_ids, POSITION_ORDER))

    from unified_pipeline import ZED2

    zed = ZED2()
    try:
        if args.show_ids:
            show_marker_ids(zed)
            return

        print("ArUco ID mapping:", ", ".join(
            f"{marker_id}={position}" for marker_id, position in marker_positions.items()
        ))
        calibrate_from_zed(zed, marker_positions)
    finally:
        zed.close()


if __name__ == "__main__":
    main()
