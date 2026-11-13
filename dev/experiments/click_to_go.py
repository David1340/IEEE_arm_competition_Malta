"""Click a point in the ZED image; an arm's tip goes there.

The click is turned into a 3D point with the ZED point cloud, then into the shared TABLE frame with the ArUco
calibration (calibration.json's camera_to_table, from `python 02_calibrate.py camera`), then into the chosen
arm's base frame (follower_pink at y = +0.25, follower_blue at y = -0.25; see calibration/calibrate.py). The arm
moves with the same M-FRPSO + RRT-Connect planning as control/IK_control.py.

    python dev/experiments/click_to_go.py                  # both arms: the one nearest the click moves (g)
    python dev/experiments/click_to_go.py --arm follower_blue   # only that arm
    python dev/experiments/click_to_go.py --hover 0.08     # stop 8 cm above the clicked surface (default 5 cm)
    python dev/experiments/click_to_go.py --no-robot --gui # no arm: plan and show the move in PyBullet only

Keys in the camera window:
    left click   choose a target (shown with its base-frame x y z)
    g            go to the target
    o / c        open / close the gripper at the target on the next g (toggle)
    h            home pose (control/arm_config.py HOME_DEG, gripper HOME_GRIPPER); the arms also start there
    q / Esc      quit
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from calibration.calibrate import FILE as CALIBRATION_FILE, arm_from_table, arms, camera_to_table, open_zed  # noqa: E402

WINDOW = "click to go"
from core.arms import HOME_DEG, HOME_GRIPPER  # noqa: E402

HOME_Q = np.radians(HOME_DEG)
PATCH = 4  # pixels around the click whose point-cloud median is used


def click_to_camera_xyz(cloud, u, v, patch=PATCH):
    """Camera-frame xyz at pixel (u, v): the median of the finite cloud points in a small square around it."""
    h, w = cloud.shape[:2]
    y0, y1 = max(0, v - patch), min(h, v + patch + 1)
    x0, x1 = max(0, u - patch), min(w, u + patch + 1)
    pts = cloud[y0:y1, x0:x1, :3].reshape(-1, 3).astype(float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    return None if len(pts) < 3 else np.median(pts, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hover", type=float, default=0.05, help="metres above the clicked surface for the tip")
    ap.add_argument("--offset", type=float, nargs=3, default=[0.0, 0.0, -0.04], metavar=("X", "Y", "Z"),
                    help="added to every target before IK, as in control/IK_control.py (same default), so a "
                         "clicked target means the same as typing it at IK_control's prompt")
    ap.add_argument("--arm", default="nearest",
                    help="follower_pink, follower_blue, or nearest (default: both connected; the arm whose base is "
                         "nearest the clicked point moves)")
    ap.add_argument("--min-z", type=float, default=0.01, help="never aim the tip lower than this (base z, m)")
    ap.add_argument("--no-robot", action="store_true", help="do not connect to the arm; preview only")
    ap.add_argument("--gui", action="store_true", help="open the PyBullet visualizer")
    ap.add_argument("--free-wrist-roll", action="store_true", help="allow the planner to vary wrist roll")
    ap.add_argument("--orient-down", action="store_true", help="constrain the gripper to point down")
    ap.add_argument("--serial", type=int, help="the ZED's serial number (default: the first camera)")
    ap.add_argument("--width", type=int, default=1280, help="window width in pixels")
    from hardware.arm import add_motion_args, apply_motion_args

    add_motion_args(ap)
    args = ap.parse_args()
    apply_motion_args(args)

    T_table = camera_to_table()
    if T_table is None:
        raise SystemExit(f"No camera_to_table in {CALIBRATION_FILE}: run `python 02_calibrate.py camera "
                         "--marker-xy ...` first (see its docstring)")
    names = list(arms()) if args.arm == "nearest" else [args.arm]
    bases = {name: np.array([b["x"], b["y"]]) for name, b in arms().items() if name in names}
    if len(bases) != len(names):
        raise SystemExit(f"unknown arm {args.arm!r}; known: {', '.join(arms())}")

    import pybullet as p
    import pyzed.sl as sl
    from hardware.arm import EEController, connect_robot, follow_path, move_to, park_and_disconnect, read_joints_rad

    ctrls, robots = {}, {}
    try:
        for name in list(names):
            robot = None
            if not args.no_robot:
                try:
                    robot = connect_robot(name)
                except (ConnectionError, OSError) as error:
                    if len(names) == 1:
                        raise
                    print(f"{name}: could not connect ({str(error).strip().splitlines()[0]}); continuing without it. "
                          f"Check its port in arms.json (`python 01_setup_usb.py --check`) and the arm's power.")
                    names.remove(name)
                    bases.pop(name)
                    continue
            # PyBullet allows one GUI window: the first arm gets it
            ctrls[name] = EEController(gui=args.gui and not ctrls, lock_wrist_roll=not args.free_wrist_roll, arm=name)
            robots[name] = robot
            if robots[name]:
                print(f"{name}: going to the home pose ...")
                follow_path(robot, [read_joints_rad(robot), list(HOME_Q)], HOME_GRIPPER)
                ctrls[name].set_sim_state(read_joints_rad(robots[name]))
            else:
                ctrls[name].set_sim_state(list(HOME_Q))
            print(f"{name}: base at x {bases[name][0]:.3f} y {bases[name][1]:.3f} m (table frame)"
                  + ("" if robots[name] else ", simulation only"))
        orient = p.getQuaternionFromEuler([math.pi, 0, 0]) if args.orient_down else None
        offset = np.array(args.offset)
        zed = open_zed(args.serial)
    except BaseException:
        for robot in robots.values():
            if robot:
                park_and_disconnect(robot)
        raise

    image, cloud_mat, runtime = sl.Mat(), sl.Mat(), sl.RuntimeParameters()
    state = {"click": None, "target": None, "arm": None, "status": "click a point", "gripper": None}
    frame = cloud = None

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or cloud is None:
            return
        scale = frame.shape[1] / args.width if args.width < frame.shape[1] else 1.0
        u, v = int(round(x * scale)), int(round(y * scale))
        xyz_cam = click_to_camera_xyz(cloud, u, v)
        if xyz_cam is None:
            state.update(click=(u, v), target=None, status="no depth at that pixel: click again")
            return
        surface = T_table[:3, :3] @ xyz_cam + T_table[:3, 3]
        arm = min(bases, key=lambda name: np.linalg.norm(surface[:2] - bases[name]))
        target_table = surface + [0.0, 0.0, args.hover]
        target = (arm_from_table(arm) @ np.append(target_table, 1.0))[:3]
        target[2] = max(target[2], args.min_z)
        state.update(click=(u, v), target=target, arm=arm,
                     status=f"{arm}: x {target[0]:.3f} y {target[1]:.3f} z {target[2]:.3f} m  (g = go)")
        print(f"clicked surface at x {surface[0]:.3f} y {surface[1]:.3f} z {surface[2]:.3f} m (table frame) "
              f"-> {arm} target {np.round(target, 3)} m (its base frame)")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(WINDOW, on_mouse)
    sized = False
    print(__doc__.split("Keys in the camera window:")[1])
    try:
        while True:
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                zed.retrieve_measure(cloud_mat, sl.MEASURE.XYZRGBA)
                frame = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)
                cloud = np.array(cloud_mat.get_data()[:, :, :3], np.float32)
            if frame is None:
                cv2.waitKey(1)
                continue

            view = frame.copy()
            if state["click"] is not None:
                cv2.drawMarker(view, state["click"], (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
            grip = {None: "", 80.0: " | gripper: open", 0.0: " | gripper: close"}[state["gripper"]]
            text = state["status"] + grip
            cv2.putText(view, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
            cv2.putText(view, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if args.width < view.shape[1]:
                view = cv2.resize(view, (args.width, round(view.shape[0] * args.width / view.shape[1])))
            cv2.imshow(WINDOW, view)
            if not sized:
                cv2.resizeWindow(WINDOW, view.shape[1], view.shape[0])
                sized = True

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("o"):
                state["gripper"] = None if state["gripper"] == 80.0 else 80.0
            elif key == ord("c"):
                state["gripper"] = None if state["gripper"] == 0.0 else 0.0
            elif key == ord("h"):
                for name in names:
                    if robots[name]:
                        print(f"{name}: going home ...")
                        follow_path(robots[name], [read_joints_rad(robots[name]), list(HOME_Q)], HOME_GRIPPER)
                        ctrls[name].set_sim_state(read_joints_rad(robots[name]))
                    else:
                        ctrls[name].set_sim_state(list(HOME_Q))
                state["status"] = "home; click a point"
            elif key == ord("g"):
                if state["target"] is None:
                    state["status"] = "click a point first"
                    continue
                arm = state["arm"]
                state["status"] = f"{arm}: moving ..."
                cv2.imshow(WINDOW, view)
                cv2.waitKey(1)
                reached, final = move_to(ctrls[arm], robots[arm], state["target"] + offset,
                                         gripper=state["gripper"], orient=orient)
                if final is None:
                    state["status"] = f"{arm}: no path to that point; click another one"
                else:
                    final = final - offset
                    err = 1000 * np.linalg.norm(final - state["target"])
                    print(f"{arm}: tip at {np.round(final, 3)} m, {err:.1f} mm from the target")
                    state["status"] = (f"{arm}: {'reached' if reached else 'NOT reached'} ({err:.0f} mm off); "
                                       "click a point")
    finally:
        zed.close()
        cv2.destroyAllWindows()
        for name in names:
            if robots[name]:
                park_and_disconnect(robots[name])
            p.disconnect(physicsClientId=ctrls[name].client)

if __name__ == "__main__":
    main()
