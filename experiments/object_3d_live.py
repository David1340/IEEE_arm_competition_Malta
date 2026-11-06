"""Show YOLO objects with calibrated 3D centers from the YOLO mask and ZED point cloud.

Coordinates are transformed into the robot/table frame defined by the known
ArUco marker positions in qr_calibrate.py. The center is the mean of valid
point-cloud samples across the full segmented object mask, after depth outlier rejection.

With the silhouette fit on (default) the STL is fitted to each YOLO mask by render-and-compare
(vision/silhouette_live.py) and the reported center is the CENTRE OF THE TOP FACE
of the fitted STL; the green outline is that top face. Objects the fit cannot match (IoU below
--silhouette-min-iou) keep the point-cloud center. --no-silhouette restores the old behaviour.

Both followers are shown in the PyBullet twin at their bases in the shared TABLE frame (calibration/calibrate.py:
follower_pink at y = +0.25, follower_blue at y = -0.25), each read live (torque off) from the port in
arms.json. An arm that cannot be connected is left at its zero pose with a warning.

Usage:
    python experiments/object_3d_live.py
    python experiments/object_3d_live.py --arm follower_blue       # only one of the arms
    python experiments/object_3d_live.py --weights runs/segment/train/weights/best.pt
    python experiments/object_3d_live.py --serial 27500226
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.detection import IGNORED_OBJECT_CLASSES, OBJECT_STL_FILES  # noqa: E402
from vision.live import COLORS, Vision, add_vision_args  # noqa: E402


class PyBulletTwin:
    def __init__(self, joint_names, bases, *, gui=True):
        """bases: {arm name: (x, y, yaw_deg)} in the TABLE frame; one robot model per arm."""
        import pybullet as p
        import pybullet_data
        from control.IK_control import EE_LINK, URDF
        from control.modified_frpso_ik import load_robot_for_collision

        self.p = p
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        if self.client < 0:
            raise RuntimeError("Could not connect to PyBullet")
        try:
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
            p.setAdditionalSearchPath(str(URDF.parent), physicsClientId=self.client)
            p.setGravity(0, 0, 0, physicsClientId=self.client)
            p.setRealTimeSimulation(0, physicsClientId=self.client)

            table_half_extents = [0.75, 0.9, 0.02]
            table_collision = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=table_half_extents, physicsClientId=self.client
            )
            table_visual = p.createVisualShape(
                p.GEOM_BOX, halfExtents=table_half_extents, rgbaColor=[0.55, 0.57, 0.60, 1],
                physicsClientId=self.client,
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=table_collision,
                baseVisualShapeIndex=table_visual,
                basePosition=[0.25, 0.0, -table_half_extents[2]],
                physicsClientId=self.client,
            )
            flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
            self.robot_ids = {}
            for name, (x, y, yaw_deg) in bases.items():
                self.robot_ids[name] = p.loadURDF(
                    str(URDF), basePosition=[x, y, 0.0],
                    baseOrientation=p.getQuaternionFromEuler([0, 0, np.radians(yaw_deg)]),
                    useFixedBase=True, flags=flags, physicsClientId=self.client,
                )
            self.robot_id = next(iter(self.robot_ids.values()))
            self.joint_names = list(joint_names)
            self.joint_indices = {}
            self.link_indices = {}
            for index in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
                info = p.getJointInfo(self.robot_id, index, physicsClientId=self.client)
                self.joint_indices[info[1].decode()] = index
                self.link_indices[info[12].decode()] = index
            self.ee_link_index = self.link_indices[EE_LINK]
            self.mesh_shapes = {}
            self.object_bodies = {}
            self.object_markers = {}
            self.object_labels = {}
            self.gui = gui
            if gui:
                p.resetDebugVisualizerCamera(
                    1.65, 45, -45, [0.25, 0.0, 0.15], physicsClientId=self.client
                )
        except Exception:
            p.disconnect(physicsClientId=self.client)
            raise

    def _object_shape(self, class_name):
        class_name = class_name.strip().casefold()
        mesh_path = OBJECT_STL_FILES.get(class_name)
        if mesh_path is None:
            return None
        if not mesh_path.exists():
            raise FileNotFoundError(f"STL asset missing for {class_name}: {mesh_path}")
        if class_name not in self.mesh_shapes:
            color = COLORS[len(self.mesh_shapes) % len(COLORS)]
            rgba = [value / 255.0 for value in color[::-1]] + [0.85]
            mesh_scale = [0.001, 0.001, 0.001]
            self.mesh_shapes[class_name] = (
                self.p.createCollisionShape(
                    self.p.GEOM_MESH, fileName=str(mesh_path), meshScale=mesh_scale,
                    physicsClientId=self.client,
                ),
                self.p.createVisualShape(
                    self.p.GEOM_MESH, fileName=str(mesh_path), meshScale=mesh_scale,
                    rgbaColor=rgba, physicsClientId=self.client,
                ),
                rgba[:3],
            )
        return self.mesh_shapes[class_name]

    def update_objects(self, objects):
        active_keys = set()
        class_indices = {}
        for class_name, confidence, center, point_count in objects:
            normalized_class = class_name.strip().casefold()
            if center is None or normalized_class in IGNORED_OBJECT_CLASSES:
                continue
            mesh_shapes = self._object_shape(normalized_class)
            if mesh_shapes is None:
                continue
            occurrence = class_indices.get(normalized_class, 0)
            class_indices[normalized_class] = occurrence + 1
            key = (normalized_class, occurrence)
            active_keys.add(key)
            collision_shape, visual_shape, color = mesh_shapes
            position = [float(center[0]), float(center[1]), 0.0]
            label = f"{class_name} {confidence:.2f} center={np.round(center, 3)} m"
            if key not in self.object_bodies:
                self.object_bodies[key] = self.p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=position,
                    physicsClientId=self.client,
                )
                marker_shape = self.p.createVisualShape(
                    self.p.GEOM_SPHERE, radius=0.009, rgbaColor=[*color, 1],
                    physicsClientId=self.client,
                )
                self.object_markers[key] = self.p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=marker_shape,
                    basePosition=center.tolist(),
                    physicsClientId=self.client,
                )
                self.object_labels[key] = self.p.addUserDebugText(
                    label, (np.asarray(center) + [0, 0, 0.035]).tolist(),
                    textColorRGB=color, textSize=1.0, lifeTime=0,
                    physicsClientId=self.client,
                )
            else:
                self.p.resetBasePositionAndOrientation(
                    self.object_bodies[key], position, [0, 0, 0, 1], physicsClientId=self.client
                )
                self.p.resetBasePositionAndOrientation(
                    self.object_markers[key], center.tolist(), [0, 0, 0, 1], physicsClientId=self.client
                )
                self.p.addUserDebugText(
                    label, (np.asarray(center) + [0, 0, 0.035]).tolist(),
                    textColorRGB=color, textSize=1.0, lifeTime=0,
                    replaceItemUniqueId=self.object_labels[key], physicsClientId=self.client,
                )

        for key in set(self.object_bodies) - active_keys:
            self.p.removeBody(self.object_bodies.pop(key), physicsClientId=self.client)
            self.p.removeBody(self.object_markers.pop(key), physicsClientId=self.client)
            self.p.removeUserDebugItem(self.object_labels.pop(key), physicsClientId=self.client)

    def update_robot(self, joint_positions, gripper_position, arm=None):
        robot_id = self.robot_ids[arm] if arm is not None else self.robot_id
        for name, position in zip(self.joint_names, joint_positions):
            self.p.resetJointState(
                robot_id, self.joint_indices[name], float(position), physicsClientId=self.client
            )
        gripper_index = self.joint_indices["gripper"]
        info = self.p.getJointInfo(self.robot_id, gripper_index, physicsClientId=self.client)
        normalized = np.clip(float(gripper_position) / 100.0, 0.0, 1.0)
        gripper_angle = info[8] + normalized * (info[9] - info[8])
        self.p.resetJointState(robot_id, gripper_index, float(gripper_angle), physicsClientId=self.client)

    def close(self):
        if self.p.isConnected(self.client):
            self.p.disconnect(physicsClientId=self.client)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_vision_args(ap)
    ap.add_argument("--max-width", type=int, default=1400)
    ap.add_argument("--no-pybullet-gui", action="store_true", help="run the twin headless")
    ap.add_argument("--no-robot", action="store_true", help="use the URDF zero pose instead of live joint readings")
    ap.add_argument("--arm", default="both", help="follower_pink, follower_blue or both (default)")
    args = ap.parse_args()

    from calibration.calibrate import arm_from_table, arms, default_arm

    all_arms = arms()
    names = list(all_arms) if args.arm == "both" else [args.arm]
    if any(name not in all_arms for name in names):
        raise SystemExit(f"unknown arm {args.arm!r}; known: {', '.join(all_arms)} or both")
    bases = {name: (all_arms[name]["x"], all_arms[name]["y"], all_arms[name]["yaw_deg"]) for name in names}
    # Vision gives positions in the default arm's frame (camera_transform.npy); the twin shows the TABLE frame.
    table_from_default = np.linalg.inv(arm_from_table(default_arm()))

    vision = Vision(args)
    win = "YOLO + ZED 3D centers"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    last_report = 0.0
    fps, frames, fps_time = 0.0, 0, time.time()
    robots = {}
    twin = None

    try:
        from control.IK_control import ARM_JOINTS, connect_robot, read_joints_rad

        for name in [] if args.no_robot else names:
            try:
                robots[name] = connect_robot(name)
            except (ConnectionError, OSError, RuntimeError) as error:
                print(f"{name}: could not connect ({str(error).strip().splitlines()[0]}); shown at its zero pose. "
                      f"Check `python ports/setup_usb_ports.py --check` and the arm's power.")
                continue
            robots[name].bus.disable_torque()
            print(f"{name}: torque disabled for hand-guiding; joint feedback is live and no motor commands are sent.")
        twin = PyBulletTwin(ARM_JOINTS, bases, gui=not args.no_pybullet_gui)
        print("Coordinates: the shared TABLE frame of the ArUco calibration, metres: origin halfway between the "
              "arms, X forward, Y towards follower_pink, Z=0 at the table. Arm feedback is read-only; q/ESC quits.")
        while True:
            frame, cloud = vision.grab()
            canvas, objects = vision.detect(frame, cloud)
            objects = [(name, conf, None if centre is None else
                        (table_from_default @ np.append(np.asarray(centre, float), 1.0))[:3], count)
                       for name, conf, centre, count in objects]

            for name in names:
                robot = robots.get(name)
                if robot is None:
                    joint_positions = [0.0] * len(ARM_JOINTS)
                    gripper_position = 50.0
                else:
                    observation = robot.get_observation()
                    joint_positions = read_joints_rad(robot)
                    gripper_position = float(observation["gripper.pos"])
                twin.update_robot(joint_positions, gripper_position, arm=name)
            twin.update_objects(objects)
            twin.p.stepSimulation(physicsClientId=twin.client)

            frames += 1
            if time.time() - fps_time >= 1.0:
                elapsed = time.time() - fps_time
                fps, frames, fps_time = frames / elapsed, 0, time.time()
            cv2.putText(canvas, f"{fps:.1f} FPS | conf>={args.conf:.2f} | {len(objects)} objects",
                        (10, canvas.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            scale = min(1.0, args.max_width / canvas.shape[1])
            view = cv2.resize(canvas, None, fx=scale, fy=scale) if scale < 1.0 else canvas
            cv2.imshow(win, view)

            now = time.time()
            if now - last_report >= 1.0:
                print(f"\n{datetime.now():%H:%M:%S} | {len(objects)} objects")
                for index, (class_name, confidence, center, count) in enumerate(objects, 1):
                    if center is None:
                        print(f"  {index}. {class_name} conf={confidence:.2f}: depth unavailable ({count} points)")
                    else:
                        print(f"  {index}. {class_name} conf={confidence:.2f} TABLE: "
                              f"X={center[0]:+.3f} Y={center[1]:+.3f} Z={center[2]:.3f} m ({count} points)")
                last_report = now

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        if twin is not None:
            twin.close()
        for robot in robots.values():
            robot.disconnect()
        vision.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()