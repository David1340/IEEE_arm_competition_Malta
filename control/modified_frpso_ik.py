"""Modified Fully Resampled PSO inverse kinematics using PyBullet.

This module is simulation/planning code only; it does not connect to a real robot.
It evaluates end-effector pose with PyBullet forward kinematics and rejects
joint configurations that collide with the supplied scene bodies.

Minimal setup:
    import pybullet as p
    from control.modified_frpso_ik import ModifiedFRPSOIK, load_robot_for_collision

    client = p.connect(p.DIRECT)
    robot = load_robot_for_collision("robot.urdf", physics_client_id=client)
    planner = ModifiedFRPSOIK(robot, ee_link_index=5, joint_indices=[0, 1, 2, 3, 4],
                              obstacle_body_ids=[table_id], physics_client_id=client,
                              calibration_path="calibration/robots/so_follower/my_follower_arm.json")
    result = planner.solve(target_xyz, start_q=current_joint_angles)

For RRT integration, use planner.state_is_collision_free(q) for sampled states and
planner.edge_is_collision_free(q0, q1) for interpolated edges.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pybullet as p
from control.fine_joint_offsets import load_fine_joint_offsets

DEFAULT_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "calibration"
    / "robots"
    / "so_follower"
    / "my_follower_arm.json"
)


@dataclass
class IKResult:
    joint_positions: np.ndarray
    position_error_m: float
    orientation_error_rad: float | None
    cost: float
    iterations: int
    collision_free: bool
    success: bool


def load_robot_for_collision(
    urdf_path: str,
    *,
    base_position=(0.0, 0.0, 0.0),
    base_orientation=(0.0, 0.0, 0.0, 1.0),
    physics_client_id: int = 0,
) -> int:
    """Load a fixed-base URDF with self-collision pairs enabled."""
    flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
    return p.loadURDF(
        urdf_path,
        basePosition=base_position,
        baseOrientation=base_orientation,
        useFixedBase=True,
        flags=flags,
        physicsClientId=physics_client_id,
    )


def find_link_index(robot_id: int, link_name: str, physics_client_id: int = 0) -> int:
    """Find a PyBullet link index by its URDF link name."""
    for joint_index in range(p.getNumJoints(robot_id, physicsClientId=physics_client_id)):
        info = p.getJointInfo(robot_id, joint_index, physicsClientId=physics_client_id)
        if info[12].decode() == link_name:
            return joint_index
    raise ValueError(f"Link {link_name!r} not found on PyBullet body {robot_id}")


class ModifiedFRPSOIK:
    """M-FRPSO endpoint IK solver with PyBullet FK and collision checks.

    Particles are joint vectors. At each generation, Gaussian samples are
    resampled around elite solutions and the remaining particles are uniform.
    The M-FRPSO spread is sigma = best_cost / tau, scaled by each joint range.
    Colliding samples receive infinite cost and are excluded from the elite set.
    """

    def __init__(
        self,
        robot_id: int,
        ee_link_index: int,
        joint_indices: Sequence[int],
        obstacle_body_ids: Sequence[int] = (),
        *,
        physics_client_id: int = 0,
        collision_margin_m: float = 0.005,
        check_self_collision: bool = True,
        calibration_path: str | Path | None = DEFAULT_CALIBRATION_PATH,
        fixed_joint_values: Mapping[int, float] | None = None,
        allowed_robot_obstacle_link_pairs: Sequence[tuple[int, int, int]] = (),
        minimum_link_heights: Mapping[int, float] | None = None,
        arm: str | None = None,
    ):
        self.robot_id = robot_id
        self.ee_link_index = ee_link_index
        self.joint_indices = list(joint_indices)
        self.obstacle_body_ids = list(obstacle_body_ids)
        self.physics_client_id = physics_client_id
        self.collision_margin_m = float(collision_margin_m)
        self.check_self_collision = check_self_collision
        self.allowed_robot_obstacle_link_pairs = set(allowed_robot_obstacle_link_pairs)
        self.minimum_link_heights = {
            int(link_index): float(minimum_z)
            for link_index, minimum_z in (minimum_link_heights or {}).items()
        }

        limits = [p.getJointInfo(robot_id, index, physicsClientId=physics_client_id) for index in self.joint_indices]
        self.urdf_lower = np.asarray([item[8] for item in limits], dtype=float)
        self.urdf_upper = np.asarray([item[9] for item in limits], dtype=float)
        fine_offsets_rad = {
            name: math.radians(value) for name, value in load_fine_joint_offsets(arm=arm).items()
        }
        continuous = self.urdf_upper <= self.urdf_lower
        self.urdf_lower[continuous] = -math.pi
        self.urdf_upper[continuous] = math.pi
        self.lower = self.urdf_lower.copy()
        self.upper = self.urdf_upper.copy()
        if calibration_path is not None:
            calibration = json.loads(Path(calibration_path).read_text())
            for offset, item in enumerate(limits):
                joint_name = item[1].decode()
                if joint_name not in calibration:
                    raise ValueError(f"Calibration has no limits for planned joint {joint_name!r}")
                joint_calibration = calibration[joint_name]
                range_min = float(joint_calibration["range_min"])
                range_max = float(joint_calibration["range_max"])
                midpoint = (range_min + range_max) / 2.0
                offset_rad = fine_offsets_rad.get(joint_name, 0.0)
                calibrated_lower = math.radians((range_min - midpoint) * 360.0 / 4095.0) + offset_rad
                calibrated_upper = math.radians((range_max - midpoint) * 360.0 / 4095.0) + offset_rad
                self.lower[offset] = max(self.lower[offset], calibrated_lower)
                self.upper[offset] = min(self.upper[offset], calibrated_upper)
        if np.any(self.upper <= self.lower):
            raise ValueError("Every planned joint must have a non-empty range")
        self.ranges = self.upper - self.lower
        self.fixed_joint_offsets = {}
        for joint_index, value in (fixed_joint_values or {}).items():
            if joint_index not in self.joint_indices:
                raise ValueError(f"Fixed joint {joint_index} is not included in joint_indices")
            offset = self.joint_indices.index(joint_index)
            value = float(value)
            if not np.isfinite(value) or not self.lower[offset] <= value <= self.upper[offset]:
                raise ValueError(f"Fixed value for joint {joint_index} is outside its limits")
            self.fixed_joint_offsets[offset] = value

        parents = set()
        for joint_index in range(p.getNumJoints(robot_id, physicsClientId=physics_client_id)):
            parent_index = p.getJointInfo(robot_id, joint_index, physicsClientId=physics_client_id)[16]
            parents.add(frozenset((joint_index, parent_index)))
        self._parent_link_pairs = parents
        self._movable_joint_indices = [
            joint_index for joint_index in range(p.getNumJoints(robot_id, physicsClientId=physics_client_id))
            if p.getJointInfo(robot_id, joint_index, physicsClientId=physics_client_id)[2] != p.JOINT_FIXED
        ]

    @staticmethod
    def _orientation_error(target_quaternion, actual_quaternion) -> float:
        target = np.asarray(target_quaternion, dtype=float)
        actual = np.asarray(actual_quaternion, dtype=float)
        target /= np.linalg.norm(target)
        actual /= np.linalg.norm(actual)
        dot = float(np.clip(abs(np.dot(target, actual)), 0.0, 1.0))
        return 2.0 * math.acos(dot)

    def _set_joint_state(self, q: Sequence[float]) -> None:
        for joint_index, value in zip(self.joint_indices, q):
            p.resetJointState(
                self.robot_id,
                joint_index,
                float(value),
                physicsClientId=self.physics_client_id,
            )

    def _apply_fixed_joint_values(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).copy()
        for offset, value in self.fixed_joint_offsets.items():
            q[..., offset] = value
        return q

    def state_is_collision_free(
        self,
        q: Sequence[float],
        allowed_collision_body_ids: Sequence[int] = (),
        *,
        enforce_joint_limits: bool = True,
        check_joint_limits: bool = True,
    ) -> bool:
        """Check one joint state against scene bodies and optionally robot self-collision."""
        q = np.asarray(q, dtype=float)
        lower = self.lower if enforce_joint_limits else self.urdf_lower
        upper = self.upper if enforce_joint_limits else self.urdf_upper
        if q.shape != lower.shape or not np.isfinite(q).all():
            return False
        if check_joint_limits and (np.any(q < lower) or np.any(q > upper)):
            return False
        if any(not math.isclose(q[offset], value, rel_tol=0.0, abs_tol=1e-9)
               for offset, value in self.fixed_joint_offsets.items()):
            return False

        self._set_joint_state(q)
        for link_index, minimum_z in self.minimum_link_heights.items():
            lower_aabb, _ = p.getAABB(
                self.robot_id,
                linkIndex=link_index,
                physicsClientId=self.physics_client_id,
            )
            if lower_aabb[2] < minimum_z:
                return False
        p.performCollisionDetection(physicsClientId=self.physics_client_id)
        allowed = set(allowed_collision_body_ids)
        for body_id in self.obstacle_body_ids:
            if body_id in allowed:
                continue
            contact_points = p.getClosestPoints(
                self.robot_id,
                body_id,
                distance=self.collision_margin_m,
                physicsClientId=self.physics_client_id,
            )
            for point in contact_points:
                if (body_id, point[3], point[4]) not in self.allowed_robot_obstacle_link_pairs:
                    return False

        if self.check_self_collision:
            for point in p.getClosestPoints(
                self.robot_id,
                self.robot_id,
                distance=self.collision_margin_m,
                physicsClientId=self.physics_client_id,
            ):
                link_a, link_b = point[3], point[4]
                if link_a == link_b or frozenset((link_a, link_b)) in self._parent_link_pairs:
                    continue
                return False
        return True

    def edge_is_collision_free(
        self,
        q_start: Sequence[float],
        q_end: Sequence[float],
        *,
        max_joint_step_rad: float = math.radians(3.0),
        allowed_collision_body_ids: Sequence[int] = (),
        enforce_joint_limits: bool = True,
        check_joint_limits: bool = True,
        free_states: Sequence[Sequence[float]] = (),
    ) -> bool:
        """Check interpolated states along an edge; useful as an RRT edge validator.

        States equal to one of free_states (e.g. a tree root) are accepted without any check.
        """
        start = np.asarray(q_start, dtype=float)
        end = np.asarray(q_end, dtype=float)
        steps = max(1, int(np.ceil(np.max(np.abs(end - start)) / max_joint_step_rad)))
        for fraction in np.linspace(0.0, 1.0, steps + 1):
            q = start + fraction * (end - start)
            if any(np.allclose(q, free, atol=1e-9, rtol=0.0) for free in free_states):
                continue
            if not self.state_is_collision_free(
                q,
                allowed_collision_body_ids,
                enforce_joint_limits=enforce_joint_limits,
                check_joint_limits=check_joint_limits,
            ):
                return False
        return True

    def _pose_errors(self, target_position, target_orientation):
        link_state = p.getLinkState(
            self.robot_id,
            self.ee_link_index,
            computeForwardKinematics=True,
            physicsClientId=self.physics_client_id,
        )
        position_error = float(np.linalg.norm(np.asarray(link_state[4]) - target_position))
        orientation_error = None
        if target_orientation is not None:
            orientation_error = self._orientation_error(target_orientation, link_state[5])
        return position_error, orientation_error

    def _dls_attempt(self, target_position, target_orientation, start, *, position_weight, orientation_weight,
                     iterations=80, damping=0.02, max_step=0.15):
        """One damped-least-squares IK run from `start` with the PyBullet Jacobian (fixed joints excluded, joint
        limits by clipping). Returns the final joint vector; the body is left in that state."""
        q = self._apply_fixed_joint_values(np.clip(np.asarray(start, dtype=float), self.lower, self.upper))
        free = [offset for offset in range(len(q)) if offset not in self.fixed_joint_offsets]
        movable = self._movable_joint_indices
        column = [movable.index(index) for index in self.joint_indices]
        for _ in range(iterations):
            self._set_joint_state(q)
            state = p.getLinkState(self.robot_id, self.ee_link_index, computeLinkVelocity=0,
                                   computeForwardKinematics=True, physicsClientId=self.physics_client_id)
            error = [position_weight * (target_position - np.asarray(state[4]))]
            positions = [p.getJointState(self.robot_id, index, physicsClientId=self.physics_client_id)[0]
                         for index in movable]
            linear, angular = p.calculateJacobian(
                self.robot_id, self.ee_link_index, state[2], positions, [0.0] * len(movable), [0.0] * len(movable),
                physicsClientId=self.physics_client_id)
            rows = [position_weight * np.asarray(linear)[:, column][:, free]]
            if target_orientation is not None and orientation_weight > 0:
                inverse = p.invertTransform([0, 0, 0], state[5])[1]
                difference = p.multiplyTransforms([0, 0, 0], target_orientation, [0, 0, 0], inverse)[1]
                axis, angle = p.getAxisAngleFromQuaternion(difference)
                angle = (angle + math.pi) % (2 * math.pi) - math.pi
                error.append(orientation_weight * angle * np.asarray(axis))
                rows.append(orientation_weight * np.asarray(angular)[:, column][:, free])
            error, jacobian = np.concatenate(error), np.vstack(rows)
            if np.linalg.norm(error) < 1e-5:
                break
            step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping ** 2 * np.eye(len(error)), error)
            largest = np.max(np.abs(step))
            if largest > max_step:
                step *= max_step / largest
            q[free] = np.clip(q[free] + step, self.lower[free], self.upper[free])
        self._set_joint_state(q)
        return q

    def fast_solve(self, target_position, *, target_orientation=None, start_q=None, allowed_collision_body_ids=(),
                   position_weight=1.0, orientation_weight=0.1, position_tolerance_m=0.005,
                   orientation_tolerance_rad=math.radians(5.0), restarts=8, seed=0) -> IKResult | None:
        """A quick try before the swarm: damped least squares from start_q, then from random poses. Returns the
        first collision-free result within tolerance, or None (then the caller runs the full M-FRPSO)."""
        target_position = np.asarray(target_position, dtype=float)
        rng = np.random.default_rng(seed)
        starts = [np.asarray(start_q, dtype=float)] if start_q is not None else []
        starts += [rng.uniform(self.lower, self.upper) for _ in range(restarts)]
        for start in starts:
            q = self._dls_attempt(target_position, target_orientation, start, position_weight=1.0,
                                  orientation_weight=orientation_weight)
            position_error, orientation_error = self._pose_errors(target_position, target_orientation)
            if position_error > position_tolerance_m or (
                    orientation_error is not None and orientation_error > orientation_tolerance_rad):
                continue
            if not self.state_is_collision_free(q, allowed_collision_body_ids):
                continue
            self._set_joint_state(q)
            cost = position_weight * position_error + (orientation_weight * orientation_error
                                                       if orientation_error is not None else 0.0)
            return IKResult(joint_positions=q.copy(), position_error_m=position_error,
                            orientation_error_rad=orientation_error, cost=cost, iterations=0,
                            collision_free=True, success=True)
        return None

    def solve(
        self,
        target_position: Sequence[float],
        *,
        target_orientation: Sequence[float] | None = None,
        start_q: Sequence[float] | None = None,
        obstacle_body_ids: Sequence[int] | None = None,
        allowed_collision_body_ids: Sequence[int] = (),
        num_particles: int = 200,
        elite_count: int = 5,
        max_iterations: int = 200,
        tau: float = 0.5,
        position_weight: float = 1.0,
        orientation_weight: float = 0.1,
        position_tolerance_m: float = 0.005,
        orientation_tolerance_rad: float = math.radians(5.0),
        seed: int = 0,
        fast_first: bool = True,
    ) -> IKResult | None:
        """Find a collision-free joint configuration for a target end-effector pose.

        Returns the best collision-free result found, or None if no collision-free
        sample was found. The PyBullet body is left at the returned joint solution.
        """
        if num_particles < 2 or not 1 <= elite_count < num_particles:
            raise ValueError("Require num_particles >= 2 and 1 <= elite_count < num_particles")
        if tau <= 0 or max_iterations < 1:
            raise ValueError("tau and max_iterations must be positive")

        target_position = np.asarray(target_position, dtype=float)
        target_orientation = None if target_orientation is None else np.asarray(target_orientation, dtype=float)
        if target_position.shape != (3,) or not np.isfinite(target_position).all():
            raise ValueError("target_position must be a finite 3-vector")
        if target_orientation is not None and (target_orientation.shape != (4,) or not np.isfinite(target_orientation).all()):
            raise ValueError("target_orientation must be a finite PyBullet quaternion")

        obstacles = self.obstacle_body_ids if obstacle_body_ids is None else list(obstacle_body_ids)
        original_obstacles = self.obstacle_body_ids
        self.obstacle_body_ids = list(obstacles)
        rng = np.random.default_rng(seed)
        dimension = len(self.joint_indices)
        start = np.asarray(start_q, dtype=float) if start_q is not None else np.asarray([
            p.getJointState(self.robot_id, index, physicsClientId=self.physics_client_id)[0]
            for index in self.joint_indices
        ])
        if start.shape != (dimension,):
            self.obstacle_body_ids = original_obstacles
            raise ValueError(f"start_q must have shape ({dimension},)")

        if fast_first:
            quick = self.fast_solve(
                target_position, target_orientation=target_orientation, start_q=start,
                allowed_collision_body_ids=allowed_collision_body_ids, position_weight=position_weight,
                orientation_weight=orientation_weight, position_tolerance_m=position_tolerance_m,
                orientation_tolerance_rad=orientation_tolerance_rad, seed=seed,
            )
            if quick is not None:
                self.obstacle_body_ids = original_obstacles
                return quick

        population = self._apply_fixed_joint_values(
            rng.uniform(self.lower, self.upper, size=(num_particles, dimension))
        )
        population[0] = np.clip(start, self.lower, self.upper)
        population[0] = self._apply_fixed_joint_values(population[0])
        elite_archive: list[tuple[float, np.ndarray, float, float | None]] = []
        best_result = None
        uniform_count = num_particles // (elite_count + 1)

        for iteration in range(1, max_iterations + 1):
            # The pose error first (cheap forward kinematics), then the collision check (expensive) only for
            # the best candidates: only collision-free ones can become elites, and only a few elites are kept.
            scored = []
            for q in population:
                self._set_joint_state(q)
                position_error, orientation_error = self._pose_errors(target_position, target_orientation)
                cost = position_weight * position_error
                if orientation_error is not None:
                    cost += orientation_weight * orientation_error
                scored.append((cost, q.copy(), position_error, orientation_error))
            scored.sort(key=lambda result: result[0])
            worst_elite = elite_archive[-1][0] if len(elite_archive) >= elite_count else math.inf
            evaluated = []
            for candidate in scored:
                if len(evaluated) >= 2 * elite_count or candidate[0] >= worst_elite and len(evaluated) >= elite_count:
                    break
                if self.state_is_collision_free(candidate[1], allowed_collision_body_ids):
                    evaluated.append(candidate)

            if evaluated:
                evaluated.sort(key=lambda result: result[0])
                elite_archive = sorted(elite_archive + evaluated, key=lambda result: result[0])
                unique_elites = []
                for candidate in elite_archive:
                    if all(np.linalg.norm((candidate[1] - elite[1]) / self.ranges) > 1e-3 for elite in unique_elites):
                        unique_elites.append(candidate)
                    if len(unique_elites) == elite_count:
                        break
                elite_archive = unique_elites

                best_cost, best_q, position_error, orientation_error = elite_archive[0]
                candidate_result = IKResult(
                    joint_positions=best_q.copy(),
                    position_error_m=position_error,
                    orientation_error_rad=orientation_error,
                    cost=best_cost,
                    iterations=iteration,
                    collision_free=True,
                    success=(position_error <= position_tolerance_m and
                             (orientation_error is None or orientation_error <= orientation_tolerance_rad)),
                )
                if best_result is None or candidate_result.cost < best_result.cost:
                    best_result = candidate_result
                if candidate_result.success:
                    self._set_joint_state(candidate_result.joint_positions)
                    self.obstacle_body_ids = original_obstacles
                    return candidate_result

            if not elite_archive:
                population = self._apply_fixed_joint_values(
                    rng.uniform(self.lower, self.upper, size=(num_particles, dimension))
                )
                continue

            best_cost = elite_archive[0][0]
            normalized_sigma = min(1.0, best_cost / tau)
            samples = []
            for _, elite_q, _, _ in elite_archive:
                for _ in range(uniform_count):
                    sample = elite_q + rng.normal(0.0, normalized_sigma, size=dimension) * self.ranges
                    samples.append(self._apply_fixed_joint_values(np.clip(sample, self.lower, self.upper)))
                    if len(samples) == num_particles:
                        break
                if len(samples) == num_particles:
                    break
            while len(samples) < num_particles:
                samples.append(self._apply_fixed_joint_values(rng.uniform(self.lower, self.upper)))
            population = np.asarray(samples[:num_particles])

        self.obstacle_body_ids = original_obstacles
        if best_result is not None:
            self._set_joint_state(best_result.joint_positions)
        return best_result