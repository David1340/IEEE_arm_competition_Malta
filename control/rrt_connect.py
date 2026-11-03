"""Joint-space RRT-Connect using ModifiedFRPSOIK collision validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from control.modified_frpso_ik import ModifiedFRPSOIK


@dataclass
class RRTConnectResult:
    path: list[np.ndarray]
    iterations: int


@dataclass
class _Tree:
    nodes: list[np.ndarray]
    parents: list[int]
    rooted_at_start: bool


class RRTConnectPlanner:
    """Plan a collision-free joint path between two full configurations.

    Fixed joints are removed from the search space and restored when validating
    states. With wrist roll fixed, the SO-101 search is four-dimensional.
    """

    def __init__(
        self,
        collision_checker: ModifiedFRPSOIK,
        *,
        step_size: float = 0.2,
        max_iterations: int = 3000,
        max_joint_step_rad: float = np.deg2rad(1.0),
        goal_sample_rate: float = 0.05,
        shortcut_path: bool = True,
    ):
        if step_size <= 0 or max_iterations < 1 or max_joint_step_rad <= 0:
            raise ValueError("step_size, max_iterations, and max_joint_step_rad must be positive")
        if not 0.0 <= goal_sample_rate <= 1.0:
            raise ValueError("goal_sample_rate must be in [0, 1]")

        self.collision_checker = collision_checker
        self.step_size = float(step_size)
        self.max_iterations = int(max_iterations)
        self.max_joint_step_rad = float(max_joint_step_rad)
        self.goal_sample_rate = float(goal_sample_rate)
        self.shortcut_path = shortcut_path
        self.active_offsets = [
            offset for offset in range(len(collision_checker.lower))
            if offset not in collision_checker.fixed_joint_offsets
        ]
        if not self.active_offsets:
            raise ValueError("RRT-Connect requires at least one unfixed joint")
        self.lower = collision_checker.lower[self.active_offsets]
        self.upper = collision_checker.upper[self.active_offsets]
        self.ranges = self.upper - self.lower

    def _reduce(self, configuration, allow_outside_limits: bool = False) -> np.ndarray:
        configuration = np.asarray(configuration, dtype=float)
        checker = self.collision_checker
        if configuration.shape != checker.lower.shape or not np.isfinite(configuration).all():
            raise ValueError(f"Configurations must be finite vectors of length {len(checker.lower)}")
        if not allow_outside_limits and (
            np.any(configuration < checker.lower) or np.any(configuration > checker.upper)
        ):
            raise ValueError("Configuration is outside the joint limits")
        for offset, value in checker.fixed_joint_offsets.items():
            if not np.isclose(configuration[offset], value, atol=1e-6, rtol=0.0):
                raise ValueError(f"Fixed joint at offset {offset} must equal {value:.6f} rad")
        return configuration[self.active_offsets].copy()

    def _expand(self, configuration: np.ndarray) -> np.ndarray:
        full = np.empty_like(self.collision_checker.lower)
        full[self.active_offsets] = configuration
        for offset, value in self.collision_checker.fixed_joint_offsets.items():
            full[offset] = value
        return full

    def _distance(self, first: np.ndarray, second: np.ndarray) -> float:
        return float(np.linalg.norm((second - first) / self.ranges))

    def _edge_is_free(self, first: np.ndarray, second: np.ndarray, allowed_collision_body_ids) -> bool:
        return self.collision_checker.edge_is_collision_free(
            self._expand(first),
            self._expand(second),
            max_joint_step_rad=self.max_joint_step_rad,
            allowed_collision_body_ids=allowed_collision_body_ids,
            free_states=[self._expand(self._root)],
        )

    def _extend(self, tree: _Tree, target: np.ndarray, allowed_collision_body_ids):
        distances = [self._distance(node, target) for node in tree.nodes]
        nearest_index = int(np.argmin(distances))
        nearest = tree.nodes[nearest_index]
        distance = distances[nearest_index]
        if distance <= 1e-12:
            return "reached", nearest_index

        fraction = min(1.0, self.step_size / distance)
        candidate = nearest + fraction * (target - nearest)
        if not self._edge_is_free(nearest, candidate, allowed_collision_body_ids):
            return "trapped", nearest_index

        tree.nodes.append(candidate)
        tree.parents.append(nearest_index)
        new_index = len(tree.nodes) - 1
        status = "reached" if fraction >= 1.0 - 1e-12 else "advanced"
        return status, new_index

    def _connect(self, tree: _Tree, target: np.ndarray, allowed_collision_body_ids):
        while True:
            status, index = self._extend(tree, target, allowed_collision_body_ids)
            if status != "advanced":
                return status, index

    @staticmethod
    def _trace(tree: _Tree, index: int) -> list[np.ndarray]:
        path = []
        while index >= 0:
            path.append(tree.nodes[index])
            index = tree.parents[index]
        return list(reversed(path))

    def _shortcut(self, path: list[np.ndarray], allowed_collision_body_ids) -> list[np.ndarray]:
        if len(path) < 3:
            return path
        result = [path[0]]
        current = 0
        while current < len(path) - 1:
            next_index = len(path) - 1
            while next_index > current + 1 and not self._edge_is_free(
                path[current], path[next_index], allowed_collision_body_ids
            ):
                next_index -= 1
            result.append(path[next_index])
            current = next_index
        return result

    def plan(
        self,
        start_configuration,
        goal_configuration,
        *,
        allowed_collision_body_ids=(),
        seed: int = 0,
    ) -> RRTConnectResult | None:
        """Return full joint configurations from start to goal, or None if blocked."""
        # The start (root) is always considered free, even outside the joint limits or in collision.
        start = self._reduce(start_configuration, allow_outside_limits=True)
        goal = self._reduce(goal_configuration)
        checker = self.collision_checker
        allowed = tuple(allowed_collision_body_ids)
        self._root = start
        # Widen the checker's limits to include the root, so edges leaving it are not rejected.
        saved = checker.lower.copy(), checker.upper.copy()
        full_root = self._expand(start)
        checker.lower = np.minimum(checker.lower, full_root)
        checker.upper = np.maximum(checker.upper, full_root)
        try:
            return self._plan(start, goal, allowed, seed)
        finally:
            checker.lower, checker.upper = saved

    def _plan(self, start, goal, allowed, seed) -> RRTConnectResult | None:
        checker = self.collision_checker
        if not checker.state_is_collision_free(self._expand(goal), allowed):
            return None
        if self._distance(start, goal) <= 1e-12:
            return RRTConnectResult(path=[self._expand(start)], iterations=0)
        if self._edge_is_free(start, goal, allowed):
            return RRTConnectResult(
                path=[self._expand(start), self._expand(goal)],
                iterations=0,
            )

        rng = np.random.default_rng(seed)
        start_tree = _Tree([start], [-1], True)
        goal_tree = _Tree([goal], [-1], False)

        for iteration in range(1, self.max_iterations + 1):
            tree_a, tree_b = (start_tree, goal_tree) if iteration % 2 else (goal_tree, start_tree)
            if rng.random() < self.goal_sample_rate:
                target = tree_b.nodes[0]
            else:
                target = rng.uniform(self.lower, self.upper)

            status_a, index_a = self._extend(tree_a, target, allowed)
            if status_a == "trapped":
                continue
            status_b, index_b = self._connect(tree_b, tree_a.nodes[index_a], allowed)
            if status_b != "reached":
                continue

            path_a = self._trace(tree_a, index_a)
            path_b = self._trace(tree_b, index_b)
            if tree_a.rooted_at_start:
                reduced_path = path_a + list(reversed(path_b))[1:]
            else:
                reduced_path = path_b + list(reversed(path_a))[1:]

            if self.shortcut_path:
                reduced_path = self._shortcut(reduced_path, allowed)
            path = [self._expand(configuration) for configuration in reduced_path]
            return RRTConnectResult(path=path, iterations=iteration)

        return None