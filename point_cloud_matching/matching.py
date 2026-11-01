"""Fit the STL model clouds to a segmented ZED point cloud.

The object rests on the table, so only 4 parameters are fitted: X, Y, yaw (rotation about
the vertical) and optionally Z (by default the model base is locked to the local table
plane). For each candidate (size x resting pose) we run a small ICP from several yaw starts
and keep the best.

Partial views: the camera only sees the faces turned towards it, so at every iteration the
model is reduced to its *visible* points (normal facing the camera; exact for convex shapes).
Matching is symmetric: data -> visible model AND visible model -> data. The second term is
what tells small from large: a big model has visible surface where there is no data.

Two modes:
  "3d"  - full XYZ distances.
  "2d"  - top-down: distances in XY only. Use when depth is unreliable (dark / blue objects
          come out much flatter than they are in the ZED cloud); it still gives X, Y and yaw.
"""

from dataclasses import dataclass

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:  # small brute-force fallback
    cKDTree = None


def _nn(ref, query):
    """Distance and index of the nearest ref point for each query point."""
    if cKDTree is not None:
        return cKDTree(ref).query(query)
    d_best = np.full(len(query), np.inf)
    i_best = np.zeros(len(query), np.int64)
    for s in range(0, len(query), 512):
        q = query[s:s + 512]
        d2 = ((q[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
        i = d2.argmin(1)
        d_best[s:s + 512] = np.sqrt(d2[np.arange(len(q)), i])
        i_best[s:s + 512] = i
    return d_best, i_best


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


@dataclass
class Pose:
    yaw: float              # radians
    t: np.ndarray           # (3,) table frame; t[2] is the model base height

    def apply(self, p):
        return p @ rot_z(self.yaw).T + self.t


@dataclass
class FitResult:
    candidate: object       # models.ModelCandidate
    pose: Pose
    rms_mm: float           # symmetric trimmed RMS distance (in the mode's metric)
    coverage: float         # visible model points with data within `inlier_mm`
    data_inliers: float     # data points within `inlier_mm` of the visible model
    mode: str
    iterations: int = 0

    @property
    def score(self):
        # Lower is better; poor coverage of the visible model is penalised.
        return self.rms_mm * (1.0 + 1.5 * (1.0 - self.coverage))

    @property
    def center(self):
        return self.pose.apply(self.candidate.com[None])[0]

    @property
    def top_center(self):
        return self.pose.apply(np.array([[0.0, 0.0, self.candidate.top_z]]))[0]


def visible_mask(points_w, normals_w, cam, min_cos=0.15):
    v = cam - points_w
    return np.einsum("ij,ij->i", normals_w, v) > min_cos * np.linalg.norm(v, axis=1)


class Matcher:
    def __init__(self, data_xyz, camera_pos, table_z, mode="3d", lock_z=True,
                 max_points=1500, trim=0.85, inlier_mm=3.0, seed=0):
        rng = np.random.default_rng(seed)
        if len(data_xyz) > max_points:
            data_xyz = data_xyz[rng.choice(len(data_xyz), max_points, replace=False)]
        self.data = np.asarray(data_xyz, float)
        self.cam = np.asarray(camera_pos, float)
        self.table_z = float(table_z)
        self.mode = mode
        self.lock_z = lock_z
        self.trim = trim
        self.inlier = inlier_mm / 1000.0
        self.w = np.array([1.0, 1.0, 1.0 if mode == "3d" else 0.0])
        self._data_w = self.data * self.w
        self._data_tree = cKDTree(self._data_w) if cKDTree is not None else None

    # --- helpers --------------------------------------------------------------------------
    def _visible(self, cand, pose):
        R = rot_z(pose.yaw)
        pw = cand.points @ R.T + pose.t
        vis = visible_mask(pw, cand.normals @ R.T, self.cam)
        return pw, vis

    def _pairs(self, cand, pose):
        pw, vis = self._visible(cand, pose)
        if vis.sum() < 10:
            return None
        mv_local, mv = cand.points[vis], pw[vis]
        d_dm, i_dm = _nn(mv * self.w, self._data_w)                 # data -> model
        if self._data_tree is not None:
            d_md, i_md = self._data_tree.query(mv * self.w)        # model -> data
        else:
            d_md, i_md = _nn(self._data_w, mv * self.w)
        return mv_local, mv, d_dm, i_dm, d_md, i_md

    def _trimmed(self, d):
        return d <= np.quantile(d, self.trim)

    def evaluate(self, cand, pose, iterations=0):
        p = self._pairs(cand, pose)
        if p is None:
            return FitResult(cand, pose, np.inf, 0.0, 0.0, self.mode, iterations)
        _, _, d_dm, _, d_md, _ = p
        k1, k2 = self._trimmed(d_dm), self._trimmed(d_md)
        rms = np.sqrt((np.sum(d_dm[k1] ** 2) + np.sum(d_md[k2] ** 2)) / (k1.sum() + k2.sum()))
        return FitResult(cand, pose, rms * 1000, float(np.mean(d_md < self.inlier)),
                         float(np.mean(d_dm < self.inlier)), self.mode, iterations)

    def residuals(self, cand, pose):
        """Per data point distance (m) to the visible model, for colouring."""
        pw, vis = self._visible(cand, pose)
        if vis.sum() == 0:
            return np.full(len(self.data), np.nan)
        return _nn(pw[vis] * self.w, self._data_w)[0]

    # --- ICP ------------------------------------------------------------------------------
    def icp(self, cand, pose, iters=30, tol=1e-5):
        for it in range(1, iters + 1):
            p = self._pairs(cand, pose)
            if p is None:
                break
            mv_local, _, d_dm, i_dm, d_md, i_md = p
            k1, k2 = self._trimmed(d_dm), self._trimmed(d_md)
            src = np.vstack((mv_local[i_dm[k1]], mv_local[k2]))       # model, local frame
            dst = np.vstack((self.data[k1], self.data[i_md[k2]]))    # data, table frame
            sc, dc = src.mean(axis=0), dst.mean(axis=0)
            s, d = src - sc, dst - dc
            yaw = np.arctan2(np.sum(s[:, 0] * d[:, 1] - s[:, 1] * d[:, 0]),
                             np.sum(s[:, 0] * d[:, 0] + s[:, 1] * d[:, 1]))
            R = rot_z(yaw)
            t = dc - R @ sc
            if self.lock_z or self.mode == "2d":
                t[2] = self.table_z
            new = Pose(yaw, t)
            step = np.linalg.norm(new.t - pose.t) + 0.02 * abs(np.angle(np.exp(1j * (new.yaw - pose.yaw))))
            pose = new
            if step < tol:
                break
        return self.evaluate(cand, pose, it)

    def fit_candidate(self, cand, yaw_starts=12, keep=2, coarse_iters=8, fine_iters=40):
        cxy = np.median(self.data[:, :2], axis=0)
        t0 = np.array([cxy[0], cxy[1], self.table_z])
        # Shift the start away from the camera: the data is the near side of the object.
        away = cxy - self.cam[:2]
        nrm = np.linalg.norm(away)
        if nrm > 1e-6:
            r = 0.5 * np.ptp(cand.points[:, :2], axis=0).mean()
            t0[:2] += away / nrm * r * 0.3
        coarse = [self.icp(cand, Pose(2 * np.pi * k / yaw_starts, t0.copy()), coarse_iters)
                  for k in range(yaw_starts)]
        coarse.sort(key=lambda f: f.score)
        fine = [self.icp(cand, f.pose, fine_iters) for f in coarse[:keep]]
        return min(fine, key=lambda f: f.score)

    def fit_all(self, candidates, **kw):
        return sorted((self.fit_candidate(c, **kw) for c in candidates), key=lambda f: f.score)
