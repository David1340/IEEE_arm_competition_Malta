"""Model point clouds from the competition STL files.

For every STL this builds one candidate per distinct *stable resting pose* (the ways the
object can lie on the table, e.g. a cylinder upright or on its side), and samples points
over its surface with outward normals. Everything here is in METRES in a local object frame:

    origin = centre of mass projected onto the table (x = y = 0), z = 0 at the table,
    +z up, so the model's 3D centre is (0, 0, com_z) and its top is at z = height.

The normals let the matcher keep only the faces the camera can actually see. All official
shapes are convex, so "faces pointing towards the camera" is exactly the visible part.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STL_DIR = ROOT / "stl"

# YOLO class name -> STL base name. The organisers' "hexagonal prism" is a pentagonal prism
# in the official 3MF; the YOLO class is "hexagonal" and the STLs are hexagonal_prism_*.stl.
CLASS_TO_STL = {
    "cube": "cube",
    "cylinder": "cylinder",
    "dodecahedron": "dodecahedron",
    "hexagonal": "hexagonal_prism",
    "hexagonal_prism": "hexagonal_prism",
    "pentagonal": "hexagonal_prism",
    "pentagonal_prism": "hexagonal_prism",
}
SIZES = ("small", "large")


def read_stl(path):
    """Binary or ASCII STL -> (n, 3, 3) triangle array in the file's units."""
    data = Path(path).read_bytes()
    if len(data) >= 84:
        n = int.from_bytes(data[80:84], "little")
        if 84 + 50 * n == len(data):
            rec = np.frombuffer(data, dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")],
                                count=n, offset=84)
            return rec["v"].astype(np.float64)
    verts = [list(map(float, line.split()[1:4])) for line in data.decode(errors="ignore").splitlines()
             if line.strip().startswith("vertex")]
    return np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3)


def _rotation_a_to_b(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if c < -1 + 1e-9:  # opposite: rotate 180 deg about any perpendicular axis
        axis = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1.0, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def _mesh_props(tris):
    """Unit outward normals, areas, volume and centre of mass of a closed mesh."""
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(area2, 1e-12)[:, None]
    signed = np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])) / 6.0
    volume = signed.sum()
    com = (signed[:, None] * tris.sum(axis=1) / 4.0).sum(axis=0) / volume
    return normals, area2 / 2, volume, com


def _point_in_triangles(p, tris2d):
    a, b, c = tris2d[:, 0], tris2d[:, 1], tris2d[:, 2]

    def side(p1, p2):
        return (p[0] - p2[:, 0]) * (p1[:, 1] - p2[:, 1]) - (p1[:, 0] - p2[:, 0]) * (p[1] - p2[:, 1])

    d1, d2, d3 = side(a, b), side(b, c), side(c, a)
    neg = (d1 < -1e-9) | (d2 < -1e-9) | (d3 < -1e-9)
    pos = (d1 > 1e-9) | (d2 > 1e-9) | (d3 > 1e-9)
    return bool(np.any(~(neg & pos)))


def stable_poses(tris):
    """Rotations that put the mesh on each face whose support polygon contains the centre of
    mass, with symmetric duplicates merged. Returns [(R, base_area)], largest base first."""
    normals, areas, _, com = _mesh_props(tris)
    keys = np.round(normals, 3)
    groups = {}
    for i, k in enumerate(map(tuple, keys)):
        groups.setdefault(k, []).append(i)
    poses = []
    for idx in groups.values():
        n = normals[idx].mean(axis=0)
        n /= np.linalg.norm(n)
        R = _rotation_a_to_b(n, np.array([0, 0, -1.0]))
        face = tris[idx] @ R.T
        if not _point_in_triangles(R @ com, face[:, :, :2]):
            continue  # would tip over
        poses.append((R, areas[idx].sum()))

    # Merge poses that look the same (height + horizontal second moments)
    def signature(R):
        v = tris.reshape(-1, 3) @ R.T
        h = np.ptp(v[:, 2])
        xy = v[:, :2] - v[:, :2].mean(axis=0)
        ev = np.sort(np.linalg.eigvalsh(xy.T @ xy / len(xy)))
        return np.array([h, *np.sqrt(ev)])

    unique = []
    for R, area in sorted(poses, key=lambda p: -p[1]):
        s = signature(R)
        for u in unique:
            if np.allclose(s, u[2], rtol=0.05, atol=2.0):  # mm; the official pentagon is slightly irregular
                u[1] += area
                break
        else:
            unique.append([R, area, s])
    return [(R, area) for R, area, _ in unique]


def sample_surface(tris, normals, spacing, rng):
    """Roughly uniform random surface samples, about one point per spacing^2."""
    areas = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    n = int(np.clip(areas.sum() / spacing ** 2, 500, 20000))
    face = rng.choice(len(tris), size=n, p=areas / areas.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    t = tris[face]
    pts = (1 - s)[:, None] * t[:, 0] + (s * (1 - r2))[:, None] * t[:, 1] + (s * r2)[:, None] * t[:, 2]
    return pts, normals[face]


@dataclass
class ModelCandidate:
    shape: str            # STL base name, e.g. "cube"
    size: str             # "small" / "large"
    pose_index: int
    pose_name: str        # "upright", "on side", ...
    points: np.ndarray    # (n, 3) metres, object frame (see module docstring)
    normals: np.ndarray   # (n, 3) unit outward normals
    tris: np.ndarray      # (m, 3, 3) metres, object frame (for drawing)
    height: float         # metres
    com: np.ndarray       # (3,) centre of mass in object frame: (0, 0, com_z)
    top_z: float          # = height
    top_tris: np.ndarray = None    # (k, 3, 3) the flat top face (upward faces at the top)
    top_center: np.ndarray = None  # (3,) area centroid of the top face, object frame
    top_edges: np.ndarray = None   # (e, 2, 3) outline of the top face, for drawing
    sym_order: int = 1             # rotational symmetry about the vertical: n-fold, 0 = any angle
    top_kind: str = "flat"         # "flat" face, or "ridge" (prism / cylinder lying on its side)

    @property
    def label(self):
        return f"{self.size} {self.shape} - {self.pose_name}"


def top_face(tris, height, cos_min=0.99, z_tol=1e-4):
    """Triangles of the flat top face, its area centroid and its outline edges.
    A cylinder on its side has no flat top; its top-most strip is used instead."""
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    nz = cross[:, 2] / np.maximum(2 * area, 1e-15)
    sel = (nz > cos_min) & (tris[..., 2].min(axis=1) > height - z_tol)
    kind = "flat"
    if area[sel].sum() < 2e-6:  # < 2 mm^2: no flat top, the top is a ridge (edge or curved line)
        sel = (nz > 0.3) & (tris[..., 2].max(axis=1) > height - z_tol)
        kind = "ridge"
    top = tris[sel]
    w = area[sel]
    centre = (top.mean(axis=1) * w[:, None]).sum(axis=0) / w.sum()
    centre[2] = height
    # Outline = edges used by only one top triangle
    count = {}
    for t in np.round(top * 1e6).astype(np.int64):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = tuple(sorted((tuple(t[a]), tuple(t[b]))))
            count[key] = count.get(key, 0) + 1
    edges = np.array([k for k, c in count.items() if c == 1], dtype=np.float64) / 1e6
    return top, centre, edges, kind


def symmetry_order(points, tol=0.6e-3):
    """n-fold rotational symmetry about the vertical through the origin (0 = any angle)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points)

    def matches(angle):
        c, s = np.cos(angle), np.sin(angle)
        p = points @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]).T
        return np.median(tree.query(p)[0]) < tol

    if matches(np.radians(7.0)) and matches(np.radians(19.0)):
        return 0
    for n in (6, 5, 4, 3, 2):
        if matches(2 * np.pi / n):
            return n
    return 1


def _pose_name(shape, pose_index, n_poses, height, xy_extent):
    if n_poses == 1:
        return "resting"
    if shape in ("cylinder", "hexagonal_prism"):
        return "upright" if height >= max(xy_extent) - 1e-4 else "on side"
    return f"pose {pose_index + 1}"


def load_candidates(shape, sizes=SIZES, spacing_mm=1.5, seed=0, stl_dir=STL_DIR):
    """All (size, resting pose) model candidates for one STL shape name."""
    rng = np.random.default_rng(seed)
    out = []
    for size in sizes:
        path = Path(stl_dir) / f"{shape}_{size}.stl"
        if not path.exists():
            continue
        tris_mm = read_stl(path)
        poses = stable_poses(tris_mm)
        for k, (R, _) in enumerate(poses):
            t = tris_mm @ R.T
            v = t.reshape(-1, 3)
            normals, _, _, com = _mesh_props(t)
            shift = np.array([com[0], com[1], v[:, 2].min()])
            t = (t - shift) / 1000.0
            com = (com - shift) / 1000.0
            pts, nrm = sample_surface(t, normals, spacing_mm / 1000.0, rng)
            height = float(np.ptp(t[..., 2]))
            top, top_c, edges, kind = top_face(t, height)
            dense, _ = sample_surface(t, normals, 0.7e-3, np.random.default_rng(1))
            out.append(ModelCandidate(shape, size, k, "", pts, nrm, t, height, com, height,
                                      top, top_c, edges, symmetry_order(dense), kind))
    # Name poses after all sizes are loaded so small and large share names
    for c in out:
        n = sum(1 for d in out if d.size == c.size)
        c.pose_name = _pose_name(shape, c.pose_index, n, c.height, np.ptp(c.points[:, :2], axis=0))
    return out


def candidates_for_class(class_name, **kw):
    shape = CLASS_TO_STL.get(str(class_name).lower())
    return load_candidates(shape, **kw) if shape else []


if __name__ == "__main__":
    for shape in sorted(set(CLASS_TO_STL.values())):
        for c in load_candidates(shape):
            sym = "any angle" if c.sym_order == 0 else f"{c.sym_order}-fold"
            a = 0.5 * np.linalg.norm(np.cross(c.top_tris[:, 1] - c.top_tris[:, 0],
                                              c.top_tris[:, 2] - c.top_tris[:, 0]), axis=1).sum() * 1e6
            print(f"{c.label:36s} height {c.height * 1000:5.1f} mm  top {c.top_kind:5s} {a:6.0f} mm^2  "
                  f"top centre ({c.top_center[0] * 1000:+.1f}, {c.top_center[1] * 1000:+.1f}) mm  symmetry {sym}")
