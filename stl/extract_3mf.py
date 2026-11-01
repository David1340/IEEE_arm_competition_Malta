"""Extract every object in a 3MF project (e.g. a PrusaSlicer plate) into separate binary STL files.

Each object is written in the pose it has on the print bed (the plate transform's rotation and
scale are applied), so it is a stable resting pose like the one on the competition table.
The object is centred in X/Y and its lowest point is at Z = 0. Units are millimetres.

Objects are named from the slicer's object names when present, and tagged small/large by
size when the same shape appears more than once.

Usage:
    python stl/extract_3mf.py stl/pick_n_place_objects.3mf
    python stl/extract_3mf.py some.3mf --out stl/
"""

import argparse
import re
import struct
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

NS = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}


def parse_transform(text):
    """3MF transform "m00 m01 m02 m10 m11 m12 m20 m21 m22 m30 m31 m32" -> (3x3, translation).
    Points are row vectors: p' = p @ R + t."""
    if not text:
        return np.eye(3), np.zeros(3)
    v = np.array([float(x) for x in text.split()]).reshape(4, 3)
    return v[:3], v[3]


def slicer_names(zf):
    """Object id -> name from PrusaSlicer / Bambu metadata, if present."""
    names = {}
    for cfg in ("Metadata/Slic3r_PE_model.config", "Metadata/model_settings.config"):
        if cfg not in zf.namelist():
            continue
        root = ET.fromstring(zf.read(cfg))
        for obj in root.iter("object"):
            for md in obj.iter("metadata"):
                if md.get("key") == "name" and md.get("type", "object") == "object":
                    names[obj.get("id")] = md.get("value")
                    break
    return names


def clean_name(name):
    name = re.sub(r"\.stl$", "", name, flags=re.I)
    name = re.sub(r"\s*\(\d+\)$", "", name)  # "cube (1)" -> "cube"
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_").lower() or "object"


def write_stl(path, verts, tris):
    v = verts[tris]  # (n, 3, 3)
    normals = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    with open(path, "wb") as f:
        f.write(path.stem.encode()[:80].ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tris)))
        rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        rec["n"], rec["v"] = normals, v
        f.write(rec.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("threemf", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()

    with zipfile.ZipFile(args.threemf) as zf:
        model = ET.fromstring(zf.read("3D/3dmodel.model"))
        names = slicer_names(zf)

    meshes = {}
    for obj in model.iterfind("m:resources/m:object", NS):
        mesh = obj.find("m:mesh", NS)
        if mesh is None:
            continue  # component-only objects are not used by these plates
        verts = np.array([[float(v.get(k)) for k in "xyz"] for v in mesh.iterfind("m:vertices/m:vertex", NS)])
        tris = np.array([[int(t.get(k)) for k in ("v1", "v2", "v3")] for t in mesh.iterfind("m:triangles/m:triangle", NS)])
        meshes[obj.get("id")] = (verts, tris)

    objects = []
    for item in model.iterfind("m:build/m:item", NS):
        oid = item.get("objectid")
        if oid not in meshes:
            continue
        verts, tris = meshes[oid]
        R, _ = parse_transform(item.get("transform"))
        v = verts @ R
        if np.linalg.det(R) < 0:  # mirrored: keep normals pointing outwards
            tris = tris[:, ::-1]
        lo, hi = v.min(0), v.max(0)
        v = v - [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]
        objects.append((clean_name(names.get(oid, f"object_{oid}")), v, tris))

    groups = defaultdict(list)
    for obj in objects:
        groups[obj[0]].append(obj)

    args.out.mkdir(parents=True, exist_ok=True)
    for base, objs in sorted(groups.items()):
        objs.sort(key=lambda o: np.ptp(o[1], axis=0).max())
        if len(objs) == 2:
            tags = ["small", "large"]
        elif len(objs) == 1:
            tags = [None]
        else:
            tags = [f"size{i}" for i in range(len(objs))]
        for tag, (_, v, tris) in zip(tags, objs):
            path = args.out / (f"{base}_{tag}.stl" if tag else f"{base}.stl")
            write_stl(path, v, tris)
            dx, dy, dz = np.ptp(v, axis=0)
            print(f"{path.name:32s} {len(tris):5d} tris   {dx:6.1f} x {dy:6.1f} x {dz:6.1f} mm")


if __name__ == "__main__":
    main()