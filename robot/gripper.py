"""How far the SO-101's jaws are apart, and where the middle of the gap is, for any opening and depth.

Measured once on the URDF's meshes (so101ik's Model.mesh_points), in so101ik's tip frame (gripper_frame_link: the
fixed jaw's grasp point; x towards the moving jaw is negative, z along the fingers), the wrist roll and pitch
making no difference there. depth = how far back from the finger ends (m); opening = the gripper's 0-100 %.
The real arm closed on a 3 cm cylinder at 21.6 %, where this gives 3.4 cm with the squeeze: close enough.
"""

from functools import lru_cache

import numpy as np

GRIPS = np.arange(0.0, 72.5, 2.5)            # % openings tabulated
DEPTHS = np.arange(0.0025, 0.0351, 0.0025)    # m back from the finger ends


@lru_cache(maxsize=1)
def _table():
    """gap[g, d] and middle x[g, d] (m) for GRIPS x DEPTHS."""
    from so101ik import Model

    m = Model(stock=True)
    mp = m.mesh_points()
    gap = np.full((len(GRIPS), len(DEPTHS)), np.nan)
    mid = np.full_like(gap, np.nan)
    for i, g in enumerate(GRIPS):
        p = m.ik(0.30, 0.0, 0.05, roll=-90, pitch=-60, gripper=float(g))
        f = m.frames(p.deg)
        Ti = np.linalg.inv(f["gripper_frame_link"])
        local = lambda n: ((np.c_[mp[n], np.ones(len(mp[n]))] @ f[n].T) @ Ti.T)[:, :3]
        G, M = local("gripper_link"), local("moving_jaw_so101_v1_link")
        for j, d in enumerate(DEPTHS):
            fixed = G[(np.abs(G[:, 2] + d) < 0.003) & (np.abs(G[:, 1]) < 0.012) & (G[:, 0] > -0.02)]
            moving = M[np.abs(M[:, 2] + d) < 0.003]
            if len(fixed) and len(moving):
                fx, mx = fixed[:, 0].min(), moving[:, 0].max()
                gap[i, j], mid[i, j] = fx - mx, (fx + mx) / 2
    return gap, mid


def _at(arr, grip, depth):
    """arr at (grip, depth), linear in the opening, nearest tabulated depth with a value."""
    j = int(np.argmin(np.abs(DEPTHS - depth)))
    col = arr[:, j]
    ok = np.isfinite(col)
    return float(np.interp(grip, GRIPS[ok], col[ok]))


def gap(grip, depth):
    """The gap between the jaws (m) at depth back from the finger ends, with the gripper open grip %."""
    return _at(_table()[0], grip, depth)


def centre(grip, depth):
    """The middle of the gap in the tip frame (m): (x, 0, -depth)."""
    return np.array([_at(_table()[1], grip, depth), 0.0, -depth])


def opening_for(width, depth, margin, lo=12.0, hi=60.0):
    """The smallest opening (%) whose gap at depth is width + 2 margin, within lo..hi. (opening, gap there)."""
    need = width + 2 * margin
    for g in np.arange(lo, hi + 0.01, 0.5):
        if gap(g, depth) >= need:
            return float(g), gap(g, depth)
    return hi, gap(hi, depth)
