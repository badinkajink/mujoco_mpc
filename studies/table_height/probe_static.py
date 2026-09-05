#!/usr/bin/env python3
"""Is the brace posture a slab requires statically stable on the feet alone?

The brace is only load-bearing after the pad touches; everything before that the
legs have to hold. This checks the re-solved pose at each slab height against the
actual foot support polygon -- the convex hull of the robot's real contact points
with the floor, taken from the compiled meshes, with the table moved out of reach
so it can carry nothing.

If the CoM sits inside that hull the pose can be held without the table, and a
failure to reach it is a control problem. If it sits outside, the brace has to be
caught dynamically and reaching the pose is not enough.
"""
import argparse, json, os, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import retarget as R
from analyze_pose import pristine_model, geometry


def support_margin(m, q):
    """Signed distance (m) from the CoM ground projection to the hull edge.

    Positive = inside the polygon. The hull is built from the FLOOR contacts the
    pose actually makes, so it is the compiled foot geometry, not a rectangle.
    """
    d = mujoco.MjData(m)
    d.qpos[:] = q
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    pts = []
    for c in range(d.ncon):
        b1 = m.geom_bodyid[d.contact[c].geom1]
        b2 = m.geom_bodyid[d.contact[c].geom2]
        if b1 == 0 or b2 == 0:
            pts.append(d.contact[c].pos[:2].copy())
    if len(pts) < 3:
        return None, 0, None
    pts = np.array(pts)
    hull = _hull(pts)
    com = d.subtree_com[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")][:2]
    return _signed_dist(com, hull), len(pts), com.copy()


def _cross2(u, v):
    return float(u[0] * v[1] - u[1] * v[0])


def _hull(p):
    p = np.unique(np.round(p, 6), axis=0)
    p = p[np.lexsort((p[:, 1], p[:, 0]))]
    def half(pp):
        out = []
        for q in pp:
            while len(out) >= 2 and _cross2(out[-1] - out[-2], q - out[-2]) <= 0:
                out.pop()
            out.append(q)
        return out
    return np.array(half(p)[:-1] + half(p[::-1])[:-1])


def _signed_dist(pt, hull):
    """Min distance to any edge; negative if the point is outside."""
    n, best, inside = len(hull), 1e9, True
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        e = b - a
        if _cross2(e, pt - a) < 0:
            inside = False
        t = np.clip(np.dot(pt - a, e) / max(1e-12, np.dot(e, e)), 0, 1)
        best = min(best, np.linalg.norm(pt - (a + t * e)))
    return best if inside else -best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "figs", "static.json"))
    a = ap.parse_args()
    m = pristine_model()
    tb = R.nid(m, mujoco.mjtObj.mjOBJ_BODY, "table")
    m.body_pos[tb][0] += 10.0          # the table carries nothing in this test
    # The keyframes park the soles ON the floor with no penetration, so MuJoCo
    # reports no contact at all. A collision margin on the floor makes the sole
    # corners register; the contact points are still the compiled foot geometry.
    for gi in range(m.ngeom):
        if m.geom_bodyid[gi] == 0:
            m.geom_margin[gi] = 0.02
    g = geometry(m)
    rows = []
    print("re-solved brace pose, table removed, both feet on the floor")
    print("%-8s %-10s %-10s %-8s" % ("face", "margin mm", "contacts", "verdict"))
    for h in np.round(np.arange(0.685, 1.1851, 0.025), 3):
        t = g["pad0"].copy();  t[2] += h - g["nom"]
        t2 = g["pad20"].copy(); t2[2] += h - g["nom"]
        q, _ = R.solve(m, g["q0"], t, t2)
        mm, n, com = support_margin(m, q)
        rows.append(dict(h=float(h), margin_mm=None if mm is None else 1000 * mm,
                         contacts=n))
        print("%-8.3f %-10s %-10d %s"
              % (h, "--" if mm is None else "%+.1f" % (1000 * mm), n,
                 "holdable" if (mm or -1) > 0 else "NOT holdable on the feet"))
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
