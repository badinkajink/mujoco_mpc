#!/usr/bin/env python3
"""Per-tick, per-rollout brace-pad contact along the horizon, from a sample dump.

    ./contact_stats.py runs/<run>/samples [--from 23 --to 26] [--json out.json]

For every dumped tick in the window and every rollout (k = -1 the centre, -2 the
fold, 0..N-1 the candidates) it walks the 101 horizon states through mj_forward on
the compiled task model and records the pad gap (bottom of `left_forearm_pad`
minus the table face) at each step, the first step at which the gap closes
(<= contact_tol), and the gap at the end of the horizon. Prints one line per tick:
how many candidates land the pad inside the horizon, and the mean J of landers vs
hoverers, which is where "the cost sorts contact discovery" is or is not visible.
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "../../build_cmake/mjpc/tasks/humanoid_bench/lean/Lean_H12_Magpie.xml")


def load_model():
    cwd = os.getcwd(); os.chdir(os.path.dirname(XML))
    m = mujoco.MjModel.from_xml_path(os.path.basename(XML)); os.chdir(cwd)
    return m


def pad_gap_track(m, d, Q, pad, tab):
    """gap (m) between the pad capsule's lowest point and the table face, per row of Q"""
    prad = m.geom_size[pad, 0]
    out = np.empty(len(Q))
    for i, q in enumerate(Q):
        d.qpos[:] = q; mujoco.mj_forward(m, d)
        surf = d.geom_xpos[tab, 2] + m.geom_size[tab, 2]
        # lowest point of the capsule: centre minus half-length along its axis (z of the local frame) minus radius
        ax = d.geom_xmat[pad].reshape(3, 3)[:, 2]
        half = m.geom_size[pad, 1]
        zlow = min(d.geom_xpos[pad, 2] + half * ax[2], d.geom_xpos[pad, 2] - half * ax[2]) - prad
        out[i] = zlow - surf
    return out


def tick_stats(m, d, tick_dir, pad, tab, tol):
    meta = json.load(open(os.path.join(tick_dir, "meta.json")))
    roll = pd.read_csv(os.path.join(tick_dir, "rollouts.csv"))
    sc = pd.read_csv(os.path.join(tick_dir, "scores.csv")).set_index("k")
    qcols = [c for c in roll.columns if c.startswith("q")]
    rows = []
    for k, g in roll.groupby("k"):
        g = g.sort_values("step")
        gap = pad_gap_track(m, d, g[qcols].values, pad, tab)
        hit = np.nonzero(gap <= tol)[0]
        rows.append(dict(k=int(k), J=float(g.cost.mean()), gap0=float(gap[0]), gap_end=float(gap[-1]), gap_min=float(gap.min()),
                         t_land=(float(hit[0] * meta["dt"]) if len(hit) else None),
                         rank=(int(sc.loc[k, "rank"]) if k >= 0 else None), elite=(int(sc.loc[k, "elite"]) if k >= 0 else None),
                         gap=gap.round(4).tolist()))
    return meta, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("--from", dest="t0", type=float, default=0)
    ap.add_argument("--to", dest="t1", type=float, default=1e9)
    ap.add_argument("--tol", type=float, default=0.003, help="gap at which the pad counts as landed (m)")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    m = load_model(); d = mujoco.MjData(m)
    pad = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "left_forearm_pad")
    tab = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    out = {}
    for td in sorted(glob.glob(os.path.join(a.samples, "tick*"))):
        t = float(td.rsplit("_t", 1)[1])
        if t < a.t0 or t > a.t1: continue
        meta, rows = tick_stats(m, d, td, pad, tab, a.tol)
        cand = [r for r in rows if r["k"] >= 0]
        land = [r for r in cand if r["t_land"] is not None]; hover = [r for r in cand if r["t_land"] is None]
        nom = [r for r in rows if r["k"] == -1][0]; fold = [r for r in rows if r["k"] == -2][0]
        el_land = sum(1 for r in land if r["elite"])
        print("%s  ph %d  gap0 %+.0f mm | landers %2d/%d (elites %d/%d) J %s | hoverers J %s | centre: %s, fold: %s" % (
            os.path.basename(td), meta["phase"], nom["gap0"] * 1e3, len(land), len(cand), el_land, meta["n_elite"],
            ("%.1f-%.1f" % (min(r["J"] for r in land), max(r["J"] for r in land))) if land else "-",
            ("%.1f-%.1f" % (min(r["J"] for r in hover), max(r["J"] for r in hover))) if hover else "-",
            ("lands at h=%.2f" % nom["t_land"]) if nom["t_land"] is not None else "hovers (%+.0f mm at end)" % (nom["gap_end"] * 1e3),
            ("lands at h=%.2f" % fold["t_land"]) if fold["t_land"] is not None else "hovers (%+.0f mm at end)" % (fold["gap_end"] * 1e3)))
        out[os.path.basename(td)] = dict(meta=meta, rows=rows)
    if a.json:
        json.dump(out, open(a.json, "w"))
        print("->", a.json)


if __name__ == "__main__":
    main()
