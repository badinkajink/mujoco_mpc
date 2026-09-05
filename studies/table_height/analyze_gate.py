#!/usr/bin/env python3
"""The gate every failing run dies at, measured from the logged qpos.

Rung 2 of strategy 25 carries `reach_target_table` [0.55, 0.04, 0.15], so
lean.cc builds its advance target from the slab -- near edge + 0.55 in x, table
centre - 0.04 in y, face + 0.15 in z -- and advances when the RIGHT gripper jaw
tip holds within `target_distance_tolerance` (70 mm) of it for 2 s. Cost and gate
grade the same point.

Replaying each run's qpos gives the closest approach the gate actually saw, with
no extra sim time. Runs without a `--qpos_out` dump are skipped.

usage: analyze_gate.py --runs LABEL=dir [LABEL=dir ...] --out figs
"""
import argparse, csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze import style, STATUS, SEQ, CAT
import retarget as R
from analyze_pose import pristine_model
from render_video import set_table_height

TIP = np.array([0.2254, -0.0118, -0.1062])   # kGripperTipLocal, lean.cc
RTT = (0.55, 0.04, 0.15)                     # h12_brace_targeting rung 2
TOL = 0.07


def one(m, d, tgc, grip, qcsv, scsv, face):
    ctr, half = d.geom_xpos[tgc].copy(), m.geom_size[tgc].copy()
    tgt = np.array([ctr[0] - half[0] + RTT[0], ctr[1] - RTT[1],
                    ctr[2] + half[2] + RTT[2]])
    ph = {round(float(r["t"]), 3): int(float(r["phase"]))
          for r in csv.DictReader(open(scsv))}
    best, err = 1e9, None
    for r in csv.DictReader(open(qcsv)):
        if ph.get(round(float(r["t"]), 3)) != 2:
            continue
        d.qpos[:] = [float(r["q%d" % i]) for i in range(m.nq)]
        mujoco.mj_kinematics(m, d)
        tip = d.xpos[grip] + d.xmat[grip].reshape(3, 3) @ TIP
        e = tip - tgt
        n = float(np.linalg.norm(e))
        if n < best:
            best, err = n, e.copy()
    return (None, None, tgt) if err is None else (best, err, tgt)


def collect(rundir, label):
    out = []
    s = os.path.join(rundir, "summary.csv")
    if not os.path.exists(s):
        return out
    cache = {}
    for r in csv.DictReader(open(s)):
        q = (r.get("csv") or "").replace(".csv", ".qpos.csv")
        if not q or not os.path.exists(q):
            continue
        face = float(r["table_h"])
        if face not in cache:
            m = pristine_model(); set_table_height(m, face)
            d = mujoco.MjData(m); mujoco.mj_forward(m, d)
            cache[face] = (m, d,
                           R.nid(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision"),
                           R.nid(m, mujoco.mjtObj.mjOBJ_BODY, "right_magpie_gripper"))
        best, err, tgt = one(*cache[face], q, r["csv"], face)
        out.append(dict(
            label=label, h=face, seed=int(r["seed"]),
            outcome=("fell" if r["fell"] == "1" else
                     "complete" if r["complete"] == "1" else "stalled"),
            reached_rung2=best is not None,
            closest_mm=None if best is None else 1000 * best,
            dx_mm=None if err is None else 1000 * err[0],
            dy_mm=None if err is None else 1000 * err[1],
            dz_mm=None if err is None else 1000 * err[2],
            target_z=float(tgt[2])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = []
    for spec in a.runs:
        label, path = spec.split("=", 1)
        recs += collect(path, label)
    if not recs:
        raise SystemExit("no qpos dumps found")
    got = [r for r in recs if r["closest_mm"] is not None]

    for mode in ("light", "dark"):
        c = style(mode)
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0))
        marks = {"seeded": "o", "ab-on": "s", "ab-off": "^", "lead-on": "D",
                 "lead-off": "v", "mode2": "P"}
        a0 = ax[0]
        a0.axhspan(0, 1000 * TOL, color=STATUS["complete"], alpha=0.12, lw=0)
        for r in got:
            a0.plot([r["h"]], [r["closest_mm"]], marks.get(r["label"], "o"),
                    ms=9, mfc=STATUS[r["outcome"]], mec=c["surface"], mew=1.2,
                    zorder=4)
        a0.axhline(1000 * TOL, color=STATUS["complete"], lw=1.4)
        a0.annotate("target_distance_tolerance = %.0f mm" % (1000 * TOL),
                    (0.79, 1000 * TOL), xytext=(0, 7),
                    textcoords="offset points", fontsize=8.5,
                    color=STATUS["complete"])
        a0.set_yscale("log")
        a0.set_xlabel("slab face height (m)")
        a0.set_ylabel("closest jaw tip to the gate target (mm)")
        a0.set_title("The rung-2 advance gate, replayed", loc="left",
                     fontsize=11, color=c["fg"])
        a0.grid(alpha=0.6, lw=0.6, which="both")

        a1 = ax[1]
        for k, col, lab in (("dx_mm", SEQ[5], "x (depth onto the slab)"),
                            ("dy_mm", CAT["right arm"], "y (lateral)"),
                            ("dz_mm", CAT["left wrist"], "z (height)")):
            xs = sorted({r["h"] for r in got})
            ys = [np.median([r[k] for r in got if r["h"] == h]) for h in xs]
            a1.plot(xs, ys, "o-", color=col, lw=1.8, ms=6, label=lab)
        a1.axhline(0, color=c["axis"], lw=1.0)
        a1.axhspan(-1000 * TOL, 1000 * TOL, color=STATUS["complete"], alpha=0.12,
                   lw=0)
        a1.set_xlabel("slab face height (m)")
        a1.set_ylabel("jaw tip minus target at closest approach (mm)")
        a1.set_title("and which axis is short", loc="left", fontsize=11,
                     color=c["fg"])
        a1.legend(fontsize=8.5, frameon=False)
        a1.grid(alpha=0.6, lw=0.6)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "fig_gate" +
                                 (".png" if mode == "light" else ".dark.png")),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
    json.dump(recs, open(os.path.join(a.out, "gate.json"), "w"), indent=1)
    print("%d runs, %d reached rung 2" % (len(recs), len(got)))
    for r in sorted(got, key=lambda z: (z["h"], z["label"], z["seed"])):
        print("  %-9s %.3f s%d %-9s closest %5.0f mm  dx %+5.0f dy %+5.0f dz %+5.0f"
              % (r["label"], r["h"], r["seed"], r["outcome"], r["closest_mm"],
                 r["dx_mm"], r["dy_mm"], r["dz_mm"]))


if __name__ == "__main__":
    main()
