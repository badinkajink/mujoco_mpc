#!/usr/bin/env python3
"""Every number the page quotes, computed from the run records.

    ./page_numbers.py              # print them
    (make_page.py imports N())

Sources: runs/replay_ablation.jsonl + runs/replay_tracks/ (replay_ablation.py,
the 2026-09-11..13 planner-ablation corpus re-scored for contact forces) and
runs/<batch>/scored.jsonl (analyze.py over this study's own sweeps).
"""
import json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
CELLS = ["t1_f1", "t0.8_f0.4", "t0.6_f0.3"]


def replay():
    df = pd.DataFrame([json.loads(l) for l in open(os.path.join(RUNS, "replay_ablation.jsonl"))])
    return df.drop_duplicates(["dir", "arm", "seed"], keep="last")


def scored(batch):
    p = os.path.join(RUNS, batch, "scored.jsonl")
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in open(p)])


def pooled(df):
    rows = []
    for _, r in df.iterrows():
        tr = pd.read_csv(os.path.join(RUNS, "replay_tracks", f"{r.dir}__{r.arm}_s{r.seed}.csv"))
        x = tr[(tr.pad_fz > 20) & tr.phase.between(1, 7)]
        rows.append(pd.DataFrame({"ratio": np.hypot(x.pad_fx, x.pad_fy) / x.pad_fz,
                                  "slip": x.pad_slip * 1000, "util": x.pad_util}))
    return pd.concat(rows)


def N():
    n = {}
    df = replay()
    nom = df[(df.dir == "gains_spp15") & (df.pad_loaded_s > 1.0)]
    n["repl_runs"] = len(df)
    n["repl_nom"] = len(df[df.dir == "gains_spp15"])
    n["braced"] = len(nom)
    n["pad_peak_med"] = nom.pad_ratio_max.median()
    n["pad_p95_med"] = nom.pad_ratio_p95.median()
    n["pad_over_08"] = int((nom.pad_ratio_max > 0.8).sum())
    n["pad_p95_over_08"] = int((nom.pad_ratio_p95 > 0.8).sum())
    n["foot_peak_med"] = nom.foot_ratio_max.median()
    n["foot_p99_med"] = nom.foot_ratio_p99.median()
    n["foot_over_05"] = int((nom.foot_ratio_max > 0.5).sum())
    n["pad_slide_med"] = nom.pad_slide_mm.median()
    n["pad_slide_q"] = nom.pad_slide_mm.quantile([0.1, 0.9]).tolist()
    P = pooled(nom)
    n["pool"] = len(P)
    n["slip_lt01"] = P.slip[P.ratio < 0.1].median()
    n["slip_03"] = P.slip[(P.ratio > 0.25) & (P.ratio < 0.35)].median()
    n["slip_08"] = P.slip[(P.ratio > 0.75) & (P.ratio < 0.85)].median()
    n["frac_slipping"] = (P.slip > 2).mean()
    n["frac_slip_below_cone"] = ((P.slip > 2) & (P.util < 0.95)).sum() / max(1, (P.slip > 2).sum())
    fell = df[df.fell == 1]
    n["falls"] = len(fell)
    n["falls_braced"] = 0
    for _, r in fell.iterrows():
        tr = pd.read_csv(os.path.join(RUNS, "replay_tracks", f"{r.dir}__{r.arm}_s{r.seed}.csv"))
        w = tr[tr.t >= tr.t.iloc[-1] - 3.0]
        n["falls_braced"] += int((w.pad_fz > 20).any())
    # this study's sweeps
    for b in ("b1_sd02", "b2_sd01", "b5_sd01_dz20", "b3_sd02_stiff"):
        d = scored(b)
        if d.empty:
            continue
        n[b] = {}
        for c in sorted(set(d.cell)):
            x = d[d.cell == c]
            n[b][c] = dict(n=len(x), complete=int(x.complete.sum()), fell=int(x.fell.sum()),
                           pad_slide=float(x.brace_slide_mm.median()),
                           foot_slide=float(x.foot_slide_mm.median()),
                           sole=float(x.sole_disp_mm.median()),
                           touch_v=float(x.touch_v_down.median()),
                           r_p95=float(x.br_ratio_p95.median()),
                           r_impact=float(x.br_ratio_impact.median()) if x.br_ratio_impact.notna().any() else float("nan"))
    return n


if __name__ == "__main__":
    n = N()
    for k, v in n.items():
        if isinstance(v, dict):
            print(k)
            for c, d in v.items():
                print("   ", c, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in d.items()})
        elif isinstance(v, float):
            print(f"{k:26s} {v:.4g}")
        elif isinstance(v, list):
            print(f"{k:26s} {[round(x,1) for x in v]}")
        else:
            print(f"{k:26s} {v}")
