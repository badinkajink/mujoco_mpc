#!/usr/bin/env python3
"""Merged tables and figures over sweep1 + sweep2 (2026-09-15).

Pools arm D (sweep2 dump runs) with arm M: same condition, same planner, the dump is
passive. Arm P stays separate. Scenario arms (Smean/Smax/Smin) have no belief axis.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze import load, heat, INK, INK2, MUTED, GRID, SEQ, CAT

HERE = os.path.dirname(os.path.abspath(__file__))


def load_all(runs):
    rows = []
    for run in runs:
        p = os.path.join(run, "results.jsonl")
        if os.path.exists(p):
            for r in load(run, attached_only=False):
                r["run"] = os.path.basename(run)
                r["arm_pooled"] = "M" if r["arm"] in ("M", "D") else r["arm"]
                rows.append(r)
    return rows


def cell_stats(rs):
    return {"n": len(rs), "fell": sum(r["fell"] == 1 for r in rs), "carried": sum(r["carried"] for r in rs),
            "complete": sum(r["complete"] == 1 for r in rs),
            "peak_brace_med": float(np.median([r["peak_brace_carry"] for r in rs])) if rs else float("nan")}


def grid_pooled(rows, arm):
    rs_arm = [r for r in rows if r["arm_pooled"] == arm and r["attached"]]
    ms = sorted({r["m"] for r in rs_arm}); bs = sorted({r["b"] for r in rs_arm})
    g = {(m, b): cell_stats([r for r in rs_arm if r["m"] == m and r["b"] == b]) for m in ms for b in bs}
    g = {k: v for k, v in g.items() if v["n"]}
    return ms, bs, g


def fig_grid_M(rows, media):
    ms, bs, g = grid_pooled(rows, "M")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    heat(axes[0], ms, bs, g, "carried", "Arm M (belief in the model), sweeps 1+2\ncarried through release / attached runs",
         lambda st: f"{st['carried']}/{st['n']}", 1.0)
    heat(axes[1], ms, bs, g, "fell", "fell after the attach / attached runs  (median peak brace)",
         lambda st: f"{st['fell']}/{st['n']}\n{st['peak_brace_med']:.0f} N", 1.0)
    fig.tight_layout(); fig.savefig(os.path.join(media, "fig_payload_grid_M6.png"), bbox_inches="tight"); plt.close(fig)


def pooled_tables(rows):
    att = [r for r in rows if r["attached"] and not r["arm"].startswith("S")]
    out = {}
    for name, sel in [("belief below the truth", lambda r: r["b"] < r["m"]),
                      ("belief equal to the truth", lambda r: r["b"] == r["m"]),
                      ("belief above the truth", lambda r: r["b"] > r["m"])]:
        out[name] = cell_stats([r for r in att if sel(r)])
    for b in sorted({r["b"] for r in att}):
        out[f"believed {b:g} kg"] = cell_stats([r for r in att if r["b"] == b])
    for m in sorted({r["m"] for r in att}):
        out[f"true {m:g} kg"] = cell_stats([r for r in att if r["m"] == m])
    for d in (-4, -3, -2, -1, 0, 1, 2, 3, 4):
        rs = [r for r in att if round(r["b"] - r["m"]) == d]
        if rs:
            out[f"belief error {d:+d} kg"] = cell_stats(rs)
    out["true 4 kg, believed 0 kg"] = cell_stats([r for r in att if r["m"] == 4 and r["b"] == 0])
    out["true 4 kg, believed 4 kg"] = cell_stats([r for r in att if r["m"] == 4 and r["b"] == 4])
    allrows = [r for r in rows if not r["arm"].startswith("S")]
    out["pre-attach falls"] = {"n": len(allrows), "fell": sum(1 for r in allrows if not r["attached"])}
    return out


def scenario_table(rows):
    out = {}
    for arm in ("Smean", "Smax", "Smin"):
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        att = [r for r in rs if r["attached"]]
        out[arm] = {"all": cell_stats(att), "pre_attach_falls": sum(1 for r in rs if not r["attached"]), "n_runs": len(rs),
                    "by_true": {f"{m:g}": cell_stats([r for r in att if r["m"] == m]) for m in sorted({r["m"] for r in att})},
                    "wall_med": float(np.median([r["wall_s"] for r in rs]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[os.path.join(HERE, "runs/sweep1"), os.path.join(HERE, "runs/sweep2")])
    ap.add_argument("--media", default=os.path.join(HERE, "../../docs/brace_payload/media"))
    a = ap.parse_args()
    rows = load_all(a.runs)
    print(f"{len(rows)} runs loaded; attached {sum(r['attached'] for r in rows)}")
    ms, bs, g = grid_pooled(rows, "M")
    print("Arm M+D: rows true, cols believed (fell/n, carried/n, median peak brace)")
    print("            " + "".join(f"{b:>20g} kg" for b in bs))
    for m in ms:
        print(f"  true {m:g} kg " + "".join(f"  f{g[(m,b)]['fell']}/{g[(m,b)]['n']:<2d} c{g[(m,b)]['carried']}/{g[(m,b)]['n']:<2d} {g[(m,b)]['peak_brace_med']:4.0f}N" if (m, b) in g else " " * 23 for b in bs))
    ms, bs, gp = grid_pooled(rows, "P")
    if gp:
        print("Arm P:")
        print("            " + "".join(f"{b:>20g} kg" for b in bs))
        for m in ms:
            print(f"  true {m:g} kg " + "".join(f"  f{gp[(m,b)]['fell']}/{gp[(m,b)]['n']:<2d} c{gp[(m,b)]['carried']}/{gp[(m,b)]['n']:<2d} {gp[(m,b)]['peak_brace_med']:4.0f}N" if (m, b) in gp else " " * 23 for b in bs))
    pooled = pooled_tables(rows)
    print("Pooled (P + M + D, attached):")
    for k, v in pooled.items():
        print(f"  {k:28s} fell {v['fell']}/{v['n']}" + (f"  carried {v['carried']}/{v['n']}  brace {v['peak_brace_med']:.0f} N" if 'carried' in v else ""))
    sc = scenario_table(rows)
    for arm, v in sc.items():
        print(f"  {arm}: fell {v['all']['fell']}/{v['all']['n']} carried {v['all']['carried']}/{v['all']['n']} pre-attach {v['pre_attach_falls']}/{v['n_runs']} wall {v['wall_med']:.0f} s  by true: " +
              " ".join(f"{m}kg f{s['fell']}/{s['n']} b{s['peak_brace_med']:.0f}N" for m, s in v["by_true"].items()))
    os.makedirs(a.media, exist_ok=True)
    fig_grid_M(rows, a.media)
    json.dump({"grid_M": {f"{m:g}|{b:g}": v for (m, b), v in g.items()}, "grid_P": {f"{m:g}|{b:g}": v for (m, b), v in gp.items()},
               "pooled": pooled, "scenario": sc}, open(os.path.join(HERE, "runs/sweep2/summary_merged.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
