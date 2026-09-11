#!/usr/bin/env python3
"""Figures for the planner-ablation page, from analyze.py's summary.json."""
import argparse, csv, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arms import ARMS  # noqa: E402

FAMILY = {0: "PS", 5: "CEM", 7: "iCEM", 9: "MPPI"}
COLORS = {"iCEM": "#1f77b4", "CEM": "#2ca02c", "PS": "#d62728", "MPPI": "#ff7f0e"}
PHASES = ["stand", "lean", "reach", "release", "sb1", "sb2", "sb3", "sb4", "final"]


def fig_outcomes(agg, order, out):
    arms = [a for a in order if a in agg]
    n = len(arms)
    fig, ax = plt.subplots(figsize=(max(7, 0.42 * n + 2), 3.6))
    x = np.arange(n)
    c = np.array([agg[a]["complete"] for a in arms])
    s = np.array([agg[a]["stalled"] for a in arms])
    f = np.array([agg[a]["fell"] for a in arms])
    ax.bar(x, c, color="#2b8a3e", label="completed the ladder")
    ax.bar(x, s, bottom=c, color="#b8b8b8", label="stalled (no fall)")
    ax.bar(x, f, bottom=c + s, color="#c92a2a", label="fell")
    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=60, ha="right", fontsize=8)
    for i, a in enumerate(arms):
        ax.get_xticklabels()[i].set_color(COLORS[FAMILY[agg[a]["planner"]]])
    ax.set_ylabel("runs")
    ax.set_ylim(0, max(agg[a]["n"] for a in arms) + 0.5)
    ax.legend(fontsize=8, loc="upper right", ncol=3, frameon=False)
    ax.set_title("Outcome per arm (seeds 0-2 unless noted)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_ladder(runs, order, out):
    arms = [a for a in order if any(r["arm"] == a for r in runs)]
    fig, ax = plt.subplots(figsize=(max(7, 0.42 * len(arms) + 2), 3.4))
    for i, a in enumerate(arms):
        rs = [r for r in runs if r["arm"] == a]
        for j, r in enumerate(rs):
            y = r["max_phase"]
            m = {"complete": "o", "fell": "x", "stalled": "s"}[r["outcome"]]
            ax.scatter(i + (j - (len(rs) - 1) / 2) * 0.18, y, marker=m, s=34,
                       color=COLORS[FAMILY[r["planner"]]], zorder=3)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(PHASES)))
    ax.set_yticklabels(PHASES, fontsize=8)
    ax.set_ylabel("furthest rung entered")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Rung reached per run: o completed, s stalled, x fell", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_jitter(runs, out):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for r in runs:
        j = r.get("jitter_stand_rad", float("nan"))
        if not (j > 0):
            continue
        fam = FAMILY[r["planner"]]
        m = {"complete": "o", "fell": "x", "stalled": "s"}[r["outcome"]]
        axs[0].scatter(j, r["max_phase"] + np.random.uniform(-0.12, 0.12), marker=m,
                       color=COLORS[fam], s=30, alpha=0.85)
        axs[1].scatter(j, r.get("cost_stand_mean", float("nan")), marker=m,
                       color=COLORS[fam], s=30, alpha=0.85)
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlabel("executed-control jitter during stand_up, RMS rad per 20 ms")
        ax.grid(alpha=0.3)
    axs[0].set_yticks(range(len(PHASES)))
    axs[0].set_yticklabels(PHASES, fontsize=8)
    axs[0].set_ylabel("furthest rung")
    axs[1].set_yscale("log")
    axs[1].set_ylabel("mean plant cost during stand_up")
    for fam, col in COLORS.items():
        axs[0].scatter([], [], color=col, label=fam)
    axs[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_traces(runs, arms, seed, out):
    fig, axs = plt.subplots(3, 1, figsize=(9, 6.2), sharex=True)
    for a in arms:
        rs = [r for r in runs if r["arm"] == a and r["seed"] == seed]
        if not rs:
            continue
        r = rs[0]
        rows = list(csv.DictReader(open(r["csv"])))
        t = np.array([float(x["t"]) for x in rows])
        pz = np.array([float(x["pelvis_z"]) for x in rows])
        c = np.array([float(x["cost"]) for x in rows])
        b = np.array([float(x["brace_normal_N"]) for x in rows])
        col = COLORS[FAMILY[r["planner"]]]
        ls = "-" if a in ("icem", "ps", "mppi", "cem") else "--"
        lab = "%s (%s)" % (a, r["outcome"])
        axs[0].plot(t, pz, color=col, ls=ls, lw=1.1, label=lab)
        axs[1].plot(t, c, color=col, ls=ls, lw=1.1)
        axs[2].plot(t, b, color=col, ls=ls, lw=1.1)
    axs[0].set_ylabel("pelvis z, m")
    axs[1].set_ylabel("plant cost")
    axs[1].set_yscale("log")
    axs[2].set_ylabel("left-arm table normal, N")
    axs[2].set_xlabel("sim time, s")
    axs[0].legend(fontsize=7, ncol=2, frameon=False)
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--order", default="")
    ap.add_argument("--trace_arms", default="icem,cem,ps,mppi,ps_raw01_zero")
    ap.add_argument("--trace_seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = json.load(open(a.summary))
    runs, agg = S["runs"], S["agg"]
    order = [x for x in a.order.split(",") if x] or list(agg.keys())
    fig_outcomes(agg, order, os.path.join(a.out, "outcomes.png"))
    fig_ladder(runs, order, os.path.join(a.out, "ladder.png"))
    fig_jitter(runs, os.path.join(a.out, "jitter.png"))
    fig_traces(runs, a.trace_arms.split(","), a.trace_seed, os.path.join(a.out, "traces.png"))
    print("figures ->", a.out)


if __name__ == "__main__":
    main()
