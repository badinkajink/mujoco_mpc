#!/usr/bin/env python3
"""Paper figures for the planner ablation. Reads the scored summaries written by
analyze.py (one per plan rate) and draws:

  fig_rate      completion rate vs plan rate, one line per update rule
  fig_elites    completion and executed step vs elite count k (CEM, 33 Hz)
  fig_floor     completion vs noise floor sigma at 33 Hz, CEM vs PS
  fig_step      executed step per plan, per rung, CEM vs PS vs MPPI at 33 Hz
  fig_shipped   the four planners as shipped vs at matched noise (167 Hz)

Color follows the UPDATE RULE, the entity the paper argues about, in a fixed
order: elite mean (CEM, iCEM) blue, argmin (predictive sampling) orange, softmax
mean (MPPI) aqua -- the first three categorical slots of the validated default
palette, which clear the all-pairs CVD floors. Text and axes use ink tokens, never
a series color. Wilson 95% intervals on every proportion.
"""
import argparse, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arms import ARMS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- tokens ------------------------------------------------------------------
C_ELITE, C_ARGMIN, C_SOFTMAX = "#2a78d6", "#eb6834", "#1baf7a"
C_GRAY, C_INK, C_INK2, C_GRID = "#9a9a96", "#0b0b0b", "#52514e", "#e6e5e1"
RULE = {  # arm -> (update rule label, color, marker)
    "cem": ("CEM, mean of 6 elites", C_ELITE, "o"),
    "icem": ("iCEM, mean of 6 elites", C_ELITE, "s"),
    "mppi_raw01_zero_l1": ("MPPI, softmax mean (λ = 1)", C_SOFTMAX, "D"),
    "ps_raw01_cubic": ("predictive sampling, argmin (cubic)", C_ARGMIN, "^"),
    "ps_raw01_zero": ("predictive sampling, argmin (zero-order)", C_ARGMIN, "v"),
}
RATES = [(3, 167), (6, 83), (10, 50), (15, 33)]
PH = ["stand", "lean", "reach", "release", "sb1", "sb2", "sb3", "sb4", "final"]
PH_KEYS = ["stand_up", "brace_lean", "reach", "release", "standback_r1", "standback_r2",
           "standback_r3", "standback_r4", "stand_final"]

rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.edgecolor": C_INK2, "axes.linewidth": 0.6,
    "xtick.color": C_INK2, "ytick.color": C_INK2, "text.color": C_INK,
    "axes.labelcolor": C_INK, "grid.color": C_GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42,
})
COL, DBL = 3.45, 7.1  # IEEE column and double-column widths, in


def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(path):
    p = os.path.join(HERE, path)
    return json.load(open(p)) if os.path.exists(p) else {"runs": [], "agg": {}}


def counts(S, arm):
    rs = [r for r in S["runs"] if r["arm"] == arm]
    return sum(r["outcome"] == "complete" for r in rs), len(rs)


def step_per_plan(S, arm, rung=None):
    """median over seeds of the executed step (mrad RMS per log row), only for
    runs logged one row per plan (row_dt > 0.021 s or == spp x 2 ms)."""
    v = []
    for r in S["runs"]:
        if r["arm"] != arm or "row_dt" not in r:
            continue
        if rung is None:
            j = r.get("jitter_rms_rad")
        else:
            j = r.get("jitter_phase", {}).get(rung)
        if j is not None and not math.isnan(j):
            v.append(1000 * j)
    return (float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))) if v else (math.nan,) * 3


def style_axes(ax):
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)


def errbar(ax, x, k, n, color, marker, label, dx=0.0, ms=5):
    p = k / n if n else math.nan
    lo, hi = wilson(k, n)
    ax.errorbar(x + dx, p, yerr=[[p - lo], [hi - p]], fmt=marker, color=color, ms=ms,
                mec="white", mew=0.8, capsize=2, elinewidth=1.0, lw=0, label=label, zorder=3)


# ---- fig_rate ----------------------------------------------------------------
def fig_rate(sums, out):
    fig, ax = plt.subplots(figsize=(COL, 2.75))
    xs = [hz for _, hz in RATES]
    n_arms = len(RULE)
    for j, (arm, (label, col, mk)) in enumerate(RULE.items()):
        ys, los, his, ns = [], [], [], []
        for spp, hz in RATES:
            k, n = counts(sums[spp], arm)
            ys.append(k / n if n else math.nan); ns.append(n)
            lo, hi = wilson(k, n); los.append(lo); his.append(hi)
        ys, los, his = np.array(ys), np.array(los), np.array(his)
        # a small horizontal offset (in log units) keeps the five markers at one
        # rate from stacking; segments are drawn only between measured neighbours
        off = 10 ** ((j - (n_arms - 1) / 2) * 0.022)
        xo = np.array(xs) * off
        ls = "-" if mk in ("o", "^", "D") else "--"
        for i in range(len(xs) - 1):
            if not (math.isnan(ys[i]) or math.isnan(ys[i + 1])):
                ax.plot(xo[i:i + 2], ys[i:i + 2], color=col, lw=1.4, ls=ls, zorder=2)
        for i in range(len(xs)):
            if math.isnan(ys[i]):
                continue
            ax.errorbar(xo[i], ys[i], yerr=[[ys[i] - los[i]], [his[i] - ys[i]]], fmt=mk, color=col,
                        ms=5, mec="white", mew=0.8, capsize=1.5, elinewidth=0.8, alpha=0.95, zorder=3)
        ax.plot([], [], color=col, lw=1.4, ls=ls, marker=mk, ms=5, mec="white", mew=0.8, label=label)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(["%d" % x for x in xs])
    ax.minorticks_off()
    ax.set_xlabel("plans per second")
    ax.set_ylabel("ladder completed, fraction of seeds")
    ax.set_ylim(-0.03, 1.05)
    ax.set_yticks([0, 0.5, 1.0])
    ax.axvspan(33, 45, color=C_GRAY, alpha=0.15, lw=0, zorder=0)
    ax.text(38.5, 1.02, "deploy node", ha="center", va="bottom", fontsize=6.5, color=C_INK2)
    style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, handlelength=2.4,
              columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png")
    plt.close(fig)


# ---- fig_elites --------------------------------------------------------------
def fig_elites(S, out):
    ks = [1, 2, 6, 10, 20]
    arms = ["cem_ne1", "cem_ne2", "cem", "cem_ne10", "cem_ne20"]
    fig, axs = plt.subplots(2, 1, figsize=(COL, 3.4), sharex=True,
                            gridspec_kw={"height_ratios": [1.15, 1]})
    for k, arm in zip(ks, arms):
        c, n = counts(S, arm)
        if n:
            errbar(axs[0], k, c, n, C_ELITE if k not in (1, 20) else C_GRAY, "o", None)
        m, lo, hi = step_per_plan(S, arm)
        if not math.isnan(m):
            axs[1].errorbar(k, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=5, mec="white", mew=0.8,
                            color=C_ELITE if k not in (1, 20) else C_GRAY, capsize=2, elinewidth=1.0, zorder=3)
    axs[0].set_ylabel("completed, fraction")
    axs[0].set_ylim(-0.03, 1.05); axs[0].set_yticks([0, 0.5, 1.0])
    axs[1].set_ylabel("executed step per plan,\nmrad RMS over 27 joints")
    axs[1].set_xlabel("elite count k of 20 rollouts (CEM, 33 plans/s)")
    axs[1].set_xscale("log"); axs[1].set_xticks(ks); axs[1].set_xticklabels([str(k) for k in ks])
    axs[1].minorticks_off()
    axs[0].annotate("argmin", (1, 0), xytext=(0, 14), textcoords="offset points", ha="center",
                    fontsize=6.5, color=C_INK2)
    axs[0].annotate("no selection", (20, 0), xytext=(0, 14), textcoords="offset points", ha="center",
                    fontsize=6.5, color=C_INK2)
    for ax in axs:
        style_axes(ax)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png")
    plt.close(fig)


# ---- fig_floor ---------------------------------------------------------------
def fig_floor(S15, S3, out):
    ladders = {
        "CEM, mean of 6 elites": (C_ELITE, "o", [(0.01, "cem"), (0.02, "cem_stdmin02"), (0.03, "cem_stdmin03"),
                                                 (0.05, "cem_stdmin05"), (0.10, "cem_stdmin10")]),
        "predictive sampling, argmin": (C_ARGMIN, "^", [(0.01, "ps_raw01_cubic"), (0.02, "ps_raw02_cubic"),
                                                        (0.03, "ps_raw03_cubic"), (0.05, "ps_raw05_cubic")]),
    }
    fig, axs = plt.subplots(1, 2, figsize=(DBL * 0.62, 2.4), sharey=True)
    for ax, (S, title) in zip(axs, [(S3, "167 plans/s (3 seeds)"), (S15, "33 plans/s (6 seeds)")]):
        for label, (col, mk, pts) in ladders.items():
            xs, ys, los, his = [], [], [], []
            for sig, arm in pts:
                k, n = counts(S, arm)
                if not n:
                    continue
                xs.append(sig); ys.append(k / n)
                lo, hi = wilson(k, n); los.append(lo); his.append(hi)
            if xs:
                ax.plot(xs, ys, color=col, lw=1.6, marker=mk, ms=5, mec="white", mew=0.8, label=label, zorder=3)
                ax.fill_between(xs, los, his, color=col, alpha=0.10, lw=0, zorder=1)
        ax.set_xscale("log")
        ax.set_xticks([0.01, 0.02, 0.03, 0.05, 0.1])
        ax.set_xticklabels(["0.01", "0.02", "0.03", "0.05", "0.10"])
        ax.minorticks_off()
        ax.set_xlabel("sampling std per knot, rad")
        ax.set_title(title, loc="left")
        ax.set_ylim(-0.03, 1.05); ax.set_yticks([0, 0.5, 1.0])
        style_axes(ax)
    axs[0].set_ylabel("completed, fraction")
    axs[1].legend(loc="lower left")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png")
    plt.close(fig)


# ---- fig_step ----------------------------------------------------------------
def fig_step(S, out):
    arms = [("cem", "CEM, mean of 6", C_ELITE, "o"), ("mppi_raw01_zero_l1", "MPPI, softmax (λ = 1)", C_SOFTMAX, "D"),
            ("ps_raw01_cubic", "PS, argmin", C_ARGMIN, "^")]
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    x = np.arange(len(PH_KEYS))
    for i, (arm, label, col, mk) in enumerate(arms):
        ys, los, his = [], [], []
        for ph in PH_KEYS:
            m, lo, hi = step_per_plan(S, arm, ph)
            ys.append(m); los.append(lo); his.append(hi)
        ys = np.array(ys); ok = ~np.isnan(ys)
        ax.plot(x[ok], ys[ok], color=col, lw=1.6, marker=mk, ms=4.5, mec="white", mew=0.8, label=label, zorder=3)
        ax.fill_between(x[ok], np.array(los)[ok], np.array(his)[ok], color=col, alpha=0.10, lw=0)
    ax.set_xticks(x); ax.set_xticklabels(PH)
    ax.set_ylabel("executed step per plan,\nmrad RMS over 27 joints")
    ax.set_xlabel("rung of the ladder (33 plans/s, 0.01 rad noise)")
    style_axes(ax)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png")
    plt.close(fig)


# ---- fig_shipped -------------------------------------------------------------
def fig_shipped(S3, out):
    rows = [("predictive\nsampling", "ps", "ps_raw01_cubic", C_ARGMIN),
            ("MPPI", "mppi", "mppi_raw01_zero_l1", C_SOFTMAX),
            ("CEM", "cem", "cem", C_ELITE), ("iCEM", "icem", "icem", C_ELITE)]
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    y = np.arange(len(rows))[::-1]
    for yi, (label, a_ship, a_match, col) in zip(y, rows):
        k1, n1 = counts(S3, a_ship); k2, n2 = counts(S3, a_match)
        p1 = k1 / n1 if n1 else math.nan; p2 = k2 / n2 if n2 else math.nan
        if p1 != p2:
            ax.plot([p1, p2], [yi, yi], color=col, lw=1.6, zorder=2)
            ax.plot([p1], [yi], marker="o", ms=6, color="white", mec=col, mew=1.4, zorder=3)
        ax.plot([p2], [yi], marker="o", ms=6, color=col, mec="white", mew=0.8, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(-0.05, 1.08); ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("completed, fraction of 3 seeds (167 plans/s)")
    ax.plot([], [], marker="o", ms=6, color="white", mec=C_INK2, mew=1.4, lw=0, label="as shipped")
    ax.plot([], [], marker="o", ms=6, color=C_INK2, lw=0, label="noise set to 0.01 rad")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.32))
    ax.grid(True, axis="x", zorder=0); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png")
    plt.close(fig)


# ---- fig_window --------------------------------------------------------------
def fig_window(S, out, hz=33):
    """Every arm at one plan rate as a point: x = median executed step per plan
    in the lean rung, y = completion fraction. The failure mode is annotated by
    where the failing runs died: small steps die backward at the lean onset,
    large steps die in the stand."""
    import statistics as st
    fig, ax = plt.subplots(figsize=(COL, 2.9))
    pts = []
    for arm in sorted(set(r["arm"] for r in S["runs"])):
        rs = [r for r in S["runs"] if r["arm"] == arm and "row_dt" in r]
        if len(rs) < 4:
            continue
        k = sum(r["outcome"] == "complete" for r in rs); n = len(rs)
        steps = [1000 * r["jitter_phase"]["brace_lean"] for r in rs if "brace_lean" in r.get("jitter_phase", {})]
        if not steps:
            continue
        pl = ARMS[arm][0]
        rule = "argmin" if (pl == 0 or ARMS[arm][1].get("n_elite") == 1) else ("softmax" if pl == 9 else "elite mean")
        col = {"argmin": C_ARGMIN, "softmax": C_SOFTMAX, "elite mean": C_ELITE}[rule]
        mk = {"argmin": "^", "softmax": "D", "elite mean": "o"}[rule]
        pts.append((st.median(steps), k, n, col, mk, arm, rule))
    seen = set()
    order = {"elite mean": 3, "softmax": 4, "argmin": 5}  # argmin drawn last: it shares x with cem_ne2
    for x, k, n, col, mk, arm, rule in sorted(pts, key=lambda t: order[t[6]]):
        lo, hi = wilson(k, n)
        p = k / n
        ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt=mk, color=col, ms=5.5, mec="white", mew=0.8,
                    capsize=1.5, elinewidth=0.7, alpha=0.95, zorder=order[rule],
                    label=rule if rule not in seen else None)
        seen.add(rule)
    ax.set_xscale("log")
    ax.set_xticks([3, 5, 10, 20, 50])
    ax.set_xticklabels(["3", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_xlabel("executed step per plan in the lean rung, mrad RMS over 27 joints")
    ax.set_ylabel("completed, fraction of seeds")
    ax.set_ylim(-0.03, 1.05); ax.set_yticks([0, 0.5, 1.0])
    ax.annotate("fails backward\nat the lean onset", xy=(4.2, 0.88), fontsize=6.5, color=C_INK2, ha="center")
    ax.annotate("fails in the stand", xy=(36, 0.88), fontsize=6.5, color=C_INK2, ha="center")
    style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, title=None)
    ax.set_title("%d plans/s, every arm with 6 seeds" % hz, loc="left")
    fig.tight_layout()
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "paper_figs"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sums = {3: load("runs/summary_rate_spp3.json"), 6: load("runs/summary_rate_spp6.json"),
            10: load("runs/summary_rate_spp10.json"), 15: load("runs/summary_rate_spp15.json")}
    fig_rate(sums, os.path.join(a.out, "fig_rate"))
    fig_elites(sums[15], os.path.join(a.out, "fig_elites"))
    fig_floor(sums[15], sums[3], os.path.join(a.out, "fig_floor"))
    fig_step(sums[15], os.path.join(a.out, "fig_step"))
    fig_shipped(sums[3], os.path.join(a.out, "fig_shipped"))
    fig_window(sums[15], os.path.join(a.out, "fig_window"))
    print("->", a.out)


if __name__ == "__main__":
    main()
