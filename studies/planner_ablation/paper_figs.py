#!/usr/bin/env python3
"""Paper figures for the planner ablation. Reads the scored summaries written by
analyze.py (one per plan rate) and draws:

  fig_rate      completion rate vs plan rate, one line per update rule
  fig_elites    completion and executed step vs elite count k (CEM, 33 Hz)
  fig_floor     completion vs noise floor sigma at 33 Hz, CEM vs PS
  fig_step      executed step per plan, per rung, CEM vs PS vs MPPI at 33 Hz
  fig_shipped   the four planners as shipped vs at matched noise (167 Hz)
  fig_basin     completion over sigma x plan rate per update rule (+ k, lambda, N)
  fig_window    completion vs executed step per plan, every arm at every rate,
                with the lean-onset CoM margin as the dense channel

Default input is the deploy-gains campaign (runs/summary_deploy_spp*.json, the
plant the robot is); --gains xml reads the earlier XML-gains summaries.

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
    return (min(p, max(0.0, c - h)), max(p, min(1.0, c + h)))


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
    ax.set_ylim(-0.03, 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    ax.axvline(33, color=C_GRAY, lw=6, alpha=0.18, zorder=0)
    ax.text(33, 1.07, "robot", ha="center", va="bottom", fontsize=6.5, color=C_INK2)
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


# ---- fig_basin ---------------------------------------------------------------
SIG = [0.005, 0.01, 0.02, 0.03, 0.05]
SIG_ARMS = {  # every rule at the hold spline; the cubic PS axis is the spline contrast
    "elite mean": ["cem_stdmin005", "cem", "cem_stdmin02", "cem_stdmin03", "cem_stdmin05"],
    "argmin": ["ps_raw005_zero", "ps_raw01_zero", "ps_raw02_zero", "ps_raw03_zero", "ps_raw05_zero"],
    "softmax": ["mppi_raw005_zero_l1", "mppi_raw01_zero_l1", "mppi_raw02_zero_l1", "mppi_raw03_zero_l1",
                "mppi_raw05_zero_l1"],
    "argmin cubic": ["ps_raw005_cubic", "ps_raw01_cubic", "ps_raw02_cubic", "ps_raw03_cubic", "ps_raw05_cubic"],
}

RULE_COL = {"elite mean": C_ELITE, "argmin": C_ARGMIN, "softmax": C_SOFTMAX, "argmin cubic": C_ARGMIN}
RULE_TITLE = {"elite mean": "CEM, mean of k = 6 elites", "argmin": "predictive sampling, argmin",
              "softmax": "MPPI, softmax mean, λ = 1"}
ADMISSIBLE = 2 / 3  # >= 4/6 seeds (>= 2/3 at 3 seeds); declared before the runs


def _mix(hex_col, t):
    """lightness ramp for one hue: t = 0 -> near white, t = 1 -> the hue."""
    import matplotlib.colors as mc
    c = np.array(mc.to_rgb(hex_col)); w = np.array([0.985, 0.98, 0.97])
    return tuple(w + (c - w) * t)


def _cells(ax, grid, col, xlabels, ylabels, title=None, admissible=True):
    """grid[i][j] = (k, n) or None; rows = y (bottom-up), cols = x."""
    ny, nx = len(grid), len(grid[0])
    for i in range(ny):
        for j in range(nx):
            kn = grid[i][j]
            if kn is None or kn[1] == 0:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fc="white", ec=C_GRID, lw=0.6, zorder=1))
                ax.text(j + 0.5, i + 0.5, "·", ha="center", va="center", fontsize=7, color=C_GRAY, zorder=3)
                continue
            k, n = kn; p = k / n
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fc=_mix(col, p), ec="white", lw=1.2, zorder=1))
            if admissible and p >= ADMISSIBLE - 1e-9:
                ax.add_patch(plt.Rectangle((j + 0.07, i + 0.07), 0.86, 0.86, fc="none", ec=C_INK,
                                           lw=0.9, zorder=2))
            ax.text(j + 0.5, i + 0.5, "%d/%d" % (k, n), ha="center", va="center", fontsize=6.3,
                    color="white" if p > 0.6 else C_INK, zorder=3)
    ax.set_xlim(0, nx); ax.set_ylim(0, ny)
    ax.set_xticks(np.arange(nx) + 0.5); ax.set_xticklabels(xlabels)
    ax.set_yticks(np.arange(ny) + 0.5); ax.set_yticklabels(ylabels)
    ax.tick_params(length=0, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, loc="left", fontsize=7, color=col if col != C_INK2 else C_INK, pad=2.5)


def fig_basin(sums, out):
    """Row 1: completion over sigma (rows) x plan rate (columns) per update rule.
    Row 2: the rule-specific axes -- CEM elite count k, MPPI temperature lambda
    (both x rate at 33/50 Hz), and the rollout count N at 33 Hz for all three.
    Cells at >= 2/3 of seeds are outlined; the outlined area is the basin.
    Panels are placed in inches so every cell is the same size."""
    rates = [(20, "25"), (15, "33"), (10, "50"), (3, "167")]
    sums = dict(sums); sums.setdefault(20, load("runs/summary_floor_spp20.json"))
    CELL, W, H = 0.19, COL, 3.4
    fig = plt.figure(figsize=(W, H))

    def place(x, y, nx, ny):  # x, y = lower-left corner in inches from the figure's lower-left
        return fig.add_axes([x / W, y / H, nx * CELL / W, ny * CELL / H])

    x0, gap = 0.50, 0.19
    ytop, ybot = H - 0.42 - 5 * CELL, 0.58
    basin = {}
    titles = {"elite mean": "CEM, elite mean", "argmin": "PS, argmin", "softmax": "MPPI, softmax",
              "argmin cubic": "argmin, cubic"}
    for j, rule in enumerate(["elite mean", "argmin", "softmax"]):
        ax = place(x0 + j * (4 * CELL + gap), ytop, 4, 5)
        grid = [[counts(sums[spp], arm) if sums[spp]["runs"] else None for spp, _ in rates]
                for arm in SIG_ARMS[rule]]
        _cells(ax, grid, RULE_COL[rule], [h for _, h in rates], ["%g" % x for x in SIG] if j == 0 else [""] * 5,
               title=titles[rule])
        if j == 0:
            ax.set_ylabel("sampling std σ, rad")
        if j == 1:
            ax.set_xlabel("plans per second", labelpad=1, x=1.1)
        adm = sum(1 for row in grid for kn in row if kn and kn[1] and kn[0] / kn[1] >= ADMISSIBLE - 1e-9)
        tot = sum(1 for row in grid for kn in row if kn and kn[1])
        basin[rule] = (adm, tot)
    # row 2, left: CEM k x rate (33, 50)
    ks = [(1, "cem_ne1"), (2, "cem_ne2"), (6, "cem"), (10, "cem_ne10"), (20, "cem_ne20")]
    ax = place(x0, ybot, 2, 5)
    grid = [[counts(sums[spp], arm) if sums[spp]["runs"] else None for spp in (15, 10)] for _, arm in ks]
    _cells(ax, grid, C_ELITE, ["33", "50"], [str(k) for k, _ in ks], title="CEM: elites k of 20")
    ax.set_ylabel("k"); ax.set_xlabel("plans/s", labelpad=1)
    # row 2, middle: MPPI lambda x rate
    ls = [(0.1, "mppi_raw01_zero_l0.1"), (1, "mppi_raw01_zero_l1"), (10, "mppi_raw01_zero_l10")]
    ax = place(x0 + 4 * CELL + gap + 0.15, ybot + 2 * CELL, 2, 3)
    grid = [[counts(sums[spp], arm) if sums[spp]["runs"] else None for spp in (15, 10)] for _, arm in ls]
    _cells(ax, grid, C_SOFTMAX, ["33", "50"], ["%g" % l for l, _ in ls], title="MPPI: temperature λ")
    ax.set_ylabel("λ"); ax.set_xlabel("plans/s", labelpad=1)
    # row 2, right: N x rule at 33 Hz
    Ns = [(8, ["cem_n8_ne2", "ps_raw01_zero_n8", "mppi_raw01_zero_l1_n8"]),
          (20, ["cem", "ps_raw01_zero", "mppi_raw01_zero_l1"]),
          (40, ["cem_n40_ne12", "ps_raw01_zero_n40", "mppi_raw01_zero_l1_n40"])]
    ax = place(x0 + 2 * (4 * CELL + gap) + 0.12, ybot + 2 * CELL, 3, 3)
    grid = [[counts(sums[15], arm) if sums[15]["runs"] else None for arm in arms] for _, arms in Ns]
    _cells(ax, grid, C_INK2, ["CEM", "PS", "MPPI"], [str(n) for n, _ in Ns], title="rollouts N, 33 plans/s", )
    ax.set_ylabel("N")
    fig.text(0.02, 0.012, "outlined: ≥ 2/3 of 6 seeds complete.  Hold spline, σ = 0.01 rad, N = 20, k = 6, λ = 1 unless varied.",
             fontsize=6.0, color=C_INK2)
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png")
    plt.close(fig)
    return basin


# ---- fig_window (all rates) --------------------------------------------------
def _rule_of(arm):
    pl = ARMS[arm][0]
    if pl == 0 or ARMS[arm][1].get("n_elite") == 1:
        return "argmin"
    return "softmax" if pl == 9 else "elite mean"


def fig_window_all(sums, out):
    """(a) every arm at every rate: x = median executed step per plan in the
    lean rung, y = completion fraction; marker = update rule, fill = plan rate.
    (b) the dense channel behind (a): the CoM margin at the lean onset per run
    against the same x, completing runs filled, falls hollow."""
    import statistics as st
    FILL = {15: "full", 10: "left", 6: "left", 3: "none"}
    RLAB = {15: "33 plans/s", 10: "50 / 83 plans/s", 3: "167 plans/s"}
    MK = {"argmin": "^", "softmax": "D", "elite mean": "o"}
    fig, axs = plt.subplots(2, 1, figsize=(COL, 4.4), sharex=True,
                            gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.12})
    ax, bx = axs
    pts = []
    for spp, S in sums.items():
        for arm in sorted(set(r["arm"] for r in S["runs"])):
            rs = [r for r in S["runs"] if r["arm"] == arm and "row_dt" in r]
            if len(rs) < 3:
                continue
            steps = [1000 * r["jitter_phase"]["brace_lean"] for r in rs
                     if "brace_lean" in r.get("jitter_phase", {})]
            if not steps:
                continue
            k = sum(r["outcome"] == "complete" for r in rs); n = len(rs)
            pts.append((st.median(steps), k, n, _rule_of(arm), spp, arm))
            for r in rs:
                j = r.get("jitter_phase", {}).get("brace_lean"); m = r.get("lean_onset_min_com_early")
                if j is None or m is None or (isinstance(m, float) and math.isnan(m)):
                    continue
                rule = _rule_of(arm)
                bx.plot(1000 * j, m, marker=MK[rule], ms=3.6, mec=RULE_COL[rule], mew=0.7, lw=0,
                        mfc=RULE_COL[rule] if r["outcome"] == "complete" else "white",
                        alpha=0.85, zorder=3 if r["outcome"] == "complete" else 4)
    order = {"elite mean": 3, "softmax": 4, "argmin": 5}
    for x, k, n, rule, spp, arm in sorted(pts, key=lambda t: order[t[3]]):
        p = k / n; lo, hi = wilson(k, n)
        ax.plot([x, x], [lo, hi], color=RULE_COL[rule], lw=0.5, alpha=0.3, zorder=2)
        ax.plot(x, p, marker=MK[rule], color=RULE_COL[rule], ms=5.2, fillstyle=FILL[spp],
                mec=RULE_COL[rule], mew=0.8, lw=0, alpha=0.9, zorder=order[rule])
    for rule in ["elite mean", "argmin", "softmax"]:
        ax.plot([], [], marker=MK[rule], color=RULE_COL[rule], lw=0, ms=5.2, mec=RULE_COL[rule], label=rule)
    for spp in (15, 10, 3):
        ax.plot([], [], marker="s", color=C_INK2, mec=C_INK2, lw=0, ms=5, fillstyle=FILL[spp], label=RLAB[spp])
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.02), handlelength=1.2, borderaxespad=0)
    ax.set_ylabel("completed, fraction of seeds")
    ax.set_ylim(-0.03, 1.05); ax.set_yticks([0, 0.5, 1.0])
    bx.set_xscale("log")
    bx.set_xticks([2, 3, 5, 10, 20, 50]); bx.set_xticklabels(["2", "3", "5", "10", "20", "50"])
    bx.minorticks_off()
    bx.set_xlabel("executed step per plan in the lean rung, mrad RMS over 27 joints")
    bx.set_ylabel("CoM margin at the lean onset, m\n(min over 1.2 s; negative = behind)")
    bx.plot([], [], marker="o", color=C_INK2, lw=0, ms=4, label="completed")
    bx.plot([], [], marker="o", mfc="white", mec=C_INK2, lw=0, ms=4, label="failed")
    bx.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), handlelength=1.2, borderaxespad=0)
    for a_ in axs:
        style_axes(a_)
    fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", bbox_inches="tight")
    plt.close(fig)


# ---- fig_ladder --------------------------------------------------------------
LADDER = [  # (arm, family, sigma text, spline text, update text)
    ("ps",                  "PS",   "0.03–0.36", "cubic", "argmin"),
    ("ps_scaled01_cubic",   "PS",   "0.003–0.03", "cubic", "argmin"),
    ("ps_raw01_cubic",      "PS",   "0.01",  "cubic", "argmin"),
    ("ps_raw01_zero",       "PS",   "0.01",  "hold",  "argmin"),
    ("mppi",                "MPPI", "0.03–0.36", "cubic", "softmax, λ 0.1"),
    ("mppi_scaled01_cubic", "MPPI", "0.003–0.03", "cubic", "softmax, λ 0.1"),
    ("mppi_raw01_cubic_l0.1", "MPPI", "0.01", "cubic", "softmax, λ 0.1"),
    ("mppi_raw01_cubic_l1", "MPPI", "0.01",  "cubic", "softmax, λ 1"),
    ("mppi_raw01_zero_l1",  "MPPI", "0.01",  "hold",  "softmax, λ 1"),
    ("cem_cubic",           "CEM",  "0.01*", "cubic", "mean of 6 elites"),
    ("cem",                 "CEM",  "0.01*", "hold",  "mean of 6 elites"),
    ("cem_ne1",             "CEM",  "0.01*", "hold",  "argmin of 20"),
    ("icem",                "CEM",  "0.01*", "hold, AR(1)", "mean of 6, memory"),
]


def fig_ladder(sums, out):
    """One row per configuration, grouped by planner family; the three switch
    columns name the component (noise std, spline, update rule) and the cell
    that changed from the row above is set in ink, the rest in gray. Right:
    completion at 33 plans/s (filled) and 167 plans/s (hollow), Wilson 95 %."""
    rows = LADDER
    n = len(rows)
    fig = plt.figure(figsize=(COL, 0.19 * n + 0.95))
    # table axes (left 58 %) and dot axes (right 42 %)
    tax = fig.add_axes([0.0, 0.17, 0.62, 0.76]); tax.axis("off")
    ax = fig.add_axes([0.66, 0.17, 0.33, 0.76])
    ys = []
    y = 0.0
    fam_prev = None
    for arm, fam, sig, spl, upd in rows:
        if fam_prev is not None and fam != fam_prev:
            y -= 0.6
        ys.append(y); y -= 1.0
        fam_prev = fam
    cols_x = [0.02, 0.16, 0.43, 0.66]   # family, sigma, spline, update
    tax.set_xlim(0, 1); tax.set_ylim(y + 0.4, 0.6)
    ax.set_ylim(y + 0.4, 0.6)
    hdr_y = 0.55
    for x, h in zip(cols_x[1:], ["noise std, rad", "spline", "update rule"]):
        tax.text(x, hdr_y + 0.45, h, fontsize=6.5, color=C_INK2, va="bottom")
    prev = None
    for (arm, fam, sig, spl, upd), yy in zip(rows, ys):
        first = prev is None or prev[1] != fam
        if first:
            tax.text(cols_x[0], yy, fam, fontsize=7.5, color=C_INK, va="center", weight="bold")
        for x, txt, key in zip(cols_x[1:], [sig, spl, upd], [2, 3, 4]):
            changed = first or prev[key] != txt
            tax.text(x, yy, txt, fontsize=6.4, va="center",
                     color=C_INK if changed else C_GRAY, weight="bold" if changed and not first else "normal")
        prev = (arm, fam, sig, spl, upd)
        col = {"PS": C_ARGMIN, "MPPI": C_SOFTMAX, "CEM": C_ELITE}[fam]
        for spp, fill, dy in [(15, "full", 0.17), (3, "none", -0.17)]:
            S = sums[spp]
            k, nn = counts(S, arm)
            if not nn:
                continue
            p_ = k / nn; lo, hi = wilson(k, nn)
            ax.plot([lo, hi], [yy + dy, yy + dy], color=col, lw=0.7, alpha=0.35, zorder=2)
            ax.plot(p_, yy + dy, marker="o", ms=4.6, mfc=col if fill == "full" else "white", mec=col,
                    mew=1.0, lw=0, zorder=4 if fill == "full" else 3)
    ax.set_xlim(-0.06, 1.06); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "½", "1"])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x", zorder=0); ax.set_axisbelow(True)
    ax.set_xlabel("completed, fraction of seeds")
    ax.plot([], [], marker="o", ms=5, color=C_INK2, lw=0, label="33 plans/s")
    ax.plot([], [], marker="o", ms=5, mfc="white", mec=C_INK2, lw=0, label="167 plans/s")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.3, columnspacing=0.8)
    fig.text(0.0, 0.0, "* CEM/iCEM: elite std floored at 0.01 rad; the refit std settles at 1.3 × the floor within 25 refits (0.8 s at 33 plans/s)",
             fontsize=6, color=C_INK2, va="bottom")
    fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", bbox_inches="tight")
    plt.close(fig)


# ---- fig_spline --------------------------------------------------------------
def fig_spline(S15, out):
    """(a) the executed position target of the left hip pitch over 0.4 s of
    quiet standing at 33 plans/s, logged at 500 Hz: the cubic spline moves
    between plans, the zero-order hold does not. (b) every run of the spline x
    update-rule grid at 33 Hz: time spent in the lean rung before it advanced
    (the rung advances after the posture has stayed within tolerance for the
    sustain time), x = the run fell or stalled before advancing."""
    import pandas as pd
    fig = plt.figure(figsize=(COL, 2.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.0], wspace=0.62)
    ax = fig.add_subplot(gs[0, 0])
    for arm, lab, col in [("ps_raw01_cubic", "cubic", C_ARGMIN), ("ps_raw01_zero", "hold", C_INK2)]:
        pth = os.path.join(HERE, "runs/probe500/%s_s1.state.csv" % arm)
        if not os.path.exists(pth):
            continue
        d = pd.read_csv(pth)
        w = d[(d.t >= 9.0) & (d.t <= 9.4)]
        ax.plot(w.t - 9.0, 1000 * (w.u1 - w.u1.iloc[0]), color=col, lw=1.1, label=lab)
    ax.set_xlabel("time, s (quiet stand)")
    ax.set_ylabel("hip-pitch target, mrad")
    ax.set_xticks([0, 0.2, 0.4])
    ax.legend(loc="upper left", handlelength=1.4, borderaxespad=0.2)
    ax.set_title("PS, 0.01 rad, 33 plans/s", loc="left", fontsize=7.5)
    style_axes(ax)
    bx = fig.add_subplot(gs[0, 1])
    groups = [("PS", "ps_raw01_cubic", "ps_raw01_zero", C_ARGMIN, "^"),
              ("MPPI", "mppi_raw01_cubic_l1", "mppi_raw01_zero_l1", C_SOFTMAX, "D"),
              ("CEM", "cem_cubic", "cem", C_ELITE, "o"),
              ("iCEM", "icem_cubic", "icem", C_ELITE, "s")]
    xt, xl = [], []
    for g, (fam, a_cub, a_hold, col, mk) in enumerate(groups):
        for k, (arm, lab) in enumerate([(a_cub, "cubic"), (a_hold, "hold")]):
            x0 = g * 3.0 + k * 1.15
            xt.append(x0); xl.append(lab)
            rs = sorted([r for r in S15["runs"] if r["arm"] == arm], key=lambda r: r["seed"])
            # stack seeds at the same rung side by side
            seen = {}
            for r in rs:
                y = 8.6 if r["outcome"] == "complete" else r["max_phase"]
                n = seen.get(y, 0); seen[y] = n + 1
                xx = x0 + ((n % 3) - 1) * 0.3
                yy = y + (0.28 if n < 3 else -0.28) if y == 8.6 else y
                if r["outcome"] == "complete":
                    bx.plot(xx, yy, marker=mk, ms=3.3, mfc=col, mec="white", mew=0.5, lw=0, zorder=3)
                else:
                    bx.plot(xx, y, marker="x" if r["outcome"] != "stalled" else "_", ms=4, color=col,
                            mew=1.0, lw=0, zorder=3)
        bx.text(g * 3.0 + 0.575, 9.7, fam, ha="center", va="top", fontsize=7.5, color=C_INK)
    bx.set_xticks(xt); bx.set_xticklabels(xl, fontsize=6.3, rotation=35, ha="right", rotation_mode="anchor")
    bx.set_xlim(-0.7, 3 * 3.0 + 1.15 + 0.7)
    bx.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8.6])
    bx.set_yticklabels(["stand", "lean", "reach", "release", "back 1", "2", "3", "4", "complete"])
    bx.set_ylim(-0.5, 9.8)
    bx.set_ylabel("furthest rung, per seed", labelpad=2)
    bx.plot([], [], marker="x", color=C_INK2, lw=0, ms=4.5, label="fell")
    bx.plot([], [], marker="_", color=C_INK2, lw=0, ms=4.5, mew=1.1, label="stalled")
    bx.legend(loc="center right", bbox_to_anchor=(1.02, 0.5), handlelength=1.0, borderaxespad=0.1)
    bx.grid(True, axis="y", zorder=0); bx.set_axisbelow(True)
    fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", bbox_inches="tight")
    plt.close(fig)


# ---- fig_persist -------------------------------------------------------------
PERSIST = {  # arm -> persistence.py config (knots, kind, alpha); 33 Hz, sigma 0.01, hold/cubic/linear
    "ps_raw01_cubic_k6": (6, "cubic", 0.0), "cem_cubic_k6": (6, "cubic", 0.0),
    "ps_raw01_zero_k6": (6, "zero", 0.0),
    "ps_raw01_cubic": (3, "cubic", 0.0), "cem_cubic": (3, "cubic", 0.0), "icem_a0_cubic": (3, "cubic", 0.0),
    "mppi_raw01_cubic_l1": (3, "cubic", 0.0),
    "ps_raw01_linear": (3, "linear", 0.0),
    "ps_raw01_zero": (3, "zero", 0.0), "cem": (3, "zero", 0.0), "icem_a0": (3, "zero", 0.0),
    "mppi_raw01_zero_l1": (3, "zero", 0.0),
    "icem_a03_cubic": (3, "cubic", 0.3),
    "ps_raw01_zero_k2": (2, "zero", 0.0),
    "ps_raw01_cubic_k2": (2, "cubic", 0.0),
    "icem_cubic": (3, "cubic", 0.7), "icem_keep0_cubic": (3, "cubic", 0.7),
    "icem": (3, "zero", 0.7), "icem_keep0": (3, "zero", 0.7),
    "icem_a095_cubic": (3, "cubic", 0.95),
    "ps_raw01_zero_k4": (4, "zero", 0.0), "ps_raw01_cubic_k4": (4, "cubic", 0.0),
    "ps_raw01_zero_k5": (5, "zero", 0.0), "ps_raw01_linear_k4": (4, "linear", 0.0),
}
FAM_MK = {"ps": ("^", C_ARGMIN), "mppi": ("D", C_SOFTMAX), "cem": ("o", C_ELITE), "icem": ("s", C_ELITE)}


def fig_persist(S15, out):
    """Completion at 33 plans/s against the persistence of the sampled
    perturbation of the executed action (persistence.py: the lag at which its
    autocorrelation falls below 0.8): every arm at 0.01 rad whose spline, knot
    count or noise colour differs. Cubic/linear splines are hollow, the hold
    filled."""
    import persistence
    fig, ax = plt.subplots(figsize=(COL, 2.35))
    cache = {}
    pts = []
    for arm, cfg in PERSIST.items():
        k, n = counts(S15, arm)
        if not n:
            continue
        if cfg not in cache:
            _, rho, taus = persistence.tau_p(*cfg, samples=3000)
            cache[cfg] = persistence.t_below(rho, taus, 0.8)
        fam = "icem" if arm.startswith("icem") else arm.split("_")[0]
        pts.append((cache[cfg], k, n, fam, cfg[1] == "zero", arm))
    # jitter identical x a little so co-located arms show
    seen = {}
    for x, k, n, fam, hold, arm in sorted(pts, key=lambda t: t[0]):
        key = round(x, 3); j = seen.get(key, 0); seen[key] = j + 1
        xo = x * (1 + 0.015 * (j - 1.5))
        p = k / n; lo, hi = wilson(k, n)
        mk, col = FAM_MK[fam]
        ax.plot([xo, xo], [lo, hi], color=col, lw=0.6, alpha=0.3, zorder=2)
        ax.plot(xo, p, marker=mk, ms=5, mfc=col if hold else "white", mec=col, mew=0.9, lw=0, zorder=3)
    ax.axvspan(0.20, 0.22, color=C_GRAY, alpha=0.22, lw=0, zorder=0)
    ka, na = counts(S15, "icem_a095_cubic")
    if na:
        ax.text(cache[(3, "cubic", 0.95)] * 1.06, 0.04, "AR(1) α = 0.95:\nfirst-knot std 0.3 σ,\nfalls in the stand-up",
                fontsize=5.8, color=C_INK2, ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.2, 0.3, 0.5]); ax.set_xticklabels(["0.05", "0.1", "0.2", "0.3", "0.5"])
    ax.minorticks_off()
    ax.set_xlabel("persistence of the sampled perturbation in the rollout, s\n(lag at which its autocorrelation falls below 0.8)")
    ax.set_ylabel("completed, fraction of 6 seeds")
    ax.set_ylim(-0.03, 1.05); ax.set_yticks([0, 0.5, 1.0])
    for fam, lab in [("ps", "predictive sampling"), ("mppi", "MPPI"), ("cem", "CEM"), ("icem", "iCEM")]:
        mk, col = FAM_MK[fam]
        ax.plot([], [], marker=mk, color=col, lw=0, ms=5, label=lab)
    ax.plot([], [], marker="o", mfc=C_INK2, mec=C_INK2, lw=0, ms=5, label="hold")
    ax.plot([], [], marker="o", mfc="white", mec=C_INK2, lw=0, ms=5, label="cubic / linear")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, handlelength=1.0, columnspacing=1.2)
    ax.set_title("33 plans/s, σ = 0.01 rad", loc="left")
    style_axes(ax)
    fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", bbox_inches="tight")
    plt.close(fig)


# ---- fig_floor2: the plan-rate floor and the target-speed collapse ------------
FLOOR_SETS = [(3, "runs/summary_deploy_spp3.json"), (6, "runs/summary_deploy_spp6.json"),
              (10, "runs/summary_deploy_spp10.json"), (15, "runs/summary_deploy_spp15.json"),
              (20, "runs/summary_floor_spp20.json"), (30, "runs/summary_floor_spp30.json"),
              (50, "runs/summary_floor_spp50.json"), (100, "runs/summary_floor_spp100.json")]
LAT_SETS = [(0, "runs/summary_deploy_spp15.json"), (15, "runs/summary_lat_spp15_l15.json"),
            (30, "runs/summary_lat_spp15_l30.json"), (50, "runs/summary_lat_spp15_l50.json"),
            (100, "runs/summary_lat_spp15_l100.json"), (150, "runs/summary_lat_spp15_l150.json"),
            (250, "runs/summary_lat_spp15_l250.json")]
LATC_SETS = [(0, "runs/summary_deploy_spp15.json"), (100, "runs/summary_latc_spp15_l100.json"),
             (150, "runs/summary_latc_spp15_l150.json"), (250, "runs/summary_latc_spp15_l250.json")]


def fig_floor2(out):
    """(a) completion against plans per second from 5 to 167 for the five
    update-rule x spline combinations at 0.01 rad; (b) completion against the
    executed target speed (step per plan x plans per second) for every hold arm
    at every rate, including the larger-sigma arms at 10-25 plans/s; (c)
    completion at 33 plans/s against uncompensated plan latency."""
    import statistics as st
    sums = {spp: load(p) for spp, p in FLOOR_SETS}
    fig, axs = plt.subplots(1, 3, figsize=(DBL, 2.5), gridspec_kw={"width_ratios": [1.1, 1.1, 1.0], "wspace": 0.35})
    ax, bx, cx = axs
    # (a)
    for j, (arm, (label, col, mk)) in enumerate(RULE.items()):
        xs, ys, los, his = [], [], [], []
        for spp, S in sorted(sums.items()):
            k, n = counts(S, arm)
            if not n:
                continue
            xs.append(500.0 / spp); ys.append(k / n); lo, hi = wilson(k, n); los.append(lo); his.append(hi)
        off = 10 ** ((j - 2) * 0.02)
        xs = np.array(xs) * off
        ls = "-" if mk in ("o", "^", "D") else "--"
        ax.plot(xs, ys, color=col, lw=1.3, ls=ls, marker=mk, ms=4.5, mec="white", mew=0.7, label=label, zorder=3)
        for x, y, lo, hi in zip(xs, ys, los, his):
            ax.plot([x, x], [lo, hi], color=col, lw=0.6, alpha=0.35, zorder=2)
    ax.set_xscale("log"); ax.set_xticks([5, 10, 17, 25, 33, 50, 83, 167]); ax.set_xticklabels(["5", "10", "17", "25", "33", "50", "83", "167"])
    ax.minorticks_off(); ax.set_xlabel("plans per second"); ax.set_ylabel("completed, fraction of 6 seeds")
    ax.set_ylim(-0.03, 1.05); ax.set_yticks([0, 0.5, 1.0])
    ax.axvline(33, color=C_GRAY, lw=6, alpha=0.18, zorder=0); ax.text(33, 1.06, "robot", ha="center", va="bottom", fontsize=6.5, color=C_INK2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=6, handlelength=2.0, columnspacing=0.8)
    style_axes(ax)
    # (b) the operating window: every hold arm at every rate (including the
    # larger-sigma arms at 10-25 plans/s): x = directed target speed over the
    # first 2 s of the lean, y = executed step per plan; fill = completion
    import pandas as pd

    def directed(r, T=2.0):
        try:
            d = pd.read_csv(os.path.join(HERE, r["state"]), usecols=["t"] + ["u%d" % k for k in range(27)])
            m = pd.read_csv(os.path.join(HERE, r["csv"]), usecols=["t", "phase"])
        except Exception:
            return None
        ph = m[m.phase == 1]
        if ph.empty:
            return None
        t0 = ph.t.min(); w = d[(d.t >= t0) & (d.t <= t0 + T + 0.05)]
        if len(w) < 3 or w.t.max() < t0 + T * 0.9:
            return None
        U = w[["u%d" % k for k in range(27)]].values
        return 1000 * np.sqrt(((U[-1] - U[0]) ** 2).mean()) / (w.t.iloc[-1] - w.t.iloc[0])

    pts = []
    for spp, S in sums.items():
        for arm in sorted(set(r["arm"] for r in S["runs"])):
            pl, nums = ARMS[arm]
            if pl in (0, 9) and nums.get("sampling_representation", 2) != 0:
                continue
            if pl in (5, 7) and nums.get("cem_representation", 0) != 0:
                continue
            if any(k in nums for k in ("sampling_spline_points", "agent_horizon", "agent_timestep", "sampling_trajectories")):
                continue
            if nums.get("n_elite") == 20:
                continue
            rs = [r for r in S["runs"] if r["arm"] == arm]
            if len(rs) < 3:
                continue
            steps = [1000 * r["jitter_phase"]["brace_lean"] for r in rs if "brace_lean" in r.get("jitter_phase", {})]
            dv = [x for x in (directed(r) for r in rs) if x]
            if not steps or not dv:
                continue
            k, n = counts(S, arm)
            pts.append((st.median(dv), st.median(steps), k / n, _rule_of(arm), spp, arm))
    MK = {"argmin": "^", "softmax": "D", "elite mean": "o"}
    for x, y, p, rule, spp, arm in sorted(pts, key=lambda t: t[2]):
        col = RULE_COL[rule]
        bx.plot(x, y, marker=MK[rule], ms=5, mfc=_mix(col, p), mec=col, mew=0.8, lw=0, zorder=3)
    bx.set_xscale("log"); bx.set_yscale("log")
    bx.set_xticks([10, 20, 40, 80]); bx.set_xticklabels(["10", "20", "40", "80"])
    bx.set_yticks([3, 5, 10, 20, 30]); bx.set_yticklabels(["3", "5", "10", "20", "30"])
    bx.minorticks_off()
    bx.set_xlabel("directed target speed, mrad/s\n(first 2 s of the lean, RMS over 27 joints)")
    bx.set_ylabel("executed step per plan, mrad", labelpad=1)
    for rule in ["elite mean", "argmin", "softmax"]:
        bx.plot([], [], marker=MK[rule], color=RULE_COL[rule], lw=0, ms=5, label=rule)
    for p_, lab in [(0.0, "0/6"), (0.5, "3/6"), (1.0, "6/6")]:
        bx.plot([], [], marker="s", mfc=_mix(C_INK2, p_), mec=C_INK2, lw=0, ms=5, label=lab)
    bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=3, fontsize=6, handlelength=1.0, columnspacing=0.8)
    bx.set_title("every hold arm, 5–167 plans/s, σ 0.005–0.05", loc="left", fontsize=7)
    style_axes(bx); bx.grid(True, axis="x", zorder=0)
    # (c) latency at 33 plans/s: raw (planned from the stale state) and
    # compensated (the stale state predicted forward, as the deploy node does);
    # per-arm markers for the four hold arms and the 4-arm mean as a line
    HOLD = ["cem", "icem", "ps_raw01_zero", "mppi_raw01_zero_l1"]
    lat_ms = [0, 30, 60, 100, 200, 300, 500]
    xpos = {ms: k for k, ms in enumerate(lat_ms)}
    for sets, ls, lab in [(LAT_SETS, "-", "planned from the stale state"), (LATC_SETS, ":", "stale state predicted forward")]:
        xs, means = [], []
        for L, pth in sets:
            S = load(pth); ms = 2 * L
            vals = []
            for arm in HOLD:
                k, n = counts(S, arm)
                if not n:
                    continue
                vals.append(k / n)
                col, mk = RULE[arm][1], RULE[arm][2]
                cx.plot(xpos[ms] + (HOLD.index(arm) - 1.5) * 0.12, k / n, marker=mk, ms=3.2, mfc=col if ls == "-" else "white",
                        mec=col, mew=0.7, lw=0, alpha=0.9, zorder=3)
            if vals:
                xs.append(xpos[ms]); means.append(np.mean(vals))
        cx.plot(xs, means, color=C_INK2, lw=1.4, ls=ls, zorder=2, label=lab)
    cx.set_xticks(list(xpos.values())); cx.set_xticklabels([str(m) for m in lat_ms], fontsize=6.5)
    cx.set_xlabel("plan latency, ms (33 plans/s)")
    cx.set_ylim(-0.03, 1.05); cx.set_yticks([0, 0.5, 1.0]); cx.set_yticklabels([])
    cx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1, fontsize=6, handlelength=2.0)
    cx.set_title("hold arms; line = 4-arm mean", loc="left", fontsize=7)
    style_axes(cx)
    fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", bbox_inches="tight")
    plt.close(fig)

# ---- fig_mismatch: plant-side model error, 12 seeds -----------------------------
MIS_SETS = {
    "mass": [(1.00, "runs/summary_gains_spp15.json"), (1.05, "runs/summary_mismatch_spp15_m1.05.json"),
             (1.10, "runs/summary_mismatch_spp15_m1.10.json"), (1.15, "runs/summary_mismatch_spp15_m1.15.json"),
             (1.20, "runs/summary_mismatch_spp15_m1.20.json")],
    "kp": [(1.0, "runs/summary_gains_spp15.json"), (1.5, "runs/summary_mismatch_spp15_kp1.5.json"),
           (2.0, "runs/summary_mismatch_spp15_kp2.0.json"), (3.0, "runs/summary_mismatch_spp15_kp3.0.json")],
    "mu": [(1.0, "runs/summary_gains_spp15.json"), (0.6, "runs/summary_mismatch_spp15_mu0.6.json"),
           (0.4, "runs/summary_mismatch_spp15_mu0.4.json")],
}
MIS_RULES = [  # arm, label, color, marker, linestyle
    ("cem", "CEM", C_ELITE, "o", "-"),
    ("icem", "iCEM", C_ELITE, "s", "--"),
    ("ps_raw01_zero", "PS, argmin", C_ARGMIN, "v", "-"),
    ("mppi_raw01_zero_l1", "MPPI, softmax", C_SOFTMAX, "D", "-"),
]
SIGMA_LINK = {  # arm -> [(sigma, arm at that sigma)]
    "cem": [(0.01, "cem"), (0.02, "cem_stdmin02"), (0.03, "cem_stdmin03")],
    "icem": [(0.01, "icem"), (0.03, "icem_stdmin03")],
    "ps_raw01_zero": [(0.01, "ps_raw01_zero"), (0.02, "ps_raw02_zero")],
    "mppi_raw01_zero_l1": [(0.01, "mppi_raw01_zero_l1"), (0.02, "mppi_raw02_zero_l1")],
}


def fig_mismatch(out):
    """Completion of the four hold update rules on the deploy plant at 33 plans/s
    when the PLANT (not the planner's model) has its mass, joint kp or sliding
    friction scaled; 12 seeds per cell. (d) the sigma link: the same rules with
    a wider noise std at baseline (hollow) and under mass x 1.10 (filled)."""
    fig, axs = plt.subplots(2, 2, figsize=(COL, 3.3), gridspec_kw={"hspace": 0.62, "wspace": 0.32})
    panels = [("mass", "plant mass × ", axs[0, 0]), ("kp", "plant joint kp × ", axs[0, 1]), ("mu", "plant friction × ", axs[1, 0])]
    for axis, xlabel, ax in panels:
        sets = [(x, load(p)) for x, p in MIS_SETS[axis]]
        xs_all = [x for x, _ in sets]
        for j, (arm, label, col, mk, ls) in enumerate(MIS_RULES):
            xs, ys, los, his = [], [], [], []
            for x, S in sets:
                k, n = counts(S, arm)
                if not n:
                    continue
                xs.append(x); ys.append(k / n); lo, hi = wilson(k, n); los.append(lo); his.append(hi)
            span = (max(xs_all) - min(xs_all))
            dx = (j - 1.5) * 0.012 * span
            xs = np.array(xs) + dx
            ax.plot(xs, ys, color=col, lw=1.2, ls=ls, marker=mk, ms=4, mec="white", mew=0.6, label=label, zorder=3)
            for x, y, lo, hi in zip(xs, ys, los, his):
                ax.plot([x, x], [lo, hi], color=col, lw=0.6, alpha=0.4, zorder=2)
        ax.set_xticks(xs_all)
        ax.set_xticklabels(["%g" % x for x in xs_all])
        if axis == "mu":
            ax.invert_xaxis()
        ax.set_xlabel(xlabel.strip(" ×") + " ×", labelpad=1.5)
        ax.set_ylim(-0.03, 1.05); ax.set_yticks([0, 0.5, 1.0])
        style_axes(ax)
    axs[0, 0].set_ylabel("completed, fraction of 12")
    axs[1, 0].set_ylabel("completed, fraction of 12")
    axs[0, 0].set_title("(a) mass", loc="left"); axs[0, 1].set_title("(b) joint stiffness", loc="left")
    axs[1, 0].set_title("(c) sliding friction", loc="left")
    # (d) the sigma link
    dx_ = axs[1, 1]
    base = load("runs/summary_gains_spp15.json"); mis = load("runs/summary_mismatch_spp15_m1.10.json")
    for j, (arm, label, col, mk, ls) in enumerate(MIS_RULES):
        pts = SIGMA_LINK[arm]
        off = (j - 1.5) * 0.0006
        xb, yb, xm, ym = [], [], [], []
        for sig, a in pts:
            kb, nb = counts(base, a); km, nm = counts(mis, a)
            if not nb or not nm:
                continue
            xb.append(sig + off); yb.append(kb / nb); xm.append(sig + off); ym.append(km / nm)
            lo, hi = wilson(km, nm)
            dx_.plot([sig + off] * 2, [lo, hi], color=col, lw=0.6, alpha=0.4, zorder=2)
        dx_.plot(xm, ym, color=col, lw=1.2, ls=ls, marker=mk, ms=4, mec="white", mew=0.6, zorder=3)
        dx_.plot(xb, yb, color=col, lw=0, marker=mk, ms=4, mfc="white", mec=col, mew=0.9, zorder=3)
        for x, y0, y1 in zip(xb, yb, ym):
            dx_.plot([x, x], [y1, y0], color=col, lw=0.6, ls=":", zorder=2)
    dx_.set_xticks([0.01, 0.02, 0.03]); dx_.set_xticklabels(["0.01", "0.02", "0.03"])
    dx_.set_xlim(0.006, 0.034)
    dx_.set_xlabel("sampling std σ, rad", labelpad=1.5)
    dx_.set_ylim(-0.03, 1.05); dx_.set_yticks([0, 0.5, 1.0])
    dx_.set_title("(d) σ under mass × 1.10", loc="left")
    style_axes(dx_)
    a0 = axs[0, 0]
    a0.plot([], [], marker="o", ms=4, mfc="white", mec=C_INK2, lw=0, label="(d) hollow: baseline")
    a0.plot([], [], marker="o", ms=4, color=C_INK2, mec="white", lw=0, label="(d) filled: mass × 1.10")
    a0.legend(loc="upper center", bbox_to_anchor=(1.15, -1.95), ncol=3, fontsize=6.5,
              handlelength=2.0, columnspacing=1.0, handletextpad=0.4)
    fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", bbox_inches="tight")
    plt.close(fig)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gains", default="deploy", choices=["deploy", "xml"],
                    help="which campaign to draw: deploy (runs/summary_deploy_spp*, default) or xml")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "paper_figs" if a.gains == "deploy" else "paper_figs_xml")
    os.makedirs(out, exist_ok=True)
    pre = "runs/summary_deploy_spp" if a.gains == "deploy" else "runs/summary_rate_spp"
    sums = {spp: load("%s%d.json" % (pre, spp)) for spp in (3, 6, 10, 15)}
    for spp, S in sums.items():
        print("spp %2d: %3d runs, %2d arms" % (spp, len(S["runs"]), len(set(r["arm"] for r in S["runs"]))))
    fig_rate(sums, os.path.join(out, "fig_rate"))
    fig_elites(sums[15], os.path.join(out, "fig_elites"))
    fig_floor(sums[15], sums[3], os.path.join(out, "fig_floor"))
    fig_step(sums[15], os.path.join(out, "fig_step"))
    fig_shipped(sums[3], os.path.join(out, "fig_shipped"))
    fig_window(sums[15], os.path.join(out, "fig_window"))
    basin = fig_basin(sums, os.path.join(out, "fig_basin"))
    fig_window_all(sums, os.path.join(out, "fig_window_all"))
    fig_ladder(sums, os.path.join(out, "fig_ladder"))
    fig_spline(sums[15], os.path.join(out, "fig_spline"))
    fig_persist(sums[15], os.path.join(out, "fig_persist"))
    fig_floor2(os.path.join(out, "fig_floor2"))
    fig_mismatch(os.path.join(out, "fig_mismatch"))
    json.dump({"admissible_rule": ">= 2/3 of seeds complete", "basin_cells": basin},
              open(os.path.join(out, "basin.json"), "w"), indent=1)
    print("basin (admissible / measured cells of the sigma x rate grid):", basin)
    print("->", out)


if __name__ == "__main__":
    main()
