#!/usr/bin/env python3
"""Release-plan dump analysis (sweep2, arm D): what the planner believed vs what was true.

Inputs per run: <tag>.csv (bench log, now with com_edge_belief / brace_belief = the plant
state evaluated on the planner's model) and <tag>.plan.csv (after every plan iteration in
phases >= 8, the nominal trajectory evaluated on the planner model and on the plant model).
"""
import argparse, csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130})


KEYS = ("t", "phase", "pelvis_z", "brace_normal_N", "com_beyond_foot_edge", "cop_beyond_foot_edge",
        "com_edge_belief", "cop_edge_belief", "brace_belief", "rhand_x", "com_x", "torso_tilt_deg")


def load_log(path):
    rs = [r for r in csv.DictReader(open(path)) if all(r.get(k) not in (None, "") for k in KEYS)]
    f = lambda k: np.array([float(r[k]) for r in rs])
    return {k: f(k) for k in KEYS}


def load_plan(path):
    rs = [r for r in csv.DictReader(open(path)) if all(v not in (None, "") for v in r.values())]
    if not rs:
        return None
    f = lambda k: np.array([float(r[k]) for r in rs])
    return {k: f(k) for k in rs[0].keys()}


def release_t(d):
    i = np.where(d["phase"] == 9)[0]
    return d["t"][i[0]] if len(i) else None


def fig_belief_vs_true(run, tags, labels, media, name):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.1))
    for k, (tag, lab) in enumerate(zip(tags, labels)):
        p = os.path.join(run, tag + ".csv")
        if not os.path.exists(p):
            continue
        d = load_log(p)
        t0 = release_t(d)
        if t0 is None:
            continue
        sel = (d["t"] >= t0 - 8) & (d["t"] <= t0 + 20)
        tt = d["t"][sel] - t0
        axes[0].plot(tt, d["brace_normal_N"][sel], color=CAT[k], lw=1.4, label=f"{lab}: true")
        axes[0].plot(tt, d["brace_belief"][sel], color=CAT[k], lw=1.1, ls="--", label=f"{lab}: planner's model")
        axes[1].plot(tt, 100 * d["com_beyond_foot_edge"][sel], color=CAT[k], lw=1.4)
        axes[1].plot(tt, 100 * d["com_edge_belief"][sel], color=CAT[k], lw=1.1, ls="--")
        axes[2].plot(tt, d["pelvis_z"][sel], color=CAT[k], lw=1.4)
    axes[0].set_ylabel("Brace normal force (N)"); axes[1].set_ylabel("CoM beyond toe (cm)")
    axes[2].set_ylabel("Pelvis height (m)"); axes[1].set_ylim(-24, 12)
    for ax in axes:
        ax.set_xlabel("time since forearm_brace_release entry (s)"); ax.grid(True, color=GRID, lw=0.6)
        ax.axvline(0, color=MUTED, lw=1, ls=":")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(media, name), bbox_inches="tight"); plt.close(fig)


def fig_plan_fans(run, tag, media, name, every=0.25, label="", window=(-4.0, 1.5)):
    """The planner's 1 s nominal prediction (CoM beyond toe, brace force) at each plan iteration
    in the seconds around brace lift-off, against what the plant then did."""
    lp = os.path.join(run, tag + ".plan.csv"); ll = os.path.join(run, tag + ".csv")
    if not (os.path.exists(lp) and os.path.exists(ll)):
        return
    pl = load_plan(lp); d = load_log(ll)
    if pl is None:
        return
    t0 = release_t(d)
    if t0 is None:
        return
    off = np.where((d["t"] > t0) & (d["brace_normal_N"] < 5.0))[0]
    tl = d["t"][off[0]] if len(off) else d["t"][-1]           # lift-off (or end)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4))
    sel = (d["t"] >= tl + window[0] - 1) & (d["t"] <= tl + window[1])
    axes[0].plot(d["t"][sel] - tl, 100 * d["com_beyond_foot_edge"][sel], color=INK, lw=1.8, label="what the plant did", zorder=5)
    axes[1].plot(d["t"][sel] - tl, d["brace_normal_N"][sel], color=INK, lw=1.8, label="what the plant did", zorder=5)
    tplans = np.unique(pl["t_plan"])
    tplans = tplans[(tplans >= tl + window[0]) & (tplans <= tl + window[1] - 1.0)]
    picks, nxt = [], -1e9
    for tp in tplans:
        if tp >= nxt:
            picks.append(tp); nxt = tp + every
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("fan", ["#9ec5f4", "#0d366b"])
    for j, tp in enumerate(picks):
        m = pl["t_plan"] == tp
        c = cmap(j / max(1, len(picks) - 1))
        axes[0].plot(pl["t_pred"][m] - tl, 100 * pl["com_edge_belief"][m], color=c, lw=1.0,
                     label="planner's 1 s prediction, one per plan" if j == 0 else None)
        axes[1].plot(pl["t_pred"][m] - tl, pl["brace_belief"][m], color=c, lw=1.0,
                     label="planner's 1 s prediction, one per plan" if j == 0 else None)
    axes[0].set_ylim(-45, 12); axes[1].set_ylim(-5, 110)
    axes[0].set_ylabel("CoM beyond toe (cm)"); axes[1].set_ylabel("Brace normal force (N)")
    axes[0].set_title(label, loc="left", fontsize=10)
    for ax in axes:
        ax.set_xlim(window[0], window[1]); ax.set_xlabel("time since brace lift-off (s)")
        ax.axvline(0, color=MUTED, lw=1, ls=":"); ax.grid(True, color=GRID, lw=0.6)
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(media, name), bbox_inches="tight"); plt.close(fig)


def release_table(run, tags):
    """Numbers at release entry and at brace lift-off (first brace_normal < 5 N after entry)."""
    rows = []
    for tag in tags:
        p = os.path.join(run, tag + ".csv")
        if not os.path.exists(p):
            continue
        d = load_log(p)
        t0 = release_t(d)
        if t0 is None:
            rows.append(dict(tag=tag, note="no phase 9")); continue
        i0 = np.searchsorted(d["t"], t0)
        # 1-s means just before release entry
        w = (d["t"] >= t0 - 1) & (d["t"] < t0)
        off = np.where((d["t"] > t0) & (d["brace_normal_N"] < 5.0))[0]
        r = dict(tag=tag, t_release=round(float(t0), 2),
                 brace_true_pre=round(float(d["brace_normal_N"][w].mean()), 1),
                 brace_belief_pre=round(float(d["brace_belief"][w].mean()), 1),
                 com_true_pre_cm=round(100 * float(d["com_beyond_foot_edge"][w].mean()), 1),
                 com_belief_pre_cm=round(100 * float(d["com_edge_belief"][w].mean()), 1),
                 rhand_x_pre=round(float(d["rhand_x"][w].mean()), 3),
                 t_liftoff=round(float(d["t"][off[0]] - t0), 2) if len(off) else None,
                 com_true_at_liftoff_cm=round(100 * float(d["com_beyond_foot_edge"][off[0]]), 1) if len(off) else None,
                 min_pelvis_after=round(float(d["pelvis_z"][i0:].min()), 3),
                 fell=bool(d["pelvis_z"][i0:].min() < 0.6))
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(HERE, "runs/sweep2"))
    ap.add_argument("--media", default=os.path.join(HERE, "../../docs/brace_payload/media"))
    a = ap.parse_args()
    os.makedirs(a.media, exist_ok=True)
    tags = [f"D_m{m}_b{b}_s{s}" for m in (0, 4) for b in (0, 4) for s in (0, 1, 2)]
    rows = release_table(a.run, tags)
    for r in rows:
        print(r)
    json.dump(rows, open(os.path.join(a.run, "release_table.json"), "w"), indent=1)
    fig_belief_vs_true(a.run, ["D_m0_b0_s1", "D_m0_b4_s0", "D_m4_b0_s1"],
                       ["true 0 kg, believed 0", "true 0 kg, believed 4 kg", "true 4 kg, believed 0"],
                       a.media, "fig_release_belief.png")
    fig_plan_fans(a.run, "D_m0_b0_s0", a.media, "fig_release_plan.png",
                  label="true 0 kg, believed 0 kg, seed 0: the release push-off goes over backward")
    return rows


if __name__ == "__main__":
    main()
