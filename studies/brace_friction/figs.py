#!/usr/bin/env python3
"""Figures for the brace-friction page (docs/lean/20260924-brace_friction.html).

    ./figs.py --out ../../docs/lean/media/brace_friction [--only creep,demand,...]

Retrospective figures read runs/replay_ablation.jsonl + runs/replay_tracks/
(replay_ablation.py); the sweep figures read runs/<batch>/scored.jsonl and the
per-run contact records (analyze.py, sweep.py).
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
# reference palette (dataviz skill, light surface)
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
CRIT = "#d03b3b"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
})


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name + ".png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("->", p)


def replay_table():
    df = pd.DataFrame([json.loads(l) for l in open(os.path.join(RUNS, "replay_ablation.jsonl"))])
    return df.drop_duplicates(["dir", "arm", "seed"], keep="last")


def pooled_pad_samples(df):
    rows = []
    for _, r in df.iterrows():
        tr = pd.read_csv(os.path.join(RUNS, "replay_tracks", f"{r.dir}__{r.arm}_s{r.seed}.csv"))
        ld = (tr.pad_fz > 20) & tr.phase.between(1, 7)
        x = tr[ld]
        rows.append(pd.DataFrame({"ratio": np.hypot(x.pad_fx, x.pad_fy) / x.pad_fz,
                                  "slip": x.pad_slip * 1000, "util": x.pad_util}))
    return pd.concat(rows)


def fig_creep(out):
    """Pad slip speed against the pad shear ratio, nominal-friction braced runs."""
    df = replay_table()
    nom = df[(df.dir == "gains_spp15") & (df.pad_loaded_s > 1.0)]
    P = pooled_pad_samples(nom)
    edges = np.linspace(0, 1.0, 21)
    mid = 0.5 * (edges[1:] + edges[:-1])
    g = P.groupby(pd.cut(P.ratio, edges), observed=False)
    med, lo, hi, n = g.slip.median(), g.slip.quantile(0.1), g.slip.quantile(0.9), g.size()
    keep = n.to_numpy() >= 50
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.fill_between(mid[keep], lo.to_numpy()[keep], hi.to_numpy()[keep], color=S1, alpha=0.12, lw=0)
    ax.plot(mid[keep], med.to_numpy()[keep], color=S1, label="Simulated pads (median, 10-90%)")
    # Coulomb reference at the real slab's static coefficient: no slip below, sliding at the limit
    ax.plot([0, 0.8, 0.8], [0, 0, 190], color=INK2, lw=1.2, label="Coulomb contact, μ = 0.8")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 190)
    ax.set_xlabel("Pad shear ratio |F$_{xy}$| / F$_z$")
    ax.set_ylabel("Pad slip speed (mm/s)")
    ax.set_title("Brace-pad slip against shear, nominal friction")
    ax.legend(loc="upper left")
    ax.text(0.99, 0.02, f"{len(P):,} loaded samples, {len(nom)} braced runs, 33 Hz",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED)
    save(fig, out, "fig_creep")


def fig_demand(out):
    """Peak shear demand per braced run at nominal friction: pads and feet."""
    df = replay_table()
    nom = df[(df.dir == "gains_spp15") & (df.pad_loaded_s > 1.0)]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    bins = np.linspace(0, 1.05, 22)
    ax = axes[0]
    ax.axvspan(0.8, 1.05, color=CRIT, alpha=0.07, lw=0)
    ax.hist(nom.pad_ratio_p95, bins=bins, color=S1, alpha=0.30, label="95th percentile of the hold")
    ax.hist(nom.pad_ratio_max, bins=bins, histtype="step", color=S1, lw=2.0, label="Peak")
    ax.axvline(0.8, color=INK2, lw=1.2)
    ax.text(0.79, 0.88, "Real slab \u03bc \u2248 0.8 ", transform=ax.get_xaxis_transform(),
            fontsize=8, color=INK2, ha="right", va="top")
    ax.set_xlabel("Pad shear ratio")
    ax.set_ylabel("Braced runs")
    ax.set_title("Brace pads on the slab")
    ax.legend(loc="upper left", fontsize=8)
    ax = axes[1]
    ax.axvspan(0.5, 1.05, color=CRIT, alpha=0.07, lw=0)
    ax.hist(nom.foot_ratio_p99, bins=bins, color=S2, alpha=0.30, label="99th percentile of the hold")
    ax.hist(nom.foot_ratio_max, bins=bins, histtype="step", color=S2, lw=2.0, label="Peak")
    ax.axvline(0.5, color=INK2, lw=1.2)
    ax.text(0.49, 0.64, "Aluminium on concrete,\nupper estimate \u03bc \u2248 0.5 ",
            transform=ax.get_xaxis_transform(), fontsize=8, color=INK2, ha="right", va="top")
    ax.set_xlabel("Foot shear ratio (worse foot)")
    ax.set_title("Feet on the floor")
    ax.legend(loc="upper left", fontsize=8)
    for ax, col, lim in ((axes[0], nom.pad_ratio_max, 0.8), (axes[1], nom.foot_ratio_max, 0.5)):
        ax.text(0.98, 0.52, "%d of %d runs peak\npast the real limit" % (int((col > lim).sum()), len(nom)),
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=CRIT)
    fig.suptitle("Friction the simulated brace drew on, %d braced runs with \u03bc = 1.0 everywhere" % len(nom),
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, out, "fig_demand")


BATCH_LABEL = {
    "b1_sd02": "CEM, σ 0.02",
    "b2_sd01": "CEM, σ 0.01 (deployed)",
    "b3_sd02_stiff": "CEM, σ 0.02, rigid slab",
    "b4_sd02": "CEM, σ 0.02",
    "b5_sd01_dz20": "CEM, σ 0.01, slab +20 mm",
    "b6_sd02_dz20": "CEM, σ 0.02, slab +20 mm",
}
CELL_LABEL = {"t1_f1": "Slab 1.0, floor 1.0", "t0.8_f0.4": "Slab 0.8, floor 0.4",
              "t0.6_f0.3": "Slab 0.6, floor 0.3", "t0.8_f1": "Slab 0.8, floor 1.0", "t1_f0.4": "Slab 1.0, floor 0.4"}
CELL_COLOR = {"t1_f1": S1, "t0.8_f0.4": S2, "t0.6_f0.3": S3, "t0.8_f1": S4, "t1_f0.4": S5}


def scored(batches):
    rows = []
    for b in batches:
        p = os.path.join(RUNS, b, "scored.jsonl")
        if os.path.exists(p):
            for line in open(p):
                r = json.loads(line)
                r["batch"] = b
                rows.append(r)
    return pd.DataFrame(rows)


def fig_slide(out, batches=("b1_sd02", "b2_sd01", "b5_sd01_dz20", "b3_sd02_stiff")):
    """Per-run pad slide and foot slide by friction cell, one panel per batch."""
    df = scored([b for b in batches])
    if df.empty:
        return
    bs = [b for b in batches if b in set(df.batch)]
    fig, axes = plt.subplots(2, len(bs), figsize=(3.2 * len(bs) + 0.6, 5.6), sharey="row", squeeze=False)
    rng = np.random.default_rng(0)
    for j, b in enumerate(bs):
        d = df[df.batch == b]
        cells = [c for c in CELL_LABEL if c in set(d.cell)]
        for i, (col, lab) in enumerate((("brace_slide_mm", "Pad slide while loaded (mm)"),
                                        ("foot_slide_mm", "Worse foot slide (mm)"))):
            ax = axes[i][j]
            for k, c in enumerate(cells):
                x = d[d.cell == c]
                jit = rng.uniform(-0.18, 0.18, len(x))
                ok = x.fell == 0
                ax.scatter(k + jit[ok.to_numpy()], x[col][ok], s=22, color=CELL_COLOR[c], edgecolor=SURF, lw=1.0, zorder=3)
                ax.scatter(k + jit[~ok.to_numpy()], x[col][~ok], s=34, marker="X", color=CRIT, edgecolor=SURF, lw=0.8, zorder=4)
                ax.plot([k - 0.28, k + 0.28], [x[col].median()] * 2, color=INK, lw=1.4, zorder=5)
            ax.set_xticks(range(len(cells)))
            ax.set_xticklabels([CELL_LABEL[c].replace(", ", "\n") for c in cells], fontsize=8)
            ax.grid(axis="x", visible=False)
            if j == 0:
                ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(BATCH_LABEL.get(b, b), fontsize=10)
    axes[0][0].scatter([], [], s=34, marker="X", color=CRIT, label="Run fell")
    axes[0][0].plot([], [], color=INK, lw=1.4, label="Median")
    axes[0][0].legend(loc="lower left", bbox_to_anchor=(0, 1.005, 1, 0.1), mode="expand",
                      ncol=2, fontsize=8)
    for j, b in enumerate(bs):
        axes[0][j].set_title(BATCH_LABEL.get(b, b), fontsize=10, pad=24)
    fig.tight_layout()
    save(fig, out, "fig_slide")


def fig_touchdown(out, batches=("b1_sd02", "b2_sd01", "b5_sd01_dz20")):
    """Touchdown speed of the pad against the peak shear ratio in the first 0.3 s."""
    df = scored(batches)
    if df.empty or "touch_v_down" not in df:
        return
    d = df[df.t_touch.notna() & df.br_ratio_impact.notna()]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for c in [c for c in CELL_LABEL if c in set(d.cell)]:
        x = d[d.cell == c]
        ok = x.fell == 0
        ax.scatter(x.touch_v_down[ok] * 1000, x.br_ratio_impact[ok], s=26, color=CELL_COLOR[c], edgecolor=SURF,
                   lw=1.0, label=CELL_LABEL[c], zorder=3)
        ax.scatter(x.touch_v_down[~ok] * 1000, x.br_ratio_impact[~ok], s=40, marker="X", color=CELL_COLOR[c],
                   edgecolor=SURF, lw=0.8, zorder=4)
    ax.set_xlabel("Pad downward speed at touchdown (mm/s)")
    ax.set_ylabel("Peak pad shear ratio, first 0.3 s")
    ax.set_title("Touchdown speed and impact shear", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005, 1, 0.1), mode="expand", ncol=3, fontsize=8)
    ax.text(0.99, 0.02, "× = run fell", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=INK2)
    save(fig, out, "fig_touchdown")


def fig_chain(out, batch, tag, t0=None, t1=None, name="fig_chain"):
    """One run's contact story on a shared clock: pad load and shear, slip speeds, pelvis height."""
    c = pd.read_csv(os.path.join(RUNS, batch, tag + ".contact.csv"))
    rec = [json.loads(l) for l in open(os.path.join(RUNS, batch, "results.jsonl")) if json.loads(l)["tag"] == tag][-1]
    mu_t, mu_f = float(rec["plant_table_mu"]), float(rec["plant_foot_mu"])
    enter = [float(x) for x in rec.get("enter", "").split(":") if x]
    te = c.t.iloc[-1]
    t0 = te - 6 if t0 is None else t0
    t1 = te + 0.2 if t1 is None else t1
    w = c[(c.t >= t0) & (c.t <= t1)]
    med = lambda x: pd.Series(np.asarray(x, float)).rolling(5, center=True, min_periods=1).median()
    fig, axes = plt.subplots(4, 1, figsize=(7.4, 7.8), sharex=True,
                             gridspec_kw={"hspace": 0.16, "height_ratios": [1.15, 1, 1, 0.75]})
    ax = axes[0]
    l1, = ax.plot(w.t, med(w.pd_fz), color=S1, lw=1.8)
    l2, = ax.plot(w.t, med(w.fl_fz), color=S2, lw=1.2)
    l3, = ax.plot(w.t, med(w.fr_fz), color=S3, lw=1.2)
    ax.set_ylabel("Normal force (N)")
    ax = axes[1]
    ratio = lambda fx, fy, fz, floor: np.where(fz > floor, np.hypot(fx, fy) / np.maximum(fz, 1e-9), np.nan)
    ax.plot(w.t, med(ratio(w.pd_fx, w.pd_fy, w.pd_fz, 20)), color=S1, lw=1.8)
    ax.plot(w.t, med(ratio(w.fl_fx, w.fl_fy, w.fl_fz, 50)), color=S2, lw=1.2)
    ax.plot(w.t, med(ratio(w.fr_fx, w.fr_fy, w.fr_fz, 50)), color=S3, lw=1.2)
    ax.axhline(mu_t, color=S1, lw=1.0, alpha=0.6)
    ax.axhline(mu_f, color=S2, lw=1.0, alpha=0.6)
    ax.text(t1, mu_t, "  slab \u03bc", va="center", fontsize=8, color=INK2)
    ax.text(t1, mu_f, "  floor \u03bc", va="center", fontsize=8, color=INK2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Shear / normal")
    ax = axes[2]
    ax.plot(w.t, med(w.pd_slip * 1000 * (w.pd_fz > 20)), color=S1, lw=1.8)
    ax.plot(w.t, med(w.fl_slip * 1000 * (w.fl_fz > 50)), color=S2, lw=1.2)
    ax.plot(w.t, med(w.fr_slip * 1000 * (w.fr_fz > 50)), color=S3, lw=1.2)
    ax.set_ylabel("Slip speed (mm/s)")
    ax = axes[3]
    ax.plot(w.t, w.pelvis_z, color=INK2, lw=1.8)
    ax.set_ylabel("Pelvis (m)")
    ax.set_xlabel("Time (s)")
    RUNG = ["stand", "lean", "reach", "release", "stand back 1", "stand back 2", "stand back 3",
            "stand back 4", "final stand"]
    for ax in axes:
        ax.set_xlim(t0, t1)
        for k, te_ in enumerate(enter):
            if t0 < te_ < t1:
                ax.axvline(te_, color=AXIS, lw=0.9, zorder=0)
    for k, te_ in enumerate(enter):
        if t0 < te_ < t1 and k < len(RUNG):
            axes[0].text(te_, 1.02, " " + RUNG[k], transform=axes[0].get_xaxis_transform(),
                         fontsize=8, color=MUTED, va="bottom")
    fig.legend([l1, l2, l3], ["Brace pads", "Left foot", "Right foot"], loc="upper left",
               bbox_to_anchor=(0.115, 0.945), ncol=3, frameon=False, fontsize=9)
    fig.suptitle("%s, seed %s \u2014 the run falls backward at %.1f s"
                 % (CELL_LABEL.get(rec["cell"], rec["cell"]), rec["seed"], te)
                 if str(rec.get("fell")) == "1" else
                 "%s, seed %s" % (CELL_LABEL.get(rec["cell"], rec["cell"]), rec["seed"]),
                 x=0.015, y=0.985, ha="left", fontsize=11, fontweight="bold", color=INK)
    save(fig, out, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--chain", default="", help="batch:tag[:t0:t1] for fig_chain")
    a = ap.parse_args()
    figs = {"creep": fig_creep, "demand": fig_demand, "slide": fig_slide,
            "touchdown": fig_touchdown, "assist": fig_assist}
    for k, f in figs.items():
        if not a.only or k in a.only.split(","):
            f(a.out)
    if a.chain and (not a.only or "chain" in a.only.split(",")):
        p = a.chain.split(":")
        fig_chain(a.out, p[0], p[1], float(p[2]) if len(p) > 2 else None,
                  float(p[3]) if len(p) > 3 else None, p[4] if len(p) > 4 else "fig_chain")



ASSIST_ORDER = ["none", "-40 N, once braced", "-20 N, last 50 mm", "-40 N, last 50 mm",
                "-20 N, whole approach", "-40 N, whole approach"]
ASSIST_LABEL = {"none": "No pull (baseline)"}


def fig_assist(out):
    """Outcome of the operator's pull-back by when it is applied."""
    import page_numbers
    n = page_numbers.N()
    A = n.get("assist", {})
    rows = [(k, A[k]) for k in ASSIST_ORDER if k in A]
    if len(rows) < 2:
        return
    fig, ax = plt.subplots(figsize=(7.6, 0.46 * len(rows) + 1.6))
    y = np.arange(len(rows))
    comp = np.array([r["complete"] for _, r in rows], float)
    onset = np.array([r["fell_onset"] for _, r in rows], float)
    late = np.array([r["fell_late"] for _, r in rows], float)
    G = 0.06   # surface gap between segments
    ax.barh(y, comp, height=0.52, color=S3, label="Completed the ladder")
    ax.barh(y, onset, left=comp + G, height=0.52, color=CRIT, label="Fell at lean onset (< 16 s)")
    ax.barh(y, late, left=comp + onset + 2 * G, height=0.52, color=S4, label="Fell later")
    for i, (k, r) in enumerate(rows):
        if r["complete"]:
            ax.text(r["complete"] - 0.12, i, "%d" % r["complete"], va="center", ha="right",
                    fontsize=9, color="white", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([ASSIST_LABEL.get(k, k) for k, _ in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(6.5, float((comp + onset + late).max()) + 0.5))
    ax.set_xlabel("Runs of 6 (seeds 0–5)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Pulling the robot back into the brace: outcome by when the pull is applied", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005, 1, 0.1), mode="expand", ncol=3, fontsize=8)
    ax.text(0.995, -0.30, "Slab μ 0.8, floor μ 0.4, CEM σ 0.02, 33 plans/s; the pull is "
            "unmodelled by the planner", transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=MUTED)
    save(fig, out, "fig_assist")

if __name__ == "__main__":
    main()
