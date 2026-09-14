#!/usr/bin/env python3
"""Separated panels for the paper's planner-ablation multifigure (LaTeX
subcaptions, no letters, titles or footnotes rendered into the PDFs). Each panel
is drawn at the physical width it occupies in a 516 pt IEEEtran figure* so the
8 pt serif type is not scaled by \\includegraphics.

  panel_ladder     (a) PS -> CEM one component at a time, 33 filled / 167 hollow
  panel_persist    (b) success vs persistence of the sampled perturbation
  panel_rate       (c) success vs plans per second, 5-167, sigma = 0.01 rad
  panel_basin      (d) success over sigma x plan rate per update rule (hold)
  panel_mismatch   (e) success under plant mass / joint kp / friction scaling, 12 seeds
  panel_sigma      (f) wider sigma at baseline (hollow) and under mass x 1.10 (filled)
  panel_strips     (extra) CEM elites k, MPPI lambda, rollouts N at 33/50 plans/s

Reads the same scored summaries as paper_figs.py; writes paper_panels/*.pdf|png
and copies the PDFs to ../../paper/figures/planner/panels/. The tex that uses
them is paper/planner_ablation_multifig.tex.
"""
import os, shutil, sys
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paper_figs as F  # noqa: E402  (tokens, helpers, arm tables, rcParams)

TW = 516.0 / 72.27  # IEEEtran conference \textwidth in inches
W = {"a": 0.45 * TW, "b": 0.26 * TW, "c": 0.26 * TW, "d": 0.43 * TW, "e": 0.375 * TW, "f": 0.165 * TW}
H1, H2 = 2.75, 1.55  # row heights, in
FAM = {  # family -> (marker, color, label); fill encodes the spline where it varies
    "cem": ("o", F.C_ELITE, "CEM"), "icem": ("s", F.C_ELITE, "iCEM"),
    "ps": ("v", F.C_ARGMIN, "PS"), "mppi": ("D", F.C_SOFTMAX, "MPPI"),
}


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".pdf")); fig.savefig(os.path.join(out, name + ".png"))
    plt.close(fig)


# ---- (a) ladder --------------------------------------------------------------
def panel_ladder(sums, out):
    rows = F.LADDER
    fig = plt.figure(figsize=(W["a"], H1))
    tax = fig.add_axes([0.0, 0.12, 0.70, 0.80]); tax.axis("off")
    ax = fig.add_axes([0.735, 0.12, 0.26, 0.80])
    ys, y, fam_prev = [], 0.0, None
    for arm, fam, sig, spl, upd in rows:
        if fam_prev is not None and fam != fam_prev:
            y -= 0.6
        ys.append(y); y -= 1.0; fam_prev = fam
    cols_x = [0.01, 0.145, 0.37, 0.60]
    tax.set_xlim(0, 1); tax.set_ylim(y + 0.4, 0.6); ax.set_ylim(y + 0.4, 0.6)
    for x, h in zip(cols_x[1:], ["noise std, rad", "spline", "update rule"]):
        tax.text(x, 1.0, h, fontsize=6.5, color=F.C_INK2, va="bottom")
    prev = None
    for (arm, fam, sig, spl, upd), yy in zip(rows, ys):
        first = prev is None or prev[1] != fam
        if first:
            tax.text(cols_x[0], yy, fam, fontsize=7.5, color=F.C_INK, va="center", weight="bold")
        for x, txt, key in zip(cols_x[1:], [sig, spl, upd], [2, 3, 4]):
            changed = first or prev[key] != txt
            tax.text(x, yy, txt, fontsize=6.4, va="center",
                     color=F.C_INK if changed else F.C_GRAY, weight="bold" if changed and not first else "normal")
        prev = (arm, fam, sig, spl, upd)
        col = {"PS": F.C_ARGMIN, "MPPI": F.C_SOFTMAX, "CEM": F.C_ELITE}[fam]
        for spp, fill, dy in [(15, "full", 0.17), (3, "none", -0.17)]:
            k, nn = F.counts(sums[spp], arm)
            if not nn:
                continue
            p_ = k / nn; lo, hi = F.wilson(k, nn)
            ax.plot([lo, hi], [yy + dy, yy + dy], color=col, lw=0.7, alpha=0.35, zorder=2)
            ax.plot(p_, yy + dy, marker="o", ms=4.4, mfc=col if fill == "full" else "white", mec=col,
                    mew=1.0, lw=0, zorder=4 if fill == "full" else 3)
    ax.set_xlim(-0.06, 1.06); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "½", "1"])
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x", zorder=0); ax.set_axisbelow(True)
    ax.set_xlabel("Success rate", labelpad=1)
    ax.plot([], [], marker="o", ms=4.6, color=F.C_INK2, lw=0, label="33 plans/s")
    ax.plot([], [], marker="o", ms=4.6, mfc="white", mec=F.C_INK2, lw=0, label="167 plans/s")
    fig.legend(loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=2, handletextpad=0.3, columnspacing=0.8,
               borderaxespad=0.0)
    save(fig, out, "panel_ladder")


# ---- (b) persistence -----------------------------------------------------------
def panel_persist(S15, out):
    import persistence
    fig = plt.figure(figsize=(W["b"], H1))
    ax = fig.add_axes([0.20, 0.42, 0.78, 0.55])
    cache, pts = {}, []
    for arm, cfg in F.PERSIST.items():
        k, n = F.counts(S15, arm)
        if not n:
            continue
        if cfg not in cache:
            _, rho, taus = persistence.tau_p(*cfg, samples=3000)
            cache[cfg] = persistence.t_below(rho, taus, 0.8)
        fam = "icem" if arm.startswith("icem") else arm.split("_")[0]
        pts.append((cache[cfg], k, n, fam, cfg[1] == "zero"))
    seen = {}
    for x, k, n, fam, hold in sorted(pts, key=lambda t: t[0]):
        key = round(x, 3); j = seen.get(key, 0); seen[key] = j + 1
        xo = x * (1 + 0.015 * (j - 1.5))
        p = k / n; lo, hi = F.wilson(k, n)
        mk, col, _ = FAM[fam]
        ax.plot([xo, xo], [lo, hi], color=col, lw=0.6, alpha=0.3, zorder=2)
        ax.plot(xo, p, marker=mk, ms=4.6, mfc=col if hold else "white", mec=col, mew=0.9, lw=0, zorder=3)
    ax.axvspan(0.20, 0.22, color=F.C_GRAY, alpha=0.22, lw=0, zorder=0)
    if F.counts(S15, "icem_a095_cubic")[1]:
        ax.annotate("α = 0.95", xy=(cache[(3, "cubic", 0.95)], 0.0), xytext=(cache[(3, "cubic", 0.95)] * 0.72, 0.3),
                    fontsize=6, color=F.C_INK2, ha="center",
                    arrowprops=dict(arrowstyle="-", color=F.C_INK2, lw=0.5, shrinkB=4))
    ax.set_xscale("log"); ax.set_xticks([0.05, 0.1, 0.2, 0.3, 0.5]); ax.set_xticklabels(["0.05", "0.1", "0.2", "0.3", "0.5"])
    ax.minorticks_off()
    ax.set_xlabel("Persistence of the sampled\nperturbation (s)", labelpad=2)
    ax.set_ylabel("Success rate", labelpad=2)
    ax.set_ylim(-0.03, 1.05); ax.set_yticks([0, 0.5, 1.0])
    for fam in ("cem", "icem", "ps", "mppi"):
        mk, col, lab = FAM[fam]
        ax.plot([], [], marker=mk, color=col, lw=0, ms=4.6, label=lab)
    ax.plot([], [], marker="o", mfc=F.C_INK2, mec=F.C_INK2, lw=0, ms=4.6, label="Hold")
    ax.plot([], [], marker="o", mfc="white", mec=F.C_INK2, lw=0, ms=4.6, label="Cubic, linear")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.40), ncol=2, handlelength=1.0, columnspacing=0.9,
              handletextpad=0.4, labelspacing=0.35, fontsize=6.6)
    F.style_axes(ax)
    save(fig, out, "panel_persist")


# ---- (c) plan rate -----------------------------------------------------------
RATE_ARMS = [  # arm, family, hold?, linestyle
    ("cem", "cem", True, "-"), ("icem", "icem", True, "--"),
    ("ps_raw01_zero", "ps", True, "-"), ("ps_raw01_cubic", "ps", False, ":"),
    ("mppi_raw01_zero_l1", "mppi", True, "-"),
]


def panel_rate(out):
    sums = {spp: F.load(p) for spp, p in F.FLOOR_SETS}
    fig = plt.figure(figsize=(W["c"], H1))
    ax = fig.add_axes([0.20, 0.42, 0.78, 0.52])
    for j, (arm, fam, hold, ls) in enumerate(RATE_ARMS):
        mk, col, lab = FAM[fam]
        xs, ys, los, his = [], [], [], []
        for spp, S in sorted(sums.items()):
            k, n = F.counts(S, arm)
            if not n:
                continue
            xs.append(500.0 / spp); ys.append(k / n); lo, hi = F.wilson(k, n); los.append(lo); his.append(hi)
        xs = np.array(xs) * 10 ** ((j - 2) * 0.02)
        ax.plot(xs, ys, color=col, lw=1.1, ls=ls, marker=mk, ms=4.0, mfc=col if hold else "white",
                mec="white" if hold else col, mew=0.6 if hold else 0.9, zorder=3,
                label=lab + ("" if fam != "ps" else (", hold" if hold else ", cubic")))
        for x, lo, hi in zip(xs, los, his):
            ax.plot([x, x], [lo, hi], color=col, lw=0.6, alpha=0.35, zorder=2)
    ax.set_xscale("log"); ax.set_xticks([5, 10, 17, 25, 33, 50, 83, 167])
    ax.set_xticklabels(["5", "10", "17", "", "33", "50", "83", "167"], fontsize=6.6)
    ax.minorticks_off(); ax.set_xlabel("Plans per second", labelpad=9); ax.set_ylabel("Success rate", labelpad=2)
    ax.set_ylim(-0.03, 1.05); ax.set_yticks([0, 0.5, 1.0])
    ax.axvline(33, color=F.C_GRAY, lw=6, alpha=0.18, zorder=0)
    ax.text(33, 1.07, "robot", ha="center", va="bottom", fontsize=6.3, color=F.C_INK2)
    ax.text(25, -0.19, "25", ha="center", va="top", fontsize=6.6, color=F.C_INK2, clip_on=False)
    ax.plot([25, 25], [-0.03, -0.12], color=F.C_INK2, lw=0.6, clip_on=False, zorder=1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.30), ncol=2, handlelength=1.6, columnspacing=0.8,
              handletextpad=0.4, labelspacing=0.35, fontsize=6.6)
    F.style_axes(ax)
    save(fig, out, "panel_rate")


# ---- (d) basin -----------------------------------------------------------------
def panel_basin(sums, out):
    rates = [(20, "25"), (15, "33"), (10, "50"), (3, "167")]
    sums = dict(sums); sums.setdefault(20, F.load("runs/summary_floor_spp20.json"))
    CELL, Wd, Hd = 0.18, W["d"], H2
    fig = plt.figure(figsize=(Wd, Hd))
    x0, gap, y0 = 0.45, 0.17, 0.35
    titles = {"elite mean": "CEM, elite mean", "argmin": "PS, argmin", "softmax": "MPPI, softmax"}
    for j, rule in enumerate(["elite mean", "argmin", "softmax"]):
        ax = fig.add_axes([(x0 + j * (4 * CELL + gap)) / Wd, y0 / Hd, 4 * CELL / Wd, 5 * CELL / Hd])
        grid = [[F.counts(sums[spp], arm) if sums[spp]["runs"] else None for spp, _ in rates]
                for arm in F.SIG_ARMS[rule]]
        F._cells(ax, grid, F.RULE_COL[rule], [h for _, h in rates],
                 ["%g" % x for x in F.SIG] if j == 0 else [""] * 5, title=titles[rule])
        if j == 0:
            ax.set_ylabel("Sampling std σ (rad)", labelpad=2)
        if j == 1:
            ax.set_xlabel("Plans per second", labelpad=1)
    save(fig, out, "panel_basin")


# ---- (e) model error -------------------------------------------------------------
def _mis_axis(ax, axis, xlabel, show_ytick):
    sets = [(x, F.load(p)) for x, p in F.MIS_SETS[axis]]
    xs_all = [x for x, _ in sets]
    for j, (arm, label, col, mk, ls) in enumerate(F.MIS_RULES):
        xs, ys, los, his = [], [], [], []
        for x, S in sets:
            k, n = F.counts(S, arm)
            if not n:
                continue
            xs.append(x); ys.append(k / n); lo, hi = F.wilson(k, n); los.append(lo); his.append(hi)
        xs = np.array(xs) + (j - 1.5) * 0.012 * (max(xs_all) - min(xs_all))
        ax.plot(xs, ys, color=col, lw=1.1, ls=ls, marker=mk, ms=3.6, mec="white", mew=0.5, zorder=3)
        for x, lo, hi in zip(xs, los, his):
            ax.plot([x, x], [lo, hi], color=col, lw=0.5, alpha=0.4, zorder=2)
    ax.set_xticks(xs_all)
    ax.set_xticklabels(["%g" % x if (axis != "mass" or x in (1.0, 1.1, 1.2)) else "" for x in xs_all], fontsize=6.6)
    if axis == "mu":
        ax.invert_xaxis()
    ax.set_xlabel(xlabel, labelpad=1.5)
    ax.set_ylim(-0.03, 1.05); ax.set_yticks([0, 0.5, 1.0])
    if not show_ytick:
        ax.set_yticklabels([])
    ax.tick_params(pad=1.5)
    F.style_axes(ax)


def panel_mismatch(out):
    We, He = W["e"], H2
    fig = plt.figure(figsize=(We, He))
    x0, pw, gap, y0, ph = 0.40, 0.62, 0.15, 0.36, 1.05
    for j, (axis, xlabel) in enumerate([("mass", "Plant mass ×"), ("kp", "Plant joint kp ×"), ("mu", "Plant friction ×")]):
        ax = fig.add_axes([(x0 + j * (pw + gap)) / We, y0 / He, pw / We, ph / He])
        _mis_axis(ax, axis, xlabel, show_ytick=(j == 0))
        if j == 0:
            ax.set_ylabel("Success rate", labelpad=2)
    save(fig, out, "panel_mismatch")


# ---- (f) sigma link ----------------------------------------------------------------
def panel_sigma(out):
    Wf, Hf = W["f"], H2
    fig = plt.figure(figsize=(Wf, Hf))
    ax = fig.add_axes([0.30 / Wf, 0.36 / Hf, (Wf - 0.36) / Wf, 1.05 / Hf])
    base = F.load("runs/summary_gains_spp15.json"); mis = F.load("runs/summary_mismatch_spp15_m1.10.json")
    for j, (arm, label, col, mk, ls) in enumerate(F.MIS_RULES):
        off = (j - 1.5) * 0.0006
        xb, yb, xm, ym = [], [], [], []
        for sig, a in F.SIGMA_LINK[arm]:
            kb, nb = F.counts(base, a); km, nm = F.counts(mis, a)
            if not nb or not nm:
                continue
            xb.append(sig + off); yb.append(kb / nb); xm.append(sig + off); ym.append(km / nm)
            lo, hi = F.wilson(km, nm)
            ax.plot([sig + off] * 2, [lo, hi], color=col, lw=0.5, alpha=0.4, zorder=2)
        ax.plot(xm, ym, color=col, lw=1.1, ls=ls, marker=mk, ms=3.6, mec="white", mew=0.5, zorder=3)
        ax.plot(xb, yb, color=col, lw=0, marker=mk, ms=3.6, mfc="white", mec=col, mew=0.8, zorder=3)
        for x, a_, b_ in zip(xb, yb, ym):
            ax.plot([x, x], [b_, a_], color=col, lw=0.5, ls=":", zorder=2)
    ax.set_xticks([0.01, 0.02, 0.03]); ax.set_xticklabels(["0.01", "0.02", "0.03"], fontsize=6.6); ax.set_xlim(0.006, 0.034)
    ax.set_xlabel("Sampling std σ (rad)", labelpad=1.5)
    ax.set_ylim(-0.03, 1.05); ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels([]); ax.tick_params(pad=1.5)
    F.style_axes(ax)
    save(fig, out, "panel_sigma")


# ---- (extra) k / lambda / N strips ---------------------------------------------------
def panel_strips(sums, out):
    CELL, Ws, Hs = 0.19, W["d"], 1.05
    fig = plt.figure(figsize=(Ws, Hs))
    x0, y0 = 0.50, 0.33
    ks = [(1, "cem_ne1"), (2, "cem_ne2"), (6, "cem"), (10, "cem_ne10"), (20, "cem_ne20")]
    ax = fig.add_axes([x0 / Ws, y0 / Hs, 5 * CELL / Ws, 2 * CELL / Hs])
    grid = [[F.counts(sums[spp], arm) if sums[spp]["runs"] else None for _, arm in ks] for spp in (15, 10)]
    F._cells(ax, grid, F.C_ELITE, [str(k) for k, _ in ks], ["33", "50"], title="CEM: elites k")
    ax.set_xlabel("k", labelpad=1); ax.set_ylabel("Plans/s", labelpad=2)
    ls = [(0.1, "mppi_raw01_zero_l0.1"), (1, "mppi_raw01_zero_l1"), (10, "mppi_raw01_zero_l10")]
    xl = x0 + 5 * CELL + 0.14
    ax = fig.add_axes([xl / Ws, y0 / Hs, 3 * CELL / Ws, 2 * CELL / Hs])
    grid = [[F.counts(sums[spp], arm) if sums[spp]["runs"] else None for _, arm in ls] for spp in (15, 10)]
    F._cells(ax, grid, F.C_SOFTMAX, ["%g" % l for l, _ in ls], ["", ""], title="MPPI: λ")
    ax.set_xlabel("λ", labelpad=1)
    Ns = [(8, ["cem_n8_ne2", "ps_raw01_zero_n8", "mppi_raw01_zero_l1_n8"]),
          (20, ["cem", "ps_raw01_zero", "mppi_raw01_zero_l1"]),
          (40, ["cem_n40_ne12", "ps_raw01_zero_n40", "mppi_raw01_zero_l1_n40"])]
    xn = xl + 3 * CELL + 0.42
    ax = fig.add_axes([xn / Ws, (y0 - CELL) / Hs, 3 * CELL / Ws, 3 * CELL / Hs])
    grid = [[F.counts(sums[15], arm) if sums[15]["runs"] else None for arm in arms] for _, arms in Ns]
    F._cells(ax, grid, F.C_INK2, ["CEM", "PS", "MPPI"], [str(n) for n, _ in Ns], title="Rollouts N, 33 plans/s")
    ax.set_ylabel("N", labelpad=2)
    save(fig, out, "panel_strips")


def main():
    out = os.path.join(HERE, "paper_panels"); os.makedirs(out, exist_ok=True)
    sums = {spp: F.load("runs/summary_deploy_spp%d.json" % spp) for spp in (3, 6, 10, 15)}
    panel_ladder(sums, out)
    panel_persist(sums[15], out)
    panel_rate(out)
    panel_basin(sums, out)
    panel_mismatch(out)
    panel_sigma(out)
    panel_strips(sums, out)
    dst = os.path.normpath(os.path.join(HERE, "..", "..", "paper", "figures", "planner", "panels"))
    os.makedirs(dst, exist_ok=True)
    for f in sorted(os.listdir(out)):
        if f.endswith(".pdf"):
            shutil.copy(os.path.join(out, f), dst)
    print("->", out, "and", dst)


if __name__ == "__main__":
    main()
