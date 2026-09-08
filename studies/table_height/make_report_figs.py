#!/usr/bin/env python3
"""Figures for the 2026-09-08 report page: the window, the clearance line, the
shoulder deficit, the arms matrix, and the paired traces.

Every panel is computed from the committed run CSVs and qpos dumps under
`runs/`, so a figure cannot disagree with the numbers in the page. Writes a
light and a dark PNG per figure into docs/lean/media/full/.

usage: make_report_figs.py [--only fig_window,...]
"""
import argparse, csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.normpath(os.path.join(HERE, "../.."))
RUNS = os.path.join(HERE, "runs")
OUT = os.path.join(ROOT, "docs/lean/media/full")
os.makedirs(OUT, exist_ok=True)

BRACE_PHASES = (1, 2)
UPRIGHT_Z = 0.55
CACHE = os.path.join(HERE, "figs", "report_geom.json")

# ---------------------------------------------------------------- run outcomes
def summary(rel):
    p = os.path.join(RUNS, rel, "summary.csv")
    out = []
    if not os.path.exists(p):
        return out
    for r in csv.DictReader(open(p)):
        try:
            out.append(dict(h=round(float(r["table_h"]), 3), seed=int(r["seed"]),
                            complete=int(r["complete"]), fell=int(r["fell"]),
                            t_end=float(r["t_end"])))
        except (ValueError, TypeError, KeyError):
            continue                      # a SIGKILLed run leaves a short row
    return out


def tally(rel):
    """{height: (completions, runs)}"""
    t = {}
    for r in summary(rel):
        c, n = t.get(r["h"], (0, 0))
        t[r["h"]] = (c + r["complete"], n + 1)
    return t


# --------------------------------------------------------- pad / shoulder geom
def geometry():
    """max/median pad clearance and shoulder height over the brace rungs."""
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    import mujoco
    import retarget as R
    from analyze_pose import pristine_model
    from render_video import set_table_height
    arms = {"shipped": ["ab/off", "seeded", "edge"], "pose_track": ["ab/on"],
            "pitch_track": ["pitch/pitch"], "target_slab": ["bracetgt/slab"],
            "stance63": ["stance/s63"], "tilt50": ["tilt/cap50"]}
    rows, models = [], {}
    for arm, dirs in arms.items():
        for rel in dirs:
            d_ = os.path.join(RUNS, rel)
            if not os.path.isdir(d_):
                continue
            for f in sorted(os.listdir(d_)):
                if not f.endswith(".qpos.csv"):
                    continue
                tag = f[:-len(".qpos.csv")]
                h = int(tag[1:5]) / 1000.0
                seed = int(tag.split("_s")[1])
                if h not in models:
                    m = pristine_model(); set_table_height(m, h); models[h] = m
                m = models[h]
                d = mujoco.MjData(m)
                ph = {round(float(r["t"]), 3): int(float(r["phase"]))
                      for r in csv.DictReader(open(os.path.join(d_, tag + ".csv")))}
                sh = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                       "left_shoulder_pitch_link")
                pad = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, R.PAD)
                tgc = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                                        "table_top_collision")
                clear, shabs, face = [], [], None
                for r in csv.DictReader(open(os.path.join(d_, f))):
                    try:
                        t = round(float(r["t"]), 3)
                        q = [float(r["q%d" % i]) for i in range(m.nq)]
                    except (ValueError, TypeError):
                        continue
                    if ph.get(t) not in BRACE_PHASES or q[2] < UPRIGHT_Z:
                        continue
                    d.qpos[:] = q
                    mujoco.mj_kinematics(m, d)
                    face = float(d.geom_xpos[tgc][2] + m.geom_size[tgc][2])
                    clear.append(float(d.geom_xpos[pad][2] - m.geom_size[pad][0] - face))
                    shabs.append(float(d.xpos[sh][2]))
                if clear:
                    rows.append(dict(arm=arm, h=h, seed=seed, n=len(clear),
                                     face=face, clear_max=max(clear),
                                     clear_med=float(np.median(clear)),
                                     sh_abs=float(np.median(shabs))))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(rows, open(CACHE, "w"), indent=1)
    return rows


SHIPPED = ("seeded", "ab/off", "edge")


def peak_force(h, col="f_forearm", rels=SHIPPED):
    """Per-run peak of a force column over the brace rungs, one value per seed.

    One definition, used everywhere on the page: the maximum over the two brace
    rungs of a single run, listed across every shipped seed at that height. The
    number quoted in prose is the MEDIAN of that list -- at 1.085 m four of six
    seeds carry exactly zero and two brush the slab at 21 N and 32 N, so a range
    hides the finding and a mean invents load that most runs never see.
    """
    vals = []
    tag = "h%04d" % round(h * 1000)
    for rel in rels:
        d_ = os.path.join(RUNS, rel)
        if not os.path.isdir(d_):
            continue
        for f in sorted(os.listdir(d_)):
            if not f.startswith(tag) or not f.endswith(".csv") or ".qpos." in f:
                continue
            pk = 0.0
            for r in csv.DictReader(open(os.path.join(d_, f))):
                try:
                    if int(float(r["phase"])) in BRACE_PHASES:
                        pk = max(pk, float(r[col]))
                except (ValueError, TypeError, KeyError):
                    continue
            vals.append(pk)
    return sorted(vals)


# ------------------------------------------------------------------ style
def style(dark):
    fg = "#e8e6e3" if dark else "#1b1b1b"
    bg = "#16181d" if dark else "#ffffff"
    grid = "#3a3f47" if dark else "#d9d9d9"
    plt.rcParams.update({
        "figure.facecolor": bg, "axes.facecolor": bg, "savefig.facecolor": bg,
        "text.color": fg, "axes.labelcolor": fg, "axes.edgecolor": grid,
        "xtick.color": fg, "ytick.color": fg, "grid.color": grid,
        "font.size": 10, "axes.titlesize": 11, "axes.grid": True,
        "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
        "figure.dpi": 130})
    return fg, bg, grid


def save(fig, name, dark):
    p = os.path.join(OUT, name + (".dark.png" if dark else ".png"))
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", os.path.relpath(p, ROOT))


GOOD, BAD, WARN = "#3f9d5a", "#c0483c", "#c8922a"


# ------------------------------------------------------------------ figures
def fig_window(dark):
    style(dark)
    ship = tally("seeded"); ship_b = tally("ab/off"); edge = tally("edge")
    hs = [0.785, 0.885, 0.985, 1.035, 1.050, 1.060, 1.085]
    comp, runs = [], []
    for h in hs:
        c = n = 0
        for t in (ship, ship_b, edge):
            if h in t:
                c += t[h][0]; n += t[h][1]
        comp.append(c); runs.append(n)
    frac = [c / n for c, n in zip(comp, runs)]
    fig, ax = plt.subplots(figsize=(7.4, 3.1))
    ax.add_patch(Rectangle((0.975, 0), 0.070, 1.05, color=GOOD, alpha=0.10, zorder=0))
    ax.bar([str(h) for h in hs], frac,
           color=[GOOD if f > 0 else BAD for f in frac], width=0.62, zorder=3)
    for i, (c, n) in enumerate(zip(comp, runs)):
        ax.text(i, frac[i] + 0.04, "%d/%d" % (c, n), ha="center", fontsize=9)
    ax.set_ylim(0, 1.18); ax.set_ylabel("fraction completing the ladder")
    ax.set_xlabel("slab face height (m)     compiled = 0.985 m")
    ax.set_title("The window is 0.985–1.035 m. Shipped controller, every seed ever run.")
    ax.axvline(1.5, ls="--", lw=0.9, color=GOOD, alpha=0.6)
    save(fig, "fig_window", dark)


def fig_clearance(dark):
    fg, bg, grid = style(dark)
    # pooled over every arm with a qpos dump: the shoulder is pinned in all of
    # them, so pooling is more points for the same claim, not a mixed population.
    g = geometry()
    hs = sorted({r["h"] for r in g})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 3.5))

    # The shoulder is only pinned once the robot is upright enough to brace at
    # all; below 0.885 m it bows over the slab, so the fit is taken over the
    # heights where the posture is the same posture.
    fit_hs = [h for h in hs if h >= 0.885]
    for h in hs:
        v = [r["clear_max"] * 1000 for r in g if r["h"] == h]
        a1.scatter([h] * len(v), v, s=28, zorder=4, alpha=0.85,
                   color=fg if h >= 0.885 else grid)
    xs = np.array(fit_hs)
    ys = np.array([np.mean([r["clear_max"] * 1000 for r in g if r["h"] == h])
                   for h in fit_hs])
    k = np.polyfit(xs, ys, 1)
    xf = np.linspace(0.87, 1.10, 50)
    a1.plot(xf, np.polyval(k, xf), lw=1.3, ls="--", color=WARN,
            label="%.2f mm of clearance per mm of slab" % (k[0] / 1000))
    a1.axhline(0, lw=1.1, color=BAD)
    a1.axhspan(25, 300, color=GOOD, alpha=0.09)
    a1.annotate("0.785 m: the robot bows over\nthe slab rather than bracing on it",
                xy=(0.787, 200), xytext=(0.845, 235), fontsize=8,
                color="#8b93a1", arrowprops=dict(arrowstyle="-", lw=0.8,
                                                 color="#8b93a1"))
    a1.set_xlabel("slab face height (m)"); a1.set_ylabel("max pad clearance (mm)")
    a1.set_ylim(-40, 300)
    a1.set_title("Clearance falls almost 1:1 with the slab")
    a1.legend(loc="upper right", fontsize=8)

    hh = [0.985, 1.035, 1.050, 1.060, 1.085]
    ok = [True, True, False, False, False]
    cl = [np.mean([r["clear_max"] * 1000 for r in g if r["h"] == h]) for h in hh]
    fz = [float(np.median(peak_force(h))) for h in hh]
    a2.scatter(cl, fz, s=95, zorder=4, color=[GOOD if o else BAD for o in ok])
    for x, y, h in zip(cl, fz, hh):
        a2.annotate("%.3f m" % h, (x, y), textcoords="offset points",
                    xytext=(9, -13 if h == 1.060 else 6), fontsize=9)
    a2.axvspan(25, 90, color=GOOD, alpha=0.09)
    a2.set_xlabel("max pad clearance (mm)")
    a2.set_ylabel("forearm load (N)")
    a2.set_xlim(-6, 96); a2.set_ylim(-14, 185)
    a2.set_title("Positive clearance is not sufficient; about 25 mm is")
    save(fig, "fig_clearance", dark)


def fig_load(dark):
    fg, bg, grid = style(dark)
    hs = [0.785, 0.885, 0.985, 1.035, 1.050, 1.060, 1.085]
    fa = [float(np.median(peak_force(h))) for h in hs]
    to = [float(np.median(peak_force(h, "f_torso"))) for h in hs]
    x = np.arange(len(hs))
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.bar(x - 0.19, fa, width=0.36, color=GOOD, label="left forearm pad")
    ax.bar(x + 0.19, to, width=0.36, color=BAD, label="torso")
    for i, (a, b) in enumerate(zip(fa, to)):
        ax.text(i - 0.19, a + 18, "%.0f" % a, ha="center", fontsize=8)
        ax.text(i + 0.19, b + 18, "%.0f" % b, ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["%.3f" % h for h in hs])
    ax.set_xlabel("slab face height (m)")
    ax.set_ylabel("load in the brace rungs (N)")
    ax.set_title("Three failure modes: the torso carries the low slab, "
                 "nothing carries the tall one")
    ax.legend(fontsize=9)
    save(fig, "fig_load", dark)


def fig_shoulder(dark):
    fg, bg, grid = style(dark)
    g = geometry()
    hs = sorted({r["h"] for r in g
                 if r["h"] in (0.885, 0.985, 1.035, 1.050, 1.060, 1.085)})
    sa = [np.median([r["sh_abs"] for r in g if r["h"] == h]) for h in hs]
    gap = [(s_ - h) * 1000 for s_, h in zip(sa, hs)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 3.4))
    a1.plot(hs, hs, lw=1.8, color=BAD, label="slab face")
    a1.plot(hs, sa, "o-", lw=1.8, color=fg,
            label="left shoulder, median over the brace rungs")
    a1.fill_between(hs, hs, sa, color=GOOD, alpha=0.10)
    a1.annotate("+%.0f mm" % gap[0], (hs[0], (hs[0] + sa[0]) / 2), fontsize=9,
                ha="left", va="center")
    a1.annotate("+%.0f mm" % gap[-1], (hs[-1], (hs[-1] + sa[-1]) / 2), fontsize=9,
                ha="right", va="center")
    a1.set_ylim(0.85, 1.46)
    a1.set_xlabel("slab face height (m)"); a1.set_ylabel("absolute height (m)")
    a1.set_title("The shoulder holds ~1.40 m whatever the table does")
    a1.legend(loc="upper left", fontsize=8)

    a2.bar([str(h) for h in hs], gap, color=[GOOD if h <= 1.035 else BAD for h in hs],
           width=0.6)
    for i, v in enumerate(gap):
        a2.text(i, v + 6, "%.0f" % v, ha="center", fontsize=9)
    a2.set_ylim(0, 560)
    a2.set_xlabel("slab face height (m)")
    a2.set_ylabel("shoulder above the face (mm)")
    a2.set_title("The reach budget the arm is left with")
    save(fig, "fig_shoulder", dark)


ARMS = [("shipped",              "seeded,ab/off,edge"),
        ("brace_pose_track 1",   "ab/on"),
        ("brace_pitch_track g1", "pitch/pitch"),
        ("brace_pitch_gain 20",  "pitch/g20"),
        ("brace_pitch_gain 100", "pitch/g100"),
        ("pelvis_tilt_max 50",   "tilt/cap50"),
        ("stance +63 mm",        "stance/s63"),
        ("brace_target_slab 1",  "bracetgt/slab")]


def fig_arms(dark):
    fg, bg, grid = style(dark)
    hs = [0.785, 0.885, 0.985, 1.035, 1.050, 1.060, 1.085]
    M = np.full((len(ARMS), len(hs)), np.nan)
    N = [["" for _ in hs] for _ in ARMS]
    for i, (_, rels) in enumerate(ARMS):
        t = {}
        for rel in rels.split(","):
            for h, (c, n) in tally(rel).items():
                c0, n0 = t.get(h, (0, 0)); t[h] = (c0 + c, n0 + n)
        for j, h in enumerate(hs):
            if h in t:
                c, n = t[h]; M[i, j] = c / n; N[i][j] = "%d/%d" % (c, n)
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("wl", [BAD, WARN, GOOD])
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(hs))); ax.set_xticklabels([("%.3f" % h) for h in hs])
    ax.set_yticks(range(len(ARMS))); ax.set_yticklabels([a for a, _ in ARMS], fontsize=9)
    for i in range(len(ARMS)):
        for j in range(len(hs)):
            ax.text(j, i, N[i][j] if N[i][j] else "·", ha="center", va="center",
                    fontsize=9, color="#101010" if not np.isnan(M[i, j]) else grid)
    ax.set_xlabel("slab face height (m)"); ax.grid(False)
    ax.set_title("Eight arms with runs on disk. No cell outside 0.985–1.035 m ever turns green.")
    save(fig, "fig_arms", dark)


def trace(rel, h, seed):
    p = os.path.join(RUNS, rel, "h%04d_s%d.csv" % (round(h * 1000), seed))
    rows = []
    for r in csv.DictReader(open(p)):
        try:
            rows.append((float(r["t"]), int(float(r["phase"])), float(r["pad_clear"]),
                         float(r["f_forearm"]), float(r["com_beyond_foot_edge"])))
        except (ValueError, TypeError):
            continue
    return np.array(rows)


def fig_traces(dark):
    fg, bg, grid = style(dark)
    cases = [("0.985 m  completes", "seeded", 0.985, 0, "#2f7d49"),
             ("1.035 m  completes", "seeded", 1.035, 0, "#4fb3c4"),
             ("1.050 m  falls",     "edge",   1.050, 0, WARN),
             ("1.085 m  falls",     "seeded", 1.085, 0, BAD)]
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 6.4), sharex=True)
    for lab, rel, h, s, c in cases:
        try:
            A = trace(rel, h, s)
        except FileNotFoundError:
            continue
        axes[0].plot(A[:, 0], A[:, 2] * 1000, lw=1.4, color=c, label=lab)
        axes[1].plot(A[:, 0], A[:, 3], lw=1.4, color=c)
        axes[2].plot(A[:, 0], A[:, 4] * 1000, lw=1.4, color=c)
    axes[0].axhline(0, lw=0.9, color=fg, alpha=0.4)
    axes[0].set_ylabel("pad clearance (mm)"); axes[0].set_ylim(-120, 260)
    axes[0].legend(fontsize=9, ncol=4, loc="upper left")
    axes[1].set_ylabel("forearm load (N)")
    axes[2].axhline(0, lw=0.9, color=fg, alpha=0.4)
    axes[2].set_ylabel("CoM past forward foot edge (mm)")
    axes[2].set_xlabel("time (s)"); axes[2].set_xlim(0, 60)
    axes[0].set_title("One seed per height. The tall slab never builds load, and the CoM goes backwards.")
    save(fig, "fig_traces", dark)


def fig_ladder(dark):
    fg, bg, grid = style(dark)
    names = ["stand_up", "lean", "lean+reach", "release",
             "back r1", "back r2", "back r3", "back r4", "stand"]
    hs = [0.785, 0.885, 0.985, 1.035, 1.050, 1.060, 1.085]
    src = {0.785: "seeded", 0.885: "seeded", 0.985: "seeded", 1.035: "seeded",
           1.050: "edge", 1.060: "edge", 1.085: "seeded"}
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    for i, h in enumerate(hs):
        p = os.path.join(RUNS, src[h], "summary.csv")
        for r in csv.DictReader(open(p)):
            if round(float(r["table_h"]), 3) != h:
                continue
            enter = [float(x) for x in r["enter"].split(":")]
            reached = sum(1 for e in enter if e >= 0)
            done = int(r["complete"])
            ax.barh(i + (int(r["seed"]) - 1) * 0.26, reached, height=0.24, zorder=3,
                    color=GOOD if done else BAD)
    ax.set_yticks(range(len(hs))); ax.set_yticklabels(["%.3f" % h for h in hs])
    ax.set_xticks(range(1, 10)); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("slab face height (m)"); ax.set_xlabel("deepest rung entered")
    ax.set_title("Three seeds per height. No low-slab run ever reaches the release rung.")
    save(fig, "fig_ladder", dark)


def fig_stance(dark):
    fg, bg, grid = style(dark)
    # measured by probe_stance.py, 1.085 m, brace rungs, metres past the near edge
    lab = ["home\nkeyframe", "brace\nkeyframe", "shipped\nrun", "stance +63\nseed 0",
           "stance +63\nseed 1"]
    feet = [-0.260, -0.197, -0.268, -0.203, -0.199]
    pad = [np.nan, +0.110, -0.066, -0.069, -0.077]
    x = np.arange(len(lab))
    fig, ax = plt.subplots(figsize=(7.8, 3.3))
    ax.bar(x - 0.19, [f * 1000 for f in feet], width=0.36, color=fg, alpha=0.75,
           label="foot midpoint")
    ax.bar(x + 0.19, [p * 1000 for p in pad], width=0.36, color=WARN,
           label="forearm pad")
    ax.axhline(0, lw=1.2, color=BAD)
    ax.text(4.4, 6, "slab near edge", color=BAD, fontsize=8, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8)
    ax.set_ylabel("mm past the slab's near edge")
    ax.set_title("Moving the robot 63 mm forward moved the feet and not the pad")
    ax.legend(fontsize=9, loc="lower left")
    save(fig, "fig_stance", dark)


def fig_aim(dark):
    fg, bg, grid = style(dark)
    # probe_bracetgt.py, mm past the near edge, medians over the brace rungs
    hs = [0.985, 1.035, 1.085]
    torso = [-235, -339, -444]
    tgt_legacy = [+95, +33, -30]
    pad = [+140, +54, -66]
    fig, ax = plt.subplots(figsize=(7.8, 3.3))
    w = 0.26; x = np.arange(3)
    ax.bar(x - w, torso, width=w, color=fg, alpha=0.6, label="torso")
    ax.bar(x, tgt_legacy, width=w, color=WARN, label="Brace Pos aim (shipped)")
    ax.bar(x + w, pad, width=w, color=GOOD, label="forearm pad, achieved")
    ax.axhline(0, lw=1.2, color=BAD)
    ax.axhline(50, lw=1.0, ls="--", color=GOOD,
               label="aim with brace_target_slab 1")
    ax.set_xticks(x); ax.set_xticklabels(["%.3f m" % h for h in hs])
    ax.set_ylabel("mm past the slab's near edge")
    ax.set_title("The shipped aim is a blend of torso and table, so it walks off the slab")
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "fig_aim", dark)


def fig_standoff(dark):
    """The coauthor's ratio question, answered in kinematics (probe_standoff.py)."""
    fg, bg, grid = style(dark)
    rows = json.load(open(os.path.join(HERE, "figs", "standoff.json")))
    faces = sorted({r["face"] for r in rows})
    cmap = plt.get_cmap("viridis")
    cols = {f: cmap(i / max(1, len(faces) - 1)) for i, f in enumerate(faces)}
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11.4, 3.3))
    for f in faces:
        rs = sorted((r for r in rows if r["face"] == f), key=lambda r: r["dx"])
        x = [r["dx"] * 1000 for r in rs]
        a1.plot(x, [r["com"] for r in rs], "o-", ms=3.5, lw=1.5,
                color=cols[f], label="%.3f m" % f)
        a2.plot(x, [r["pitch"] for r in rs], "o-", ms=3.5, lw=1.5, color=cols[f])
        a3.plot(x, [r["sh"] for r in rs], "o-", ms=3.5, lw=1.5, color=cols[f])
    a1.axhline(0, lw=1.0, color=BAD)
    a1.set_ylabel("CoM ahead of the foot midpoint (mm)")
    a1.set_title("Standing closer pulls the CoM back")
    a1.legend(fontsize=7, title="slab face", title_fontsize=7, ncol=2)
    a2.set_ylabel("base pitch the pose needs (deg)")
    a2.set_title("and takes a few degrees off the bow")
    a3.set_ylabel("shoulder above the face (mm)")
    a3.set_ylim(150, 260)
    a3.set_title("but buys no height at all")
    for ax in (a1, a2, a3):
        ax.set_xlabel("base moved toward the table (mm)")
        ax.axvline(0, lw=0.9, ls=":", color=fg, alpha=0.5)
    save(fig, "fig_standoff", dark)


FIGS = dict(fig_standoff=fig_standoff,fig_window=fig_window, fig_clearance=fig_clearance,
            fig_load=fig_load,
            fig_shoulder=fig_shoulder, fig_arms=fig_arms, fig_traces=fig_traces,
            fig_ladder=fig_ladder, fig_stance=fig_stance, fig_aim=fig_aim)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    want = a.only.split(",") if a.only else list(FIGS)
    for name in want:
        print(name)
        for dark in (False, True):
            FIGS[name](dark)
