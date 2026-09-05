#!/usr/bin/env python3
"""Turn a table-height sweep into the figures and summary JSON the page reads.

Input:  <runs>/summary.csv + one <tag>.csv per run (lean_bench columns).
Output: <out>/fig_*.png (+ .dark.png), summary.json, agg.json.

Conventions that matter for reading the figures:
  * `f_forearm` is `left_elbow_link` -- the `left_forearm_pad` capsule is a geom
    on it (checked against the compiled model, not assumed). `f_wrist` is the
    left wrist pad's link. Both get their own column because a brace silently
    carried by the wrist instead of the forearm is a failure this task has
    produced before, and a residual would hide it.
  * `f_robot_total` excludes table-vs-floor and table-vs-object contacts. Before
    that filter the "total" carried the table's own ~166 N of leg weight.
  * Load statistics are taken over the SEATED window (pad within 5 mm of the
    face), not the whole brace phase. A median over the whole phase is mostly
    approach with no contact and reads 0 N even when the forearm peaks at 168 N.
  * MJPC's planner is not run-to-run deterministic (measured: two identical
    invocations diverged by 3.9 s in phase-3 entry), so every per-height number
    is an aggregate over seeds and single runs are drawn but never claimed.
"""
import argparse, csv, json, math, os
from collections import OrderedDict, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CAT = OrderedDict([("left forearm", "#2a78d6"), ("left wrist", "#eb6834"),
                   ("right arm", "#1baf7a"), ("torso", "#4a3aa7"),
                   ("pelvis", "#d03b3b"), ("other", "#898781")])
STATUS = OrderedDict([("complete", "#0ca30c"), ("stalled", "#fab219"),
                      ("fell", "#d03b3b")])
INK = {"light": dict(surface="#fcfcfb", fg="#0b0b0b", mid="#52514e",
                     muted="#898781", grid="#e1e0d9", axis="#c3c2b7"),
       "dark": dict(surface="#1a1a19", fg="#ffffff", mid="#c3c2b7",
                    muted="#898781", grid="#2c2c2a", axis="#383835")}

BRACE_PHASES = ("forearm_brace_lean", "forearm_brace_release")
SEATED_M = 0.005
RIGHT_ARM = ("f_r_elbow", "f_r_wrist", "f_r_gripper")


def ramp(n):
    if n <= 1:
        return [SEQ[3]]
    return [SEQ[int(round(i))] for i in np.linspace(1, len(SEQ) - 1, n)]


def style(mode):
    c = INK[mode]
    plt.rcParams.update({
        "figure.facecolor": c["surface"], "axes.facecolor": c["surface"],
        "savefig.facecolor": c["surface"], "text.color": c["fg"],
        "axes.labelcolor": c["mid"], "axes.edgecolor": c["axis"],
        "xtick.color": c["muted"], "ytick.color": c["muted"],
        "grid.color": c["grid"], "font.size": 10, "font.family": "sans-serif",
        "axes.spines.top": False, "axes.spines.right": False})
    return c


def read_run(path):
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("t") in (None, "") or r.get("cost") in (None, ""):
                continue
            d, ok = {"phase_name": r.get("phase_name") or "?"}, True
            for k, v in r.items():
                if k == "phase_name":
                    continue
                try:
                    d[k] = float(v)
                except (TypeError, ValueError):
                    d[k] = math.nan
                    if k in ("t", "cost"):
                        ok = False
            if ok:
                out.append(d)
    return out


def col(rows, k):
    return np.array([r.get(k, math.nan) for r in rows], float)


def load_sweep(rundir):
    runs = []
    with open(os.path.join(rundir, "summary.csv")) as f:
        for r in csv.DictReader(f):
            if not r.get("csv") or not os.path.exists(r["csv"]):
                continue
            rows = read_run(r["csv"])
            if not rows:
                continue
            fell, comp = r.get("fell") == "1", r.get("complete") == "1"
            runs.append(dict(
                h=float(r["table_h"]), seed=int(r["seed"]), fell=fell,
                complete=comp, t_complete=float(r.get("t_complete") or -1),
                t_end=float(r.get("t_end") or 0), wall_s=float(r.get("wall_s") or 0),
                rows=rows, csv=r["csv"],
                qpos=r["csv"].replace(".csv", ".qpos.csv"),
                outcome="fell" if fell else ("complete" if comp else "stalled")))
    runs.sort(key=lambda d: (d["h"], d["seed"]))
    return runs


def brace_mask(rows):
    return np.array([r["phase_name"] in BRACE_PHASES for r in rows], bool)


def seated_mask(rows):
    """Pad at face level AND actually carrying load.

    ★ 2026-09-04 the height test alone is not seating. `pad_clear` is
    (pad z - radius) - face z, so an arm hanging BESIDE a too-high slab reads
    <= 0 and scores as perfectly seated: h = 1.085 came out "seated 100%" while
    every contact column read 0.00 N, because the forearm was below the face,
    outboard of the wood. Requiring contact makes the statistic mean what its
    name says, and the gap between the two masks is itself the diagnostic --
    see `at_face_mask`.
    """
    pc = col(rows, "pad_clear")
    fwd = np.nan_to_num(col(rows, "f_forearm"))
    return brace_mask(rows) & ~np.isnan(pc) & (pc <= SEATED_M) & (fwd > 0.0)


def at_face_mask(rows):
    """Geometric only: pad within SEATED_M of the face plane, load ignored."""
    pc = col(rows, "pad_clear")
    return brace_mask(rows) & ~np.isnan(pc) & (pc <= SEATED_M)


def right_arm(rows):
    return sum(np.nan_to_num(col(rows, k)) for k in RIGHT_ARM)


def reach_err(rows):
    d = np.zeros(len(rows))
    for ax in ("x", "y", "z"):
        d += (col(rows, "rhand_" + ax) - col(rows, "tgt_" + ax)) ** 2
    return np.sqrt(d)


def hlabel(h):
    return "%.3f m" % h


# ---------------------------------------------------------------- figures ----
def fig_outcome(agg, out, mode):
    c = style(mode)
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    hs = [a["h"] for a in agg]
    x = np.arange(len(hs))
    bottom = np.zeros(len(hs))
    for k, colr in STATUS.items():
        v = np.array([a["counts"].get(k, 0) for a in agg], float)
        ax.bar(x, v, 0.6, bottom=bottom, color=colr, label=k,
               edgecolor=c["surface"], linewidth=2)
        for xi, (b, vv) in enumerate(zip(bottom, v)):
            if vv:
                ax.text(xi, b + vv / 2, "%d" % vv, ha="center", va="center",
                        fontsize=9, color="#fff", fontweight="bold")
        bottom += v
    ax.set_xticks(x, [hlabel(h) for h in hs])
    ax.set_xlabel("table face height")
    ax.set_ylabel("runs")
    ax.set_yticks(range(int(bottom.max()) + 1))
    # Every bar is full height, so the legend needs its own band above them.
    ax.set_ylim(0, bottom.max() * 1.30)
    ax.grid(axis="y", lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, ncols=3, loc="upper center")
    ax.set_title("Outcome by table height (%d seeds each)" % agg[0]["n"],
                 loc="left", fontsize=12, color=c["fg"], pad=12)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_ladder(runs, out, mode):
    c = style(mode)
    fig, ax = plt.subplots(figsize=(9.2, 0.34 * len(runs) + 2.2))
    names = []
    for r in runs:
        for row in r["rows"]:
            if row["phase_name"] not in names:
                names.append(row["phase_name"])
    steps = ramp(max(2, len(names)))
    colors = {n: steps[min(i, len(steps) - 1)] for i, n in enumerate(names)}
    for j, r in enumerate(runs):
        t = col(r["rows"], "t")
        st, nm0 = t[0], r["rows"][0]["phase_name"]
        for i in range(1, len(t) + 1):
            nm = r["rows"][i]["phase_name"] if i < len(t) else None
            if nm != nm0:
                end = t[i] if i < len(t) else t[-1]
                ax.barh(j, end - st, left=st, height=0.62, color=colors[nm0],
                        edgecolor=c["surface"], linewidth=1.0)
                if nm is None:
                    break
                st, nm0 = end, nm
        ax.scatter([r["t_end"]], [j], s=40, zorder=5, color=STATUS[r["outcome"]],
                   marker={"complete": "o", "stalled": "s", "fell": "X"}[r["outcome"]],
                   edgecolor=c["surface"], linewidth=1.0)
    ax.set_yticks(range(len(runs)),
                  ["%.3f s%d" % (r["h"], r["seed"]) for r in runs], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("sim time (s)")
    ax.set_ylabel("table face height / seed")
    ax.grid(axis="x", lw=0.7, alpha=0.9); ax.set_axisbelow(True)
    ax.set_title("How far the phase ladder got, per run", loc="left",
                 fontsize=12, color=c["fg"], pad=12)
    hs = [plt.Line2D([], [], marker=m, ls="", color=STATUS[k], markersize=7, label=k)
          for k, m in (("complete", "o"), ("stalled", "s"), ("fell", "X"))]
    ax.legend(handles=hs, loc="lower right", frameon=False, fontsize=8.5, ncols=3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_load(agg, out, mode):
    c = style(mode)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(agg))
    bottom = np.zeros(len(agg))
    for lab in CAT:
        v = np.array([a["load"].get(lab) or 0.0 for a in agg], float)
        ax.bar(x, v, 0.62, bottom=bottom, label=lab, color=CAT[lab],
               edgecolor=c["surface"], linewidth=2)
        bottom += v
    for xi, (tot, a) in enumerate(zip(bottom, agg)):
        lbl = "%.0f N" % tot if tot > 0 else "never seated"
        ax.text(xi, tot + 2, lbl, ha="center", va="bottom", fontsize=8.5,
                color=c["mid"] if tot > 0 else "#d03b3b")
    ax.set_xticks(x, [hlabel(a["h"]) for a in agg])
    ax.set_xlabel("table face height")
    ax.set_ylabel("median table load while in contact (N)")
    ax.grid(axis="y", lw=0.7, alpha=0.9); ax.set_axisbelow(True)
    # Headroom for the total labels, and the legend parked over the short bars
    # on the right rather than on top of the tallest one.
    ax.set_ylim(top=max(bottom.max() * 1.18, 1.0))
    ax.legend(frameon=False, fontsize=9, ncols=2, loc="upper right")
    ax.set_title("Which body carries the brace, by table height", loc="left",
                 fontsize=12, color=c["fg"], pad=12)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_series(runs, getter, out, mode, ylabel, title, zero_label=None,
               hline=None, hline_label=None, ylim=None):
    """Every run drawn; colour is the height, seed 0 solid and the rest faint."""
    c = style(mode)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    hs = sorted({r["h"] for r in runs})
    cmap = dict(zip(hs, ramp(len(hs))))
    seen = set()
    for r in runs:
        t, v = col(r["rows"], "t"), getter(r["rows"])
        good = ~np.isnan(v)
        if not good.any():
            continue
        first = r["h"] not in seen
        seen.add(r["h"])
        ax.plot(t[good], v[good], lw=2.0 if r["seed"] == 0 else 1.0,
                alpha=1.0 if r["seed"] == 0 else 0.45, color=cmap[r["h"]],
                solid_capstyle="round", label=hlabel(r["h"]) if first else None)
    # Clip BEFORE annotating so the labels land inside the visible band. A fall
    # drives these traces to -840 mm, which flattens the +-50 mm region the
    # seating question actually lives in; the caption says the axis is clipped.
    if ylim:
        ax.set_ylim(*ylim)
    if zero_label:
        ax.axhline(0, lw=1.2, color=c["axis"], zorder=1)
        ax.annotate(zero_label, (ax.get_xlim()[1], 0), fontsize=8,
                    color=c["muted"], va="bottom", ha="right")
    if hline is not None:
        ax.axhline(hline, lw=1.4, ls=(0, (5, 3)), color="#d03b3b", zorder=1)
        ax.annotate(hline_label or "", (ax.get_xlim()[0], hline), fontsize=8,
                    color="#d03b3b", va="bottom", ha="left")
    ax.set_xlabel("sim time (s)"); ax.set_ylabel(ylabel)
    ax.grid(lw=0.7, alpha=0.9); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, ncols=3, title="table face height",
              title_fontsize=8.5, loc="best")
    ax.set_title(title, loc="left", fontsize=12, color=c["fg"], pad=12)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def run_record(r):
    b, s = brace_mask(r["rows"]), seated_mask(r["rows"])

    def st(v, mask, f=np.nanmedian):
        v = v[mask]
        v = v[~np.isnan(v)]
        return None if not len(v) else float(f(v))

    pc = col(r["rows"], "pad_clear")
    re_ = reach_err(r["rows"])
    return dict(
        table_h=r["h"], seed=r["seed"], outcome=r["outcome"], t_end=r["t_end"],
        t_complete=r["t_complete"], wall_s=r["wall_s"],
        brace_seconds=float(b.sum() * 0.02), seated_seconds=float(s.sum() * 0.02),
        seated_fraction=(float(s.sum() / b.sum()) if b.sum() else None),
        at_face_fraction=(float(at_face_mask(r["rows"]).sum() / b.sum())
                          if b.sum() else None),
        pad_clear_min_mm=(None if st(pc, b, np.nanmin) is None
                          else 1000 * st(pc, b, np.nanmin)),
        f_forearm_seated=st(col(r["rows"], "f_forearm"), s),
        f_wrist_seated=st(col(r["rows"], "f_wrist"), s),
        f_rightarm_seated=st(right_arm(r["rows"]), s),
        f_torso_seated=st(col(r["rows"], "f_torso"), s),
        f_pelvis_seated=st(col(r["rows"], "f_pelvis"), s),
        f_other_seated=st(col(r["rows"], "f_other"), s),
        f_forearm_peak=st(col(r["rows"], "f_forearm"), b, np.nanmax),
        f_total_peak=st(col(r["rows"], "f_robot_total"), b, np.nanmax),
        com_beyond_peak_mm=(None if st(col(r["rows"], "com_beyond_foot_edge"), b, np.nanmax) is None
                            else 1000 * st(col(r["rows"], "com_beyond_foot_edge"), b, np.nanmax)),
        reach_err_min_mm=(None if st(re_, b, np.nanmin) is None
                          else 1000 * st(re_, b, np.nanmin)),
        torso_tilt_peak_deg=st(col(r["rows"], "torso_tilt_deg"), b, np.nanmax),
        qpos=os.path.basename(r["qpos"]) if os.path.exists(r["qpos"]) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = load_sweep(a.runs)
    if not runs:
        raise SystemExit("no runs found in %s" % a.runs)
    recs = [run_record(r) for r in runs]

    by_h = defaultdict(list)
    for rec in recs:
        by_h[rec["table_h"]].append(rec)
    agg = []
    for h in sorted(by_h):
        g = by_h[h]
        counts = defaultdict(int)
        for rec in g:
            counts[rec["outcome"]] += 1

        def m(key):
            v = [rec[key] for rec in g if rec[key] is not None]
            return float(np.median(v)) if v else None
        agg.append(dict(
            h=h, n=len(g), counts=dict(counts),
            complete=counts.get("complete", 0),
            load={"left forearm": m("f_forearm_seated"),
                  "left wrist": m("f_wrist_seated"),
                  "right arm": m("f_rightarm_seated"),
                  "torso": m("f_torso_seated"), "pelvis": m("f_pelvis_seated"),
                  "other": m("f_other_seated")},
            seated_fraction=m("seated_fraction"),
            at_face_fraction=m("at_face_fraction"),
            pad_clear_min_mm=m("pad_clear_min_mm"),
            f_forearm_peak=m("f_forearm_peak"), f_total_peak=m("f_total_peak"),
            com_beyond_peak_mm=m("com_beyond_peak_mm"),
            reach_err_min_mm=m("reach_err_min_mm"),
            torso_tilt_peak_deg=m("torso_tilt_peak_deg"),
            t_end=m("t_end")))

    for mode in ("light", "dark"):
        sfx = ".png" if mode == "light" else ".dark.png"
        fig_outcome(agg, os.path.join(a.out, "fig_outcome" + sfx), mode)
        fig_ladder(runs, os.path.join(a.out, "fig_ladder" + sfx), mode)
        fig_load(agg, os.path.join(a.out, "fig_load" + sfx), mode)
        fig_series(runs, lambda rw: col(rw, "pad_clear") * 1000,
                   os.path.join(a.out, "fig_pad" + sfx), mode,
                   "forearm pad clearance above the face (mm)",
                   "Did the forearm seat? Pad clearance over the slab",
                   zero_label="0 = pad flat on the wood", ylim=(-260, 360))
        fig_series(runs, lambda rw: col(rw, "com_beyond_foot_edge") * 1000,
                   os.path.join(a.out, "fig_balance" + sfx), mode,
                   "CoM beyond the front foot edge (mm)",
                   "Balance margin: how far the CoM goes past the toes",
                   hline=145.0,
                   hline_label="com_cap_fwd = 145 mm (a soft cost, not a limit)",
                   ylim=(-260, 700))
        fig_series(runs, lambda rw: reach_err(rw) * 1000,
                   os.path.join(a.out, "fig_reach" + sfx), mode,
                   "right gripper to target mocap (mm)",
                   "Reach error to the target, which rides the slab")

    json.dump(recs, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    json.dump(agg, open(os.path.join(a.out, "agg.json"), "w"), indent=1)
    print("%d runs, %d heights" % (len(runs), len(agg)))
    for x in agg:
        print("  %.3f  n=%d  complete=%d  in-contact=%s  at-face=%s  "
              "forearm=%s N  peak=%s N"
              % (x["h"], x["n"], x["complete"],
                 "--" if x["seated_fraction"] is None else "%.0f%%" % (100 * x["seated_fraction"]),
                 "--" if x["at_face_fraction"] is None else "%.0f%%" % (100 * x["at_face_fraction"]),
                 "--" if x["load"]["left forearm"] is None else "%.0f" % x["load"]["left forearm"],
                 "--" if x["f_forearm_peak"] is None else "%.0f" % x["f_forearm_peak"]))


if __name__ == "__main__":
    main()
