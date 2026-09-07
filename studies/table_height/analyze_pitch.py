#!/usr/bin/env python3
"""Score the two 2026-09-06 arms against the shipped and pose-track controls.

Arms
  shipped  runs/ab/off      nothing on
  pose1    runs/ab/on       brace_pose_track 1
  pitch    runs/pitch/pitch brace_pose_track 1 + brace_pitch_track 1
  cap50    runs/tilt/cap50  pelvis_tilt_max_deg 50

Everything here is read off the logs already on disk: the summary CSV for
outcomes and per-step contact/clearance, the qpos CSV for base pitch (the free
joint quaternion, read with the same asin the release gate uses).

usage: analyze_pitch.py [--out figs] [--json figs/pitchtrack.json]
"""
import argparse, csv, json, os, sys
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze import SEQ, STATUS, INK, style

ARMS = OrderedDict([
    ("shipped",  ("runs/ab/off",      "shipped")),
    ("pose1",    ("runs/ab/on",       "brace_pose_track 1")),
    ("pitch",    ("runs/pitch/pitch", "+ brace_pitch_track 1")),
    ("g20",      ("runs/pitch/g20",   "+ pitch_track, gain 20")),
    ("g100",     ("runs/pitch/g100",  "+ pitch_track, gain 100")),
    ("cap50",    ("runs/tilt/cap50",  "pelvis_tilt_max_deg 50")),
])
ARMCOL = {"shipped": "#898781", "pose1": "#9ec5f4", "pitch": "#3987e5",
          "g20": "#184f95", "g100": "#0d366b", "cap50": "#eb6834"}
HEIGHTS = (0.785, 0.885, 0.985, 1.035, 1.085)


def pitch_deg(q):
    s = np.clip(2.0 * (q[3] * q[5] - q[6] * q[4]), -1.0, 1.0)
    return float(np.degrees(np.arcsin(s)))


def required_pitch():
    """Base pitch of the seated brace pose per slab, from probe_pitch.py."""
    p = os.path.join(HERE, "figs", "pitch.json")
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))["heights"]
    return {float(k): float(np.degrees(v["brace"]["pitch"])) for k, v in d.items()}


def tag(h, s):
    return "h%04d_s%d" % (round(h * 1000), s)


def read_one(d, h, s):
    scsv = os.path.join(HERE, d, tag(h, s) + ".csv")
    qcsv = os.path.join(HERE, d, tag(h, s) + ".qpos.csv")
    if not os.path.exists(scsv):
        return None
    rows = [r for r in csv.DictReader(open(scsv)) if r.get("t")]
    if not rows:
        return None
    def col(k, rr):
        v = [float(x[k]) for x in rr if x.get(k) not in (None, "", "nan")]
        return np.array(v) if v else np.array([np.nan])
    r2 = [x for x in rows if x["phase"] == "2"]
    rb = [x for x in rows if x["phase"] in ("1", "2")]
    out = {
        "h": h, "seed": s,
        "t_end": float(rows[-1]["t"]),
        "phases": "".join(sorted({x["phase"] for x in rows})),
        "complete": int("3" in {x["phase"] for x in rows}),
        "n2": len(r2),
        "f_forearm2": float(np.median(col("f_forearm", r2))) if r2 else float("nan"),
        "pad_clear2": float(np.median(col("pad_clear", r2))) if r2 else float("nan"),
        "f_torso_peak": float(np.nanmax(col("f_torso", rb))) if rb else float("nan"),
    }
    # completion is the ladder finishing, not merely reaching release
    out["complete"] = int(len({x["phase"] for x in rows}) >= 9)
    if os.path.exists(qcsv):
        ph = {round(float(x["t"]), 3): x["phase"] for x in rows}
        pit, pit2 = [], []
        for x in csv.DictReader(open(qcsv)):
            try:
                q = [float(x["q%d" % i]) for i in range(7)]
                t = round(float(x["t"]), 3)
            except (TypeError, ValueError):
                continue        # a truncated final row on a killed run
            p = pitch_deg(q)
            pit.append(p)
            if ph.get(t) in ("1", "2"):
                pit2.append(p)
        if pit:
            out["pitch_peak"] = float(np.max(pit))
        if pit2:
            out["pitch_brace_med"] = float(np.median(pit2))
    return out


def collect():
    req = required_pitch()
    recs = []
    for arm, (d, _) in ARMS.items():
        for h in HEIGHTS:
            for s in range(3):
                r = read_one(d, h, s)
                if r is None:
                    continue
                r["arm"] = arm
                r["pitch_req"] = req.get(h, float("nan"))
                recs.append(r)
    return recs


def by(recs, arm, h):
    return [r for r in recs if r["arm"] == arm and abs(r["h"] - h) < 1e-6]


def table(recs):
    req = required_pitch()
    w = ["", "face  req_pitch"]
    print("\n face  req  | " + " | ".join("%-22s" % ARMS[a][1] for a in ARMS))
    for h in HEIGHTS:
        cells = []
        for a in ARMS:
            g = by(recs, a, h)
            if not g:
                cells.append("%-22s" % "--")
                continue
            c = sum(r["complete"] for r in g)
            f = np.nanmedian([r.get("f_forearm2", np.nan) for r in g])
            cells.append("%d/%d  %5.1f N  %+5.1f deg" %
                         (c, len(g), f,
                          np.nanmedian([r.get("pitch_brace_med", np.nan) for r in g])))
        print(" %.3f %4.1f | " % (h, req.get(h, float("nan"))) + " | ".join(cells))
    print("\n(cells: completions/seeds, median rung-2 forearm load, "
          "median base pitch over the lean rungs)")


def fig_load(recs, out, mode):
    c = style(mode)
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    xs = np.arange(len(HEIGHTS))
    wd = 0.8 / len(ARMS)
    for i, a in enumerate(ARMS):
        ys = [np.nanmedian([r.get("f_forearm2", np.nan) for r in by(recs, a, h)] or [np.nan])
              for h in HEIGHTS]
        ax.bar(xs + i * wd - 0.4 + wd / 2, ys, wd * 0.9, color=ARMCOL[a],
               label=ARMS[a][1])
    ax.axhline(39, color=c["mid"], lw=1, ls="--")
    ax.annotate("39 N -- the lightest load any completed run carried",
                (len(HEIGHTS) - 0.5, 41), ha="right", fontsize=8, color=c["mid"])
    ax.set_xticks(xs); ax.set_xticklabels(["%.3f" % h for h in HEIGHTS])
    ax.set_xlabel("slab face height (m)")
    ax.set_ylabel("median forearm pad load in rung 2 (N)")
    ax.set_title("Forearm pad load while the targeting rung runs", loc="left")
    ax.grid(axis="y", lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(out, dpi=170); plt.close(fig)


def fig_pitch(recs, out, mode):
    c = style(mode)
    req = required_pitch()
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    hs = list(HEIGHTS)
    ax.plot(hs, [req.get(h, np.nan) for h in hs], color=c["fg"], lw=1.6,
            marker="o", ms=4, label="pitch the seated brace pose needs")
    for a in ARMS:
        ys = [np.nanmedian([r.get("pitch_brace_med", np.nan)
                            for r in by(recs, a, h)] or [np.nan]) for h in hs]
        ax.plot(hs, ys, color=ARMCOL[a], lw=1.4, marker="s", ms=4,
                label=ARMS[a][1])
    ax.set_xlabel("slab face height (m)")
    ax.set_ylabel("base pitch (deg)")
    ax.set_title("Commanded pitch against delivered pitch on the lean rungs",
                 loc="left")
    ax.grid(lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=170); plt.close(fig)


def fig_outcome(recs, out, mode):
    c = style(mode)
    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    for j, a in enumerate(ARMS):
        for i, h in enumerate(HEIGHTS):
            g = by(recs, a, h)
            n = len(g)
            k = sum(r["complete"] for r in g)
            col = (STATUS["complete"] if n and k == n else
                   STATUS["stalled"] if k else STATUS["fell"]) if n else c["grid"]
            ax.add_patch(plt.Rectangle((i - 0.45, j - 0.42), 0.9, 0.84,
                                       color=col, alpha=0.85 if n else 0.3))
            ax.text(i, j, "%d/%d" % (k, n) if n else "--", ha="center",
                    va="center", color="#ffffff" if n else c["muted"],
                    fontsize=9, fontweight="bold")
    ax.set_xlim(-0.6, len(HEIGHTS) - 0.4); ax.set_ylim(-0.6, len(ARMS) - 0.4)
    ax.set_xticks(range(len(HEIGHTS)))
    ax.set_xticklabels(["%.3f m" % h for h in HEIGHTS])
    ax.set_yticks(range(len(ARMS)))
    ax.set_yticklabels([ARMS[a][1] for a in ARMS], fontsize=9)
    ax.invert_yaxis()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Completions, 3 seeds per cell", loc="left")
    fig.tight_layout(); fig.savefig(out, dpi=170); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "figs"))
    ap.add_argument("--json", default=os.path.join(HERE, "figs", "pitchtrack.json"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    recs = collect()
    if not recs:
        sys.exit("no runs found")
    table(recs)
    json.dump(recs, open(a.json, "w"), indent=1)
    for mode in ("light", "dark"):
        sfx = "" if mode == "light" else "_dark"
        fig_load(recs, os.path.join(a.out, "pt_load%s.png" % sfx), mode)
        fig_pitch(recs, os.path.join(a.out, "pt_pitch%s.png" % sfx), mode)
        fig_outcome(recs, os.path.join(a.out, "pt_outcome%s.png" % sfx), mode)
    print("\nwrote %s and 3 figure pairs in %s" % (a.json, a.out))


if __name__ == "__main__":
    main()
