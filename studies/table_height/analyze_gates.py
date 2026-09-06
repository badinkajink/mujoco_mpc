#!/usr/bin/env python3
"""Figures for the two constants that bound the braced-lean height window.

Reads `figs/pitch.json` (probe_pitch.py), `figs/basin.json` (probe_basin.py) and
`figs/reachset.json` (probe_reachset.py), plus the run outcomes, and draws:

  fig_pitchgate   required brace pitch per slab against `standback_pitch_release`
  fig_basinaim    where the posture target's right arm points, three ways
  fig_window      the two conditions and the observed window on one axis

usage: analyze_gates.py --figs figs [--runs LABEL=dir ...]
"""
import argparse, csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze import style, STATUS, SEQ, CAT

TOL = 0.07


def outcomes(specs):
    """height -> complete count / total, over the shipped-baseline runs."""
    n, k = {}, {}
    for spec in specs:
        _, path = spec.split("=", 1)
        s = os.path.join(path, "summary.csv")
        if not os.path.exists(s):
            continue
        for r in csv.DictReader(open(s)):
            h = float(r["table_h"])
            n[h] = n.get(h, 0) + 1
            k[h] = k.get(h, 0) + (1 if r["complete"] == "1" else 0)
    return {h: (k[h], n[h]) for h in n}


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_pitchgate(P, out, mode):
    c = style(mode)
    hs = sorted(float(h) for h in P["heights"])
    gate, erect = np.degrees(P["gate"]), np.degrees(P["erect"])
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for tag, col, lab in (("brace", CAT["left forearm"], "brace seated on the slab"),
                          ("reach", CAT["right arm"], "brace + jaw tip on the rung-2 target")):
        ys = [np.degrees(P["heights"]["%.3f" % h][tag]["pitch"]) for h in hs]
        ax.plot(hs, ys, "o-", color=col, lw=2.0, ms=7, label=lab, zorder=4)
    ax.axhline(gate, color=STATUS["fell"], lw=1.6)
    ax.annotate("standback_pitch_release  %.1f deg" % gate, (hs[0], gate),
                xytext=(2, 5), textcoords="offset points", fontsize=9,
                color=STATUS["fell"])
    ax.axhline(erect, color=STATUS["stalled"], lw=1.2, ls="--")
    ax.annotate("brace_erect_target  %.1f deg" % erect, (hs[0], erect),
                xytext=(2, -13), textcoords="offset points", fontsize=9,
                color=STATUS["stalled"])
    ax.axhspan(gate, 60, color=STATUS["fell"], alpha=0.08, lw=0)
    ax.set_ylim(8, 52)
    ax.set_xlabel("slab face height (m)")
    ax.set_ylabel("base pitch the pose requires (deg)")
    ax.set_title("How far the robot must bow to reach each slab, against the "
                 "gate that lets it go", loc="left", fontsize=11, color=c["fg"])
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.grid(alpha=0.6, lw=0.6)
    save(fig, out, "fig_pitchgate" + (".png" if mode == "light" else ".dark.png"))


def fig_basinaim(B, out, mode):
    c = style(mode)
    hs = sorted(float(h) for h in B["heights"])
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    fixed = [B["heights"]["%.3f" % h]["fixed_mm"] for h in hs]
    solved = [max(B["heights"]["%.3f" % h]["solved_mm"], 0.3) for h in hs]
    ax.axhspan(0.2, 1000 * TOL, color=STATUS["complete"], alpha=0.12, lw=0)
    ax.axhline(1000 * TOL, color=STATUS["complete"], lw=1.4)
    ax.annotate("target_distance_tolerance  70 mm", (hs[0], 1000 * TOL),
                xytext=(2, 5), textcoords="offset points", fontsize=9,
                color=STATUS["complete"])
    ax.plot(hs, fixed, "s-", color=SEQ[4], lw=2.0, ms=7,
            label="reach_arm_q, the shipped basin-lock config", zorder=4)
    ax.plot(hs, solved, "o-", color=CAT["right arm"], lw=2.0, ms=7,
            label="right arm solved for this slab", zorder=4)
    ax.set_yscale("log")
    ax.set_xlabel("slab face height (m)")
    ax.set_ylabel("jaw tip to the rung-2 gate target (mm)")
    ax.set_title("Where the posture target's right arm points",
                 loc="left", fontsize=11, color=c["fg"])
    ax.legend(fontsize=9, frameon=False, loc="upper center")
    ax.grid(alpha=0.6, lw=0.6, which="both")
    save(fig, out, "fig_basinaim" + (".png" if mode == "light" else ".dark.png"))


def fig_window(P, B, obs, out, mode):
    """The pitch bound is a prediction; the run outcome is what happened."""
    c = style(mode)
    hs = sorted(float(h) for h in P["heights"])
    gate = np.degrees(P["gate"])
    fig, ax = plt.subplots(figsize=(8.6, 2.9))
    ok_pitch = [np.degrees(P["heights"]["%.3f" % h]["brace"]["pitch"]) <= gate
                for h in hs]
    ok_run = [obs.get(h, (0, 0))[0] > 0 for h in hs]
    rows = [("predicted: brace pitch clears the release gate", ok_pitch, None),
            ("measured: shipped runs that complete", ok_run,
             ["%d/%d" % obs.get(h, (0, 0)) for h in hs])]
    for j, (lab, ok, notes) in enumerate(rows):
        y = -1.15 * j
        for i, h in enumerate(hs):
            ax.add_patch(plt.Rectangle((i - 0.40, y - 0.34), 0.80, 0.68,
                                       color=STATUS["complete"] if ok[i]
                                       else STATUS["fell"], alpha=0.8, lw=0))
            if notes:
                ax.text(i, y, notes[i], ha="center", va="center", fontsize=9.5,
                        color="#ffffff")
        ax.text(-0.60, y, lab, ha="right", va="center", fontsize=10,
                color=c["fg"])
    for i, h in enumerate(hs):
        ax.text(i, 0.62, "%.3f" % h, ha="center", fontsize=10, color=c["fg"])
    ax.text(-0.60, 0.62, "slab face (m)", ha="right", fontsize=10,
            color=c["mid"])
    ax.set_xlim(-4.6, len(hs) - 0.4)
    ax.set_ylim(-1.75, 0.95)
    ax.axis("off")
    ax.set_title("The pitch bound accounts for the low end and not the high end",
                 loc="left", fontsize=11, color=c["fg"], x=-0.0)
    fig.text(0.5, -0.04,
             "1.085 m clears the pitch gate by 11.4 deg and still fails, which "
             "is the reach, not the bow.",
             ha="center", fontsize=9, color=c["mid"])
    save(fig, out, "fig_window" + (".png" if mode == "light" else ".dark.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    a = ap.parse_args()
    P = json.load(open(os.path.join(a.figs, "pitch.json")))
    B = json.load(open(os.path.join(a.figs, "basin.json")))
    obs = outcomes(a.runs)
    for mode in ("light", "dark"):
        fig_pitchgate(P, a.figs, mode)
        fig_basinaim(B, a.figs, mode)
        fig_window(P, B, obs, a.figs, mode)
    print("wrote fig_pitchgate, fig_basinaim, fig_window to", a.figs)
    hs = sorted(float(h) for h in P["heights"])
    print("\n face   brace pitch  release gate  margin   shipped complete")
    for h in hs:
        p = np.degrees(P["heights"]["%.3f" % h]["brace"]["pitch"])
        k, n = obs.get(h, (0, 0))
        print(" %.3f   %8.1f  %11.1f  %+7.1f   %d/%d"
              % (h, p, np.degrees(P["gate"]), np.degrees(P["gate"]) - p, k, n))


if __name__ == "__main__":
    main()
