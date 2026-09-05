#!/usr/bin/env python3
"""How far forward the base travels, against the line `Brace Reach Lead` draws.

lean.cc's `Brace Reach Lead` charges base_x past `brace_lead_x0` plus
`brace_lead_gain` times the measured brace load, and is exactly zero below that
line. Its weight is 0.0 in every strategy JSON in the repo, so it has never been
on. This plots what every run in the study did against that line, which is the
whole argument for turning it on.

base_x is data->qpos[0] -- the same frame the estimator feeds and the recording
logs -- and it is only in the `--qpos_out` dump, so a run without one is skipped.

usage: analyze_lead.py --runs A=runs/seeded B=runs/ab/on --out figs
"""
import argparse, csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze import style, STATUS, SEQ, CAT
import retarget as R
from analyze_pose import pristine_model


def collect(rundir, label):
    out = []
    p = os.path.join(rundir, "summary.csv")
    if not os.path.exists(p):
        return out
    for r in csv.DictReader(open(p)):
        q = (r.get("csv") or "").replace(".csv", ".qpos.csv")
        if not q or not os.path.exists(q):
            continue
        rows = list(csv.DictReader(open(q)))
        if len(rows) < 2:
            continue
        t = np.array([float(x["t"]) for x in rows])
        bx = np.array([float(x["q0"]) for x in rows])
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
        out.append(dict(
            label=label, h=float(r["table_h"]), seed=int(r["seed"]),
            outcome=("fell" if r["fell"] == "1" else
                     "complete" if r["complete"] == "1" else "stalled"),
            peak=float(bx.max()), t_peak=float(t[int(bx.argmax())]),
            over_s=float(dt * int((bx > X0).sum()))))
    return out


X0 = 0.24        # overwritten from the model in main()


def main():
    global X0
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="LABEL=path pairs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = pristine_model()
    X0 = R.num(m, "brace_lead_x0", 0.24)
    gain = R.num(m, "brace_lead_gain", 0.10)

    recs = []
    for spec in a.runs:
        label, path = spec.split("=", 1)
        recs += collect(path, label)
    if not recs:
        raise SystemExit("no qpos dumps found")

    for mode in ("light", "dark"):
        c = style(mode)
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        marks = {"seeded": "o", "on": "s", "off": "^", "lead-on": "D",
                 "lead-off": "v"}
        for r in recs:
            ax.plot([r["peak"]], [r["over_s"]], marks.get(r["label"], "o"),
                    ms=9, mfc=STATUS[r["outcome"]], mec=c["surface"], mew=1.2,
                    zorder=4)
            ax.annotate("%.3f" % r["h"], (r["peak"], r["over_s"]),
                        xytext=(0, 9), textcoords="offset points", ha="center",
                        fontsize=7.5, color=c["muted"])
        ax.axvline(X0, color=CAT["pelvis"], lw=1.6, ls="--")
        ax.annotate("brace_lead_x0 = %.2f m,\nopening +%.2f m per unit brace load"
                    % (X0, gain), (X0, 0.60 * ax.get_ylim()[1]),
                    xytext=(9, 0), textcoords="offset points", fontsize=8.5,
                    color=CAT["pelvis"], va="center")
        ax.set_xlabel("peak base x during the run (m)")
        ax.set_ylabel("seconds spent past the line")
        ax.set_title("Forward base travel against the line the unused cost draws",
                     loc="left", fontsize=11, color=c["fg"])
        ax.grid(alpha=0.6, lw=0.6)
        handles = [plt.Line2D([], [], marker="o", ls="", ms=8, mfc=v,
                              mec=c["surface"], label=k) for k, v in STATUS.items()]
        seen = []
        for r in recs:
            if r["label"] not in seen:
                seen.append(r["label"])
        handles += [plt.Line2D([], [], marker=marks.get(k, "o"), ls="", ms=8,
                               mfc=c["muted"], mec=c["surface"], label=k)
                    for k in seen]
        ax.legend(handles=handles, loc="upper right", fontsize=8.5, frameon=False,
                  ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "fig_lead" +
                                 (".png" if mode == "light" else ".dark.png")),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
    json.dump(recs, open(os.path.join(a.out, "lead.json"), "w"), indent=1)
    print("%d runs with a qpos dump; brace_lead_x0 = %.2f m" % (len(recs), X0))
    for r in sorted(recs, key=lambda z: (z["label"], z["h"], z["seed"])):
        print("  %-8s %.3f s%d  %-9s peak %.3f  %5.1f s past"
              % (r["label"], r["h"], r["seed"], r["outcome"], r["peak"],
                 r["over_s"]))


if __name__ == "__main__":
    main()
