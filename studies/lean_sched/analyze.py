#!/usr/bin/env python3
"""Score a lean schedule sweep and emit the figure + tables the docpage needs.

Reads `summary.csv` (one row per run, written incrementally by sweep.py) plus each
run's state CSV, and reports per variant:

  completed / fell / timed out      -- robustness, as counts (n is small; say so)
  t_complete                        -- the number the whole exercise is about
  per-phase dwell                   -- from the `enter=` times, so a cut schedule
                                       can be checked against what it asked for
  brace load + duty                 -- did it actually seat the arm, or just
                                       finish the clock faster with nothing on
                                       the table

usage: analyze.py --runs DIR --out DIR
"""
import argparse, csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HELD = 9999.0
PHASES = ["stand_up", "brace_lean", "release", "sb_r1", "sb_r2", "sb_r3",
          "sb_r4", "stand_end"]
BRACE_PHASE = 1
FORCE_ON = 5.0          # N: "the link is carrying something", not sensor noise


def read_summary(d):
    rows = []
    with open(os.path.join(d, "summary.csv")) as f:
        for r in csv.DictReader(f):
            if not r.get("variant"):
                continue
            r["seed"] = int(r["seed"])
            r["fell"] = r.get("fell") == "1"
            r["complete"] = r.get("complete") == "1"
            r["t_complete"] = float(r.get("t_complete") or -1)
            r["t_end"] = float(r.get("t_end") or 0)
            r["enter"] = [float(x) for x in (r.get("enter") or "").split(":") if x]
            rows.append(r)
    return rows


def brace_stats(csv_path):
    """Peak and duty of table load on the brace links during the brace phase."""
    t, f, ph = [], [], []
    try:
        with open(csv_path) as fh:
            for r in csv.DictReader(fh):
                try:
                    t.append(float(r["t"]))
                    f.append(float(r["f_elbow"]) + float(r["f_forearm"]))
                    ph.append(int(r["phase"]))
                except (TypeError, ValueError):
                    continue
    except FileNotFoundError:
        return (0.0, 0.0)
    if not t:
        return (0.0, 0.0)
    f = np.array(f); ph = np.array(ph)
    m = ph == BRACE_PHASE
    if not m.any():
        return (0.0, 0.0)
    return (float(f[m].max()), float((f[m] > FORCE_ON).mean()))


def phase_dwell(rs):
    """Median measured dwell per phase, over the runs that got through it.

    This is where a cut actually lands, and it is the only place the SLOP shows
    up: a phase whose measured dwell exceeds its `success_sustain_time` is not
    running on the timer alone.
    """
    done = [r for r in rs if r["complete"] and len(r["enter"]) >= 2]
    if not done:
        return []
    n = max(len(r["enter"]) for r in done)
    out = []
    for i in range(n - 1):
        d = [r["enter"][i + 1] - r["enter"][i] for r in done
             if len(r["enter"]) > i + 1
             and r["enter"][i] >= 0 and r["enter"][i + 1] >= 0]
        out.append(round(float(np.median(d)), 2) if d else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = read_summary(a.runs)
    variants = sorted({r["variant"] for r in rows})

    table = []
    for v in variants:
        rs = [r for r in rows if r["variant"] == v]
        done = [r for r in rs if r["complete"]]
        fell = [r for r in rs if r["fell"]]
        tc = [r["t_complete"] for r in done]
        peaks, duties = [], []
        for r in rs:
            p, d = brace_stats(r["csv"])
            peaks.append(p); duties.append(d)
        table.append(dict(
            variant=v, n=len(rs), completed=len(done), fell=len(fell),
            t_complete_med=float(np.median(tc)) if tc else float("nan"),
            t_complete_all=sorted(round(x, 1) for x in tc),
            brace_peak_med=float(np.median(peaks)) if peaks else 0.0,
            brace_duty_med=float(np.median(duties)) if duties else 0.0,
            reached_phase=[int(sum(1 for e in rr["enter"] if e >= 0)) for rr in rs],
            phase_dwell=phase_dwell(rs),
        ))

    with open(os.path.join(a.out, "analysis.json"), "w") as f:
        json.dump(dict(rows=len(rows), variants=table), f, indent=2)

    # ---- figure: what actually ended each run, and when -------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]
    for i, v in enumerate(variants):
        rs = sorted([r for r in rows if r["variant"] == v], key=lambda r: r["seed"])
        for j, r in enumerate(rs):
            y = i + (j - (len(rs) - 1) / 2) * 0.22
            end = r["t_complete"] if r["complete"] else r["t_end"]
            ax.barh(y, end, height=0.18,
                    color="#1baf7a" if r["complete"] else "#e34948")
            for e in r["enter"]:
                if e >= 0:
                    ax.plot([e], [y], "|", color="#0b0b0b", ms=7, mew=1.0)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([v.replace("h12_recovery_noreach_", "") for v in variants])
    ax.set_xlabel("sim seconds")
    ax.set_title("run duration per seed  (green = completed, red = fell)\n"
                 "ticks = phase entries", fontsize=9)
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1]
    for v in variants:
        rs = [r for r in rows if r["variant"] == v]
        for r in rs:
            t, f = [], []
            try:
                with open(r["csv"]) as fh:
                    for q in csv.DictReader(fh):
                        try:
                            t.append(float(q["t"]))
                            f.append(float(q["f_elbow"]) + float(q["f_forearm"]))
                        except (TypeError, ValueError):
                            continue
            except FileNotFoundError:
                continue
            ax.plot(t, f, lw=1.0,
                    color="#2a78d6" if "both50" in v else "#eb6834",
                    alpha=0.8,
                    label=v.replace("h12_recovery_noreach_", "")
                    if r["seed"] == 0 else None)
    ax.axhline(FORCE_ON, color="#7a7873", lw=0.8, ls="--")
    ax.set_xlabel("sim seconds"); ax.set_ylabel("elbow + forearm load on table [N]")
    ax.set_title("did the arm actually seat?", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "schedule.png"), dpi=140)

    hdr = ("variant", "n", "done", "fell", "t_complete median", "brace peak N",
           "brace duty")
    print("%-10s %3s %5s %5s %18s %13s %10s" % hdr)
    for r in table:
        print("%-10s %3d %5d %5d %18s %13.0f %9.0f%%" % (
            r["variant"].replace("h12_recovery_noreach_", ""), r["n"],
            r["completed"], r["fell"],
            ("%.1f" % r["t_complete_med"]) if r["completed"] else "-",
            r["brace_peak_med"], 100 * r["brace_duty_med"]))


if __name__ == "__main__":
    main()
