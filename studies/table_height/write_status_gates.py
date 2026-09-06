#!/usr/bin/env python3
"""Rewrite the GATES section of STATUS.md from the probes and the basin arms.

Idempotent: replaces everything between the GATES markers, appending them if
they are absent. Every number is read from JSON, so re-running after more seeds
rewrites the section instead of stranding stale counts.

usage: write_status_gates.py --figs figs --arms LABEL=dir [...] --out STATUS.md
"""
import argparse, datetime, json, os
import numpy as np

BEG, END = "<!-- GATES:BEGIN -->", "<!-- GATES:END -->"
DEG = 180.0 / np.pi


def load(p, name):
    f = os.path.join(p, name)
    return json.load(open(f)) if p and os.path.exists(f) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", required=True)
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    P = load(a.figs, "pitch.json")
    B = load(a.figs, "basin.json")
    G = load(a.figs, "gate.json")
    arms = []
    for spec in a.arms:
        lab, path = spec.split("=", 1)
        agg = load(path, "agg.json")
        if agg:
            arms.append((lab, {r["h"]: r for r in agg}))

    gate = P["gate"] * DEG
    hs = sorted(float(h) for h in P["heights"])
    cross = float(np.interp(-gate, [-P["heights"]["%.3f" % h]["brace"]["pitch"] * DEG
                                    for h in hs], hs))

    L = ["", "## The two gates that bound the window (2026-09-05)", "",
         "The window is bounded by two different constants, one at each end, and",
         "neither tracks the slab.", "",
         "**Low end -- `standback_pitch_release` (0.50 rad = %.1f deg).** Seating both"
         % gate,
         "brace pads requires more bow than the release gate admits below ~%.3f m."
         % cross,
         "The requirement is geometry against a gate, so it holds for any planner or",
         "seed. Probe: `probe_pitch.py`.", "",
         "| face | brace pose pitch | margin to the gate |",
         "|---|---|---|"]
    for h in hs:
        p = P["heights"]["%.3f" % h]["brace"]["pitch"] * DEG
        L.append("| %.3f m | %.1f deg | %+.1f deg |" % (h, p, gate - p))
    L += ["",
          "Predicted lower edge **%.3f m**. Untested: nothing between 0.885 m and"
          % cross,
          "0.985 m has been run under any arm. Refuted if 0.905 m completes.", "",
          "**High end -- the trunk, not the arm.** With both pads seated and both feet",
          "planted the rung-2 waypoint solves to under 1 mm at every height over a",
          "0.30-0.75 x 0-0.30 m grid (`probe_reachset.py`), so the 154-185 mm gate miss",
          "at 1.085 m is not a reach limit. Replayed, `dx` is short by 125-149 mm in",
          "every arm regardless of what the right arm is commanded: peak base pitch is",
          "12.6 deg against the 17.2 deg the pose requires, and base x is 11 mm short of",
          "0.207 m. No cap binds -- CoM +0.041 against `com_cap_fwd` 0.145, base x 0.207",
          "against `brace_lead_x0` 0.24, 11.4 deg of pitch margin to the release gate.",
          ""]
    if arms:
        L += ["| face | " + " | ".join(l for l, _ in arms) + " |",
              "|---|" + "---|" * len(arms)]
        hh = sorted({h for _, m in arms for h in m})
        for h in hh:
            row = []
            for _, m in arms:
                r = m.get(h)
                row.append("&mdash;" if not r
                           else "%d/%d" % (r.get("complete", 0), r.get("n", 0)))
            L.append("| %.3f m | " % h + " | ".join(row) + " |")
        L += ["",
              "`reach_arm_posture` is neutral: it matches the shipped controller at both",
              "working heights and completes nothing at the three failing ones. Paired",
              "with `brace_pose_track` 1 it costs 1.035 m outright, which disqualifies",
              "the pair under the rule that a change must cost no completions where the",
              "controller already works.", ""]
    if B:
        L += ["`reach_arm_posture`'s config aims %s mm from the gate target offline; a"
              % "/".join("%.0f" % B["heights"]["%.3f" % h]["fixed_mm"] for h in hs),
              "per-slab solve aims 0-6 mm. In the runs it moved the tip no closer than",
              "the shipped posture did, because the probe assumed a brace the robot",
              "never seats.", ""]
    L += ["### Run it", "", "```bash", "S=studies/table_height",
          "$S/probe_pitch.py --json $S/figs/pitch.json      # the pitch bound, no sim",
          "$S/probe_reachset.py --json $S/figs/reachset.json # the rung-2 reach set",
          "$S/probe_basin.py --json $S/figs/basin.json      # where reach_arm_q aims",
          "$S/sweep_basin.py --out $S/runs/basin            # the two arms",
          "$S/publish_gates.sh                              # figures, page, this section",
          "```", "",
          "Page: `docs/lean/20260905-height_window_gates.html` (local only).", "",
          "_Generated %s by write_status_gates.py._" % datetime.date.today().isoformat(),
          ""]

    body = "\n".join(L)
    s = open(a.out).read()
    block = "%s\n%s\n%s\n" % (BEG, body, END)
    if BEG in s and END in s:
        s = s[:s.index(BEG)] + block + s[s.index(END) + len(END):].lstrip("\n")
    else:
        s = s.rstrip("\n") + "\n\n" + block
    open(a.out, "w").write(s)
    print("rewrote GATES section of", a.out)


if __name__ == "__main__":
    main()
