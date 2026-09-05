#!/usr/bin/env python3
"""Refresh the brace-posture section of STATUS.md, between HTML markers.

STATUS.md stays the single generated source of truth for numbers. The baseline
half is written by write_status.py; this writes the half about the minimal-change
work and rewrites it in place, so re-running after another sweep updates the
numbers instead of appending a second copy.
"""
import argparse, json, os, sys, datetime

BEGIN = "<!-- POSE:BEGIN -->"
END = "<!-- POSE:END -->"


def load(p, n):
    f = os.path.join(p, n)
    return json.load(open(f)) if p and os.path.exists(f) else None


def counts(agg):
    return {round(a["h"], 3): (a["complete"], a["n"]) for a in (agg or [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", required=True)
    ap.add_argument("--arms", nargs="*", default=[],
                    help="LABEL=figsdir, in the order to show")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    gate = load(a.figs, "gate.json")
    arms = []
    for spec in a.arms:
        label, d = spec.split("=", 1)
        agg = load(d, "agg.json")
        if agg:
            arms.append((label, counts(agg)))

    L = [BEGIN, "", "## The minimal-change work (2026-09-05)", ""]
    # the headline is computed, so a later arm that DOES widen the window
    # rewrites this sentence instead of contradicting the table under it.
    base = dict(arms).get("shipped", {})
    fail = sorted(h for h, (c, n) in base.items() if c == 0)
    gained = sorted({h for lbl, c in arms if lbl != "shipped"
                     for h in fail if h in c and c[h][0] > 0})
    if gained:
        L.append("**%s now completes.** Arms: %s."
                 % (", ".join("%.3f m" % h for h in gained),
                    ", ".join(lbl for lbl, _ in arms)))
    else:
        L.append("**The window did not move.** No arm below completes a single "
                 "seed at %s, and every arm still completes every seed at the "
                 "compiled height. Each is free where the controller already "
                 "works, and each changes the mechanism without changing the "
                 "outcome." % ", ".join("%.3f m" % h for h in fail))
    L.append("")
    if arms:
        hs = sorted({h for _, c in arms for h in c})
        L.append("| face | " + " | ".join(k for k, _ in arms) + " |")
        L.append("|---|" + "---|" * len(arms))
        for h in hs:
            row = []
            for _, c in arms:
                row.append("%d/%d" % c[h] if h in c else "&mdash;")
            L.append("| %.3f m | %s |" % (h, " | ".join(row)))
        L.append("")

    if gate:
        got = [r for r in gate if r["closest_mm"] is not None]
        inside = [r for r in got if r["closest_mm"] <= 70.0]
        outside = [r for r in got if r["closest_mm"] > 70.0]
        ok_in = sum(1 for r in inside if r["outcome"] == "complete")
        ok_out = sum(1 for r in outside if r["outcome"] == "complete")
        L += [
            "### What the sweep is actually blocked on",
            "",
            "Rung 2 carries `reach_target_table` [0.55, 0.04, 0.15], so",
            "`TransitionLocked` overwrites `total_distance` with the distance from",
            "the right gripper jaw tip to `(near_edge+0.55, ctr_y-0.04, face+0.15)`",
            "and `target_distance_tolerance` (70 mm) is the gate. Replayed from the",
            "qpos dumps over %d runs: **%d of %d runs that came within 70 mm"
            % (len(got), ok_in, len(inside)),
            "completed the ladder, and %d of %d that did not.** No other measured"
            % (ok_out, len(outside)),
            "quantity separates the outcomes.",
            "",
            "Which axis is short at the closest approach (median per height):",
            "",
            "| face | dx (mm) | dy (mm) | dz (mm) | closest (mm) |",
            "|---|---|---|---|---|",
        ]
        import statistics as st
        for h in sorted({r["h"] for r in got}):
            g = [r for r in got if r["h"] == h]
            L.append("| %.3f m | %+.0f | %+.0f | %+.0f | %.0f |"
                     % (h, st.median(r["dx_mm"] for r in g),
                        st.median(r["dy_mm"] for r in g),
                        st.median(r["dz_mm"] for r in g),
                        st.median(r["closest_mm"] for r in g)))
        L.append("")

    L += [
        "### Run it",
        "",
        "```bash",
        "S=studies/table_height",
        "$S/sweep_ab.py    --out $S/runs/ab      # pose_track off vs on, one binary",
        "$S/sweep_lead.py  --out $S/runs/lead    # Brace Reach Lead 400, +/- retarget",
        "$S/sweep_mode2.py --out $S/runs/mode    # pose_track 2, reaching arm too",
        "$S/sweep_tol.py   --out $S/runs/tol     # DIAGNOSTIC: open the gate to 0.20",
        "$S/publish_pose.sh                      # figures, videos, page, this section",
        "```",
        "",
        "Page: `docs/lean/20260905-brace_posture_retarget.html` (local only).",
        "",
        "_Generated %s by write_status_pose.py._" % datetime.date.today().isoformat(),
        "",
        END,
    ]
    block = "\n".join(L)

    s = open(a.out).read() if os.path.exists(a.out) else ""
    if BEGIN in s and END in s:
        s = s[:s.index(BEGIN)] + block + s[s.index(END) + len(END):]
    else:
        s = s.rstrip() + "\n\n" + block + "\n"
    open(a.out, "w").write(s)
    print("refreshed the pose section of", a.out)


if __name__ == "__main__":
    main()
