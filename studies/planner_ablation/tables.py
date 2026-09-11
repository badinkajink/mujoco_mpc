#!/usr/bin/env python3
"""Emit HTML tables for the planner-ablation page from analyze.py's summary.json.

    tables.py --summary S.json --arms icem,cem,ps,mppi > table.html

Columns: arm, what differs from the shipped iCEM, n, completed, fell, stalled,
furthest rung per run, median t_complete, median stand jitter (rad per 20 ms),
median stand cost, median peak brace normal, median min reach error.
"""
import argparse, html, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arms import ARMS  # noqa: E402

FAMILY = {0: "PS", 5: "CEM", 7: "iCEM", 9: "MPPI"}
PHASES = ["stand", "lean", "reach", "release", "sb1", "sb2", "sb3", "sb4", "final"]


def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return ("%%.%df" % nd) % x


def describe(arm):
    pl, nums = ARMS[arm]
    parts = [FAMILY[pl]]
    for k, v in nums.items():
        parts.append("%s=%g" % (k, v))
    return ", ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--caption", default="")
    a = ap.parse_args()
    S = json.load(open(a.summary))
    agg = S["agg"]
    runs = S["runs"]
    out = ["<div class=scroll><table>"]
    if a.caption:
        out.append("<caption>%s</caption>" % html.escape(a.caption))
    out.append("<tr><th>arm</th><th>planner and overrides</th><th>n</th><th>completed</th>"
               "<th>fell</th><th>collapsed</th><th>stalled</th><th>furthest rung, per seed</th>"
               "<th>t<sub>complete</sub> median, s</th><th>stand jitter, mrad / 20 ms</th>"
               "<th>stand cost</th><th>peak brace, N</th><th>min reach err, mm</th></tr>")
    for arm in a.arms.split(","):
        if arm not in agg:
            continue
        g = agg[arm]
        rs = sorted([r for r in runs if r["arm"] == arm], key=lambda r: int(r["seed"]))
        rungs = " / ".join("%s<sup>%s</sup>" % (PHASES[r["max_phase"]], {"complete": "✓", "fell": "✗", "collapsed": "✗", "stalled": "·"}[r["outcome"]]) for r in rs)
        cls = {"iCEM": "icem", "CEM": "cem", "PS": "ps", "MPPI": "mppi"}[FAMILY[g["planner"]]]
        out.append("<tr class=%s><td><code>%s</code></td><td>%s</td><td>%d</td><td><b>%d</b></td><td>%d</td><td>%d</td><td>%d</td>"
                   "<td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                   % (cls, arm, html.escape(describe(arm)), g["n"], g["complete"], g["fell"], g["collapsed"], g["stalled"], rungs,
                      fmt(g["t_complete_median"]), fmt(1000 * g["jitter_stand_median"], 1),
                      fmt(g["cost_stand_median"], 1), fmt(g["brace_peak_median"], 0),
                      fmt(g["reach_min_mm_median"], 0)))
    out.append("</table></div>")
    print("\n".join(out))


if __name__ == "__main__":
    main()
