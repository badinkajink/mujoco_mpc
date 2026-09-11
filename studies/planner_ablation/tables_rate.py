#!/usr/bin/env python3
"""Plan-rate table: completions per arm at each --spp, with the failure rung."""
import json, os, sys, html
HERE = os.path.dirname(os.path.abspath(__file__))
PHASES = ["stand", "lean", "reach", "release", "sb1", "sb2", "sb3", "sb4", "final"]
LEVELS = [("runs/summary.json", 3, "167 Hz"), ("runs/summary_spp6.json", 6, "83 Hz"),
          ("runs/summary_spp10.json", 10, "50 Hz"), ("runs/summary_spp15.json", 15, "33 Hz")]
ARMS = ["icem", "cem", "ps_raw01_cubic", "ps_raw01_zero", "cem_ne1_fixed01_nom", "mppi_raw01_zero_l1", "ps"]


def cell(S, arm):
    rs = sorted([r for r in S["runs"] if r["arm"] == arm], key=lambda r: r["seed"])
    if not rs:
        return "—"
    k = sum(r["outcome"] == "complete" for r in rs)
    fails = ["%s@%.0fs" % (PHASES[r["max_phase"]], r["t_end"]) for r in rs if r["outcome"] != "complete"]
    return "<b>%d/%d</b>%s" % (k, len(rs), (" <span class=muted>(" + ", ".join(fails) + ")</span>") if fails else "")


def main():
    sums = []
    for path, spp, hz in LEVELS:
        p = os.path.join(HERE, path)
        sums.append((json.load(open(p)) if os.path.exists(p) else {"runs": []}, spp, hz))
    out = ["<div class=scroll><table><caption>%s</caption>" % html.escape(sys.argv[1] if len(sys.argv) > 1 else "")]
    out.append("<tr><th>arm</th>" + "".join("<th>--spp %d (%s)</th>" % (spp, hz) for _, spp, hz in sums) + "</tr>")
    for arm in ARMS:
        out.append("<tr><td><code>%s</code></td>" % arm + "".join("<td>%s</td>" % cell(S, arm) for S, _, _ in sums) + "</tr>")
    out.append("</table></div>")
    print("\n".join(out))


if __name__ == "__main__":
    main()
