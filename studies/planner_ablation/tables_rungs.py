#!/usr/bin/env python3
"""Per-rung durations and task channels per arm (completing runs), 33 plans/s."""
import json, os, sys, html, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
PH = [("brace_lean", "lean"), ("reach", "reach"), ("release", "release"), ("standback_r1", "back 1"),
      ("standback_r2", "back 2"), ("standback_r3", "back 3"), ("standback_r4", "back 4")]
ARMS = [("icem", "iCEM"), ("cem", "CEM, k = 6"), ("cem_ne2", "CEM, k = 2"), ("ps_raw01_zero", "PS, hold"),
        ("mppi_raw01_zero_l0.1", "MPPI, λ = 0.1"), ("mppi_raw01_zero_l1", "MPPI, λ = 1"), ("mppi_raw01_zero_l10", "MPPI, λ = 10")]


def med(v):
    v = [x for x in v if x is not None and x == x]
    return st.median(v) if v else float("nan")


def main():
    S = json.load(open(os.path.join(HERE, "runs/summary_deploy_spp15.json")))
    cap = sys.argv[1] if len(sys.argv) > 1 else ""
    out = ["<div class=scroll><table><caption>%s</caption>" % html.escape(cap),
           "<tr><th>arm</th><th>completed</th><th>t<sub>complete</sub>, s</th>" + "".join("<th>%s, s</th>" % l for _, l in PH) +
           "<th>brace peak, N</th><th>reach on the trunk, fraction</th><th>reach error, mm</th></tr>"]
    for arm, label in ARMS:
        rs = [r for r in S["runs"] if r["arm"] == arm]
        comp = [r for r in rs if r["outcome"] == "complete"]
        cells = ["%.1f" % med([r["rung_dur"].get(k) for r in comp]) for k, _ in PH]
        out.append("<tr><td>%s</td><td>%d/%d</td><td>%.1f</td>%s<td>%.0f</td><td>%.2f</td><td>%.0f</td></tr>" % (
            label, len(comp), len(rs), med([r["t_complete"] for r in comp]), "".join("<td>%s</td>" % c for c in cells),
            med([r.get("brace_peak_N") for r in rs]), med([r.get("seated_frac") for r in rs]), med([r.get("reach_min_mm") for r in rs])))
    out.append("</table></div>")
    print("\n".join(out))


if __name__ == "__main__":
    main()
