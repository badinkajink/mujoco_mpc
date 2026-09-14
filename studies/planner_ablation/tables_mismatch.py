#!/usr/bin/env python3
"""Completions per cell of the plant-side mismatch ladders (12 seeds), 33 plans/s,
plus the sigma-link cells, for the page (Table 6)."""
import json, os, sys, html
HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = [("cem", "CEM"), ("icem", "iCEM"), ("ps_raw01_zero", "PS, argmin"), ("mppi_raw01_zero_l1", "MPPI, softmax")]
CELLS = [("baseline", "runs/summary_gains_spp15.json"),
         ("mass × 1.05", "runs/summary_mismatch_spp15_m1.05.json"), ("mass × 1.10", "runs/summary_mismatch_spp15_m1.10.json"),
         ("mass × 1.15", "runs/summary_mismatch_spp15_m1.15.json"), ("mass × 1.20", "runs/summary_mismatch_spp15_m1.20.json"),
         ("kp × 1.5", "runs/summary_mismatch_spp15_kp1.5.json"), ("kp × 2", "runs/summary_mismatch_spp15_kp2.0.json"),
         ("kp × 3", "runs/summary_mismatch_spp15_kp3.0.json"),
         ("friction × 0.6", "runs/summary_mismatch_spp15_mu0.6.json"), ("friction × 0.4", "runs/summary_mismatch_spp15_mu0.4.json")]
LINK = [("cem_stdmin02", "CEM, σ 0.02"), ("cem_stdmin03", "CEM, σ 0.03"), ("icem_stdmin03", "iCEM, σ 0.03"),
        ("ps_raw02_zero", "PS, σ 0.02"), ("mppi_raw02_zero_l1", "MPPI, σ 0.02")]


def counts(S, arm):
    rs = [r for r in S["runs"] if r["arm"] == arm]
    return sum(r["outcome"] == "complete" for r in rs), len(rs)


def stance_falls(S, arm):
    return sum(1 for r in S["runs"] if r["arm"] == arm and r["outcome"] == "fell" and r["max_phase"] == 0)


def main():
    cap = sys.argv[1] if len(sys.argv) > 1 else ""
    sums = {name: json.load(open(os.path.join(HERE, p))) for name, p in CELLS}
    out = ["<div class=scroll><table><caption>%s</caption>" % html.escape(cap),
           "<tr><th>plant</th>" + "".join("<th>%s</th>" % l for _, l in ARMS) + "<th>stance falls (CEM / iCEM)</th></tr>"]
    for name, _ in CELLS:
        S = sums[name]
        cells = []
        for arm, _ in ARMS:
            k, n = counts(S, arm)
            cells.append("<td>%d/%d</td>" % (k, n))
        out.append("<tr><td style='white-space:nowrap'>%s</td>%s<td>%d / %d</td></tr>" % (
            name, "".join(cells), stance_falls(S, "cem"), stance_falls(S, "icem")))
    out.append("<tr><th colspan=6>σ link (baseline → mass × 1.10)</th></tr>")
    b, m = sums["baseline"], sums["mass × 1.10"]
    row = []
    for arm, label in LINK:
        kb, nb = counts(b, arm); km, nm = counts(m, arm)
        row.append("<td style='white-space:nowrap'>%s: %d/%d → %d/%d</td>" % (label, kb, nb, km, nm))
    out.append("<tr>" + "".join(row) + "<td></td></tr>" if len(row) == 5 else "<tr>" + "".join(row) + "</tr>")
    out.append("</table></div>")
    print("\n".join(out))


if __name__ == "__main__":
    main()
