#!/usr/bin/env python3
"""Emit the HTML result tables for the page from runs/<run>/summary.json.

Prints two <table> blocks (sweep A, sweep B) to stdout; the page author pastes
them in. Numbers only; the sentences around them are written by hand.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
run = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "runs/sweep1")
S = json.load(open(os.path.join(run, "summary.json")))
OBJ_ORDER = ["light", "nominal", "heavyish", "slippery", "grippy", "heavy"]
OBJ_LABEL = {"light": "0.2 kg, μ 0.4", "nominal": "0.5 kg, μ 0.4", "heavyish": "1.0 kg, μ 0.4",
             "slippery": "0.5 kg, μ 0.2", "grippy": "0.5 kg, μ 0.8", "heavy": "2.0 kg, μ 0.4"}
FSLIDE = {"light": 0.78, "nominal": 1.96, "heavyish": 3.92, "slippery": 0.98, "grippy": 3.92, "heavy": 7.85}


def cell(st):
    if not st:
        return "<td>—</td>"
    t = f"{st['t_done_med']:.1f} s" if st["t_done_med"] is not None else "—"
    return (f"<td class=\"num\"><b>{st['success']}/{st['n']}</b><br><span class=\"small\">{st['final_err_med']*100:.1f} cm · {t} · "
            f"{st['peak_f_lp_med']:.1f} N</span></td>")


A, B = S["A"], S["B"]
print("<div class=\"tablewrap\"><table>")
print("<tr><th>Object (true m, μ)</th><th class=\"num\">μ m g</th>" + "".join(
    f"<th class=\"num\">m̂ = {r:g}×</th>" for r in A["ratios_m"]) + "".join(
    f"<th class=\"num\">μ̂ = {r:g}×</th>" for r in A["ratios_mu"]) + "</tr>")
for o in OBJ_ORDER:
    if not A["grid_m"].get(o):
        continue
    row = f"<tr><td>{OBJ_LABEL[o]}</td><td class=\"num\">{FSLIDE[o]:.1f} N</td>"
    row += "".join(cell(A["grid_m"][o].get(str(r)) or A["grid_m"][o].get(r)) for r in A["ratios_m"])
    row += "".join(cell(A["grid_mu"][o].get(str(r)) or A["grid_mu"][o].get(r)) for r in A["ratios_mu"])
    print(row + "</tr>")
print("</table></div>")
print()
print("<div class=\"tablewrap\"><table>")
print("<tr><th>Object (true m, μ)</th><th class=\"num\">μ m g</th>" + "".join(
    f"<th class=\"num\">F<sub>max</sub> = {r:g}× μmg</th>" for r in B["ratios"]) + "</tr>")
for o in OBJ_ORDER:
    if not B["grid"].get(o):
        continue
    row = f"<tr><td>{OBJ_LABEL[o]}</td><td class=\"num\">{FSLIDE[o]:.1f} N</td>"
    row += "".join(cell(B["grid"][o].get(str(r)) or B["grid"][o].get(r)) for r in B["ratios"])
    print(row + "</tr>")
print("</table></div>")
