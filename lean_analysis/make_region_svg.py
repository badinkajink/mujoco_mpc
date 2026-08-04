#!/usr/bin/env python3
"""Render the static-equilibrium CoM regions from regions.json as inline SVG.

Inline rather than a PNG so the docpage stays a single self-contained file and
the plot restyles itself in dark mode with the rest of the page (strokes use
currentColor-adjacent CSS variables, not baked-in hex).

Top view, world x to the RIGHT (away from the robot, toward the table) and world
y UP the page. Drawn per contact set: the actuation-limited region, the pose's
CoM, the contact points, and -- for reference in every panel -- the legs-only
region, so the growth is visible rather than asserted.
"""
import json
import sys

import numpy as np

W, H = 300, 250
PAD = 26
ORDER = ["legs", "palm", "elbow+forearm", "elbow+forearm+hip"]
LABEL = {"legs": "legs only", "palm": "palm",
         "elbow+forearm": "elbow + forearm", "elbow+forearm+hip": "elbow + forearm + hip"}
COLOR = {"legs": "var(--ink3)", "palm": "var(--s3)",
         "elbow+forearm": "var(--s1)", "elbow+forearm+hip": "var(--s5)"}


def main():
    R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "regions.json"))

    allpts = []
    for k in ORDER:
        allpts += R[k]["actuated"] + [R[k]["com"]] + R[k]["pts"]
    A = np.array(allpts)
    x0, x1 = A[:, 0].min(), A[:, 0].max()
    y0, y1 = A[:, 1].min(), A[:, 1].max()
    # square the aspect so the region's shape is not distorted
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2 * 1.12
    x0, x1, y0, y1 = cx - half, cx + half, cy - half, cy + half
    sx = lambda x: PAD + (x - x0) / (x1 - x0) * (W - 2 * PAD)
    sy = lambda y: H - PAD - (y - y0) / (y1 - y0) * (H - 2 * PAD)

    def poly(pts, **kw):
        if not pts:
            return ""
        s = " ".join("%.1f,%.1f" % (sx(p[0]), sy(p[1])) for p in pts)
        att = " ".join('%s="%s"' % (k.replace("_", "-"), v) for k, v in kw.items())
        return '<polygon points="%s" %s/>' % (s, att)

    out = ['<div class="grid2">']
    for k in ORDER:
        r = R[k]
        g = [f'<svg viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="equilibrium region, {LABEL[k]}">']
        # scale bar: 10 cm
        bx, by = PAD, H - 10
        blen = 0.10 / (x1 - x0) * (W - 2 * PAD)
        g.append(f'<line x1="{bx}" y1="{by}" x2="{bx+blen:.1f}" y2="{by}" '
                 f'stroke="var(--ink3)" stroke-width="1.5"/>')
        g.append(f'<text x="{bx+blen+5:.1f}" y="{by+3.5}" font-size="9" '
                 f'fill="var(--ink3)">10 cm</text>')
        # legs-only reference in every panel
        if k != "legs":
            g.append(poly(R["legs"]["actuated"], fill="none", stroke="var(--ink3)",
                          stroke_width="1", stroke_dasharray="3 3", opacity="0.75"))
        g.append(poly(r["actuated"], fill=COLOR[k], fill_opacity="0.16",
                      stroke=COLOR[k], stroke_width="1.8"))
        # contact points
        for p in r["pts"]:
            g.append('<circle cx="%.1f" cy="%.1f" r="2" fill="var(--ink3)" '
                     'opacity="0.65"/>' % (sx(p[0]), sy(p[1])))
        # CoM
        g.append('<circle cx="%.1f" cy="%.1f" r="4.5" fill="none" stroke="var(--ink)" '
                 'stroke-width="1.8"/>' % (sx(r["com"][0]), sy(r["com"][1])))
        g.append('<circle cx="%.1f" cy="%.1f" r="1.6" fill="var(--ink)"/>'
                 % (sx(r["com"][0]), sy(r["com"][1])))
        g.append('</svg>')
        out.append(
            '<figure><div class="plot">%s</div>'
            '<figcaption><b style="color:var(--ink)">%s</b> — margin '
            '<b style="color:var(--ink)">%.0f mm</b> actuated, %.0f mm contact-only. '
            'Dashed = legs-only region for reference.</figcaption></figure>'
            % ("".join(g), LABEL[k], 1000 * r["margin_a"], 1000 * r["margin_c"]))
    out.append('</div>')
    open("region_svg.html", "w").write("\n".join(out))
    print("wrote region_svg.html")


if __name__ == "__main__":
    main()
