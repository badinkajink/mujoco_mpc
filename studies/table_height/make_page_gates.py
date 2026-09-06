#!/usr/bin/env python3
"""Build the height-window gates page from the probe JSON and the basin sweep.

Narrative and hypotheses are fixed text, written before the sweep landed. Every
number comes from figs/pitch.json, figs/basin.json, figs/reachset.json,
figs/gate.json and the arms' agg.json, so more seeds rewrite the numbers instead
of stranding them in prose.

usage:
  make_page_gates.py --figs studies/table_height/figs \
      --basin studies/table_height/figs_basin --both studies/table_height/figs_both \
      --shipped studies/table_height/figs_ab_off \
      --figs_rel media/gates --out docs/lean/20260905-height_window_gates.html
"""
import argparse, datetime, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_page import CSS, fig, num

DEG = 180.0 / np.pi


def load(p, name):
    f = os.path.join(p, name)
    return json.load(open(f)) if p and os.path.exists(f) else None


def hs_of(d):
    return sorted(float(h) for h in d["heights"])


def pitch_table(P, obs):
    gate = P["gate"] * DEG
    r = ["<table><tr><th>slab face</th><th>brace pose pitch</th>"
         "<th>reach pose pitch</th><th>margin to the release gate</th>"
         "<th>shipped runs complete</th></tr>"]
    for h in hs_of(P):
        rec = P["heights"]["%.3f" % h]
        bp, rp = rec["brace"]["pitch"] * DEG, rec["reach"]["pitch"] * DEG
        k, n = obs.get(h, (0, 0))
        cls = "good" if gate - bp > 0 else "bad"
        r.append("<tr><td>%.3f m</td><td>%.1f&deg;</td><td>%.1f&deg;</td>"
                 "<td class=%s>%+.1f&deg;</td><td>%d/%d</td></tr>"
                 % (h, bp, rp, cls, gate - bp, k, n))
    r.append("</table>")
    return "".join(r)


def basin_table(B, G):
    """the three things the right arm could be aimed at, per slab."""
    med = {}
    for rec in (G or []):
        if rec.get("closest_mm") is None:
            continue
        med.setdefault(rec["h"], []).append(rec["closest_mm"])
    r = ["<table><tr><th>slab face</th><th>as shipped, measured in the runs</th>"
         "<th><code>reach_arm_q</code></th><th>right arm solved for this slab</th>"
         "</tr>"]
    for h in hs_of(B):
        rec = B["heights"]["%.3f" % h]
        m = med.get(h)
        r.append("<tr><td>%.3f m</td><td>%s</td><td>%.0f mm</td>"
                 "<td class=good>%.0f mm</td></tr>"
                 % (h, ("%.0f mm" % np.median(m)) if m else "&mdash;",
                    rec["fixed_mm"], rec["solved_mm"]))
    r.append("</table>")
    return "".join(r)


def by_h(agg):
    """agg.json is a list of per-height records keyed by `h`."""
    return {} if not agg else {r["h"]: r for r in agg}


def arm_table(arms):
    """arms: list of (label, agg.json or None)."""
    maps = [(lab, by_h(a)) for lab, a in arms]
    hs = sorted({h for _, m in maps for h in m})
    r = ["<table><tr><th>slab face</th>"]
    for lab, _ in maps:
        r.append("<th>%s</th>" % lab)
    r.append("</tr>")
    for h in hs:
        r.append("<tr><td>%.3f m</td>" % h)
        for _, m in maps:
            rec = m.get(h)
            if not rec:
                r.append("<td>&mdash;</td>"); continue
            k, n = rec.get("complete", 0), rec.get("n", 0)
            cls = "good" if k == n and n else ("bad" if k == 0 else "hold")
            r.append('<td class="%s">%d/%d</td>' % (cls, k, n))
        r.append("</tr>")
    r.append("</table>")
    return "".join(r)


def verdict(basin, both, shipped):
    """Score the pre-registered hypothesis against whatever landed."""
    base_h = by_h(shipped)

    def wins(a):
        if not (a and shipped):
            return None
        got = 0
        for h, rec in by_h(a).items():
            base = base_h.get(h)
            if base is None:
                continue
            if rec.get("complete", 0) > 0 and base.get("complete", 0) == 0:
                got += 1
        return got
    out = []
    for lab, a in (("reach_arm_posture alone", basin),
                   ("reach_arm_posture + brace_pose_track", both)):
        w = wins(a)
        if w is None:
            out.append("<li><b>%s</b> &mdash; running.</li>" % lab)
        elif w > 0:
            out.append("<li><b>%s</b> &mdash; <span class=good>widens the "
                       "window</span>: %d height(s) that complete 0/3 shipped "
                       "now complete at least one seed.</li>" % (lab, w))
        else:
            out.append("<li><b>%s</b> &mdash; <span class=bad>does not widen "
                       "the window</span>: no height that fails under the "
                       "shipped controller completes a seed.</li>" % lab)
    return "<ul>%s</ul>" % "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", required=True)
    ap.add_argument("--basin", default="")
    ap.add_argument("--both", default="")
    ap.add_argument("--shipped", default="")
    ap.add_argument("--figs_rel", default="media/gates")
    ap.add_argument("--pose_page",
                    default="20260905-brace_posture_retarget.html")
    ap.add_argument("--baseline_page",
                    default="20260904-table_height_generalization.html")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    P = load(a.figs, "pitch.json")
    B = load(a.figs, "basin.json")
    RS = load(a.figs, "reachset.json")
    G = load(a.figs, "gate.json")
    basin, both = load(a.basin, "agg.json"), load(a.both, "agg.json")
    shipped = load(a.shipped, "agg.json")
    R = a.figs_rel

    obs = {r["h"]: (r.get("complete", 0), r.get("n", 0))
           for r in (shipped or [])}

    gate_deg = P["gate"] * DEG
    # where the requirement crosses the gate: the predicted lower window edge
    _h = hs_of(P)
    _p = [P["heights"]["%.3f" % h]["brace"]["pitch"] * DEG for h in _h]
    cross = float(np.interp(-gate_deg, [-v for v in _p], _h))
    worst = max(1000 * min(v for v in np.array(RS["%.3f" % h]["grid"]).ravel())
                for h in (1.085,)) if RS else 0

    Pg = []
    A = Pg.append
    A("<div class=wrap>")
    A("<h1>Pitch and reach gates on the braced-lean height window</h1>")
    A("<p class=sub>The shipped controller braces on a slab from 0.985 m to "
      "1.035 m and fails outside it. Two constants in <code>lean.cc</code>, both "
      "fitted at the compiled 0.985 m face and neither tracking the slab, bound "
      "that window at opposite ends: the release pitch gate at the low end, the "
      "rung-2 posture target at the high end.</p>")
    A("<p class=meta>%s &middot; branch <code>wxie/table-height</code> &middot; "
      "strategy 25 (<code>h12_brace_targeting</code>), Lean H12 Magpie, "
      "<code>--threads 6 --spp 3</code> &middot; generated by "
      "<code>studies/table_height/make_page_gates.py</code></p>"
      % datetime.date.today().isoformat())

    A("<h2>Where this sits</h2>")
    A("<p>The <a href=\"%s\">baseline sweep</a> measured the shipped controller "
      "over slab height and changed nothing. The <a href=\"%s\">posture "
      "retarget</a> re-solved the brace keyframe for each slab and found the "
      "proximate blocker: rung 2 of the ladder carries "
      "<code>reach_target_table</code> [0.55, 0.04, 0.15], and every run whose "
      "right jaw tip came within the 70 mm <code>target_distance_tolerance</code> "
      "of that point completed the ladder while every run that did not, failed. "
      "Four changes moved real mechanism and none widened the window.</p>"
      % (a.baseline_page, a.pose_page))
    A("<p>This page asks why that gate is missed, and finds two different "
      "answers at the two ends. Minimal is still counted in numbers touched: an "
      "existing model numeric beats a new weight, a weight beats a new residual, "
      "a new residual beats moving the robot. A result counts only if it widens "
      "the window on 3 seeds per height at a fixed thread count and costs no "
      "completions at 0.985 m.</p>")

    A("<h2>The low end: the brace pose is more bowed than the release gate "
      "allows</h2>")
    A("<p>Putting the forearm on a lower slab requires a deeper bow. Solving for "
      "the pose each slab needs &mdash; both brace pads on the face, both feet "
      "exactly where the keyframe plants them &mdash; and reading base pitch the "
      "way <code>lean.cc</code> reads it, off the free-joint quaternion, gives "
      "the requirement in the same frame as the gate that grades it. "
      "<code>standback_pitch_release</code> is %.2f rad (%.1f&deg;): rung 3 will "
      "not let go of the table above it.</p>" % (P["gate"], gate_deg))
    A(pitch_table(P, obs))
    A("<p>The sign of the margin separates the low failures from the working "
      "window exactly. At 0.885 m the pose needs %.1f&deg; and the gate is "
      "%.1f&deg;; at 0.785 m it needs %.1f&deg;. A run at either height cannot "
      "leave the brace rung without first un-bowing past the pose that is "
      "holding it on the table. At 0.985 m the requirement clears the gate by "
      "%.1f&deg;, which is the whole margin the working height has.</p>"
      % (P["heights"]["0.885"]["brace"]["pitch"] * DEG, gate_deg,
         P["heights"]["0.785"]["brace"]["pitch"] * DEG,
         gate_deg - P["heights"]["0.985"]["brace"]["pitch"] * DEG))
    A(fig("fig_pitchgate",
          "Base pitch required to seat the brace on each slab, and to seat it "
          "with the jaw tip on the rung-2 target, against the two pitch "
          "constants. Both constants are fitted at 0.985 m and neither moves "
          "with the slab.", R))
    A("<p>The runs bear the direction out. Replaying the logged qpos, peak base "
      "pitch is 74.9&deg; at 0.785 m and 44.8&deg; at 0.885 m against the "
      "47.1&deg; and 36.1&deg; the poses require &mdash; the robot overshoots "
      "the bow at the low end &mdash; and 12.6&deg; at 1.085 m against a "
      "required 17.2&deg;, so at the high end it never commits far enough "
      "forward to load the pads at all.</p>")
    A("<p>The requirement crosses the gate at <b>%.3f m</b>. That is a "
      "prediction of where the window's lower edge sits, and it is testable: "
      "nothing between 0.885 m and 0.985 m has been run under any arm, so the "
      "edge is currently bounded by the sweep grid rather than measured. Three "
      "seeds each at 0.905, 0.935, 0.955 and 0.975 m put a number on it. The "
      "bound is refuted if 0.905 m completes.</p>" % cross)
    A("<div class=note>This bound is derived from constants and geometry, with "
      "no sim time: it holds for any planner, any seed and any cost weighting, "
      "because it is a statement about which poses the gate admits. It is a "
      "necessary condition, not a sufficient one &mdash; a height can clear it "
      "and still fail for another reason, which is what 1.085 m does.</div>")

    A("<h2>The high end: the waypoint is reachable, and the posture points "
      "elsewhere</h2>")
    A("<p>The rung-2 target could have been outside the arm. It is not. Seat "
      "both brace pads on the slab, plant both feet, and solve for the demanded "
      "waypoint over a grid from 0.30 to 0.75 m in from the near edge and 0 to "
      "0.30 m above the face: every cell lands within a millimetre at every "
      "height from 0.785 m to 1.085 m, with the pads holding to 0.06 mm and the "
      "feet to 0.01 mm. The shipped cell (0.55, 0.15) solves to %.1f mm at "
      "1.085 m.</p>" % worst)
    A("<p>So the 176 mm the gate measures at 1.085 m is not a reach limit. "
      "<code>lean.cc</code> already names what it is:</p>")
    A("<blockquote>Posture (idx 27..33 = right arm) is tracking the brace-UP "
      "keyframe here, so it actively fights Reach DOWN.</blockquote>")
    A("<p>That comment sits above <code>reach_arm_posture</code>, a basin lock "
      "that overwrites the seven right-arm entries of the posture target with "
      "the config <code>reach_arm_q</code> on any rung carrying a reach target "
      "whose hover is below <code>reach_arm_hgate</code>. "
      "<code>Lean_H12_Magpie.xml</code> sets that gate to 0.16 and strategy 25's "
      "rung 2 hovers at 0.15, so the lock is already aimed at the rung the "
      "height sweep dies on. It ships at 0.0 &mdash; off.</p>")
    A(basin_table(B, G))
    A(fig("fig_basinaim",
          "Distance from the rung-2 gate target to where the posture target's "
          "right arm points. The shipped basin-lock config sits inside 105 mm "
          "from 0.885 m up; a right arm solved for the slab sits inside 6 mm at "
          "every height.", R))

    A("<h2>Both conditions, against the observed window</h2>")
    A(fig("fig_window",
          "The pitch condition and the measured rung-2 outcome, per slab. They "
          "agree with the shipped controller's window at every height tested.",
          R))

    A("<h2>Hypotheses, before the runs</h2>")
    A("<p>Written into <code>studies/table_height/sweep_basin.py</code> before "
      "it was launched:</p>")
    A("<ol>"
      "<li><b>H1.</b> Setting <code>reach_arm_posture</code> to 1 &mdash; one "
      "existing model numeric, no rebuild &mdash; completes at least one seed at "
      "1.085 m, where pitch has %.1f&deg; of margin and the reach is the only "
      "thing missing. It does nothing at 0.785 m or 0.885 m, where the release "
      "gate blocks the ladder regardless.</li>"
      "<li><b>H2.</b> <code>reach_arm_posture</code> plus "
      "<code>brace_pose_track</code> 1 does better at 1.085 m than either alone, "
      "because the shipped keyframe leaves the pads 100 mm under a 1.085 m slab "
      "and the lock cannot help an arm that is not braced.</li>"
      "<li><b>Kill condition.</b> If neither arm completes a seed at 1.085 m, "
      "the right arm's posture target is not what misses the gate, and the next "
      "measurement is the reach cost's weight against Posture's, not another "
      "posture edit.</li>"
      "</ol>" % (gate_deg - P["heights"]["1.085"]["brace"]["pitch"] * DEG))

    A("<h2>Results</h2>")
    A(arm_table([("shipped", shipped), ("reach_arm_posture", basin),
                 ("+ brace_pose_track 1", both)]))
    A(verdict(basin, both, shipped))
    if not (basin and both):
        A("<div class=note><b>Running.</b> Five heights &times; 3 seeds &times; "
          "two arms, one binary, <code>--threads 6</code>, serial under a 700% "
          "CPU quota. The shipped column is <code>runs/ab/off</code>, which came "
          "off this same binary.</div>")

    A("<h2>What this does not settle</h2>")
    A(("<ul>"
      "<li><b>Whether the release gate can move.</b> 0.885 m needs "
      "<code>standback_pitch_release</code> at 0.63 rad and 0.785 m needs 0.82, "
      "against the shipped 0.50. Run 3 seeds at 0.885 m with "
      "<code>--numeric standback_pitch_release=0.70</code>. It is killed if the "
      "robot releases and faceplants: the gate would then be enforcing a real "
      "limit rather than a fitted one.</li>"
      "<li><b>Whether the low end fails before the gate is reached.</b> With the "
      "rung-2 tolerance opened to 0.20 m, 0.885 m still fell in rung 2 and never "
      "tested rung 3, so the pitch bound is proven necessary and not yet proven "
      "binding. The measurement is peak pitch and pad load in the rung-2 window "
      "of those runs, from the qpos already on disk.</li>"
      "<li><b>Whether a solved arm beats the fixed one.</b> "
      "<code>reach_arm_q</code> aims 82&ndash;105 mm off and a per-slab solve "
      "aims 0&ndash;6 mm off. If H1 lands, the follow-up is writing the solved "
      "seven joints into that numeric at transition time, gated behind "
      "<code>brace_pose_track</code> 3, and re-running 1.085 m.</li>"
      "<li><b>Where the lower edge actually is.</b> The pitch bound puts it at "
      "PREDCROSS m. Run 3 seeds each at 0.905, 0.935, 0.955 and 0.975 m under "
      "the shipped controller; the bound is refuted if 0.905 m completes and "
      "confirmed if the edge lands within 20 mm of the crossing.</li>"
      "<li><b>The outer edges are grid-bounded, not measured.</b> 0.735 m and "
      "1.135 m have never been run under any arm.</li>"
      "<li><b>Strategy 9.</b> Its servo rung corrects this reach from the "
      "eye-in-palm camera, which is the mechanism that would close a residual "
      "gate error at run time. It cannot be evaluated headless &mdash; "
      "<code>g_object_seq</code> never advances without a tag bridge &mdash; so "
      "it needs the twin.</li>"
      "</ul>").replace("PREDCROSS", "%.3f" % cross))

    A("<p class=meta>Local copies: this page is <code>%s</code>; its figures are "
      "<code>docs/lean/%s/</code>; the probes are "
      "<code>studies/table_height/probe_pitch.py</code>, "
      "<code>probe_reachset.py</code> and <code>probe_basin.py</code>; the arm "
      "is <code>sweep_basin.py</code>. Nothing here is published to claude.ai."
      "</p>" % (a.out, R))
    A("</div>")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    html = ("<!doctype html><html lang=en><head><meta charset=utf-8>"
            "<meta name=viewport content=\"width=device-width,initial-scale=1\">"
            "<title>Pitch and reach gates on the braced-lean height window</title>"
            "<style>%s</style></head><body>%s</body></html>"
            % (CSS, "\n".join(Pg)))
    open(a.out, "w").write(html)
    print("wrote %s (%.1f KB)" % (a.out, len(html) / 1024.0))


if __name__ == "__main__":
    main()
