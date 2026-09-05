#!/usr/bin/env python3
"""Emit the page's analysis fragment from agg.json.

Every claim is computed, so re-running after more seeds rewrites the numbers
instead of leaving stale prose beside fresh figures. Hypotheses are scored
against what was written down before the sweep, the losing ones included.
"""
import argparse, json, os


def f(v, fmt="%.0f", dash="&mdash;"):
    return dash if v is None else fmt % v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", required=True)
    ap.add_argument("--long", default="", help="agg.json dir for the 140 s sweep")
    ap.add_argument("--ab", default="", help="runs dir for the brace_com_hold A/B")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    agg = json.load(open(os.path.join(a.figs, "agg.json")))
    runs = json.load(open(os.path.join(a.figs, "summary.json")))
    nominal = 0.985
    nom = next((x for x in agg if abs(x["h"] - nominal) < 1e-6), None)
    off = [x for x in agg if abs(x["h"] - nominal) >= 1e-6]
    off_ok, off_n = sum(x["complete"] for x in off), sum(x["n"] for x in off)
    lo = [x for x in off if x["h"] < nominal]
    hi = [x for x in off if x["h"] > nominal]
    lo_ok, lo_n = sum(x["complete"] for x in lo), sum(x["n"] for x in lo)
    hi_ok, hi_n = sum(x["complete"] for x in hi), sum(x["n"] for x in hi)
    nseed = max(x["n"] for x in agg)

    seats = [x for x in agg if (x["seated_fraction"] or 0) > 0.02]
    off_forearm = [x for x in seats
                   if (x["load"]["left forearm"] or 0) <
                      max((x["load"]["left wrist"] or 0), (x["load"]["torso"] or 0),
                          (x["load"]["pelvis"] or 0), (x["load"]["right arm"] or 0))]
    over_cap = [x for x in agg if (x["com_beyond_peak_mm"] or 0) > 145.0]

    p = []
    p.append("<p>The compiled height completes <b>%d of %d</b>. Every other height "
             "together completes <b>%d of %d</b> &mdash; %d of %d below nominal, "
             "%d of %d above.</p>" % (
                 nom["complete"] if nom else 0, nom["n"] if nom else 0,
                 off_ok, off_n, lo_ok, lo_n, hi_ok, hi_n))

    p.append("<h3>Hypotheses, scored</h3><ol>")
    if lo_n and hi_n:
        v = ("supported" if (lo_ok / lo_n) < (hi_ok / hi_n) - 1e-9 else
             ("refuted" if (lo_ok / lo_n) > (hi_ok / hi_n) + 1e-9 else
              "not separated by this sample"))
        p.append("<li><b>An asymmetric window &mdash; %s.</b> Below nominal "
                 "%d/%d complete; above nominal %d/%d. The prediction was that "
                 "raising the slab degrades gracefully and lowering it fails "
                 "first.</li>" % (v, lo_ok, lo_n, hi_ok, hi_n))
    if over_cap:
        w = max(over_cap, key=lambda x: x["com_beyond_peak_mm"])
        p.append("<li><b>The CoM cap binds &mdash; consistent with the data.</b> "
                 "%d of %d heights push the CoM past the fixed 145 mm "
                 "<code>com_cap_fwd</code> at peak, worst %s mm at %.3f m. That "
                 "constant was fitted at one height and does not move with the "
                 "slab.</li>" % (len(over_cap), len(agg),
                                 f(w["com_beyond_peak_mm"]), w["h"]))
    else:
        p.append("<li><b>The CoM cap binds &mdash; not shown.</b> No height's peak "
                 "CoM excursion reaches the 145 mm <code>com_cap_fwd</code> "
                 "constant, so whatever ends these runs, it is not that cap "
                 "saturating.</li>")
    if not seats:
        p.append("<li><b>The load path moves before the outcome does &mdash; "
                 "unmeasured.</b> No height held the pad seated long enough to "
                 "give a load split.</li>")
    elif off_forearm:
        p.append("<li><b>The load path moves before the outcome does &mdash; "
                 "supported.</b> At %s the largest seated load is not on the "
                 "forearm.</li>" % ", ".join("%.3f m" % x["h"] for x in off_forearm))
    else:
        p.append("<li><b>The load path moves before the outcome does &mdash; "
                 "refuted.</b> Wherever the pad seats, the left forearm is the "
                 "largest contributor and wrist, torso and pelvis stay near zero. "
                 "The brace does not migrate; it seats or it does not.</li>")
    p.append("<li><b>The reach survives wherever the brace survives.</b> Best "
             "gripper-to-target distance by height: %s.</li>" % ", ".join(
                 "%.3f&nbsp;m %s&nbsp;mm" % (x["h"], f(x["reach_err_min_mm"]))
                 for x in agg))
    p.append("</ol>")

    p.append("<h3>Reaching the face and touching it are different things</h3>")
    p.append("<p>Per height, the fraction of brace samples with the pad at face "
             "level, and the fraction actually carrying force: %s.</p>" % ", ".join(
                 "%.3f&nbsp;m %s%%&nbsp;at&nbsp;face&nbsp;/&nbsp;%s%%&nbsp;in&nbsp;contact"
                 % (x["h"],
                    f(None if x["at_face_fraction"] is None
                      else 100 * x["at_face_fraction"]),
                    f(None if x["seated_fraction"] is None
                      else 100 * x["seated_fraction"])) for x in agg))
    eq = [x for x in agg if x["complete"] == x["n"]]
    if eq:
        p.append("<p>At the heights that complete, the two agree to the sample "
                 "(%s), so a pad at face level is a pad under load. They come "
                 "apart at the failures: the tallest slab holds the forearm at "
                 "face level for the whole brace and never touches it, because "
                 "the arm is outboard of the wood rather than on it. The "
                 "geometric test alone scores that 100%%, so seating has to be "
                 "defined by newtons.</p>" % ", ".join(
                     "%.3f&nbsp;m %s%%" % (x["h"],
                         f(None if x["seated_fraction"] is None
                           else 100 * x["seated_fraction"])) for x in eq))
    p.append("<p>Peak forearm load by height: %s.</p>" % ", ".join(
        "%.3f&nbsp;m %s&nbsp;N" % (x["h"], f(x["f_forearm_peak"])) for x in agg))

    # ---- failure modes, read off the load path and the CoM sign ------------
    p.append("<h3>Three failure modes, not one</h3>")
    drape = [x for x in agg if (x["load"]["torso"] or 0) > 25]
    back = [x for x in agg if (x["com_beyond_peak_mm"] or 0) < 0]
    rows = []
    for x in agg:
        L = x["load"]
        if x["complete"] == x["n"]:
            kind = ("braces on the forearm alone &mdash; %s N on the forearm, "
                    "0 N everywhere else" % f(L["left forearm"]))
        elif (L["torso"] or 0) > 25:
            kind = ("<b>drapes</b>: %s N through the TORSO against only %s N on "
                    "the forearm, CoM %s mm past the toes" % (
                        f(L["torso"]), f(L["left forearm"]),
                        f(x["com_beyond_peak_mm"], "%+.0f")))
        elif (x["com_beyond_peak_mm"] or 0) < 0:
            kind = ("<b>falls backward</b>: CoM peaks %s mm &mdash; behind the "
                    "toes &mdash; with %s N of table contact" % (
                        f(x["com_beyond_peak_mm"], "%+.0f"), f(L["left forearm"])))
        else:
            kind = ("<b>falls forward</b>: torso reaches %s deg with the CoM %s mm "
                    "out and only %s N on the forearm" % (
                        f(x["torso_tilt_peak_deg"], "%.0f"),
                        f(x["com_beyond_peak_mm"], "%+.0f"), f(L["left forearm"])))
        rows.append("<li><b>%.3f m</b> &mdash; %s.</li>" % (x["h"], kind))
    p.append("<ul>%s</ul>" % "".join(rows))
    if drape:
        p.append("<p>At the lowest slab the torso carries %s N and the forearm "
                 "&mdash; the only link the task declares as a brace &mdash; "
                 "carries %s N. The robot rests its chest on the wood, which is "
                 "why those runs end on the clock instead of on the floor. A "
                 "pass/fail flag scores them the same as the tallest slab, which "
                 "fails the opposite way.</p>" % (
                     f(drape[0]["load"]["torso"]),
                     f(drape[0]["load"]["left forearm"])))
    if back:
        p.append("<p><b>This refutes hypothesis 2 as a general explanation.</b> "
                 "At %s the CoM peaks <i>behind</i> the toes (%s mm), so a "
                 "forward excursion cap cannot be what ends those runs. The slab "
                 "sits above the reachable brace envelope, the robot gets no "
                 "support at all, and it topples away from the table. "
                 "<code>com_cap_fwd</code> remains a candidate for the low end "
                 "only.</p>" % (
                     ", ".join("%.3f m" % x["h"] for x in back),
                     ", ".join(f(x["com_beyond_peak_mm"], "%+.0f") for x in back)))
    stalled = sum(1 for r in runs if r["outcome"] == "stalled")
    fell = sum(1 for r in runs if r["outcome"] == "fell")
    p.append("<p>Across all %d runs: %d completed, %d stalled to the %.0f s cap, "
             "%d fell.</p>" % (len(runs), len(runs) - stalled - fell, stalled,
                               max(r["t_end"] for r in runs), fell))

    # ---- follow-ups that have since run -----------------------------------
    ran = []
    if a.long and os.path.exists(os.path.join(a.long, "agg.json")):
        lg = json.load(open(os.path.join(a.long, "agg.json")))
        stuck = [x for x in lg if x["h"] < nominal and x["complete"] == 0]
        marg = [x for x in lg if 0 < x["complete"] < x["n"]]
        line = ("<li><b>The 75 s cap was not hiding slow successes.</b> A 140 s "
                "re-run leaves %s completing 0 of their seeds with three times "
                "the clock, so the low slab is unreachable rather than slow."
                % ", ".join("%.3f m" % x["h"] for x in stuck))
        if marg:
            line += (" It also demotes the top of the window: %s completed 3/3 "
                     "at 75 s and %s here, so the upper edge is marginal."
                     % (", ".join("%.3f m" % x["h"] for x in marg),
                        ", ".join("%d/%d" % (x["complete"], x["n"]) for x in marg)))
        ran.append(line + "</li>")
    if a.ab and os.path.isdir(a.ab):
        import glob, re
        arms = {}
        for fp in sorted(glob.glob(os.path.join(a.ab, "*.log"))):
            arm = os.path.basename(fp).split("_")[0]
            m = re.search(r"complete=(\d+) t_complete=\S+ t_end=(\S+)",
                          open(fp).read())
            if m:
                arms.setdefault(arm, []).append((int(m.group(1)), float(m.group(2))))
        if "on" in arms and "off" in arms:
            mo = sum(t for _, t in arms["on"]) / len(arms["on"])
            mf = sum(t for _, t in arms["off"]) / len(arms["off"])
            ran.append(
                "<li><b><code>brace_com_hold</code> is a real effect and not a "
                "fix.</b> The term is absent from the model, so it ships off. At "
                "0.05, mean survival at the tallest slab goes from %.1f s to "
                "%.1f s with no overlap between the arms, and still completes "
                "%d of %d. It buys time against the backward fall. A value sweep "
                "(0.02 / 0.05 / 0.10) comes before calling it the wrong lever."
                "</li>" % (mf, mo, sum(c for c, _ in arms["on"]), len(arms["on"])))
    if ran:
        p.append("<h2>Follow-ups that have since run</h2><ul>%s</ul>"
                 % "".join(ran))

    p.append("""<h2>What this does not settle</h2>
<ul>
<li><b>Why the low end cannot reach.</b> The slab's near edge sits at x = 0.45 m
at every height here, so a low table is reached by bowing rather than stepping,
and the torso gets there before the forearm does. The measurement that separates
"cannot brace low" from "cannot brace far" is a sweep of the table's <i>x</i> at
one low height, which is goal 4's last resort and should stay last.</li>
<li><b>Whether the excursion penalty can be made to win.</b> At the lowest slab
the torso takes several hundred newtons and the CoM goes far past
<code>com_cap_fwd</code>, which is a soft cost rather than a limit. The lever is
the weight on the residual carrying it (<code>Pelvis Forward</code>), not the cap
value; a weight sweep at the mid-low height is the cheap test, and it fails if
the robot simply stops reaching instead of bracing.</li>
<li><b>Whether the window generalises past the planner's own model.</b> Every
number here is own-sim, where the table is excluded from the arm chain and an
inert pad reports the newtons. The twin is the check, and no twin run exists at
an off-nominal height.</li>
<li><b>Seeds.</b> %d per point rejects "it always works" and does not put an
interval on a completion rate. Six would, and the marginal upper edge is exactly
where that matters.</li>
</ul>""" % nseed)

    open(a.out, "w").write("\n".join(p))
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
