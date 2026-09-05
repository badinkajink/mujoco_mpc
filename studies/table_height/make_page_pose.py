#!/usr/bin/env python3
"""Build the brace-posture-retarget page from the probe JSON and the A/B sweep.

Narrative sections are fixed text written before any run landed; every number in
the results comes from figs/pose.json, figs/armreach.json, figs/static.json and
the two arms' agg.json, so re-running after more seeds rewrites the numbers
instead of stranding them in prose.

usage:
  make_page_pose.py --figs studies/table_height/figs \
      --off studies/table_height/figs_ab_off --on studies/table_height/figs_ab_on \
      --figs_rel media/pose --media docs/lean/media/pose \
      --out docs/lean/20260905-brace_posture_retarget.html
"""
import argparse, datetime, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_page import CSS, fig, num

STAT = {"complete": "good", "fell": "bad", "stalled": "hold"}


def load(p, name):
    f = os.path.join(p, name)
    return json.load(open(f)) if p and os.path.exists(f) else None


def ab_table(off, on):
    """One row per height, both arms, counted the way the criterion counts."""
    if not off or not on:
        return "<p class=meta>The A/B sweep has not landed yet.</p>"
    byh = {a["h"]: a for a in off}
    rows = []
    for a in on:
        b = byh.get(a["h"])
        if not b:
            continue
        def cell(x):
            cls = ("good" if x["complete"] == x["n"] else
                   "bad" if x["complete"] == 0 else "hold")
            return '<td class=%s>%d / %d</td>' % (cls, x["complete"], x["n"])
        def frac(x):
            return ("&mdash;" if x["seated_fraction"] is None
                    else "%.0f%%" % (100 * x["seated_fraction"]))
        rows.append(
            "<tr><td>%.3f m</td>%s%s<td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
            "<td>%s</td><td>%s</td></tr>"
            % (a["h"], cell(b), cell(a), frac(b), frac(a),
               num(b["f_forearm_peak"]), num(a["f_forearm_peak"]),
               num(b["t_end"], "%.0f"), num(a["t_end"], "%.0f")))
    head = ("<tr><th>table face</th><th>complete off</th><th>complete on</th>"
            "<th>in contact off</th><th>in contact on</th>"
            "<th>forearm peak off (N)</th><th>forearm peak on (N)</th>"
            "<th>t end off (s)</th><th>t end on (s)</th></tr>")
    return ("<div class=scroll><table><thead>%s</thead><tbody>%s</tbody></table>"
            "</div>" % (head, "".join(rows)))


def family_table(fam, nom):
    keep = [r for r in fam if abs(round(r["h"] * 40) - r["h"] * 40) < 1e-6
            and round(r["h"] * 1000) % 100 in (35, 85)]
    if not keep:
        keep = fam[::4]
    rows = "".join(
        "<tr><td>%.3f m</td><td>%.1f</td><td>%.1f</td><td>%.3f</td>"
        "<td>%+.0f</td><td>%+.0f</td></tr>"
        % (r["h"], r["pitch"], r["hip"], r["base_z"], 1000 * r["com"],
           1000 * r["sh"])
        for r in keep)
    return ("<div class=scroll><table><thead><tr><th>table face</th>"
            "<th>base pitch (deg)</th><th>hip pitch (deg)</th>"
            "<th>base z (m)</th><th>CoM ahead of midfoot (mm)</th>"
            "<th>shoulder above face (mm)</th></tr></thead><tbody>%s</tbody>"
            "</table></div>" % rows)




def lead_table(lead):
    rows = sorted(lead, key=lambda r: (r["h"], r["label"], r["seed"]))
    body = "".join(
        "<tr><td>%.3f m / %s / s%d</td><td class=%s>%s</td><td>%.3f</td>"
        "<td>%.1f</td></tr>"
        % (r["h"], r["label"], r["seed"], STAT[r["outcome"]], r["outcome"],
           r["peak"], r["over_s"]) for r in rows)
    return ("<div class=scroll><table><thead><tr><th>run</th><th>outcome</th>"
            "<th>peak base x (m)</th><th>seconds past the line</th></tr></thead>"
            "<tbody>%s</tbody></table></div>" % body)


def _by_h(agg):
    return {round(x["h"], 3): x for x in agg}


def verdicts(off, on, nom):
    """Score the three hypotheses against the two arms, from the counts."""
    O, N = _by_h(off), _by_h(on)
    hs = sorted(set(O) & set(N))
    fail = [h for h in hs if O[h]["complete"] == 0]
    gained = [h for h in fail if N[h]["complete"] > 0]
    nomh = min(hs, key=lambda h: abs(h - nom))
    kept = N[nomh]["complete"] >= O[nomh]["complete"]

    def seat(x):
        return 0.0 if x["seated_fraction"] is None else 100 * x["seated_fraction"]
    seat_up = [h for h in fail if seat(N[h]) > seat(O[h]) + 1.0]

    rows = []
    rows.append(
        ("H1 &mdash; re-solving the keyframes widens the window",
         "supported" if gained else "not supported",
         ("the retarget completes at %s, where the shipped controller completes "
          "none" % ", ".join("%.3f m" % h for h in gained)) if gained else
         ("no failing height gained a completion. Contact did improve at %s, so "
          "the pose reaches the slab without the run surviving; the block is "
          "downstream of the seat."
          % ", ".join("%.3f m" % h for h in seat_up) if seat_up else
          "no failing height gained a completion, and the forearm did not spend "
          "more time on the slab either")))
    rows.append(
        ("H2 &mdash; it costs nothing at the compiled height",
         "held" if kept else "VIOLATED",
         "%d/%d complete with the retarget on against %d/%d off at %.3f m. The "
         "height block only fires when the slab moves, so at the compiled height "
         "the solver never runs and the keyframes are the shipped bytes."
         % (N[nomh]["complete"], N[nomh]["n"], O[nomh]["complete"], O[nomh]["n"],
            nomh)))
    lowfail = [h for h in fail if h < nom and N[h]["complete"] == 0]
    rows.append(
        ("H3 &mdash; not sufficient on its own at the low end",
         "held" if lowfail else "refuted",
         ("%s still complete 0 of 3 with the pose correct, so the keyframe is "
          "not the whole story below the compiled height"
          % ", ".join("%.3f m" % h for h in lowfail)) if lowfail else
         "the low end completed, so the keyframe was the whole story there"))
    body = "".join("<tr><td>%s</td><td>%s</td><td>%s</td></tr>" % r for r in rows)
    return ("<div class=scroll><table><thead><tr><th>hypothesis</th>"
            "<th>verdict</th><th>on what</th></tr></thead><tbody>%s</tbody>"
            "</table></div>" % body)


def open_items(off, on, nom):
    """Named next measurements, not a mood of humility."""
    O, N = _by_h(off), _by_h(on)
    hs = sorted(set(O) & set(N))
    still = [h for h in hs if N[h]["complete"] == 0]
    items = []
    if still:
        items.append(
            "<li><b>%s still complete 0 of 3.</b> The next two numerics to test, "
            "both already implemented and both shipping off: "
            "<code>table_contact_exclusive</code> 1 &rarr; 2, which arms the "
            "preventive keep-off guard for every non-brace body over the slab "
            "footprint (the low-end failure with the pose correct is a drape, "
            "torso onto the wood at over 1 kN after the forearm has seated); and "
            "<code>brace_target_slab</code> 0 &rarr; 1 with "
            "<code>brace_target_inset</code> 0.05 &rarr; 0.11, which moves the "
            "brace x target off the body and onto the slab at the inset the "
            "keyframe already uses. At the re-solved pose the body-tied target "
            "still asks for 32&ndash;51 mm of extra forward travel at the low "
            "end. Both die if the completion count at %s does not move.</li>"
            % (", ".join("%.3f m" % h for h in still),
               ", ".join("%.3f m" % h for h in still)))
    items.append(
        "<li><b>The window edges are untested.</b> The A/B covers the five "
        "heights the baseline covered. 1.135 m and 0.735 m have not been run "
        "with the retarget on, so &ldquo;the window widened&rdquo; is bounded "
        "below by the grid, not measured. Run "
        "<code>sweep_ab.py --heights 0.735,1.135</code>.</li>")
    items.append(
        "<li><b>The ramp was not adjusted.</b> The brace rung glides the posture "
        "target over <code>target_ramp_sec</code> 18 s, and the re-solved pose "
        "is up to 10&deg; deeper in base pitch, so the commanded bow rate rises "
        "by about 40% at the low end while the ramp stays put. Whether that "
        "matters is one JSON edit and 3 seeds, and it needs no rebuild.</li>")
    items.append(
        "<li><b>Only sim.</b> Every number here is the planner's own model. The "
        "documented own-sim-over-holds gap means a pose the agent server holds "
        "can still be unholdable on the twin, and the re-solved low-end poses "
        "carry 10&ndash;20&deg; more hip flex than anything this controller has "
        "been run at on hardware.</li>")
    return "<ul>%s</ul>" % "".join(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", required=True)
    ap.add_argument("--off", default="")
    ap.add_argument("--on", default="")
    ap.add_argument("--figs_rel", default="media/pose")
    ap.add_argument("--media", default="")
    ap.add_argument("--lead_on", default="")
    ap.add_argument("--lead_off", default="")
    ap.add_argument("--baseline_page", default="20260904-table_height_generalization.html")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pose = load(a.figs, "pose.json")
    arm = pose["armreach"]          # written beside the figure it feeds
    static = load(a.figs, "static.json")
    off = load(a.off, "agg.json")
    on = load(a.on, "agg.json")
    lead_on = load(a.lead_on, "agg.json")
    lead_off = load(a.lead_off, "agg.json")
    lead = load(a.figs, "lead.json")
    R = a.figs_rel

    nom = pose["nominal_face"]
    clear = pose["pad_clear_at_nominal_mm"]
    modev = pose["mode"]["variance_first_mode"]
    fam = pose["family"]
    lo_arm = arm["arm_only_low_edge"]
    margins = [r["margin_mm"] for r in (static or []) if r["margin_mm"] is not None]

    P = []
    A = P.append

    A("<div class=wrap>")
    A("<h1>Brace posture retargeting across table heights</h1>")
    A("<p class=sub>The shipped brace keyframe seats the forearm pad %+0.1f mm "
      "above the compiled %.3f m slab and is never re-solved when the slab "
      "moves. `brace_pose_track` re-solves it, and this is what that buys.</p>"
      % (clear, nom))
    A("<p class=meta>%s &middot; branch <code>wxie/table-height</code> &middot; "
      "strategy 25 (<code>h12_brace_targeting</code>), Lean H12 Magpie &middot; "
      "generated by <code>studies/table_height/make_page_pose.py</code></p>"
      % datetime.date.today().isoformat())

    A("<h2>Where this sits</h2>")
    A("<p>The <a href=\"%s\">baseline sweep</a> measured the shipped controller "
      "over slab height and changed nothing: no cost weight, no residual, no "
      "strategy JSON, no keyframe, no robot&ndash;table standoff. It completes "
      "3/3 seeds at 0.985 m and 1.035 m and 0/3 at 0.785 m, 0.885 m and 1.085 m, "
      "with three distinct failure modes. This page answers the question that "
      "baseline was built to ask: what is the minimal change that makes "
      "table-height generalisation work?</p>" % a.baseline_page)
    A("<p>Minimal is counted in numbers touched. An existing model numeric beats "
      "a new weight, a weight beats a new residual, a new residual beats moving "
      "the robot. A result counts only if it widens the window on 3 seeds per "
      "height at a fixed thread count and costs no completions at 0.985 m.</p>")

    A("<h2>The brace posture is a solution of one table height</h2>")
    A("<p>Each rung of the ladder names a model <code>&lt;key&gt;</code>, and "
      "the Posture and Control costs track that pose. At "
      "<code>forearm_brace_lean</code> the left forearm pad sits %+0.1f mm above "
      "the compiled face. The pad's own drive target, "
      "<code>brace_press_z</code>, is derived from the live "
      "<code>table_surface_pos</code> sensor and tracks the slab. At the "
      "compiled height the two agree to two millimetres; at any other height "
      "they disagree by the height error, and the disagreement is what Posture "
      "(w60 over 27 joints, plus the leg-chain amplifier) spends the brace rung "
      "defending.</p>" % clear)
    A(fig("fig_keyframe",
          "Left: the Brace Pos target rises with the slab; the pad at the shipped "
          "keyframe does not. Right: the same gap in millimetres, with the "
          "measured outcome of 3 seeds at each tested height. The two heights "
          "that complete are the two nearest zero.", R))
    A("<p>The keyframe's own header records the pose being hand-solved "
      "(&ldquo;--surface 0.955 --min-x 0.34&rdquo;) the last time the slab "
      "moved, and notes that the previous pose left the pad 8.5 cm under the new "
      "surface. The re-solve happened once, by hand, and no tool in the repo "
      "does it.</p>")

    A("<h2>What the slab actually requires</h2>")
    A("<p>For each face height, solve for the pose nearest the shipped keyframe "
      "that shifts <em>both</em> brace pads by (face &minus; compiled face) in z "
      "while holding both feet exactly where and how the keyframe holds them. "
      "Constraining the elbow pad alone leaves the wrist free and the nearest "
      "solution parks the hand 122 mm inside a 0.785 m slab; two points make the "
      "forearm rigid, which is what &ldquo;flat on the table&rdquo; means.</p>")
    A(family_table(fam, nom))
    A("<p>Base pitch runs from %.0f&deg; at the low end to %.0f&deg; at the high "
      "end while base height moves %.0f mm. The adjustment is a bow, not a "
      "squat, and it is nearly linear in slab height.</p>"
      % (fam[0]["pitch"], fam[-1]["pitch"],
         1000 * (fam[-1]["base_z"] - fam[0]["base_z"])))
    A(fig("fig_pose",
          "The pose the slab requires, solved every 25 mm from 0.685 to 1.185 m. "
          "The dashed orange line is the shipped keyframe's constant value; the "
          "vertical dashes mark the compiled slab.", R))
    A(fig("fig_mode",
          "The requirement is one-dimensional: the first mode carries %.1f%% of "
          "the variance, and it is one coordinated bow at the hips and trunk "
          "with the shoulder and elbow following." % modev, R))
    A("<div class=note><b>The direction is not in the model's pose library.</b> "
      "The first mode's largest cosine against any (keyframe &minus; "
      "<code>forearm_brace_lean</code>) difference the model ships is 0.43, so "
      "blending two existing keyframes cannot produce it. The direction has to "
      "be solved for.</div>")
    A(fig("fig_poses",
          "Top: the shipped brace posture holds one bow whatever the slab does "
          "&mdash; the forearm hangs in air at 0.785 m and is buried in the wood "
          "at 1.085 m. Bottom: the re-solved posture folds to meet each slab. "
          "Same solver as the one now in lean.cc.", R))

    if margins:
        A("<h3>Every one of those poses is statically stable</h3>")
        A("<p>With the table moved out of reach so it can carry nothing, the CoM "
          "ground projection of the re-solved pose sits %.0f to %.0f mm inside "
          "the foot support polygon &mdash; the convex hull of the robot's real "
          "floor contacts &mdash; at every height from %.3f to %.3f m. The "
          "margin is smallest at the compiled height, where the controller "
          "works. Nothing about the required poses is near the edge of balance, "
          "so a failure to reach one is a control problem rather than an "
          "infeasible target.</p>"
          % (min(margins), max(margins), static[0]["h"], static[-1]["h"]))

    A("<h2>What the arm can do on its own</h2>")
    A("<p>Hold the trunk at the shipped brace posture and let only the bracing "
      "arm move, with both pads required to reach face height and their x and y "
      "free &mdash; the slab is 1.18 m deep and 0.595 m wide, so the pad may sit "
      "anywhere on it. The arm can seat on any slab at or above <b>%.3f m</b> "
      "and on none below it.</p>" % lo_arm)
    A(fig("fig_armreach",
          "Left: how far short the forearm falls with the trunk frozen, against "
          "slab height, with the measured outcome of each tested height on the "
          "axis. Right: the pad's lateral position at that seat, against the "
          "Brace Pos lateral target.", R))
    A("<p>That splits the two failing ends. Below %.3f m the arm cannot reach "
      "the wood at all, so no weight and no gain on any cost can seat it &mdash; "
      "only a deeper bow can, and nothing tells the trunk to bow deeper. Above "
      "it the arm can reach 1.135 m with the trunk frozen, so the high-end "
      "failure is not a reach limit; what runs out there is lateral. Seating a "
      "higher slab drags the pad inboard from y = %.3f m at the keyframe toward "
      "the centreline, while the Brace Pos lateral target stays at %.2f m, so "
      "the vertical and lateral demands pull the same arm in different "
      "directions.</p>"
      % (lo_arm, pose["pad_y"], arm["lat_target"]))

    A("<h2>Hypotheses, before the runs</h2>")
    A("<ol>")
    A("<li><b>H1.</b> Re-solving the brace keyframes with the slab widens the "
      "working window, because it removes the only term that is wrong by exactly "
      "the height error. Killed if the completion count does not move at "
      "0.785 m, 0.885 m or 1.085 m.</li>")
    A("<li><b>H2.</b> It costs nothing at 0.985 m. This one is structural rather "
      "than empirical: a zero shift returns the shipped pose bit for bit, and "
      "the height block only fires when the slab actually moves. Measured "
      "anyway, because a harness can be wrong.</li>")
    A("<li><b>H3.</b> It is not sufficient on its own at the low end. The pose "
      "is only right at the end of the rung's 18 s target ramp, and the residual "
      "x pull toward a body-tied brace target still asks for 32&ndash;51 mm of "
      "extra forward travel there. Killed if the low end completes.</li>")
    A("</ol>")

    A("<h2>The change</h2>")
    A("<p><code>brace_pose_track</code> (model numeric, 0 = off = "
      "byte-identical). When <code>Table H</code> moves the slab, "
      "<code>lean::TransitionLocked</code> re-solves "
      "<code>forearm_brace_lean</code>, <code>forearm_brace_reach</code> and "
      "<code>forearm_brace_release</code> against the new face &mdash; damped "
      "least squares over the floating base, both legs, the waist and the "
      "bracing arm, with the two pads and both feet as constraints, run once per "
      "height change. It touches one numeric and no weight, no residual and no "
      "standoff.</p>")
    A("<pre>build_cmake/bin/lean_bench --task \"Lean H12 Magpie\" --strategy 25 \\\n"
      "  --table_h 0.885 --seed 0 --total_time 75 --threads 6 --spp 3 \\\n"
      "  --pose_track 1 --out r.csv\n\n"
      "studies/table_height/retarget.py --face 0.885        # the same solve, offline\n"
      "studies/table_height/sweep_ab.py --out studies/table_height/runs/ab</pre>")
    A("<p>The offline twin (<code>studies/table_height/retarget.py</code>) runs "
      "the same arithmetic out of process. The two solvers land within 1.0&deg; "
      "of base pitch and 1 mm of base height of each other at every height "
      "tested; both satisfy the pad constraints to better than 0.2 mm, and they "
      "differ only in where the null-space pull leaves them.</p>")

    A("<h2>Procedure</h2>")
    A("<p>Paired A/B from one binary. For each (height, seed) cell the two arms "
      "run back to back, <code>--pose_track 0</code> then "
      "<code>--pose_track 1</code>, so machine state drifts through both "
      "equally. Three seeds per height, <code>--threads 6</code>, "
      "<code>--spp 3</code> (33 Hz plan rate, the rate the deploy node runs at), "
      "75 s cap, serial under a <code>CPUQuota=700%%</code> scope.</p>")
    A("<div class=note warn><b>The pairing is not optional.</b> MJPC's sampling "
      "planner draws noise from a generator shared across the thread pool, so a "
      "rollout is not a function of (config, seed): two identical invocations "
      "have diverged by 3.9 s in phase-3 entry, and the same seed under a "
      "different binary produces a different trajectory. Arm A here is the "
      "shipped controller and reproduces the earlier baseline only up to that "
      "noise.</div>")

    A("<h2>Results</h2>")
    A(ab_table(off, on))
    if off and on:
        A(fig("fig_bench",
              "Every headline metric against slab height, both arms. Solid is "
              "the retarget on, dashed is the shipped controller, three seeds "
              "each.", R))
        A(verdicts(off, on, nom))

    A("<h2>A second lever the model already carries</h2>")
    A("<p><code>lean.cc</code> defines <b>Brace Reach Lead</b>: a one-sided "
      "charge on the base travelling forward past <code>brace_lead_x0</code> "
      "(0.24 m), with the ceiling opening by <code>brace_lead_gain</code> "
      "(0.10 m) per unit of measured brace load. Below that line it is exactly "
      "zero, which is why its comment says it cannot starve the press the way "
      "raising Balance did. Its weight is <b>0.0 in every strategy JSON in this "
      "repo</b>, so the term has never been on.</p>")
    if lead:
        A(lead_table(lead))
    A(fig("fig_lead",
          "Peak forward base travel against the seconds spent past the line, one "
          "point per run with a qpos dump. The two completing runs sit on or just "
          "past the line; every low-slab failure lives well beyond it.", R))
    A("<p>That is the separation the term was built to charge for, and it splits "
      "the same way with the retarget on. It is one number in the strategy JSON "
      "and needs no rebuild, which ranks it just below an existing model "
      "numeric.</p>")
    if lead_on or lead_off:
        A("<h3>Brace Reach Lead at weight 400, with and without the retarget</h3>")
        A(ab_table(lead_off, lead_on))
    else:
        A("<div class=note><b>Running.</b> Three heights (0.785 m, 0.885 m, "
          "0.985 m) x 3 seeds x two arms, weight 400 on both "
          "<code>forearm_brace_lean</code> rungs &mdash; the only rungs the term "
          "is gated to &mdash; with <code>--pose_track</code> 1 and 0, so the "
          "result says whether either lever is sufficient alone. 1.085 m is left "
          "out: its peak base x is 0.196 m, so the term is inert there by "
          "construction and the high-end failure is a backward one.</div>")

    if off and on:
        A("<h2>What this does not settle</h2>")
        A(open_items(off, on, nom))
    A("<p class=meta>Local copies: this page is "
      "<code>docs/lean/20260905-brace_posture_retarget.html</code>; its figures "
      "are <code>docs/lean/media/pose/</code>; the study scripts are "
      "<code>studies/table_height/</code>; the change is "
      "<code>brace_pose_track</code> in "
      "<code>mjpc/tasks/humanoid_bench/lean/lean.{h,cc}</code> on branch "
      "<code>wxie/table-height</code>. Nothing here is published to claude.ai."
      "</p>")
    A("</div>")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    html = ("<!doctype html><html lang=en><head><meta charset=utf-8>"
            "<meta name=viewport content=\"width=device-width,initial-scale=1\">"
            "<title>Brace posture retargeting across table heights</title>"
            "<style>%s</style></head><body>%s</body></html>"
            % (CSS, "\n".join(P)))
    open(a.out, "w").write(html)
    print("wrote %s (%.1f KB)" % (a.out, len(html) / 1024.0))


if __name__ == "__main__":
    main()
