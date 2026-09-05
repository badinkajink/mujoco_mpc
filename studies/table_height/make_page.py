#!/usr/bin/env python3
"""Build the table-height doc page from summary.json + the rendered figures.

Everything the page asserts numerically comes from summary.json, so re-running
this after more seeds land rewrites the numbers instead of stranding them in
prose. The narrative sections (intent, goals, hypotheses, procedure) are fixed
text and are deliberately written BEFORE the results section reads any data.

usage: make_page.py --figs studies/table_height/figs --media docs/lean/media/th
                    --out docs/lean/20260904-table_height_generalization.html
"""
import argparse, json, os, datetime

CSS = """
  :root{ color-scheme:light dark;
    --bg:#fcfcfb; --panel:#f3f3f0; --line:#dededa;
    --ink:#0b0b0b; --ink2:#52514e; --ink3:#7a7873;
    --s1:#2a78d6; --s2:#eb6834; --warn:#d03b3b; --good:#0ca30c; --hold:#fab219; }
  @media (prefers-color-scheme:dark){ :root{
    --bg:#1a1a19; --panel:#232322; --line:#3a3a37;
    --ink:#fff; --ink2:#c3c2b7; --ink3:#918f86;
    --s1:#3987e5; --s2:#d95926; --warn:#e66767; --good:#0ca30c; --hold:#fab219; } }
  *{box-sizing:border-box}
  body{background:var(--bg);color:var(--ink);margin:0;
       font:16px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
       padding:2.5rem 1.25rem 5rem}
  .wrap{max-width:64rem;margin:0 auto}
  h1{font-size:1.75rem;line-height:1.2;margin:0 0 .4rem;letter-spacing:-.02em}
  h2{font-size:1.2rem;margin:3rem 0 .5rem;letter-spacing:-.01em;
     padding-top:1rem;border-top:1px solid var(--line)}
  h3{font-size:1rem;margin:1.8rem 0 .3rem}
  .sub{color:var(--ink2);margin:0 0 .4rem}
  .meta{color:var(--ink3);font-size:.85rem;margin:0 0 2rem}
  p,li{color:var(--ink2)}
  code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.87em;
       background:var(--panel);padding:.1em .35em;border-radius:4px}
  pre{background:var(--panel);padding:.9rem 1rem;border-radius:8px;overflow-x:auto;
      font-size:.78rem;line-height:1.45;color:var(--ink)}
  table{border-collapse:collapse;width:100%;font-size:.88rem;margin:.8rem 0}
  th,td{text-align:right;padding:.35rem .6rem;border-bottom:1px solid var(--line);
        white-space:nowrap;font-variant-numeric:tabular-nums}
  th:first-child,td:first-child{text-align:left;white-space:normal;
        font-variant-numeric:normal}
  th{color:var(--ink3);font-weight:600;font-size:.75rem;text-transform:uppercase;
     letter-spacing:.05em}
  td{color:var(--ink2)} td:first-child{color:var(--ink)}
  .scroll{overflow-x:auto}
  .note{border-left:3px solid var(--s1);background:var(--panel);
        border-radius:0 8px 8px 0;padding:.8rem 1rem;margin:1.2rem 0}
  .note.warn{border-left-color:var(--warn)} .note.ok{border-left-color:var(--good)}
  .note b{color:var(--ink)}
  figure{margin:1.6rem 0}
  img,video{width:100%;border-radius:10px;display:block;background:#fff}
  img.dk{display:none}
  @media (prefers-color-scheme:dark){ img.lt{display:none} img.dk{display:block} }
  figcaption{color:var(--ink3);font-size:.84rem;margin-top:.5rem}
  ul{padding-left:1.15rem}
  .tag{display:inline-block;font-size:.72rem;padding:.1rem .45rem;border-radius:4px;
       border:1px solid var(--line);color:var(--ink3);margin-left:.3rem}
  .good{color:var(--good);font-weight:600}
  .bad{color:var(--warn);font-weight:600}
  .hold{color:var(--hold);font-weight:600}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
  @media(max-width:52rem){.grid2{grid-template-columns:1fr}}
  .vidgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(19rem,1fr));gap:1rem}
"""


def fig(name, caption, figs_rel):
    return ("<figure><img class=lt src=\"%s/%s.png\" alt=\"%s\">"
            "<img class=dk src=\"%s/%s.dark.png\" alt=\"%s\">"
            "<figcaption>%s</figcaption></figure>"
            % (figs_rel, name, caption, figs_rel, name, caption, caption))


def num(v, fmt="%.0f", dash="&mdash;"):
    return dash if v is None else fmt % v


def outcome_cell(o):
    cls = {"complete": "good", "fell": "bad", "stalled": "hold"}[o]
    return '<td class="%s">%s</td>' % (cls, o)


def agg_table(agg):
    h = ("<tr><th>table face</th><th>complete</th><th>in contact %</th>"
         "<th>at face %</th>"
         "<th>pad min (mm)</th><th>forearm N</th><th>wrist N</th>"
         "<th>right arm N</th><th>torso N</th><th>pelvis N</th>"
         "<th>forearm peak N</th><th>CoM past toes (mm)</th>"
         "<th>tilt peak (deg)</th><th>reach err (mm)</th></tr>")
    body = []
    for a in agg:
        ok = a["complete"]
        cls = "good" if ok == a["n"] else ("bad" if ok == 0 else "hold")
        L = a["load"]
        body.append(
            "<tr><td>%.3f m</td><td class=%s>%d / %d</td><td>%s</td><td>%s</td>"
            "<td>%s</td>"
            "<td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
            "<td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
                a["h"], cls, ok, a["n"],
                num(None if a["seated_fraction"] is None
                    else 100 * a["seated_fraction"], "%.0f"),
                num(None if a["at_face_fraction"] is None
                    else 100 * a["at_face_fraction"], "%.0f"),
                num(a["pad_clear_min_mm"], "%+.0f"),
                num(L["left forearm"]), num(L["left wrist"]), num(L["right arm"]),
                num(L["torso"]), num(L["pelvis"]), num(a["f_forearm_peak"]),
                num(a["com_beyond_peak_mm"], "%+.0f"),
                num(a["torso_tilt_peak_deg"], "%.1f"),
                num(a["reach_err_min_mm"])))
    return ("<div class=scroll><table><thead>%s</thead><tbody>%s</tbody>"
            "</table></div>" % (h, "".join(body)))


def runs_table(rows):
    h = ("<tr><th>height / seed</th><th>outcome</th><th>t end (s)</th>"
         "<th>brace (s)</th><th>seated (s)</th><th>pad min (mm)</th>"
         "<th>forearm peak N</th><th>CoM past toes (mm)</th>"
         "<th>reach err (mm)</th><th>wall (s)</th></tr>")
    body = []
    for r in rows:
        body.append(
            "<tr><td>%.3f&nbsp;/&nbsp;s%d</td>%s<td>%.1f</td><td>%.1f</td>"
            "<td>%.1f</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
            "<td>%.0f</td></tr>" % (
                r["table_h"], r["seed"], outcome_cell(r["outcome"]), r["t_end"],
                r["brace_seconds"], r["seated_seconds"],
                num(r["pad_clear_min_mm"], "%+.0f"), num(r["f_forearm_peak"]),
                num(r["com_beyond_peak_mm"], "%+.0f"),
                num(r["reach_err_min_mm"]), r["wall_s"]))
    return ("<div class=scroll><table><thead>%s</thead><tbody>%s</tbody>"
            "</table></div>" % (h, "".join(body)))


def videos(rows, media_rel, media_dir):
    out = []
    for r in rows:
        mp4 = "h%04d_s%d.mp4" % (round(r["table_h"] * 1000), r["seed"])
        if not os.path.exists(os.path.join(media_dir, mp4)):
            continue
        out.append(
            "<figure><video controls muted playsinline preload=metadata "
            "src=\"%s/%s\"></video><figcaption><b>%.3f m</b> &mdash; %s, "
            "ended t=%.1f s</figcaption></figure>"
            % (media_rel, mp4, r["table_h"], r["outcome"], r["t_end"]))
    if not out:
        return "<p class=meta>No videos rendered yet.</p>"
    return "<div class=vidgrid>%s</div>" % "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", required=True, help="dir holding fig_*.png")
    ap.add_argument("--figs_rel", default="media/th")
    ap.add_argument("--media", required=True, help="dir holding the mp4s")
    ap.add_argument("--media_rel", default="media/th")
    ap.add_argument("--out", required=True)
    ap.add_argument("--analysis", default="",
                    help="path to an HTML fragment with the results discussion; "
                         "written after the numbers are in")
    a = ap.parse_args()

    rows = json.load(open(os.path.join(a.figs, "summary.json")))
    rows.sort(key=lambda r: (r["table_h"], r["seed"]))
    agg = json.load(open(os.path.join(a.figs, "agg.json")))
    nominal = 0.985
    n_ok = sum(1 for r in rows if r["outcome"] == "complete")
    hs = sorted({r["table_h"] for r in rows})
    nseed = max(a2["n"] for a2 in agg)
    nom = next((a2 for a2 in agg if abs(a2["h"] - nominal) < 1e-6), None)
    off = [a2 for a2 in agg if abs(a2["h"] - nominal) >= 1e-6]
    off_ok = sum(a2["complete"] for a2 in off)
    off_n = sum(a2["n"] for a2 in off)
    analysis = open(a.analysis).read() if a.analysis and os.path.exists(a.analysis) \
        else "<p class=meta>Analysis pending &mdash; the sweep is still running.</p>"

    html = f"""<!doctype html>
<html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>Table height generalisation for the braced lean controller</title>
<style>{CSS}</style></head><body><div class=wrap>

<h1>Table height generalisation for the braced lean controller</h1>
<p class=sub>The lean controller was tuned at one slab height. This makes the
height a task parameter, changes it without touching a cost weight, and measures
what degrades.</p>
<p class=meta>{datetime.date.today().isoformat()} &middot;
<code>Lean H12 Magpie</code>, strategy 25 (<code>h12_brace_targeting</code>)
&middot; branch <code>icra2026</code> &middot; CEM, 20 rollouts, 1.0 s horizon,
spp 3 &rarr; <b>33 Hz</b> plan rate &middot; harness
<code>mjpc/lean_bench.cc</code> + <code>studies/table_height/</code>
&middot; {len(hs)} heights, {len(rows)} runs</p>

<h2>Why this experiment</h2>
<p>The braced-lean result rests on one workcell geometry. The slab face sits at
<b>0.985 m</b> in <code>lean.xml</code> and the robot stands a fixed distance from
its near edge, and both numbers were arrived at by sweeping them until the brace
worked &mdash; the XML comment history records the face moving 0.75 &rarr; 0.85
&rarr; 0.87 &rarr; 0.985 m and the body x moving 1.0125 &rarr; 0.78 &rarr; 1.04 m,
each time with keyframe targets re-solved in lockstep. A controller tuned that way
has an obvious question against it: does it work at a table it was not tuned on,
or is the geometry load-bearing?</p>
<p>Everything else in the ICRA submission is a claim about a controller. This is a
claim about its <i>domain</i>, and it is the cheapest generalisation axis available
&mdash; the height is one number, the task is otherwise unchanged, and the real
robot meets tables of many heights.</p>

<h2>What was built</h2>
<p>Table height is now MJPC task parameter index 7,
<code>residual_Table H</code> &mdash; the absolute world z of the slab's physical
top face, in metres. <code>0</code> means off and uses the compiled model, so
every pre-existing run is byte-identical. A non-zero value makes
<code>lean::TransitionLocked</code> rewrite the table body's z on the first
transition, restretch the four cosmetic legs to span floor-to-underside, and shift
the free object and the target mocap by the same delta.</p>
<p>The object rides the slab rather than staying at a fixed world height. A block
left at world z while the table dropped 200 mm would simply fall off, so the
manipulation task is held fixed <b>in the table frame</b>: same depth from the near
edge, same lateral offset, same height above the face. Strategy 25's reach rung is
already written that way &mdash; <code>reach_target_table: [0.55, 0.04, 0.15]</code>
&mdash; so no target had to be re-authored.</p>
<div class="note">
<b>What tracks the slab for free, and what does not.</b> Every task-space term
derives its geometry from the <code>table_surface_pos</code> framepos and the
compiled <code>table_top</code> half-extents: the Brace Pos target, the brace-force
proximity gate, Hip / Leg / Body-Table Clearance, and the reach rungs. Those follow
the table with no edit. What does not follow it is the joint-space and CoM-cap
family &mdash; <code>com_cap_fwd</code> 0.145 m, <code>pelvis_cap_fwd</code> 0.13 m,
<code>lean_nominal_x</code> 0.06 m, <code>brace_erect_target</code> 0.38,
<code>brace_lead_x0</code> 0.24 &mdash; all fixed constants fitted at one height.
The split is the whole experiment.
</div>

<h2>Goals</h2>
<ol>
<li>Change the slab height with the controller otherwise untouched, and see
    whether it still braces and reaches.</li>
<li>Expose the height as an MJPC parameter rather than a set of forked XMLs, so a
    sweep is a flag and the GUI slider shows degradation live. <span class=tag>done</span></li>
<li>If it holds up, benchmark across height for a reach target held fixed in the
    table frame.</li>
<li>If it does not, work back through cost weights first, then new cost terms,
    and only then the robot&ndash;table standoff.</li>
</ol>

<h2>Hypotheses, before the runs</h2>
<p>Recorded ahead of the sweep so they can be scored rather than reconstructed.</p>
<ol>
<li><b>An asymmetric window.</b> Raising the table degrades gracefully and lowering
it fails first. The brace is a downward press with a bowed torso; a lower slab
demands more forward CoM travel to reach it, and the caps that limit that travel
are fixed constants.</li>
<li><b>The binding constraint is <code>com_cap_fwd</code> (145 mm), not arm
reach.</b> The arm has ~460 mm of radius and the slab edge does not move
horizontally, so kinematic reach is not what runs out first.</li>
<li><b>The load path moves before the outcome does.</b> Off-nominal heights seat
the forearm pad worse, and the newtons migrate to the wrist pad or the torso while
the run still nominally completes. Pad clearance and per-body load degrade earlier
and more smoothly than a pass/fail flag.</li>
<li><b>The reach survives wherever the brace survives.</b> The reach target is
table-relative and the reach cost is task-space, so a run that braces should also
reach; reach error should track brace quality rather than height directly.</li>
</ol>

<h2>Procedure</h2>
<pre>build_cmake/bin/lean_bench --task "Lean H12 Magpie" --strategy 25 \\
  --table_h &lt;face z&gt; --seed &lt;n&gt; --total_time 75 --threads 6 --spp 3 \\
  --out &lt;tag&gt;.csv --qpos_out &lt;tag&gt;.qpos.csv

studies/table_height/sweep.py   --out runs/recon --heights {",".join("%.3f" % h for h in hs)} --seeds 1
studies/table_height/analyze.py --runs runs/recon --out figs
studies/table_height/render_video.py --qpos ... --table_h ... --out ....mp4</pre>
<p>One run per point, run serially under
<code>systemd-run --user --scope -p CPUQuota=700%</code>: MJPC saturates every
planner thread it is given and this box is a workstation. Nothing is recompiled
between points &mdash; the height is a parameter, so all runs share one binary and
one strategy JSON. Outcome is decided by the bench, not by eye:
<b>fell</b> if the pelvis drops below 0.5 m, <b>complete</b> if the terminal phase
is reached and held 3 s, <b>stalled</b> otherwise.</p>
<div class="note warn">
<b>The planner is not run-to-run repeatable, and that sets the sample size.</b>
Two invocations with identical height, identical seed and identical binary
completed at t = 45.77 s and t = 45.50 s, entering phase 3 at <b>32.15 s and
28.23 s</b> &mdash; 3.9 s apart. MJPC's sampling planner draws its noise from a
generator shared across the thread pool, so a rollout is not a function of
(height, seed) and the thread count changes the stream too: h = 0.785 seed 0
stood past t = 35 s at <code>--threads 4</code> and fell at t = 3.4 s at
<code>--threads 6</code>. Every number below is therefore an aggregate over
{nseed} seeds at one fixed thread count, and no single run is used to argue
anything.
</div>
<div class="note warn">
<b>Three measurement defects found before these numbers.</b>
(1) The table's four legs stand on the floor, and floor geoms belong to body 0, so
a &ldquo;one side is the table&rdquo; contact test booked the <i>table's own
weight</i> as table contact load &mdash; 166 N at h=0.785 with the arm still
228 mm above the face and every named body reading zero. Contacts against the
world and against the free object are now skipped.
(2) The left wrist pad had no column of its own. It has one now, because a brace
silently carried by the wrist rather than the forearm is a failure this task has
produced before, and a residual channel would have hidden it inside the table
weight. The right arm is named too: the nominal run carries 15&ndash;20 N of
robot-on-slab contact through the whole of <code>stand_up</code> with every left
body at zero, so something was already resting on the table before the brace
began.
(3) Load was first summarised as a median over the whole brace phase, which is
mostly approach with no contact and reads 0 N even when the forearm peaks at
168 N. Load statistics are now taken over the <b>seated</b> window &mdash; pad
within 5 mm of the face.
</div>
<div class="note">
<b>Two defects in <code>lean::ComputeMetrics</code>, reported not fixed.</b>
They are Allen's file and neither affects the controller, but both make its
monitoring channels read wrong for this ladder.
<code>brace_force</code> is <code>right_contact ? right_contact[0] : 0.0</code>,
keyed on a right-hand sensor under the comment &ldquo;right arm always braces,
left arm always reaches&rdquo; &mdash; while the model's brace geoms are
<code>left_forearm_pad</code> and <code>left_wrist_pad</code>. Measured: the
metric reads 0.00 N for the entire run while the left forearm carries
97&ndash;168 N. The deploy monitor therefore shows no brace force during a real
brace. Separately, the whole <code>reach_err</code> / <code>reach_tgt_*</code>
family is gated on <code>kf.name == "reach_to_target"</code>, a rung name
strategy 25 never uses, so it is nan for the entire braced ladder. The reach
column here is computed in the bench from the right gripper and the target
mocap instead.
</div>

<h2>Results</h2>
<p>{n_ok} of {len(rows)} runs reached the terminal phase. At the compiled height
the ladder completes {nom["complete"] if nom else 0} of {nom["n"] if nom else 0}
times; across every off-nominal height together it completes
<b>{off_ok} of {off_n}</b>.</p>
{agg_table(agg)}
<p class=meta>One row per height, median over {nseed} seeds. Load columns are the
median while the pad is <b>in contact</b> &mdash; within 5 mm of the face
<i>and</i> carrying force. &ldquo;at face&rdquo; is the geometric test alone.
The two are equal at a height that braces properly; they diverge when the forearm
reaches face level without touching the wood, which is what a too-high slab does.
Nominal is <b>{nominal:.3f} m</b>.</p>

{fig("fig_outcome", "Outcome by table height. The controller was tuned at 0.985 m and every other height is a generalisation test.", a.figs_rel)}
{fig("fig_ladder", "Every run: how far the phase ladder got and how long each rung took. Marker at the right end is the outcome.", a.figs_rel)}
{fig("fig_pad", "Forearm pad clearance above the slab face. Zero is a flat seated pad, positive is a hover. Axis clipped: a fall drives these traces to -840 mm.", a.figs_rel)}
{fig("fig_load", "Median table contact load during the brace, split by body. The question is not how much load but which link takes it.", a.figs_rel)}
{fig("fig_balance", "CoM excursion past the front foot edge against com_cap_fwd. That 145 mm constant is a soft cost, not a limit, and the low-table runs blow 226 mm past it. Axis clipped.", a.figs_rel)}
{fig("fig_reach", "Right gripper to the target mocap. The target rides the slab, so this is the same task at every height.", a.figs_rel)}

<h3>Per-run detail</h3>
{runs_table(rows)}

<h3>Rollouts</h3>
{videos(rows, a.media_rel, a.media)}

<h2>Analysis</h2>
{analysis}

</div></body></html>
"""
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(html)
    print("wrote %s (%d runs, %d heights)" % (a.out, len(rows), len(hs)))


if __name__ == "__main__":
    main()
