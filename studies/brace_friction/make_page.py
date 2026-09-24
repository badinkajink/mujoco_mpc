#!/usr/bin/env python3
"""Build docs/lean/20260924-brace_friction.html from the run records.

    ./make_page.py

Figures come from figs.py (PNG under docs/lean/media/brace_friction/, copied to
JPEG here), videos from render.py, numbers from page_numbers.py. House CSS is the
one studies/alg1_filmstrips/make_page.py uses.
"""
import html, os, sys
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from page_numbers import N  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "../../docs/lean")
MEDIA = os.path.join(DOCS, "media/brace_friction")
OUT = os.path.join(DOCS, "20260924-brace_friction.html")

CSS = '''
<style>
:root{--bg:#f7f6f2;--ink:#1e1e1c;--muted:#5f5d57;--rule:#dcd9d0;--accent:#c8102e;--card:#ffffff;--code:#efece4;}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#16171a;--ink:#ecebe6;--muted:#a3a19a;--rule:#33353a;--accent:#ff5c6e;--card:#1e2024;--code:#26282d;}}
:root[data-theme="dark"]{--bg:#16171a;--ink:#ecebe6;--muted:#a3a19a;--rule:#33353a;--accent:#ff5c6e;--card:#1e2024;--code:#26282d;}
body{background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:16px;line-height:1.5;margin:0;padding:0 16px;padding-block:24px 64px;}
main{max-width:1180px;margin:0 auto;}
h1{font-size:1.7rem;font-weight:600;line-height:1.2;margin:0 0 6px;text-wrap:balance;}
h2{font-size:1.25rem;font-weight:600;margin:48px 0 6px;padding-top:18px;border-top:1px solid var(--rule);text-wrap:balance;}
h3{font-size:1.05rem;font-weight:600;margin:28px 0 4px;}
p{max-width:78ch;margin:8px 0;}
ul{max-width:78ch;padding-left:1.2em;} li{margin:5px 0;}
.lede{color:var(--muted);max-width:78ch;}
code,.path{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:0.86em;}
code{background:var(--code);padding:1px 5px;border-radius:3px;}
table{border-collapse:collapse;margin:14px 0 10px;font-size:0.93rem;}
th,td{text-align:left;padding:6px 14px 6px 0;border-bottom:1px solid var(--rule);vertical-align:top;}
th{font-weight:500;color:var(--muted);}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:0.9em;white-space:nowrap;}
.wrap{overflow-x:auto;}
nav{display:flex;flex-wrap:wrap;gap:6px 18px;margin:14px 0 0;font-size:0.93rem;}
nav a{color:var(--accent);text-decoration:none;}
nav a:hover,nav a:focus{text-decoration:underline;}
figure{margin:22px 0 0;display:flex;flex-direction:column;gap:8px;}
figcaption{max-width:90ch;font-size:0.95rem;color:var(--muted);}
figcaption strong{color:var(--ink);font-weight:600;}
.path{display:block;color:var(--muted);margin-top:2px;}
figure img,figure video{max-width:100%;height:auto;display:block;background:var(--card);border:1px solid var(--rule);}
.vids{display:grid;grid-template-columns:1fr;gap:20px;}
a:focus-visible{outline:2px solid var(--accent);outline-offset:2px;}
.key{background:var(--card);border:1px solid var(--rule);border-left:3px solid var(--accent);padding:10px 14px;margin:16px 0;max-width:78ch;}
.key p{margin:4px 0;}
pre{background:var(--code);border:1px solid var(--rule);border-radius:4px;padding:12px 14px;overflow-x:auto;max-width:78ch;font-size:0.86em;line-height:1.45;}
pre code{background:none;padding:0;font-size:1em;}
</style>'''
FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">')


def jpg(name, w=1700, q=88):
    src = os.path.join(MEDIA, name + ".png")
    dst = os.path.join(MEDIA, name + ".jpg")
    if os.path.exists(src) and (not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src)):
        im = Image.open(src).convert("RGB")
        if im.width > w:
            im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
        im.save(dst, "JPEG", quality=q, optimize=True)
    return "media/brace_friction/" + name + ".jpg"


def figure(name, head, cap, src_note):
    rel = jpg(name)
    if not os.path.exists(os.path.join(MEDIA, name + ".jpg")):
        return ""
    return f'''
<figure>
  <figcaption><strong>{head}</strong> {cap}<span class="path">{html.escape(src_note)}</span></figcaption>
  <a href="{rel}" target="_blank" rel="noopener"><img src="{rel}" alt="{html.escape(head)}" loading="lazy"></a>
</figure>'''


def video(name, head, cap, src_note):
    p = os.path.join(MEDIA, name + ".mp4")
    if not os.path.exists(p):
        return ""
    rel = "media/brace_friction/" + name + ".mp4"
    return f'''
<figure>
  <figcaption><strong>{head}</strong> {cap}<span class="path">{html.escape(src_note)}</span></figcaption>
  <video src="{rel}" controls muted loop playsinline preload="metadata"></video>
</figure>'''


def cell_rows(n, batch, label):
    d = n.get(batch)
    if not d:
        return ""
    names = {"t1_f1": "1.0 / 1.0 (shipped)", "t0.8_f0.4": "0.8 / 0.4 (measured estimate)",
             "t0.6_f0.3": "0.6 / 0.3", "t0.8_f1": "0.8 / 1.0", "t1_f0.4": "1.0 / 0.4"}
    out = []
    for c in ("t1_f1", "t0.8_f0.4", "t0.6_f0.3", "t0.8_f1", "t1_f0.4"):
        if c not in d:
            continue
        r = d[c]
        out.append(
            f'<tr><td>{label}</td><td>{names.get(c, c)}</td><td class="num">{r["complete"]}/{r["n"]}</td>'
            f'<td class="num">{r["fell"]}</td><td class="num">{r["r_p95"]:.2f}</td>'
            f'<td class="num">{r["pad_slide"]:.0f}</td><td class="num">{r["foot_slide"]:.0f}</td>'
            f'<td class="num">{r["sole"]:.0f}</td></tr>')
    return "\n".join(out)


ASSIST_ORDER = ["none", "-40 N, once braced", "-20 N, last 50 mm", "-40 N, last 50 mm",
                "-20 N, whole approach", "-40 N, whole approach"]


def assist_table(n):
    A = n.get("assist", {})
    rows = [(k, A[k]) for k in ASSIST_ORDER if k in A]
    if len(rows) < 2:
        return ""
    out = []
    for k, r in rows:
        num = lambda v, f="%.2f": ("&ndash;" if v != v else f % v)
        out.append(f'<tr><td>{"No pull" if k == "none" else html.escape(k)}</td>'
                   f'<td class="num">{r["complete"]}/{r["n"]}</td>'
                   f'<td class="num">{r["fell_onset"]}</td><td class="num">{r["fell_late"]}</td>'
                   f'<td class="num">{num(r["r_impact"])}</td><td class="num">{num(r["r_p95"])}</td>'
                   f'<td class="num">{num(r["pad_slide"], "%.0f")}</td>'
                   f'<td class="num">{num(r["foot_slide"], "%.0f")}</td></tr>')
    return ('<div class="wrap"><table><thead><tr><th>Pull</th><th class="num">Completed</th>'
            '<th class="num">Fell at onset</th><th class="num">Fell later</th>'
            '<th class="num">Impact shear</th><th class="num">Pad shear p95</th>'
            '<th class="num">Pad slide (mm)</th><th class="num">Foot slide (mm)</th>'
            '</tr></thead><tbody>' + "\n".join(out) + '</tbody></table></div>')


def build():
    n = N()
    sweeps = "".join(cell_rows(n, b, lab) for b, lab in (
        ("b1_sd02", "σ 0.02"), ("b2_sd01", "σ 0.01 (deployed)"),
        ("b5_sd01_dz20", "σ 0.01, slab +20 mm"), ("b3_sd02_stiff", "σ 0.02, rigid slab")))
    sweep_table = f'''
<div class="wrap"><table>
<thead><tr><th>Planner</th><th>Plant μ, slab / floor</th><th class="num">Completed</th><th class="num">Fell</th>
<th class="num">Pad shear p95</th><th class="num">Pad slide (mm)</th><th class="num">Foot slide (mm)</th>
<th class="num">Sole moved (mm)</th></tr></thead>
<tbody>
{sweeps}
</tbody></table></div>
<p class="lede">Medians over the seeds that produced a contact record (12 per cell, less one
&mu;&nbsp;0.6/0.3 run the host's memory watchdog killed mid-run). Pad slide is the contact-point slip
integrated over the time the pads carry more than 20&nbsp;N; foot slide the same for the worse foot
above 50&nbsp;N; sole moved is how far the sole's material point ends up from where it stood when the
lean rung began.</p>''' if sweeps else ""

    sections = []
    sections.append(("ladder", "The friction ladder never varied the brace", f'''
<p>With <code>--plant_friction_scale 0.4</code>, the two surfaces that carry the brace stay at
&mu;&nbsp;=&nbsp;1.0. The forearm pad and the wrist pad meet the slab through explicit
<code>&lt;pair&gt;</code> elements, and MuJoCo takes a declared pair's friction from
<code>pair_friction</code>; the flag scales <code>geom_friction</code> only. Everything else on the
slab &mdash; the gripper box, the jaw, the object &mdash; does drop to 0.4, and so do the feet.</p>
<div class="wrap"><table>
<thead><tr><th>Contact</th><th>Route</th><th class="num">&mu; at <code>--plant_friction_scale 0.4</code></th></tr></thead>
<tbody>
<tr><td><code>left_forearm_pad</code> &ndash; slab</td><td>declared <code>&lt;pair&gt;</code></td><td class="num">1.00</td></tr>
<tr><td><code>left_wrist_pad</code> &ndash; slab</td><td>declared <code>&lt;pair&gt;</code></td><td class="num">1.00</td></tr>
<tr><td><code>left_gripper_jaw_a</code> &ndash; slab</td><td>geom, slab <code>priority=1</code></td><td class="num">0.40</td></tr>
<tr><td><code>left_gripper_collision</code> &ndash; slab</td><td>geom, slab <code>priority=1</code></td><td class="num">0.40</td></tr>
<tr><td>object &ndash; slab</td><td>geom, slab <code>priority=1</code></td><td class="num">0.40</td></tr>
<tr><td>sole &ndash; floor</td><td>geom &times; geom</td><td class="num">0.40</td></tr>
</tbody></table></div>
<p>So the &mu;&nbsp;&times;0.6 and &times;0.4 rows of the planner ablation (CEM 11/12 and 11/12,
<code>studies/planner_ablation/HANDOFF.md</code> &sect;2j-result) are a foot-friction ladder with the
arm&ndash;table contact pinned at 1.0. The axis the hardware crossed was not on the grid.
<code>--plant_table_mu</code> and <code>--plant_foot_mu</code> set each surface directly, pairs
included, and echo what the contact set ends up using.</p>'''))

    sections.append(("demand", "Friction the simulated brace draws on", f'''
<p>Across the {n['braced']} braced runs of the ablation corpus, all at &mu;&nbsp;=&nbsp;1.0 on both
surfaces, the brace pads' peak shear ratio has median {n['pad_peak_med']:.2f} and passes 0.8 in
{n['pad_over_08']} of them; the worse foot's peak has median {n['foot_peak_med']:.2f} and passes 0.5 in
{n['foot_over_05']}. The shear ratio |F<sub>xy</sub>|/F<sub>z</sub> is the friction coefficient that
contact needs in order not to slide, so those runs are asking both surfaces for more than the real
ones can supply &mdash; a slab at &mu;&nbsp;&asymp;&nbsp;0.8 and aluminium on smooth concrete
somewhere near 0.3&ndash;0.5.</p>
<p>Sustained demand is lower than peak: the pads' 95th percentile over a hold has median
{n['pad_p95_med']:.2f} and exceeds 0.8 in only {n['pad_p95_over_08']} runs, the feet's 99th percentile
median {n['foot_p99_med']:.2f}. The excursions past the real limit are brief, which is consistent with
the hardware failing at the moments of load transfer rather than through the whole hold.</p>
{figure("fig_demand", "Shear demand at &mu; = 1.0.",
        "Per-run peak and sustained shear ratio over the braced rungs, from the "
        "planner-ablation corpus re-scored for contact forces. Shaded band = past the real surface's "
        "coefficient.", "studies/brace_friction/figs.py fig_demand")}'''))

    sections.append(("creep", "The simulated contact creeps instead of sticking", f'''
<p>A loaded brace pad slides the whole time it is loaded. Pooling {n['pool']:,} loaded samples, the
median contact slip speed is {n['slip_lt01']:.1f}&nbsp;mm/s below a shear ratio of 0.1,
{n['slip_03']:.0f}&nbsp;mm/s at 0.3, and {n['slip_08']:.0f}&nbsp;mm/s at 0.8. The pads are slipping
faster than 2&nbsp;mm/s for {n['frac_slipping']*100:.0f}% of their loaded time, and in
{n['frac_slip_below_cone']*100:.0f}% of that the contact sits below its own friction cone &mdash; it is
not sliding because it ran out of friction. Over one braced hold the pad's contact point travels a
median {n['pad_slide_med']:.0f}&nbsp;mm along the slab (10&ndash;90%:
{n['pad_slide_q'][0]:.0f}&ndash;{n['pad_slide_q'][1]:.0f}&nbsp;mm), and the pad centre's path over the
wood matches that to within about 10%, so it is the contact sliding, not the arm rolling.</p>
<p>The mechanism is the contact model. The brace pairs ship <code>solref 0.02</code> with a soft
impedance onset (<code>solimp 0.015 1 0.022</code>) and the slab geom <code>solref 0.08</code>; friction
in MuJoCo is a regularised constraint, so tangential force is produced <em>by</em> tangential
displacement. A contact carrying half its friction limit therefore creeps at a steady rate rather than
holding still, and there is no breakaway: the curve below has no flat stick segment anywhere.</p>
{figure("fig_creep", "Slip against shear.",
        "Median and 10&ndash;90% band of pad contact slip speed against shear ratio, pooled over the "
        "braced runs, with a Coulomb contact at &mu; = 0.8 for reference: zero until the limit, then "
        "sliding.", "studies/brace_friction/figs.py fig_creep")}
<p>The consequence for this study is that lowering &mu; on the plant cannot, on its own, produce the
hardware failure: it moves the creep rate and the point at which the cone is reached, but there is
still no moment at which the brace releases. The arms that would change that are
<code>cone="elliptic"</code>, <code>noslip_iterations</code>, and a stiffer contact; the first probe
is in the open questions below.</p>'''))

    sections.append(("sweep", "Friction set per surface", f'''
<p>These runs set slab and floor friction on the plant directly and leave the planner's model at
&mu;&nbsp;=&nbsp;1.0, which is what the robot's planner believes. Strategy 25 on the deploy joint
gains at 33&nbsp;plans/s, CEM, 12 seeds per cell. They also log contact every 2&nbsp;ms rather than
once per plan, and reproduce the creep at that resolution, so the slide numbers above are not an
artefact of sampling the ablation corpus at 33&nbsp;Hz.</p>
{sweep_table}
<p>Completion is flat across the three cells while the sliding is not: 7, 9 and 7 runs finish the
ladder, and the worse foot's slide over one run goes 41&nbsp;&rarr;&nbsp;127&nbsp;&rarr;&nbsp;397&nbsp;mm
as the floor coefficient drops. The pads' slide barely moves, because at a lower coefficient the pad
reaches its cone sooner but was already creeping below it. A friction sweep scored on completion
alone reports that this controller is insensitive to friction; scored on how far the feet travelled,
it reports a tenfold change.</p>
{figure("fig_slide", "How far the contacts slide.",
        "Per-run pad slide and worse-foot slide by friction cell; &times; marks a run that fell, "
        "the bar is the median.", "studies/brace_friction/figs.py fig_slide")}
<p>Two of the &mu;&nbsp;0.6/0.3 falls do run the hardware's sequence: the loaded foot reaches its
friction cone and skates, the brace unloads, and the robot goes over backwards, with the worse foot
having slid 736 and 886&nbsp;mm by the end. They are the only runs in this study that do.</p>'''))

    sections.append(("touchdown", "Descent speed and the impact", f'''
<p>With the slab where the planner expects it, the pad meets it at 112&ndash;318&nbsp;mm/s downward
in every one of the 36 runs, and the peak shear over the next 0.3&nbsp;s runs 0.2&ndash;0.8. Raising
the plant's slab 20&nbsp;mm above the planner's model does <em>not</em> make that impact harder, which
is the opposite of what this arm was built to test: 7 of 9 seeds per cell then graze the higher
surface at 2&ndash;20&nbsp;mm/s, during a slower part of the arm's swing, and settle onto it. The
pads end up carrying less, not more &mdash; median pad slide 107&nbsp;mm against 199 at the same
friction, and no pad load at all at &mu;&nbsp;=&nbsp;1.0, where 4 of 9 runs fall before the brace
ever forms.</p>
<p>Descent speed in this task is set by the lean rung's 18&nbsp;s posture ramp in
<code>h12_brace_targeting.json</code>, and the bench has no flag that shortens it. Getting a
genuinely faster approach means a copy of that strategy with a shorter
<code>target_ramp_sec</code> on rung 1, which is the next measurement rather than a result here.</p>
{figure("fig_touchdown", "Touchdown speed against impact shear.",
        "Pad downward speed averaged over the 20 ms before first contact, against the peak shear "
        "ratio in the 0.3 s after it. The +20 mm arm puts the plant's slab higher than the planner's "
        "model, so the pad is met earlier in its descent.",
        "studies/brace_friction/figs.py fig_touchdown")}'''))

    sections.append(("assist", "Pulling the robot back into the brace", f'''
<p>On hardware the lean is made to settle by an operator pulling the robot backwards as it comes
down, so it drops into the brace rather than shearing forward along the slab.
<code>--assist_fx</code> is that force: a backward pull on the torso written into
<code>xfrc_applied</code>, which the planner's state copy does not carry, so the planner is as blind
to it as it is to a person. <code>--assist_gap</code> decides when it acts &mdash; only while the
forearm pad is within that distance of the slab face.</p>
{assist_table(n)}
<p class="lede">Six seeds per condition at slab&nbsp;&mu;&nbsp;0.8 / floor&nbsp;&mu;&nbsp;0.4, CEM
&sigma;&nbsp;0.02. "Whole approach" is a 150&nbsp;mm gate, which is open from the moment the lean
rung begins because the pad already hovers 66&nbsp;mm above the face during the stand.</p>
{figure("fig_assist", "When the pull is applied decides whether it hurts.",
        "Outcome of six seeds per condition, split by whether the run finished the ladder, fell at "
        "the lean onset before the brace ever formed, or fell later.",
        "studies/brace_friction/figs.py fig_assist")}
<p>No window tested makes the brace better. Held through the approach, a 20&nbsp;N pull takes the
ladder from 5 of 6 completions to 0 of 6, and all six falls are the lean-onset backward fall before
the brace ever forms; 40&nbsp;N does the same. Confined to the last 50&nbsp;mm of the approach,
20&nbsp;N is neutral (5 of 6, as the baseline) and 40&nbsp;N costs four runs. Applied only after the
brace is down, 40&nbsp;N loses all six.</p>
<p>The mechanism reading is that the pull never touches the quantity it is supposed to fix. The pad's
peak shear ratio over the 0.3&nbsp;s after touchdown is 0.50&ndash;0.59 in every condition including
the baseline, and the pad's slide along the slab goes up, not down, under the one pull that does not
cost completions (310&nbsp;mm against 158). What the pull does change is the margin at the lean onset,
which is where this ladder has least to spare: the controller is driving the CoM forward over the feet
on a 12&nbsp;s ramp, and a backward force applied there is the disturbance it is least able to absorb.
The planner sees the resulting state at 33&nbsp;Hz, as it does on the robot, but not the force.</p>
<p>The honest conclusion is that this intervention cannot be tuned against this plant. On hardware the
pull works because it counteracts a real slip; here there is no breakaway for it to counteract, so it
enters as a pure disturbance. Getting the pull-back dialled in in simulation needs the contact model
fixed first &mdash; the order of work is the open questions below, not a sweep over force and timing.</p>
'''))

    sections.append(("chain", "One run, contact by contact", f'''
{figure("fig_chain", "The contact record of a single run.",
        "Normal force, shear ratio against each surface's coefficient, contact slip speed and pelvis "
        "height on one clock.", "studies/brace_friction/figs.py fig_chain")}
<div class="vids">
{video("vid_mu_compare", "Shipped friction against the measured estimate.",
       "Same seed, same planner, left at &mu; 1.0/1.0 and right at 0.6/0.3. Arrows are the shear "
       "force each surface applies to the robot, drawn flat on the surface (green below 0.6 of that "
       "surface's &mu;, amber to 0.95, red at the limit). Dots trace the pad's contact line and any "
       "sole that slid.", "studies/brace_friction/render.py")}
{video("vid_collapse", "The one sequence that matches the hardware.",
       "&mu; 0.6/0.3, seed 2, real time. The left foot reaches its friction cone and skates, the "
       "brace unloads, and the robot goes over backwards; the worse foot has slid 886 mm by the end.",
       "studies/brace_friction/render.py")}
{video("vid_dz20", "A slab 20 mm higher than the planner believes.",
       "Deployed settings, &mu; 0.8/0.4. The pad is met while the torso is still coming down.",
       "studies/brace_friction/render.py")}
</div>'''))

    sections.append(("repro", "Reproducing this", '''
<p>Everything here is on branch <code>wxie/brace-payload</code> in <code>mujoco_mpc</code>, study
directory <code>studies/brace_friction/</code>.</p>
<pre><code>cmake --build build_cmake --target lean_bench -j 6

# the sweeps (2 jobs x 6 threads through resguard.sh run; ~80 s per run)
studies/brace_friction/run_batches.sh

# score, figures, videos, page
studies/brace_friction/analyze.py runs/b1_sd02
studies/brace_friction/replay_ablation.py --dirs gains_spp15,mismatch_spp15_mu0.6,mismatch_spp15_mu0.4 \\
    --arms all --out runs/replay_ablation.jsonl --tracks runs/replay_tracks
studies/brace_friction/figs.py --out ../../docs/lean/media/brace_friction
MUJOCO_GL=egl studies/brace_friction/render.py --runs runs/b1_sd02 \\
    --tags t1_f1_s3,t0.6_f0.3_s3 --out ../../docs/lean/media/brace_friction/vid_mu_compare.mp4
studies/brace_friction/make_page.py</code></pre>
<p>New <code>lean_bench</code> flags: <code>--plant_table_mu</code>, <code>--plant_foot_mu</code>,
<code>--planner_table_mu</code>, <code>--planner_foot_mu</code> (sliding friction per surface, plant
and planner separately, pairs included); <code>--plant_table_stiff 1</code> (slab contacts to
<code>solref 0.01</code>, <code>solimp 0.95 0.99 0.001</code>); <code>--plant_table_dz</code> (the
plant's slab offset from the planner's); <code>--contact_out f.csv --contact_hz 500</code> (per-step
brace, foot and pad contact forces, friction-cone use, contact slip speed, pad and sole positions).
<code>replay_ablation.py</code> reconstructs the same channels from the 2026-09 ablation's state
tracks, which recorded state but not contact forces.</p>'''))

    sections.append(("open", "What this does not settle", '''
<ul>
<li><strong>Both coefficients are estimates.</strong> Drag the forearm pad across the real slab under
a 50&ndash;150 N normal load with an inline force gauge and read the breakaway and sliding values; do
the same for a foot on the lab floor. This page assumes 0.8 for the slab, as reported from the
hardware, and 0.3&ndash;0.5 for aluminium on smooth concrete. Both feed <code>--plant_table_mu</code> /
<code>--plant_foot_mu</code> directly.</li>
<li><strong>The creep is not fixed by lowering &mu;.</strong> <code>--plant_table_stiff</code>
shortens the compliance length but leaves MuJoCo's friction regularised. The test: at
<code>solref 0.01</code>, does the pad's slide over a hold at shear ratio 0.3 fall below
20 mm? If it does not, the next lever is <code>cone="elliptic"</code> plus
<code>noslip_iterations</code>, which is a two-line change in
<code>Lean_H12_Magpie.xml</code> and a re-run of batch b1.</li>
<li><strong>The planner is never told the friction.</strong> <code>--planner_table_mu</code> /
<code>--planner_foot_mu</code> exist and are unswept. The measurement: batch b1 cells with the
planner's model at 0.8/0.4 as well, asking whether a planner that knows the limit keeps the pad's
shear below it, or whether the shear is set by the posture targets regardless.</li>
<li><strong>Completion is the wrong score here.</strong> Runs complete with the soles hundreds of
millimetres from where they started. Until a run is scored on sole displacement and pad slide as
well as on reaching the last rung, a friction sweep will keep reporting that the controller is
robust.</li>
<li><strong>Nothing here makes the robot come down faster.</strong> The bench's descent is the
lean rung's 18&nbsp;s <code>target_ramp_sec</code>, and <code>--plant_table_dz</code> turned out to
soften the touch rather than sharpen it. The measurement: a copy of
<code>h12_brace_targeting.json</code> with rung 1's ramp at 9 and 4.5&nbsp;s, run at
&mu;&nbsp;0.8/0.4, asking whether impact shear crosses 0.8 and whether the pads then unload. Pair it
with the hardware number &mdash; log the pad's vertical speed from the robot's state estimate through
the lean rung and compare it with the 112&ndash;318&nbsp;mm/s the bench shows.</li>
<li><strong>The pull-back was tested at one place and two sizes.</strong> 20 and 40&nbsp;N on
<code>torso_link</code>, against a 674&nbsp;N robot. A person steadying the machine may take the
pelvis rather than the torso, may push up as well as back, and lets go on contact. The measurement
that would settle it: instrument the real assist &mdash; a load cell in line with whoever is holding
the robot &mdash; and replay the measured force profile through <code>--assist_fx</code> with
<code>--assist_body pelvis</code>.</li>
<li><strong>No estimator noise and no actuator dynamics.</strong> The bench drives the plant from
true state through an ideal joint PD. Both add phase lag at exactly the moment the brace lands.</li>
</ul>'''))

    nav = "\n".join(f'<a href="#{i}">{html.escape(t)}</a>' for i, t, _ in sections)
    body = "\n".join(f'<h2 id="{i}">{html.escape(t)}</h2>{b}' for i, t, b in sections)
    lede = f'''<p class="lede">On the real H1-2 the braced lean slips at the arm, then at the feet,
then collapses, and it is made to settle by an operator pulling the robot backwards into the brace.
In simulation none of that happens. Two reasons, both measured here: the friction robustness ladder
this project already ran never lowered arm&ndash;table friction at all, and MuJoCo's contact lets a
loaded brace pad creep continuously rather than hold and break away, so the plant has no event that
corresponds to the arm letting go &mdash; and no purchase for the pull-back that fixes it on
hardware. {n['repl_runs']} runs of the existing corpus re-scored for contact forces, plus new sweeps
with friction set per surface and with the operator's pull applied at three different moments.</p>'''
    doc = "\n".join(["<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
                     '<meta name="viewport" content="width=device-width, initial-scale=1">',
                     "<title>Brace and foot friction</title>", FONTS, CSS, "</head><body>",
                     "<main>",
                     "<h1>Arm&ndash;table and foot&ndash;floor friction in the braced lean</h1>",
                     lede, f"<nav>{nav}</nav>", body, "</main></body></html>"])
    os.makedirs(DOCS, exist_ok=True)
    open(OUT, "w").write(doc)
    print("->", OUT, len(doc), "bytes")


if __name__ == "__main__":
    build()
