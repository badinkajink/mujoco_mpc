#!/usr/bin/env python3
"""Contact-sheet pages for the Algorithm 1 filmstrips, with 1800 px JPEG copies of
the figures under docs/lean/media/alg1_filmstrips/.

    ./make_page.py filmstrips   # docs/lean/20260921-alg1_filmstrips.html   (lean onset, sigma sets)
    ./make_page.py contact      # docs/lean/20260921-alg1_brace_contact.html (the pad landing)

Each page is a config below: a list of sections, each a run + tick prefix + the
figure suffixes to show, with the numbers for the caption written by hand from
render_alg1.py / contact_stats.py output.
"""
import html, os, sys
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "../../docs/lean")
MEDIA = os.path.join(DOCS, "media/alg1_filmstrips")
os.makedirs(MEDIA, exist_ok=True)

STAGE = {
    "A_center": ("line 1", "θ̄ ← Center(θ)", "Six frames of θ̄'s rollout over the 1 s horizon (faint grey = x₀), its running cost, and the spline θ̄ itself for eight joints."),
    "B_spread": ("lines 3–4", "ε(k) ~ N(0, Σ), θ(k) ← θ̄ + ε(k)", "Σ as a 3 knots × 27 joints heatmap of the std sampled at this tick (refit elite std, floored at std_min), and the 20 perturbed splines around θ̄."),
    "C3_fan_strip": ("line 5", "roll out f from x₀, all 20 at once", "The fan as a filmstrip: every rollout at h = 0.25, 0.5, 0.75, 1.0 s, tinted by J(k) (yellow = lowest); black = θ̄'s rollout; traces = hands and pelvis."),
    "C1_rollouts_fan": ("lines 5–6", "J(k) ← mean cost", "All 20 end states on one image next to the sorted J(k) bars."),
    "C2_rollouts_strips": ("lines 5–6", "four rollouts, frame by frame", "Best, two mid-ranked and worst candidate as filmstrips with their running cost against θ̄'s."),
    "D_fold": ("line 8", "θ ← Fold({θ(k)}, {J(k)})", "The 6 elites in the J bars, the elite fan with the rollout of their mean (red outline), and the elite splines → mean for eight joints."),
    "D2_fold_strips": ("line 8", "θ̄'s rollout over θ's rollout", "Both from the same x₀; the difference is one iteration."),
    "E_overview": ("all", "the four stages on one panel", "Centre, spread, fan, fold."),
    "G1_contact": ("lines 5–8", "the brace contact inside the rollouts", "Detail camera on the forearm pad at h = 1 s with pad and right-hand traces; pad clearance to the table face along every rollout (grey band = landed); J(k) bars with landers solid and hoverers hatched, elites shaded."),
    "G2_contact_strip": ("line 5", "the pad landing along the horizon", "Detail camera, all 20 rollouts at h = 0.25, 0.5, 0.75, 1.0 s; the frame title counts how many have the pad down by then."),
    "F_sigma_compare": ("line 5", "the fan at three sampling widths", "Separate runs of the same seed, each at t = 12.51 s, all 20 rollouts at h = 1 s."),
}
FULL = ["A_center", "B_spread", "C3_fan_strip", "C1_rollouts_fan", "C2_rollouts_strips", "D_fold", "D2_fold_strips", "E_overview"]

CSS = '''
<style>
:root{--bg:#f7f6f2;--ink:#1e1e1c;--muted:#5f5d57;--rule:#dcd9d0;--accent:#c8102e;--card:#ffffff;--code:#efece4;}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#16171a;--ink:#ecebe6;--muted:#a3a19a;--rule:#33353a;--accent:#ff5c6e;--card:#1e2024;--code:#26282d;}}
:root[data-theme="dark"]{--bg:#16171a;--ink:#ecebe6;--muted:#a3a19a;--rule:#33353a;--accent:#ff5c6e;--card:#1e2024;--code:#26282d;}
body{background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:16px;line-height:1.5;margin:0;padding:0 20px;padding-block:24px 64px;}
main{max-width:1480px;margin:0 auto;}
h1{font-size:1.7rem;font-weight:600;line-height:1.2;margin:0 0 6px;text-wrap:balance;}
h2{font-size:1.25rem;font-weight:600;margin:48px 0 6px;padding-top:18px;border-top:1px solid var(--rule);text-wrap:balance;}
h3{font-size:1.05rem;font-weight:600;margin:28px 0 4px;}
p{max-width:78ch;margin:6px 0;}
ul{max-width:78ch;padding-left:1.2em;} li{margin:4px 0;}
.lede{color:var(--muted);}
code,.path{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:0.86em;}
code{background:var(--code);padding:1px 5px;border-radius:3px;}
table{border-collapse:collapse;margin:14px 0 10px;font-size:0.93rem;}
th,td{text-align:left;padding:6px 14px 6px 0;border-bottom:1px solid var(--rule);vertical-align:top;}
th{font-weight:500;color:var(--muted);}
td.num{font-variant-numeric:tabular-nums;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:0.9em;white-space:nowrap;}
.wrap{overflow-x:auto;}
nav{display:flex;flex-wrap:wrap;gap:6px 18px;margin:14px 0 0;font-size:0.93rem;}
nav a{color:var(--accent);text-decoration:none;}
nav a:hover,nav a:focus{text-decoration:underline;}
figure{margin:22px 0 0;display:flex;flex-direction:column;gap:8px;}
figcaption{max-width:90ch;font-size:0.95rem;}
.eyebrow{display:inline-block;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:0.78em;letter-spacing:0.04em;text-transform:uppercase;color:var(--accent);margin-right:10px;}
.path{display:block;color:var(--muted);margin-top:2px;}
figure img{max-width:100%;height:auto;display:block;background:var(--card);border:1px solid var(--rule);}
a:focus-visible{outline:2px solid var(--accent);outline-offset:2px;}
</style>'''
FONTS = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">'


def jpg(src_rel, w=1800, q=88):
    src = os.path.join(HERE, "figs", src_rel + ".png")
    dst_name = src_rel.replace("/", "__") + ".jpg"
    dst = os.path.join(MEDIA, dst_name)
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
        im = Image.open(src).convert("RGB")
        if im.width > w:
            im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
        im.save(dst, "JPEG", quality=q, optimize=True)
    return "media/alg1_filmstrips/" + dst_name


def fig_block(src_rel, suffix, extra_cap=""):
    line, head, cap = STAGE[suffix]
    rel = jpg(src_rel)
    return f'''
<figure>
  <figcaption><span class="eyebrow">{html.escape(line)}</span><strong>{html.escape(head)}</strong> {html.escape(cap)} {html.escape(extra_cap)}
  <span class="path">figs/{html.escape(src_rel)}.png · clean/</span></figcaption>
  <a href="{rel}" target="_blank" rel="noopener"><img src="{rel}" alt="{html.escape(head)}" loading="lazy"></a>
</figure>'''


def build(page):
    parts = [f'<title>{html.escape(page["title_tag"])}</title>', FONTS, CSS, "<main>", f'<h1>{html.escape(page["h1"])}</h1>', page["lede"], "<nav>"]
    for sec in page["sections"]:
        parts.append(f'<a href="#{sec["id"]}">{html.escape(sec["nav"])}</a>')
    parts.append("</nav>")
    parts.append(page.get("intro", ""))
    for sec in page["sections"]:
        parts.append(f'<h2 id="{sec["id"]}">{html.escape(sec["title"])}</h2>{sec.get("body", "")}')
        for item in sec["figs"]:
            src, suffix = item[0], item[1]
            parts.append(fig_block(src, suffix, item[2] if len(item) > 2 else ""))
    parts.append(page.get("outro", ""))
    parts.append("</main>")
    out = os.path.join(DOCS, page["file"])
    open(out, "w").write("\n".join(parts))
    print("->", out)


# ----------------------------------------------------------------------------- page 1: lean onset
FILMSTRIPS = dict(
    file="20260921-alg1_filmstrips.html", title_tag="Algorithm 1 filmstrips",
    h1="One CEM planning iteration on the braced lean: filmstrips for Algorithm 1",
    lede='''<p class="lede">Talk assets, 2026-09-21. Every figure is rendered from what the CEM planner held at one plan tick of <code>lean_bench</code>
(strategy 25, 0.985 m slab, seed 1, deploy gains, 33 plans/s, N = 20, k = 6, 1 s horizon at 10 ms, 3 knots, zero-order hold):
the centre spline and its rollout, the sampled Σ and ε(k), all 20 candidate rollouts with J(k), the elite set, and the folded spline rolled out from the same x₀.
Full-resolution PNG and PDF sit next to the source: <code>mujoco_mpc/studies/alg1_filmstrips/figs/&lt;run&gt;/</code>, with a <code>clean/</code> copy of each figure without title text for slides.
Method and regeneration: <code>studies/alg1_filmstrips/README.md</code>. These ticks are the lean onset (t = 12.5–13.5 s, 0.5–1.5 s into the lean rung), the transition the lean-onset falls come from;
the companion page <code>20260921-alg1_brace_contact.html</code> does the same at the brace-contact moment.</p>''',
    intro='''
<div class="wrap"><table>
<tr><th>run</th><th>std_min</th><th>outcome by 30 s</th><th>tick shown</th><th>J(θ̄)</th><th>samples best / worst</th><th>fold</th><th>pelvis spread mean / max</th><th>hand spread mean / max</th></tr>
<tr><td><code>cem_stdmin05_s1</code></td><td class="num">0.05 rad</td><td>lean 12.58 s, reach 25.02 s, no fall</td><td class="num">13.50 s</td><td class="num">186.0</td><td class="num">192.4 / 340.1</td><td class="num">180.7</td><td class="num">103 / 217 mm</td><td class="num">126–136 / 251–285 mm</td></tr>
<tr><td><code>cem_s1</code> (deployed)</td><td class="num">0.01 rad</td><td>lean 12.00 s, reach 24.01 s, release 27.10 s, no fall</td><td class="num">12.51 s</td><td class="num">87.0</td><td class="num">78.7 / 123.1</td><td class="num">81.5</td><td class="num">25 / 54 mm</td><td class="num">36–42 / 89–98 mm</td></tr>
<tr><td><code>cem_stdmin03_s1</code></td><td class="num">0.03 rad</td><td>backward fall at 14.74 s (lean onset)</td><td class="num">13.50 s</td><td class="num">452.3</td><td class="num">430.1 / 531.9</td><td class="num">443.9</td><td class="num">61 / 130 mm</td><td class="num">67–71 / 125–161 mm</td></tr>
</table></div>
<p>Spread is the distance at h = 1 s between each candidate's body position and θ̄'s rollout, over the 20 candidates.
Two things the pictures show that the tables did not: the third knot sits at h = 1.0 s and under the hold acts on the last 10 ms only, so nothing selects on it and it random-walks under the refit (largest entry of Σ at every tick; the spline panels let it clip);
and every rollout's running cost is sawtoothed for the first 0.5 s and smooth after the second knot, which is not investigated here.</p>''',
    sections=[
        dict(id="cem_stdmin05_s1", nav="σ floor 50 mrad, t = 13.50 s", title="σ floor 50 mrad, t = 13.50 s (primary set of this page)",
             body='''<p>std_min 0.05 rad, five times the deployed floor and the top of the CEM basin that still completes ≥ 4/6 at 33 plans/s.
Chosen because the fan is wide enough to read: the 20 rollouts end 103 mm apart at the pelvis on average (217 max), 126–136 mm at the hands (251–285 max).
J(θ̄) 186.0; samples 192.4–340.1; fold 180.7, below every single sample. Σ used 50–185 mrad.</p>''',
             figs=[("cem_stdmin05_s1/t13.5_" + s, s) for s in FULL]),
        dict(id="cem_s1", nav="σ floor 10 mrad (deployed), t = 12.51 s", title="σ floor 10 mrad (as deployed), t = 12.51 s",
             body='''<p>The robot's own setting, 0.5 s into the lean rung. Pelvis spread 25 mm mean / 54 max, hands 36–42 / 89–98.
J(θ̄) 87.0; samples 78.7–123.1; fold 81.5. Σ used 10–127 mrad (hip pitch and torso at knots 0–1 sit at 20–60; the 127 is the h = 1 s knot).</p>''',
             figs=[("cem_s1/t12.5_" + s, s) for s in FULL]),
        dict(id="extra", nav="σ comparison and the collapsed fan", title="σ comparison and the collapsed fan", body="",
             figs=[("compare/t12.5_F_sigma_compare", "F_sigma_compare", "The 30 mrad run is already in its lean-onset backward fall: every rollout predicts it, and the plant goes over at 14.74 s."),
                   ("cem_stdmin03_s1/t13.5_C3_fan_strip", "C3_fan_strip", "σ floor 30 mrad, t = 13.50 s: all 20 rollouts topple backward within the horizon; no candidate recovers. The plant fell 1.2 s later."),
                   ("cem_stdmin03_s1/t13.5_C1_rollouts_fan", "C1_rollouts_fan", "The same tick, scored: J spans 430–532 and the sort no longer separates recoveries from falls, because there are none.")]),
    ])

# ----------------------------------------------------------------------------- page 2: brace contact
CONTACT = dict(
    file="20260921-alg1_brace_contact.html", title_tag="Brace-contact discovery",
    h1="Brace-contact discovery inside one CEM iteration: the forearm pad landing, rollout by rollout",
    lede='''<p class="lede">Talk assets, 2026-09-21. Same method as <code>20260921-alg1_filmstrips.html</code>, at the moment that matters: the plan tick at which some of the 20 sampled rollouts
land the forearm pad on the slab and others do not, with the deployed planner settings (CEM, std_min 0.01 rad, k = 6, 33 plans/s, deploy gains, strategy 25).
Ticks were found by dumping every 0.25 s from 19 to 29 s in four runs (<code>run_contact_dumps.sh</code>) and scoring each dump with <code>contact_stats.py</code>:
for every rollout, the gap between the lowest point of <code>left_forearm_pad</code> and the table face along the horizon, landed = within 3 mm.
Full-resolution PNG and PDF: <code>mujoco_mpc/studies/alg1_filmstrips/figs/&lt;run&gt;/</code>, <code>clean/</code> for slides; method and commands in <code>studies/alg1_filmstrips/README.md</code>.</p>''',
    intro='''
<div class="wrap"><table>
<tr><th>run</th><th>tick</th><th>pad gap at x₀</th><th>landers / 20</th><th>elite landers / 6</th><th>θ̄</th><th>fold</th><th>J(θ̄)</th><th>samples best / worst</th><th>J fold</th><th>plant first loads the pad</th></tr>
<tr><td><code>cem_stdmin01_s10_dense</code></td><td class="num">24.27 s</td><td class="num">+28 mm</td><td class="num">4</td><td class="num">3</td><td>hovers, +17 mm at h = 1</td><td>hovers, +6 mm</td><td class="num">1603.3</td><td class="num">1540.2 / 1707.1</td><td class="num">1603.2</td><td class="num">24.9 s (122 N peak)</td></tr>
<tr><td><code>cem_stdmin01_s10_dense</code></td><td class="num">24.51 s</td><td class="num">+26 mm</td><td class="num">20</td><td class="num">6</td><td>lands at h = 0.48 s</td><td>lands at h = 0.46 s</td><td class="num">980.8</td><td class="num">878.9 / 3349.0</td><td class="num">892.0</td><td class="num">24.9 s</td></tr>
<tr><td><code>cem_stdmin01_s9_dense</code></td><td class="num">24.51 s</td><td class="num">+15 mm</td><td class="num">9</td><td class="num">5</td><td>hovers, +3 mm at h = 1</td><td>lands at h = 0.61 s</td><td class="num">1222.1</td><td class="num">1100.8 / 1300.6</td><td class="num">1168.2</td><td class="num">24.88 s (119 N peak)</td></tr>
<tr><td><code>cem_stdmin01_s1_dense</code></td><td class="num">24.00 s</td><td class="num">+26 mm</td><td class="num">19</td><td class="num">6</td><td>lands at h = 0.86 s</td><td>lands at h = 0.79 s</td><td class="num">1139.9</td><td class="num">983.1 / 1247.1</td><td class="num">1036.7</td><td class="num">24.52 s (111 N peak)</td></tr>
<tr><td><code>cem_stdmin02_s0_dense</code> (std_min 0.02)</td><td class="num">23.01 s</td><td class="num">+43 mm</td><td class="num">20</td><td class="num">6</td><td>lands at h = 0.58 s</td><td>lands at h = 0.60 s</td><td class="num">530.3</td><td class="num">549.2 / 776.8</td><td class="num">554.2</td><td class="num">23.64 s (139 N peak)</td></tr>
</table></div>
<p>In every deployed-setting run the pad is held 23–43 mm above the slab through rung 1 with 0 of 20 rollouts landing it; the landing appears in the rollouts within one or two plan ticks of the rung-2 transition at 24.0 s
(rung 2 lowers the brace target: <code>target_distance_tolerance</code> 0.07 m, <code>target_ramp_sec</code> 3 s), and the plant loads the pad 0.4–0.9 s after that. The std_min 0.02 run found the contact inside rung 1, 1.5 s before the schedule asked for it.</p>''',
    sections=[
        dict(id="s10", nav="seed 10, t = 24.27 s: four find it", title="Seed 10, t = 24.27 s: four rollouts land the pad, and they are the three best",
             body='''<p>Pad at +28 mm. The three lowest-J candidates (k = 14, 4, 11; J 1540–1576) are exactly three of the four that land the pad inside the horizon; every hoverer scores 1603–1606,
and the fourth lander (k = 12) is the worst at 1707. θ̄ rises and then descends to +17 mm; the elite mean of three landers and three hoverers descends to +6 mm, halfway.
Rollout spread at h = 1 s: pelvis 23 / 62 mm, hands 30 / 64. One tick later (24.51 s, below) all 20 land by h ≈ 0.5 s; the plant loads the pad at 24.9 s.</p>''',
             figs=[("cem_stdmin01_s10_dense/t24.27_G1_contact", "G1_contact"), ("cem_stdmin01_s10_dense/t24.27_G2_contact_strip", "G2_contact_strip")] +
                  [("cem_stdmin01_s10_dense/t24.27_" + s, s) for s in FULL] +
                  [("cem_stdmin01_s10_dense/t24.51_G1_contact", "G1_contact", "The next dumped tick, 24.51 s: 20 of 20 land, θ̄ at h = 0.48 s, the fold at 0.46 s; J 878.9–3349.0 (the 3349 is a rollout that lands and then loses balance)."),
                   ("cem_stdmin01_s10_dense/t24.51_G2_contact_strip", "G2_contact_strip", "24.51 s.")]),
        dict(id="s9", nav="seed 9, t = 24.51 s: the fold lands it", title="Seed 9, t = 24.51 s: nine land, five of the six elites are landers, the centre hovers and the fold lands",
             body='''<p>Pad at +15 mm. θ̄'s own rollout bottoms out at +3 mm and never touches; 9 of 20 samples land (J 1100.8–1289.5, against 1182.2–1300.6 for the 11 hoverers), 5 of the 6 elites are landers,
and the folded plan lands the pad at h = 0.61 s. J(fold) 1168.2 against J(θ̄) 1222.1. Rollout spread at h = 1 s: pelvis 14 / 39 mm, hands 17–28 / 44–73. The plant loads the pad at 24.88 s.</p>''',
             figs=[("cem_stdmin01_s9_dense/t24.5_G1_contact", "G1_contact"), ("cem_stdmin01_s9_dense/t24.5_G2_contact_strip", "G2_contact_strip")] +
                  [("cem_stdmin01_s9_dense/t24.5_" + s, s) for s in FULL]),
        dict(id="extra", nav="seed 1 at the transition; std_min 0.02 finds it early", title="Seed 1 at the rung transition, and the std_min 0.02 run finding the contact early", body="",
             figs=[("cem_stdmin01_s1_dense/t24.00_G1_contact", "G1_contact", "Seed 1, t = 24.00 s, the tick on the rung-2 transition: 19 of 20 land at h ≈ 0.8 s; the one hoverer is the worst (J 1247 against 983–1223); the fold lands 70 ms before the centre."),
                   ("cem_stdmin02_s0_dense/t23.01_G1_contact", "G1_contact", "std_min 0.02, seed 0, t = 23.01 s, still in rung 1 with the pad at +43 mm: all 20 rollouts land it at h ≈ 0.6 s and the plant loads it at 23.64 s, 1.5 s before the deployed floor's runs."),
                   ("cem_stdmin02_s0_dense/t23.01_G2_contact_strip", "G2_contact_strip", "The same tick as a strip.")]),
    ],
    outro='''
<h2 id="notes">What the dumps say about the bench itself, and what to measure next</h2>
<p>Looking for the contact tick exposed a property of the schedule on this plant. In the planner-ablation runs on the deploy plant at 33 plans/s (<code>studies/planner_ablation/runs/gains_spp15</code>),
shipped CEM (std_min 0.01) is scored complete in 10 of 12 seeds, but the forearm carries load in only 3 of those 10 (seeds 1, 9, 10: 117–147 N). The other seven lean to ~20°, hold the pad 10–30 mm above the slab
and walk through the rungs on 12.0 s intervals. The rung's contact gate accepts a believed pad gap under <code>brace_contact_zmargin</code> = 80 mm (<code>lean.cc</code>, "believed pad gap"), so a hovering pad passes the 2 s verify and the schedule advances unbraced.
At std_min 0.02–0.05 every completing seed braces (157–365 N); PS-hold 10/12, MPPI 11/12, iCEM 7/12.</p>
<ul>
<li>Re-score the ablation page with a brace criterion next to completion: <code>f_forearm</code> &gt; 20 N sustained for 1 s inside rungs 1–2 (the column is in every <code>*.csv</code>; <code>analyze.py</code> reads them). Completion and brace are different counts at the deployed floor.</li>
<li>On the robot logs, check whether the measured brace force was loaded before the reach rung fired (the wrist/forearm F/T channel against the phase index). If the real robot also advances on a believed gap, the 80 mm margin is doing on hardware what it does here.</li>
<li>Whether the schedule should ask for the contact in rung 1: the std_min 0.02 run lands it 1.5 s earlier without the rung-2 target. Rerun <code>run_contact_dumps.sh "0.02:0 0.02:1 0.02:2"</code> and count the first-landing tick per seed.</li>
</ul>''')

PAGES = {"filmstrips": FILMSTRIPS, "contact": CONTACT}

if __name__ == "__main__":
    which = sys.argv[1:] or list(PAGES)
    for w in which:
        build(PAGES[w])
    print("media:", len(os.listdir(MEDIA)), "files,", sum(os.path.getsize(os.path.join(MEDIA, f)) for f in os.listdir(MEDIA)) // 1024, "KB")
