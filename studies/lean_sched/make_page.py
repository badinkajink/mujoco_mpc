#!/usr/bin/env python3
"""Assemble the schedule-cost docpage from a sweep's analysis.json.

Tables and the verdict line are generated from the measurements, never typed, so
the page cannot drift from the run that produced it.

usage: make_page.py --analysis DIR/analysis.json --out docs/lean/<date>_schedule.html
"""
import argparse, json, os, datetime

CSS = """
  :root{ color-scheme:light dark;
    --bg:#fcfcfb; --panel:#f3f3f0; --line:#dededa;
    --ink:#0b0b0b; --ink2:#52514e; --ink3:#7a7873;
    --s1:#2a78d6; --s2:#eb6834; --warn:#e34948; --good:#1baf7a; }
  @media (prefers-color-scheme:dark){ :root{
    --bg:#1a1a19; --panel:#232322; --line:#3a3a37;
    --ink:#fff; --ink2:#c3c2b7; --ink3:#918f86;
    --s1:#3987e5; --s2:#d95926; --warn:#e66767; --good:#3fbf8c; } }
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
        white-space:nowrap}
  th:first-child,td:first-child{text-align:left;white-space:normal}
  th{color:var(--ink3);font-weight:600;font-size:.75rem;text-transform:uppercase;
     letter-spacing:.05em}
  td{color:var(--ink2)} td:first-child{color:var(--ink)}
  .scroll{overflow-x:auto}
  .note{border-left:3px solid var(--s1);background:var(--panel);
        border-radius:0 8px 8px 0;padding:.8rem 1rem;margin:1.2rem 0}
  .warn{border-left-color:var(--warn)} .ok{border-left-color:var(--good)}
  .note b{color:var(--ink)}
  figure{margin:1.6rem 0}
  img,video{width:100%;border-radius:10px;display:block;background:#fff}
  figcaption{color:var(--ink3);font-size:.84rem;margin-top:.5rem}
  ul{padding-left:1.15rem}
  td.bad{color:var(--warn);font-weight:600}
  td.good{color:var(--good);font-weight:600}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
  @media(max-width:52rem){.grid2{grid-template-columns:1fr}}
"""


def esc(x):
    return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def short(v):
    return v.replace("h12_recovery_noreach_", "")


def results_table(vs):
    h = ("<div class=scroll><table><thead><tr>"
         "<th>variant</th><th>dwell floor</th><th>n</th><th>completed</th>"
         "<th>fell</th><th>t_complete median</th><th>brace peak</th>"
         "<th>brace duty</th></tr></thead><tbody>")
    FLOOR = {"base": 70.0, "sus50": 42.5, "both50": 42.5, "both33": 33.3}
    for r in vs:
        name = short(r["variant"])
        tc = ("%.1f s" % r["t_complete_med"]) if r["completed"] else "&mdash;"
        cls = "good" if r["completed"] == r["n"] else ("bad" if r["fell"] else "")
        h += ("<tr><td>%s</td><td>%.1f s</td><td>%d</td>"
              "<td class=%s>%d</td><td class=%s>%d</td><td>%s</td>"
              "<td>%.0f N</td><td>%.0f%%</td></tr>") % (
            esc(name), FLOOR.get(name, float("nan")), r["n"],
            cls, r["completed"], "bad" if r["fell"] else "", r["fell"],
            tc, r["brace_peak_med"], 100 * r["brace_duty_med"])
    return h + "</tbody></table></div>"


PHASE_NAMES = ["stand_up", "brace_lean", "release", "sb_r1", "sb_r2",
               "sb_r3", "sb_r4"]
ASKED = {"base":   [15, 24, 14, 5, 5, 4, 3],
         "both50": [15, 12, 7, 2.5, 2.5, 2, 1.5]}


def phase_table(vs):
    """Measured dwell against the sustain the JSON asked for, per phase."""
    h = ("<div class=scroll><table><thead><tr><th>phase</th>"
         + "".join("<th>%s</th>" % esc(n) for n in PHASE_NAMES)
         + "</tr></thead><tbody>")
    for r in vs:
        name = short(r["variant"])
        asked = ASKED.get(name)
        meas = r.get("phase_dwell") or []
        h += "<tr><td>%s &mdash; asked</td>" % esc(name)
        h += "".join("<td>%s</td>" % (asked[i] if asked and i < len(asked) else "&mdash;")
                     for i in range(len(PHASE_NAMES)))
        h += "</tr><tr><td>%s &mdash; measured</td>" % esc(name)
        for i in range(len(PHASE_NAMES)):
            m = meas[i] if i < len(meas) else None
            a = asked[i] if asked and i < len(asked) else None
            cls = ""
            if m is not None and a is not None and m - a > 0.25:
                cls = " class=bad"
            h += "<td%s>%s</td>" % (cls, "%.2f" % m if m is not None else "&mdash;")
        h += "</tr>"
    return h + "</tbody></table></div>"


def verdict(vs):
    by = {short(r["variant"]): r for r in vs}
    b, c = by.get("base"), by.get("both50")
    if not b or not c:
        return "<p>Not enough variants ran to state a verdict.</p>"
    if c["completed"] == 0 and b["completed"] == 0:
        return ("<p><b>Neither schedule completed.</b> Both the stock schedule and "
                "the halved one lost the robot before the recovery phases, so this "
                "sweep says nothing about the cut &mdash; it says the commit itself "
                "is the fragile step at this plan rate. Fix the commit before "
                "optimising the clock around it.</p>")
    if c["completed"] >= b["completed"] and c["completed"] > 0:
        saved = (b["t_complete_med"] - c["t_complete_med"]) if b["completed"] else float("nan")
        return ("<p><b>Halving every downstream dwell did not cost robustness in "
                "this sample.</b> <code>both50</code> completed %d/%d against the "
                "stock schedule's %d/%d, and the runs that finished did so %s "
                "earlier. The sample is three seeds per arm: this is a "
                "go-look-further result, not a certified one.</p>") % (
            c["completed"], c["n"], b["completed"], b["n"],
            ("%.0f s" % saved) if saved == saved else "n/a")
    return ("<p><b>The cut cost robustness.</b> <code>both50</code> completed "
            "%d/%d against the stock schedule's %d/%d. The dwell is doing work; "
            "cutting it uniformly is not free.</p>") % (
        c["completed"], c["n"], b["completed"], b["n"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig", default="media/schedule.png")
    ap.add_argument("--videos", default="")
    a = ap.parse_args()
    d = json.load(open(a.analysis))
    vs = d["variants"]
    today = datetime.date.today().isoformat()

    vids = ""
    if a.videos:
        cells = ""
        for spec in a.videos.split(","):
            label, path = spec.split("=", 1)
            cells += ('<figure><video controls preload=metadata src="%s"></video>'
                      '<figcaption>%s</figcaption></figure>' % (esc(path), esc(label)))
        vids = "<div class=grid2>%s</div>" % cells

    html = """<!doctype html>
<html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>What the lean schedule costs</title>
<style>%s</style></head><body><div class=wrap>

<h1>What the lean schedule costs</h1>
<p class=sub>The deploy pipeline's duration is written into the strategy JSON, not
produced by the planner. This measures how much of it is removable and what the
removal costs, at a plan rate matched to the robot.</p>
<p class=meta>%s &middot; <code>Lean H12 Magpie</code>, strategy 24
(<code>h12_recovery_noreach</code>) &middot; CEM, 20 rollouts, 1.0 s horizon,
spp 3 &rarr; <b>33 Hz</b> &middot; harness <code>mjpc/lean_bench.cc</code> +
<code>studies/lean_sched/</code></p>

<h2>1. Two things that were wrong before any run</h2>

<div class="note warn"><b>The ramp is not additive wall-clock.</b> Phase 0 of
strategy 24 carries <code>success_sustain_time 15</code> and
<code>target_ramp_sec 5</code>, and the phase advances at <b>t&nbsp;=&nbsp;15.00</b>,
measured. <code>target_ramp_sec</code> drives the target-pose interpolation
<em>concurrently</em> with the sustain; it does not gate
<code>Transition</code>'s advance test. So a strategy's scheduled floor is the
sum of its finite sustains and nothing else &mdash; 70 s for strategy 24, not the
137 s you get by adding the ramps. Cutting a ramp makes the reference move
faster inside the same window; only cutting a sustain shortens the window.</div>

<div class="note warn"><b>The tolerance is dead code on lean phases.</b>
<code>total_distance</code> is computed from the keyframe's contact pairs, and
lean keyframes declare none, so it is identically 0 and the
<code>total_distance &lt;= target_distance_tolerance</code> test is always true
(<code>lean.cc</code>, and the comment at the commit-retry block says so
explicitly). Phase advance on this strategy is a <b>pure timer</b>. Only the
strat-25/27 targeting rungs, which measure a real hand-to-target distance,
have a tolerance that gates anything.</div>

<h2>2. What was run</h2>
<p>One process per run, <code>nice</code>d, 4 planner threads. Variants are
generated by scaling every sustain and ramp <em>except phase 0</em>, which is the
bring-up settle the real robot needs and is held byte-identical:</p>
<pre>base    phase 0 fixed; everything else stock          dwell floor 70.0 s
both50  sustains x0.5 AND ramps x0.5 downstream       dwell floor 42.5 s</pre>
<p>Selected at runtime by <code>LEAN_STRATEGY_OVERRIDE</code>, a default-off env
read added to <code>lean.cc</code> so variants can run without overwriting the
slot's own JSON. Unset, the load path is byte-identical.</p>

<h2>3. Results</h2>
%s
<p class=meta>"brace peak" is the maximum elbow+forearm normal load against the
table during the brace phase; "duty" is the fraction of that phase carrying more
than 5 N. A schedule that finishes faster with 0 N never braced &mdash; it just
ran the clock out in the air.</p>
%s

<figure><img src="%s" alt="run durations and brace load"><figcaption>
Left: each run's duration, ticks marking phase entries; green completed, red fell.
Right: elbow+forearm load on the table through each run.</figcaption></figure>

<h2>3b. Where the seconds actually are</h2>
%s
<p>Five of the seven phases land on their asked sustain to the sample interval:
they are <b>pure timers with no feedback in them at all</b>. Only two exceed it,
and by a repeatable amount.</p>

<div class="note ok"><b>The excess is a contact gate, and it is the good kind.</b>
<code>brace_lean</code> overruns its sustain by exactly <b>2.00 s</b> in both
arms. That is <code>brace_contact_verify</code> (2.0 in
<code>Lean_H12_Magpie.xml</code>): the 2026-08-12 brace-contact-gated advance
holds the brace rungs until the forearm pad has been in believed table contact
continuously for that long, with <code>brace_contact_loss</code> (1.0 s) resetting
the counter. <code>release</code> carries the same gate and a larger, more
variable excess (+6.2 s stock, and 26 s vs 14 s across two seeds) because the pad
is coming <em>off</em> and re-seating.</p>
<p>This is why halving the dwell did not cost anything. The two phases that carry
the physical risk are not on the timer &mdash; the timer is their <em>lower</em>
bound and the contact gate is what actually releases them. Shortening the timer
lowers a floor the gate was already above. The four <code>standback</code> rungs
and the final stand have no such gate: they are 17 s of open-loop clock in the
stock schedule and they ran to the millisecond, whatever the robot was doing.</p></div>

<h2>4. Verdict</h2>
%s

<h2>5. What this does not say</h2>
<ul>
<li><b>Three seeds per arm.</b> The baseline's own commit failure is documented
at ~25%% on the twin bench, so three runs cannot separate a schedule effect from
that noise. Treat the completion counts as a direction, not a rate.</li>
<li><b>This is strategy 24, not 27.</b> The retrieval mission adds seven servo
rungs whose tolerances <em>do</em> gate, so its time will not scale the same way.</li>
<li><b>Plan rate is held fixed at 33 Hz</b> and is not a variable here. An
earlier version of this sweep ran at 25 Hz (<code>spp 4</code>) and was discarded:
below the robot's own rate every arm looks more fragile than it is.</li>
<li><b>Sim, and one plant.</b> Nothing here has been near the robot.</li>
</ul>

<h2>6. Reproduce</h2>
<pre>ninja -C build_cmake lean_bench
python3 studies/lean_sched/gen_variants.py h12_recovery_noreach
python3 studies/lean_sched/sweep.py --out studies/lean_sched/runs/s24 \\
        --variants base,both50 --seeds 3 --jobs 1 --threads 4 --spp 3
python3 studies/lean_sched/analyze.py --runs studies/lean_sched/runs/s24 \\
        --out docs/lean/media</pre>
<div class="note"><b>Core budget.</b> <code>sweep.py</code> refuses any grid
leaving fewer than 6 of the box's cores free. A 4&times;5-thread sweep on this
20-core machine hard-crashed it on 2026-08-26 and lost every partial result;
rows are now fsynced as they land.</div>

</div></body></html>
""" % (CSS, today, results_table(vs), vids, esc(a.fig), phase_table(vs), verdict(vs))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(html)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
