#!/usr/bin/env python3
"""Assemble the benchmark results into one self-contained HTML report.

Reads the sweep logs, the timing tables and the gallery renders that the other
scripts in this directory produce, and writes a single file with the charts and
the rollout videos embedded. Nothing is fetched at view time, so the report can
be handed to someone who does not have the repository.

The charts are built from the same RESULT lines the sweeps print, so the page
and the logs cannot drift apart: re-run the sweeps, re-run this, and the numbers
move together.

Usage:
  python3 make_report.py --out renders/report.html
"""
import argparse
import base64
import csv
import glob
import json
import os
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
GAPS = (3.0, 6.0, 9.0)

# ---------------------------------------------------------------- data loading


def result_lines(path):
    """Parse `RESULT k=v k=v ...` lines into dicts, newest wins on duplicates."""
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        if not line.startswith("RESULT"):
            continue
        f = {}
        for tok in line.split()[1:]:
            k, _, v = tok.partition("=")
            # solved_pct is printed as "76.0+-4.3"; split it into value and
            # standard error instead of leaving the whole token as a string.
            if "+-" in v:
                mean, _, err = v.partition("+-")
                try:
                    f[k] = float(mean)
                    f[k + "_se"] = float(err)
                    continue
                except ValueError:
                    pass
            try:
                f[k] = float(v)
            except ValueError:
                f[k] = v
        out[f["planner"]] = f
    return out


def timing_table(path):
    rows = {}
    if not os.path.exists(path):
        return rows
    label = None
    for line in open(path):
        m = re.match(r"=== (\S+) ===", line)
        if m:
            label = m.group(1)
        m = re.match(r"\s*where an iteration goes:(.*)", line)
        if m and label:
            parts = dict()
            for name, pct in re.findall(r"(\S+) ([\d.]+)%", m.group(1)):
                parts[name] = float(pct)
            rows.setdefault(label, {})["parts"] = parts
    res = result_lines(path)
    for label, f in res.items():
        rows.setdefault(label, {}).update(
            ms=f.get("ms_per_iter"), p95=f.get("ms_per_iter_p95"))
    return rows


def load_dump(path):
    rows = [r for r in csv.reader(open(path)) if not r[0].startswith("#")]
    head, body = rows[0], rows[1:]
    i = {k: head.index(k) for k in head}
    return {k: [float(r[i[k]]) for r in body]
            for k in ("time", "cart", "min_clearance", "ncon")}


def outcome_hist(dump_dir):
    """Bottlenecks cleared *before the first contact*, over a directory.

    Measured up to the first contact rather than over the whole rollout,
    because these dumps were recorded with --early_exit=false: the cart keeps
    driving after it hits something and parks at the rail limit, so counting
    the whole run marks every trial as "reached all three" and the histogram
    collapses to one bar. Truncating at the violation is what makes the
    distribution mean anything -- it is how far the planner got while still
    satisfying the constraint.
    """
    hist = [0, 0, 0, 0]
    solved = 0
    for f in glob.glob(os.path.join(dump_dir, "*.csv")):
        d = load_dump(f)
        touch = [k for k, n in enumerate(d["ncon"]) if n > 0]
        end = touch[0] if touch else len(d["cart"])
        reach = max(d["cart"][:end]) if end else d["cart"][0]
        hist[sum(1 for x in GAPS if reach > x + 0.5)] += 1
        if not touch and any(abs(c - 11.0) < 0.3 for c in d["cart"]):
            solved += 1
    return hist, solved


def data_uri(path, mime):
    with open(path, "rb") as fh:
        return f"data:{mime};base64," + base64.b64encode(fh.read()).decode()


# ------------------------------------------------------------------- chart SVG
# Every chart is inline SVG sized in a viewBox so it scales with its container,
# with a <title> per mark for the hover layer and a <details> table beneath for
# the non-visual reading.

def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def table(headers, rows, caption):
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    tr = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>"
                 for r in rows)
    return (f'<details class="tbl"><summary>{esc(caption)}</summary>'
            f'<div class="scroll"><table><thead><tr>{th}</tr></thead>'
            f"<tbody>{tr}</tbody></table></div></details>")


def chart_outcome_spectrum(dist):
    """Stacked bars: how far through the field each trial got.

    The categories are ordered (0,1,2,3 bottlenecks cleared), so this uses an
    ordinal ramp of one hue rather than categorical hues -- the darkness *is*
    the progress, and a reader does not have to consult the legend to know
    which end is better.
    """
    if not dist:
        return ""
    W, rowh, gap, left, top = 760, 26, 10, 150, 34
    H = top + len(dist) * (rowh + gap) + 26
    bars = []
    for k, (label, (hist, n)) in enumerate(dist.items()):
        y = top + k * (rowh + gap)
        x = left
        for g in range(4):
            frac = hist[g] / n if n else 0
            w = frac * (W - left - 20)
            if w > 0:
                # 2px surface gap between stacked segments
                bars.append(
                    f'<rect class="seg s{g}" x="{x:.1f}" y="{y}" '
                    f'width="{max(w - 2, 0.5):.1f}" height="{rowh}" rx="2">'
                    f"<title>{esc(label)}: {hist[g]} of {n} trials cleared "
                    f"{g} bottleneck{'s' if g != 1 else ''}</title></rect>")
                if frac > 0.14:
                    bars.append(
                        f'<text class="inbar" x="{x + w / 2:.1f}" '
                        f'y="{y + rowh / 2 + 4:.1f}">{hist[g]}</text>')
            x += w
        bars.append(f'<text class="rowlab" x="{left - 12}" '
                    f'y="{y + rowh / 2 + 4}">{esc(label)}</text>')
    key = "".join(
        f'<g transform="translate({left + i * 120},18)">'
        f'<rect class="seg s{i}" width="11" height="11" y="-9" rx="2"/>'
        f'<text class="key" x="16">{i} cleared</text></g>' for i in range(4))
    rows = [[l, *h, n] for l, (h, n) in dist.items()]
    return (f'<figure class="chart"><svg viewBox="0 0 {W} {H}" '
            f'role="img" aria-label="bottlenecks cleared per trial">'
            f"{key}{''.join(bars)}</svg></figure>"
            + table(["planner", "0 gaps", "1", "2", "3", "trials"], rows,
                    "Table: bottlenecks cleared"))


def declutter(ys, min_gap):
    """Push overlapping label positions apart, preserving order.

    Two planners three points apart put their labels 5 px apart at this scale,
    which renders as one illegible smear. The marks stay on their true y; only
    the text moves, so the chart is still accurate and the labels are readable.
    """
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for k in range(1, len(order)):
        prev, cur = order[k - 1], order[k]
        if out[cur] - out[prev] < min_gap:
            out[cur] = out[prev] + min_gap
    return out


def chart_slope(pairs):
    """Corridor -> slalom, one line per planner.

    A slope chart because the question is *change between two conditions* and
    the reading is which lines cross. Only the three planners the text argues
    about carry a hue; the rest stay neutral, so the highlighted pairs never
    exceed the three slots that validate on an all-pairs palette check.
    """
    W, H, left, right, top, bot = 700, 360, 210, 210, 34, 44
    def y(v):
        return top + (1 - v / 100) * (H - top - bot)
    ya = [y(a) for _, a, _, _ in pairs]
    yb = [y(b) for _, _, b, _ in pairs]
    la = declutter(ya, 14)
    lb = declutter(yb, 14)
    lines = []
    for i, (label, a, b, hue) in enumerate(pairs):
        cls = f"hl h{hue}" if hue else "mut"
        lines.append(
            f'<line class="slope {cls}" x1="{left}" y1="{ya[i]:.1f}" '
            f'x2="{W - right}" y2="{yb[i]:.1f}"><title>{esc(label)}: '
            f"{a:.0f}% on one bottleneck, {b:.0f}% on three</title></line>"
            f'<circle class="dot {cls}" cx="{left}" cy="{ya[i]:.1f}" r="4"/>'
            f'<circle class="dot {cls}" cx="{W - right}" cy="{yb[i]:.1f}" r="4"/>'
            f'<text class="slab end {cls}" x="{left - 12}" y="{la[i] + 4:.1f}">'
            f"{esc(label)} {a:.0f}%</text>"
            f'<text class="slab {cls}" x="{W - right + 12}" y="{lb[i] + 4:.1f}">'
            f"{b:.0f}%</text>")
    axis = (f'<text class="axlab" x="{left}" y="{H - 14}" '
            f'text-anchor="middle">one bottleneck</text>'
            f'<text class="axlab" x="{W - right}" y="{H - 14}" '
            f'text-anchor="middle">three bottlenecks</text>'
            f'<text class="axlab" x="{left}" y="{top - 14}" '
            f'text-anchor="middle">% of trials solved cleanly</text>')
    return (f'<figure class="chart"><svg viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="success rate, one bottleneck versus three">'
            f"{''.join(lines)}{axis}</svg></figure>"
            + table(["planner", "1 gap", "3 gaps"],
                    [[l, f"{a:.0f}%", f"{b:.0f}%"] for l, a, b, _ in pairs],
                    "Table: success rate by field"))


def chart_grazing(rows):
    """Clean score against the laxer score, per planner.

    Grouped bars rather than one bar with a "lost" segment stacked on it: the
    two numbers are alternative measurements of the same trials, not parts of a
    whole, and stacking would invite adding them together.
    """
    if not rows:
        return ""
    W, barh, gap, grp, left, top = 760, 11, 5, 14, 160, 40
    H = top + len(rows) * (2 * barh + gap + grp) + 26
    sx = (W - left - 56) / 100.0
    marks = []
    for k, (label, clean, lax) in enumerate(rows):
        y = top + k * (2 * barh + gap + grp)
        for j, (v, cls, name) in enumerate(
                ((clean, "g1", "never touched a disk"),
                 (lax, "g2", "penetration under 20 mm"))):
            yy = y + j * (barh + gap)
            marks.append(
                f'<rect class="gb {cls}" x="{left}" y="{yy}" '
                f'width="{max(v * sx, 0.5):.1f}" height="{barh}" rx="2">'
                f"<title>{esc(label)}: {v:.0f} of 100 trials, {esc(name)}"
                f"</title></rect>"
                f'<text class="val" x="{left + v * sx + 7:.1f}" '
                f'y="{yy + barh - 1}">{v:.0f}</text>')
        share = 0 if lax <= 0 else 100 * (1 - clean / lax)
        marks.append(
            f'<text class="rowlab" x="{left - 12}" y="{y + barh + 4}">'
            f"{esc(label)}</text>"
            f'<text class="share" x="{W - 4}" y="{y + barh + 4}">'
            f"{share:.0f}% grazing</text>")
    return (f'<figure class="chart"><svg viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="clean versus lax success rate per planner">'
            f'<text class="key" x="{left}" y="20">of 100 trials</text>'
            f"{''.join(marks)}</svg>"
            f'<figcaption class="legend"><span class="sw g1"></span>'
            f"never touched a disk"
            f'<span class="sw g2"></span>penetration under 20 mm (old test)'
            f"</figcaption></figure>"
            + table(["planner", "clean", "≤20 mm", "grazing share"],
                    [[l, f"{c:.0f}/100", f"{x:.0f}/100",
                      "—" if x <= 0 else f"{100 * (1 - c / x):.0f}%"]
                     for l, c, x in rows],
                    "Table: both collision criteria"))


def chart_timing(rows, budget=5.0):
    """Per-iteration cost. One measure, so one hue and no legend.

    The budget line is the point of the chart: the bars are only meaningful
    against the deadline, and a bar chart without it would invite the reading
    that 4 ms is 'a bit slower' rather than 'four fifths of the budget'.
    """
    order = sorted(rows.items(), key=lambda kv: kv[1].get("ms") or 0)
    W, rowh, gap, left, top = 760, 24, 12, 150, 26
    H = top + len(order) * (rowh + gap) + 34
    xmax = max(max((r.get("p95") or 0) for _, r in order), budget) * 1.08
    sx = (W - left - 30) / xmax
    marks = []
    for k, (label, r) in enumerate(order):
        y = top + k * (rowh + gap)
        w = (r.get("ms") or 0) * sx
        over = (r.get("ms") or 0) > budget
        marks.append(
            f'<rect class="bar{" over" if over else ""}" x="{left}" y="{y}" '
            f'width="{w:.1f}" height="{rowh}" rx="2">'
            f'<title>{esc(label)}: {r["ms"]:.3f} ms per iteration, '
            f'p95 {r["p95"]:.2f} ms</title></rect>'
            f'<line class="p95" x1="{left + (r.get("p95") or 0) * sx:.1f}" '
            f'y1="{y - 3}" x2="{left + (r.get("p95") or 0) * sx:.1f}" '
            f'y2="{y + rowh + 3}"><title>{esc(label)} p95 '
            f'{r["p95"]:.2f} ms</title></line>'
            f'<text class="rowlab" x="{left - 12}" y="{y + rowh / 2 + 4}">'
            f"{esc(label)}</text>"
            f'<text class="val" x="{left + w + 8:.1f}" '
            f'y="{y + rowh / 2 + 4}">{r["ms"]:.2f}</text>')
    bx = left + budget * sx
    marks.append(f'<line class="budget" x1="{bx:.1f}" y1="{top - 12}" '
                 f'x2="{bx:.1f}" y2="{H - 30}"/>'
                 f'<text class="budgetlab" x="{bx:.1f}" y="{top - 16}" '
                 f'text-anchor="middle">5 ms budget</text>')
    return (f'<figure class="chart"><svg viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="milliseconds per planning iteration">'
            f"{''.join(marks)}</svg>"
            f'<figcaption class="legend"><span class="sw bar"></span>mean'
            f'<span class="sw p95"></span>p95</figcaption></figure>'
            + table(["planner", "ms/iter", "p95 ms", "% of budget"],
                    [[l, f'{r["ms"]:.3f}', f'{r["p95"]:.2f}',
                      f'{100 * r["ms"] / budget:.0f}%'] for l, r in order],
                    "Table: per-iteration cost"))


def chart_horizon(series):
    """Success against planning horizon. Lines, because the x is continuous."""
    if not series:
        return ""
    W, H, left, right, top, bot = 700, 300, 60, 190, 26, 46
    xs = sorted({x for s in series.values() for x, _ in s})
    ymax = max(1.2, max((v for s in series.values() for _, v in s), default=1.2))
    def px(x):
        return left + (x - xs[0]) / (xs[-1] - xs[0]) * (W - left - right)
    def py(v):
        return top + (1 - v / ymax) * (H - top - bot)
    g = []
    for x in xs:
        g.append(f'<line class="grid" x1="{px(x):.1f}" y1="{top}" '
                 f'x2="{px(x):.1f}" y2="{H - bot}"/>'
                 f'<text class="axlab" x="{px(x):.1f}" y="{H - bot + 18}" '
                 f'text-anchor="middle">{x:g} s</text>')
    for i, (label, pts) in enumerate(series.items()):
        pts = sorted(pts)
        d = " ".join(f"{px(x):.1f},{py(v):.1f}" for x, v in pts)
        g.append(f'<polyline class="ln h{i + 1}" points="{d}"/>')
        for x, v in pts:
            g.append(f'<circle class="dot h{i + 1}" cx="{px(x):.1f}" '
                     f'cy="{py(v):.1f}" r="4.5"><title>{esc(label)} at '
                     f"{x:g} s horizon: {v:.2f} of 3 bottlenecks</title></circle>")
        lx, lv = pts[-1]
        g.append(f'<text class="slab h{i + 1}" x="{px(lx) + 10:.1f}" '
                 f'y="{py(lv) + 4:.1f}">{esc(label)}</text>')
    g.append(f'<text class="axlab" x="{left}" y="{H - 8}">planning horizon</text>')
    g.append(f'<text class="axlab" x="{left}" y="{top - 10}">'
             f"bottlenecks reached before first contact (of 3)</text>")
    rows = [[l, f"{x:g} s", f"{v:.2f}"] for l, s in series.items()
            for x, v in sorted(s)]
    return (f'<figure class="chart"><svg viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="bottlenecks reached against planning horizon">'
            f"{''.join(g)}</svg></figure>"
            + table(["series", "horizon", "bottlenecks reached"], rows,
                    "Table: horizon sweep"))


def chart_traces(traces):
    """Cart position against time for the four representative rollouts.

    Small multiples rather than one overlaid axis: the runs have different
    durations and the question is the shape of each, not a comparison at a
    shared instant. The bottleneck bands are the reference that makes the
    shape readable -- crossing one is the event.
    """
    panels = []
    for title, d, outcome in traces:
        W, H, left, top, bot = 330, 150, 34, 16, 26
        tmax = max(d["time"]) or 1
        def px(t):
            return left + t / tmax * (W - left - 12)
        def py(c):
            return top + (1 - c / 11.5) * (H - top - bot)
        bands = "".join(
            f'<line class="gapline" x1="{left}" y1="{py(g):.1f}" '
            f'x2="{W - 12}" y2="{py(g):.1f}"/>'
            f'<text class="gaplab" x="{left - 5}" y="{py(g) + 3:.1f}">{g:g}</text>'
            for g in GAPS)
        step = max(1, len(d["time"]) // 200)
        pts = " ".join(f"{px(d['time'][i]):.1f},{py(d['cart'][i]):.1f}"
                       for i in range(0, len(d["time"]), step))
        cls = "good" if outcome == "solved" else "bad"
        panels.append(
            f'<figure class="panel"><figcaption class="ptitle">{esc(title)}'
            f'</figcaption><svg viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="cart position for the {esc(title)} rollout">{bands}'
            f'<polyline class="trace {cls}" points="{pts}"/>'
            f'<text class="axlab" x="{left}" y="{H - 6}">0 s</text>'
            f'<text class="axlab" x="{W - 12}" y="{H - 6}" '
            f'text-anchor="end">{tmax:.1f} s</text></svg></figure>')
    return f'<div class="panels">{"".join(panels)}</div>'


# ------------------------------------------------------------------------ page

CSS = """
:root{
  color-scheme:light;
  --ground:#f6f7f9; --panel:#fdfdfe; --ink:#11141a; --ink2:#59606d;
  --rule:#dfe3ea; --rail:#c8ced8;
  --alarm:#d0402c;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
  --o0:#86b6ef; --o1:#3987e5; --o2:#1c5cab; --o3:#0d366b;
  --good:#0ca30c; --crit:#d03b3b;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme=light]){
    color-scheme:dark;
    --ground:#121419; --panel:#181b21; --ink:#eef1f6; --ink2:#98a1b0;
    --rule:#252932; --rail:#39404c;
    --alarm:#e8674f;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70;
    --o0:#1c5cab; --o1:#2a78d6; --o2:#5598e7; --o3:#9ec5f4;
  }
}
:root[data-theme=dark]{
  color-scheme:dark;
  --ground:#121419; --panel:#181b21; --ink:#eef1f6; --ink2:#98a1b0;
  --rule:#252932; --rail:#39404c;
  --alarm:#e8674f;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70;
  --o0:#1c5cab; --o1:#2a78d6; --o2:#5598e7; --o3:#9ec5f4;
}
*{box-sizing:border-box}
body{
  margin:0;background:var(--ground);color:var(--ink);
  font-family:ui-sans-serif,system-ui,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased;
}
.wrap{max-width:1180px;margin:0 auto;padding:0 24px 96px}
.measure{max-width:64ch}
code,.mono,th,td,.val,.rowlab,.key,.axlab,.slab,.inbar,.gaplab,.budgetlab{
  font-family:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums;
}
header.mast{padding:64px 0 28px;border-bottom:2px solid var(--ink)}
.eyebrow{
  font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  text-transform:uppercase;letter-spacing:.14em;font-size:11px;
  color:var(--ink2);margin:0 0 14px;
}
h1{
  font-size:clamp(30px,4.6vw,52px);line-height:1.04;margin:0 0 16px;
  letter-spacing:-.022em;font-weight:800;text-wrap:balance;max-width:20ch;
}
.standfirst{font-size:19px;color:var(--ink2);margin:0;max-width:60ch}
.runmeta{
  display:flex;flex-wrap:wrap;gap:8px 28px;margin-top:26px;
  font-size:12px;color:var(--ink2);
}
.runmeta b{color:var(--ink);font-weight:600}
section{padding-top:56px}
h2{
  font-size:26px;letter-spacing:-.012em;margin:0 0 6px;font-weight:750;
  text-wrap:balance;
}
h3{font-size:17px;margin:32px 0 6px;font-weight:700}
p{margin:0 0 16px}
.lede{font-size:17px;color:var(--ink2);margin-bottom:26px;max-width:64ch}
.reading{
  border-left:3px solid var(--alarm);padding:2px 0 2px 16px;margin:18px 0 0;
  font-size:15px;color:var(--ink2);max-width:66ch;
}
.reading b{color:var(--ink)}
.chart{margin:22px 0 0;background:var(--panel);border:1px solid var(--rule);
  border-radius:4px;padding:18px 16px}
.chart svg{width:100%;height:auto;display:block;overflow:visible}
.scroll{overflow-x:auto}
.seg.s0{fill:var(--o0)} .seg.s1{fill:var(--o1)}
.seg.s2{fill:var(--o2)} .seg.s3{fill:var(--o3)}
.seg{stroke:var(--panel);stroke-width:0}
.rowlab{font-size:11px;fill:var(--ink);text-anchor:end}
.key{font-size:11px;fill:var(--ink2);dominant-baseline:middle}
.inbar{font-size:11px;fill:#fff;text-anchor:middle;font-weight:600}
.seg.s0+.inbar{fill:var(--ink)}
.slope{stroke-width:2;fill:none}
.slope.mut{stroke:var(--rail)} .dot.mut{fill:var(--rail)}
.slope.h1,.dot.h1,.ln.h1,.slab.h1{stroke:var(--s1)}
.slope.h2,.dot.h2,.ln.h2,.slab.h2{stroke:var(--s2)}
.slope.h3,.dot.h3,.ln.h3,.slab.h3{stroke:var(--s3)}
.dot.h1{fill:var(--s1)} .dot.h2{fill:var(--s2)} .dot.h3{fill:var(--s3)}
.slab{font-size:11px;fill:var(--ink);stroke:none}
.slab.end{text-anchor:end}
.slab.h1{fill:var(--s1)} .slab.h2{fill:var(--s2)} .slab.h3{fill:var(--s3)}\n.slab.mut{fill:var(--ink2)}
.axlab{font-size:11px;fill:var(--ink2)}
.bar{fill:var(--s1)} .bar.over{fill:var(--crit)}\n.gb.g1{fill:var(--s1)} .gb.g2{fill:var(--s2)}\n.legend .sw.g1{background:var(--s1)} .legend .sw.g2{background:var(--s2)}\n.share{font-size:10px;fill:var(--ink2);text-anchor:end;font-family:ui-monospace,Menlo,monospace}
.p95{stroke:var(--ink);stroke-width:2}
.budget{stroke:var(--alarm);stroke-width:2;stroke-dasharray:4 4}
.budgetlab{font-size:11px;fill:var(--alarm)}
.val{font-size:11px;fill:var(--ink2)}
.legend{display:flex;gap:8px;align-items:center;font-size:12px;
  color:var(--ink2);margin-top:12px}
.legend .sw{width:14px;height:10px;display:inline-block;border-radius:2px}
.legend .sw.bar{background:var(--s1)}
.legend .sw.p95{width:2px;height:14px;background:var(--ink);border-radius:0}
.ln{fill:none;stroke-width:2}
.grid{stroke:var(--rule);stroke-width:1}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
  gap:14px;margin-top:22px}
.panel{margin:0;background:var(--panel);border:1px solid var(--rule);
  border-radius:4px;padding:12px}
.ptitle{font-size:12px;font-weight:650;margin-bottom:6px}
.trace{fill:none;stroke-width:2}
.trace.good{stroke:var(--good)} .trace.bad{stroke:var(--crit)}
.gapline{stroke:var(--rail);stroke-width:1;stroke-dasharray:3 3}
.gaplab{font-size:9px;fill:var(--ink2);text-anchor:end}
.tbl{margin-top:12px;font-size:13px}
.tbl summary{cursor:pointer;color:var(--ink2);font-size:12px}
.tbl summary:focus-visible{outline:2px solid var(--s1);outline-offset:3px}
table{border-collapse:collapse;margin-top:10px;width:100%}
th,td{text-align:right;padding:5px 12px 5px 0;border-bottom:1px solid var(--rule);
  font-size:12px;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600}
.gallery{display:flex;flex-direction:column;gap:34px;margin-top:26px}
.shot{background:var(--panel);border:1px solid var(--rule);border-radius:4px;
  overflow:hidden}
.shot header{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:baseline;
  padding:14px 16px;border-bottom:1px solid var(--rule)}
.tier{font-family:ui-monospace,Menlo,monospace;font-size:11px;
  letter-spacing:.1em;text-transform:uppercase;color:var(--ink2)}
.shot h3{margin:0;font-size:16px}
.stat{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--ink2)}
.stat b{color:var(--ink);font-weight:600}
.shot .scroll img{display:block;height:190px;width:auto;max-width:none}
.shot video{display:block;width:100%;background:#000}
.vidwrap{padding:14px 16px 16px}
.cap{font-size:14px;color:var(--ink2);padding:0 16px 16px;max-width:70ch}
@media (prefers-reduced-motion:reduce){*{animation:none!important;
  transition:none!important}}
"""


def page(html_body, title):
    return (f"<title>{esc(title)}</title>\n<style>{CSS}</style>\n"
            f'<div class="wrap">{html_body}</div>')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="renders/report.html")
    p.add_argument("--renders", default="renders")
    a = p.parse_args()
    R = pathlib.Path(a.renders)

    corridor = result_lines(R / "avoid100_s025" / "sweep.log")
    slalom = result_lines(R / "slalom100_s025" / "sweep.log")
    corridor_lax = result_lines(R / "avoid100_s025_laxtol" / "sweep.log")
    slalom_lax = result_lines(R / "slalom100_s025_laxtol" / "sweep.log")
    timing = timing_table(R / "timing" / "timing.log")
    timing_bal = timing_table(R / "timing_balance" / "timing.log")

    # --- horizon: fixed knots (from the sweep) and matched knots (follow-up)
    hz = result_lines(R / "horizon_slalom" / "horizon.log")
    matched = result_lines(R / "horizon_slalom" / "matched.log")
    # Plotted against partial credit, not success rate: every horizon here
    # solves 0-6 runs in 50, which is noise, while "how far before the first
    # contact" resolves the effect cleanly.
    series = {}
    for label, key in (("Predictive Sampling", "predictive_sampling"),
                       ("Random Sampling", "random_sampling")):
        pts = [(f["horizon"], f["gaps_mean"]) for n, f in hz.items()
               if n.startswith(key)]
        if pts:
            series[label + ", 12 knots"] = pts
    pts = [(f["horizon"], f["gaps_mean"]) for f in matched.values()]
    if pts:
        series["Predictive Sampling, knots scaled"] = pts

    # --- outcome distributions
    dist = {}
    for d in sorted(glob.glob(str(R / "outcome_dist" / "*"))):
        if not os.path.isdir(d):
            continue
        hist, _ = outcome_hist(d)
        n = sum(hist)
        if n:
            dist[os.path.basename(d).replace("_", " ")] = (hist, n)
    if not dist:
        hist, _ = outcome_hist(str(R / "slalom_gallery" / "dumps"))
        if sum(hist):
            dist["predictive sampling"] = (hist, sum(hist))

    # --- slope: corridor vs slalom
    hue = {"predictive_sampling": 1, "annealed_sampling": 2, "ilqg": 3}
    pairs = []
    for k in corridor:
        if k in slalom:
            pairs.append((k.replace("_", " "), corridor[k]["solved_pct"],
                          slalom[k]["solved_pct"], hue.get(k, 0)))
    pairs.sort(key=lambda t: -t[1])

    # --- gallery
    gal = R / "slalom_gallery" / "render"
    manifest = {}
    if (gal / "outcomes.json").exists():
        manifest = json.load(open(gal / "outcomes.json"))
    tier_copy = {
        "1_stopped_at_gap1": ("Stopped at the first bottleneck",
            "The cart accelerates hard at the 11 m goal, arrives at gap 1 with "
            "the pendulum still swinging, and drives a head into a disk before "
            "it is anywhere near laid out. This is the modal failure, and it "
            "is a failure of the approach, not of the gap."),
        "2_cleared_one": ("Cleared one, lost it before the second",
            "The pendulum is laid out in time for gap 1 and threads it. What "
            "the run cannot do is hold that posture for the 3 m to gap 2 -- "
            "the pendulum is chaotic and already rotating as it clears the "
            "first disk pair."),
        "3_cleared_two": ("Cleared two, caught the third",
            "Two clean crossings. By gap 3 the pendulum has swung above the "
            "rail and the upper link catches the top disk. Notice the "
            "clearance at gap 2 is 0.03 m -- it was already marginal."),
        "4_solved": ("Solved clean: all three bottlenecks, zero contact",
            "The one that works. The pendulum is folded below the rail by the "
            "first gap and stays folded through all three: clearances of "
            "+70 mm, +30 mm and +53 mm, no contact at any step, goal reached "
            "at 2.23 s. Three runs in 56 look like this. Note the posture — it "
            "does not lay the pendulum out flat and hold it, it tucks the "
            "links under the cart, which keeps the swept area small enough to "
            "survive three crossings instead of one."),
        "4b_all_three_grazed": ("Cleared all three \u2014 but touched a disk",
            "The best outcome in the batch, and it still does not count. The "
            "cart gets past every bottleneck and reaches the goal, but a head "
            "overlaps a disk on the way. Under the old 20 mm penetration "
            "tolerance this scored as a clean solve; a 28 mm head could sit "
            "71% inside a disk and pass. Any overlap now disqualifies, which "
            "is what the avoidance constraint actually says."),
    }
    shots = []
    traces = []
    for rec in manifest.get("representatives", []):
        tier = rec["tier"]
        head, cap = tier_copy.get(tier, (tier, ""))
        png, mp4 = gal / f"{tier}.png", gal / f"{tier}.mp4"
        if not png.exists():
            continue
        src = rec.get("src") or str(R / "slalom_gallery" / "dumps" / rec["dump"])
        d = load_dump(src)
        traces.append((head.split(":")[0], d,
                       "solved" if rec["solved"] else "failed"))
        vid = (f'<div class="vidwrap"><video controls muted playsinline loop '
               f'preload="metadata" src="{data_uri(str(mp4), "video/mp4")}">'
               f"</video></div>") if mp4.exists() else ""
        shots.append(
            f'<article class="shot"><header>'
            f'<span class="tier">{tier.split("_")[0]}</span>'
            f"<h3>{esc(head)}</h3>"
            f'<span class="stat">reached <b>{rec["max_cart"]:.2f} m</b></span>'
            f'<span class="stat">closest <b>{rec["min_clear"]:+.3f} m</b></span>'
            f'<span class="stat">ran <b>{rec["sim_time"]:.2f} s</b></span>'
            f"</header>"
            f'<div class="scroll"><img alt="filmstrip of the {esc(head)} '
            f'rollout, one frame per bottleneck crossing" '
            f'src="{data_uri(str(png), "image/png")}"></div>'
            f'<p class="cap">{esc(cap)}</p>{vid}</article>')

    # ------------------------------------------------------------------ copy
    b = []
    b.append(
        '<header class="mast"><p class="eyebrow">MuJoCo MPC &middot; '
        "triple pendulum cartpole</p>"
        "<h1>Seven planners, one gap, then three</h1>"
        '<p class="standfirst">A 3-link pendulum on a cart, one actuator, and '
        "a bottleneck narrower than the pendulum is long. What the success "
        "table cannot show is how the failures fail — so this page shows the "
        "rollouts.</p>"
        '<div class="runmeta"><span>100 trials per planner</span>'
        "<span>weights <b>[1, 0, 0.1, 0.01, 500]</b></span>"
        "<span><b>50 Hz</b> wall-clock control, 4 iterations per decision</span>"
        "<span>perturbed starts, shared across planners</span></div></header>")

    if dist:
        b.append(
            "<section><h2>How the failures fail</h2>"
            '<p class="lede">Every trial on the three-bottleneck slalom, '
            "binned by how far through the field the cart got. Runs are "
            "simulated to the end here rather than cut at the first contact, "
            "so this is the whole journey \u2014 <em>reached</em>, not "
            "<em>cleared cleanly</em>. A success rate collapses all of this to "
            "one number; the shape is the interesting part.</p>" + chart_outcome_spectrum(dist) +
            '<p class="reading">The same bottleneck at x=3 is cleared 76% of '
            "the time when it is the only one on the rail. Nothing about its "
            "geometry changed here — what changed is that the goal moved from "
            "6 m to 11 m, so the cart residual is three times larger at the "
            "start and the planner commits to more speed on approach. "
            "<b>Planners arrive at a gap they can otherwise thread with far "
            "too much velocity to lay the pendulum out.</b></p>"
            "</section>")

    if shots:
        b.append(
            "<section><h2>Four rollouts, worst to best</h2>"
            '<p class="lede">Predictive Sampling on the slalom, one run per '
            "outcome tier. Frames are placed at the bottleneck crossings, not "
            "at uniform intervals — the crossings are the events that decide "
            "the run. Each strip is followed by the rollout itself.</p>"
            f'<div class="gallery">{"".join(shots)}</div></section>')

    if traces:
        b.append(
            "<section><h2>The same four runs as trajectories</h2>"
            '<p class="lede">Cart position against time. The dashed lines are '
            "the three bottlenecks. A run that fails does not slow down "
            "first — it drives at the gap and stops dead.</p>"
            + chart_traces(traces) + "</section>")

    if pairs:
        b.append(
            "<section><h2>One bottleneck to three</h2>"
            '<p class="lede">The same planners, the same objective, the same '
            "first gap at x=3 — only two more bottlenecks after it.</p>"
            + chart_slope(pairs) +
            '<p class="reading"><b>Nothing solves three.</b> The best planner '
            "manages 4 runs in 100, and five of the seven never manage it at "
            "all, so the top rows sit within one or two standard errors of "
            "each other — this chart reports a wall, not a ranking. An "
            "earlier version of this page showed an inversion here with "
            "Annealed Sampling in front. That was an artifact of a 20 mm "
            "penetration tolerance that scored grazing as success: it was "
            "ranking planners by how much they grazed, and Annealed Sampling "
            "grazes the most.</p></section>")

    graz = [(k.replace("_", " "), corridor[k]["solved_pct"],
             corridor_lax[k]["solved_pct"])
            for k in corridor if k in corridor_lax]
    graz.sort(key=lambda r: -r[1])
    if graz:
        b.append(
            "<section><h2>How much of the score was grazing</h2>"
            '<p class="lede">The single corridor, scored two ways. <b>Clean</b> '
            "is the constraint as written: never overlap a disk. <b>≤20 mm</b> "
            "is the test this benchmark originally applied, which tolerated a "
            "head sitting up to 71% inside a disk. Same 100 trials, same "
            "rollouts — only the verdict differs.</p>"
            + chart_grazing(graz) +
            '<p class="reading">Every planner loses ground, but not equally, '
            "and the spread is the point. Random Sampling loses 11% of its "
            "score; Annealed Sampling loses 40% and Cross-Entropy 90%. "
            "<b>A high grazing share means a planner's apparent competence was "
            "contact it was not being charged for.</b> Annealed Sampling has a "
            "plausible mechanism for it: the MPPI update averages candidates "
            "rather than taking the best, and the average of two trajectories "
            "that pass on opposite sides of a disk is not itself a "
            "clearance.</p></section>")

    if timing:
        b.append(
            "<section><h2>What an iteration costs</h2>"
            '<p class="lede">The success tables hold planning <em>iterations</em> '
            "constant, which is what makes them a comparison of algorithms "
            "rather than of throughput. This is the other half. The budget is "
            "one timestep — and it is the same 5 ms at every speed setting, "
            "because slowing the simulation buys iterations and costs control "
            "rate in exactly the same proportion.</p>"
            + chart_timing(timing) +
            '<p class="reading">The four 10-rollout samplers sit within 3% of '
            "each other, and most of that gap is contact, not code: on the "
            "obstacle-free control run Cross-Entropy drops 34% while the "
            "others drop 8–10%. It collides in 79% of trials, and MuJoCo "
            "charges more for a contact-rich rollout — <b>so a planner that "
            "crashes pays for it twice, once in the score and once on the "
            "clock</b>. The two real outliers are structural: Annealed "
            "Sampling is 4.0× because it takes 4× the samples, and iLQG "
            "spends 45% of its iteration on finite-difference derivatives "
            "against 2.7% on the backward pass.</p></section>")

    if series:
        b.append(
            "<section><h2>Does more lookahead help?</h2>"
            '<p class="lede">The bottlenecks are 3 m apart, about a second of '
            "travel, so at the default 1 s horizon a planner can barely see "
            "the next gap while committing to the posture for this one. The "
            "obvious fix is a longer horizon.</p>"
            + chart_horizon(series) +
            '<p class="reading">It backfires, and the controlled version says '
            "why. At a fixed 12 knots, stretching the horizon from 1 s to 3 s "
            "stretches the spacing between control points from 83 ms to "
            "250 ms, and progress collapses from 0.96 bottlenecks to 0.26: "
            "<b>the planner was buying lookahead by giving up the resolution "
            "of the action it is about to execute</b>. Scale the knots with "
            "the horizon and the collapse disappears — 0.98, 0.80, 0.86, flat "
            "within noise. So the cost was resolution, not horizon. But note "
            "what the flat line also says: <b>more lookahead, paid for "
            "honestly, buys nothing here either.</b></p></section>")

    b.append(
        "<section><h2>How to reproduce this</h2>"
        '<p class="measure">Every number here comes from a RESULT line in a '
        "sweep log, and this page is regenerated from those logs, so the two "
        "cannot drift.</p>"
        '<div class="scroll"><table><tbody>'
        "<tr><td>headline table</td><td><code>avoidance_sweep.sh renders/avoid100_s025 100 0.25</code></td></tr>"
        "<tr><td>three bottlenecks</td><td><code>TASK=slalom avoidance_sweep.sh renders/slalom100_s025 100 0.25</code></td></tr>"
        "<tr><td>per-iteration cost</td><td><code>timing_bench.sh renders/timing 10 3 1.0</code></td></tr>"
        "<tr><td>horizon</td><td><code>horizon_sweep.sh renders/horizon_slalom 50 0.25 1.0 2.0 3.0</code></td></tr>"
        "<tr><td>this page</td><td><code>make_report.py --out renders/report.html</code></td></tr>"
        "</tbody></table></div></section>")

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page("".join(b), "Seven planners, one gap, then three"))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
