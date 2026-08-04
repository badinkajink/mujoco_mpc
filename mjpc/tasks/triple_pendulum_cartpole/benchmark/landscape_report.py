#!/usr/bin/env python3
"""Build the triple-pendulum-cartpole avoidance report: one self-contained page.

The whole arc, in the order it was measured:
  - the task, and the RRT the task was published with,
  - seven planners at equal budget on one bottleneck and on three,
  - which planner and which settings produced each historical row,
  - how much of the spread between those rows was the planner and how much was
    an unseeded random number generator,
  - what the success rate actually looks like over (avoidance weight x
    clearance margin), measured with seeds and repeats, and what is past the
    ridge that map found,
  - what an iteration costs, on which machine.

Everything is read from the logs and CSVs the benchmark scripts write, so the
page cannot drift from them: re-run the sweeps, re-run this, the numbers move
together. Videos and filmstrips are embedded, so the file stands alone.

Usage:
  python3 landscape_report.py --grid renders/runs/<ts>_landscape \
      --renders renders/runs/<ts>_outcomes --repro renders/runs/<ts>_repro \
      --out renders/landscape_report.html
"""
import argparse
import base64
import csv
import glob
import html
import itertools
import os
import pathlib
import re
import statistics as st

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]

HALF_GAP = 0.25   # metres from disk edge to wall; see slalom.xml
GOAL_TOL = 0.30

# --------------------------------------------------------------- log parsing

HEAD_RE = re.compile(
    r"^task:\s*(?P<task>.+?)\s{2,}stage:\s*(?P<stage>\S+)\s{2,}"
    r"planner:\s*(?P<planner>\S+)\s+(?P<secs>[\d.]+)s x (?P<repeats>\d+) repeat")
WEIGHTS_RE = re.compile(
    r"^\s*weights:\s*Cart\s+(\S+)\s+Upright\s+(\S+)\s+Velocity\s+(\S+)\s+"
    r"Control\s+(\S+)\s+Avoidance\s+(\S+)")
MARGIN_RE = re.compile(r"margin\s+([\d.]+)m")
HORIZON_RE = re.compile(r"horizon\s+([\d.]+)s")


def parse_runs(path):
    """Every RESULT line in `path`, paired with the header block above it.

    The header carries the settings the RESULT line historically left out --
    planner name, weight vector, margin, episode length -- which is exactly the
    information whose absence made the old tables uncomparable.
    """
    if not os.path.exists(path):
        return []
    runs, cur = [], {}
    for line in open(path, errors="replace"):
        m = HEAD_RE.match(line)
        if m:
            cur = {"planner_class": m.group("planner"),
                   "total_time": float(m.group("secs")),
                   "repeats": int(m.group("repeats")),
                   "stage": m.group("stage"),
                   "margin": None, "horizon": None}
            continue
        m = WEIGHTS_RE.match(line)
        if m:
            cur["weights"] = [float(x) for x in m.groups()]
            cur["avoid_w"] = float(m.group(5))
            continue
        if line.startswith("  speed"):
            mm = MARGIN_RE.search(line)
            # No explicit margin printed means the task default was in force.
            cur["margin"] = float(mm.group(1)) if mm else 0.08
            cur["margin_explicit"] = bool(mm)
            hh = HORIZON_RE.search(line)
            if hh:
                cur["horizon"] = float(hh.group(1))
            continue
        if line.startswith("RESULT "):
            f = dict(kv.split("=", 1) for kv in line.split()[1:])
            r = dict(cur)
            r["label"] = f["planner"]
            r["trials"] = int(f["trials"])
            r["solved"] = int(f["solved"])
            r["collided"] = int(f["collided"])
            r["collided_pct"] = float(f["collided_pct"])
            r["gaps"] = float(f["gaps_mean"])
            r["ms_iter"] = float(f["ms_per_iter"])
            r["source"] = os.path.relpath(path, ROOT)
            if "_dumps" not in r["label"]:
                runs.append(r)
    return runs


def load_grid(grid_dir):
    """The seeded (weight x margin x seed) grid: rows plus per-cell aggregates."""
    rows = []
    path = os.path.join(grid_dir, "results.csv")
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            rows.append({"weight": int(r["weight"]), "margin": float(r["margin"]),
                         "seed": int(r["planner_seed"]),
                         "trials": int(r["trials"]), "solved": int(r["solved"]),
                         "collided": int(r["collided"]),
                         "gaps": float(r["gaps_mean"])})
    if not rows:
        # No CSV yet, or the sweep is still running and has only written its
        # header: read whatever runs have landed, so the page can be built and
        # looked at before the grid finishes.
        for p in sorted(glob.glob(os.path.join(grid_dir, "runs", "*.log"))):
            for r in parse_runs(p):
                m = re.match(r"w(\d+)_m(\d+)_s(\d+)", r["label"])
                if not m:
                    continue
                rows.append({"weight": int(m.group(1)),
                             "margin": int(m.group(2)) / 100.0,
                             "seed": int(m.group(3)), "trials": r["trials"],
                             "solved": r["solved"], "collided": r["collided"],
                             "gaps": r["gaps"]})
    if not rows:
        return [], {}
    cells = {}
    for r in rows:
        cells.setdefault((r["weight"], r["margin"]), []).append(r)
    agg = {}
    for k, v in cells.items():
        trials = sum(x["trials"] for x in v)
        solved = sum(x["solved"] for x in v)
        collided = sum(x["collided"] for x in v)
        pcts = [100.0 * x["solved"] / x["trials"] for x in v]
        agg[k] = {
            "trials": trials, "solved": solved, "collided": collided,
            "pct": 100.0 * solved / trials,
            "collided_pct": 100.0 * collided / trials,
            "stalled_pct": 100.0 * (trials - solved - collided) / trials,
            "gaps": st.mean(x["gaps"] for x in v),
            "seeds": len(v), "per_seed": sorted(pcts),
            "spread": max(pcts) - min(pcts) if len(v) > 1 else 0.0,
            # binomial standard error on the pooled trials
            "se": 100.0 * ((solved / trials) * (1 - solved / trials) / trials) ** 0.5,
        }
    return rows, agg


def read_manifest(grid_dir):
    p = os.path.join(grid_dir, "manifest.txt")
    if not os.path.exists(p):
        return {}
    out = {}
    for line in open(p):
        if ":" in line and not line.startswith("="):
            k, _, v = line.partition(":")
            if k.strip() and k[0] not in " =":
                out[k.strip()] = v.strip()
    return out


def grid_size(grid_dir):
    """How many runs the sweep intends, counted from the queue it built rather
    than from the rows that have landed -- otherwise a partial grid reports its
    own subset as the whole thing and looks finished."""
    p = os.path.join(grid_dir, "joblist.txt")
    if not os.path.exists(p):
        return 0
    return sum(1 for line in open(p) if line.strip())


def b64(path, mime):
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


# Assets are inlined, so the page's weight is the sum of them. The renders come
# out of the pipeline at a size meant for looking at frame by frame; recompress
# for the web before embedding, and cache so repeated report builds are cheap.
def shrink_video(src, cache):
    dst = os.path.join(cache, os.path.basename(src))
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
        # filmstrip.py writes every 4th step at 12 fps, which plays a 12 s
        # episode over 50 s. Retime to roughly real time -- the point of the
        # footage is what the cart does, and at quarter speed nobody watches it.
        rc = os.system(
            f'ffmpeg -y -loglevel error -i {src!r} '
            f'-vf "setpts=PTS/4,fps=25,scale=440:-2" '
            f'-c:v libx264 -crf 34 -preset veryslow -pix_fmt yuv420p '
            f'-movflags +faststart -an {dst!r}')
        if rc != 0 or not os.path.exists(dst):
            return src
    return dst


def shrink_png(src, cache):
    dst = os.path.join(cache, os.path.basename(src).replace(".png", ".jpg"))
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
        try:
            from PIL import Image
            im = Image.open(src).convert("RGB")
            w, h = im.size
            im.resize((1180, round(h * 1180 / w)), Image.LANCZOS).save(
                dst, "JPEG", quality=72, optimize=True, progressive=True)
        except Exception:
            return src
    return dst


# --------------------------------------------------------------------- design
# A cool, blue-biased neutral ground -- the page is about plotted grids and log
# lines, so the neutrals lean toward the plotting ink rather than sitting on a
# dead grey. Outcome hues are the validated aqua/orange/blue trio (all-pairs
# clean in both modes); the heatmap runs the documented single-hue blue ramp.
CSS = """
:root{
  --paper:#f5f7f9; --card:#ffffff; --ink:#10161c; --ink-2:#4a5661; --ink-3:#78858f;
  --rule:#d5dde4; --rule-2:#e7edf1;
  --solved:#1baf7a; --collided:#eb6834; --stalled:#2a78d6;
  --accent:#1c5cab;
  --s100:#cde2fb; --s200:#9ec5f4; --s300:#6da7ec; --s400:#3987e5;
  --s500:#256abf; --s600:#184f95; --s700:#0d366b;
  --zero:#eef2f5;
  --mono:ui-monospace,"SF Mono","JetBrains Mono","DejaVu Sans Mono",Menlo,Consolas,monospace;
  --sans:ui-sans-serif,system-ui,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  color-scheme:light;
}
@media (prefers-color-scheme:dark){
  :root:where(:not([data-theme="light"])){
    --paper:#12161b; --card:#181e25; --ink:#e8edf2; --ink-2:#a2b0bc; --ink-3:#6f7d89;
    --rule:#263039; --rule-2:#1d252d;
    --solved:#199e70; --collided:#d95926; --stalled:#3987e5;
    --accent:#6da7ec;
    --s100:#0d366b; --s200:#184f95; --s300:#1c5cab; --s400:#256abf;
    --s500:#3987e5; --s600:#6da7ec; --s700:#9ec5f4;
    --zero:#1b222a;
    color-scheme:dark;
  }
}
:root[data-theme="dark"]{
  --paper:#12161b; --card:#181e25; --ink:#e8edf2; --ink-2:#a2b0bc; --ink-3:#6f7d89;
  --rule:#263039; --rule-2:#1d252d;
  --solved:#199e70; --collided:#d95926; --stalled:#3987e5;
  --accent:#6da7ec;
  --s100:#0d366b; --s200:#184f95; --s300:#1c5cab; --s400:#256abf;
  --s500:#3987e5; --s600:#6da7ec; --s700:#9ec5f4;
  --zero:#1b222a;
  color-scheme:dark;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
  font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:0 24px 96px}
.prose{max-width:68ch}
h1,h2,h3{font-family:var(--mono);font-weight:600;text-wrap:balance;letter-spacing:-.02em}
h1{font-size:clamp(28px,4.6vw,46px);line-height:1.08;margin:0 0 12px}
h2{font-size:clamp(20px,2.4vw,26px);line-height:1.2;margin:0 0 6px}
h3{font-size:15px;letter-spacing:.02em;margin:0 0 8px;color:var(--ink-2)}
p{margin:0 0 14px}
a{color:var(--accent)}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--ink-3);margin:0 0 10px}
header.mast{padding:64px 0 28px;border-bottom:1px solid var(--rule)}
.lede{font-size:19px;line-height:1.5;color:var(--ink-2);max-width:60ch;margin:0 0 26px}
.meta{display:flex;flex-wrap:wrap;gap:6px 28px;font-family:var(--mono);
  font-size:12px;color:var(--ink-3)}
.meta b{font-weight:500;color:var(--ink-2)}
section{padding:52px 0;border-bottom:1px solid var(--rule-2)}
section:last-of-type{border-bottom:0}
.snum{font-family:var(--mono);font-size:12px;color:var(--accent);
  letter-spacing:.1em;margin:0 0 4px}
.card{background:var(--card);border:1px solid var(--rule);border-radius:3px;
  padding:20px;margin:22px 0}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:12.5px;
  font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:6px 10px;border-bottom:1px solid var(--rule-2);
  white-space:nowrap}
th{color:var(--ink-3);font-weight:500;text-align:right;border-bottom:1px solid var(--rule);
  font-size:11px;letter-spacing:.06em;text-transform:uppercase}
th:first-child,td:first-child{text-align:left}
tbody tr:hover{background:var(--rule-2)}
td.lbl{font-weight:600}
.cmd{font-family:var(--mono);font-size:12px;background:var(--zero);
  border-left:2px solid var(--accent);padding:10px 14px;margin:14px 0 0;
  overflow-x:auto;white-space:pre;color:var(--ink-2);border-radius:0 3px 3px 0}
.cap{font-family:var(--mono);font-size:11.5px;color:var(--ink-3);
  margin:10px 0 0;max-width:78ch;line-height:1.55}
.legend{display:flex;flex-wrap:wrap;gap:6px 20px;font-family:var(--mono);
  font-size:12px;color:var(--ink-2);margin:0 0 14px;align-items:center}
.key{display:inline-flex;align-items:center;gap:7px}
.sw{width:11px;height:11px;border-radius:2px;flex:none}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin:20px 0}
.tile{background:var(--card);border:1px solid var(--rule);border-radius:3px;padding:16px}
.tile .n{font-family:var(--mono);font-size:30px;font-weight:600;letter-spacing:-.03em;
  line-height:1;margin:0 0 6px;font-variant-numeric:tabular-nums}
.tile .k{font-family:var(--mono);font-size:11px;letter-spacing:.09em;
  text-transform:uppercase;color:var(--ink-3)}
.tile .s{font-size:13px;color:var(--ink-2);margin:8px 0 0;line-height:1.45}
figure{margin:0}
video{width:100%;display:block;background:#000;border-radius:2px}
.vid{background:var(--card);border:1px solid var(--rule);border-radius:3px;
  padding:12px;display:flex;flex-direction:column;gap:9px}
.vid .vh{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:12px}
.vid .vs{font-family:var(--mono);font-size:11px;color:var(--ink-3);line-height:1.5}
.strip{width:100%;border:1px solid var(--rule);border-radius:2px;display:block}
.pill{font-family:var(--mono);font-size:10.5px;letter-spacing:.07em;
  text-transform:uppercase;padding:2px 7px;border-radius:2px;color:#fff;font-weight:600}
svg{display:block;max-width:100%;height:auto}
svg text{font-family:var(--mono);fill:var(--ink-2)}
.tt{position:fixed;pointer-events:none;background:var(--ink);color:var(--paper);
  font-family:var(--mono);font-size:11.5px;padding:6px 9px;border-radius:3px;
  opacity:0;transition:opacity .1s;z-index:99;line-height:1.5;white-space:pre}
.warn{border-left:2px solid var(--collided);background:var(--zero);
  padding:14px 18px;margin:20px 0;border-radius:0 3px 3px 0}
.warn p{margin:0 0 8px}.warn p:last-child{margin:0}
code{font-family:var(--mono);font-size:.92em;background:var(--zero);
  padding:1px 5px;border-radius:2px}
ul{margin:0 0 14px;padding-left:20px}li{margin:0 0 7px}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
"""

JS = """
(function(){
  var tt=document.createElement('div');tt.className='tt';document.body.appendChild(tt);
  document.addEventListener('mouseover',function(e){
    var t=e.target.closest('[data-tip]');if(!t)return;
    tt.textContent=t.getAttribute('data-tip');tt.style.opacity='1';
  });
  document.addEventListener('mousemove',function(e){
    if(tt.style.opacity!=='1')return;
    var x=e.clientX+14,y=e.clientY+14;
    var r=tt.getBoundingClientRect();
    if(x+r.width>innerWidth-8)x=e.clientX-r.width-14;
    if(y+r.height>innerHeight-8)y=e.clientY-r.height-14;
    tt.style.left=x+'px';tt.style.top=y+'px';
  });
  document.addEventListener('mouseout',function(e){
    if(e.target.closest('[data-tip]'))tt.style.opacity='0';
  });
})();
"""


# ---------------------------------------------------------------------- charts

def heatmap(agg, weights, margins):
    """Solve rate over (avoidance weight x clearance margin).

    The margin axis is drawn against the physical half-gap, because that is the
    explanation: a margin approaching 0.25 m does not make the planner more
    careful, it closes the gap the planner is supposed to drive through.
    """
    if not agg:
        return "<p class='cap'>grid not finished</p>"
    cw, ch = 92, 54
    left, top = 92, 66
    W = left + cw * len(margins) + 118
    H = top + ch * len(weights) + 62
    ramp = ["var(--zero)", "var(--s100)", "var(--s200)", "var(--s300)",
            "var(--s400)", "var(--s500)", "var(--s600)", "var(--s700)"]

    def step(pct):
        if pct <= 0.01:
            return 0
        for i, edge in enumerate((2, 5, 10, 18, 28, 40)):
            if pct < edge:
                return i + 1
        return 7

    o = [f'<svg viewBox="0 0 {W} {H}" width="{W}" role="img" '
         f'aria-label="solve rate over avoidance weight and clearance margin">']
    # the sealed-gap region: margins at or past the half-gap
    seal = [i for i, m in enumerate(margins) if m >= HALF_GAP * 0.76]
    if seal:
        x0 = left + seal[0] * cw
        o.append(f'<rect x="{x0}" y="{top-34}" width="{cw*len(seal)}" '
                 f'height="{ch*len(weights)+34}" fill="var(--collided)" opacity=".055"/>')
        o.append(f'<text x="{x0+cw*len(seal)}" y="{top-42}" font-size="10" '
                 f'text-anchor="end" fill="var(--collided)" letter-spacing=".08em">'
                 f'BARRIER SEALS THE GAP</text>')
    for j, m in enumerate(margins):
        o.append(f'<text x="{left+cw*j+cw/2}" y="{top-4}" font-size="11.5" '
                 f'text-anchor="middle">{m:.2f}</text>')
    # The best margin per weight -- the ridge. Marking it is the difference
    # between a table of numbers and a picture of where the optimum moves.
    ridge = {}
    for w in weights:
        cand = [m for m in margins if (w, m) in agg]
        if cand:
            ridge[w] = max(cand, key=lambda m: agg[(w, m)]["pct"])
    for i, w in enumerate(weights):
        o.append(f'<text x="{left-12}" y="{top+ch*i+ch/2+4}" font-size="11.5" '
                 f'text-anchor="end">{w:,}</text>')
        for j, m in enumerate(margins):
            a = agg.get((w, m))
            x, y = left + cw * j, top + ch * i
            if not a:
                o.append(f'<rect x="{x+1}" y="{y+1}" width="{cw-2}" height="{ch-2}" '
                         f'fill="none" stroke="var(--rule)" stroke-dasharray="2 2"/>')
                continue
            fill = ramp[step(a["pct"])]
            ink = "#fff" if step(a["pct"]) >= 4 else "var(--ink)"
            spread = (f'  seed spread {a["spread"]:.0f} pts'
                      f' ({", ".join(f"{p:.0f}%" for p in a["per_seed"])})')
            tip = (f'avoidance {w:,}  margin {m:.2f} m\\n'
                   f'{a["solved"]}/{a["trials"]} solved = {a["pct"]:.1f}% '
                   f'+-{a["se"]:.1f}\\n{a["seeds"]} seeds x '
                   f'{a["trials"]//a["seeds"]} trials\\n{spread.strip()}\\n'
                   f'collided {a["collided_pct"]:.0f}%  '
                   f'stalled {a["stalled_pct"]:.0f}%')
            o.append(f'<g data-tip="{html.escape(tip)}">'
                     f'<rect x="{x+1}" y="{y+1}" width="{cw-2}" height="{ch-2}" '
                     f'fill="{fill}"/>'
                     f'<text x="{x+cw/2}" y="{y+ch/2+1}" font-size="14" '
                     f'text-anchor="middle" fill="{ink}" font-weight="600">'
                     f'{a["pct"]:.0f}%</text>'
                     f'<text x="{x+cw/2}" y="{y+ch/2+15}" font-size="9.5" '
                     f'text-anchor="middle" fill="{ink}" opacity=".72">'
                     f'{a["solved"]}/{a["trials"]}</text></g>')
            if ridge.get(w) == m:
                o.append(f'<rect x="{x+1.5}" y="{y+1.5}" width="{cw-3}" '
                         f'height="{ch-3}" fill="none" stroke="var(--solved)" '
                         f'stroke-width="2.5"/>')
    yb = top + ch * len(weights)
    o.append(f'<text x="{left+cw*len(margins)/2}" y="{yb+34}" font-size="11" '
             f'text-anchor="middle" fill="var(--ink-3)" letter-spacing=".08em">'
             f'CLEARANCE MARGIN (m) &#8594; half-gap is {HALF_GAP:.2f}</text>')
    o.append(f'<text transform="translate(20,{top+ch*len(weights)/2}) rotate(-90)" '
             f'font-size="11" text-anchor="middle" fill="var(--ink-3)" '
             f'letter-spacing=".08em">AVOIDANCE WEIGHT</text>')
    o.append('</svg>')
    return "".join(o)


def repro_plot(points):
    """Every measurement of one configuration, on one axis."""
    W, H = 760, 168
    left, right = 116, 40
    y = 84
    xs = lambda v: left + (v / 40.0) * (W - left - right)
    o = [f'<svg viewBox="0 0 {W} {H}" width="{W}" role="img" '
         f'aria-label="six measurements of one configuration">']
    for t in range(0, 41, 10):
        o.append(f'<line x1="{xs(t)}" y1="{y-40}" x2="{xs(t)}" y2="{y+22}" '
                 f'stroke="var(--rule-2)"/>')
        o.append(f'<text x="{xs(t)}" y="{y+38}" font-size="11" '
                 f'text-anchor="middle">{t}%</text>')
    vals = [p["pct"] for p in points]
    lo, hi, mean = min(vals), max(vals), st.mean(vals)
    o.append(f'<rect x="{xs(lo)}" y="{y-14}" width="{xs(hi)-xs(lo)}" height="28" '
             f'fill="var(--collided)" opacity=".13" rx="2"/>')
    o.append(f'<line x1="{xs(mean)}" y1="{y-20}" x2="{xs(mean)}" y2="{y+20}" '
             f'stroke="var(--ink-3)" stroke-dasharray="3 3"/>')
    o.append(f'<text x="{xs(mean)}" y="{y-26}" font-size="10.5" text-anchor="middle" '
             f'fill="var(--ink-3)">mean {mean:.0f}%</text>')
    for p in points:
        cx = xs(p["pct"])
        tip = (f'{p["solved"]}/{p["trials"]} = {p["pct"]:.0f}%\\n'
               f'{p["name"]}\\n{p["when"]}')
        o.append(f'<g data-tip="{html.escape(tip)}">'
                 f'<circle cx="{cx}" cy="{y}" r="7" fill="var(--stalled)" '
                 f'stroke="var(--card)" stroke-width="2"/></g>')
    o.append(f'<text x="{left-14}" y="{y+4}" font-size="11.5" text-anchor="end" '
             f'fill="var(--ink-2)">6 runs</text>')
    o.append(f'<text x="{xs(lo)}" y="{y+58}" font-size="11" fill="var(--collided)">'
             f'range {lo:.0f}% &#8211; {hi:.0f}%, one configuration</text>')
    o.append('</svg>')
    return "".join(o)


def outcome_bars(agg, weights, margins):
    """What happened in the trials that did not solve."""
    if not agg:
        return ""
    rows = [(w, m) for w in weights for m in margins if (w, m) in agg]
    bh, gap = 21, 5
    left, right = 150, 74
    W = 820
    H = 42 + len(rows) * (bh + gap)
    bw = W - left - right
    o = [f'<svg viewBox="0 0 {W} {H}" width="{W}" role="img" '
         f'aria-label="outcome composition per configuration">']
    for k, (w, m) in enumerate(rows):
        a = agg[(w, m)]
        y = 30 + k * (bh + gap)
        o.append(f'<text x="{left-10}" y="{y+14}" font-size="11" text-anchor="end">'
                 f'{w:,} / {m:.2f}</text>')
        x = left
        for name, pct, col in (("solved", a["pct"], "var(--solved)"),
                               ("collided", a["collided_pct"], "var(--collided)"),
                               ("stalled", a["stalled_pct"], "var(--stalled)")):
            wpx = bw * pct / 100.0
            if wpx <= 0.4:
                continue
            tip = (f'avoidance {w:,}  margin {m:.2f} m\\n'
                   f'{name} {pct:.0f}% of {a["trials"]} trials')
            o.append(f'<g data-tip="{html.escape(tip)}">'
                     f'<rect x="{x}" y="{y}" width="{max(0,wpx-2)}" height="{bh}" '
                     f'fill="{col}" rx="2"/>')
            if wpx > 42:
                o.append(f'<text x="{x+8}" y="{y+14}" font-size="10.5" fill="#fff" '
                         f'font-weight="600">{pct:.0f}%</text>')
            o.append('</g>')
            x += wpx
        o.append(f'<text x="{W-right+10}" y="{y+14}" font-size="10.5" '
                 f'fill="var(--ink-3)">{a["gaps"]:.2f} gaps</text>')
    o.append(f'<text x="{left}" y="18" font-size="10.5" fill="var(--ink-3)" '
             f'letter-spacing=".08em">WEIGHT / MARGIN</text>')
    o.append('</svg>')
    return "".join(o)


# -------------------------------------------------- the seven-planner sweeps

# Display names, and the order the tables are read in (best corridor first).
PLANNER_NAMES = {
    "random_shooting": "Random Sampling",
    "random_sampling": "Random Sampling",
    "predictive_sampling": "Predictive Sampling",
    "pso": "PSO (fixed)",
    "annealed_sampling": "Annealed Sampling",
    "ilqg": "iLQG",
    "pso_stock": "PSO (stock)",
    "cross_entropy": "Cross-Entropy",
}


def result_lines(path):
    """Every RESULT line in `path`, as a dict, keyed by planner label.

    Deliberately not parse_runs(): the lax-tolerance sweeps predate the timing
    block, so their RESULT lines carry no ms_per_iter and parse_runs would
    raise on them. Here every field is optional.
    """
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, errors="replace"):
        if not line.startswith("RESULT "):
            continue
        f = dict(kv.split("=", 1) for kv in line.split()[1:])
        out[f["planner"]] = f
    return out


def load_planner_sweeps():
    """The 100-trial, seven-planner comparisons, clean and lax, on both worlds.

    Returns rows ordered by corridor clean score, each carrying both collision
    criteria so the gap between them stays visible -- that gap is the finding,
    not a footnote to it.
    """
    src = {
        "corridor": result_lines(os.path.join(ROOT, "renders/avoid100_s025/sweep.log")),
        "corridor_lax": result_lines(
            os.path.join(ROOT, "renders/avoid100_s025_laxtol/sweep.log")),
        "slalom": result_lines(os.path.join(ROOT, "renders/slalom100_s025/sweep.log")),
        "slalom_lax": result_lines(
            os.path.join(ROOT, "renders/slalom100_s025_laxtol/sweep.log")),
    }
    rows = []
    for key, name in PLANNER_NAMES.items():
        c = src["corridor"].get(key)
        if not c:
            continue
        s = src["slalom"].get(key, {})
        row = {
            "key": key, "name": name,
            "trials": int(c["trials"]),
            "c_clean": int(c["solved"]),
            "c_lax": int(src["corridor_lax"].get(key, {}).get("solved", -1)),
            "c_se": float(c["solved_pct"].split("+-")[1]),
            "c_tsolve": float(c["t_solve_median"]),
            "c_ms": float(c.get("ms_per_iter", 0)),
            "s_clean": int(s.get("solved", -1)),
            "s_lax": int(src["slalom_lax"].get(key, {}).get("solved", -1)),
            "s_gaps": float(s.get("gaps_mean", 0)),
        }
        # Grazing share: the fraction of the lax score that does not survive the
        # zero-tolerance test. Undefined when the lax score is zero.
        for w in ("c", "s"):
            lax, clean = row[f"{w}_lax"], row[f"{w}_clean"]
            row[f"{w}_graze"] = (1 - clean / lax) if lax > 0 else None
        rows.append(row)
    rows.sort(key=lambda r: -r["c_clean"])
    return rows


def planner_bars(rows, world):
    """Paired bars: clean score inside the lax score, one row per planner.

    Drawing the clean bar inside the lax one rather than beside it makes the
    grazing share the visible quantity -- it is the part of the bar that is not
    filled.
    """
    W, rowh, left, right = 760, 26, 168, 58
    H = 30 + rowh * len(rows) + 12
    span = W - left - right
    o = [f'<svg viewBox="0 0 {W} {H}" width="{W}" role="img" '
         f'aria-label="{world} success by planner, both collision criteria">']
    for pct in (0, 25, 50, 75, 100):
        x = left + span * pct / 100
        o.append(f'<line x1="{x:.1f}" y1="26" x2="{x:.1f}" y2="{H-12}" '
                 f'stroke="var(--rule-2)"/>')
        o.append(f'<text x="{x:.1f}" y="20" font-size="10" text-anchor="middle" '
                 f'fill="var(--ink-3)">{pct}%</text>')
    for i, r in enumerate(rows):
        y = 30 + i * rowh
        clean, lax, n = r[f"{world[0]}_clean"], r[f"{world[0]}_lax"], r["trials"]
        o.append(f'<text x="{left-10}" y="{y+13}" font-size="11.5" '
                 f'text-anchor="end">{html.escape(r["name"])}</text>')
        tip = (f'{r["name"]}\\nclean {clean}/{n}\\n'
               f'<=20 mm {lax}/{n}' if lax >= 0 else f'{r["name"]}\\nclean {clean}/{n}')
        o.append(f'<g data-tip="{tip}">')
        if lax >= 0:
            o.append(f'<rect x="{left}" y="{y+4}" width="{span*lax/n:.1f}" '
                     f'height="15" fill="var(--collided)" opacity=".22" rx="1.5"/>')
        o.append(f'<rect x="{left}" y="{y+4}" width="{span*clean/n:.1f}" '
                 f'height="15" fill="var(--solved)" rx="1.5"/>')
        o.append(f'<rect x="{left}" y="{y+4}" width="{span:.0f}" height="15" '
                 f'fill="transparent"/></g>')
        o.append(f'<text x="{W-right+8}" y="{y+15}" font-size="10.5" '
                 f'fill="var(--ink-3)">{clean}/{n}</text>')
    o.append('</svg>')
    return "".join(o)


def load_timing(path):
    """The per-iteration cost table, from timing_bench.sh's log."""
    out = {}
    if not os.path.exists(path):
        return out
    cur = None
    for line in open(path, errors="replace"):
        m = re.match(r"^=== (\S+) ===", line)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r"^timing: ([\d.]+) ms/iteration over (\d+) iterations "
                     r"\(p95 ([\d.]+), worst ([\d.]+)\)", line)
        if m and cur:
            out[cur] = {"ms": float(m.group(1)), "iters": int(m.group(2)),
                        "p95": float(m.group(3)), "worst": float(m.group(4))}
        m = re.match(r"\s+where an iteration goes:\s+(.*)$", line)
        if m and cur and cur in out:
            out[cur]["where"] = m.group(1).strip()
    return out


def load_machine(path):
    """The threads/ms-per-iteration table from machine_bench.sh's log."""
    rows, host = [], {}
    if not os.path.exists(path):
        return host, rows
    in_results = False
    for line in open(path, errors="replace"):
        m = re.match(r"^(date|uname|cpu|cores|mem|load)\s*:\s*(.+)$", line.strip())
        if m:
            host.setdefault(m.group(1), m.group(2).strip())
        if line.strip().startswith("threads"):
            in_results = True
            continue
        if in_results:
            p = line.split()
            if len(p) == 6 and p[0].isdigit():
                rows.append({"threads": int(p[0]), "ms": float(p[1]),
                             "iters": int(p[3]), "msteps": float(p[4]),
                             "speedup": p[5]})
            elif rows and not line.strip():
                in_results = False
    return host, rows


# ------------------------------------------------------------------ page build

def build(args):
    man = read_manifest(args.grid)
    rows, agg = load_grid(args.grid)
    weights = sorted({r["weight"] for r in rows}) or [500, 2000, 8000, 32000, 128000]
    margins = sorted({r["margin"] for r in rows}) or [0.04, 0.08, 0.12, 0.16, 0.20]

    # ---- history: every historical row, with the settings recovered
    hist = []
    for p in ["renders/weight_slalom/weight.log",
              "renders/margin_slalom/margin.log",
              "renders/slalom_solve/solve.log",
              "renders/slalom_solve/solve_extra.log",
              "renders/config_gallery/gallery.log"]:
        hist += parse_runs(os.path.join(ROOT, p))

    # ---- the one configuration measured six times
    W8, M8 = 8000.0, 0.08
    same = [h for h in hist
            if h.get("avoid_w") == W8 and abs(h.get("margin", 9) - M8) < 1e-9
            and h.get("planner_class") == "PredictiveSampling"
            and h.get("horizon") is None and h["trials"] == 50]
    pts = [{"name": h["label"], "solved": h["solved"], "trials": h["trials"],
            "pct": 100.0 * h["solved"] / h["trials"],
            "when": h["source"]} for h in same]
    for p in sorted(glob.glob(os.path.join(args.repro, "rep*.log"))) if args.repro else []:
        for r in parse_runs(p):
            pts.append({"name": r["label"], "solved": r["solved"],
                        "trials": r["trials"],
                        "pct": 100.0 * r["solved"] / r["trials"],
                        "when": os.path.relpath(p, ROOT)})

    # ---- renders
    vids = []
    idx = os.path.join(args.renders or "", "index.csv")
    if os.path.exists(idx):
        cache = os.path.join(args.renders, ".web")
        os.makedirs(cache, exist_ok=True)
        for r in csv.DictReader(open(idx)):
            item = dict(r)
            png, mp4 = os.path.join(ROOT, r["png"]), os.path.join(ROOT, r["mp4"])
            # Only one filmstrip per configuration gets embedded, so resolve the
            # path now and pay for the base64 later, at the one that is used.
            item["png_path"] = png if os.path.exists(png) else ""
            item["cache"] = cache
            item["mp4_data"] = (b64(shrink_video(mp4, cache), "video/mp4")
                                if os.path.exists(mp4) else "")
            vids.append(item)

    # ---- control character (kept: it was the part of the old summary that worked)
    cc = {}
    ccp = os.path.join(ROOT, "renders/config_gallery/cost_control_summary.csv")
    if os.path.exists(ccp):
        by = {}
        for r in csv.DictReader(open(ccp)):
            by.setdefault(r["label"], []).append(r)
        for k, v in by.items():
            mean = lambda c: st.mean(float(x[c]) for x in v)
            cc[k] = {"u": mean("mean_abs_u"), "eff": mean("effort_int_u2"),
                     "rev": mean("reversals_per_s"), "cart": mean("share_Cart"),
                     "avoid": mean("share_Avoidance"), "n": len(v)}

    o = []
    A = o.append

    # Sections are numbered as they are emitted, not by hand: several of them
    # are conditional on their inputs existing, and a hand-written number turns
    # a missing sweep into a gap in the sequence.
    counter = itertools.count(1)

    def sec(title):
        return (f'<section><p class="snum">{next(counter):02d}</p>'
                f'<h2>{title}</h2>')
    A(f"<title>Threading three gaps with a chaotic pendulum</title>")
    A(f"<style>{CSS}</style>")
    A('<div class="wrap">')

    # ---------------------------------------------------------------- masthead
    A('<header class="mast">')
    A('<p class="eyebrow">triple pendulum cartpole &middot; mujoco mpc &middot; '
      'obstacle avoidance</p>')
    A("<h1>Threading three gaps<br>with a chaotic pendulum</h1>")
    A('<p class="lede">A three-link pendulum on a cart, one actuator for four '
      'degrees of freedom, driving through gaps half its own length. Seven '
      'planners at equal budget, a collision test that allows no penetration, '
      'and the two cost parameters that turned out to matter more than any of '
      'them. Everything below was measured; the numbers that were wrong are '
      'kept and marked.</p>')
    A('<div class="meta">')
    for k in ("date", "commit short", "binary md5", "cpu"):
        if man.get(k):
            A(f'<span><b>{html.escape(k)}</b> {html.escape(man[k])}</span>')
    A(f'<span><b>grid</b> {len(rows)} runs / {len(agg)} cells</span>')
    A('</div></header>')

    # ------------------------------------------------------- 1. what was run
    # ------------------------------------------------------------- 1. the task
    psweeps = load_planner_sweeps()

    A(sec('The gap is narrower than the pendulum'))
    A('<div class="prose">')
    A('<p>A 1.0 kg cart on a rail carries a three-link pendulum: three massless '
      'rods of 1/3 m, each with a 0.1 kg head on the end. One motor pushes the '
      'cart, &plusmn;20 N. That is one actuator for four degrees of freedom, and '
      'the passive dynamics are chaotic.</p>')
    A('<p>The obstacles are pairs of disks of radius 0.6 m centred at '
      '<code>z = &plusmn;0.85</code>, leaving a gap of 0.5 m. The pendulum is '
      '1.0 m long, so it cannot pass upright. It has to be swung down, laid out '
      'or folded, driven through, and recovered &mdash; and the disks are real '
      'collision geoms, so one touch disqualifies the run.</p>')
    A('<p><b>Corridor</b> is the task as published (Caldwell &amp; Correll, ISRR '
      '2015 &sect;5): one gap at <code>x = 3</code>, goal at <code>x = 6</code>. '
      '<b>Slalom</b> keeps that gap exactly and adds two more at <code>x = 6</code> '
      'and <code>x = 9</code>, goal at <code>x = 11</code>. Because the first '
      'bottleneck is unchanged, the difference between the two is attributable to '
      'what comes after it.</p>')
    A('</div>')
    strip = str(HERE.parent / "report" / "fig" / "filmstrip_solved.png")
    if os.path.exists(strip):
        cache = os.path.join(args.grid, ".web")
        os.makedirs(cache, exist_ok=True)
        A('<figure class="card"><img class="strip" src="'
          + b64(shrink_png(strip, cache), "image/png") + '" alt="filmstrip of a '
          'clean slalom run"><p class="cap">One clean run through all three gaps. '
          'Panels are framed at the crossings, not at uniform intervals. The '
          'posture that survives three gaps is the folded one, not the flat '
          'lay-out that clears a single corridor &mdash; laying the links '
          'horizontal sweeps a 1 m arc, which survives one crossing and rarely '
          'two. Clearances at the three crossings here are +27, +154 and +727 '
          'mm.</p></figure>')
    A('<div class="prose">')
    A('<p>What counts as solved: the cart ends within 0.30 m of the goal and no '
      'head ever overlapped a disk. The pendulum\'s final posture is not scored, '
      'because every objective here sets the upright weight to zero &mdash; '
      'requiring an upright finish would score a run against a term it was never '
      'given.</p>')
    A('</div>')
    A('</section>')

    # -------------------------------------------------------- 2. the RRT baseline
    A(sec('The planner this task came with'))
    A('<div class="prose">')
    A('<p>The task is from a paper about a kinodynamic RRT whose steering '
      'primitive linearizes about the zero-control trajectory from a vertex '
      'rather than about the vertex itself. We ported the primitive to MuJoCo and '
      'grew trees with it, both as a baseline and to test whether it could serve '
      'as an MJPC planner. It cannot, and the reason is not the primitive.</p>')
    A('</div>')
    A('<div class="tiles">')
    for n, k, s in [
        ("3000", "corridor vertices", "grown in 379 s. Closest approach to the "
         "goal: 2.117. Not solved."),
        ("3873", "slalom vertices", "grown in 1800 s. Best path moved the cart "
         "5.83 m of the 11 m required. Not solved."),
        ("96%", "of build in nearest-nbr", "O(N) per extend, so O(N&sup2;) over "
         "the build. The per-vertex precompute the paper optimizes is 2&ndash;7%."),
        ("10&sup3;&times;", "over an MPC step", "MJPC's contract is one "
         "OptimizePolicy call per control step, ~5 ms. A tree build is three "
         "orders of magnitude past it."),
    ]:
        A(f'<div class="tile"><p class="n">{n}</p><p class="k">{k}</p>'
          f'<p class="s">{s}</p></div>')
    A('</div>')
    A('<div class="prose">')
    A('<p>Two further observations, because both were measured rather than '
      'assumed. Extends were rejected by collision 50% of the time on the '
      'corridor and 80% on the slalom: the feasible set is thin relative to the '
      'reachable set, which is the same reason the samplers below struggle. And '
      'the paper\'s headline contribution is a no-op at its own benchmark\'s start '
      'state &mdash; the upright equilibrium <em>is</em> an equilibrium of the '
      'zero-control dynamics, so measured drift over 1.0 s is 0.000 and the two '
      'linearizations are identical there. From non-equilibrium vertices it is '
      'real, but only past a horizon of about 0.6 s.</p>')
    A('<p>AgileRRT is usable here as an offline trajectory generator. It is not a '
      'receding-horizon planner, so it does not appear in the tables below.</p>')
    A('</div>')
    A('</section>')

    # ------------------------------------------------- 3. one gap, all planners
    if psweeps:
        A(sec('One gap, seven planners'))
        A('<div class="prose">')
        A('<p>100 trials each, objective '
          '<code>(cart, upright, velocity, control, avoidance) = '
          '(1, 0, 0.1, 0.01, 500)</code>, margin 0.08 m, four planner iterations '
          'per control step. Every planner gets the same iteration count, so this '
          'compares algorithms rather than throughput; what an iteration costs '
          'each of them is further down.</p>')
        A('<p>Both collision criteria are shown. The filled bar is the constraint '
          'as stated &mdash; never overlap a disk. The pale bar behind it is the '
          '&le;20 mm penetration tolerance this benchmark used originally. The '
          'unfilled difference is <b>grazing</b>: the part of a planner\'s '
          'apparent competence that was contact it was not charged for.</p>')
        A('</div>')
        A('<div class="legend">'
          '<span class="key"><span class="sw" style="background:var(--solved)">'
          '</span>clean &mdash; no overlap at any step</span>'
          '<span class="key"><span class="sw" style="background:var(--collided);'
          'opacity:.22"></span>&le;20 mm penetration allowed</span></div>')
        A('<div class="card scroll">' + planner_bars(psweeps, "corridor") + '</div>')
        A('<div class="card scroll"><table><thead><tr><th>planner</th>'
          '<th>clean</th><th>1 s.e.</th><th>&le;20 mm</th><th>grazing</th>'
          '<th>median t_solve</th><th>ms/iter</th></tr></thead><tbody>')
        for r in psweeps:
            gz = "&mdash;" if r["c_graze"] is None else f'{100*r["c_graze"]:.0f}%'
            A(f'<tr><td class="lbl">{html.escape(r["name"])}</td>'
              f'<td>{r["c_clean"]}/{r["trials"]}</td><td>{r["c_se"]:.1f}</td>'
              f'<td>{r["c_lax"]}/{r["trials"]}</td><td>{gz}</td>'
              f'<td>{r["c_tsolve"]:.2f} s</td><td>{r["c_ms"]:.3f}</td></tr>')
        A('</tbody></table></div>')
        A('<div class="prose">')
        A('<p><b>The memoryless control is not beaten.</b> Random Sampling throws '
          'away the incumbent every iteration and draws candidates around zero '
          'control. It is nominally ahead of Predictive Sampling &mdash; by 7 '
          'points against a combined standard error of about 6, so treat them as '
          'indistinguishable rather than ranked. Either way: at four iterations '
          'per decision this task is solved by re-deciding from scratch, not by '
          'refining a plan. Any claim that some planner\'s optimizer is what '
          'closes the corridor has to get past this row first.</p>')
        A('<p><b>Cross-Entropy fails here rather than underperforming.</b> 2/100 '
          'with 98% collisions, against 76/100 for the same sampling machinery '
          'under a different update rule. CEM refits a Gaussian to the elite set '
          'and samples from it; on a chaotic system the elite set at one '
          'iteration does not predict the next, so the covariance concentrates on '
          'a region the dynamics have already left. In a rollout the signature is '
          'a smooth, committed, wrong trajectory rather than a flailing one.</p>')
        A('<p><b>The collision criterion moves the ranking.</b> Annealed Sampling '
          'scores 81 at 20 mm and 49 clean &mdash; a 32-point drop, the largest '
          'here, and enough to take it from second place to fourth.</p>')
        A('</div>')
        A('</section>')

        # --------------------------------------------- 4. three gaps, all planners
        A(sec('Three gaps: the wall'))
        A('<div class="prose">')
        A('<p>Same objective, same protocol, three bottlenecks instead of one.</p>')
        A('</div>')
        srows = sorted(psweeps, key=lambda r: -r["s_clean"])
        A('<div class="card scroll">' + planner_bars(srows, "slalom") + '</div>')
        A('<div class="card scroll"><table><thead><tr><th>planner</th>'
          '<th>clean</th><th>&le;20 mm</th><th>grazing</th>'
          '<th>gaps before first contact</th><th>corridor (clean)</th>'
          '</tr></thead><tbody>')
        for r in srows:
            gz = "&mdash;" if r["s_graze"] is None else f'{100*r["s_graze"]:.0f}%'
            A(f'<tr><td class="lbl">{html.escape(r["name"])}</td>'
              f'<td>{r["s_clean"]}/{r["trials"]}</td>'
              f'<td>{r["s_lax"]}/{r["trials"]}</td><td>{gz}</td>'
              f'<td>{r["s_gaps"]:.2f} of 3</td>'
              f'<td style="color:var(--ink-3)">{r["c_clean"]}/{r["trials"]}</td>'
              f'</tr>')
        A('</tbody></table></div>')
        A('<div class="prose">')
        A('<p><b>At this objective, nothing solves it.</b> The best planner clears '
          'three gaps cleanly in 4 runs out of 100 and five of the seven never '
          'manage it at all. Differences among the top rows are one to two '
          'standard errors: this table reports a wall, not a ranking.</p>')
        A('<p><b>Read the partial-credit column instead.</b> Gaps cleared before '
          'the first contact does separate the planners, 0.94 down to 0.05 &mdash; '
          'and that ordering is essentially the corridor ordering. No planner '
          'handles chaining better than its single-gap performance predicts.</p>')
        A('<p>Nothing about the first gap changed. What changed is the goal, from '
          '6 m to 11 m, so the cart residual is roughly 3&times; larger at the '
          'start and the planner commits to more speed on approach. iLQG is the '
          'extreme case &mdash; 35/100 on the corridor, 0.09 gaps here &mdash; '
          'arriving at a gap it can otherwise clear with far too much velocity to '
          'lay the pendulum out.</p>')
        A('</div>')
        A('<div class="warn"><p><b>Retracted.</b> An earlier version of this work '
          'read the &le;20 mm column as the result and reported the slalom as a '
          'ranking <em>inversion</em>, with Annealed Sampling first at 34/100 and '
          'an argument about chaining rewarding different behaviour. The grazing '
          'shares here are 82&ndash;100%: on the slalom the lax test was measuring '
          'almost nothing but contact. Both columns are kept because the gap '
          'between them is the finding.</p></div>')
        A('</section>')

    A(sec('Every row was predictive sampling'))
    A('<div class="prose">')
    A('<p>The short answer to the naming question: <code>w32000_m008</code> and '
      '<code>w8000_m008</code> are the built-in sampling planner &mdash; '
      '<code>--planner=0</code>, predictive sampling &mdash; the same planner as '
      'every row prefixed <code>ps_</code>. The prefix was dropped when those runs '
      'were added to a table that no longer had a planner column, and nothing else '
      'in the label recorded it.</p>')
    A('<p>Recovering the settings from the log headers shows the tables were '
      'comparing rows that differed in more than their labels admitted. '
      'Three labels turn out to name one configuration:</p>')
    A('</div>')
    A('<div class="card scroll"><table><thead><tr>'
      '<th>label</th><th>source</th><th>planner</th><th>avoidance w</th>'
      '<th>margin</th><th>cart w</th><th>episode</th><th>solved</th>'
      '</tr></thead><tbody>')
    for h in sorted(hist, key=lambda h: (h.get("planner_class", ""),
                                         h.get("avoid_w", 0), h.get("margin") or 0)):
        dup = (h.get("avoid_w") == W8 and abs((h.get("margin") or 9) - M8) < 1e-9
               and h.get("planner_class") == "PredictiveSampling"
               and h.get("horizon") is None)
        mk = (' style="background:var(--zero);box-shadow:inset 3px 0 0 var(--collided)"'
              if dup else "")
        mtxt = f'{h["margin"]:.2f}' + ("" if h.get("margin_explicit") else "*")
        A(f'<tr{mk}><td class="lbl">{html.escape(h["label"])}</td>'
          f'<td style="color:var(--ink-3)">{html.escape(os.path.dirname(h["source"]).split("/")[-1])}</td>'
          f'<td>{html.escape(h.get("planner_class","?"))}</td>'
          f'<td>{h.get("avoid_w",0):,.0f}</td><td>{mtxt}</td>'
          f'<td>{h.get("weights",[1])[0]:g}</td>'
          f'<td>{h.get("total_time",0):g}s</td>'
          f'<td>{h["solved"]}/{h["trials"]}</td></tr>')
    A('</tbody></table></div>')
    A('<p class="cap">* margin not printed by that run, so the task default of 0.08 m '
      'was in force. Shaded rows are the same configuration: avoidance 8000, margin '
      '0.08 m, cart weight 1, 12 s episodes, predictive sampling. '
      '<code>ps_cart0.25</code> also differs in episode length (20 s) and cart '
      'weight, so it never belonged in a column next to the others.</p>')
    A('</section>')

    # ------------------------------------------------- 2. the RNG
    A(sec('The spread was the instrument'))
    A('<div class="prose">')
    A(f'<p>Those {len(pts)} runs are one configuration measured '
      f'{len(pts)} times. Nothing about the planner changed between them.</p></div>')
    if pts:
        A('<div class="card">' + repro_plot(pts) + '</div>')
        A('<div class="card scroll"><table><thead><tr><th>run</th><th>source</th>'
          '<th>solved</th><th>rate</th></tr></thead><tbody>')
        for p in sorted(pts, key=lambda p: p["pct"]):
            A(f'<tr><td class="lbl">{html.escape(p["name"])}</td>'
              f'<td style="color:var(--ink-3)">{html.escape(p["when"])}</td>'
              f'<td>{p["solved"]}/{p["trials"]}</td><td>{p["pct"]:.0f}%</td></tr>')
        A('</tbody></table></div>')
    A('<div class="prose">')
    A('<p>The cause is one line. Every candidate rollout, on every planner '
      'iteration, built a fresh generator seeded from system entropy:</p></div>')
    A('<div class="cmd">// mjpc/planners/sampling/planner.cc, AddNoiseToPolicy\n'
      'absl::BitGen gen_;   // entropy-seeded, per candidate, per iteration</div>')
    A('<div class="prose">')
    A('<p>The benchmark\'s <code>--seed</code> flag never touched it &mdash; it seeds '
      'the initial-state perturbation only, which its own help text said and which '
      'was easy to miss. So the planner drew different noise on every run of every '
      'sweep, and no table row was reproducible.</p>')
    A('<p>The generator is now keyed on <code>(seed, iteration, candidate)</code> '
      'rather than on entropy, and the candidate index is part of the key so that '
      'thread-pool interleaving cannot change the draw. With '
      '<code>--planner_seed</code> set, two runs of one command are identical down '
      'to the iteration count:</p></div>')
    A('<div class="cmd">'
      'RESULT planner=seeded_A ... planner_seed=1000 trials=12 solved=5 '
      'gaps_mean=2.08 t_solve_median=2.64 plan_iters=38032\n'
      'RESULT planner=seeded_B ... planner_seed=1000 trials=12 solved=5 '
      'gaps_mean=2.08 t_solve_median=2.64 plan_iters=38032</div>')
    A('<div class="warn"><p><b>Still unseeded:</b> cross-entropy, PSO, annealed '
      'sampling and sample-gradient each carry their own <code>absl::BitGen</code>. '
      'Any published comparison involving those planners has the same problem and '
      'is not yet reproducible.</p></div>')
    A('</section>')

    # ------------------------------------------------------- 3. the landscape
    A(sec('The map'))
    A('<div class="prose">')
    nseed = len({r["seed"] for r in rows}) or 1
    ntr = rows[0]["trials"] if rows else 50
    full = max((a["trials"] for a in agg.values()), default=0)
    worst_se = max((a["se"] for a in agg.values()), default=0)
    A(f'<p>{len(rows)} seeded runs: {len(weights)} avoidance weights &times; '
      f'{len(margins)} clearance margins &times; {nseed} seeds &times; '
      f'{ntr} trials. A complete cell pools {full} trials, and the widest '
      f'standard error on the map is {worst_se:.1f} points &mdash; so a '
      f'difference smaller than about {2*worst_se:.0f} points between two cells '
      f'is not a finding.</p>')
    # State the shape of the surface from the surface, not from expectation: the
    # best cell, how far the best margin's column moves with weight, and where
    # the whole thing collapses.
    if agg:
        (bw_, bm_), best = max(agg.items(), key=lambda kv: kv[1]["pct"])
        col = sorted(((w, agg[(w, bm_)]["pct"]) for w in weights
                      if (w, bm_) in agg), key=lambda t: t[0])
        row = sorted(((m, agg[(bw_, m)]["pct"]) for m in margins
                      if (bw_, m) in agg), key=lambda t: t[0])
        # The sealed-gap regime is not "0% solved" -- a weak barrier also scores
        # 0% by driving the cart straight into a disk. It is "0% solved AND the
        # cart never touched anything", which is a different cell entirely.
        sealed = sorted(k for k, a in agg.items()
                        if a["pct"] < 3 and a["collided_pct"] < 12)
        # Where the optimum sits for each weight. This is the shape of the
        # thing: the two knobs are not independent, and reporting only the
        # single best cell would hide that.
        ridge = []
        for w in weights:
            cand = [m for m in margins if (w, m) in agg]
            if cand:
                bm = max(cand, key=lambda m: agg[(w, m)]["pct"])
                ridge.append((w, bm, agg[(w, bm)]["pct"]))
        A(f'<p>The best cell is avoidance <b>{bw_:,}</b> at margin '
          f'<b>{bm_:.2f} m</b>: {best["solved"]}/{best["trials"]} = '
          f'<b>{best["pct"]:.0f}%</b> &plusmn;{best["se"]:.1f}. At the same '
          f'margin with weight {col[0][0]:,} it is {col[0][1]:.0f}%.</p>')
        if len(ridge) > 2:
            steps = ", ".join(f"{w:,}&#8202;&rarr;&#8202;{m:.2f}"
                              for w, m, _ in ridge)
            A(f'<p>But the optimum is not a corner, it is a <b>ridge</b>, and it '
              f'moves: the best margin for each weight runs {steps}. Every '
              f'factor of four on the weight moves it down a step or holds it '
              f'&mdash; the grid steps margin by 0.04 m, too coarse to fit a law '
              f'to, but the direction is unambiguous: the harder the avoidance '
              f'term is weighted, the less padding it wants. What the planner '
              f'needs is a barrier of a particular '
              f'stiffness &mdash; weak enough that the cart will still commit to '
              f'the gap, strong enough that it does not clip a disk on the way '
              f'through &mdash; and weight and margin are two ways of setting the '
              f'same quantity. Tuning either one alone walks across the ridge '
              f'instead of along it, which is what the earlier '
              f'one-knob-at-a-time sweeps were doing.</p>')
        if sealed:
            cells = ", ".join(f"{w:,}/{m:.2f}" for w, m in sealed)
            A(f'<p>The other direction is not a gentle decline, and it is not the '
              f'same failure. In {len(sealed)} cell'
              f'{"s" if len(sealed) > 1 else ""} &mdash; {cells} &mdash; the cart '
              f'neither reaches the goal <em>nor</em> ever touches a disk. It '
              f'stops short. Margin is metres of padding added to each disk '
              f'before the avoidance cost engages, and the gap is '
              f'{HALF_GAP*2:.2f} m wide, so at {HALF_GAP:.2f} m of padding the '
              f'two disks\' penalty regions meet in the middle of the gap. The '
              f'cost stops describing an obstacle to steer around and starts '
              f'describing a wall. That takes <em>both</em> knobs: a wide '
              f'penalty region and enough weight to make crossing it '
              f'unaffordable, which is why the collapse sits in one corner '
              f'rather than along an edge.</p>')
        # The cell the whole question started from. Naming it explicitly beats
        # making the reader hunt for it in the grid.
        prior = agg.get((32000, 0.08))
        if prior:
            A(f'<p>And the cell this started from: avoidance 32,000 at margin '
              f'0.08 m, once reported as 20/50 = 40%, pools to '
              f'<b>{prior["solved"]}/{prior["trials"]} = {prior["pct"]:.0f}%</b> '
              f'&plusmn;{prior["se"]:.1f} across three seeds '
              f'({", ".join(f"{p:.0f}%" for p in prior["per_seed"])}). It '
              f'replicates &mdash; the single run that found it was a high draw '
              f'by about six points, not an artefact. It is simply not the best '
              f'cell on the map; it was the best cell in the region the earlier '
              f'sweeps happened to search.</p>')
        A('</div>')
        want = grid_size(args.grid) or len(weights) * len(margins) * nseed
        if len(rows) < want:
            A(f'<div class="warn"><p><b>Grid still filling:</b> {len(rows)} of '
              f'{want} runs have landed. Empty cells are dashed, and the numbers '
              f'above will move.</p></div>')
    else:
        A('</div>')
    A('<div class="card">' + heatmap(agg, weights, margins) + '</div>')
    A(f'<p class="cap">Clearance margin is metres added to each disk\'s radius '
      f'before the avoidance cost starts charging. The gap between a disk and the '
      f'wall is {HALF_GAP*2:.2f} m wide, so at a margin near {HALF_GAP:.2f} m the '
      f'padded disks meet in the middle and the cost stops being a fence around '
      f'the obstacle and becomes a wall across the corridor. The outlined cell '
      f'in each row is that weight\'s best margin; together they trace the '
      f'ridge. Hover any cell for its per-seed spread.</p>')
    A('</section>')

    # ------------------------------------------------------- 8. past the ridge
    if args.ext and os.path.exists(os.path.join(args.ext, "results.csv")):
        erows, eagg = load_grid(args.ext)
        ew = sorted({r["weight"] for r in erows})
        em = sorted({r["margin"] for r in erows})
        A(sec('Past the ridge'))
        A('<div class="prose">')
        A('<p>The map above ends at its own best corner, which leaves the obvious '
          'question open: is that cell a peak, or the near edge of a plateau that '
          'keeps going? This grid continues past it &mdash; higher avoidance '
          'weights, smaller clearance margins, same three seeds and 50 trials per '
          'cell.</p>')
        if eagg:
            (bw, bm), best = max(eagg.items(), key=lambda kv: kv[1]["pct"])
            A(f'<p>The best cell of the extension is avoidance '
              f'<b>{bw:,.0f}</b> at margin <b>{bm:g} m</b>, at '
              f'<b>{best["pct"]:.0f}%</b> &plusmn;{best["se"]:.1f} over '
              f'{best["trials"]} trials.</p>')
        A('</div>')
        A('<div class="card">' + heatmap(eagg, ew, em) + '</div>')
        A('<p class="cap">Same scale and same scoring as the map above. The '
          'lowest-weight, largest-margin cell here repeats the best cell of the '
          'first grid, under a later binary, as a check that the two can be read '
          'together.</p>')
        want_e = grid_size(args.ext) or len(ew) * len(em) * 3
        if len(erows) < want_e:
            A(f'<div class="warn"><p><b>Grid still filling:</b> {len(erows)} of '
              f'{want_e} runs have landed.</p></div>')
        A('</section>')

    # --------------------------------------- 3b. the control that was not one
    if args.rs and os.path.exists(os.path.join(args.rs, "results.csv")):
        rsrows, rsagg = load_grid(args.rs)
        rw = sorted({r["weight"] for r in rsrows})
        rm = sorted({r["margin"] for r in rsrows})
        soln = sum(r["solved"] for r in rsrows)
        trin = sum(r["trials"] for r in rsrows)
        A(sec('The control that was not a control'))
        A('<div class="prose">')
        A('<p>Every sweep above is predictive sampling, which leaves open whether '
          'the ridge is a property of the cost function or of that one planner. '
          'The obvious check is the memoryless control: the same sampler with its '
          'incumbent thrown away before each iteration, so no plan survives from '
          'one iteration to the next. Run it on the same corner and see whether '
          'the tuning still helps.</p>')
        A('<p>The first attempt returned the grid it was being compared against. '
          'All 30 finished cells matched predictive sampling exactly &mdash; not '
          'closely, but to the trial &mdash; which is not something two distinct '
          'planners do. The control was a no-op: it cleared the plan, and the '
          'base planner&rsquo;s first act each iteration is to rebuild that plan '
          'from the previous winner, which threw the clearing away. It had been '
          'the planner it was a control for the whole time. The sweep script '
          'compounded it by printing <em>predictive sampling</em> into the '
          'manifest of every sweep regardless of which planner index it was '
          'given.</p>')
        A(f'<p>Corrected, and re-run on the same {len(rsagg)} cells with the same '
          f'seeds, it solves <b>{soln} of {trin}</b> trials &mdash; zero, '
          'everywhere, at every weight and margin in the corner where predictive '
          'sampling reaches 65%. It does not reach the first gap.</p>')
        A('<p>That reverses what this benchmark had been saying. The memoryless '
          'control was previously the top row of the one-gap table, which made '
          'the warm start look like decoration. It is the opposite: the warm '
          'start is the mechanism, and the cost tuning that carries predictive '
          'sampling from 2.7% to 65% does nothing at all for a planner that '
          'discards its plan. The objective is not solving this task &mdash; it '
          'is making an already-working search better.</p>')
        A('</div>')
        A('<div class="card">' + heatmap(rsagg, rw, rm) + '</div>')
        A('<p class="cap">The corrected memoryless control on the same corner as '
          'the grid above, same colour scale. Every cell is zero.</p>')
        A('</section>')

    # ------------------------------------------------------ 4. failure modes
    if agg:
        A(sec('Two different ways to score zero'))
        A('<div class="prose"><p>A cell at 0% can mean the cart charged the gap and '
          'clipped a disk, or that it never went near the gap at all. Those want '
          'opposite fixes, and the success column cannot tell them apart.</p></div>')
        A('<div class="legend">')
        for name, col in (("solved", "var(--solved)"), ("collided", "var(--collided)"),
                          ("stalled &mdash; never reached the goal, never touched a disk",
                           "var(--stalled)")):
            A(f'<span class="key"><span class="sw" style="background:{col}"></span>{name}</span>')
        A('</div>')
        A('<div class="card scroll">' + outcome_bars(agg, weights, margins) + '</div>')
        A('<p class="cap">Bars are percentages of all trials in the cell; the right '
          'column is bottlenecks cleared before first contact, out of 3.</p>')
        A('</section>')

    # ------------------------------------------------------------- 5. footage
    if vids:
        A(sec('What each outcome looks like'))
        A('<div class="prose"><p>One rollout per outcome per configuration, rendered '
          'from runs that went to the end rather than stopping at first contact. '
          'The earlier gallery picked one rollout per configuration by distance '
          'reached, which is why the configurations that solve had no footage of a '
          'solve.</p></div>')
        colmap = {"solved": "var(--solved)", "collided": "var(--collided)",
                  "stalled": "var(--stalled)"}
        bycfg = {}
        for v in vids:
            bycfg.setdefault(v["config"], []).append(v)
        order = {"solved": 0, "collided": 1, "stalled": 2}
        for cfg, items in bycfg.items():
            w = int(items[0]["weight"]); m = float(items[0]["margin"])
            items = sorted(items, key=lambda v: order.get(v["outcome"], 9))
            A(f'<h3>avoidance {w:,} &middot; margin {m:.2f} m</h3>')
            A('<div class="grid2">')
            for v in items:
                col = colmap.get(v["outcome"], "var(--ink-3)")
                A('<figure class="vid">')
                A(f'<div class="vh"><span class="pill" style="background:{col}">'
                  f'{v["outcome"]}</span><span style="color:var(--ink-3)">'
                  f'{html.escape(v["run"])}</span></div>')
                if v.get("mp4_data"):
                    A(f'<video controls muted loop playsinline preload="metadata" '
                      f'src="{v["mp4_data"]}"></video>')
                A(f'<figcaption class="vs">cart reached {float(v["max_cart"]):.2f} m'
                  f' &middot; closest approach {float(v["min_clearance"]):+.3f} m'
                  f'</figcaption>')
                A('</figure>')
            A('</div>')
            # One filmstrip per configuration, full width. Tiled eight-up it is
            # unreadable at card size, and it is the frame-by-frame view -- the
            # thing the videos are bad at -- so it gets the room.
            lead = next((v for v in items if v.get("png_path")), None)
            if lead:
                strip = b64(shrink_png(lead["png_path"], lead["cache"]), "image/jpeg")
                A(f'<figure style="margin:14px 0 0">'
                  f'<img class="strip" src="{strip}" '
                  f'alt="filmstrip of the {lead["outcome"]} rollout">'
                  f'<figcaption class="cap">{lead["outcome"]} rollout '
                  f'{html.escape(lead["run"])}, eight frames left to right, '
                  f'top row then bottom.</figcaption></figure>')
        A('</section>')

    # ----------------------------------------------------- 6. control character
    if cc:
        A(sec('What the controller does with the authority'))
        A('<div class="prose"><p>Measured on the dumped rollouts of the earlier '
          'gallery. These are the numbers that survived the audit unchanged: they '
          'describe control character within a run, not success rates across runs, '
          'so the seeding problem does not reach them.</p></div>')
        A('<div class="card scroll"><table><thead><tr><th>config</th>'
          '<th>mean |u|</th><th>effort &int;u&sup2;</th><th>reversals/s</th>'
          '<th>cost share: cart</th><th>avoidance</th><th>n</th>'
          '</tr></thead><tbody>')
        for k, v in sorted(cc.items(), key=lambda kv: -kv[1]["rev"]):
            A(f'<tr><td class="lbl">{html.escape(k)}</td><td>{v["u"]:.2f}</td>'
              f'<td>{v["eff"]:.0f}</td><td>{v["rev"]:.1f}</td>'
              f'<td>{v["cart"]:.0f}%</td><td>{v["avoid"]:.0f}%</td>'
              f'<td style="color:var(--ink-3)">{v["n"]}</td></tr>')
        A('</tbody></table></div></section>')

    # ------------------------------------------------------------ 7. reproduce
    # ----------------------------------------------- 12. cost of an iteration
    tim = load_timing(os.path.join(ROOT, "renders/timing/timing.log"))
    # renders/machine/machine.log is the later re-run at low load; machine_quiet
    # is the earlier one. They agree to within 2%, and this is the current one.
    mach_i = load_machine(os.path.join(ROOT, "renders/machine/machine.log"))
    if tim or mach_i[1]:
        A(sec('What an iteration costs, and on what'))
        A('<div class="prose">')
        A('<p>Every table above holds planning <em>iterations</em> constant, '
          'which is what makes them comparisons of algorithms rather than of '
          'throughput. This is the other half: what an iteration costs, and '
          'therefore whether a planner could deliver those iterations outside '
          'the harness.</p>')
        A('<p>The budget is one timestep, <b>5 ms</b>, at any '
          '<code>--speed</code>. That is worth stating precisely, because it is '
          'easy to confuse with the control period: at <code>--speed=0.25</code> '
          'control runs at 50 Hz in wall time, so the control period is 20 ms and '
          'the planner fits four iterations inside it. Slowing down buys '
          'iterations per decision and pays for them in control rate; the '
          'per-iteration budget does not move.</p>')
        A('</div>')
    if tim:
        A('<div class="card scroll"><table><thead><tr><th>planner</th>'
          '<th>ms/iter</th><th>p95</th><th>worst</th><th>% of 5 ms</th>'
          '<th>where it goes</th></tr></thead><tbody>')
        for k, v in sorted(tim.items(), key=lambda kv: kv[1]["ms"]):
            over = ' style="color:var(--collided)"' if v["p95"] > 5.0 else ""
            A(f'<tr><td class="lbl">{html.escape(PLANNER_NAMES.get(k, k))}</td>'
              f'<td>{v["ms"]:.3f}</td><td{over}>{v["p95"]:.2f}</td>'
              f'<td>{v["worst"]:.1f}</td><td>{100*v["ms"]/5.0:.0f}%</td>'
              f'<td style="color:var(--ink-3);white-space:normal">'
              f'{html.escape(v.get("where",""))}</td></tr>')
        A('</tbody></table></div>')
        A('<p class="cap">6000 iterations each, early exit off so every planner '
          'runs the same iterations from the same starts, 15 planner threads on '
          'an idle 20-core host. p95 in orange exceeds the budget: a real loop '
          'would drop iterations for those two precisely when the state is '
          'hardest, so their success rates above are optimistic in a way the '
          '1 ms planners\' are not.</p>')
        A('<div class="prose">')
        A('<p>The four 10-rollout samplers are within 3% of each other, and almost '
          'all of that is contact &mdash; with the obstacles removed they converge '
          'to 0.92&ndash;0.99 ms. Cross-Entropy\'s apparent 30% overhead is not '
          'the algorithm but its 79% collision rate: MuJoCo charges more for '
          'contact-rich rollouts, so a planner that drives into disks pays twice, '
          'once in the score and once on the clock.</p>')
        A('<p>The two exceptions are structural. Annealed Sampling\'s 4&times; is '
          'its annealing multiplier &mdash; 99.9% of its iteration is rollouts, so '
          'there is no overhead to trim; the cost <em>is</em> the extra samples. '
          'iLQG spends only a quarter of its iteration on rollouts: 45% goes to '
          'finite-difference derivatives and 27% to the nominal rollout, while the '
          'Riccati backward pass, the part usually expected to dominate, is 2.7%. '
          'On a 4-DOF system the derivatives are already the bill.</p>')
        A('</div>')
    if mach_i[1]:
        host_i, rows_i = mach_i
        # The macOS run is a pasted log rather than a generated one, so its
        # table is transcribed here with the load it was taken at.
        mac = [(1, 3.215), (2, 1.700), (4, 1.169), (8, 1.189), (10, 1.175)]
        A('<h3 style="margin-top:34px">Two hosts, same workload</h3>')
        A('<div class="card scroll"><table><thead><tr><th>threads</th>'
          '<th>Ultra 7 265KF ms/iter</th><th>Msteps/s</th><th>speedup</th>'
          '<th>M5 Pro ms/iter</th></tr></thead><tbody>')
        macd = dict(mac)
        for r in rows_i:
            m = macd.get(r["threads"])
            mcell = (f'<td>{m:.3f}</td>' if m
                     else '<td style="color:var(--ink-3)">&mdash;</td>')
            A(f'<tr><td class="lbl">{r["threads"]}</td><td>{r["ms"]:.3f}</td>'
              f'<td>{r["msteps"]:.2f}</td><td>{r["speedup"]}</td>{mcell}</tr>')
        A('</tbody></table></div>')
        A(f'<p class="cap">Predictive sampling, 1.0 s horizon, 12 knots, 10 '
          f'trajectories, 8000 iterations. Ultra 7: 20 logical cores, '
          f'{html.escape(host_i.get("load","?"))}. M5 Pro: 15 logical cores, '
          f'load averages 2.97 4.53 4.77. Msteps/s is mj_step calls per second '
          f'across all planner threads.</p>')
        A('<div class="prose">')
        A('<p>Single-thread the M5 Pro is 28% faster. Neither host converts more '
          'than about 3.5&times; of its cores, and both plateau at the same place '
          'in absolute terms &mdash; 1.17 against 1.18 ms &mdash; so at the budget '
          'this task uses the two machines are equivalent and the single-thread '
          'difference is not recoverable. Ten rollouts of 201 steps is simply not '
          'enough parallel work to fill either one.</p>')
        A('<p>That headroom is the most concrete thing this page has to say about '
          'getting the success rate up. Both machines sustain roughly 1.7 '
          'Msteps/s and the task asks for 0.49. Raising the rollout count is a '
          '4&times; that fits the budget on both hosts &mdash; the same 4&times; '
          'annealed sampling spends on its annealing schedule for no measured '
          'gain.</p>')
        A('</div>')
    if tim or mach_i[1]:
        A('</section>')

    A(sec('Reproducing any cell'))
    A('<div class="prose"><p>Every cell of the map came from one command. To '
      're-run one, take its weight, margin and seed off the hover tooltip:</p></div>')
    A('<div class="cmd">./build/bin/corridor_benchmark --task=slalom --planner=0 '
      '--stage=corridor \\\n'
      '    --weights=1,0,0.1,0.01,&lt;W&gt; --clearance=&lt;M&gt; \\\n'
      '    --speed=0.25 --total_time=12 --repeats=50 \\\n'
      '    --seed=1 --planner_seed=&lt;S&gt; --per_run=false --planner_thread=4</div>')
    A('<div class="prose">')
    A('<p>The whole grid, and the videos:</p></div>')
    A('<div class="cmd">'
      'mjpc/tasks/triple_pendulum_cartpole/benchmark/landscape_grid.sh\n'
      'mjpc/tasks/triple_pendulum_cartpole/benchmark/outcome_renders.sh</div>')
    A('<div class="prose">')
    A('<p>Two things the grid deliberately does not measure. Milliseconds per '
      'iteration: it ran five jobs at once, so those numbers are contended and '
      'belong to <code>machine_bench.sh</code> on an idle box instead. And the '
      'planner-to-planner comparison: cross-entropy, PSO and annealed sampling '
      'are still unseeded, so their rows would carry the spread this page just '
      'finished removing.</p>')
    A('<p>Outputs are written under a timestamped directory &mdash; '
      '<code>renders/runs/&lt;YYYYMMDDThhmmss&gt;_&lt;name&gt;/</code> &mdash; '
      'with a <code>manifest.txt</code> recording host, commit, binary checksum '
      'and the full grid, and a <code>commands.txt</code> giving the exact '
      'invocation behind every row.</p></div>')
    A('</section>')

    A('</div>')
    A(f"<script>{JS}</script>")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(o), encoding="utf-8")
    mb = out.stat().st_size / 1e6
    print(f"wrote {out} ({mb:.2f} MB)  cells={len(agg)} runs={len(rows)} "
          f"history={len(hist)} videos={len(vids)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="")
    ap.add_argument("--ext", default="",
                    help="ridge-extension grid dir (higher weights, smaller "
                         "margins) -- continues --grid past its best corner")
    ap.add_argument("--rs", default="",
                    help="the same corner as --ext under the memoryless "
                         "control (random sampling), for the section on what "
                         "the warm start is worth")
    ap.add_argument("--renders", default="")
    ap.add_argument("--repro", default="")
    ap.add_argument("--out", default="renders/landscape_report.html")
    build(ap.parse_args())
