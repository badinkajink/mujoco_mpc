#!/usr/bin/env python3
"""Assemble docs/lean/<date>-planner_ablation.html from page_body.html (the
prose, with {{TABLE:arm1,arm2,...}} and {{FIG:name.png}} markers), analyze.py's
summary.json and the figures under docs/lean/media/planner_ablation/."""
import argparse, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
STYLE = """
:root{--bg:#fafbfd;--fg:#1d2835;--muted:#546474;--rule:#d6dde6;--th:#e9eef5;--card:#ffffff;--code:#eef1f5;--link:#075eaa;--ok:#1b5e20;
  --icem:#1f77b4;--cem:#2ca02c;--ps:#d62728;--mppi:#ff7f0e}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#14181d;--fg:#e4e8ee;--muted:#9aa5b4;--rule:#2c343d;--th:#1f262e;--card:#181d23;--code:#222932;--link:#7db5e8;--ok:#8fd19e}}
:root[data-theme="dark"]{--bg:#14181d;--fg:#e4e8ee;--muted:#9aa5b4;--rule:#2c343d;--th:#1f262e;--card:#181d23;--code:#222932;--link:#7db5e8;--ok:#8fd19e}
body{font:17px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;max-width:1180px;margin:40px auto;padding:0 24px;color:var(--fg);background:var(--bg)}
h1{font-size:32px;line-height:1.15;text-wrap:balance}h2{margin-top:36px}h3{margin-top:24px}
p{max-width:78ch}ul{max-width:80ch}
table{border-collapse:collapse;width:100%;font-size:13.5px;background:var(--card);font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}
th{background:var(--th)}caption{text-align:left;font-size:14px;color:var(--muted);padding:6px 0;caption-side:top}
.scroll{overflow-x:auto;margin:12px 0}
code,pre{background:var(--code);font-size:13.5px;border-radius:3px;padding:0 3px}pre{padding:14px;white-space:pre-wrap}
img{max-width:100%;display:block;margin:8px auto;background:#fff;border-radius:4px}
a{color:var(--link)}.muted{color:var(--muted)}
figure{margin:18px 0}figcaption{font-size:14px;color:var(--muted);max-width:90ch}
figure.paper{background:var(--card);padding:12px 12px 4px;border:1px solid var(--rule);border-radius:4px}
figure.paper img{max-width:min(100%,760px)}
tr.icem td:first-child{border-left:4px solid var(--icem)}tr.cem td:first-child{border-left:4px solid var(--cem)}
tr.ps td:first-child{border-left:4px solid var(--ps)}tr.mppi td:first-child{border-left:4px solid var(--mppi)}
td b{color:var(--ok)}td.rungs{white-space:nowrap;font-size:12px}
.vids{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px;margin:12px 0}
figure.vid{margin:0}figure.vid video{width:100%;display:block;background:#000;border-radius:4px}figure.vid figcaption{font-size:13px}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default=os.path.join(HERE, "page_body.html"))
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--media_rel", default="media/planner_ablation")
    ap.add_argument("--title", required=True)
    ap.add_argument("--nruns", type=int, default=0, help="scored runs across every rate, for the header")
    a = ap.parse_args()
    body = open(a.body).read()

    def table(m):
        arms = m.group(1)
        cap = m.group(2) or ""
        r = subprocess.run([sys.executable, os.path.join(HERE, "tables.py"), "--summary", a.summary,
                            "--arms", arms, "--caption", cap], capture_output=True, text=True, check=True)
        return r.stdout

    body = re.sub(r"\{\{TABLE:([^}|]+)(?:\|([^}]*))?\}\}", table, body)

    def table2(m):
        # {{TABLE2:<summary path relative to this dir>:arms|caption}} -- a table
        # from a different scored set (the spp=15 wave reuses arm names)
        path, arms, cap = m.group(1), m.group(2), m.group(3) or ""
        r = subprocess.run([sys.executable, os.path.join(HERE, "tables.py"), "--summary",
                            os.path.join(HERE, path), "--arms", arms, "--caption", cap],
                           capture_output=True, text=True, check=True)
        return r.stdout

    body = re.sub(r"\{\{TABLE2:([^}:]+):([^}|]+)(?:\|([^}]*))?\}\}", table2, body)

    def table_rate(m):
        r = subprocess.run([sys.executable, os.path.join(HERE, "tables_rate.py"), m.group(1) or ""],
                           capture_output=True, text=True, check=True)
        return r.stdout

    body = re.sub(r"\{\{TABLE_RATE(?:\|([^}]*))?\}\}", table_rate, body)
    body = re.sub(r"\{\{FIG:([^}|]+)(?:\|([^}]*))?\}\}",
                  lambda m: '<figure><img src="%s/%s" alt="%s"><figcaption>%s</figcaption></figure>'
                  % (a.media_rel, m.group(1), m.group(1), m.group(2) or ""), body)
    # paper figures (IEEE column width): shown at a fixed width so they read as in print
    body = re.sub(r"\{\{PFIG:([^}|]+)(?:\|([^}]*))?\}\}",
                  lambda m: '<figure class=paper><img src="%s/%s" alt="%s"><figcaption>%s</figcaption></figure>'
                  % (a.media_rel, m.group(1), m.group(1), m.group(2) or ""), body)
    S = json.load(open(a.summary))
    body = body.replace("{{NRUNS}}", str(a.nruns or len(S["runs"])))
    left = sorted(set(re.findall(r"\{\{[A-Z_]+\}\}", body)))
    if left:
        print("unresolved markers:", ", ".join(left), file=sys.stderr)
    html = ("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>%s</title><style>%s</style>\n%s\n</html>\n" % (a.title, STYLE, body))
    open(a.out, "w").write(html)
    print("->", a.out)


if __name__ == "__main__":
    main()
