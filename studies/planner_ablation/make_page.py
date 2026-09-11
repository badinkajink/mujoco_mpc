#!/usr/bin/env python3
"""Assemble docs/lean/<date>-planner_ablation.html from page_body.html (the
prose, with {{TABLE:arm1,arm2,...}} and {{FIG:name.png}} markers), analyze.py's
summary.json and the figures under docs/lean/media/planner_ablation/."""
import argparse, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
STYLE = ("body{font:17px/1.55 system-ui,sans-serif;max-width:1180px;margin:40px auto;padding:0 24px;"
         "color:#1d2835;background:#fafbfd}h1{font-size:32px;line-height:1.15}h2{margin-top:36px}"
         "h3{margin-top:24px}table{border-collapse:collapse;width:100%;font-size:13.5px;background:white}"
         "th,td{padding:6px 8px;border-bottom:1px solid #d6dde6;text-align:left;vertical-align:top}"
         "th{background:#e9eef5}caption{text-align:left;font-size:14px;color:#546474;padding:6px 0}"
         ".scroll{overflow-x:auto;margin:12px 0}code,pre{background:#eef1f5;font-size:13.5px}pre{padding:14px;white-space:pre-wrap}"
         "img{max-width:100%;display:block;margin:8px auto}a{color:#075eaa}.muted{color:#546474}"
         "figure{margin:18px 0}figcaption{font-size:14px;color:#546474}"
         "tr.icem td:first-child{border-left:4px solid #1f77b4}tr.cem td:first-child{border-left:4px solid #2ca02c}"
         "tr.ps td:first-child{border-left:4px solid #d62728}tr.mppi td:first-child{border-left:4px solid #ff7f0e}"
         "td b{color:#1b5e20}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default=os.path.join(HERE, "page_body.html"))
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--media_rel", default="media/planner_ablation")
    ap.add_argument("--title", required=True)
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
    body = re.sub(r"\{\{FIG:([^}|]+)(?:\|([^}]*))?\}\}",
                  lambda m: '<figure><img src="%s/%s" alt="%s"><figcaption>%s</figcaption></figure>'
                  % (a.media_rel, m.group(1), m.group(1), m.group(2) or ""), body)
    html = ("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>%s</title><style>%s</style>\n%s\n</html>\n" % (a.title, STYLE, body))
    open(a.out, "w").write(html)
    print("->", a.out)


if __name__ == "__main__":
    main()
