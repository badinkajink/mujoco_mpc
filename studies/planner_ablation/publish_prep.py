#!/usr/bin/env python3
"""Make the artifact copy of the built page: strip the document wrapper (the
artifact host supplies it), keep <title> and <style>; the media files are published alongside (pass
--inline to embed them as data URIs instead)."""
import base64, os, re, sys
src, out = sys.argv[1], sys.argv[2]
html = open(src).read()
docs_dir = os.path.dirname(os.path.abspath(src))
title = re.search(r"<title>(.*?)</title>", html).group(1)
style = re.search(r"<style>(.*?)</style>", html, re.S).group(1)
body = html.split("</style>", 1)[1].replace("</html>", "").strip()


def inline(m):
    p = os.path.join(docs_dir, m.group(1))
    b = base64.b64encode(open(p, "rb").read()).decode()
    return 'src="data:image/png;base64,%s"' % b


if "--inline" in sys.argv:
    body = re.sub(r'src="(media/[^"]+\.png)"', inline, body)
open(out, "w").write("<title>%s</title>\n<style>%s</style>\n%s\n" % (title, style, body))
print("->", out, os.path.getsize(out) // 1024, "KB")
