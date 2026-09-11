#!/usr/bin/env python3
"""Compact per-arm completion table across the four plan rates of the
deploy-plant campaign (runs/summary_deploy_spp*.json)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arms import ARMS
sums = {}
for spp in (15, 10, 6, 3):
    p = os.path.join(HERE, "runs/summary_deploy_spp%d.json" % spp)
    sums[spp] = json.load(open(p)) if os.path.exists(p) else {"runs": []}
arms = [a for a in ARMS if any(r["arm"] == a for S in sums.values() for r in S["runs"])]
print("%-26s %8s %8s %8s %8s" % ("arm", "33/s", "50/s", "83/s", "167/s"))
for a in arms:
    cells = []
    for spp in (15, 10, 6, 3):
        rs = [r for r in sums[spp]["runs"] if r["arm"] == a]
        cells.append("%d/%d" % (sum(r["outcome"] == "complete" for r in rs), len(rs)) if rs else "·")
    print("%-26s %8s %8s %8s %8s" % ((a,) + tuple(cells)))
