#!/usr/bin/env bash
# swap_table_low.sh low|nominal -- planner table 5 cm LOW (z 0.49, the 09-13 strat-11 grid config that seated slowly
# without a dive: the planner believes the slab is 5 cm lower than the real 0.985 face, so the forearm meets the real
# table before the CoM commits) or nominal (z 0.54). Copies into the build tree; restart the node after.
set -e; D="$(cd "$(dirname "$0")" && pwd)"; X="$D/lean.xml"
case "$1" in low) sed -i 's|<body name="table" pos="1.0400 0 0.5400">|<body name="table" pos="1.0400 0 0.4900">|' "$X";; nominal) sed -i 's|<body name="table" pos="1.0400 0 0.4900">|<body name="table" pos="1.0400 0 0.5400">|' "$X";; *) echo "usage: $0 low|nominal"; exit 2;; esac
cd "$D/../../../../build" && ninja copy_model_resources >/dev/null 2>&1
grep -o 'name="table" pos="[^"]*"' "$X"; md5sum "$X" "$D/../../../../build/mjpc/tasks/humanoid_bench/lean/lean.xml" | awk '{print $1}' | uniq | wc -l | sed 's/^1$/[swap] build tree synced/;s/^2$/[swap] MISMATCH/'
