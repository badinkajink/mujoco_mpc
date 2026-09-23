#!/usr/bin/env bash
# swap_aug17_numerics.sh aug|now  -- put the August-17 (strat 22 era) task numerics into the LIVE model
# (agent_planner 5=CEM, com_y_offset 0, lateral_center_braced 0, brace_flat_gate 0.17, stance_width_min 0.30,
# standback_pitch_release 0.42, ...) or restore the current 09-15 numerics. Copies into the build tree. Restart the node after.
set -e; D="$(cd "$(dirname "$0")" && pwd)"; X="$D/Lean_H12_Magpie.xml"
case "$1" in aug) cp "$X.aug17numerics" "$X";; now) cp "$X.current_0915" "$X";; *) echo "usage: $0 aug|now"; exit 2;; esac
cd "$D/../../../../build" && ninja copy_model_resources >/dev/null 2>&1
md5sum "$X" "$D/../../../../build/mjpc/tasks/humanoid_bench/lean/Lean_H12_Magpie.xml" | awk '{print $1}' | uniq | wc -l | sed 's/^1$/[swap] build tree synced/;s/^2$/[swap] MISMATCH/'
grep -o 'name="agent_planner" data="[^"]*"\|name="com_y_offset" data="[^"]*"\|name="lateral_center_braced" data="[^"]*"' "$X"
