#!/usr/bin/env bash
# swap_strat25_kterms.sh 0|3|6|9|live  -- put the k-term brace-cost variant into the strat 25 slot (h12_brace_targeting.json)
# for the hardware cost ablation, or restore the live file. Copies into the build tree; restart the node after.
set -e; D="$(cd "$(dirname "$0")" && pwd)"; J="$D/h12_brace_targeting.json"
V="/home/the2xman/Desktop/h12/logs/hyperparamneter sweep/better_figs/cost_sweep/variants"
case "$1" in 0|3|6|9) cp "$V/k$1.json" "$J"; echo "[swap] strat 25 = k$1 ($1 brace terms)";; live) cp "$J.live_0915" "$J"; echo "[swap] strat 25 = live file restored";; *) echo "usage: $0 0|3|6|9|live"; exit 2;; esac
cd "$D/../../../../../build" && ninja copy_model_resources >/dev/null 2>&1
md5sum "$J" "$D/../../../../../build/mjpc/tasks/humanoid_bench/lean/strategies/h12_brace_targeting.json" | awk '{print $1}' | uniq | wc -l | sed 's/^1$/[swap] build tree synced/;s/^2$/[swap] MISMATCH/'
