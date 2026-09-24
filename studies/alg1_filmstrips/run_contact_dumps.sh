#!/bin/bash
# Dense sample dumps around the brace-contact moment: one lean_bench run per
# (std_min, seed), every plan tick at or after each 0.25 s mark from --from to
# --to, two runs at a time through resguard (3.7 GB RSS each at 6 threads).
#
#   ./run_contact_dumps.sh "0.01:1 0.01:9 0.01:10 0.02:0" [from=19] [to=29] [total_time=30]
#
# Output: runs/cem_stdmin<SS>_s<SEED>_dense/{metrics.csv,qpos.csv,bench.log,samples/tickNN_tT/}
set -u
cd "$(dirname "$0")"
BIN=../../build_cmake/bin/lean_bench
JOBS=${1:-"0.01:1 0.01:9 0.01:10 0.02:0"}
FROM=${2:-19}; TO=${3:-29}; TT=${4:-30}
AT=$(python3 -c "import numpy as np; print(','.join('%.2f' % t for t in np.arange($FROM, $TO + 1e-9, 0.25)))")
run_one() {
  local sm=$1 seed=$2
  local tag; tag=$(printf "cem_stdmin%s_s%d_dense" "$(echo "$sm" | sed 's/^0\.//')" "$seed")
  mkdir -p "runs/$tag"
  ~/.claude/bin/resguard.sh run --mem 5G --cpu 700 -- "$BIN" --task "Lean H12 Magpie" --strategy 25 \
    --seed "$seed" --total_time "$TT" --threads 6 --spp 15 --gains deploy \
    --numeric agent_allocate_active_only=1 --numeric agent_planner=5 --numeric "std_min=$sm" \
    --out "runs/$tag/metrics.csv" --qpos_out "runs/$tag/qpos.csv" \
    --samples_out "runs/$tag/samples" --samples_at "$AT" > "runs/$tag/bench.log" 2>&1
  echo "$tag: $(grep -o 'fell=[01] complete=[01].*t_end=[0-9.]*' "runs/$tag/bench.log")"
}
n=0
for job in $JOBS; do
  sm=${job%%:*}; seed=${job##*:}
  run_one "$sm" "$seed" &
  n=$((n + 1))
  if [ $((n % 2)) -eq 0 ]; then wait; fi
done
wait
