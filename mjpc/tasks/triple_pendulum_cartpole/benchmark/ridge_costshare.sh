#!/usr/bin/env bash
# Cost-share characterization along the ridge of the weight x margin landscape.
#
# config_gallery.sh characterizes the configurations that were interesting
# before the landscape sweep ran, and every one of them solves the slalom at
# under 35%. That leaves the cost-share table describing what the planner pays
# for when it is mostly failing. This script runs the same characterization on
# the four cells along the ridge, from its start at (128000, 0.04) to its far
# end at (8192000, 0.01), so the table can be read as a progression in solve
# rate rather than a list of near-misses.
#
# Only the dump pass is run: the solve rates for these cells already exist in
# the landscape grid, and repeating them here would cost hours and produce a
# second, differently-seeded set of numbers for the same cells.
#
# Contention does not affect what comes out. The planner gets a fixed number of
# iterations per control step, not a wall-clock budget, so this is safe to run
# alongside a sweep -- it only takes longer.
#
# Usage: ridge_costshare.sh [out_dir] [dumps_per_config]
set -u

BIN=${BIN:-./build/bin/corridor_benchmark}
OUT=${1:-renders/ridge_costshare}
DUMPS=${2:-8}
TOTAL_TIME=${TOTAL_TIME:-12}
SPEED=${SPEED:-0.25}
SEED=${SEED:-1}
PLANNER_SEED=${PLANNER_SEED:-1000}
THREADS=${THREADS:-4}
BENCH=$(dirname "$0")

mkdir -p "$OUT"
LOG="$OUT/costshare.log"
SUM="$OUT/cost_control_summary.csv"
: > "$LOG"; rm -f "$SUM"

{
  echo "=== ridge cost share ==="
  echo "date        : $(date -Is)"
  echo "host        : $(uname -sr) $(uname -m)"
  echo "load at start: $(uptime | sed 's/.*load average/load average/')"
  echo "commit      : $(git rev-parse HEAD)"
  echo "binary      : $BIN"
  echo "binary md5  : $(md5sum "$BIN" | cut -d' ' -f1)"
  echo "fixed       : total_time=${TOTAL_TIME}s speed=$SPEED seed=$SEED"
  echo "              planner_seed=$PLANNER_SEED dumps=$DUMPS early_exit=false"
  echo "              (early exit off: a rollout cut at first contact cannot"
  echo "               show what the planner did afterwards)"
  echo
} | tee -a "$LOG"

# label|planner_idx|weights|margin  -- the ridge, in ascending solve rate
CONFIGS=${CONFIGS:-"
ps_w128000_m004|0|1,0,0.1,0.01,128000|0.04
ps_w512000_m002|0|1,0,0.1,0.01,512000|0.02
ps_w2048000_m002|0|1,0,0.1,0.01,2048000|0.02
ps_w8192000_m001|0|1,0,0.1,0.01,8192000|0.01
rs_w8192000_m001|9|1,0,0.1,0.01,8192000|0.01
"}

while IFS='|' read -r label idx weights margin; do
  [ -z "${label:-}" ] && continue
  echo "=== $label ===" | tee -a "$LOG"
  D="$OUT/$label"; mkdir -p "$D"
  $BIN --task=slalom --planner="$idx" --stage=corridor --weights="$weights" \
       --clearance="$margin" --speed="$SPEED" --total_time="$TOTAL_TIME" \
       --repeats="$DUMPS" --seed="$SEED" --planner_seed="$PLANNER_SEED" \
       --per_run=false --planner_thread="$THREADS" --early_exit=false \
       --dump="$D/r.csv" --dump_runs="$DUMPS" --label="${label}_dumps" \
       >> "$LOG" 2>&1

  for f in "$D"/r_*.csv; do
    [ -e "$f" ] || continue
    python3 "$BENCH/cost_control.py" --dump "$f" --weights "$weights" \
        --margin "$margin" --out "${f%.csv}_cc" --summary "$SUM" \
        --label "$label" >> "$LOG" 2>&1 || true
  done
  echo "  dumps: $(ls "$D"/r_*.csv 2>/dev/null | wc -l)" | tee -a "$LOG"
done <<< "$CONFIGS"

echo "wrote $SUM" | tee -a "$LOG"
