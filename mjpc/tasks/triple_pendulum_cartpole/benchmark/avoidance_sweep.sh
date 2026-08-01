#!/usr/bin/env bash
# Run every planner on the obstacle-avoidance question at a fixed objective and
# a fixed number of trials, and tabulate the success rates.
#
# This is the sweep for one question -- "can this planner drive the cart
# through the corridor without hitting a disk" -- as opposed to sweep.sh, which
# walks all three stages to diagnose where a planner fails. Here the objective
# is stated on the command line rather than taken from the stage, so the table
# says exactly what was optimized.
#
# Every planner sees the same trials: --seed fixes the per-trial initial
# perturbation, so trial k is the same start for all of them. Without that the
# comparison is between planners AND between problem sets.
#
# Usage: avoidance_sweep.sh [out_dir] [repeats] [speed] [extra flags...]
#   avoidance_sweep.sh renders/avoid100 100 0.25
#   avoidance_sweep.sh renders/avoid_h2 100 0.25 --horizon=2.0
set -u

BIN=${BIN:-./build/bin/corridor_benchmark}
OUT=${1:-renders/avoidance}
REPEATS=${2:-100}
SPEED=${3:-0.25}
shift 3 2>/dev/null || shift $#
EXTRA=("$@")

WEIGHTS=${WEIGHTS:-1,0,0.1,0.01,500}
TOTAL_TIME=${TOTAL_TIME:-20}
STAGE=${STAGE:-corridor}
SEED=${SEED:-1}

mkdir -p "$OUT/dumps"
LOG="$OUT/sweep.log"
: > "$LOG"

# label:planner_index:extra_flags
VARIANTS=(
  "predictive_sampling:0:"
  "ilqg:2:"
  "cross_entropy:5:"
  "pso:7:--pso_publish_evaluated=1"
  "pso_stock:7:--pso_publish_evaluated=0"
  "annealed_sampling:8:"
  "random_sampling:9:"
)

# Provenance. Every row of the table has to come from one build of the binary,
# so record which one: a rebuild part-way through a sweep is the easy way to
# end up with a table whose rows were measured by different code.
echo "task=${TASK:-corridor} weights=$WEIGHTS stage=$STAGE speed=$SPEED repeats=$REPEATS seed=$SEED extra=${EXTRA[*]:-none}" | tee -a "$LOG"
echo "binary=$(md5sum "$BIN" | cut -d' ' -f1) commit=$(git rev-parse --short HEAD) date=$(date -Is)" | tee -a "$LOG"

for v in "${VARIANTS[@]}"; do
  label=${v%%:*}; rest=${v#*:}
  idx=${rest%%:*}; extra=${rest#*:}
  echo "=== $label ===" | tee -a "$LOG"
  # shellcheck disable=SC2086
  $BIN --task="${TASK:-corridor}" --planner="$idx" --stage="$STAGE" --weights="$WEIGHTS" \
       --speed="$SPEED" --total_time="$TOTAL_TIME" --repeats="$REPEATS" \
       --seed="$SEED" --per_run=false --label="$label" \
       --dump="$OUT/dumps/${label}.csv" --dump_runs=2 \
       $extra "${EXTRA[@]}" 2>&1 | tee -a "$LOG"
  echo | tee -a "$LOG"
done

echo "=== summary ===" | tee -a "$LOG"
awk '
  /^RESULT/ {
    delete f
    for (i = 2; i <= NF; i++) { split($i, kv, "="); f[kv[1]] = kv[2] }
    if (!header++) printf "%-20s %10s %14s %10s %9s\n",
                          "planner", "solved", "solved_pct", "collided", "t_solve"
    printf "%-20s %10s %14s %9s%% %8ss\n", f["planner"],
           f["solved"] "/" f["trials"], f["solved_pct"],
           f["collided_pct"], f["t_solve_median"]
  }
' "$LOG" | tee -a "$LOG"

echo "wrote $LOG"
