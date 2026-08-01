#!/usr/bin/env bash
# Run every planner across every stage of the triple-pendulum-cartpole task and
# collect both the summary lines and a per-step dump of the first repeat.
#
# The dumps are what make the numbers checkable: aggregate cost cannot tell
# "threaded the corridor and captured" from "drove through with the pendulum
# whirling", and on this task the second scores better. Render them with
# filmstrip.py before believing any ranking.
#
# Usage: sweep.sh [output_dir] [total_time] [repeats]
set -u

BIN=${BIN:-./bin/corridor_benchmark}
OUT=${1:-sweep}
TOTAL_TIME=${2:-20}
REPEATS=${3:-3}

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
STAGES=(corridor balance combined)

for stage in "${STAGES[@]}"; do
  for v in "${VARIANTS[@]}"; do
    label=${v%%:*}; rest=${v#*:}
    idx=${rest%%:*}; extra=${rest#*:}
    echo "=== stage=$stage planner=$label ===" | tee -a "$LOG"
    # shellcheck disable=SC2086
    $BIN --planner="$idx" --stage="$stage" \
         --total_time="$TOTAL_TIME" --repeats="$REPEATS" \
         --dump="$OUT/dumps/${stage}_${label}.csv" $extra 2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"
  done
done

echo "wrote $LOG and $OUT/dumps/"
