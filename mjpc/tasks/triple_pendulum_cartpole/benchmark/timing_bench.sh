#!/usr/bin/env bash
# What one planning iteration costs, per planner.
#
# The success-rate sweeps hold planning *iterations* constant, which is what
# makes them a comparison of algorithms rather than of throughput. This script
# measures the other half: what an iteration costs, and therefore whether a
# planner could deliver those iterations outside the benchmark.
#
# The deadline is one control period. At --speed s the planner is asked for 1/s
# iterations per control step, and one simulated second has to be produced in
# 1/s real seconds -- so the speed cancels and the condition is simply that one
# iteration fits inside one timestep (5 ms here). A planner above that line is
# being given more iterations by the harness than it could ever get in a real
# loop, and its success rate should be read with that in mind.
#
# --early_exit is off on purpose: every planner then runs exactly the same
# number of iterations from the same starts, so the comparison is not skewed by
# one planner failing early and timing only its cheap first second.
#
# Run this with nothing else on the machine. Timing is the one measurement here
# that CPU contention silently corrupts.
#
# Usage: timing_bench.sh [out_dir] [total_time] [repeats] [speed]
set -u

BIN=${BIN:-./build/bin/corridor_benchmark}
OUT=${1:-renders/timing}
TOTAL_TIME=${2:-10}
REPEATS=${3:-3}
SPEED=${4:-1.0}

TASK=${TASK:-corridor}
# STAGE=balance removes the obstacles from the model entirely, which is the
# contact-free control for this measurement: planners that fail more end up in
# contact more, and MuJoCo charges more for a contact-rich rollout, so a
# corridor-stage timing table confounds "expensive algorithm" with "bad
# outcome". Comparing the two stages separates them.
STAGE=${STAGE:-corridor}
WEIGHTS=${WEIGHTS:-1,0,0.1,0.01,500}
SEED=${SEED:-1}

mkdir -p "$OUT"
LOG="$OUT/timing.log"
: > "$LOG"
echo "task=$TASK stage=$STAGE weights=$WEIGHTS speed=$SPEED total_time=$TOTAL_TIME repeats=$REPEATS seed=$SEED" | tee -a "$LOG"
echo "binary=$(md5sum "$BIN" | cut -d' ' -f1) commit=$(git rev-parse --short HEAD) date=$(date -Is)" | tee -a "$LOG"
echo "host=$(nproc) cores, planner threads=$(( $(nproc) - 5 ))" | tee -a "$LOG"

VARIANTS=(
  "predictive_sampling:0:"
  "ilqg:2:"
  "cross_entropy:5:"
  "pso:7:--pso_publish_evaluated=1"
  "annealed_sampling:8:"
  "random_sampling:9:"
)

for v in "${VARIANTS[@]}"; do
  label=${v%%:*}; rest=${v#*:}
  idx=${rest%%:*}; extra=${rest#*:}
  echo "=== $label ===" | tee -a "$LOG"
  # shellcheck disable=SC2086
  $BIN --task="$TASK" --planner="$idx" --stage="$STAGE" --weights="$WEIGHTS" \
       --speed="$SPEED" --total_time="$TOTAL_TIME" --repeats="$REPEATS" \
       --seed="$SEED" --early_exit=false --per_run=false --label="$label" \
       $extra 2>&1 | tee -a "$LOG"
  echo | tee -a "$LOG"
done

echo "=== summary ===" | tee -a "$LOG"
awk '
  /^RESULT/ {
    delete f
    for (i = 2; i <= NF; i++) { split($i, kv, "="); f[kv[1]] = kv[2] }
    if (!header++)
      printf "%-22s %10s %10s %8s %10s %9s\n",
             "planner", "ms/iter", "p95", "iters", "rollout%", "vs 5ms"
    printf "%-22s %10.3f %10.3f %8d %9.0f%% %8.0f%%\n", f["planner"],
           f["ms_per_iter"], f["ms_per_iter_p95"], f["plan_iters"],
           100 * f["rollout_frac"], 100 * f["ms_per_iter"] / 5.0
  }
' "$LOG" | tee -a "$LOG"

echo "wrote $LOG"
