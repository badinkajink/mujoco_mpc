#!/usr/bin/env bash
# Sweep the planning horizon on the three-bottleneck slalom.
#
# The slalom spaces its gaps 3 m apart, which is roughly 0.8-1.0 s of travel at
# the speeds the cart reaches -- so at the default 1 s horizon a planner can
# barely see the next gap while committing to the posture for this one. That is
# the whole reason the task exists, and this sweep is the measurement: if
# lookahead is what the task is short of, success should climb with horizon; if
# the task is really about reacting fast to a chaotic pendulum, it will not.
#
# Random Sampling is swept alongside Predictive Sampling on purpose. It has no
# memory between iterations, so any gain it makes from a longer horizon comes
# from seeing further within a single iteration rather than from better
# refinement of a plan -- which separates the two explanations.
#
# Usage: horizon_sweep.sh [out_dir] [repeats] [speed] [horizons...]
set -u

BIN=${BIN:-./build/bin/corridor_benchmark}
OUT=${1:-renders/horizon}
REPEATS=${2:-100}
SPEED=${3:-0.25}
shift 3 2>/dev/null || shift $#
HORIZONS=("$@")
[ ${#HORIZONS[@]} -eq 0 ] && HORIZONS=(0.5 1.0 2.0 3.0)

TASK=${TASK:-slalom}
WEIGHTS=${WEIGHTS:-1,0,0.1,0.01,500}
TOTAL_TIME=${TOTAL_TIME:-20}
SEED=${SEED:-1}
VARIANTS=${VARIANTS:-"predictive_sampling:0 random_sampling:9"}

# Knots per second of horizon. Empty leaves sampling_spline_points at whatever
# the XML says (12), which spreads a fixed knot budget over a longer horizon --
# so a 3 s sweep is also measuring a 3x coarser control signal at t=0, the part
# that actually gets executed. Set KNOTS_PER_SEC=12 to hold knot spacing at the
# 1 s task's 83 ms and measure lookahead on its own. agent_horizon and
# sampling_spline_points are not independent knobs; sweeping one without the
# other measures their product.
KNOTS_PER_SEC=${KNOTS_PER_SEC:-}

mkdir -p "$OUT"
LOG="$OUT/horizon.log"
: > "$LOG"
echo "task=$TASK weights=$WEIGHTS speed=$SPEED repeats=$REPEATS seed=$SEED horizons=${HORIZONS[*]} knots_per_sec=${KNOTS_PER_SEC:-xml}" | tee -a "$LOG"
echo "binary=$(md5sum "$BIN" | cut -d' ' -f1) commit=$(git rev-parse --short HEAD) date=$(date -Is)" | tee -a "$LOG"

for v in $VARIANTS; do
  label=${v%%:*}; idx=${v##*:}
  for h in "${HORIZONS[@]}"; do
    knots=()
    if [ -n "$KNOTS_PER_SEC" ]; then
      n=$(awk -v h="$h" -v k="$KNOTS_PER_SEC" 'BEGIN{printf "%d", int(h*k + 0.5)}')
      knots=(--spline_points="$n")
    fi
    echo "=== $label horizon=$h ${knots[*]+${knots[*]}} ===" | tee -a "$LOG"
    $BIN --task="$TASK" --planner="$idx" --stage=corridor --weights="$WEIGHTS" \
         --speed="$SPEED" --total_time="$TOTAL_TIME" --repeats="$REPEATS" \
         --seed="$SEED" --horizon="$h" --per_run=false \
         ${knots[@]+"${knots[@]}"} \
         --label="${label}_h${h}" 2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"
  done
done

echo "=== summary ===" | tee -a "$LOG"
awk '
  /^RESULT/ {
    delete f
    for (i = 2; i <= NF; i++) { split($i, kv, "="); f[kv[1]] = kv[2] }
    if (!header++) printf "%-24s %8s %10s %14s %10s %9s\n",
                          "run", "horizon", "solved", "solved_pct", "collided", "gaps"
    printf "%-24s %8s %10s %14s %9s%% %9s\n", f["planner"], f["horizon"],
           f["solved"] "/" f["trials"], f["solved_pct"], f["collided_pct"],
           f["gaps_mean"] "/" f["num_gaps"]
  }
' "$LOG" | tee -a "$LOG"

echo "wrote $LOG"
