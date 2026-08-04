#!/usr/bin/env bash
# Planning-speed characterization of the MACHINE, for comparing hosts.
#
# timing_bench.sh answers "which planner costs more"; this answers "how fast is
# this box", which needs two different numbers:
#
#   1. Single-thread rollout throughput -- per-core speed, the thing that does
#      not improve by buying more cores. This is the number to compare across
#      CPUs with different core counts.
#   2. Thread scaling -- how much of the extra cores the planner actually
#      converts into iterations. MJPC parallelizes over rollouts, so a planner
#      with 10 trajectories cannot use more than 10 threads no matter what the
#      machine has.
#
# The portable unit is mj_step throughput: trajectories x horizon_steps per
# iteration, divided by the time an iteration took. That is independent of
# planner, horizon and knot count, so a row here is comparable to a row taken on
# any other host with any other settings.
#
# Every run is --early_exit=false with a fixed seed and a fixed iteration count,
# so two machines execute the identical workload rather than a similar one.
#
# Usage: machine_bench.sh [out_dir] [threads...]
set -u

BIN=${BIN:-./build/bin/corridor_benchmark}
OUT=${1:-renders/machine}
shift 2>/dev/null || true
THREADS=("$@")
[ ${#THREADS[@]} -eq 0 ] && THREADS=(1 2 4 8 12 16)

PLANNER=${PLANNER:-0}          # Predictive Sampling: 10 rollouts, no annealing
TOTAL_TIME=${TOTAL_TIME:-2.0}
REPEATS=${REPEATS:-5}
HORIZON=${HORIZON:-1.0}
KNOTS=${KNOTS:-12}
TRAJ=${TRAJ:-10}               # sampling_trajectories in slalom.xml

mkdir -p "$OUT"
LOG="$OUT/machine.log"
: > "$LOG"

{
  echo "=== host ==="
  echo "date        : $(date -Is)"
  echo "uname       : $(uname -srm)"
  if [ -r /proc/cpuinfo ]; then
    echo "cpu         : $(awk -F': ' '/model name/{print $2; exit}' /proc/cpuinfo)"
    echo "cores       : $(nproc) logical, $(lscpu 2>/dev/null | awk -F: '/^Core\(s\) per socket/{gsub(/ /,"",$2); c=$2} /^Socket\(s\)/{gsub(/ /,"",$2); s=$2} END{print (c&&s)?c*s:"?"}') physical"
    echo "mem         : $(awk '/MemTotal/{printf "%.1f GB", $2/1048576}' /proc/meminfo)"
  else
    echo "cpu         : $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
    echo "cores       : $(sysctl -n hw.ncpu 2>/dev/null || echo ?) logical"
    echo "mem         : $(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.1f GB", $1/1073741824}')"
  fi
  echo "load        : $(uptime | sed 's/.*load average/load average/')"
  echo "binary md5  : $(md5sum "$BIN" 2>/dev/null | cut -d' ' -f1 || md5 -q "$BIN")"
  echo "commit      : $(git rev-parse --short HEAD)"
  echo "workload    : planner=$PLANNER horizon=${HORIZON}s knots=$KNOTS trajectories=$TRAJ"
  echo "              total_time=${TOTAL_TIME}s repeats=$REPEATS early_exit=false"
  echo
} | tee -a "$LOG"

echo "run with nothing else on the machine; this is the one measurement CPU contention corrupts." | tee -a "$LOG"
echo | tee -a "$LOG"

for t in "${THREADS[@]}"; do
  echo "--- threads=$t ---" >> "$LOG"
  $BIN --task=slalom --planner="$PLANNER" --stage=corridor \
       --weights=1,0,0.1,0.01,500 --speed=0.25 --total_time="$TOTAL_TIME" \
       --repeats="$REPEATS" --seed=1 --per_run=false --early_exit=false \
       --planner_thread="$t" --horizon="$HORIZON" --spline_points="$KNOTS" \
       --num_trajectory="$TRAJ" \
       --label="threads_$t" 2>&1 | grep -E "^timing:|^RESULT" >> "$LOG"
done

{
  echo "=== results ==="
  echo
  printf "%8s %12s %10s %10s %16s %10s\n" \
         "threads" "ms/iter" "p95" "iters" "Msteps/s" "speedup"
  awk -v hor="$HORIZON" '
    /^--- threads=/ { split($0, a, "="); sub(/ ---/, "", a[2]); t = a[2] }
    /^timing:/ { ms = $2; p95 = $8; sub(/,/, "", p95) }
    /^RESULT/ {
      delete f; for (i = 2; i <= NF; i++) { split($i, kv, "="); f[kv[1]] = kv[2] }
      steps = hor / 0.005 + 1
      # Rollout count comes off the RESULT line, which the binary reads back
      # from the model after the override was applied -- not from $TRAJ. Those
      # disagreed until 2026-08-03: $TRAJ fed this arithmetic and the header
      # but was never passed to the binary, so TRAJ=40 reported 4x the true
      # throughput for a run that had used 10 rollouts.
      thr = f["num_trajectory"] * steps / (ms / 1000.0) / 1e6
      if (base == 0) base = thr
      printf "%8s %12.3f %10.3f %10s %16.2f %9.2fx\n",
             t, ms, p95, f["plan_iters"], thr, thr / base
    }
  ' "$LOG"
  echo
  echo "Msteps/s is mj_step calls per second across all planner threads"
  echo "($TRAJ rollouts x $(awk -v h="$HORIZON" 'BEGIN{print int(h/0.005)+1}') steps per iteration)."
  echo "Compare the threads=1 row across machines for per-core speed; compare the"
  echo "speedup column for how much of the extra cores each host converts."
} | tee -a "$LOG"

echo
echo "wrote $LOG"
