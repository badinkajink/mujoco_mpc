#!/usr/bin/env bash
# Map the slalom success rate over (avoidance weight x clearance margin), the
# two knobs that moved it from 0/50 to 20/50 across a dozen earlier ad-hoc
# sweeps.
#
# Those earlier sweeps could not be compared to each other: they were run with
# --planner_seed unset, so the sampling noise came from system entropy and the
# same configuration measured anywhere from 6/50 to 16/50 depending on when it
# ran. Every run here is seeded, and every cell is repeated over several seeds,
# so a table row means "this configuration, this seed" and can be reproduced
# exactly from the command recorded next to it.
#
# Outputs, all under a timestamped $OUT:
#   manifest.txt  host, commit, binary md5, working-tree state, the full grid
#   runs/*.log    raw stdout of every run, one file per cell x seed
#   results.csv   one row per run: every input knob and every output metric
#   commands.txt  the exact command line for each row of results.csv
#
# Usage: landscape_grid.sh [out_dir] [repeats]
set -u

BIN=${BIN:-./build/bin/corridor_benchmark}
OUT=${1:-renders/runs/$(date +%Y%m%dT%H%M%S)_landscape}
REPEATS=${2:-50}

# The grid. Weight is the Avoidance cost weight; margin is residual_Clearance,
# the metres added to each disk's radius before the avoidance cost starts
# charging. The half-gap between a disk and the wall is 0.25 m, so margins
# approaching that are expected to seal the gap rather than fence the disk.
WEIGHTS=${WEIGHTS:-"500 2000 8000 32000 128000"}
MARGINS=${MARGINS:-"0.04 0.08 0.12 0.16 0.20"}
SEEDS=${SEEDS:-"1000 2000 3000"}

TOTAL_TIME=${TOTAL_TIME:-12}
SPEED=${SPEED:-0.25}
SEED=${SEED:-1}          # initial-state perturbation; same starts everywhere
PLANNER=${PLANNER:-0}    # see PLANNER_NAMES below; 0 is predictive sampling

# Index -> name, mirroring corridor_benchmark.cc's kPlannerLabel. The manifest
# used to print "predictive sampling" unconditionally, which meant a sweep run
# with PLANNER=9 documented itself as a sweep of a planner it never ran. Read
# the name from the index instead.
PLANNER_NAMES=(PredictiveSampling Gradient iLQG iLQS RobustSampling \
               CrossEntropy SampleGradient PSO AnnealedSampling RandomSampling)
PLANNER_NAME=${PLANNER_NAMES[$PLANNER]:-unknown}

# Concurrency. Outcomes do not depend on it -- the planner gets a fixed number
# of iterations per control step, not a wall-clock budget -- but ms/iter does,
# so timing from this sweep is contended and must not be quoted. Measure speed
# with machine_bench.sh on an idle box instead.
JOBS=${JOBS:-5}
THREADS=${THREADS:-4}

mkdir -p "$OUT/runs"
MANIFEST="$OUT/manifest.txt"
CSV="$OUT/results.csv"
CMDS="$OUT/commands.txt"

{
  echo "=== landscape grid ==="
  echo "date        : $(date -Is)"
  echo "host        : $(uname -sr) $(uname -m)"
  echo "cpu         : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"
  echo "cores       : $(nproc) logical"
  echo "load at start: $(uptime | sed 's/.*load average/load average/')"
  echo "commit      : $(git rev-parse HEAD)"
  echo "commit short: $(git rev-parse --short HEAD)"
  echo "tree        : $(git status --porcelain | grep -c . ) modified/untracked paths"
  echo "binary      : $BIN"
  echo "binary md5  : $(md5sum "$BIN" | cut -d' ' -f1)"
  echo "binary mtime: $(date -Is -r "$BIN")"
  echo
  echo "task        : slalom, stage=corridor, planner=$PLANNER ($PLANNER_NAME)"
  echo "fixed       : total_time=${TOTAL_TIME}s speed=$SPEED repeats=$REPEATS seed=$SEED"
  echo "              early_exit=true (verdict is monotone, so it costs no information)"
  echo "varied      : weights = $WEIGHTS"
  echo "              margins = $MARGINS"
  echo "              planner_seeds = $SEEDS"
  echo "concurrency : $JOBS jobs x $THREADS planner threads"
  echo "              ms/iter in results.csv is CONTENDED; do not quote it as speed"
  echo
  echo "weights vector is Cart,Upright,Velocity,Control,Avoidance = 1,0,0.1,0.01,<w>"
  echo "Upright is 0, so the pendulum's state is not scored; the goal set is"
  echo "the cart within 0.30 m of x=11.0."
} > "$MANIFEST"
cat "$MANIFEST"

echo "label,weight,margin,planner_seed,trials,solved,solved_pct,solved_se,collided,collided_pct,gaps_mean,t_solve_median,ms_per_iter_contended,plan_iters,wall_s" > "$CSV"
: > "$CMDS"

run_cell() {
  local w=$1 m=$2 ps=$3
  local label
  label=$(printf "w%06d_m%03d_s%d" "$w" "$(python3 -c "print(round($m*100))")" "$ps")
  local log="$OUT/runs/$label.log"
  "$BIN" --task=slalom --planner="$PLANNER" --stage=corridor \
       --weights="1,0,0.1,0.01,$w" --clearance="$m" \
       --speed="$SPEED" --total_time="$TOTAL_TIME" --repeats="$REPEATS" \
       --seed="$SEED" --planner_seed="$ps" --per_run=false \
       --planner_thread="$THREADS" --label="$label" > "$log" 2>&1
  echo "$label|$BIN --task=slalom --planner=$PLANNER --stage=corridor --weights=1,0,0.1,0.01,$w --clearance=$m --speed=$SPEED --total_time=$TOTAL_TIME --repeats=$REPEATS --seed=$SEED --planner_seed=$ps --per_run=false --planner_thread=$THREADS" >> "$CMDS"
  echo "done $label"
}
export -f run_cell
export BIN OUT PLANNER SPEED TOTAL_TIME REPEATS SEED THREADS CMDS

# Build the job list, then feed it to a fixed pool of workers.
JOBLIST="$OUT/joblist.txt"
: > "$JOBLIST"
for w in $WEIGHTS; do
  for m in $MARGINS; do
    for ps in $SEEDS; do
      echo "$w $m $ps" >> "$JOBLIST"
    done
  done
done
echo "queued $(wc -l < "$JOBLIST") runs, $JOBS at a time"

xargs -a "$JOBLIST" -P "$JOBS" -L1 bash -c 'run_cell $0 $1 $2'

# Collect. Parsing the RESULT line rather than the human-readable summary keeps
# the table and the logs from being able to disagree.
python3 - "$OUT" "$CSV" <<'PY'
import glob, os, re, sys
out, csv_path = sys.argv[1], sys.argv[2]
rows = []
for p in sorted(glob.glob(os.path.join(out, "runs", "*.log"))):
    txt = open(p).read()
    m = re.search(r"^RESULT (.*)$", txt, re.M)
    if not m:
        print(f"  no RESULT line in {p}")
        continue
    f = dict(kv.split("=", 1) for kv in m.group(1).split())
    label = f["planner"]
    lm = re.match(r"w(\d+)_m(\d+)_s(\d+)", label)
    w, mm, ps = int(lm.group(1)), int(lm.group(2)) / 100.0, int(lm.group(3))
    pct, se = f["solved_pct"].split("+-")
    rows.append((label, w, mm, ps, f["trials"], f["solved"], pct, se,
                 f["collided"], f["collided_pct"], f["gaps_mean"],
                 f["t_solve_median"], f["ms_per_iter"], f["plan_iters"],
                 f["wall_total_s"]))
with open(csv_path, "a") as fh:
    for r in sorted(rows, key=lambda r: (r[1], r[2], r[3])):
        fh.write(",".join(str(x) for x in r) + "\n")
print(f"wrote {len(rows)} rows to {csv_path}")
PY

echo "=== $OUT ==="
column -s, -t < "$CSV"
