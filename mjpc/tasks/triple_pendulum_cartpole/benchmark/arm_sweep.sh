#!/usr/bin/env bash
# Run a list of named configurations ("arms") x seeds, one RESULT row each.
#
# landscape_grid.sh sweeps a rectangle: every weight against every margin. Some
# questions are not rectangles. "Does the sliding plan help at the three best
# ridge cells" is three (weight, margin) pairs picked out of a grid, crossed
# with one binary; "does raising the rollout count help" holds the cell fixed
# and varies a knob the grid script does not know about. Expressing either as a
# rectangle either runs cells nobody asked for or hides the varied knob inside
# the label. So arms are listed explicitly here, one per line, and the knob
# that distinguishes them is passed through as flags.
#
# The arms file: one arm per line, blank lines and #-comments ignored.
#
#     <arm_name> <weight> <margin> [extra flags ...]
#
# e.g.
#     nt10   8192000 0.01 --num_trajectory=10
#     slide1 2048000 0.02 --sliding_plan=1
#
# Every arm is run at every seed in $SEEDS, so a row is (arm, seed) and the
# per-seed spread is recoverable -- pooling three seeds into one binomial
# understates it (see RELIABILITY_PLAN.md section 1).
#
# Outputs, all under a timestamped $OUT:
#   manifest.txt  host, load, commit, binary md5, the arms, the fixed settings
#   arms.txt      a copy of the arms file as given, so the run is self-contained
#   runs/*.log    raw stdout, one file per arm x seed
#   results.csv   one row per run, parsed from the RESULT line
#   commands.txt  the exact command line for each row
#
# Usage: arm_sweep.sh <out_dir> <repeats> <arms_file>
set -u

BIN=${BIN:-./build/bin/corridor_benchmark}
OUT=${1:?out_dir required}
REPEATS=${2:-50}
ARMS_FILE=${3:?arms_file required}

SEEDS=${SEEDS:-"1000 2000 3000"}
TOTAL_TIME=${TOTAL_TIME:-12}
SPEED=${SPEED:-0.25}
SEED=${SEED:-1}          # initial-state perturbation; same starts everywhere
PLANNER=${PLANNER:-0}

PLANNER_NAMES=(PredictiveSampling Gradient iLQG iLQS RobustSampling \
               CrossEntropy SampleGradient PSO AnnealedSampling RandomSampling)
PLANNER_NAME=${PLANNER_NAMES[$PLANNER]:-unknown}

# Outcomes do not depend on concurrency -- the planner gets a fixed iteration
# count per control step, not a wall-clock budget -- but ms/iter does. Timing
# from this sweep is contended and must not be quoted as planner speed.
JOBS=${JOBS:-5}
THREADS=${THREADS:-4}

if [[ -e "$OUT" ]]; then
  echo "refusing to write into existing $OUT" >&2
  exit 1
fi
mkdir -p "$OUT/runs"
MANIFEST="$OUT/manifest.txt"
CSV="$OUT/results.csv"
CMDS="$OUT/commands.txt"
cp "$ARMS_FILE" "$OUT/arms.txt"

{
  echo "=== arm sweep ==="
  echo "date        : $(date -Is)"
  echo "host        : $(uname -sr) $(uname -m)"
  echo "cpu         : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"
  echo "cores       : $(nproc) logical"
  echo "load at start: $(uptime | sed 's/.*load average/load average/')"
  echo "foreign jobs : $(ps -eo comm --no-headers | grep -c '^corridor_benchm') corridor_benchmark process(es) already running"
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
  echo "              iterations/control step = 1/speed = $(python3 -c "print(1.0/$SPEED)"), held across arms"
  echo "varied      : the arms below, x planner_seeds = $SEEDS"
  echo "concurrency : $JOBS jobs x $THREADS planner threads"
  echo "              ms/iter in results.csv is CONTENDED; do not quote it as speed"
  echo
  echo "arms (name weight margin flags):"
  grep -v '^\s*\(#\|$\)' "$ARMS_FILE" | sed 's/^/  /'
  echo
  echo "weights vector is Cart,Upright,Velocity,Control,Avoidance = 1,0,0.1,0.01,<weight>"
  echo "Upright is 0, so the pendulum's state is not scored; the goal set is"
  echo "the cart within 0.30 m of x=11.0."
  echo
  echo "num_trajectory and sliding_plan on each RESULT line are read back off"
  echo "the model after the overrides were applied, not echoed from the flags,"
  echo "so a row cannot claim a setting the planner did not receive."
} > "$MANIFEST"
cat "$MANIFEST"

echo "label,arm,weight,margin,planner_seed,num_trajectory,sliding_plan,trials,solved,solved_pct,solved_se,collided,collided_pct,gaps_mean,t_solve_median,ms_per_iter_contended,ms_per_iter_p95_contended,plan_iters,wall_s" > "$CSV"
: > "$CMDS"

run_cell() {
  local arm=$1 w=$2 m=$3 ps=$4; shift 4
  local label="${arm}_s${ps}"
  local log="$OUT/runs/$label.log"
  "$BIN" --task=slalom --planner="$PLANNER" --stage=corridor \
       --weights="1,0,0.1,0.01,$w" --clearance="$m" \
       --speed="$SPEED" --total_time="$TOTAL_TIME" --repeats="$REPEATS" \
       --seed="$SEED" --planner_seed="$ps" --per_run=false \
       --planner_thread="$THREADS" --label="$label" "$@" > "$log" 2>&1
  echo "$label|$BIN --task=slalom --planner=$PLANNER --stage=corridor --weights=1,0,0.1,0.01,$w --clearance=$m --speed=$SPEED --total_time=$TOTAL_TIME --repeats=$REPEATS --seed=$SEED --planner_seed=$ps --per_run=false --planner_thread=$THREADS $*" >> "$CMDS"
  echo "done $label"
}
export -f run_cell
export BIN OUT PLANNER SPEED TOTAL_TIME REPEATS SEED THREADS CMDS

JOBLIST="$OUT/joblist.txt"
: > "$JOBLIST"
while read -r arm w m rest; do
  [[ -z "${arm:-}" || "$arm" == \#* ]] && continue
  for ps in $SEEDS; do
    echo "$arm $w $m $ps $rest" >> "$JOBLIST"
  done
done < "$ARMS_FILE"
echo "queued $(wc -l < "$JOBLIST") runs, $JOBS at a time"

# Longest arms first, so the pool does not finish with one 80-rollout cell
# running alone while four workers idle. Ordering does not affect outcomes.
xargs -a "$JOBLIST" -P "$JOBS" -L1 bash -c 'run_cell "$@"' _

python3 - "$OUT" "$CSV" <<'PY'
import glob, os, re, sys
out, csv_path = sys.argv[1], sys.argv[2]

# Recover (arm, weight, margin) from the command line rather than from the
# label: the label is chosen by the caller and can say anything, the command is
# what ran.
cmds = {}
with open(os.path.join(out, "commands.txt")) as fh:
    for line in fh:
        label, cmd = line.rstrip("\n").split("|", 1)
        cmds[label] = cmd

rows = []
for p in sorted(glob.glob(os.path.join(out, "runs", "*.log"))):
    txt = open(p).read()
    m = re.search(r"^RESULT (.*)$", txt, re.M)
    if not m:
        print(f"  no RESULT line in {p}")
        continue
    f = dict(kv.split("=", 1) for kv in m.group(1).split())
    label = f["planner"]
    cmd = cmds.get(label, "")
    w = re.search(r"--weights=[\d.,]*?,([\d.]+)\s", cmd + " ")
    weight = w.group(1) if w else ""
    arm = label.rsplit("_s", 1)[0]
    pct, se = f["solved_pct"].split("+-")
    rows.append((label, arm, weight, f["clearance"], f["planner_seed"],
                 f["num_trajectory"], f["sliding_plan"],
                 f["trials"], f["solved"], pct, se,
                 f["collided"], f["collided_pct"], f["gaps_mean"],
                 f["t_solve_median"], f["ms_per_iter"], f["ms_per_iter_p95"],
                 f["plan_iters"], f["wall_total_s"]))
with open(csv_path, "a") as fh:
    for r in sorted(rows, key=lambda r: (r[1], int(r[4]))):
        fh.write(",".join(str(x) for x in r) + "\n")
print(f"wrote {len(rows)} rows to {csv_path}")
PY

echo "=== $OUT ==="
column -s, -t < "$CSV"
