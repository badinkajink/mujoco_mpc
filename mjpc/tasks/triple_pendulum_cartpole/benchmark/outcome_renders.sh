#!/usr/bin/env bash
# Render one video per *outcome* per configuration, not one per configuration.
#
# The earlier gallery rendered a single "best" rollout for each config, which
# meant the configurations that actually solved the slalom had no footage of a
# solve -- the picker ranked by distance reached before first contact, and for
# a config that solves 40% of the time that is not the same thing. Worse, a
# config whose failure is "never enters the gap" and one whose failure is
# "enters and clips the disk" produced pictures that look alike at a glance.
#
# So: dump a batch of rollouts, classify each one by what happened, and render
# a representative of each class that occurred.
#
#   solved    cart reached the goal band and never overlapped a disk
#   collided  clearance went negative at some point
#   stalled   neither -- ran out of time still short of the goal
#
# Usage: outcome_renders.sh [out_dir] [dumps_per_config]
set -u

BIN=${BIN:-./build/bin/corridor_benchmark}
OUT=${1:-renders/runs/$(date +%Y%m%dT%H%M%S)_outcomes}
DUMPS=${2:-16}
BENCH=$(dirname "$0")

TOTAL_TIME=${TOTAL_TIME:-12}
SPEED=${SPEED:-0.25}
SEED=${SEED:-1}
PLANNER_SEED=${PLANNER_SEED:-1000}
THREADS=${THREADS:-2}
GOAL_X=${GOAL_X:-10.70}     # goal 11.0 m, tolerance 0.30 m

# label|weight|margin -- chosen to span the landscape's regimes, not to
# flatter it: a strong solver, a middling one, a config that collides because
# the barrier is too weak, and one that never enters the gap because the
# barrier has sealed it.
CONFIGS=${CONFIGS:-"
w32000_m008|32000|0.08
w8000_m008|8000|0.08
w500_m008|500|0.08
w8000_m020|8000|0.20
"}

mkdir -p "$OUT"
LOG="$OUT/renders.log"
INDEX="$OUT/index.csv"
: > "$LOG"
echo "config,weight,margin,run,outcome,max_cart,min_clearance,png,mp4" > "$INDEX"

{
  echo "date       : $(date -Is)"
  echo "commit     : $(git rev-parse --short HEAD)"
  echo "binary md5 : $(md5sum "$BIN" | cut -d' ' -f1)"
  echo "dumps      : $DUMPS per config, planner_seed=$PLANNER_SEED, early_exit=false"
  echo "goal band  : cart >= $GOAL_X"
} | tee -a "$LOG"

while IFS='|' read -r label w m; do
  [ -z "${label:-}" ] && continue
  echo "=== $label ===" | tee -a "$LOG"
  D="$OUT/$label"; mkdir -p "$D"

  # Rollouts must run to the end, not stop at first contact: a run cut at its
  # first overlap cannot show what the planner did next, and "what it did next"
  # is the whole difference between the failure modes.
  "$BIN" --task=slalom --planner=0 --stage=corridor \
       --weights="1,0,0.1,0.01,$w" --clearance="$m" \
       --speed="$SPEED" --total_time="$TOTAL_TIME" --repeats="$DUMPS" \
       --seed="$SEED" --planner_seed="$PLANNER_SEED" --per_run=false \
       --planner_thread="$THREADS" --early_exit=false \
       --dump="$D/r.csv" --dump_runs="$DUMPS" --label="${label}_dumps" \
       >> "$LOG" 2>&1

  # Classify every dumped rollout, then name one representative per class.
  python3 - "$D" "$GOAL_X" > "$D/picks.txt" <<'PY'
import csv, glob, sys
d, goal = sys.argv[1], float(sys.argv[2])
best = {}
for p in sorted(glob.glob(d + "/r_*.csv")):
    with open(p) as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith("#")))
    if not rows:
        continue
    max_cart = max(float(r["cart"]) for r in rows)
    min_clr = min(float(r["min_clearance"]) for r in rows)
    if min_clr < 0:
        outcome = "collided"
    elif max_cart >= goal:
        outcome = "solved"
    else:
        outcome = "stalled"
    # Within a class keep the most representative: for solved, the earliest to
    # get there is the cleanest to watch; for the failures, the one that got
    # furthest shows the most of the attempt.
    key = -max_cart if outcome == "solved" else max_cart
    if outcome not in best or key > best[outcome][0]:
        best[outcome] = (key, p, max_cart, min_clr)
for outcome, (_, p, max_cart, min_clr) in best.items():
    print(f"{outcome} {p} {max_cart:.3f} {min_clr:.4f}")
PY
  cat "$D/picks.txt" | tee -a "$LOG"

  while read -r outcome path max_cart min_clr; do
    [ -z "${outcome:-}" ] && continue
    run=$(basename "$path" .csv)
    png="$D/${label}_${outcome}.png"; mp4="$D/${label}_${outcome}.mp4"
    MUJOCO_GL=egl python3 "$BENCH/filmstrip.py" --dump "$path" \
        --out "$png" --video "$mp4" --track >> "$LOG" 2>&1 \
      && echo "$label,$w,$m,$run,$outcome,$max_cart,$min_clr,$png,$mp4" >> "$INDEX" \
      || echo "  filmstrip failed for $label/$outcome" | tee -a "$LOG"
  done < "$D/picks.txt"
done <<< "$CONFIGS"

echo "=== $INDEX ==="
column -s, -t < "$INDEX"
