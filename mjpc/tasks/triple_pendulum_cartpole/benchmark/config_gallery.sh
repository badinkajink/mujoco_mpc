#!/usr/bin/env bash
# One pass over the weight/planner configurations that matter, producing for
# each: a benchmark result line, dumped rollouts, a filmstrip + MP4 of the best
# rollout it produced, and a cost-vs-control characterization.
#
# This exists because the interesting configurations were found by hand across
# a dozen separate sweeps whose only record was a scrollback buffer. Everything
# here lands on disk under $OUT, and the run ends by writing $OUT/SUMMARY.md --
# so the results can be read once, at the end, instead of watched.
#
# Usage: config_gallery.sh [out_dir] [repeats]
set -u

BIN=${BIN:-./build/bin/corridor_benchmark}
OUT=${1:-renders/config_gallery}
REPEATS=${2:-50}
DUMPS=${DUMPS:-8}
TOTAL_TIME=${TOTAL_TIME:-12}
SPEED=${SPEED:-0.25}
SEED=${SEED:-1}
THREADS=${THREADS:-15}
BENCH=$(dirname "$0")

mkdir -p "$OUT"
LOG="$OUT/gallery.log"
SUM="$OUT/cost_control_summary.csv"
: > "$LOG"; rm -f "$SUM"

echo "binary=$(md5sum "$BIN" | cut -d' ' -f1) commit=$(git rev-parse --short HEAD) date=$(date -Is)" | tee -a "$LOG"
echo "repeats=$REPEATS speed=$SPEED total_time=$TOTAL_TIME seed=$SEED" | tee -a "$LOG"

# label|planner_idx|weights|margin|extra flags
CONFIGS=${CONFIGS:-"
ps_w500_m008|0|1,0,0.1,0.01,500|0.08|
ps_w8000_m008|0|1,0,0.1,0.01,8000|0.08|
ps_w8000_m020|0|1,0,0.1,0.01,8000|0.20|
ps_w32000_m020|0|1,0,0.1,0.01,32000|0.20|
rs_w8000_m020|9|1,0,0.1,0.01,8000|0.20|
pso_w8000_m020|7|1,0,0.1,0.01,8000|0.20|
anneal_h1_w500|8|1,0,0.1,0.01,500|0.08|--horizon=1.0 --spline_points=12
anneal_h2_w500|8|1,0,0.1,0.01,500|0.08|--horizon=2.0 --spline_points=24
anneal_h2_w8000_m020|8|1,0,0.1,0.01,8000|0.20|--horizon=2.0 --spline_points=24
"}

while IFS='|' read -r label idx weights margin extra; do
  [ -z "${label:-}" ] && continue
  echo "=== $label ===" | tee -a "$LOG"
  D="$OUT/$label"; mkdir -p "$D"
  # Two passes, because the two products want opposite settings. The scored
  # pass uses --early_exit (the verdict is monotone, so it costs no information
  # and is several times cheaper). The dump pass runs a handful of trials to the
  # end, because a rollout cut at its first contact cannot show what the planner
  # did afterwards, and the renders are the point of this script.
  # shellcheck disable=SC2086
  $BIN --task=slalom --planner="$idx" --stage=corridor --weights="$weights" \
       --clearance="$margin" --speed="$SPEED" --total_time="$TOTAL_TIME" \
       --repeats="$REPEATS" --seed="$SEED" --per_run=false \
       --planner_thread="$THREADS" --label="$label" \
       $extra 2>&1 | tee -a "$LOG"
  # shellcheck disable=SC2086
  $BIN --task=slalom --planner="$idx" --stage=corridor --weights="$weights" \
       --clearance="$margin" --speed="$SPEED" --total_time="$TOTAL_TIME" \
       --repeats="$DUMPS" --seed="$SEED" --per_run=false \
       --planner_thread="$THREADS" --early_exit=false \
       --dump="$D/r.csv" --dump_runs="$DUMPS" --label="${label}_dumps" \
       $extra >> "$LOG" 2>&1

  # Rank the dumped rollouts by how far the cart got before its first overlap,
  # and render the best one. Picking by outcome rather than by run index is the
  # difference between a gallery and a screenshot of run 0.
  best=$(python3 - "$D" <<'PY'
import csv, glob, sys
best, best_key = None, (-1, -1)
for p in sorted(glob.glob(sys.argv[1] + "/r_*.csv")):
    with open(p) as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith("#")))
    if not rows: continue
    reach, clean = 0.0, True
    for r in rows:
        if float(r["min_clearance"]) < 0: clean = False; break
        reach = float(r["cart"])
    key = (1 if clean else 0, reach)
    if key > best_key: best_key, best = key, p
print(best or "")
PY
)
  echo "  best rollout: ${best:-none}" | tee -a "$LOG"
  [ -z "$best" ] && continue

  MUJOCO_GL=egl python3 "$BENCH/filmstrip.py" --dump "$best" \
      --out "$D/$label.png" --video "$D/$label.mp4" --track \
      >> "$LOG" 2>&1 || echo "  filmstrip failed" | tee -a "$LOG"

  for f in "$D"/r_*.csv; do
    python3 "$BENCH/cost_control.py" --dump "$f" --weights "$weights" \
        --margin "$margin" --out "${f%.csv}_cc" --summary "$SUM" \
        --label "$label" >> "$LOG" 2>&1 || true
  done
done <<< "$CONFIGS"

{
  echo "# Slalom configuration gallery"
  echo
  echo "\`$(basename "$0") $OUT $REPEATS\` — commit \`$(git rev-parse --short HEAD)\`, $(date -Is)"
  echo
  echo "Every row is $REPEATS trials on \`slalom.xml\`, scored with --early_exit"
  echo "(the verdict is monotone, so it costs no information). The renders and"
  echo "cost/control panels come from a separate $DUMPS-trial pass run to the end."
  echo "\`gaps\` is partial credit: bottlenecks cleared before the first contact,"
  echo "out of 3. Run-to-run spread on \`gaps\` is about +/-0.15 at n=50, so"
  echo "differences smaller than that are not findings."
  echo
  echo "| config | solved | gaps | collided | ms/iter | p95 |"
  echo "|---|---|---|---|---|---|"
  grep -h '^RESULT' "$LOG" | grep -v '_dumps ' | awk '{
    delete f; for (i=2;i<=NF;i++){split($i,kv,"="); f[kv[1]]=kv[2]}
    printf "| %s | %s/%s | %s | %s%% | %s | %s |\n", f["planner"], f["solved"],
           f["trials"], f["gaps_mean"], f["collided_pct"], f["ms_per_iter"],
           f["ms_per_iter_p95"]
  }'
  echo
  echo "Renders: \`$OUT/<config>/<config>.png\` (filmstrip) and \`.mp4\`."
  echo "Cost-vs-control panels: \`$OUT/<config>/r_N_cc.png\`, per-step term"
  echo "breakdown in the matching \`.csv\`, and one row per rollout in"
  echo "\`$(basename "$SUM")\`."
  echo
  if [ -f "$SUM" ]; then
    echo "## Control character per configuration"
    echo
    echo "Means over the dumped rollouts of each config."
    echo
    echo "| config | saturated % | mean abs u | effort int u^2 | reversals/s | cost share: Cart | Avoidance |"
    echo "|---|---|---|---|---|---|---|"
    python3 - "$SUM" <<'PY'
import csv, sys, collections, statistics as st
rows = list(csv.DictReader(open(sys.argv[1])))
by = collections.defaultdict(list)
for r in rows: by[r["label"]].append(r)
for k, v in by.items():
    m = lambda c: st.mean(float(x[c]) for x in v)
    print(f"| {k} | {m('saturated_pct'):.0f} | {m('mean_abs_u'):.2f} | "
          f"{m('effort_int_u2'):.0f} | {m('reversals_per_s'):.1f} | "
          f"{m('share_Cart'):.0f}% | {m('share_Avoidance'):.0f}% |")
PY
  fi
} > "$OUT/SUMMARY.md"

echo "wrote $OUT/SUMMARY.md"
