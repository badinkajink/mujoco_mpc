#!/usr/bin/env bash
# Rebuild the height-window gates page from whatever probes and arms exist.
# Idempotent: a missing arm drops out of the page rather than failing the build.
set -u
cd /home/humanoid/Programs/Humanoid_Simulation/mujoco_mpc
S=studies/table_height
PAGE=docs/lean/20260905-height_window_gates.html
MEDIA=docs/lean/media/gates
mkdir -p $MEDIA

$S/probe_pitch.py    --json $S/figs/pitch.json    > $S/figs/pitch.txt    2>&1 || true
$S/probe_basin.py    --json $S/figs/basin.json    > $S/figs/basin.txt    2>&1 || true
[ -s $S/figs/reachset.json ] || $S/probe_reachset.py --json $S/figs/reachset.json || true

RUNS="seeded=$S/runs/seeded"
for SPEC in ab-off:ab/off ab-on:ab/on gate-open:tol basin:basin/basin both:basin/both; do
  L=${SPEC%%:*}; D=${SPEC#*:}
  [ -s $S/runs/$D/summary.csv ] && RUNS="$RUNS $L=$S/runs/$D"
done
$S/analyze_gate.py  --runs $RUNS --out $S/figs || true
$S/analyze_gates.py --figs $S/figs --runs seeded=$S/runs/seeded || true

ARGS=""
for ARM in basin both; do
  if [ -s $S/runs/basin/$ARM/summary.csv ]; then
    $S/analyze.py --runs $S/runs/basin/$ARM --out $S/figs_$ARM || true
    [ -f $S/figs_$ARM/agg.json ] && ARGS="$ARGS --$ARM $S/figs_$ARM"
  fi
done

for H in 0885 0985 1085; do
  Q=$S/runs/basin/both/h${H}_s0.qpos.csv
  [ -f "$Q" ] || continue
  [ -f "$MEDIA/both_h${H}_s0.mp4" ] && continue
  FACE=$(python3 -c "print('%.3f' % (int('${H}') / 1000.0))")
  nice -n 19 $S/render_video.py --qpos $Q --states $S/runs/basin/both/h${H}_s0.csv \
      --table_h $FACE --out $MEDIA/both_h${H}_s0.mp4 || true
done

cp $S/figs/fig_pitchgate*.png $S/figs/fig_basinaim*.png $S/figs/fig_window*.png \
   $MEDIA/ 2>/dev/null
$S/make_page_gates.py --figs $S/figs --shipped $S/figs_ab_off $ARGS \
   --figs_rel media/gates --out $PAGE
echo "=== PUBLISHED $(date +%H:%M) -> $PAGE ==="
