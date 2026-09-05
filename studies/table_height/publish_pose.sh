#!/usr/bin/env bash
# Aggregate whatever brace-posture runs exist and rebuild the page. Idempotent:
# every arm is optional, and a missing one drops out of the page rather than
# failing the build.
set -u
cd /home/humanoid/Programs/Humanoid_Simulation/mujoco_mpc
S=studies/table_height
PAGE=docs/lean/20260905-brace_posture_retarget.html
MEDIA=docs/lean/media/pose
mkdir -p $MEDIA

[ -s $S/runs/ab/off/summary.csv ] && $S/analyze.py --runs $S/runs/ab/off --out $S/figs_ab_off
[ -s $S/runs/ab/on/summary.csv ] && $S/analyze.py --runs $S/runs/ab/on --out $S/figs_ab_on \
    --overlay $S/figs_ab_off --overlay_label "brace_pose_track off"

LEAD_ARGS=""
for ARM in off on; do
  if [ -s $S/runs/lead/$ARM/summary.csv ]; then
    $S/analyze.py --runs $S/runs/lead/$ARM --out $S/figs_lead_$ARM || true
    LEAD_ARGS="$LEAD_ARGS --lead_$ARM $S/figs_lead_$ARM"
  fi
done

$S/analyze_pose.py --baseline $S/runs/seeded --out $S/figs || true
RUNS="seeded=$S/runs/seeded"
for D in ab/off ab/on lead/off lead/on; do
  [ -s $S/runs/$D/summary.csv ] && RUNS="$RUNS $(echo $D | tr / -)=$S/runs/$D"
done
$S/analyze_lead.py --runs $RUNS --out $S/figs || true
$S/render_pose.py --out $S/figs/fig_poses.png || true

for H in 0785 0885 0985 1035 1085; do
  Q=$S/runs/ab/on/h${H}_s0.qpos.csv
  [ -f "$Q" ] || continue
  [ -f "$MEDIA/on_h${H}_s0.mp4" ] && continue
  FACE=$(python3 -c "print('%.3f'%(${H}/1000.0))")
  $S/render_video.py --qpos $Q --states $S/runs/ab/on/h${H}_s0.csv \
      --table_h $FACE --out $MEDIA/on_h${H}_s0.mp4 || true
done

cp $S/figs/fig_keyframe*.png $S/figs/fig_pose.png $S/figs/fig_pose.dark.png \
   $S/figs/fig_mode*.png $S/figs/fig_armreach*.png $S/figs/fig_poses*.png \
   $S/figs/fig_lead*.png $MEDIA/ 2>/dev/null
[ -f $S/figs_ab_on/fig_bench.png ] && cp $S/figs_ab_on/fig_bench*.png \
   $S/figs_ab_on/fig_pad*.png $S/figs_ab_on/fig_balance*.png $MEDIA/ 2>/dev/null

$S/make_page_pose.py --figs $S/figs \
   $( [ -f $S/figs_ab_off/agg.json ] && echo "--off $S/figs_ab_off" ) \
   $( [ -f $S/figs_ab_on/agg.json ] && echo "--on $S/figs_ab_on" ) \
   $LEAD_ARGS --figs_rel media/pose --media $MEDIA --out $PAGE
echo "=== PUBLISHED $(date +%H:%M) -> $PAGE ==="
