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
for SPEC in ab-off:ab/off ab-on:ab/on lead-off:lead/off lead-on:lead/on \
            pose2:mode/mode2 gate-open:tol; do
  L=${SPEC%%:*}; D=${SPEC#*:}
  [ -s $S/runs/$D/summary.csv ] && RUNS="$RUNS $L=$S/runs/$D"
done
$S/analyze_lead.py --runs $RUNS --out $S/figs || true
$S/analyze_gate.py --runs $RUNS --out $S/figs || true
[ -s $S/runs/mode/mode2/summary.csv ] && $S/analyze.py --runs $S/runs/mode/mode2 --out $S/figs_mode2 || true
[ -s $S/runs/tol/summary.csv ] && $S/analyze.py --runs $S/runs/tol --out $S/figs_tol || true
$S/render_pose.py --out $S/figs/fig_poses.png || true

for ARM in on off; do
  for H in 0785 0885 0985 1035 1085; do
    Q=$S/runs/ab/$ARM/h${H}_s0.qpos.csv
    [ -f "$Q" ] || continue
    [ -f "$MEDIA/${ARM}_h${H}_s0.mp4" ] && continue
    FACE=$(python3 -c "print('%.3f' % (int('${H}') / 1000.0))")
    nice -n 19 $S/render_video.py --qpos $Q --states $S/runs/ab/$ARM/h${H}_s0.csv \
        --table_h $FACE --out $MEDIA/${ARM}_h${H}_s0.mp4 || true
  done
done

cp $S/figs/fig_keyframe*.png $S/figs/fig_pose.png $S/figs/fig_pose.dark.png \
   $S/figs/fig_mode*.png $S/figs/fig_armreach*.png $S/figs/fig_poses*.png \
   $S/figs/fig_lead*.png $S/figs/fig_gate*.png $MEDIA/ 2>/dev/null
[ -f $S/figs_ab_on/fig_bench.png ] && cp $S/figs_ab_on/fig_bench*.png \
   $S/figs_ab_on/fig_pad*.png $S/figs_ab_on/fig_balance*.png $MEDIA/ 2>/dev/null

$S/make_page_pose.py --figs $S/figs \
   $( [ -f $S/figs_ab_off/agg.json ] && echo "--off $S/figs_ab_off" ) \
   $( [ -f $S/figs_ab_on/agg.json ] && echo "--on $S/figs_ab_on" ) \
   $( [ -f $S/figs_mode2/agg.json ] && echo "--mode2 $S/figs_mode2" ) \
   $LEAD_ARGS --figs_rel media/pose --media $MEDIA --out $PAGE
ARMS="shipped=$S/figs_ab_off pose_track1=$S/figs_ab_on"
[ -f $S/figs_lead_off/agg.json ] && ARMS="$ARMS lead=$S/figs_lead_off"
[ -f $S/figs_lead_on/agg.json ] && ARMS="$ARMS lead+pose=$S/figs_lead_on"
[ -f $S/figs_mode2/agg.json ] && ARMS="$ARMS pose_track2=$S/figs_mode2"
[ -f $S/figs_tol/agg.json ] && ARMS="$ARMS gate_opened=$S/figs_tol"
$S/write_status_pose.py --figs $S/figs --arms $ARMS --out $S/STATUS.md
echo "=== PUBLISHED $(date +%H:%M) -> $PAGE ==="
