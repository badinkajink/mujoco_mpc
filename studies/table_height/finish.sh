#!/usr/bin/env bash
# Unattended completion of the table-height study.
#   1. figures + summary from the seeded sweep
#   2. one video per height (seed 0)
#   3. analysis fragment + doc page
#   4. STATUS.md
#   5. supplementary sweep: the 75 s cap censored the stalled points, so re-run
#      the whole grid at 140 s, 2 seeds, and record what changed
# Each step is `|| true` so one failure cannot strand the rest.
set -u
cd /home/humanoid/Programs/Humanoid_Simulation/mujoco_mpc
S=studies/table_height
PAGE=docs/lean/20260904-table_height_generalization.html
export MPLBACKEND=Agg

echo "=== [1/5] analyze $(date +%H:%M) ==="
nice -n 15 python3 $S/analyze.py --runs $S/runs/seeded --out $S/figs || true

echo "=== [2/5] videos $(date +%H:%M) ==="
mkdir -p docs/lean/media/th
for H in 0.785 0.885 0.985 1.035 1.085; do
  TAG=$(python3 -c "print('h%04d_s0'%round($H*1000))")
  Q=$S/runs/seeded/$TAG.qpos.csv
  [ -f "$Q" ] || { echo "  skip $TAG (no qpos)"; continue; }
  nice -n 15 python3 $S/render_video.py --qpos "$Q" --states $S/runs/seeded/$TAG.csv \
      --table_h $H --fps 20 --out docs/lean/media/th/$TAG.mp4 \
      --title "table face $H m  ·  seed 0" || true
done

echo "=== [3/5] page $(date +%H:%M) ==="
cp -f $S/figs/*.png $S/figs/*.json docs/lean/media/th/ 2>/dev/null || true
nice -n 15 python3 $S/write_analysis.py --figs $S/figs --out $S/figs/analysis.html || true
nice -n 15 python3 $S/make_page.py --figs $S/figs --figs_rel media/th \
    --media docs/lean/media/th --media_rel media/th \
    --analysis $S/figs/analysis.html --out $PAGE || true

echo "=== [4/5] status $(date +%H:%M) ==="
nice -n 15 python3 $S/write_status.py --figs $S/figs --out $S/STATUS.md || true

echo "=== [5/5] long-cap sweep $(date +%H:%M) ==="
# The 75 s cap censors "stalled": a run that neither fell nor finished may simply
# have been slow. 140 s is ~3x the nominal completion time, so a point that still
# does not finish is genuinely stuck rather than clipped.
nice -n 15 python3 $S/sweep.py --out $S/runs/long \
    --heights 0.785,0.885,0.985,1.035,1.085 --seeds 2 --slot 25 \
    --total_time 140 --jobs 1 --threads 6 --cpu_quota 700 --video_seed 9 || true
nice -n 15 python3 $S/analyze.py --runs $S/runs/long --out $S/figs_long || true
{
  echo ""
  echo "## Long-cap check (--total_time 140, 2 seeds), finished $(date '+%Y-%m-%d %H:%M')"
  echo ""
  echo '```'
  grep "done h" $S/runs/long.log 2>/dev/null | sed 's/task=Lean H12 Magpie //;s/enter=.*//' || echo "(no runs)"
  echo '```'
} >> $S/STATUS.md 2>/dev/null || true

echo "=== ALLDONE $(date +%H:%M) ==="
