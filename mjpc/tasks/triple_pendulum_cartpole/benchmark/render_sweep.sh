#!/usr/bin/env bash
# Render every (stage, planner) dump produced by sweep.sh to a filmstrip PNG and
# an MP4, plus a per-step text trace.
#
# The point is to make each planner's behaviour visible rather than inferred:
# on this task the aggregate cost ranking and the "what did it actually do"
# ranking disagree, so the videos are the primary artifact and the cost table
# is the secondary one.
#
# Usage: render_sweep.sh [sweep_dir] [repeat_index]
set -u

SWEEP=${1:-renders/sweep}
REP=${2:-0}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT="$SWEEP/render"
mkdir -p "$OUT"
export MUJOCO_GL=${MUJOCO_GL:-egl}

TRACE="$OUT/traces.txt"
: > "$TRACE"

for csv in "$SWEEP"/dumps/*_"$REP".csv; do
  base=$(basename "$csv" "_$REP.csv")
  stage=${base%%_*}
  echo "=== $base ===" | tee -a "$TRACE"
  # --stage matters: the balance stage ran with the obstacles removed, and
  # without telling the renderer that, every balance video shows a corridor
  # that was not there.
  python3 "$HERE/filmstrip.py" --dump "$csv" --stage "$stage" \
      --out "$OUT/$base.png" --video "$OUT/$base.mp4" \
      --video-stride 8 2>&1 | tee -a "$TRACE"
  echo | tee -a "$TRACE"
done

echo "wrote $OUT"
