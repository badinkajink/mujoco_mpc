#!/bin/bash
# Launch the full h12_lowerbody stand-task weight sweep + residual ablation.
# Run inside hams_ros:  bash .../tuning/run_sweep.sh [workers] [outdir]
#
# Populations of 25 candidates x N=25-episode evaluation. The search ranks on a
# cheaper episode budget (CRN-paired) and reserves N=25 for the honest
# evaluation of baseline / best / every ablation. See tune_weights.py.
set -uo pipefail

WORKERS="${1:-6}"
CPW="${3:-5}"          # cores per worker; planner_threads = CPW-3, so keep >=5
TS="$(date +%Y%m%d-%H%M%S)"
OUTDIR="${2:-/home/code/mujoco_mpc/mjpc/tasks/h12_lowerbody/tuning/runs/sweep_${TS}}"

cd /home/code/mujoco_mpc
export PYTHONPATH=/home/code/mujoco_mpc/python

cleanup(){ for p in $(pgrep -x agent_server); do kill -9 "$p" 2>/dev/null; done; }
trap cleanup EXIT
cleanup   # start from a clean slate (no orphan servers stealing cores)
sleep 1

echo "outdir: ${OUTDIR}"
python3 mjpc/tasks/h12_lowerbody/tuning/tune_weights.py \
  --mode both \
  --iterations 5 \
  --population 25 \
  --search-episodes 5 \
  --episodes 25 \
  --horizon 3.0 \
  --ctrl-dt 0.02 \
  --planner-steps 1 \
  --warmup-steps 15 \
  --push-vel 0.5 \
  --push-interval 1.5 \
  --workers "${WORKERS}" \
  --cores-per-worker "${CPW}" \
  --ablate-ref default \
  --outdir "${OUTDIR}"
rc=$?
echo "sweep exit code: ${rc}"
exit ${rc}
