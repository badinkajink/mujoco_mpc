#!/bin/bash
# Fourth queue: fill in the 33 Hz row (spp=15) with the cubic-spline PS arm
# and plain CEM, so the deploy-rate table has the elite-mean vs argmin pair
# at both splines.
cd "$(dirname "$0")"
export SWEEP_MEM_MAX=5G
while pgrep -f "queue3.sh" > /dev/null; do sleep 60; done
echo "[queue4] queue3.sh finished $(date)"
./sweep.py --out runs/wave3_spp15 --arms ps_raw01_cubic,cem,mppi_raw01_zero_l1 --seeds 0,1,2 --jobs 2 --threads 6 --total_time 75 --spp 15 --force > runs/wave3b_spp15.log 2>&1
echo "[queue4] wave3b finished $(date)"
