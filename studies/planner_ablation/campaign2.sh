#!/bin/bash
# Continuation of campaign_rate.sh with the deploy-gains test inserted first:
# waits for the 50 Hz sweep, runs the 33 Hz arms with the deploy node's KP/KV
# on plant + planner, then finishes the 83 Hz and 167 Hz rows.
cd "$(dirname "$0")"
export SWEEP_MEM_MAX=5G
while pgrep -f "sweep.py --out runs/rate_spp10" > /dev/null; do sleep 30; done
echo "[campaign2] spp10 done $(date)"
common="--seeds 0,1,2,3,4,5 --jobs 2 --threads 6 --total_time 75 --cpu_quota 550 --nice 15 --log_per_plan --force"
./sweep.py --out runs/gains_spp15 --arms cem,icem,ps_raw01_cubic,cem_ne2,cem_stdmin02,mppi_raw01_zero_l1 --spp 15 --gains deploy $common > runs/gains_spp15.log 2>&1
echo "[campaign2] gains_spp15 done $(date)"
./sweep.py --out runs/rate_spp6 --batch R --spp 6 $common > runs/rate_spp6.log 2>&1
echo "[campaign2] spp6 done $(date)"
./sweep.py --out runs/rate_spp3 --batch R --seeds 3,4,5 --jobs 2 --threads 6 --total_time 75 --cpu_quota 550 --nice 15 --log_per_plan --force --spp 3 > runs/rate_spp3.log 2>&1
echo "[campaign2] spp3 done $(date)"
