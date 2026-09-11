#!/bin/bash
# Deploy-gains campaign (2026-09-11 15:10 MDT). Every run: lean_bench --gains deploy
# (the deploy node's KP/KV table on plant + planner). Replaces the XML-gains
# rows of campaign_rate.sh / campaign2.sh, which measured a different plant.
# Order = priority: the robot's rate first, then the 167 Hz rows behind the
# original PS/MPPI claim, then the 50 Hz slice of the basin, then 83 Hz, then
# the 167 Hz basin at 3 seeds. Each sweep is resumable (same --out).
cd "$(dirname "$0")"
export SWEEP_MEM_MAX=5G
common="--jobs 2 --threads 6 --total_time 75 --cpu_quota 550 --nice 15 --log_per_plan --gains deploy --force"
six="--seeds 0,1,2,3,4,5"
# 1. 33 Hz: everything (R + shipped ps/mppi + sigma/k/lambda/N axes); 6 done arms are skipped
./sweep.py --out runs/gains_spp15 --batch R,A,S,K,T,N --spp 15 $six $common > runs/gains_spp15b.log 2>&1
echo "[campaign3] spp15 done $(date)"
# 2. 167 Hz: the five update rules + shipped ps/mppi
./sweep.py --out runs/gains_spp3 --batch R,A --spp 3 $six $common > runs/gains_spp3.log 2>&1
echo "[campaign3] spp3 done $(date)"
# 3. 50 Hz: R + sigma + k + lambda
./sweep.py --out runs/gains_spp10 --batch R,S,K,T --spp 10 $six $common > runs/gains_spp10.log 2>&1
echo "[campaign3] spp10 done $(date)"
# 4. 83 Hz: R
./sweep.py --out runs/gains_spp6 --batch R --spp 6 $six $common > runs/gains_spp6.log 2>&1
echo "[campaign3] spp6 done $(date)"
# 5. 167 Hz: sigma axis, 3 seeds
./sweep.py --out runs/gains_spp3 --batch S --spp 3 --seeds 0,1,2 $common > runs/gains_spp3b.log 2>&1
echo "[campaign3] spp3 sigma done $(date)"
