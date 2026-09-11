#!/bin/bash
# Overnight driver (started 17:36 after a duplicate-launch cleanup: campaign5's
# spp3 sweep had kept running when its bash was killed, and campaign6 launched a
# second one on the same dir; both were killed by PID). 3 jobs x 6 threads,
# CPUQuota 600% per scope, nice 15. Every sweep is resumable (same --out).
cd "$(dirname "$0")"
export SWEEP_MEM_MAX=5G
common="--jobs 3 --threads 6 --total_time 75 --cpu_quota 600 --nice 15 --log_per_plan --gains deploy --force"
six="--seeds 0,1,2,3,4,5"
./sweep.py --out runs/gains_spp3 --batch R,A,X --spp 3 $six $common > runs/gains_spp3.log 2>&1
echo "[campaign7] spp3 done $(date)"
./sweep.py --out runs/gains_spp10 --batch R,S,S2,K,T,X --spp 10 $six $common > runs/gains_spp10.log 2>&1
echo "[campaign7] spp10 done $(date)"
./sweep.py --out runs/gains_spp6 --batch R --spp 6 $six $common > runs/gains_spp6.log 2>&1
echo "[campaign7] spp6 done $(date)"
./sweep.py --out runs/gains_spp3 --batch S,S2 --spp 3 --seeds 0,1,2 $common > runs/gains_spp3b.log 2>&1
echo "[campaign7] spp3 sigma done $(date)"
