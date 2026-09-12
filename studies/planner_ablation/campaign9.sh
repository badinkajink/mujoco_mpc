#!/bin/bash
# Continuation of campaign8.sh (bash killed 20:38 to insert batch Z at 33 Hz
# after the running batch-Y sweep). 3 jobs x 6 threads overnight.
cd "$(dirname "$0")"
export SWEEP_MEM_MAX=5G
while kill -0 2540605 2>/dev/null; do sleep 15; done
echo "[campaign9] Y sweep exited $(date)"
common="--jobs 3 --threads 6 --total_time 75 --cpu_quota 600 --nice 15 --log_per_plan --gains deploy --force"
six="--seeds 0,1,2,3,4,5"
./sweep.py --out runs/gains_spp15 --batch Z --spp 15 $six $common > runs/gains_spp15f.log 2>&1
echo "[campaign9] spp15 Z done $(date)"
./sweep.py --out runs/gains_spp10 --batch R,S,S2,K,T,X --spp 10 $six $common > runs/gains_spp10.log 2>&1
echo "[campaign9] spp10 done $(date)"
./sweep.py --out runs/gains_spp6 --batch R --spp 6 $six $common > runs/gains_spp6.log 2>&1
echo "[campaign9] spp6 done $(date)"
./sweep.py --out runs/gains_spp3 --batch S,S2 --spp 3 --seeds 0,1,2 $common > runs/gains_spp3b.log 2>&1
echo "[campaign9] spp3 sigma done $(date)"
