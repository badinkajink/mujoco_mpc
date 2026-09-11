#!/bin/bash
# Overnight continuation of campaign5.sh (user away from 16:45): 3 jobs x 6
# threads (18 of 20 cores, 3 x 3.7 GB), CPUQuota 600% per scope, nice 15.
# Same order and dirs as campaign5.sh; every sweep resumable.
cd "$(dirname "$0")"
export SWEEP_MEM_MAX=5G
while kill -0 1949527 2>/dev/null; do sleep 20; done
echo "[campaign6] previous 33 Hz sweep exited $(date)"
common="--jobs 3 --threads 6 --total_time 75 --cpu_quota 600 --nice 15 --log_per_plan --gains deploy --force"
six="--seeds 0,1,2,3,4,5"
./sweep.py --out runs/gains_spp15 --batch R,A,S,S2,K,T,N,X --spp 15 $six $common > runs/gains_spp15d.log 2>&1
echo "[campaign6] spp15 done $(date)"
./sweep.py --out runs/gains_spp3 --batch R,A,X --spp 3 $six $common > runs/gains_spp3.log 2>&1
echo "[campaign6] spp3 done $(date)"
./sweep.py --out runs/gains_spp10 --batch R,S,S2,K,T,X --spp 10 $six $common > runs/gains_spp10.log 2>&1
echo "[campaign6] spp10 done $(date)"
./sweep.py --out runs/gains_spp6 --batch R --spp 6 $six $common > runs/gains_spp6.log 2>&1
echo "[campaign6] spp6 done $(date)"
./sweep.py --out runs/gains_spp3 --batch S,S2 --spp 3 --seeds 0,1,2 $common > runs/gains_spp3b.log 2>&1
echo "[campaign6] spp3 sigma done $(date)"
