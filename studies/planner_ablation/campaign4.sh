#!/bin/bash
# Continuation of campaign3.sh (killed 15:22 so the spline x update-rule arms
# could go in at 33 Hz before the 167 Hz block). Waits for the running 33 Hz
# sweep, then the same order as campaign3.sh with batch X added at 33/50 Hz.
cd "$(dirname "$0")"
export SWEEP_MEM_MAX=5G
while kill -0 1949527 2>/dev/null; do sleep 20; done
echo "[campaign4] campaign3 spp15 sweep exited $(date)"
common="--jobs 2 --threads 6 --total_time 75 --cpu_quota 550 --nice 15 --log_per_plan --gains deploy --force"
six="--seeds 0,1,2,3,4,5"
./sweep.py --out runs/gains_spp15 --batch R,A,S,K,T,N,X --spp 15 $six $common > runs/gains_spp15c.log 2>&1
echo "[campaign4] spp15 done $(date)"
./sweep.py --out runs/gains_spp3 --batch R,A,X --spp 3 $six $common > runs/gains_spp3.log 2>&1
echo "[campaign4] spp3 done $(date)"
./sweep.py --out runs/gains_spp10 --batch R,S,K,T,X --spp 10 $six $common > runs/gains_spp10.log 2>&1
echo "[campaign4] spp10 done $(date)"
./sweep.py --out runs/gains_spp6 --batch R --spp 6 $six $common > runs/gains_spp6.log 2>&1
echo "[campaign4] spp6 done $(date)"
./sweep.py --out runs/gains_spp3 --batch S --spp 3 --seeds 0,1,2 $common > runs/gains_spp3b.log 2>&1
echo "[campaign4] spp3 sigma done $(date)"
