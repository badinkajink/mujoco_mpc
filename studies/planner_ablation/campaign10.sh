#!/bin/bash
# Continuation of campaign9.sh (bash killed 21:00 to insert batch W, the
# 0.2-0.3 s persistence points, at 33 Hz after the running 50 Hz sweep).
cd "$(dirname "$0")"
export SWEEP_MEM_MAX=5G
while kill -0 2593159 2>/dev/null; do sleep 30; done
echo "[campaign10] spp10 sweep exited $(date)"
common="--jobs 3 --threads 6 --total_time 75 --cpu_quota 600 --nice 15 --log_per_plan --gains deploy --force"
six="--seeds 0,1,2,3,4,5"
./sweep.py --out runs/gains_spp15 --batch W --spp 15 $six $common > runs/gains_spp15g.log 2>&1
echo "[campaign10] spp15 W done $(date)"
./sweep.py --out runs/gains_spp6 --batch R --spp 6 $six $common > runs/gains_spp6.log 2>&1
echo "[campaign10] spp6 done $(date)"
./sweep.py --out runs/gains_spp3 --batch S,S2 --spp 3 --seeds 0,1,2 $common > runs/gains_spp3b.log 2>&1
echo "[campaign10] spp3 sigma done $(date)"
