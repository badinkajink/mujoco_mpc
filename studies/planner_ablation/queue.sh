#!/bin/bash
# Overnight queue for the planner ablation: waits for wave 1, then runs the
# remaining waves back to back. Each wave is resumable on its own.
cd "$(dirname "$0")"
export SWEEP_MEM_MAX=5G
while pgrep -f "sweep.py --out runs/wave1" > /dev/null; do sleep 30; done
echo "[queue] wave1 finished $(date)"
# icem seeds 0/1 were double-written by a duplicate launch (see runs/wave1.log
# note); redo them into a clean dir so the shipped control is uncontaminated.
./sweep.py --out runs/wave1_redo --arms icem --seeds 0,1 --jobs 2 --threads 6 --total_time 75 --force > runs/wave1_redo.log 2>&1
echo "[queue] wave1_redo finished $(date)"
./sweep.py --out runs/wave2 --batch C,D --seeds 0,1,2 --jobs 2 --threads 6 --total_time 75 --force > runs/wave2.log 2>&1
echo "[queue] wave2 finished $(date)"
# deploy-rate check: spp=15 -> a plan every 30 ms (33 Hz) instead of 6 ms
./sweep.py --out runs/wave3_spp15 --arms icem,ps_raw01_zero,cem_ne1_fixed01_nom,ps --seeds 0,1,2 --jobs 2 --threads 6 --total_time 75 --spp 15 --force > runs/wave3_spp15.log 2>&1
echo "[queue] wave3 finished $(date)"
