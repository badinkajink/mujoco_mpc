#!/bin/bash
# Third queue: plan-rate ladder. spp counts 2 ms plant steps: 6 -> 83 Hz,
# 10 -> 50 Hz (the twin's documented floor), 15 -> 33 Hz (wave 3).
cd "$(dirname "$0")"
export SWEEP_MEM_MAX=5G
while pgrep -f "queue2.sh" > /dev/null; do sleep 60; done
echo "[queue3] queue2.sh finished $(date)"
./sweep.py --out runs/wave5_spp10 --arms icem,ps_raw01_cubic --seeds 0,1,2 --jobs 2 --threads 6 --total_time 75 --spp 10 --force > runs/wave5_spp10.log 2>&1
echo "[queue3] wave5_spp10 finished $(date)"
./sweep.py --out runs/wave5_spp6 --arms icem,ps_raw01_cubic --seeds 0,1,2 --jobs 2 --threads 6 --total_time 75 --spp 6 --force > runs/wave5_spp6.log 2>&1
echo "[queue3] wave5_spp6 finished $(date)"
