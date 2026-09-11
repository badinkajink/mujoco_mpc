#!/bin/bash
# Second overnight queue: waits for queue.sh, then runs batch E (budget and
# lookahead within iCEM, and the PS noise ladder around 0.01).
cd "$(dirname "$0")"
export SWEEP_MEM_MAX=5G
while pgrep -f "queue.sh" > /dev/null; do sleep 60; done
echo "[queue2] queue.sh finished $(date)"
./sweep.py --out runs/wave4 --batch E --seeds 0,1,2 --jobs 2 --threads 6 --total_time 75 --force > runs/wave4.log 2>&1
echo "[queue2] wave4 finished $(date)"
