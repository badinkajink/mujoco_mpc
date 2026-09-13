#!/bin/bash
# Continuation of campaign11.sh (bash killed 01:10 to insert FS -- the step-size
# x low-plan-rate test -- ahead of W and the sigma seeds). 3 jobs x 6 threads.
cd "$(dirname "$0")"
export SWEEP_MEM_MAX=5G
while kill -0 1519960 2>/dev/null; do sleep 20; done
echo "[campaign12] latency 50 sweep exited $(date)"
common="--jobs 3 --threads 6 --total_time 75 --cpu_quota 600 --nice 15 --log_per_plan --gains deploy --force"
six="--seeds 0,1,2,3,4,5"
./sweep.py --out runs/gains_spp15 --batch H --spp 15 $six $common > runs/gains_spp15h.log 2>&1
echo "[campaign12] H done $(date)"
for spp in 20 30 50; do
  ./sweep.py --out runs/floor_spp$spp --batch FS --spp $spp $six $common > runs/floor_spp${spp}b.log 2>&1
  echo "[campaign12] FS spp$spp done $(date)"
done
./sweep.py --out runs/perturb_spp15_p02 --batch R --spp 15 --perturb 0.02 $six $common > runs/perturb_spp15_p02.log 2>&1
echo "[campaign12] perturb done $(date)"
./sweep.py --out runs/gains_spp10 --batch W --spp 10 $six $common > runs/gains_spp10w.log 2>&1
echo "[campaign12] W spp10 done $(date)"
./sweep.py --out runs/gains_spp3 --batch W --spp 3 $six $common > runs/gains_spp3w.log 2>&1
echo "[campaign12] W spp3 done $(date)"
./sweep.py --out runs/gains_spp3 --batch S,S2 --spp 3 $six $common > runs/gains_spp3c.log 2>&1
echo "[campaign12] sigma seeds 3-5 done $(date)"
