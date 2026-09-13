#!/bin/bash
# Continuation of campaign12.sh (bash killed 01:20 to add the longer latencies:
# 100 ms cost nothing at 33 plans/s, so 200 / 300 / 500 ms next). 3 jobs x 6 threads.
cd "$(dirname "$0")"
export SWEEP_MEM_MAX=5G
while kill -0 1537646 2>/dev/null; do sleep 20; done
echo "[campaign13] H sweep exited $(date)"
common="--jobs 3 --threads 6 --total_time 75 --cpu_quota 600 --nice 15 --log_per_plan --gains deploy --force"
six="--seeds 0,1,2,3,4,5"
for spp in 20 30 50; do
  ./sweep.py --out runs/floor_spp$spp --batch FS --spp $spp $six $common > runs/floor_spp${spp}b.log 2>&1
  echo "[campaign13] FS spp$spp done $(date)"
done
for L in 100 150 250; do
  ./sweep.py --out runs/lat_spp15_l$L --batch R --spp 15 --latency $L $six $common > runs/lat_spp15_l$L.log 2>&1
  echo "[campaign13] latency $L done $(date)"
done
./sweep.py --out runs/perturb_spp15_p02 --batch R --spp 15 --perturb 0.02 $six $common > runs/perturb_spp15_p02.log 2>&1
echo "[campaign13] perturb done $(date)"
./sweep.py --out runs/gains_spp10 --batch W --spp 10 $six $common > runs/gains_spp10w.log 2>&1
echo "[campaign13] W spp10 done $(date)"
./sweep.py --out runs/gains_spp3 --batch W --spp 3 $six $common > runs/gains_spp3w.log 2>&1
echo "[campaign13] W spp3 done $(date)"
./sweep.py --out runs/gains_spp3 --batch S,S2 --spp 3 $six $common > runs/gains_spp3c.log 2>&1
echo "[campaign13] sigma seeds 3-5 done $(date)"
