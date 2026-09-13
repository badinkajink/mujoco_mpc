#!/bin/bash
# 2026-09-13 00:15, user asleep. 3 jobs x 6 threads. In order:
#  V   video arms at 33 Hz, seed 1, with the qpos track (renders for the page)
#  F   plan-rate floor: batch R at 25 / 16.7 / 10 / 5 plans/s (spp 20/30/50/100)
#  L   uncompensated plan latency at 33 Hz: 30 / 60 / 100 ms (latency_steps 15/30/50)
#  H   horizon, budget N=4, rollout step 20 ms at 33 Hz
#  P   initial perturbation 20 mm (bench default 3 mm) at 33 Hz, batch R
#  W   persistence-band arms at 50 and 167 plans/s
#  S3  sigma axis at 167 plans/s, seeds 3-5 (the 3-seed cells -> 6)
cd "$(dirname "$0")"
export SWEEP_MEM_MAX=5G
common="--jobs 3 --threads 6 --total_time 75 --cpu_quota 600 --nice 15 --log_per_plan --gains deploy --force"
six="--seeds 0,1,2,3,4,5"
./sweep.py --out runs/video_spp15 --batch V --spp 15 --seeds 1 --qpos $common > runs/video_spp15.log 2>&1
echo "[campaign11] video done $(date)"
for spp in 20 30 50 100; do
  ./sweep.py --out runs/floor_spp$spp --batch R --spp $spp $six $common > runs/floor_spp$spp.log 2>&1
  echo "[campaign11] floor spp$spp done $(date)"
done
for L in 15 30 50; do
  ./sweep.py --out runs/lat_spp15_l$L --batch R --spp 15 --latency $L $six $common > runs/lat_spp15_l$L.log 2>&1
  echo "[campaign11] latency $L done $(date)"
done
./sweep.py --out runs/gains_spp15 --batch H --spp 15 $six $common > runs/gains_spp15h.log 2>&1
echo "[campaign11] H done $(date)"
./sweep.py --out runs/perturb_spp15_p02 --batch R --spp 15 --perturb 0.02 $six $common > runs/perturb_spp15_p02.log 2>&1
echo "[campaign11] perturb done $(date)"
./sweep.py --out runs/gains_spp10 --batch W --spp 10 $six $common > runs/gains_spp10w.log 2>&1
echo "[campaign11] W spp10 done $(date)"
./sweep.py --out runs/gains_spp3 --batch W --spp 3 $six $common > runs/gains_spp3w.log 2>&1
echo "[campaign11] W spp3 done $(date)"
./sweep.py --out runs/gains_spp3 --batch S,S2 --spp 3 $six $common > runs/gains_spp3c.log 2>&1
echo "[campaign11] sigma seeds 3-5 done $(date)"
