#!/bin/bash
# Continuation of campaign13.sh (its bash killed 02:35; campaign14 was mis-chained
# and killed). Waits for the running latency sweep, then: latency 500 ms,
# perturbation, plant-only mismatch, W, sigma seeds 3-5. 3 jobs x 6 threads.
cd "$(dirname "$0")"
export SWEEP_MEM_MAX=5G
while kill -0 1634990 2>/dev/null; do sleep 20; done
echo "[campaign15] latency sweep exited $(date)"
common="--jobs 3 --threads 6 --total_time 75 --cpu_quota 600 --nice 15 --log_per_plan --gains deploy --force"
six="--seeds 0,1,2,3,4,5"
for L in 150 250; do
  ./sweep.py --out runs/lat_spp15_l$L --batch R --spp 15 --latency $L $six $common > runs/lat_spp15_l$L.log 2>&1
  echo "[campaign15] latency $L done $(date)"
done
./sweep.py --out runs/perturb_spp15_p02 --batch R --spp 15 --perturb 0.02 $six $common > runs/perturb_spp15_p02.log 2>&1
echo "[campaign15] perturb done $(date)"
for kp in 0.5 2.0; do
  ./sweep.py --out runs/mismatch_spp15_kp$kp --batch R --spp 15 --plant_kp_scale $kp $six $common > runs/mismatch_spp15_kp$kp.log 2>&1
  echo "[campaign15] mismatch kp $kp done $(date)"
done
for m in 1.1 1.25; do
  ./sweep.py --out runs/mismatch_spp15_m$m --batch R --spp 15 --plant_mass_scale $m $six $common > runs/mismatch_spp15_m$m.log 2>&1
  echo "[campaign15] mismatch mass $m done $(date)"
done
./sweep.py --out runs/gains_spp10 --batch W --spp 10 $six $common > runs/gains_spp10w.log 2>&1
echo "[campaign15] W spp10 done $(date)"
./sweep.py --out runs/gains_spp3 --batch W --spp 3 $six $common > runs/gains_spp3w.log 2>&1
echo "[campaign15] W spp3 done $(date)"
./sweep.py --out runs/gains_spp3 --batch S,S2 --spp 3 $six $common > runs/gains_spp3c.log 2>&1
echo "[campaign15] sigma seeds 3-5 done $(date)"
