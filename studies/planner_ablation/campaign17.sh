#!/bin/bash
# 2026-09-13 14:00: model-mismatch ladder for the sim-to-real claim. Plant-only
# mismatch (planner keeps the model): mass x 1.05/1.10/1.15/1.20, kp x 1.5/2/3,
# sliding friction x 0.6/0.4 (the real table went from mu 0.3 to 0.8 with grip
# tape). Four hold rules at 0.01 rad, 12 seeds (0-11) per cell; baseline at 12
# seeds; then the sigma link under mass x 1.10. 3 jobs x 6 threads.
cd "$(dirname "$0")"
export SWEEP_MEM_MAX=5G
common="--jobs 3 --threads 6 --total_time 75 --cpu_quota 600 --nice 15 --log_per_plan --gains deploy --force"
twelve="--seeds 0,1,2,3,4,5,6,7,8,9,10,11"
./sweep.py --out runs/gains_spp15 --batch M --spp 15 $twelve $common > runs/gains_spp15m.log 2>&1
echo "[campaign17] baseline seeds 6-11 done $(date)"
for m in 1.05 1.10 1.15 1.20; do
  ./sweep.py --out runs/mismatch_spp15_m$m --batch M --spp 15 --plant_mass_scale $m $twelve $common > runs/mismatch_spp15_m$m.log 2>&1
  echo "[campaign17] mass $m done $(date)"
done
for kp in 1.5 2.0 3.0; do
  ./sweep.py --out runs/mismatch_spp15_kp$kp --batch M --spp 15 --plant_kp_scale $kp $twelve $common > runs/mismatch_spp15_kp$kp.log 2>&1
  echo "[campaign17] kp $kp done $(date)"
done
for f in 0.6 0.4; do
  ./sweep.py --out runs/mismatch_spp15_mu$f --batch M --spp 15 --plant_friction_scale $f $twelve $common > runs/mismatch_spp15_mu$f.log 2>&1
  echo "[campaign17] friction $f done $(date)"
done
./sweep.py --out runs/mismatch_spp15_m1.10 --batch MS --spp 15 --plant_mass_scale 1.10 $twelve $common > runs/mismatch_spp15_m1.10s.log 2>&1
echo "[campaign17] sigma link under mass 1.10 done $(date)"
./sweep.py --out runs/gains_spp15 --batch MS --spp 15 $twelve $common > runs/gains_spp15ms.log 2>&1
echo "[campaign17] sigma link baseline done $(date)"
