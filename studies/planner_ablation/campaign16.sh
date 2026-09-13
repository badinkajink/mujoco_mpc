#!/bin/bash
# After campaign15.sh (W + sigma seeds): latency WITH the node's compensation,
# the 25 plans/s column of the basin (S, S2), and the low-rate follow-ups
# (iCEM sigma window, CEM k at sigma 0.02) at 25/16.7/10 plans/s.
cd "$(dirname "$0")"
export SWEEP_MEM_MAX=5G
while pgrep -x -f "/bin/bash ./campaign15.sh" > /dev/null; do sleep 60; done
echo "[campaign16] campaign15 exited $(date)"
common="--jobs 3 --threads 6 --total_time 75 --cpu_quota 600 --nice 15 --log_per_plan --gains deploy --force"
six="--seeds 0,1,2,3,4,5"
for L in 100 150 250; do
  ./sweep.py --out runs/latc_spp15_l$L --batch R --spp 15 --latency $L --latency_compensate $six $common > runs/latc_spp15_l$L.log 2>&1
  echo "[campaign16] compensated latency $L done $(date)"
done
./sweep.py --out runs/floor_spp20 --batch S,S2 --spp 20 $six $common > runs/floor_spp20c.log 2>&1
echo "[campaign16] basin spp20 done $(date)"
for spp in 20 30 50; do
  ./sweep.py --out runs/floor_spp$spp --batch FS2 --spp $spp $six $common > runs/floor_spp${spp}d.log 2>&1
  echo "[campaign16] FS2 spp$spp done $(date)"
done
