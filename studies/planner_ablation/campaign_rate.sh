#!/bin/bash
# Deploy-rate campaign (2026-09-11): the five update rules at four plan rates,
# and the CEM elite-count / noise-floor ladders plus the PS floor ladder and the
# MPPI temperature ladder at 33 Hz. Six seeds, per-plan logging, threads fixed
# at 6. Daytime run: two jobs under a 550% CPUQuota each (11 of 20 cores).
cd "$(dirname "$0")"
export SWEEP_MEM_MAX=5G
common="--seeds 0,1,2,3,4,5 --jobs 2 --threads 6 --total_time 75 --cpu_quota 550 --nice 15 --log_per_plan --force"
./sweep.py --out runs/rate_spp15 --batch R,L --spp 15 $common > runs/rate_spp15.log 2>&1
echo "[campaign] spp15 done $(date)"
./sweep.py --out runs/rate_spp10 --batch R --arms cem_stdmin03,ps_raw03_cubic --spp 10 $common > runs/rate_spp10.log 2>&1
echo "[campaign] spp10 done $(date)"
./sweep.py --out runs/rate_spp6 --batch R --spp 6 $common > runs/rate_spp6.log 2>&1
echo "[campaign] spp6 done $(date)"
# 167 Hz already has seeds 0-2 for these arms (waves 1/2); add 3-5 per-plan logged
./sweep.py --out runs/rate_spp3 --batch R --seeds 3,4,5 --jobs 2 --threads 6 --total_time 75 --cpu_quota 550 --nice 15 --log_per_plan --force --spp 3 > runs/rate_spp3.log 2>&1
echo "[campaign] spp3 done $(date)"
