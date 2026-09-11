#!/bin/bash
# Score every plan-rate level into its own summary; a later --runs dir overrides
# an earlier one for the same (arm, seed), so the per-plan-logged campaign runs
# supersede the 50 Hz-logged pilots of the same seeds.
cd "$(dirname "$0")"
./analyze.py --runs runs/wave1 runs/wave1_redo runs/wave2 runs/wave4 runs/rate_spp3 --out runs/summary_rate_spp3.json > /dev/null
./analyze.py --runs runs/wave5_spp6 runs/rate_spp6 --out runs/summary_rate_spp6.json > /dev/null
./analyze.py --runs runs/wave5_spp10 runs/rate_spp10 --out runs/summary_rate_spp10.json > /dev/null
./analyze.py --runs runs/wave3_spp15 runs/rate_spp15 --out runs/summary_rate_spp15.json \
  --order cem,icem,mppi_raw01_zero_l0.1,mppi_raw01_zero_l1,mppi_raw01_zero_l10,ps_raw01_cubic,ps_raw01_zero,cem_ne1,cem_ne2,cem_ne10,cem_ne20,cem_stdmin02,cem_stdmin03,cem_stdmin05,cem_stdmin10,ps_raw02_cubic,ps_raw03_cubic,ps_raw05_cubic
