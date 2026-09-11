#!/bin/bash
# Score every plan-rate level into its own summary. Default = the deploy-gains
# campaign (lean_bench --gains deploy, runs/gains_spp*), the plant the robot is.
# `./score_rates.sh xml` scores the earlier XML-gains dirs (a different plant;
# kept for the record). A later --runs dir overrides an earlier one for the
# same (arm, seed).
cd "$(dirname "$0")"
ORDER=cem,icem,mppi_raw01_zero_l1,ps_raw01_cubic,ps_raw01_zero,ps,mppi,cem_stdmin005,cem_stdmin02,cem_stdmin03,cem_stdmin05,cem_stdmin10,ps_raw005_cubic,ps_raw02_cubic,ps_raw03_cubic,ps_raw05_cubic,mppi_raw005_zero_l1,mppi_raw02_zero_l1,mppi_raw03_zero_l1,mppi_raw05_zero_l1,cem_ne1,cem_ne2,cem_ne10,cem_ne20,mppi_raw01_zero_l0.1,mppi_raw01_zero_l10,cem_n8_ne2,cem_n40_ne12,ps_raw01_cubic_n8,ps_raw01_cubic_n40,mppi_raw01_zero_l1_n8,mppi_raw01_zero_l1_n40
if [ "${1:-deploy}" = "xml" ]; then
  ./analyze.py --runs runs/wave1 runs/wave1_redo runs/wave2 runs/wave4 runs/rate_spp3 --out runs/summary_rate_spp3.json > /dev/null
  ./analyze.py --runs runs/wave5_spp6 runs/rate_spp6 --out runs/summary_rate_spp6.json > /dev/null
  ./analyze.py --runs runs/wave5_spp10 runs/rate_spp10 --out runs/summary_rate_spp10.json > /dev/null
  ./analyze.py --runs runs/wave3_spp15 runs/rate_spp15 --out runs/summary_rate_spp15.json --order $ORDER
else
  for spp in 3 6 10 15; do
    [ -f runs/gains_spp$spp/results.jsonl ] || continue
    ./analyze.py --runs runs/gains_spp$spp --out runs/summary_deploy_spp$spp.json --order $ORDER > /dev/null
  done
  ./analyze.py --runs runs/gains_spp15 --out runs/summary_gains_spp15.json --order $ORDER > /dev/null  # old name, same data
fi
