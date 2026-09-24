#!/bin/bash
# Friction study batches in priority order, one after the other (each batch is
# 2 jobs x 6 threads through resguard.sh run). Seed-major inside a batch, so an
# interrupted batch leaves balanced cells; re-running a line resumes it.
set -u
cd "$(dirname "$0")"
CELLS="1:1,0.8:0.4,0.6:0.3"
# 1. CEM at sigma 0.02 (braces in 11/12 nominal seeds): nominal, real estimate, kinetic bracket
./sweep.py --out runs/b1_sd02 --cells $CELLS --seeds 0-11 --jobs 2 --numeric std_min=0.02
# 2. deployed CEM (sigma 0.01) on a slab 20 mm higher than it believes
./sweep.py --out runs/b5_sd01_dz20 --cells 1:1,0.8:0.4 --seeds 0-11 --jobs 2 --table_dz 0.02
# 3. deployed CEM at the three frictions
./sweep.py --out runs/b2_sd01 --cells $CELLS --seeds 0-11 --jobs 2
