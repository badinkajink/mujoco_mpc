#!/bin/bash
# Second wave: the operator's pull-back assist, and the stiff-contact probe.
# The assist is an unmodelled backward force on the torso through the lean-in
# rungs (1-2) — what a person does to the robot on hardware so it settles into
# the brace instead of shearing forward. One directory per force so the run tags
# do not collide. All at the measured-estimate friction, sigma 0.02, 6 seeds.
set -u
cd "$(dirname "$0")"
for FX in -20 -40 -60; do
  ./sweep.py --out "runs/a${FX#-}_sd02" --cells 0.8:0.4 --seeds 0-5 --jobs 2 \
      --numeric std_min=0.02 --assist_fx "$FX"
done
./sweep.py --out runs/b3_sd02_stiff --cells 0.8:0.4 --seeds 0-5 --jobs 2 --numeric std_min=0.02 --stiff
