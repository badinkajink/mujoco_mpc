#!/bin/bash
# Second wave: WHEN the operator's pull-back is applied, and the stiff-contact
# probe. The pull is an unmodelled backward force on the torso (xfrc_applied),
# so the planner is as blind to it as it is to a person. All at the measured
# friction estimate (slab 0.8 / floor 0.4), sigma 0.02, 6 seeds, one directory
# per condition.
#   g15  = pull whenever the pad is within 150 mm of the face, i.e. through the
#          whole lean-in (the pad already sits 66 mm above the face at rung
#          entry, so this is "pull the whole way down")
#   g05  = pull only over the last 50 mm of the approach
#   ph2  = pull only after the brace is down (rung 2 onward)
set -u
cd "$(dirname "$0")"
./sweep.py --out runs/a40_g15 --cells 0.8:0.4 --seeds 0-5 --jobs 2 --numeric std_min=0.02 --assist_fx -40
./sweep.py --out runs/a20_g05 --cells 0.8:0.4 --seeds 0-5 --jobs 2 --numeric std_min=0.02 --assist_fx -20 --assist_gap 0.05
./sweep.py --out runs/a40_g05 --cells 0.8:0.4 --seeds 0-5 --jobs 2 --numeric std_min=0.02 --assist_fx -40 --assist_gap 0.05
./sweep.py --out runs/a40_ph2 --cells 0.8:0.4 --seeds 0-5 --jobs 2 --numeric std_min=0.02 --assist_fx -40 \
    --assist_from_phase 2 --assist_until_phase 3 --assist_gap 0
./sweep.py --out runs/b3_sd02_stiff --cells 0.8:0.4 --seeds 0-5 --jobs 2 --numeric std_min=0.02 --stiff
