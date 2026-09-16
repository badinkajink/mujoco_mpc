#!/bin/bash
# sweep2 (2026-09-15): the three follow-ups from the 2026-09-14 page, in the
# order their answers are needed. One results.jsonl, resumable. Every job goes
# through ~/.claude/bin/resguard.sh (gate + scope + watchdog); 2 jobs x 6 threads
# = 7.4 GB RSS, 14 threads -- the box's whole compute budget with the desktop open,
# so nothing else heavy runs alongside (the 2026-09-15 reboot was a third process).
#   usage: ./run_sweep2.sh [out_dir] [stage ...]     (no stage = all five)
#   1. D     release-plan dump on the (true 0|4) x (believed 0|4) pairs, 3 seeds     12 runs
#   2. M,P   the believed-3 kg column, seeds 0-2                                    24 runs
#   3. Smean, Smax   scenario baseline over {0,2,4} kg, seeds 0-2                   24 runs
#   4. M     seeds 3-5 on the full 4 x 5 grid                                       60 runs
#   5. Smin  the optimistic aggregation, seeds 0-2                                  12 runs
set -u
cd "$(dirname "$0")"
OUT=${1:-runs/sweep2}; shift || true
STAGES=${*:-1 2 3 4 5}
mkdir -p "$OUT"
LOG="$OUT/driver.log"
run() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; python3 sweep.py --out "$OUT" "$@" >> "$LOG" 2>&1; }
for st in $STAGES; do
  case $st in
    1) run --arms D --trues 0,4 --beliefs 0,4 --seeds 0,1,2;;
    2) run --arms M,P --trues 0,1,2,4 --beliefs 3 --seeds 0,1,2;;
    3) run --arms Smean,Smax --trues 0,1,2,4 --seeds 0,1,2;;
    4) run --arms M --trues 0,1,2,4 --beliefs 0,1,2,3,4 --seeds 3,4,5;;
    5) run --arms Smin --trues 0,1,2,4 --seeds 0,1,2;;
  esac
done
echo "[$(date +%H:%M:%S)] stages $STAGES done" >> "$LOG"
