#!/bin/bash
# Interleaved honest-twin A/B: planner residual OFF vs ON (arms+waist, 2x64 DaD net), strat 25 target B3, d37,
# creep-fixed twin plant (_real_ns). One bench at a time, each under a cgroup memory cap. Summary appended to $SUM.
set -u
H12=~/Desktop/h12; MB=$H12/mujoco_mpc/mujoco_mpc/mbrl
N=${1:-5}; DUR=${2:-150}; STRAT=${STRAT:-25}; TGT=${TGT:-B3}
W=$MB/out/residual_dadpre10s.bin
SUM=$H12/logs/grid37_twin/d37/AB_residual_$(date +%m%d_%H%M).md
echo "# twin A/B residual OFF vs ON  strat $STRAT $TGT  $(date)" > "$SUM"
echo "| run | arm | residual line | plan/s | gate tail | twin end |" >> "$SUM"; echo "|---|---|---|---|---|---|" >> "$SUM"
run_one() {  # arm r
  local arm=$1 r=$2 suf="_${1}_r${2}"
  local env=""; [ "$arm" = res ] && env="H12_RESIDUAL_MLP=$W H12_RESIDUAL_SCALE=0,1,1"
  echo "[ab] $(date +%H:%M:%S) start $arm r$r"
  env SCENE_VAR=_real_ns TAG_SUFFIX=$suf $env systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 \
    bash $H12/scratch_bench/grid37_twin_noassist.sh $STRAT $TGT $DUR > $H12/logs/grid37_twin/d37/ab${suf}.log 2>&1
  local out=$H12/logs/grid37_twin/d37/s${STRAT}_${TGT}_noassist${suf}
  local rl; rl=$(grep -m1 "planner residual ON" $out/node.log 2>/dev/null | sed 's/.*ON //' ); [ -z "$rl" ] && rl="off"
  local pr; pr=$(grep -oE "plan=[0-9]+/s" $out/node.log 2>/dev/null | tail -20 | grep -oE "[0-9]+" | sort -n | awk '{a[NR]=$1} END{print a[int(NR/2)+1]}')
  local gate; gate=$(grep -E "lean-gate\]" $out/node.log 2>/dev/null | grep -E "rung [0-9]|TIMEOUT|RELEAS|seat" | tail -2 | sed 's/.*lean-gate\]//' | tr '\n' ';' | cut -c1-120)
  local tw; tw=$(grep -E "twin-harness|FALL|fall|stop-on-fall" $out/twin.log 2>/dev/null | tail -1 | cut -c1-90)
  echo "| $r | $arm | $rl | $pr | $gate | $tw |" >> "$SUM"
  echo "[ab] $(date +%H:%M:%S) done $arm r$r : plan $pr  gate: $gate  twin: $tw"
  sleep 5
}
for r in $(seq 1 $N); do run_one base $r; run_one res $r; done
echo "[ab] ALL DONE -> $SUM"; cat "$SUM"
