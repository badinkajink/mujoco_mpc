#!/bin/bash
# Interleaved honest-twin A/B of the PARAMETRIC correction: planner joint damping 10 -> 30 (param_id2 stage D, at the
# bound) in the planner model include (source + build copy, restored on exit). No residual. One bench at a time.
set -u
H12=~/Desktop/h12; MJ=$H12/mujoco_mpc/mujoco_mpc; N=${1:-5}; DUR=${2:-180}; STRAT=${STRAT:-37}; TGT=${TGT:-B3}; DAMP=${DAMP:-30}
# the h1_2 include exists only in the BUILD tree (generated/patched by the build); patch and restore that copy
B=$MJ/build/mjpc/tasks/humanoid_bench/lean/h1_2/h1_2_modified_magpie.xml
cp "$B" /tmp/claude-1000/h1_2_modified_magpie.xml.pre
restore() { cp /tmp/claude-1000/h1_2_modified_magpie.xml.pre "$B"; echo "[damp-ab] build include restored ($(grep -oE '<joint damping="[0-9.]+"' "$B" | head -1))"; }
trap restore EXIT
set_damp() { sed -i -E "s|<joint damping=\"[0-9.]+\" armature=\"0.1\"/>|<joint damping=\"$1\" armature=\"0.1\"/>|" "$B"
  grep -oE '<joint damping="[0-9.]+" armature="0.1"/>' "$B" | head -1 | grep -q "damping=\"$1\"" || { echo "damping patch failed"; exit 1; }; }
SUM=$H12/logs/grid37_twin/d37/AB_damp${DAMP}_$(date +%m%d_%H%M).md; echo "# damping 10 vs $DAMP  strat $STRAT $TGT  $(date)" > "$SUM"
run_one() { local arm=$1 r=$2 suf="_${1}_r${2}"
  if [ "$arm" = damp ]; then set_damp $DAMP; else set_damp 10; fi
  echo "[damp-ab] $(date +%H:%M:%S) start $arm r$r"
  env SCENE_VAR=_real_ns TAG_SUFFIX=$suf systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 \
    bash $H12/scratch_bench/grid37_twin_noassist.sh $STRAT $TGT $DUR > $H12/logs/grid37_twin/d37/ab${suf}.log 2>&1
  local out=$H12/logs/grid37_twin/d37/s${STRAT}_${TGT}_noassist${suf}
  grep -oE "plan=[0-9]+/s" $out/node.log 2>/dev/null | tail -3 | tr '\n' ' '; echo " <- $arm r$r"; sleep 5; }
for r in $(seq 1 $N); do run_one base10 $r; run_one damp $r; done
echo "[damp-ab] ALL DONE"
