#!/usr/bin/env bash
# A/B: does Allen's `brace_com_hold` term rescue the TOO-HIGH slab?
#
# WHY. At face 1.085 all 3 seeds fall BACKWARD -- CoM peaks at -85 mm, i.e.
# behind the toes -- with ~1 N of table contact. lean.cc's `brace_com_hold`
# comment describes that exact signature ("every backward fall of hp133-161 shows
# the CoM walking from +5 cm to -10 cm at a rung fire") and adds a one-sided
# "keep the CoM at least this far ahead of midfoot while an arm is on the table"
# term. The numeric is ABSENT from Lean_H12_Magpie.xml, so it defaults to 0 = off.
#
# This is goal 4's first rung (existing cost terms before new ones, standoff last).
#
# SAFETY: only the BUILD COPY of the model is edited, never the source XML, and it
# is restored on every exit path. A rebuild regenerates it anyway.
set -u
cd /home/humanoid/Programs/Humanoid_Simulation/mujoco_mpc
S=studies/table_height
M=build_cmake/mjpc/tasks/humanoid_bench/lean/Lean_H12_Magpie.xml
BK=$M.abbackup
H=1.085

restore() { [ -f "$BK" ] && mv -f "$BK" "$M" && echo "restored $M"; }
trap restore EXIT INT TERM

# wait for the long-cap sweep to finish so we never run two benches at once
for i in $(seq 1 240); do
  grep -q ALLDONE $S/finish.log 2>/dev/null && break
  sleep 30
done

cp -f "$M" "$BK"
mkdir -p $S/runs/bracehold
for ARM in off on; do
  cp -f "$BK" "$M"
  if [ "$ARM" = on ]; then
    python3 - "$M" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
anchor = '<numeric name="com_cap_fwd" data="0.145"/>'
assert s.count(anchor) == 1, "com_cap_fwd anchor not unique"
s = s.replace(anchor, anchor +
  '\n        <numeric name="brace_com_hold" data="0.05"/>', 1)
open(p, 'w').write(s)
print("brace_com_hold=0.05 injected")
PY
  fi
  for SEED in 0 1 2; do
    OUT=$S/runs/bracehold/${ARM}_s${SEED}.csv
    echo "=== arm=$ARM seed=$SEED $(date +%H:%M) ==="
    systemd-run --user --scope --quiet -p CPUQuota=700% -p MemoryMax=6G nice -n 15 \
      build_cmake/bin/lean_bench --task "Lean H12 Magpie" --strategy 25 \
      --table_h $H --seed $SEED --total_time 90 --threads 6 --spp 3 \
      --out "$OUT" > $S/runs/bracehold/${ARM}_s${SEED}.log 2>&1
    grep -h "bench-summary" $S/runs/bracehold/${ARM}_s${SEED}.log | sed 's/task=Lean H12 Magpie //;s/enter=.*//'
  done
done
restore; trap - EXIT

{
  echo ""
  echo "## brace_com_hold A/B at ${H} m (3 seeds each), $(date '+%Y-%m-%d %H:%M')"
  echo ""
  echo "Baseline = the shipped model (numeric absent, term off)."
  echo "Treatment = brace_com_hold 0.05, injected into the BUILD copy only."
  echo ""
  echo '```'
  grep -h "bench-summary" $S/runs/bracehold/*.log 2>/dev/null \
    | sed 's/task=Lean H12 Magpie //;s/enter=.*//' || echo "(no runs)"
  echo '```'
} >> $S/STATUS.md
echo "=== ABDONE $(date +%H:%M) ==="
