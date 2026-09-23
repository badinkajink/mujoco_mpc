#!/usr/bin/env bash
# swap_planner.sh cem|icem -- agent_planner numeric 5 (CEM, the August strat-22 config) or 7 (iCEM, live since 08-26). Restart the node after.
set -e; D="$(cd "$(dirname "$0")" && pwd)"; X="$D/Lean_H12_Magpie.xml"
case "$1" in cem) sed -i 's|<numeric name="agent_planner" data="7"/>|<numeric name="agent_planner" data="5"/>|' "$X";; icem) sed -i 's|<numeric name="agent_planner" data="5"/>|<numeric name="agent_planner" data="7"/>|' "$X";; *) echo "usage: $0 cem|icem"; exit 2;; esac
cd "$D/../../../../build" && ninja copy_model_resources >/dev/null 2>&1; grep -o 'name="agent_planner" data="[^"]*"' "$X"
