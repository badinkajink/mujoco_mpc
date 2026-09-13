#!/usr/bin/env bash
# sync_bridge_to_robot.sh -- 2026-09-12: push the laptop's tag_bridge_node.py (multi-id block tags, detection age,
# --object-ambig-ratio) to the robot PC and make lean_bringup.sh launch it with the block-tag list, so the user's
# usual two launches on the robot PC bring up everything strat 11/12 need. Idempotent; backs up what it replaces.
#   ./sync_bridge_to_robot.sh          # do it
#   ./sync_bridge_to_robot.sh --check  # only report
set -euo pipefail
HS="$(cd "$(dirname "$0")" && pwd)"
SSH="$HOME/.local/bin/ssh_unitree"
RB=/home/unitree/HAMS/mujoco_mpc/mjpc/deploy/helper_scripts/tag_bridge_node.py
BR=/home/unitree/lean_deploy/lean_bringup.sh
IDS="30,31,32,33,34"
local_md5=$(md5sum "$HS/tag_bridge_node.py" | cut -c1-32)
echo "[sync] laptop bridge md5 $local_md5"
remote_md5=$("$SSH" "md5sum $RB | cut -c1-32" 2>/dev/null | tail -1 || true)
echo "[sync] robot  bridge md5 ${remote_md5:-<unreachable>}"
[ -z "$remote_md5" ] && { echo "[sync] robot PC unreachable -- power it on and rerun"; exit 2; }
line=$("$SSH" "grep -n 'tag_bridge_node.py' $BR | grep -v '^\s*#' | head -3" 2>/dev/null | tail -3)
echo "[sync] bringup launch line(s):"; echo "$line"
if [ "${1:-}" = "--check" ]; then exit 0; fi
if [ "$local_md5" != "$remote_md5" ]; then
  "$SSH" "cp -n $RB $RB.bak_pre_multiid_0912 2>/dev/null || true"
  "$SSH" "cat > $RB" < "$HS/tag_bridge_node.py"
  echo "[sync] bridge copied (backup: $RB.bak_pre_multiid_0912)"
  "$SSH" "cd $(dirname $RB) && /home/unitree/h12_cerg_controller/.venv/bin/python3 -m py_compile tag_bridge_node.py && echo '[sync] robot copy compiles'"
else
  echo "[sync] bridge already identical"
fi
# bringup: add the id list to the bridge launch line(s) that lack it
"$SSH" "grep -q -- '--object-tag-id $IDS --object-ambig-ratio 1.0' $BR && echo '[sync] bringup already has --object-tag-id $IDS' || { cp -n $BR $BR.bak_pre_multiid_0912; sed -i -E '/tag_bridge_node.py/{/^\s*#/!{/--object-tag-id/!s/(tag_bridge_node.py[^\\\\]*)/\1 --object-tag-id $IDS --object-ambig-ratio 1.0/}}' $BR; grep -n 'tag_bridge_node.py' $BR | grep -v '^\s*#' | head -3; echo '[sync] bringup patched (backup $BR.bak_pre_multiid_0912)'; }"
echo "[sync] done. Node side (laptop): h12_control_node built 2026-09-12 16:32 already filters the old servo bus to id 30 and feeds ids 30-34 to the centroid bus."
