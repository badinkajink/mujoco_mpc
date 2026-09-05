#!/usr/bin/env bash
# Wait for the A/B sweep, publish, then wait for the lead sweep and publish again.
set -u
cd /home/humanoid/Programs/Humanoid_Simulation/mujoco_mpc
S=studies/table_height
for i in $(seq 1 900); do grep -q "A/B done" $S/ab.log 2>/dev/null && break; sleep 30; done
echo "=== A/B finished $(date +%H:%M) ==="
$S/publish_pose.sh
echo "=== POSEDONE $(date +%H:%M) ==="
for i in $(seq 1 900); do grep -q "lead sweep done" $S/lead.log 2>/dev/null && break; sleep 30; done
echo "=== lead finished $(date +%H:%M) ==="
$S/publish_pose.sh
echo "=== LEADDONE $(date +%H:%M) ==="
