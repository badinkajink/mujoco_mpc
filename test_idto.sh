#!/bin/bash
# Quick test of IDTO planner in headless mode

set -e

TASK="Cartpole"
DURATION=5

echo "Testing IDTO planner on $TASK (${DURATION}s, headless)..."
echo ""

# Temporarily set IDTO as default planner for Cartpole
TASK_XML="mjpc/tasks/cartpole/task.xml"
cp "$TASK_XML" "${TASK_XML}.bak"

# Set planner to IDTO (id=7)
sed -i 's/<numeric name="agent_planner" data="[0-9]*"/<numeric name="agent_planner" data="7"/' "$TASK_XML"

# Also reduce iterations for faster testing
sed -i 's/<numeric name="agent_horizon" data="[0-9.]*"/<numeric name="agent_horizon" data="0.3"/' "$TASK_XML"

echo "Running testspeed with IDTO planner..."
build/bin/testspeed \
    --task="$TASK" \
    --total_time=$DURATION \
    --planner_thread=1 \
    --steps_per_planning_iteration=4

# Restore original
mv "${TASK_XML}.bak" "$TASK_XML"

echo ""
echo "Test complete!"
