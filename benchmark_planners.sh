#!/bin/bash
# Benchmark IDTO vs other planners in headless mode
# Usage: ./benchmark_planners.sh [task_name] [duration]

TASK=${1:-"Cartpole"}
DURATION=${2:-10}
THREADS=1

echo "============================================"
echo "Benchmarking Planners on: $TASK"
echo "Duration: ${DURATION}s"
echo "============================================"
echo ""

# Build if needed
if [ ! -f "build/bin/testspeed" ]; then
    echo "Building testspeed..."
    ninja -C build testspeed
fi

# Function to run benchmark with specific planner
run_benchmark() {
    local planner_id=$1
    local planner_name=$2
    local task_file=$3

    echo "Running $planner_name (planner=$planner_id)..."

    # Temporarily modify task XML to use this planner
    if [ -f "$task_file" ]; then
        # Backup original
        cp "$task_file" "${task_file}.bak"

        # Replace planner value
        sed -i "s/<numeric name=\"agent_planner\" data=\"[0-9]*\"/<numeric name=\"agent_planner\" data=\"$planner_id\"/" "$task_file"
    fi

    # Run benchmark
    build/bin/testspeed \
        --task="$TASK" \
        --total_time=$DURATION \
        --planner_thread=$THREADS \
        --steps_per_planning_iteration=4 \
        2>&1 | tee "/tmp/bench_${planner_name}.log"

    # Extract cost
    local cost=$(grep "Average cost" "/tmp/bench_${planner_name}.log" | awk '{print $NF}')
    local time=$(grep "Total wall time" "/tmp/bench_${planner_name}.log" | grep -oP '\d+\.\d+(?=\sx realtime)')

    echo "$planner_name: Cost=$cost, Realtime factor=$time"
    echo ""

    # Restore original
    if [ -f "${task_file}.bak" ]; then
        mv "${task_file}.bak" "$task_file"
    fi
}

# Find task XML file
TASK_FILE=$(find mjpc/tasks -name "task.xml" | xargs grep -l "$TASK" | head -1)

if [ -z "$TASK_FILE" ]; then
    # Try cartpole specifically
    TASK_FILE="mjpc/tasks/cartpole/task.xml"
fi

echo "Using task file: $TASK_FILE"
echo ""

# Run benchmarks for different planners
# 0=Sampling, 1=Gradient, 2=iLQG, 3=iLQS, 7=IDTO

echo "=== iLQG (Baseline) ==="
run_benchmark 2 "iLQG" "$TASK_FILE"

echo "=== IDTO (New) ==="
run_benchmark 7 "IDTO" "$TASK_FILE"

echo "=== Sampling (Reference) ==="
run_benchmark 0 "Sampling" "$TASK_FILE"

echo ""
echo "============================================"
echo "Benchmark Complete!"
echo "============================================"
echo ""
echo "Results summary:"
grep "Average cost" /tmp/bench_*.log | sed 's/.*bench_/  /' | sed 's/.log:/: /'
