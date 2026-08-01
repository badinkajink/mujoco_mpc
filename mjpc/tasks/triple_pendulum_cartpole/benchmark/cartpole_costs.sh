#!/usr/bin/env bash
# Average cost per step on the stock Cartpole task, per planner.
#
# Cartpole is the smallest task in the tree that all of these planners solve, so
# it is where a regression in one of them shows up as a number rather than as a
# behaviour. It is a sanity check, not the headline: aggregate cost per step
# cannot distinguish outcomes on a task with obstacles, which is why the
# corridor benchmark exists. See ../README.md.
#
# testspeed has no --planner flag -- it reads `agent_planner` from the model --
# so this patches the built copy of cartpole/task.xml in place and restores it
# afterwards. It edits build output, never the source tree.
#
# Usage: cartpole_costs.sh [total_time] [repeats] [planner_threads]
set -u

BINDIR=${BINDIR:-./build/bin}
# The binary resolves models against build/mjpc/tasks, not build/bin/tasks.
XML=${XML:-./build/mjpc/tasks/cartpole/task.xml}
TOTAL_TIME=${1:-10}
REPEATS=${2:-3}
THREADS=${3:-6}

[ -f "$XML" ] || { echo "no $XML; build the mjpc target first"; exit 1; }

BACKUP=$(mktemp)
cp "$XML" "$BACKUP"
restore() { cp "$BACKUP" "$XML"; rm -f "$BACKUP"; }
trap restore EXIT

grep -q 'name="agent_planner"' "$XML" || {
  echo "cartpole task.xml has no agent_planner numeric; cannot select planners"
  exit 1
}

# label:planner_index:extra_numeric_edits (sed expressions, applied after the
# planner is set)
VARIANTS=(
  "sampling:0:"
  "cross_entropy:5:"
  "pso:7:s/pso_publish_evaluated\" data=\"[0-9.]*\"/pso_publish_evaluated\" data=\"1\"/"
  "pso_stock:7:s/pso_publish_evaluated\" data=\"[0-9.]*\"/pso_publish_evaluated\" data=\"0\"/"
  "annealed_sampling:8:"
  "random_sampling:9:"
)

printf "%-20s %s\n" "planner" "average cost per step (lower is better)"
for v in "${VARIANTS[@]}"; do
  label=${v%%:*}; rest=${v#*:}
  idx=${rest%%:*}; extra=${rest#*:}

  cp "$BACKUP" "$XML"
  sed -i "s/name=\"agent_planner\" data=\"[0-9.]*\"/name=\"agent_planner\" data=\"$idx\"/" "$XML"
  # Equal rollout budget. Cartpole's XML sets neither, and the library defaults
  # disagree: PSO defaults to 20 particles while the samplers default to 10, so
  # without this the table compares budgets rather than planners.
  sed -i "s|<custom>|<custom>\n    <numeric name=\"sampling_trajectories\" data=\"10\" />\n    <numeric name=\"pso_num_particles\" data=\"10\" />\n    <numeric name=\"pso_publish_evaluated\" data=\"1\" />|" "$XML"
  [ -n "$extra" ] && sed -i "$extra" "$XML"

  costs=()
  for ((k = 0; k < REPEATS; k++)); do
    c=$($BINDIR/testspeed --task=Cartpole --total_time="$TOTAL_TIME" \
          --planner_thread="$THREADS" 2>&1 |
        awk '/Average cost per step/ {print $NF}')
    costs+=("${c:-NA}")
  done
  printf "%-20s %s\n" "$label" "${costs[*]}"
done
