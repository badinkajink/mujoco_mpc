#!/usr/bin/env bash
# 3-second DDS read: est pelvis vs head-cam abs anchor. Prints CONVERGED when gap < 3 cm.
# Run any time (between runs, after an estop) before launching the node.
exec "$HOME/Desktop/h12/h1_mujoco/.venv/bin/python" "$(dirname "$0")/est_check.py" "$@"
