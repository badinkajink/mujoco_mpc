#!/usr/bin/env bash
# est_bringup.sh -- one-command NATIVE (laptop) bringup of an estimator variant
# plus whatever sensor bridges it needs. The controller / safety layer / estop
# are NOT started here -- you run those yourself as usual.
#
#   ./est_bringup.sh v1                     # baseline estimator only
#   ./est_bringup.sh v3                     # v3 estimator only
#   ./est_bringup.sh v4                     # v4, no bridges (== v3 behaviour)
#   ./est_bringup.sh v4 lidar               # livox driver + FAST-LIO + lio_bridge + v4
#   ./est_bringup.sh v4 april               # realsense cams + tag_bridge + v4
#   ./est_bringup.sh v4 lidar april         # everything
#   ./est_bringup.sh v4 april -- --out-topic rt/sportmodestate_est
#                                           # anything after -- goes to the estimator
#
# Each background piece is health-gated (waits for its output topic to actually
# produce a message) before the next starts; the estimator runs in the
# FOREGROUND, so Ctrl-C here tears the whole set down. Logs land in
# ~/Desktop/h12/est_bringup_logs/<timestamp>/.
#
# ONE estimator at a time on the robot network -- they all publish
# rt/sportmodestate by default.

set -u
HS="$HOME/Desktop/h12/mujoco_mpc/mujoco_mpc/mjpc/deploy/helper_scripts"
VENV_PY="$HOME/Desktop/h12/h1_mujoco/.venv/bin/python"
VENV_SITE="$HOME/Desktop/h12/h1_mujoco/.venv/lib/python3.10/site-packages"
ROS_SETUP="/opt/ros/humble/setup.bash"
HAMS_SETUP="$HOME/Desktop/HAMS/core_ws/install/setup.bash"
BUNDLE="$HOME/Desktop/h12/table_tags/table_tag_bundle.yaml"

# ---------- args ------------------------------------------------------------
EST=""; WANT_LIDAR=0; WANT_APRIL=0; EST_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    v1|v3|v4) EST="$1" ;;
    lidar)    WANT_LIDAR=1 ;;
    april|aruco|tags) WANT_APRIL=1 ;;
    --) shift; EST_ARGS=("$@"); break ;;
    *) echo "unknown arg: $1  (usage: est_bringup.sh v1|v3|v4 [lidar] [april] [-- est-flags])"; exit 2 ;;
  esac
  shift
done
[ -z "$EST" ] && { echo "pick an estimator: v1 | v3 | v4"; exit 2; }
if [ "$EST" != "v4" ] && { [ $WANT_LIDAR -eq 1 ] || [ $WANT_APRIL -eq 1 ]; }; then
  echo "NOTE: only v4 consumes the aux bus -- bridges with $EST would publish into the void."
  hint="v4"; [ $WANT_LIDAR -eq 1 ] && hint="$hint lidar"; [ $WANT_APRIL -eq 1 ] && hint="$hint april"
  echo "      Refusing; use '$hint'."
  exit 2
fi
case "$EST" in
  v1) EST_FILE="base_estimator_node.py" ;;
  v3) EST_FILE="base_estimator_node_v3.py" ;;
  v4) EST_FILE="base_estimator_node_v4.py" ;;
esac

# 2026-08-13: for v4+APRIL, default the aux gate to ALWAYS unless the caller
# passed their own --aux-gate. The planted-regime deferral was protection
# against the July LIO *velocity* aux; april is position-only + 2s tau, and
# with the default gate the corrections sat unused through entire stands ->
# ~3.5 mm/s believed drift = the run-6/7 "march". Every real brace success
# (runs 8-15) and every bench since runs with always. LIDAR mode keeps the
# regime gate (velocity aux DID hurt stands).
if [ "$EST" = "v4" ] && [ $WANT_APRIL -eq 1 ] && [ $WANT_LIDAR -eq 0 ]; then
  has_gate=0
  for a in "${EST_ARGS[@]}"; do case "$a" in --aux-gate*) has_gate=1 ;; esac; done
  if [ $has_gate -eq 0 ]; then
    EST_ARGS+=(--aux-gate always)
    echo "[bringup] v4+april: defaulting '--aux-gate always' (pass your own --aux-gate to override)"
  fi
fi

LOGDIR="$HOME/Desktop/h12/est_bringup_logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOGDIR"
PIDS=()

cleanup() {
  trap - INT TERM EXIT
  echo; echo "[bringup] shutting down ${#PIDS[@]} background process group(s)..."
  # kill the whole PROCESS GROUP (-pid): 'ros2 run' wraps the real node, and
  # the livox node ignores SIGINT when detached -- escalate INT->TERM->KILL.
  # (a surviving livox node keeps the lidar's UDP stream and starves the next run)
  for p in "${PIDS[@]}"; do kill -INT -- "-$p" 2>/dev/null; done
  sleep 2
  for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null; done
  sleep 1
  for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null; done
  echo "[bringup] logs: $LOGDIR"
}
trap cleanup INT TERM EXIT

launch_bg() {  # launch_bg <name> <cmd...>
  # Each launch gets its OWN session/process group so cleanup can kill the
  # whole tree. setsid may FORK (then $! is a dead wrapper and its group id is
  # useless -- the bug that leaked fast_lio/livox), so the new session writes
  # its true group-leader pid to a file and cleanup kills THAT group.
  local name="$1"; shift
  echo "[bringup] starting $name  (log: $LOGDIR/$name.log)"
  setsid bash -c 'echo "$$" >"$1"; shift; exec "$@"' _ "$LOGDIR/$name.pgid" "$@" \
    >"$LOGDIR/$name.log" 2>&1 &
  sleep 0.5
  PIDS+=("$(cat "$LOGDIR/$name.pgid" 2>/dev/null || echo "$!")")
}

topic_alive() {  # one quick probe: is anyone already publishing this?
  timeout 5 ros2 topic echo --qos-profile sensor_data --once "$1" >/dev/null 2>&1
}

wait_topic() {  # wait_topic <topic> <timeout_s>
  # measure REAL elapsed time: a probe on a not-yet-existing topic fails in
  # ~1s (echo exits early), so counting 5s per attempt burned the whole
  # window before the driver even finished its sensor handshake.
  local topic="$1" tmo="$2" start=$SECONDS
  echo -n "[bringup] waiting for $topic "
  while [ $((SECONDS - start)) -lt "$tmo" ]; do
    if topic_alive "$topic"; then
      echo " OK"; return 0
    fi
    echo -n "."; sleep 2
  done
  echo " TIMEOUT (${tmo}s)"
  echo "[bringup] $topic never produced data -- check $LOGDIR/. Aborting."
  exit 1
}

# ---------- bridges ---------------------------------------------------------
if [ $WANT_LIDAR -eq 1 ] || [ $WANT_APRIL -eq 1 ]; then
  # The HAMS overlay was built inside docker as /home/code/core_ws -- its
  # install-space symlinks only resolve natively through this one link:
  if [ ! -e /home/code ]; then
    echo "[bringup] /home/code missing -- the docker-built HAMS overlay's symlinks"
    echo "          are broken natively. One-time fix (needs sudo):"
    echo "              sudo ln -s $HOME/Desktop/HAMS /home/code"
    exit 1
  fi
  # ROS setup scripts read variables they haven't set yet -> incompatible with
  # set -u; relax it just for the sourcing.
  set +u
  # shellcheck disable=SC1090
  source "$ROS_SETUP"; source "$HAMS_SETUP"
  set -u
  # Match the ROBOT's ROS docker world (docker-compose: rmw_cyclonedds_cpp +
  # ROS_DOMAIN_ID=1 + tuned cyclonedds.xml) -- on FastRTPS/domain-0 the laptop
  # and the robot-side drivers can never see each other.
  if [ ! -e /opt/ros/humble/lib/librmw_cyclonedds_cpp.so ]; then
    echo "[bringup] rmw_cyclonedds_cpp missing -- the robot's ROS stack runs CycloneDDS"
    echo "          on domain 1; install the matching RMW once:"
    echo "              sudo apt install ros-humble-rmw-cyclonedds-cpp"
    exit 1
  fi
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  # Domain must match where the robot-side drivers RUN: the onboard PC's HOST
  # env (camera launch, firmware topics) is domain 0; the HAMS docker is
  # domain 1. Default 0; override with EST_ROS_DOMAIN=1 for docker-launched.
  export ROS_DOMAIN_ID="${EST_ROS_DOMAIN:-0}"
  export CYCLONEDDS_URI="file://$HS/cyclonedds_laptop.xml"
  # the ros2 CLI daemon caches graph state per RMW/domain -- restart it so the
  # topic gates don't answer from a stale FastRTPS/domain-0 daemon
  ros2 daemon stop >/dev/null 2>&1 || true
fi

if [ $WANT_LIDAR -eq 1 ]; then
  # If the robot's ONBOARD bringup is already streaming the lidar, consume it
  # over DDS -- never run a second driver (the MID360 streams to ONE host).
  if topic_alive /livox/lidar; then
    echo "[bringup] /livox/lidar already published (robot-side driver) -- skipping local driver."
    LOCAL_LIVOX=0
  else
    LOCAL_LIVOX=1
  fi
fi
if [ $WANT_LIDAR -eq 1 ] && [ $LOCAL_LIVOX -eq 1 ]; then
  # Laptop-side lidar config (clone of the humanoid-validated h1_bringup one:
  # lidar .123.120, inverted mount roll 180) -- HAMS files stay untouched.
  # The config pins the HOST ip; it must match this laptop's address on the
  # robot wire (DHCP could hand out a different one after a re-plug).
  LIDAR_CFG="$HS/MID360_config_laptop.json"
  HOST_IP=$(python3 -c "import json;print(json.load(open('$LIDAR_CFG'))['MID360']['host_net_info']['point_data_ip'])")
  if ! ip -4 addr show | grep -q "inet ${HOST_IP}/"; then
    echo "[bringup] $LIDAR_CFG host ip is $HOST_IP but this machine doesn't"
    echo "          hold that address -- lidar data would stream into the void."
    echo "          Edit host_net_info there to your current 192.168.123.x IP."
    exit 1
  fi
  # mirror of h1_real_drivers.launch.py's inlined driver params, but with the
  # laptop config (the stock msg_MID360_launch.py reads the stock 192.168.1.x
  # sample config, which is not this robot)
  # liblivox_lidar_sdk_shared.so: the docker image installed Livox-SDK2 to
  # /usr/local; natively it lives in ~/.local (built 2026-07-30, no sudo).
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.local/lib"
  if ! ls "$HOME/.local/lib/liblivox_lidar_sdk_shared.so" >/dev/null 2>&1 && \
     ! ldconfig -p | grep -q liblivox_lidar_sdk_shared; then
    echo "[bringup] liblivox_lidar_sdk_shared.so not found -- build Livox-SDK2:"
    echo "          git clone --depth 1 https://github.com/Livox-SDK/Livox-SDK2 &&"
    echo "          cmake -B build -DCMAKE_INSTALL_PREFIX=\$HOME/.local && make -C build -j8 install"
    exit 1
  fi
  launch_bg livox_driver ros2 run livox_ros_driver2 livox_ros_driver2_node --ros-args \
    -p xfer_format:=1 -p multi_topic:=0 -p data_src:=0 -p publish_freq:=10.0 \
    -p output_data_type:=0 -p frame_id:=livox_frame \
    -p lvx_file_path:=/home/livox/livox_test.lvx \
    -p user_config_path:="$LIDAR_CFG" \
    -p cmdline_input_bd_code:=livox0000000001
  wait_topic /livox/lidar 30
fi
if [ $WANT_LIDAR -eq 1 ]; then
  # mirror of h1_navigation.launch.py's fast_lio node with the tuned config
  # (stock mapping.launch.py would use FAST_LIO's default params instead)
  launch_bg fast_lio ros2 run fast_lio fastlio_mapping --ros-args \
    --params-file "$HOME/Desktop/HAMS/core_ws/src/h1_bringup/config/mid360.yaml"
  wait_topic /Odometry 40
  # venv python (NOT system python3 + PYTHONPATH): unitree_sdk2py is an
  # EDITABLE install whose .pth only resolves under the venv's own site setup;
  # rclpy still imports via the ROS PYTHONPATH set by the sourced overlay.
  launch_bg lio_bridge "$VENV_PY" "$HS/lio_bridge_node.py" --twist-frame world
  sleep 2
fi

if [ $WANT_APRIL -eq 1 ]; then
  if grep -q "x: 0.000, y: 0.000" "$BUNDLE" 2>/dev/null; then
    echo "[bringup] $BUNDLE still has placeholder centres -- fill/verify it first."
    exit 1
  fi
  # The head D435i's USB is on the ROBOT's onboard PC -- normally its driver
  # runs THERE (h1_real_drivers.launch.py) and the laptop just consumes the
  # topic over DDS. Launch locally only if nothing is publishing AND this
  # machine has the driver installed (needs: sudo apt install ros-humble-realsense2-camera).
  if topic_alive /realsense/head/color/image_raw/compressed; then
    echo "[bringup] head camera already published (robot-side driver) -- skipping local launch."
  else
    if ! ros2 pkg prefix realsense2_camera >/dev/null 2>&1; then
      echo "[bringup] no /realsense/head/... publisher on the network, and this machine"
      echo "          lacks realsense2_camera. Either start the camera on the ROBOT"
      echo "          (its onboard bringup owns the head-cam USB), or install locally:"
      echo "              sudo apt install ros-humble-realsense2-camera"
      exit 1
    fi
    launch_bg realsense ros2 launch cl_realsense h12_rs_cams.launch.py
    wait_topic /realsense/head/color/image_raw/compressed 40
  fi
  # venv python for the same editable-install reason as lio_bridge above
  launch_bg tag_bridge "$VENV_PY" "$HS/tag_bridge_node.py" --bundle "$BUNDLE"
  sleep 2
fi

# ---------- estimator (foreground) ------------------------------------------
echo "[bringup] launching $EST ($EST_FILE) in the FOREGROUND -- Ctrl-C stops everything."
cd "$HS"
"$VENV_PY" "$HS/$EST_FILE" ${EST_ARGS[@]+"${EST_ARGS[@]}"}
