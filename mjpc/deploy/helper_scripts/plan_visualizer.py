#!/usr/bin/env python3
"""plan_visualizer.py -- debug view of what the MJPC lower-body planner is thinking.

Renders TWO robots into one frame, live, and writes an mp4 on exit:

  * the MEASURED robot in its NORMAL colours -- joints straight off rt/lowstate,
    base pose from the RW-EKF estimator on rt/sportmodestate_est (the same signal
    path the deploy core itself plans from), and
  * a semi-transparent BLUE GHOST sweeping the planner's best trajectory, read off
    rt/mjpc/plan (the deploy core's planner thread publishes it as JSON -- see
    deploy_common.cc AppendPlanJson + the --plan_topic flag).

So the ghost is literally "where MJPC wants the robot to be"; the coloured robot is
where it actually is. The gap between them is the story.

By default (--ghost-mode sweep) the ghost's plan step cycles 0 -> end of horizon in
REAL TIME, so each ~1 s loop walks from "where MJPC thinks the robot is now" out to
"where it thinks the robot will be a second from now", and its alpha fades with that
lookahead (bright = near, faint = far).

WHAT "1:1 WITH THE ROBOT" MEANS -- READ BEFORE FILING A BUG
----------------------------------------------------------
Plan row 0 is NOT an approximation of the measured robot, it IS the measured robot:
Trajectory::Rollout seeds it with the planner's input state verbatim (trajectory.cc:130
`mju_copy(states.data(), state, dim_state)`), and the deploy core re-seeds that from
rt/lowstate + the estimator every control tick. A ghost pinned to step 0 is therefore a
tautology -- pixel-exact 1:1 forever, INCLUDING while the robot falls. That is why
sweep, not step 0, is the default.

But a near-1:1 ghost can also be CORRECT: over a 1.0 s horizon on a STABLE stand the
plan moves the base only ~1.8 cm and the 27 joints ~0.029 rad on average (measured off
the live CEM planner). A balancing robot's plan says "stay put", so the ghost SHOULD sit
near the robot. Judge this DURING a fall, in the first few hundred ms -- that is when
the plan's intended recovery diverges from what the robot actually does, and that gap is
the deploy gap (see deploy_common.h:92-110: "The plan was valid in the planner's model
and unexecutable in the node's"). A settled or fallen robot barely mj_steps, so the rows
collapse toward each other again and 1:1 returns -- correctly.

READ THIS BEFORE "FIXING" THE OFFSET BASE
-----------------------------------------
The ghost's BASE will generally NOT sit on the measured robot's base, and that is
CORRECT, not a bug: the plan carries the PLANNER's own base belief (its rollout
state), while the measured robot is drawn at the ESTIMATOR's. Divergence between
those two is a real, load-bearing signal -- arguably the most useful thing this
tool shows (it is how you see the planner solving for a robot that isn't where it
thinks it is). --align-ghost-base parks the ghost on the measured base to compare
POSTURE alone; it is OFF by default on purpose. Don't make it the default.

AUTHENTIC: reads only rt/lowstate + the estimator's rt/sportmodestate_est + the
planner's own plan. No simulator ground truth (no rt/sportmodestate truth topic,
no /robocasa/*) -- so this behaves identically against the sim and the real robot.

DDS, not rclpy: this is a pure unitree_sdk2py worker, like base_estimator_node.py
next to it. rclpy's rmw_cyclonedds and unitree_sdk2's bundled CycloneDDS both
export libddsc/libddscxx.so.0 and corrupt the heap when one process loads both --
the ROS parameter surface lives in the h12_deploy_mjpc shell that execs this.

Run (inside hams_ros, with the sim + bringup up):
  MUJOCO_GL=osmesa python3 mjpc/deploy/helper_scripts/plan_visualizer.py \
      --domain $ROS_DOMAIN_ID --out /tmp/mjpc_plan.mp4
"""
import argparse
import json
import os
import signal
import sys
import time

import numpy as np
import mujoco

# The iface auto-pin + quat helper are the estimator's, reused rather than forked
# (same directory, same DDS conventions). runpy.run_path does not put this script's
# directory on sys.path, so add it before the sibling import.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base_estimator_node import IMU_OFFSET, _pick_iface, _quat2mat  # noqa: E402

NJ = 27          # H1-2 handless actuated joints == lowstate motor order (motor_state[0:27]);
                 # motor_state is a fixed 35-element array -- rows 27..34 are not real joints.
NMOTOR_REAL = 27


def _model_scene_default():
    """The SAME staged task XML the controller loads (build tree, not source tree:
    the robot body is generated at build time), so geom ids + the qpos layout match
    the plan rows exactly. Mirrors estimator_node.py's resolution."""
    root = os.environ.get("MJPC_ROOT", "/home/code/mujoco_mpc")
    tasks = os.environ.get("MJPC_TASKS_DIR", os.path.join(root, "build", "mjpc", "tasks"))
    return os.path.join(tasks, "humanoid_bench", "stabilize", "Stabilize_H12_Magpie.xml")


def _parse_plan(raw, want_nq):
    """rt/mjpc/plan JSON -> (dict, (H, nq) qpos rows, (H,) times). Returns Nones on
    anything unexpected: a debug viewer must never take the run down over a malformed
    frame."""
    d = json.loads(raw)
    nq, H = int(d["nq"]), int(d["horizon"])
    if H <= 0:
        return None, None, None
    # DIVERGED ROLLOUT -> the tail rows are STALE, not merely wrong.
    # Trajectory::NoisyRollout stamps `horizon = steps` (trajectory.cc:118) BEFORE the
    # step loop, then bails on divergence (:179-183) WITHOUT rolling horizon back -- so
    # rows [t+1 .. H-1] still hold the PREVIOUS iteration's qpos while the message
    # advertises a full 101. Drawing those paints a confident ghost from a plan that
    # never existed.
    # This path is REACHABLE HERE, contrary to the usual "a diverged rollout scores
    # 1e6 and never wins" reasoning: that argument is about SamplingPlanner, whose
    # BestTrajectory() returns trajectory[winner] after an ascending sort by
    # total_return. This task runs CEM (agent_planner=5), and
    # CrossEntropyPlanner::BestTrajectory() returns &nominal_trajectory with NO
    # selection at all -- a diverged nominal is published verbatim. And it diverges
    # exactly when the robot is falling, which is when you are watching.
    if d.get("failure"):
        raise ValueError("planner reported failure=true (diverged rollout); its tail "
                         "rows are stale -- refusing to draw a ghost from it")
    q = np.asarray(d["qpos"], dtype=float)
    # The publisher walks traj->horizon rows (NOT the 512-row Allocate() capacity),
    # so this should always hold; check anyway -- a silent reshape of the wrong
    # length would draw a plausible, wrong ghost.
    if q.size != nq * H:
        raise ValueError(f"qpos has {q.size} values, expected nq*horizon = {nq}*{H}")
    if nq != want_nq:
        raise ValueError(f"plan nq={nq} != model nq={want_nq} -- plan and viewer "
                         "are on different models")
    t = np.asarray(d.get("times", []), dtype=float)
    if t.size != H:
        raise ValueError(f"times has {t.size} values, expected horizon = {H}")
    return d, q.reshape(H, nq), t


def main():
    # Under `ros2 launch` stdout is a pipe, not a tty, so Python block-buffers it and
    # every line below -- including the "no plan received" warning that is this tool's
    # main diagnostic -- would sit invisible in a 8 KB buffer for minutes. Line-buffer
    # so the launch log shows them as they happen.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:      # pre-3.7 / an already-wrapped stream
        pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=_model_scene_default(),
                    help="task XML; MUST be the one the controller loads (qpos layout parity)")
    ap.add_argument("--domain", type=int,
                    default=int(os.environ.get("ROS_DOMAIN_ID", "0")))
    ap.add_argument("--iface", default="", help="'' = auto-pin the robot subnet")
    ap.add_argument("--no-auto-iface", action="store_true",
                    help="skip the auto-pin and let the SDK autodetermine (twin/sim same-host)")
    ap.add_argument("--lowstate-topic", default="rt/lowstate")
    ap.add_argument("--plan-topic", default="rt/mjpc/plan")
    ap.add_argument("--sportstate-topic", default="rt/sportmodestate_est",
                    help="ESTIMATOR output (real-parity). Never point this at ground truth.")
    ap.add_argument("--out", default="/tmp/mjpc_plan.mp4")
    # 10, not 30: osmesa is a SOFTWARE rasteriser and this scene costs ~96 ms/frame
    # with shadows off (measured in hams_ros; ~207 ms with them on) -- about 10 fps,
    # near-identical at 640x480, so it is geometry/shadow-bound, not fill-bound. The
    # mp4's fps tag is fixed at open time, so a target the loop cannot hit produces a
    # video that PLAYS FAST (30 tag / 4 real = 7.5x) and silently lies about the
    # robot's dynamics. 10 is achievable -> playback is ~real-time.
    ap.add_argument("--fps", type=float, default=10.0,
                    help="render/write rate; decoupled from the 200 Hz lowstate. Keep at or "
                         "below what the renderer can sustain (~10 on osmesa) or the mp4 "
                         "plays faster than reality -- the exit line reports the shortfall.")
    ap.add_argument("--shadows", action="store_true",
                    help="re-enable shadows + reflections (2.2x slower on osmesa). Off by "
                         "default: they cost more than half the frame time and a ghost that "
                         "casts its own shadow only muddies the comparison.")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--camera", default="",
                    help="named model camera; '' = a free camera tracking the pelvis")
    # Defaults picked by rendering the stabilize scene at several azimuths: 45 puts the
    # task's table off to the side instead of across the robot's legs, so the lower body
    # (the half this controller actually drives) stays visible.
    ap.add_argument("--cam-azimuth", type=float, default=45.0)
    ap.add_argument("--cam-elevation", type=float, default=-10.0)
    ap.add_argument("--cam-distance", type=float, default=2.6)
    ap.add_argument("--ghost-rgba", default="0.25,0.45,1.0,0.35",
                    help="the blue ghost's colour (r,g,b,a). In sweep mode the alpha here is "
                         "the NEAR alpha; --ghost-alpha-far is the far end of the ramp.")
    # WHY sweep IS THE DEFAULT, and why step 0 was the worst possible one.
    # Trajectory::Rollout seeds row 0 with the planner's input state verbatim
    # (trajectory.cc:130 `mju_copy(states.data(), state, dim_state)`), and the deploy
    # core re-seeds that from rt/lowstate + the estimator every control tick. So plan
    # row 0 IS the measured robot, bit-for-bit -- a ghost pinned there is a tautology
    # that renders pixel-exact 1:1 forever, INCLUDING while the robot falls. The plan's
    # information lives entirely in rows 1..H-1. Sweeping is what shows it.
    ap.add_argument("--ghost-mode", choices=("sweep", "single"), default="sweep",
                    help="sweep (default): one ghost whose plan step cycles 0 -> end of horizon "
                         "in real time, so each loop walks from 'now' out to the full lookahead. "
                         "single: a static ghost at --ghost-step.")
    ap.add_argument("--sweep-rate", type=float, default=1.0,
                    help="sweep speed as a multiple of real time (1.0 = the ghost advances one "
                         "second of PLAN per second of wall, i.e. it moves at the speed the plan "
                         "actually predicts). Indexed off the plan's own `times`, never a "
                         "hardcoded dt -- agent_timestep is a task XML numeric and can change.")
    ap.add_argument("--ghost-alpha-far", type=float, default=0.15,
                    help="sweep mode: ghost alpha at the END of the horizon; it ramps from "
                         "--ghost-rgba's alpha (near) down to this. This is what makes a single "
                         "frame legible -- bright ghost = near-term, faint = a full second out. "
                         "Without it the sweep is ambiguous when you scrub frame by frame.")
    ap.add_argument("--ghost-step", type=int, default=-1,
                    help="single mode only: which plan step to pose the ghost at. -1 = the LAST "
                         "step (the full lookahead). Do not use 0: that is the measured state by "
                         "construction and draws a ghost on top of the robot.")
    ap.add_argument("--ghost-trail", type=int, default=0,
                    help=">0: also draw every Nth later step as a fading trail (0 = single ghost)")
    ap.add_argument("--plan-stale-sec", type=float, default=1.0,
                    help="draw NO ghost if the newest plan is older than this (WALL s). A frozen "
                         "ghost that still looks live is worse than no ghost.")
    ap.add_argument("--align-ghost-base", action="store_true",
                    help="park the ghost on the MEASURED base to compare posture alone. OFF by "
                         "default: the planner-vs-estimator base gap is real signal (see docstring)")
    ap.add_argument("--fallback-z", type=float, default=0.98,
                    help="pelvis height used until the estimator's sportmodestate arrives")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="stop after this many WALL seconds (0 = until signalled). Guards an "
                         "unattended run from filling the disk.")
    a = ap.parse_args()

    ghost_rgba = np.array([float(x) for x in a.ghost_rgba.split(",")], dtype=np.float32)
    if ghost_rgba.size != 4:
        raise SystemExit("--ghost-rgba wants 4 comma-separated floats (r,g,b,a)")

    # osmesa: no GPU needed, and unlike egl it works under both `docker run --gpus`
    # and compose. The shell sets this too; default it here so a bare run works.
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import cv2  # after MUJOCO_GL, and only once we know we're rendering

    if not os.path.isfile(a.scene):
        raise SystemExit(f"[viz] scene not found: {a.scene} -- is MJPC_TASKS_DIR right "
                         "(the BUILD tree, not the source tree)?")
    model = mujoco.MjModel.from_xml_path(a.scene)
    nq = model.nq
    if nq < 7 + NJ:
        raise SystemExit(f"[viz] model nq={nq} < 7+{NJ}: not an H1-2 free-base model")

    # ONE model, TWO mjData. The ghost MUST share the measured robot's model: mesh
    # geoms are drawn by display-list index (baseMesh + geom->dataid), so geoms from
    # a second model would render the wrong meshes.
    d_meas = mujoco.MjData(model)
    d_ghost = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    for d in (d_meas, d_ghost):
        if home >= 0:
            mujoco.mj_resetDataKeyframe(model, d, home)
        else:
            mujoco.mj_resetData(model, d)
    # Task/object slots past the robot (qpos[34:]) keep their home defaults, exactly
    # as the deploy core's scratch mjData does.

    # The task XML declares a 640x480 offscreen framebuffer, and Renderer REFUSES any
    # size above it ("Image width W > framebuffer width 640"). Raise it on OUR OWN
    # in-memory model instead of editing the shared task XML -- the controller loads
    # that file too, and this is a debug viewer's private copy.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, a.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, a.height)
    renderer = mujoco.Renderer(model, height=a.height, width=a.width, max_geom=10000)
    vopt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    cam = mujoco.MjvCamera()
    cam_id = -1
    if a.camera:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, a.camera)
        if cam_id < 0:
            print(f"[viz] WARNING: no camera '{a.camera}' in the model -> free camera")
    if cam_id >= 0:
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = cam_id
    else:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth, cam.elevation, cam.distance = (
            a.cam_azimuth, a.cam_elevation, a.cam_distance)

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

    if a.no_auto_iface:
        iface, why = None, "forced autodetermine (--no-auto-iface)"
    else:
        iface, why = _pick_iface(a.iface)
    print(f"[viz] DDS interface = {iface or 'autodetermine'} ({why}), domain {a.domain}")
    if iface:
        ChannelFactoryInitialize(a.domain, iface)
    else:
        ChannelFactoryInitialize(a.domain)

    # Cache the latest of each; NEVER render inside a DDS callback (that would stall
    # the reader thread for the whole render).
    latest = {"ls": None, "sp": None, "plan": None, "plan_rx": 0.0}
    ls_sub = ChannelSubscriber(a.lowstate_topic, LowState_)
    ls_sub.Init(lambda m: latest.__setitem__("ls", m), 10)
    sp_sub = ChannelSubscriber(a.sportstate_topic, SportModeState_)
    sp_sub.Init(lambda m: latest.__setitem__("sp", m), 10)
    pl_sub = ChannelSubscriber(a.plan_topic, String_)
    # stamp the arrival: sweep-mode staleness is judged on WALL time (the plan's own
    # `stamp` is planner/plant time and freezes exactly when the core wedges -- the one
    # case we need to detect).
    def _on_plan(m):
        latest["plan"] = m.data
        latest["plan_rx"] = time.monotonic()
    pl_sub.Init(_on_plan, 1)

    stop = {"v": False}

    def _on_sig(signum, _frame):
        stop["v"] = True     # let the loop unwind so the mp4 gets finalised

    signal.signal(signal.SIGINT, _on_sig)
    signal.signal(signal.SIGTERM, _on_sig)

    out_path = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             a.fps, (a.width, a.height))
    if not writer.isOpened():
        raise SystemExit(f"[viz] could not open {out_path} for writing")

    print(f"[viz] measured='{a.lowstate_topic}' + '{a.sportstate_topic}' | "
          f"plan='{a.plan_topic}' | -> {out_path} @ {a.fps:.0f} fps "
          f"{a.width}x{a.height}"
          + (f" | max {a.max_seconds:.0f}s" if a.max_seconds > 0 else ""))

    frames = 0
    plan_seen = False
    warned = {"plan": False}
    period = 1.0 / a.fps if a.fps > 0 else 0.0
    t_start = time.monotonic()
    next_frame = t_start
    try:
        while not stop["v"]:
            now = time.monotonic()
            if now < next_frame:
                time.sleep(min(next_frame - now, 0.005))
                continue
            next_frame += period
            if a.max_seconds > 0 and (now - t_start) >= a.max_seconds:
                print(f"[viz] --max-seconds {a.max_seconds:.0f} reached")
                break

            ls, sp, plan_raw = latest["ls"], latest["sp"], latest["plan"]
            if ls is None:
                continue          # nothing measured yet -> don't write a blank frame

            # ---- measured robot: joints from lowstate, base from the estimator ----
            quat = np.array([ls.imu_state.quaternion[k] for k in range(4)])  # wxyz (MuJoCo)
            for i in range(NMOTOR_REAL):
                d_meas.qpos[7 + i] = ls.motor_state[i].q
            d_meas.qpos[3:7] = quat
            if sp is not None:
                # pelvis = imu_site - R*kImuOffset: the estimator publishes the IMU-SITE
                # pose (twin convention), and the deploy core backs the pelvis out the
                # same way (deploy_common.cc fill_state). Both ends must agree or the
                # base is wrong.
                site_p = np.array([sp.position[k] for k in range(3)])
                d_meas.qpos[0:3] = site_p - _quat2mat(quat) @ IMU_OFFSET
            else:
                d_meas.qpos[0:3] = (0.0, 0.0, a.fallback_z)

            # ---- ghost: the planner's best trajectory ----
            # ALWAYS index the NEWEST plan; never latch one and play it out. Plans land
            # at ~20 Hz, so ~20 refreshes happen per 1 s sweep. Receding-horizon plans
            # are similar frame-to-frame so the sweep still reads smooth -- and when it
            # DOES jump, that jump is a real planner instability worth seeing. Latching
            # would instead show a plan up to a full second stale by each cycle's end.
            have_ghost = False
            rows = ptimes = None
            if plan_raw is not None and (now - latest["plan_rx"]) <= a.plan_stale_sec:
                try:
                    _, rows, ptimes = _parse_plan(plan_raw, nq)
                    have_ghost = rows is not None
                except (ValueError, KeyError, json.JSONDecodeError) as e:
                    if not warned["plan"]:
                        print(f"[viz] WARNING: unusable plan on '{a.plan_topic}': {e}")
                        warned["plan"] = True
            if have_ghost and not plan_seen:
                span = float(ptimes[-1] - ptimes[0])
                print(f"[viz] first plan: horizon={rows.shape[0]} nq={rows.shape[1]} "
                      f"lookahead={span:.3f}s mode={a.ghost_mode}")
                plan_seen = True

            # mjv_addGeoms reads geom_xpos/xmat -> the kinematics MUST be current.
            mujoco.mj_forward(model, d_meas)
            if cam.type == mujoco.mjtCamera.mjCAMERA_FREE:
                # track the pelvis, framed slightly high so the torso doesn't ride the edge
                cam.lookat[:] = d_meas.qpos[0:3] + np.array([0.0, 0.0, 0.1])
            renderer.update_scene(d_meas, camera=cam, scene_option=vopt)
            # update_scene repopulates the scene each frame, so re-apply per frame.
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1 if a.shadows else 0
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1 if a.shadows else 0

            if have_ghost:
                H = rows.shape[0]
                # -1 (the default) = the LAST step. Note the old clamp was
                # min(max(step, 0), H-1), which silently folded -1 onto 0 -- i.e. onto
                # the measured state -- so the "last step" convention never worked.
                base_step = (H - 1) if a.ghost_step < 0 else min(a.ghost_step, H - 1)
                frac = 0.0
                if a.ghost_mode == "sweep":
                    # Index off the plan's OWN times, not a hardcoded dt: agent_timestep
                    # is a task XML numeric. sweep_t is free-running wall time scaled by
                    # --sweep-rate, wrapped over the horizon's true span, so at rate 1.0
                    # the ghost advances one second of plan per second of wall -- it
                    # moves at the speed the plan actually predicts.
                    span = float(ptimes[-1] - ptimes[0])
                    if span > 0:
                        sweep_t = ((now - t_start) * a.sweep_rate) % span
                        base_step = int(np.argmin(np.abs(ptimes - (ptimes[0] + sweep_t))))
                        frac = sweep_t / span
                steps = [base_step]
                if a.ghost_trail > 0:
                    steps += list(range(steps[0] + a.ghost_trail, H, a.ghost_trail))
                for n, t in enumerate(steps):
                    d_ghost.qpos[:] = rows[t]
                    if a.align_ghost_base:      # OFF by default -- see the docstring
                        d_ghost.qpos[0:7] = d_meas.qpos[0:7]
                    # The ghost is RENDER-ONLY: mjv_addGeoms reads geom_xpos/xmat, which
                    # mj_kinematics computes. mj_forward would additionally solve the
                    # dynamics nothing here consumes.
                    mujoco.mj_kinematics(model, d_ghost)
                    n0 = renderer.scene.ngeom
                    mujoco.mjv_addGeoms(model, d_ghost, vopt, pert,
                                        mujoco.mjtCatBit.mjCAT_DYNAMIC, renderer.scene)
                    # Tint ONLY the geoms we just appended -- [0, n0) is the measured
                    # robot and must keep its normal colours.
                    rgba = ghost_rgba.copy()
                    if a.ghost_mode == "sweep" and not n:
                        # Encode LOOKAHEAD as alpha: bright at t+0, faint at the end of
                        # the horizon. This is what makes sweep mode legible -- the
                        # ghost's clock is decoupled from the robot's, so without it a
                        # single frame is ambiguous about how far ahead the ghost is.
                        rgba[3] = ghost_rgba[3] + (a.ghost_alpha_far - ghost_rgba[3]) * frac
                    if n:      # trail steps fade with distance into the future
                        rgba[3] *= max(0.25, 1.0 - n / max(len(steps) - 1, 1))
                    for i in range(n0, renderer.scene.ngeom):
                        g = renderer.scene.geoms[i]
                        g.rgba[:] = rgba
                        # matid = -1 is MANDATORY, not cosmetic: setMaterial keeps the
                        # material when mjVIS_TEXTURE && matid >= 0, and the material
                        # then OVERRIDES rgba -> a ghost in the robot's normal colours.
                        # (MJPC's own Agent::ModifyScene swaps matid for the same reason.)
                        g.matid = -1

            frame = renderer.render()
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frames += 1
            if frames % (int(a.fps) * 10 or 300) == 0:
                print(f"[viz] {frames} frames ({(now - t_start):.0f}s wall)"
                      + ("" if plan_seen else f" -- STILL NO PLAN on '{a.plan_topic}'"),
                      flush=True)
    except KeyboardInterrupt:
        # SIGINT from `ros2 launch` lands here if it beats the handler above.
        pass
    finally:
        # An unreleased VideoWriter leaves a truncated, unplayable file -- this
        # release() is the whole reason the loop is wrapped.
        writer.release()
        wall = time.monotonic() - t_start
        dur = frames / a.fps if a.fps > 0 else 0.0
        achieved = frames / wall if wall > 0 else 0.0
        size = os.path.getsize(out_path) if os.path.isfile(out_path) else 0
        print(f"[viz] wrote {out_path}: {frames} frames, {dur:.1f}s of video from "
              f"{wall:.0f}s wall, {size/1e6:.2f} MB", flush=True)
        # The mp4's fps tag was fixed at open time. If the loop could not keep up, the
        # file plays FAST -- say so rather than hand over a video that misrepresents
        # the robot's timing.
        if frames > 0 and a.fps > 0 and achieved < 0.8 * a.fps:
            print(f"[viz] NOTE: rendered {achieved:.1f} fps but the mp4 is tagged "
                  f"{a.fps:.0f} fps -> it plays ~{a.fps/achieved:.1f}x FASTER than real "
                  f"time. Re-run with --fps {achieved:.0f} (or params fps: {achieved:.0f}) "
                  "for real-time playback.", flush=True)
        if not plan_seen:
            print("[viz] WARNING: no plan was ever received on "
                  f"'{a.plan_topic}' -- is the controller core running with "
                  "--plan_topic? (controller_launcher.py passes it)", flush=True)


if __name__ == "__main__":
    main()
