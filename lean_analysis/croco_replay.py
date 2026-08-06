#!/usr/bin/env python3
"""Replay a crocoddyl plan in MuJoCo, score it, and render video.

crocoddyl solved the plan against RIGID, PRESCRIBED contacts on a Pinocchio
model.  MuJoCo has soft contacts, a real table it can collide with anywhere, and
position-servo actuators.  Whether the plan survives that is the only question
that matters, and it is not answerable from the solver's own cost.

TWO THINGS THIS GETS RIGHT, both of which have burned this study before:

 1. THE ACTUATORS ARE POSITION SERVOS, not torque sources.  Every actuator is
    `kp = 150, kd = 5` (read off the model, not assumed), applying

        tau = kp * (ctrl - q) - kd * qvel

    so writing the plan's TORQUES into d.ctrl commands a joint angle of a few
    hundred radians, and writing nothing at all leaves the robot commanded to
    q = 0 while the reported contact forces describe a robot that is not the one
    being planned for.  The faithful replay inverts the servo law at the planned
    state:

        ctrl = q_plan + (tau_plan + kd * v_plan) / kp

    `--mode position` (ctrl = q_plan, i.e. just track the trajectory) is kept as
    a comparison, because it is what a WBC handoff would actually do.

 2. THE PLAN IS AT dt = 0.02 AND MuJoCo RUNS AT 0.002.  Each plan node is held
    for the intervening substeps rather than one MuJoCo step per node, which
    would replay a 2 s trajectory in 0.2 s and call the resulting flailing a
    tracking failure.

usage: croco_replay.py --mode elbow+forearm [--ctrl torque|position] [--video]
"""

import argparse
import json
import os

import numpy as np

import croco_bridge as cb          # first: sets RTLD_GLOBAL (see that module)
import contact_select as cs
import mujoco

W, H = 960, 720


def servo_gains(m):
    """(kp, kd) per actuator, read off the MJCF position actuators."""
    kp = m.actuator_gainprm[:m.nu, 0].copy()
    kd = -m.actuator_biasprm[:m.nu, 2].copy()
    if not np.allclose(-m.actuator_biasprm[:m.nu, 1], kp):
        raise RuntimeError("actuators are not the expected position servos "
                           "(biasprm[1] != -gainprm[0]); check the model")
    return kp, kd


def replay(mode_tag, ctrl_mode="torque", dt_plan=0.02, run_dir="runs/2026-08-05_session12/croco",
           video=None, fps=30, speed=1.0):
    xs = np.load(os.path.join(run_dir, f"xs_{mode_tag}.npy"))
    us = np.load(os.path.join(run_dir, f"us_{mode_tag}.npy"))
    with open(os.path.join(run_dir, f"plan_{mode_tag}.json")) as fh:
        plan = json.load(fh)
    subset = plan["subset"]

    m, d = cs.load()
    kp, kd = servo_gains(m)
    nsub = max(1, int(round(dt_plan / m.opt.timestep)))

    # start exactly where the plan starts, table included
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, plan["start"])
    qpos_full = m.key_qpos[kid].copy()
    nq = cb.NQ_ROBOT
    d.qpos[:] = cb.pin_to_mj(xs[0][:nq], qpos_full)
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)

    renderer = frames = None
    if video:
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, W)
        m.vis.global_.offheight = max(m.vis.global_.offheight, H)
        renderer = mujoco.Renderer(m, H, W, max_geom=m.ngeom + 64)
        frames = []
        frame_every = max(1, int(round(speed / (fps * dt_plan))))

    tbl = cs.bid(m, "table")
    brace_bodies = {s: cs.bid(m, cs.SITES[s][0]) for s in subset}
    feet = [cs.bid(m, f) for f in cs.FEET]
    log = []

    for k in range(len(us)):
        q_plan = xs[k][:nq]
        v_plan = xs[k][nq:]
        # Pinocchio joint order == MJCF joint order for the 27 actuated joints,
        # and both put them after the free joint, so the slice is direct.
        qj, vj = q_plan[7:], v_plan[6:]
        if ctrl_mode == "kinematic":
            # No dynamics at all: drive MuJoCo straight onto the planned
            # configuration.  This renders what crocoddyl BELIEVES it planned, so
            # a plan that is geometrically wrong can be told apart from a plan
            # that is fine but untrackable.  Contact forces here are meaningless.
            d.qpos[:] = cb.pin_to_mj(q_plan, qpos_full)
            d.qvel[:] = 0
            mujoco.mj_forward(m, d)
        elif ctrl_mode == "torque":
            d.ctrl[:] = qj + (us[k] + kd * vj) / kp
            for _ in range(nsub):
                mujoco.mj_step(m, d)
        else:
            d.ctrl[:] = qj
            for _ in range(nsub):
                mujoco.mj_step(m, d)

        rec = dict(k=k, t=k * dt_plan,
                   pelvis_z=float(d.qpos[2]),
                   tracking_err=float(np.linalg.norm(d.qpos[7:nq] - qj)),
                   com_x=float(d.subtree_com[1][0]))
        # contact forces actually realised, per bracing body and per foot
        f_brace = {s: 0.0 for s in subset}
        f_feet = 0.0
        buf = np.zeros(6)
        for c in range(d.ncon):
            con = d.contact[c]
            b1, b2 = m.geom_bodyid[con.geom[0]], m.geom_bodyid[con.geom[1]]
            mujoco.mj_contactForce(m, d, c, buf)
            fn = abs(float(buf[0]))
            for s, bid_ in brace_bodies.items():
                if {b1, b2} == {bid_, tbl} or (bid_ in (b1, b2) and tbl in (b1, b2)):
                    f_brace[s] += fn
            if b1 in feet or b2 in feet:
                f_feet += fn
        rec.update({f"F_{s}": f_brace[s] for s in subset})
        rec["F_feet"] = f_feet
        log.append(rec)

        if renderer is not None and k % frame_every == 0:
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = (0.75, 0.0, 0.90)
            camera.distance, camera.azimuth, camera.elevation = 3.0, 120, -14
            renderer.update_scene(d, camera=camera)
            frames.append(renderer.render())

    if renderer is not None:
        import imageio.v2 as imageio
        imageio.mimsave(video, frames, fps=fps, quality=8, macro_block_size=1)
        print(f"wrote {video}  ({len(frames)} frames)")

    return log, plan


def summarise(log, plan, ctrl_mode):
    last = log[-1]
    fell = min(r["pelvis_z"] for r in log) < 0.55
    print(f"\n--- MuJoCo replay ({ctrl_mode} inversion) ---")
    print(f"  steps                {len(log)}   ({log[-1]['t']:.2f} s)")
    print(f"  pelvis z  final      {last['pelvis_z']:.3f} m   "
          f"min {min(r['pelvis_z'] for r in log):.3f} m"
          f"   {'TOPPLED' if fell else 'upright'}")
    print(f"  joint tracking error final {last['tracking_err']:.4f} rad, "
          f"max {max(r['tracking_err'] for r in log):.4f} rad")
    for s in plan["subset"]:
        col = [r[f"F_{s}"] for r in log]
        print(f"  brace {s:9s} force  final {col[-1]:7.1f} N   peak {max(col):7.1f} N")
    print(f"  feet normal force    final {last['F_feet']:7.1f} N   "
          f"(robot weight 673.6 N)")
    return dict(fell=bool(fell), final_pelvis_z=last["pelvis_z"],
                max_tracking_err=max(r["tracking_err"] for r in log),
                final_tracking_err=last["tracking_err"],
                brace_force_final={s: log[-1][f"F_{s}"] for s in plan["subset"]},
                brace_force_peak={s: max(r[f"F_{s}"] for r in log)
                                  for s in plan["subset"]},
                feet_force_final=last["F_feet"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="elbow+forearm")
    ap.add_argument("--ctrl", default="torque", choices=["torque", "position", "kinematic"])
    ap.add_argument("--dir", default="runs/2026-08-05_session12/croco")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    tag = args.mode.replace("+", "_")
    vid = (os.path.join(args.dir, f"replay_{tag}_{args.ctrl}.mp4")
           if args.video else None)
    log, plan = replay(tag, ctrl_mode=args.ctrl, run_dir=args.dir, video=vid,
                       fps=args.fps, speed=args.speed)
    res = summarise(log, plan, args.ctrl)

    out = os.path.join(args.dir, f"replay_{tag}_{args.ctrl}.json")
    with open(out, "w") as fh:
        json.dump({"mode": args.mode, "ctrl": args.ctrl, "summary": res,
                   "log": log}, fh, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
