#!/usr/bin/env python3
"""sim_object_tag.py -- STRAT 27 synthetic gripper-cam object detector (twin only).

The real pipeline is: gripper RealSense sees the tag30 on the block ->
tag_bridge_node --gripper solves its pose -> rt/object_tag. On the TWIN there is
no camera, so this node SIMULATES that detection ANALYTICALLY (no rendering):
it subscribes to the twin's lowstate + sportmodestate, FK's the right wrist,
and computes where a perfectly-calibrated gripper camera WOULD see the block,
then publishes rt/object_tag in the same contract the node consumes with
--object_servo (SportState: position = tag translation in the CAMERA OPTICAL
frame, velocity = rvec placeholder, mode = tag id).

It is the exact INVERSE of the servo composition in lean.cc, using the SAME
hand-eye extrinsic (pass --cam-pos / --cam-rpy-deg identical to the model's
grip_cam_pos / grip_cam_rpy_deg numerics), so a correctly-wired loop recovers
the true object position. Point of the tool: exercise the servo consumer, the
grasp gate, and the relay end-to-end on the twin with a KNOWN object offset,
before any real camera or calibration exists.

  --object-offset dx dy dz : place the block this far from its nominal scene
        pose (default 0). The servo should then shift the reach target by the
        same amount -- the decisive loop test.
  --noise-mm / --rate-hz   : detection realism.
  --occlude-below M        : stop publishing when the grasp centre is closer
        than M to the block (jaws occlude the tag) -> exercises freeze-and-go.

Run (twin venv + unitree_sdk2py, same shell as the bench):
  python3 sim_object_tag.py --object-offset 0.03 -0.02 0.0
  python3 sim_object_tag.py --selftest      # DDS-free math check
"""
import argparse, math, os, sys, time
import numpy as np

H12 = os.path.expanduser("~/Desktop/h12")
SCENE = f"{H12}/h1_mujoco/unitree_robots/h1_2/scene_handless_magpie_table_block.xml"
# block nominal in world (scene_handless_magpie_table_block.xml, depth 0.55)
BLOCK_NOMINAL = np.array([1.000, -0.16, 1.010])
GRASP_LOCAL = np.array([0.19, -0.0038, 0.0])   # grasp centre in gripper frame
IMU_OFFSET = np.array([-0.04452, -0.01891, 0.27756])  # IMU site in pelvis frame
                                                      # (deploy_common.h kImuOffset)


def opt_to_wrist(rpy_deg):
    import mujoco
    q = np.zeros(4); R = np.zeros(9)
    mujoco.mju_euler2Quat(q, np.radians(rpy_deg), "xyz")
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def object_in_camera(obj_world, wrist_pos, wrist_R, cam_pos, R_ow):
    """Inverse of lean.cc's servo composition: world object -> camera optical."""
    cam_world = wrist_pos + wrist_R @ cam_pos
    in_wrist = wrist_R.T @ (obj_world - cam_world)      # object in wrist frame
    return R_ow.T @ in_wrist                            # -> optical frame


def _selftest():
    import mujoco
    ok = True
    R_ow = opt_to_wrist([-90.0, 0.0, -90.0])
    cam_pos = np.array([0.06, 0.0, 0.05])
    # random wrist pose + object; forward then the lean.cc inverse must recover it
    rng = [0.3, -0.1, 1.0]
    wrist_pos = np.array(rng)
    aa = np.array([0.2, -0.3, 0.5]); q = np.zeros(4); Rm = np.zeros(9)
    mujoco.mju_axisAngle2Quat(q, aa/np.linalg.norm(aa), np.linalg.norm(aa))
    mujoco.mju_quat2Mat(Rm, q); wrist_R = Rm.reshape(3, 3)
    for off in ([0, 0, 0], [0.03, -0.02, 0.0], [-0.05, 0.04, 0.01]):
        obj = BLOCK_NOMINAL + np.array(off)
        t_cam = object_in_camera(obj, wrist_pos, wrist_R, cam_pos, R_ow)
        # lean.cc forward
        p_world = wrist_R @ (R_ow @ t_cam + cam_pos) + wrist_pos
        good = np.allclose(p_world, obj, atol=1e-9); ok &= good
        print(f"  offset {np.array(off)} -> t_cam {t_cam.round(4)} recover_err "
              f"{np.linalg.norm(p_world-obj):.1e}  {'ok' if good else 'FAIL'}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    ap.add_argument("--cam-pos", type=float, nargs=3, default=[0.06, 0.0, 0.05],
                    help="MUST equal the model's grip_cam_pos numeric")
    ap.add_argument("--cam-rpy-deg", type=float, nargs=3, default=[-90.0, 0.0, -90.0],
                    help="MUST equal the model's grip_cam_rpy_deg numeric")
    ap.add_argument("--tag-id", type=int, default=30)
    ap.add_argument("--rate-hz", type=float, default=15.0)
    ap.add_argument("--noise-mm", type=float, default=2.0)
    ap.add_argument("--occlude-below", type=float, default=0.03,
                    help="stop publishing when grasp centre within this of the "
                         "block (jaws occlude tag) -> freeze-and-go")
    ap.add_argument("--max-range", type=float, default=0.8)
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--iface", default=None)
    ap.add_argument("--force", action="store_true",
                    help="publish regardless of visibility (pure wiring test): "
                         "the composition is exact at any arm pose, so the "
                         "node's servo delta should equal --object-offset")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftest())

    import mujoco
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    wy = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    gb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_magpie_gripper")
    R_ow = opt_to_wrist(a.cam_rpy_deg)
    cam_pos = np.array(a.cam_pos)
    obj_world = BLOCK_NOMINAL + np.array(a.object_offset)
    print(f"[sim-obj] block at world {obj_world} (nominal {BLOCK_NOMINAL} + "
          f"offset {a.object_offset}); cam_pos {cam_pos} rpy {a.cam_rpy_deg}")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from base_estimator_node import _pick_iface
    from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                             ChannelSubscriber, ChannelPublisher)
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
    iface, why = _pick_iface(a.iface)
    ChannelFactoryInitialize(a.domain, iface) if iface else ChannelFactoryInitialize(a.domain)

    latest = {"ls": None, "ss": None}
    ChannelSubscriber("rt/lowstate", LowState_).Init(
        lambda x: latest.__setitem__("ls", x), 10)
    ChannelSubscriber("rt/sportmodestate", SportModeState_).Init(
        lambda x: latest.__setitem__("ss", x), 10)
    pub = ChannelPublisher("rt/object_tag", SportModeState_); pub.Init()
    out = unitree_go_msg_dds__SportModeState_()

    period = 1.0 / a.rate_hz
    n_pub = n_occ = n_far = 0
    t_last = time.time()
    rs = np.random.RandomState(1)   # deterministic (Math.random ban n/a here, but keep repeatable)
    while True:
        time.sleep(period)
        ls, ss = latest["ls"], latest["ss"]
        if ls is None or ss is None:
            continue
        # rebuild qpos: base from sportstate, joints from lowstate motors 0..26
        quat = np.array(list(ls.imu_state.quaternion))   # wxyz
        Rb = np.zeros(9); mujoco.mju_quat2Mat(Rb, quat); Rb = Rb.reshape(3, 3)
        # sportmodestate.position is the IMU-SITE world pose; pelvis (free joint)
        # = site - R*IMU_OFFSET, exactly as the node reconstructs it.
        d.qpos[0:3] = np.array(list(ss.position)) - Rb @ IMU_OFFSET
        d.qpos[3:7] = quat
        for i in range(27):
            d.qpos[7 + i] = ls.motor_state[i].q
        mujoco.mj_forward(m, d)
        wpos = d.xpos[wy].copy(); wR = d.xmat[wy].reshape(3, 3).copy()
        grasp_world = d.xmat[gb].reshape(3, 3) @ GRASP_LOCAL + d.xpos[gb]
        # visibility gates
        dist_to_obj = np.linalg.norm(grasp_world - obj_world)
        t_cam = object_in_camera(obj_world, wpos, wR, cam_pos, R_ow)
        if (not a.force) and dist_to_obj < a.occlude_below:
            n_occ += 1
        else:
            if (not a.force) and (t_cam[2] <= 0.02 or np.linalg.norm(t_cam) > a.max_range):
                n_far += 1                     # behind camera / out of range
            else:
                t_cam = t_cam + rs.randn(3) * (a.noise_mm / 1000.0)
                out.position[0], out.position[1], out.position[2] = map(float, t_cam)
                out.velocity[0] = out.velocity[1] = out.velocity[2] = 0.0
                out.mode = a.tag_id
                pub.Write(out)
                n_pub += 1
        if time.time() - t_last > 2.0:
            t_last = time.time()
            print(f"[sim-obj] pub={n_pub} occluded={n_occ} out-of-view={n_far} "
                  f"grasp-to-block={dist_to_obj*100:.1f}cm", flush=True)


if __name__ == "__main__":
    main()
