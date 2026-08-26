#!/usr/bin/env python3
"""hand_eye_check.py -- STRAT 27 gripper-cam hand-eye VALIDATOR / calibrator.

The FORWARD twin of sim_object_tag.py. Where sim_object_tag turns a known world
object into a fake rt/object_tag, THIS tool does the real direction: it reads the
REAL detector's rt/object_tag (tag30 in the CAMERA OPTICAL frame from
tag_bridge_node --gripper) + the robot state (rt/lowstate + rt/sportmodestate),
FK's the right wrist, and applies the SAME hand-eye composition the node's
--object_servo does (lean.cc):

    p_world = wrist_pos + wrist_R @ (R_optical->wrist @ t_cam + cam_pos)

then prints where it BELIEVES the tag is in the world. Compare that to where you
physically placed tag30:
  * matches your surveyed spot        -> hand-eye is correct.
  * off by a consistent ROTATION      -> grip_cam_rpy_deg (esp. the roll) is off;
                                         the residual tells you which way.
  * off by a consistent TRANSLATION   -> grip_cam_pos is off by that vector.

NO brace, NO full control node, arm in ANY safe pose. This is the isolated servo
test + the ticket-03 hand-eye calibration rig in one.

Defaults match the current model numerics (grip_cam_pos 0.108 0 0,
grip_cam_rpy_deg -180 90 90). Pass --cam-pos / --cam-rpy-deg to try candidates
live without restarting the node. Pass --true-pos X Y Z (your surveyed tag world
position) to print the error directly and a suggested translation fix.

Run (same shell/env as the real detector; robot connected):
  python3 hand_eye_check.py                          # print believed tag world pose
  python3 hand_eye_check.py --true-pos 1.00 -0.16 1.01
  python3 hand_eye_check.py --cam-rpy-deg -180 90 0  # try a different roll live
  python3 hand_eye_check.py --selftest               # DDS-free round-trip vs sim_object_tag
"""
import argparse, os, sys, time
import numpy as np

H12 = os.path.expanduser("~/Desktop/h12")
SCENE = f"{H12}/h1_mujoco/unitree_robots/h1_2/scene_handless_magpie_table_block.xml"
IMU_OFFSET = np.array([-0.04452, -0.01891, 0.27756])   # deploy_common.h kImuOffset
SERVO_NOMINAL = np.array([0.55, 0.16, 0.025])          # model servo_nominal (B3)


def opt_to_wrist(rpy_deg):
    import mujoco
    q = np.zeros(4); R = np.zeros(9)
    mujoco.mju_euler2Quat(q, np.radians(rpy_deg), "xyz")
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def compose_world(t_cam, wrist_pos, wrist_R, cam_pos, R_ow):
    """lean.cc servo composition: camera-optical tag -> world."""
    return wrist_R @ (R_ow @ t_cam + cam_pos) + wrist_pos


def nominal_world(m):
    """B3 world point from the table geom + servo_nominal, exactly as lean.cc."""
    import mujoco
    tg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
    import numpy as _np
    # geom world pos requires a forward; use the model's geom_pos+body_pos via a data
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    tc = d.geom_xpos[tg]
    half_depth = m.geom_size[tg][0]      # x half-size (approach axis)
    face = tc[2] + m.geom_size[tg][2]
    return _np.array([tc[0] - half_depth + SERVO_NOMINAL[0],
                      tc[1] - SERVO_NOMINAL[1],
                      face + SERVO_NOMINAL[2]])


def _selftest():
    import mujoco
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sim_object_tag import object_in_camera, BLOCK_NOMINAL
    ok = True
    for rpy, cpos in ([-180.0, 90.0, 90.0], [0.108, 0.0, 0.0]), ([-90.0, 0.0, -90.0], [0.06, 0.0, 0.05]):
        R_ow = opt_to_wrist(rpy); cam_pos = np.array(cpos)
        wrist_pos = np.array([0.3, -0.1, 1.0])
        aa = np.array([0.2, -0.3, 0.5]); q = np.zeros(4); Rm = np.zeros(9)
        mujoco.mju_axisAngle2Quat(q, aa/np.linalg.norm(aa), np.linalg.norm(aa))
        mujoco.mju_quat2Mat(Rm, q); wR = Rm.reshape(3, 3)
        for off in ([0, 0, 0], [0.03, -0.02, 0.0], [-0.05, 0.04, 0.01]):
            obj = BLOCK_NOMINAL + np.array(off)
            t_cam = object_in_camera(obj, wrist_pos, wR, cam_pos, R_ow)   # inverse
            p = compose_world(t_cam, wrist_pos, wR, cam_pos, R_ow)        # forward
            good = np.allclose(p, obj, atol=1e-9); ok &= good
            print(f"  rpy{rpy} off{np.array(off)} recover_err {np.linalg.norm(p-obj):.1e} "
                  f"{'ok' if good else 'FAIL'}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam-pos", type=float, nargs=3, default=[0.108, 0.0, 0.0],
                    help="grip_cam_pos candidate (default = current model numeric)")
    ap.add_argument("--cam-rpy-deg", type=float, nargs=3, default=[-180.0, 90.0, 90.0],
                    help="grip_cam_rpy_deg candidate (default = current model numeric)")
    ap.add_argument("--true-pos", type=float, nargs=3, default=None,
                    help="surveyed tag world position; prints error + suggested fix")
    ap.add_argument("--object-topic", default="rt/object_tag")
    ap.add_argument("--tag-id", type=int, default=30)
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--iface", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftest())

    import mujoco
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    wy = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    R_ow = opt_to_wrist(a.cam_rpy_deg)
    cam_pos = np.array(a.cam_pos)
    nom = nominal_world(m)
    print(f"[hand-eye] cam_pos {cam_pos} rpy {a.cam_rpy_deg} | B3 nominal world {nom.round(3)}")
    if a.true_pos:
        print(f"[hand-eye] surveyed true tag world = {np.array(a.true_pos).round(3)}")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from base_estimator_node import _pick_iface
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    iface, why = _pick_iface(a.iface)
    ChannelFactoryInitialize(a.domain, iface) if iface else ChannelFactoryInitialize(a.domain)

    latest = {"ls": None, "ss": None, "obj": None}
    ChannelSubscriber("rt/lowstate", LowState_).Init(lambda x: latest.__setitem__("ls", x), 10)
    ChannelSubscriber("rt/sportmodestate", SportModeState_).Init(lambda x: latest.__setitem__("ss", x), 10)
    ChannelSubscriber(a.object_topic, SportModeState_).Init(lambda x: latest.__setitem__("obj", x), 10)
    print(f"[hand-eye] waiting for {a.object_topic} + rt/lowstate + rt/sportmodestate "
          f"(iface {iface or 'auto'}: {why}) ...")

    errs = []
    t_last = 0.0
    while True:
        time.sleep(0.1)
        ls, ss, obj = latest["ls"], latest["ss"], latest["obj"]
        if ls is None or ss is None or obj is None:
            continue
        if int(obj.mode) != a.tag_id:
            continue
        t_cam = np.array([obj.position[0], obj.position[1], obj.position[2]])
        quat = np.array(list(ls.imu_state.quaternion))
        Rb = np.zeros(9); mujoco.mju_quat2Mat(Rb, quat); Rb = Rb.reshape(3, 3)
        d.qpos[0:3] = np.array(list(ss.position)) - Rb @ IMU_OFFSET
        d.qpos[3:7] = quat
        for i in range(27):
            d.qpos[7 + i] = ls.motor_state[i].q
        mujoco.mj_forward(m, d)
        wpos = d.xpos[wy].copy(); wR = d.xmat[wy].reshape(3, 3).copy()
        p_world = compose_world(t_cam, wpos, wR, cam_pos, R_ow)
        delta = p_world - nom                          # what the servo would apply
        if time.time() - t_last > 1.0:
            t_last = time.time()
            line = (f"[hand-eye] believed tag world = ({p_world[0]:+.3f} {p_world[1]:+.3f} "
                    f"{p_world[2]:+.3f})  servo-delta vs B3 = ({delta[0]:+.3f} "
                    f"{delta[1]:+.3f} {delta[2]:+.3f})  t_cam=({t_cam[0]:+.3f} "
                    f"{t_cam[1]:+.3f} {t_cam[2]:+.3f})")
            if a.true_pos:
                e = p_world - np.array(a.true_pos)
                errs.append(e)
                em = np.mean(errs[-30:], axis=0)
                line += (f"\n           ERROR vs surveyed = ({e[0]:+.3f} {e[1]:+.3f} "
                         f"{e[2]:+.3f})  |{np.linalg.norm(e)*1000:.0f}mm|  "
                         f"mean30=({em[0]:+.3f} {em[1]:+.3f} {em[2]:+.3f}) "
                         f"-> if steady, subtract it from grip_cam_pos")
            print(line, flush=True)


if __name__ == "__main__":
    main()
