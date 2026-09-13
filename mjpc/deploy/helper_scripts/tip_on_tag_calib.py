#!/usr/bin/env python3
"""tip_on_tag_calib.py -- TIP-ON-TAG hand-eye calibration for the gripper (wrist) camera.

WHAT IT MEASURES. With the gripper's CENTRELINE TIP (the point on the gripper axis at the far
edge of the jaw plates, local (0.2254, -0.0118, 0.0) in right_magpie_gripper) physically placed on
the CENTRE of a tag, the tag position the detector reports in the CAMERA frame *is* the tip's
position in the camera frame:   c_tip = t_cam.   That is a rigid constant of the gripper, so
every pose must give the same vector; the spread across poses is the calibration's own error.
No base pose, no table survey, no estimator anywhere in this.

WHAT IT OUTPUTS.
  1. c_tip (tip in camera frame) = mean over poses, with the per-pose spread.
  2. grip_cam_pos consistent with c_tip and the CURRENT rotation:  tip_wrist = cam_pos + R c_tip
     =>  cam_pos = tip_wrist - R c_tip   (tip_wrist from the model).
  3. rotation CHECK: the tag lies FLAT on the table, so its normal (from the detector's rvec)
     must come out vertical after R and the wrist FK (IMU roll/pitch + encoders). Reports the
     tilt residual per pose (diagnostic only: a single normal cannot pin all 3 rotation dof, so the
     rotation stays the 08-28 value unless the tilt says it is wrong).
  4. the exact XML lines to paste (source XML + `ninja copy_model_resources`).

HOW TO DO IT (robot on, safety layer + tag_bridge --gripper --gripper-publish running, arm free):
  a. Put ONE tag flat on the table (tag 30 or any of 31-34; pass --tag-id). Tape it down.
  b. Lower the gripper so BOTH jaw plates' far edges rest on the table with the tag centred
     between them, jaw axis across the tag. The centreline tip is then at table height, at the
     jaw far edge, centred = the tag centre. Check the camera sees the tag (the script prints Hz).
  c. Press ENTER; the script averages 2 s of detections and stores the pose.
  d. Lift, change the WRIST ORIENTATION (roll the wrist +-30 deg, pitch it, come in from a
     different arm configuration), put the tip back on the tag centre, ENTER. Do 4-6 poses.
  e. Type q. Read the summary. Paste the two numerics, rebuild resources, restart the node.

Run (same env as hand_eye_check.py):
  .venv/bin/python tip_on_tag_calib.py [--tag-id 30] [--iface enx...] [--cam-rpy-deg -13.1 88.5 185.7]
  .venv/bin/python tip_on_tag_calib.py --selftest      # DDS-free math check
"""
import argparse, os, sys, time, threading
import numpy as np

SCENE = os.path.expanduser(
    "~/Desktop/h12/mujoco_mpc/mujoco_mpc/build/mjpc/tasks/humanoid_bench/lean/Lean_H12_Magpie.xml")
TIP_LOCAL = np.array([0.2254, -0.0118, 0.0])   # CENTRELINE tip, right_magpie_gripper frame

def rpy_to_R(rpy_deg):
    import mujoco
    q = np.zeros(4); mujoco.mju_euler2Quat(q, np.radians(np.asarray(rpy_deg, float)), "xyz")
    R = np.zeros(9); mujoco.mju_quat2Mat(R, q); return R.reshape(3, 3)

def R_to_rpy(R):
    import mujoco
    q = np.zeros(4); mujoco.mju_mat2Quat(q, R.reshape(9))
    # xyz euler from quaternion (mujoco has no quat2euler; do it via matrix)
    sy = -R[2, 0]; cy = np.sqrt(max(0.0, 1 - sy * sy))
    if cy > 1e-6:
        r = np.arctan2(R[2, 1], R[2, 2]); p = np.arcsin(np.clip(sy, -1, 1)); y = np.arctan2(R[1, 0], R[0, 0])
    else:
        r = np.arctan2(-R[1, 2], R[1, 1]); p = np.arcsin(np.clip(sy, -1, 1)); y = 0.0
    return np.degrees([r, p, y])

def tip_in_wrist(m):
    import mujoco
    d = mujoco.MjData(m); mujoco.mj_kinematics(m, d)
    wy = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    gt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_magpie_gripper")
    Rw = d.xmat[wy].reshape(3, 3); Rg = d.xmat[gt].reshape(3, 3)
    return Rw.T @ (d.xpos[gt] + Rg @ TIP_LOCAL - d.xpos[wy])

def kabsch(A, B):
    """R minimising |R a_i - b_i| for unit vectors a_i (rows of A) -> b_i (rows of B)."""
    H = A.T @ B; U, S, Vt = np.linalg.svd(H); D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T

def summarize(poses, rpy_deg, m, verbose=True):
    """poses: list of dict(t_cam, rvec, wR). Returns (c_tip, cam_pos, rpy_fit or None)."""
    import cv2
    R = rpy_to_R(rpy_deg); tw = tip_in_wrist(m)
    C = np.array([p["t_cam"] for p in poses]); c_tip = C.mean(0); spread = C - c_tip
    cam_pos = tw - R @ c_tip
    if verbose:
        print("\n=== TIP-ON-TAG RESULT (%d poses) ===" % len(poses))
        print("tip in WRIST frame (model)       : %s m" % np.round(tw, 4))
        for i, p in enumerate(poses):
            print("  pose %d  t_cam = (%+.4f %+.4f %+.4f)  dev from mean %.1f mm  n=%d" % (
                i + 1, *p["t_cam"], 1000 * np.linalg.norm(spread[i]), p["n"]))
        print("c_tip (tip in CAMERA frame)      : (%+.4f %+.4f %+.4f) m   spread %.1f mm rms, %.1f mm max" % (
            *c_tip, 1000 * np.sqrt((spread ** 2).sum(1).mean()), 1000 * np.linalg.norm(spread, axis=1).max()))
        print("grip_cam_pos with rpy %s       : (%+.4f %+.4f %+.4f)" % (np.round(rpy_deg, 1), *cam_pos))
    # rotation check: tag normal must be world-up
    rpy_fit = None; have = [p for p in poses if p.get("wR") is not None and p.get("rvec") is not None]
    if have:
        n_cam = []; n_wr = []
        for p in have:
            Rct, _ = cv2.Rodrigues(np.asarray(p["rvec"], float)); nc = Rct[:, 2]        # tag z in camera frame
            n_cam.append(nc); n_wr.append(p["wR"].T @ np.array([0, 0, 1.0]))            # world up in wrist frame
        n_cam = np.array(n_cam); n_wr = np.array(n_wr)
        tilt = [np.degrees(np.arccos(np.clip(abs((R @ a) @ b), -1, 1))) for a, b in zip(n_cam, n_wr)]
        if verbose:
            print("rotation check (tag normal vs world up through current rpy): tilt per pose %s deg" %
                  np.round(tilt, 1))
        if verbose:
            print("  (tilt > ~3 deg on every pose = the 08-28 rotation is off; a tag normal cannot fix all 3 dof,")
            print("   so the rotation is NOT refit here -- report it and we redo the 5-pose consistency solve.)")
    if verbose:
        print("\nPASTE (Lean_H12_Magpie.xml, then `ninja -C build copy_model_resources`, restart the node):")
        print('  <numeric name="grip_cam_pos" data="%.4f %.4f %.4f"/>' % tuple(cam_pos))
        print('  <numeric name="grip_cam_rpy_deg" data="%.1f %.1f %.1f"/>   (unchanged unless you take the refit)' % tuple(rpy_deg))
    return c_tip, cam_pos, rpy_fit

def analyse_record(z, a, m):
    """Find still tip-on-tag windows in a --record file and run the summary on them."""
    A = z["rows"]; rpy = list(z["rpy_in"]) if "rpy_in" in z.files else a.cam_rpy_deg
    if len(A) < 10:
        print("[calib] no detections in the record."); return
    t = A[:, 0]; tc = A[:, 1:4]; rv = A[:, 4:7]; wR = A[:, 7:16].reshape(-1, 3, 3); q = A[:, 19:26]
    # still = wrist joints not moving AND t_cam steady, over a sliding window
    win = a.still_seconds; poses = []; i = 0; n = len(t)
    while i < n:
        j = np.searchsorted(t, t[i] + win)
        if j - i >= 5 and j <= n:
            seg_tc = tc[i:j]; seg_q = q[i:j]
            jitter = np.sqrt(((seg_tc - seg_tc.mean(0)) ** 2).sum(1).mean()) * 1000
            qmove = np.degrees(np.abs(seg_q - seg_q.mean(0)).max())
            if jitter < a.still_mm and qmove < 0.5:
                # extend the window while it stays still
                k = j
                while k < n and t[k] - t[i] < 6.0:
                    s2 = tc[i:k + 1]; q2 = q[i:k + 1]
                    if np.sqrt(((s2 - s2.mean(0)) ** 2).sum(1).mean()) * 1000 < a.still_mm and np.degrees(np.abs(q2 - q2.mean(0)).max()) < 0.5: k += 1
                    else: break
                poses.append(dict(t_cam=tc[i:k].mean(0), rvec=rv[i:k].mean(0), wR=wR[i:k].mean(0), n=k - i, t0=t[i], t1=t[k - 1]))
                i = k; continue
        i += 1
    if not poses:
        print("[calib] no still windows found (need >= %.1f s with t_cam jitter < %.1f mm and wrist joints still). Loosen --still-mm / --still-seconds." % (win, a.still_mm)); return
    # merge windows that are the SAME pose (t_cam within 4 mm and wrist orientation within 2 deg)
    merged = []
    for p in poses:
        for q_ in merged:
            dR = p["wR"] @ q_["wR"].T; ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
            if np.linalg.norm(p["t_cam"] - q_["t_cam"]) < 0.004 and ang < 2.0:
                w = q_["n"] + p["n"]; q_["t_cam"] = (q_["t_cam"] * q_["n"] + p["t_cam"] * p["n"]) / w; q_["n"] = w; q_["t1"] = p["t1"]; break
        else:
            merged.append(dict(p))
    print("[calib] %d still window(s) -> %d distinct pose(s):" % (len(poses), len(merged)))
    for k, p in enumerate(merged):
        print("   pose %d: t=%.0f..%.0f s (%d det)  t_cam (%+.4f %+.4f %+.4f)" % (k + 1, p["t0"] - t[0], p["t1"] - t[0], p["n"], *p["t_cam"]))
    summarize(merged, rpy, m)

def _selftest():
    import mujoco
    m = mujoco.MjModel.from_xml_path(SCENE)
    rpy = [-13.1, 88.5, 185.7]; R = rpy_to_R(rpy); tw = tip_in_wrist(m)
    cam_pos_true = np.array([0.12, 0.02, -0.03]); c_tip_true = R.T @ (tw - cam_pos_true)
    rng = np.random.default_rng(0); poses = []
    for k in range(5):
        # random wrist orientation; tag flat => its normal in camera frame = R^T (wR^T up)
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax); ang = rng.uniform(0.2, 0.8)
        q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, ax, ang); wR = np.zeros(9); mujoco.mju_quat2Mat(wR, q); wR = wR.reshape(3, 3)
        n_w = wR.T @ np.array([0, 0, 1.0]); n_c = R.T @ n_w
        # build an rvec whose z column is n_c
        z = n_c; x = np.cross(z, [0, 1, 0]); x /= np.linalg.norm(x); y = np.cross(z, x); Rct = np.stack([x, y, z], 1)
        import cv2; rvec, _ = cv2.Rodrigues(Rct)
        poses.append(dict(t_cam=c_tip_true + rng.normal(scale=0.001, size=3), rvec=rvec.ravel(), wR=wR, n=40))
    c_tip, cam_pos, rpy_fit = summarize(poses, rpy, m, verbose=False)
    e = np.linalg.norm(cam_pos - cam_pos_true)
    print("[selftest] cam_pos recovered to %.1f mm (noise 1 mm/axis) -> %s" % (1000 * e, "ok" if e < 0.004 else "FAIL"))
    return 0 if e < 0.004 else 1

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag-id", type=int, default=30)
    ap.add_argument("--cam-rpy-deg", type=float, nargs=3, default=[-13.1, 88.5, 185.7],
                    help="CURRENT grip_cam_rpy_deg (used to derive grip_cam_pos and check the rotation)")
    ap.add_argument("--object-topic", default="rt/object_tag")
    ap.add_argument("--avg-seconds", type=float, default=2.0)
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--iface", default=None)
    ap.add_argument("--out", default=os.path.expanduser("~/Desktop/h12/logs/tip_on_tag_calib_%s.npz" % time.strftime("%m%d_%H%M%S")))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--record", type=float, default=0.0,
                    help="HANDS-FREE mode: record every detection + wrist FK for this many seconds "
                         "(no ENTER needed), save to --out, then auto-analyse: still windows where the "
                         "tag is seen and the wrist is not moving are taken as tip-on-tag poses.")
    ap.add_argument("--analyze", default=None, help="offline: analyse a --record npz (no DDS)")
    ap.add_argument("--still-seconds", type=float, default=1.5, help="min still window length")
    ap.add_argument("--still-mm", type=float, default=3.0, help="t_cam jitter allowed inside a still window (mm rms)")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftest())
    if a.analyze:
        import mujoco
        m = mujoco.MjModel.from_xml_path(SCENE)
        analyse_record(np.load(a.analyze, allow_pickle=True), a, m); return
    import mujoco
    m = mujoco.MjModel.from_xml_path(SCENE); d = mujoco.MjData(m)
    wy = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from base_estimator_node import _pick_iface
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    iface, why = _pick_iface(a.iface)
    ChannelFactoryInitialize(a.domain, iface) if iface else ChannelFactoryInitialize(a.domain)
    buf = {"ls": None, "obj": [], "n_obj": 0}
    lock = threading.Lock()
    def on_obj(x):
        if int(x.mode) != a.tag_id: return
        with lock:
            buf["obj"].append((time.time(), np.array(x.position[:3], float), np.array(x.velocity[:3], float))); buf["n_obj"] += 1
            if len(buf["obj"]) > 400: buf["obj"] = buf["obj"][-400:]
    ChannelSubscriber("rt/lowstate", LowState_).Init(lambda x: buf.__setitem__("ls", x), 10)
    ChannelSubscriber(a.object_topic, SportModeState_).Init(on_obj, 10)
    print(f"[calib] tag {a.tag_id} on {a.object_topic}, rt/lowstate (iface {iface or 'auto'}: {why}); rpy {a.cam_rpy_deg}")
    print("[calib] tip in wrist frame (model): %s" % np.round(tip_in_wrist(m), 4))
    print("[calib] put the CENTRELINE TIP on the tag centre, keep still, press ENTER to capture; 'q' + ENTER to finish.")
    # cyclonedds installs its own SIGINT handler after ChannelFactoryInitialize -> KeyboardInterrupt never
    # fires. Install ours AFTER DDS init so Ctrl-C / kill -INT / kill -TERM end the record and SAVE.
    import signal as _sig
    stop = {"f": False}
    for _s in (_sig.SIGINT, _sig.SIGTERM):
        _sig.signal(_s, lambda *_: stop.__setitem__("f", True))
    if a.record > 0:
        print(f"[calib] RECORD mode {a.record:.0f} s -> {a.out}. Put the tip on the tag centre and hold still >= {a.still_seconds} s per pose; "
              f"move between poses. Ctrl-C ends early.")
        T0 = time.time(); rows = []; seen = 0; last_print = 0.0
        try:
            while time.time() - T0 < a.record and not stop["f"]:
                time.sleep(0.05)
                with lock:
                    new = buf["obj"][seen - buf["n_obj"]:] if buf["n_obj"] > seen else []
                    seen = buf["n_obj"]; ls = buf["ls"]
                if ls is None: continue
                quat = np.array(list(ls.imu_state.quaternion)); q = np.array([ls.motor_state[i].q for i in range(27)])
                d.qpos[:] = 0; d.qpos[3:7] = quat; d.qpos[7:34] = q; mujoco.mj_kinematics(m, d)
                wR = d.xmat[wy].reshape(3, 3).copy(); wp = d.xpos[wy].copy()
                for (tt, tc, rv) in new: rows.append((tt, *tc, *rv, *wR.ravel(), *wp, *q[20:27]))
                if time.time() - last_print > 2.0:
                    last_print = time.time(); print("\r[calib] t=%4.0f s  detections %5d  (%2d/s)  " % (time.time() - T0, len(rows), len(new) / 2.0 if new else 0), end="", flush=True)
        except KeyboardInterrupt:
            print("\n[calib] stopped early.")
        A = np.array(rows) if rows else np.zeros((0, 25))
        np.savez(a.out, rows=A, cols="t tcx tcy tcz rvx rvy rvz wR00..wR22 wpx wpy wpz q_rsp q_rsr q_rsy q_re q_wr q_wp q_wy", rpy_in=np.array(a.cam_rpy_deg), tag_id=a.tag_id)
        print("\n[calib] saved %d detections -> %s" % (len(A), a.out))
        analyse_record(np.load(a.out, allow_pickle=True), a, m); return

    poses = []
    while True:
        # live status line
        t0 = time.time(); last = 0
        while True:
            time.sleep(0.25)
            with lock: n = buf["n_obj"]; recent = [o for o in buf["obj"] if o[0] > time.time() - 1.0]
            hz = len(recent); ls_ok = buf["ls"] is not None
            print("\r[calib] tag %d: %2d Hz   lowstate: %s   poses stored: %d   (ENTER=capture, q=finish) " % (
                a.tag_id, hz, "ok " if ls_ok else "NO ", len(poses)), end="", flush=True)
            import select
            if select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline().strip(); break
        if line.lower().startswith("q"):
            break
        # capture: average --avg-seconds of detections, wrist FK from IMU + encoders
        time.sleep(a.avg_seconds)
        with lock:
            win = [o for o in buf["obj"] if o[0] > time.time() - a.avg_seconds]; ls = buf["ls"]
        if len(win) < 5 or ls is None:
            print("\n[calib]   NOT stored: %d detections in window / lowstate %s -- is the tag in view and the arm still?" % (len(win), ls is not None)); continue
        T = np.array([o[1] for o in win]); RV = np.array([o[2] for o in win])
        quat = np.array(list(ls.imu_state.quaternion)); d.qpos[:] = 0; d.qpos[3:7] = quat
        for i in range(27): d.qpos[7 + i] = ls.motor_state[i].q
        mujoco.mj_kinematics(m, d); wR = d.xmat[wy].reshape(3, 3).copy()
        p = dict(t_cam=T.mean(0), rvec=RV.mean(0), wR=wR, n=len(T), t_std=T.std(0))
        poses.append(p)
        print("\n[calib]   stored pose %d: t_cam (%+.4f %+.4f %+.4f) m, jitter %.1f mm, n=%d" % (
            len(poses), *p["t_cam"], 1000 * np.linalg.norm(p["t_std"]), len(T)))
        if len(poses) >= 2:
            summarize(poses, a.cam_rpy_deg, m, verbose=False)
            C = np.array([q["t_cam"] for q in poses]); print("[calib]   c_tip spread so far: %.1f mm max" % (1000 * np.linalg.norm(C - C.mean(0), axis=1).max()))
    if not poses:
        print("\n[calib] nothing captured."); return
    c_tip, cam_pos, rpy_fit = summarize(poses, a.cam_rpy_deg, m)
    np.savez(a.out, t_cam=np.array([p["t_cam"] for p in poses]), rvec=np.array([p["rvec"] for p in poses]),
             wR=np.array([p["wR"] for p in poses]), n=np.array([p["n"] for p in poses]), rpy_in=np.array(a.cam_rpy_deg),
             c_tip=c_tip, cam_pos=cam_pos, rpy_fit=np.array(rpy_fit if rpy_fit is not None else [np.nan] * 3), tag_id=a.tag_id)
    print("[calib] saved", a.out)

if __name__ == "__main__":
    main()
