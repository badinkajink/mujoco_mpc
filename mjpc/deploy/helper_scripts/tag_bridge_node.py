#!/usr/bin/env python3
"""tag_bridge_node.py -- AprilTag bundle on the lean table -> rt/aux_odom (mode 2).

SCOPE (do not let it creep): this feeder exists for ONE reason -- when load
bearing happens, the planted-foot assumption the base estimator rests on can
give out, and the estimate with it. The static tag bundle on the table is an
external, drift-free position reference that keeps the estimator honest through
exactly that regime. It feeds the v4 AUX BUS as a POSITION-ONLY source
(SportModeState_.mode == 2: xy anchor only, the velocity state is never
touched -- 30 Hz differenced fiducial velocity would be noise). It has NO role
in the lean pipeline: brace targets, table position, stage logic all stay
config/model-driven and never see this node's output. If this bridge dies, v4
degrades to v3 within --aux-age-max. Nothing else notices.

CHAIN:
    head D435i color (cl_realsense real / mujoco_ros_bridge sim)
      -> 36h11 detection (cv2.aruco) of the SURVEYED bundle (table_tag_bundle.yaml)
      -> solvePnP over every visible corner  => T_cam<-table
      -> fixed camera-in-torso extrinsic (--cam-pos/--cam-euler ★VERIFY against
         the URDF/robot_tf before trusting -- wrong extrinsic = biased anchor)
      -> Rz(-waist, motor 12) torso->pelvis  => pelvis pose IN TABLE FRAME
      -> continuous yaw alignment vs the IMU quat (same trick as lio_bridge)
      -> publish pelvis xy in the IMU-world yaw frame, mode=2

TAG CONVENTION (must match how the tags were physically mounted): every tag
lies flat on the top surface with its printed PAGE-TOP toward the table's far
end (+x of the survey frame). => tag-image 'up' = +x_table, tag-image 'right'
= -y_table, tag normal = +z.

Run (needs ROS2 overlay + the twin venv on PYTHONPATH, like lio_bridge):
  python3 tag_bridge_node.py --bundle ~/Desktop/h12/table_tags/table_tag_bundle.yaml
Offline math check (no ROS2, no DDS, no camera):
  python3 tag_bridge_node.py --selftest
"""
import argparse
import math
import os
import time

import numpy as np
import cv2

TORSO_MOTOR = 12
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
# corner direction vectors in the TABLE frame (see TAG CONVENTION above):
U = np.array([1.0, 0.0, 0.0])          # tag-image 'up'    -> +x table
R = np.array([0.0, -1.0, 0.0])         # tag-image 'right' -> -y table


def _rz(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _euler_xyz(r, p, y):
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rzm = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rzm @ Ry @ Rx


def optical_in_torso(pitch_down):
    """R (torso <- OPTICAL camera frame) for a forward-facing camera pitched
    DOWN by pitch_down rad. OpenCV optical convention: z forward, x image-right,
    y image-down. Torso: x fwd, y left, z up."""
    ct, st = math.cos(pitch_down), math.sin(pitch_down)
    z_opt = np.array([ct, 0.0, -st])         # looking forward-down
    x_opt = np.array([0.0, -1.0, 0.0])       # image right = robot right
    y_opt = np.cross(z_opt, x_opt)           # image down
    return np.column_stack([x_opt, y_opt, z_opt])


def _quat_wxyz_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _yaw(Rm):
    return math.atan2(Rm[1, 0], Rm[0, 0])


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def load_bundle(path):
    """{id: (centre_xyz, size, rot)} from table_tag_bundle.yaml. rot = optional
    per-tag mounting rotation in quarter-turns CCW (about table +z) away from
    the page-top-toward-+x convention; absent = 0 = the old identity behaviour.
    ★ 2026-08-17: added so a tag taped down a quarter-turn off (real tags 0/1
    after the centreline move) can be DECLARED instead of re-stuck."""
    import re
    tags = {}
    for line in open(os.path.expanduser(path)):
        mrow = re.search(r"\{id:\s*(\d+),\s*size:\s*([\d.]+),\s*x:\s*([-\d.]+),"
                         r"\s*y:\s*([-\d.]+),\s*z:\s*([-\d.]+)", line)
        if mrow:
            tid = int(mrow.group(1))
            mrot = re.search(r"rot:\s*(-?\d+)", line)
            tags[tid] = (np.array([float(mrow.group(3)), float(mrow.group(4)),
                                   float(mrow.group(5))]), float(mrow.group(2)),
                         int(mrot.group(1)) % 4 if mrot else 0)
    return tags


def tag_corners_table(centre, size, rot=0):
    """Object points in the table frame, aruco corner order (TL, TR, BR, BL).
    rot = quarter-turns CCW the physical print is mounted away from the
    page-top-toward-+x convention (0 = stock)."""
    h = size / 2.0
    u, r = U, R
    for _ in range(rot % 4):
        u = np.array([-u[1], u[0], 0.0])     # rotz +90
        r = np.array([-r[1], r[0], 0.0])
    return np.array([centre + h * u - h * r,     # top-left
                     centre + h * u + h * r,     # top-right
                     centre - h * u + h * r,     # bottom-right
                     centre - h * u - h * r])    # bottom-left


class TagCore:
    """Detection-to-anchor math, ROS/DDS-free (selftestable).
    step(detections, K, dist, imu_quat, waist, dt) -> (xy, yaw_off, err_px) or None.
    detections: {id: 4x2 image corners in aruco order}."""

    def __init__(self, bundle, cam_pos, cam_R, yaw_tau,
                 min_tags=1, max_reproj_px=3.0):
        self.bundle = bundle
        self.T_tc_R = np.asarray(cam_R, float)       # torso <- OPTICAL camera
        self.T_tc_p = np.asarray(cam_pos, float)     # camera origin in torso frame
        self.yaw_tau = yaw_tau
        self.min_tags = min_tags
        self.max_reproj = max_reproj_px
        self.yaw_off = None
        self.n_solved = 0
        self.n_rejected = 0
        # Physical mounting rotation, auto-locked on the first gate-passing
        # frame: the code assumes printed page-top toward table +x, but a
        # mounting one quarter-turn off shifts every corner correspondence by
        # one slot -- the PnP still converges, just with reproj ~= tagpx/sqrt2
        # (measured 2026-07-30 on the real table: k=0 gave a rock-steady
        # 63.9px, k=1 gave 1.0px). Trying np.roll(corners, k) for k=0..3 and
        # locking the winner makes the bridge convention-proof (real vs sim).
        self.corner_rot = None

    def _solve(self, detections, K, dist, rot_k):
        obj, img = [], []
        for tid, corners in detections.items():
            if tid not in self.bundle:
                continue
            ent = self.bundle[tid]
            c, s = ent[0], ent[1]
            rk = ent[2] if len(ent) > 2 else 0   # per-tag mounting rot (2026-08-17)
            obj.append(tag_corners_table(c, s, rk))
            img.append(np.roll(np.asarray(corners, float).reshape(4, 2),
                               rot_k, axis=0))
        if len(obj) < self.min_tags:
            return None
        obj = np.concatenate(obj).astype(np.float64)
        img = np.concatenate(img).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)))
        return rvec, tvec, err

    def step(self, detections, K, dist, imu_quat, waist, dt):
        if self.corner_rot is None:
            tries = {k: self._solve(detections, K, dist, k) for k in range(4)}
            tries = {k: v for k, v in tries.items() if v is not None}
            if not tries:
                self.n_rejected += 1
                return None
            best = min(tries, key=lambda k: tries[k][2])
            if tries[best][2] > self.max_reproj:
                self.n_rejected += 1                 # nothing passes -> no lock yet
                return None
            self.corner_rot = best
            print(f"[tag] mounting rotation LOCKED: corner_rot={best} "
                  f"(reproj {tries[best][2]:.2f}px; "
                  f"others {[f'{k}:{v[2]:.0f}px' for k, v in sorted(tries.items()) if k != best]})",
                  flush=True)
            r = tries[best]
        else:
            r = self._solve(detections, K, dist, self.corner_rot)
        if r is None:
            self.n_rejected += 1
            return None
        rvec, tvec, err = r
        if err > self.max_reproj:
            self.n_rejected += 1                     # bad solve: blur/partial/mis-id
            return None
        R_ct, _ = cv2.Rodrigues(rvec)                # camera <- table
        t_ct = tvec.reshape(3)
        # invert: table <- camera
        R_tc = R_ct.T
        p_cam_table = -R_tc @ t_ct
        # camera -> torso -> pelvis
        R_table_torso = R_tc @ self.T_tc_R.T
        p_torso_table = p_cam_table - R_table_torso @ self.T_tc_p
        R_table_pelvis = R_table_torso @ _rz(-float(waist))
        # yaw alignment to the IMU world (lio_bridge trick: both frames agree on
        # gravity, so one LPF'd yaw offset relates them; tracks IMU yaw drift)
        R_wt_imu = _quat_wxyz_to_mat(np.asarray(imu_quat, float))
        R_wp_imu = R_wt_imu @ _rz(-float(waist))
        off = _wrap(_yaw(R_wp_imu) - _yaw(R_table_pelvis))
        if self.yaw_off is None:
            self.yaw_off = off
        else:
            alpha = dt / (self.yaw_tau + dt)
            self.yaw_off = _wrap(self.yaw_off + alpha * _wrap(off - self.yaw_off))
        xy = (_rz(self.yaw_off) @ p_torso_table)[:2]
        self.n_solved += 1
        # p_torso_table (un-rotated table-frame pose) rides along for the
        # 2026-08-13 --abs-world mode; legacy callers unpack 3 values.
        return xy, self.yaw_off, err, p_torso_table


class ObjectTagCore:
    """★ 2026-08-24 STRAT 27: single-tag object localizer for the GRIPPER cam.

    FIREWALL (the whole point of this being a separate class + topic): output
    goes to rt/object_tag ONLY -- it never touches rt/aux_odom, yaw_off, the
    believed base pose, or the table frame. The gripper cam rides on the moving
    arm; letting it anywhere near the estimator would inject arm motion into
    the base estimate. The consumer (h12_control_node grasp phase) composes
    T_cam<-tag with its OWN believed wrist FK + the fixed hand-eye extrinsic.

    Math: IPPE_SQUARE PnP on the 4 corners of one known-size tag. Quality
    gates: (1) reprojection error <= max_reproj_px; (2) flip-ambiguity -- IPPE
    returns both planar solutions; if the runner-up's reprojection error is
    within ambig_ratio of the winner's, the view is too shallow to trust the
    orientation and the frame is REJECTED (the 45 deg-approach lesson: shallow
    tag views have a pose flip ambiguity).
    step(corners, K, dist) -> (tvec, rvec, err_px) in the CAMERA OPTICAL frame
    or None."""

    def __init__(self, tag_id, tag_size, max_reproj_px=2.0, ambig_ratio=2.0):
        self.tag_id = int(tag_id)
        h = tag_size / 2.0
        # object points in the TAG frame, aruco order (TL,TR,BR,BL), tag plane
        # z=0, x right, y up (IPPE_SQUARE convention wants this exact order)
        self.obj = np.array([[-h,  h, 0.0], [ h,  h, 0.0],
                             [ h, -h, 0.0], [-h, -h, 0.0]])
        self.max_reproj = max_reproj_px
        self.ambig_ratio = ambig_ratio
        self.n_solved = 0
        self.n_rejected = 0

    def step(self, corners, K, dist):
        img = np.asarray(corners, float).reshape(4, 2)
        try:
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                self.obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error:
            self.n_rejected += 1
            return None
        if n < 1:
            self.n_rejected += 1
            return None
        errs = [float(np.ravel(e)[0]) for e in errs]
        order = np.argsort(errs)
        best = int(order[0])
        if errs[best] > self.max_reproj:
            self.n_rejected += 1
            return None
        # flip-ambiguity gate: a clear winner must beat the alternative by
        # ambig_ratio in reprojection error, else the pose is untrustworthy
        if n > 1:
            second = int(order[1])
            if errs[second] < self.ambig_ratio * max(errs[best], 1e-6):
                self.n_rejected += 1
                return None
        self.n_solved += 1
        return (tvecs[best].reshape(3), rvecs[best].reshape(3), errs[best])


def _selftest():
    ok = True
    # synthetic bundle = the locked v3 symmetric layout
    bundle = {0: (np.array([0.10, 0.15, 0.0]), 0.10),
              1: (np.array([0.10, 0.45, 0.0]), 0.10),
              10: (np.array([0.28, 0.20, 0.0]), 0.05),
              11: (np.array([0.28, 0.40, 0.0]), 0.05),
              2: (np.array([0.62, 0.15, 0.0]), 0.10),
              3: (np.array([0.62, 0.45, 0.0]), 0.10),
              20: (np.array([1.05, 0.30, 0.0]), 0.12)}
    K = np.array([[900.0, 0, 960], [0, 900.0, 540], [0, 0, 1]])
    dist = np.zeros(5)

    # ground truth: torso yawed 0.25 in a table frame; camera on the torso,
    # OPTICAL frame pitched down 55 deg, 0.62m up / 0.05 fwd of the torso
    # origin; torso 0.45m behind the table origin, centred on the width;
    # waist 0.3; IMU world yawed +0.7 from the table frame.
    cam_pos = np.array([0.05, 0.0, 0.62])
    cam_R = optical_in_torso(math.radians(55.0))
    waist, yaw_imu_world = 0.3, 0.7
    R_table_torso_true = _rz(0.25)
    p_torso_true = np.array([-0.45, 0.2975, 0.95])
    R_table_cam = R_table_torso_true @ cam_R
    p_cam = p_torso_true + R_table_torso_true @ cam_pos
    # camera<-table for projection
    R_ct = R_table_cam.T
    t_ct = -R_ct @ p_cam
    rvec, _ = cv2.Rodrigues(R_ct)

    detections = {}
    for tid, (c, s) in bundle.items():
        proj, _ = cv2.projectPoints(tag_corners_table(c, s), rvec, t_ct, K, dist)
        detections[tid] = proj.reshape(4, 2)

    # IMU torso quat consistent with the ground truth: R_world_torso =
    # Rz(yaw_imu_world) @ R_table_torso (worlds differ by pure yaw)
    yaw_t = _yaw(_rz(yaw_imu_world) @ R_table_torso_true)
    imu_quat = np.array([math.cos((yaw_t + waist) * 0.5), 0, 0,
                         math.sin((yaw_t + waist) * 0.5)])
    # ^ torso quat = pelvis yaw + waist; pelvis yaw = yaw_t - ... keep simple:
    # the core only uses yaw(R_wt_imu @ Rz(-waist)); feed R_wt = Rz(yaw_t):
    imu_quat = np.array([math.cos(yaw_t * 0.5), 0, 0, math.sin(yaw_t * 0.5)])

    core = TagCore(bundle, cam_pos, cam_R, yaw_tau=0.0)
    out = core.step(detections, K, dist, imu_quat, waist, dt=0.033)
    assert out is not None, "solve failed"
    xy, yoff, err = out[0], out[1], out[2]
    want_xy = (_rz(yaw_imu_world) @ p_torso_true)[:2]
    e = float(np.linalg.norm(xy - want_xy))
    good = e < 1e-6 and err < 0.1
    ok &= good
    print(f"[selftest] tag full bundle : xy_err={e:.2e} m reproj={err:.3f}px "
          f"yaw_off={yoff:+.3f} (expect {yaw_imu_world - 0.0:+.3f}-ish yaw frame) "
          f"({'ok' if good else 'FAIL'})")

    # subset: only the two 5 cm tags visible (deep-lean case)
    sub = {k: detections[k] for k in (10, 11)}
    core2 = TagCore(bundle, cam_pos, cam_R, yaw_tau=0.0)
    out2 = core2.step(sub, K, dist, imu_quat, waist, dt=0.033)
    e2 = float(np.linalg.norm(out2[0] - want_xy))
    good = e2 < 1e-5
    ok &= good
    print(f"[selftest] tag 2-tag subset: xy_err={e2:.2e} m ({'ok' if good else 'FAIL'})")

    # corrupted detection must be REJECTED by the reprojection gate, not fused
    bad = dict(detections)
    bad[0] = bad[0] + np.array([25.0, -18.0])       # shove one tag 30px off
    core3 = TagCore(bundle, cam_pos, cam_R, yaw_tau=0.0)
    out3 = core3.step(bad, K, dist, imu_quat, waist, dt=0.033)
    good = out3 is None and core3.n_rejected == 1
    ok &= good
    print(f"[selftest] tag reproj gate : corrupted bundle rejected={out3 is None} "
          f"({'ok' if good else 'FAIL'})")

    # wrong extrinsic must show up as a biased anchor (this is WHY --cam-pos
    # needs verification): document the sensitivity, 1cm cam error -> ~1cm bias
    core4 = TagCore(bundle, cam_pos + np.array([0.01, 0, 0]), cam_R, yaw_tau=0.0)
    out4 = core4.step(detections, K, dist, imu_quat, waist, dt=0.033)
    bias = float(np.linalg.norm(out4[0] - want_xy))
    good = 0.005 < bias < 0.02
    ok &= good
    print(f"[selftest] tag extrinsic   : +1cm cam-pos error -> {bias*100:.1f}cm anchor bias "
          f"(calibrate before trusting) ({'ok' if good else 'FAIL'})")

    # ---- ObjectTagCore (strat 27 gripper cam) ------------------------------
    # ground truth: 45mm tag 0.25m ahead of the camera, facing it, yawed 10 deg
    oc = ObjectTagCore(30, 0.045)
    R_ct_o = _euler_xyz(math.pi, 0.0, math.radians(10.0))  # tag z toward cam
    t_true = np.array([0.02, -0.01, 0.25])
    rvec_o, _ = cv2.Rodrigues(R_ct_o)
    proj_o, _ = cv2.projectPoints(oc.obj, rvec_o, t_true, K, dist)
    r_o = oc.step(proj_o.reshape(4, 2), K, dist)
    # sub-mm bar (near-frontal is IPPE's numerically worst regime; 0.36mm
    # measured on this synthetic view -- far under the jaw capture margin)
    good = r_o is not None and float(np.linalg.norm(r_o[0] - t_true)) < 1e-3
    ok &= good
    print(f"[selftest] obj tag pose    : "
          f"{'t_err=%.2e m reproj=%.3fpx ok' % (np.linalg.norm(r_o[0]-t_true), r_o[2]) if r_o else 'SOLVE FAILED'} "
          f"({'ok' if good else 'FAIL'})")
    # near-frontal views are ambiguity-safe; a corrupted detection must reject
    r_bad = ObjectTagCore(30, 0.045).step(
        proj_o.reshape(4, 2) + np.array([12.0, -9.0]) * np.array([[1], [0], [1], [0]]),
        K, dist)
    good = r_bad is None
    ok &= good
    print(f"[selftest] obj tag gate    : corrupted corners rejected={r_bad is None} "
          f"({'ok' if good else 'FAIL'})")

    print(f"[selftest] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default="~/Desktop/h12/table_tags/table_tag_bundle.yaml")
    ap.add_argument("--image-topic", default="/realsense/head/color/image_raw/compressed")
    ap.add_argument("--info-topic", default="/realsense/head/color/camera_info")
    ap.add_argument("--raw", action="store_true", help="image topic is raw, not compressed")
    ap.add_argument("--aux-topic", default="rt/aux_odom")
    ap.add_argument("--lowstate-topic", default="rt/lowstate")
    ap.add_argument("--cam-pos", type=float, nargs=3,
                    default=[0.11109, 0.0175, 0.68789],
                    help="camera origin in the TORSO frame (m). Default = "
                         "camera_link from h1_2_handless_ros.urdf (via the "
                         "CL_Assets h1_2_magpie model, camera_joint). Residual "
                         "risk: the D435i COLOR sensor sits a few mm off the "
                         "camera body origin -- the move-table-10cm field check "
                         "still applies (1cm error = 1cm anchor bias).")
    ap.add_argument("--cam-pitch-deg", type=float, default=50.760,
                    help="head camera OPTICAL-frame pitch-down (deg) in the torso "
                         "frame. Default = 0.8859291 rad from the same URDF "
                         "camera_joint.")
    ap.add_argument("--abs-world", action="store_true",
                    help="publish ABSOLUTE model-world xy (table corner at "
                         "0.45,0.2975) instead of the imu-yaw-rotated relative "
                         "pose; pair with est --aux-abs")
    ap.add_argument("--yaw-tau", type=float, default=30.0)
    ap.add_argument("--min-tags", type=int, default=1)
    ap.add_argument("--max-reproj-px", type=float, default=3.0)
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--iface", default=None)
    ap.add_argument("--no-auto-iface", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    # ★ 2026-08-24 STRAT 27 gripper-cam object channel (opt-in, log-only until
    # --gripper-publish). See ObjectTagCore docstring for the FIREWALL contract.
    ap.add_argument("--gripper", action="store_true",
                    help="enable the gripper-cam tag30 object channel")
    ap.add_argument("--gripper-image-topic",
                    default="/realsense/right_hand/color/image_raw/compressed")
    ap.add_argument("--gripper-info-topic",
                    default="/realsense/right_hand/color/camera_info")
    ap.add_argument("--gripper-raw", action="store_true")
    ap.add_argument("--object-topic", default="rt/object_tag",
                    help="DDS topic for the object tag pose (CAMERA optical "
                         "frame; consumer composes with believed wrist FK)")
    ap.add_argument("--object-tag-id", type=int, default=30)
    ap.add_argument("--object-tag-size", type=float, default=0.045)
    ap.add_argument("--gripper-publish", action="store_true",
                    help="actually publish rt/object_tag (default: LOG ONLY)")
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(_selftest())

    bundle = load_bundle(a.bundle)
    if not bundle:
        raise SystemExit(f"[tag] no tags parsed from {a.bundle} -- fill the layout rows first")
    print(f"[tag] bundle: {len(bundle)} tags {sorted(bundle)} from {a.bundle}")

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from base_estimator_node import _pick_iface
    from unitree_sdk2py.core.channel import (ChannelFactoryInitialize, ChannelSubscriber,
                                             ChannelPublisher)
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_

    if a.no_auto_iface:
        iface = None
    else:
        iface, why = _pick_iface(a.iface)
        print(f"[tag] DDS interface = {iface or 'autodetermine'} ({why})")
    ChannelFactoryInitialize(a.domain, iface) if iface else ChannelFactoryInitialize(a.domain)

    latest = {"ls": None, "K": None, "dist": None}
    ls_sub = ChannelSubscriber(a.lowstate_topic, LowState_)
    ls_sub.Init(lambda m: latest.__setitem__("ls", m), 10)
    pub = ChannelPublisher(a.aux_topic, SportModeState_)
    pub.Init()
    out_msg = unitree_go_msg_dds__SportModeState_()

    core = TagCore(bundle, a.cam_pos, optical_in_torso(math.radians(a.cam_pitch_deg)),
                   a.yaw_tau, a.min_tags, a.max_reproj_px)
    det = cv2.aruco.ArucoDetector(DICT)
    stats = {"n": 0, "t0": time.time(), "last": 0.0}

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, CompressedImage, Image

    # ---- gripper-cam object channel (strat 27, opt-in) ---------------------
    obj_core = None
    obj_pub = None
    obj_msg = None
    obj_state = {"K": None, "dist": None, "n": 0, "t0": time.time(), "last": 0.0}
    if a.gripper:
        obj_core = ObjectTagCore(a.object_tag_id, a.object_tag_size)
        if a.gripper_publish:
            obj_pub = ChannelPublisher(a.object_topic, SportModeState_)
            obj_pub.Init()
            obj_msg = unitree_go_msg_dds__SportModeState_()
        print(f"[obj] gripper-cam channel ON: tag{a.object_tag_id} "
              f"({a.object_tag_size*1000:.0f}mm) {a.gripper_image_topic} -> "
              f"{'DDS ' + a.object_topic if a.gripper_publish else 'LOG ONLY'} "
              f"(CAMERA-frame pose; FIREWALLED from aux/estimator)")

    class Bridge(Node):
        def __init__(self):
            super().__init__("tag_bridge")
            qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(CameraInfo, a.info_topic, self.on_info, qos)
            if a.raw:
                self.create_subscription(Image, a.image_topic, self.on_raw, qos)
            else:
                self.create_subscription(CompressedImage, a.image_topic, self.on_jpg, qos)
            self.get_logger().info(f"tag_bridge: {a.image_topic} -> DDS '{a.aux_topic}' "
                                   f"(mode=2 POSITION-ONLY, estimator aux; no pipeline role)")
            if a.gripper:
                self.create_subscription(CameraInfo, a.gripper_info_topic,
                                         self.on_obj_info, qos)
                if a.gripper_raw:
                    self.create_subscription(Image, a.gripper_image_topic,
                                             self.on_obj_raw, qos)
                else:
                    self.create_subscription(CompressedImage, a.gripper_image_topic,
                                             self.on_obj_jpg, qos)

        def on_obj_info(self, msg):
            obj_state["K"] = np.array(msg.k, float).reshape(3, 3)
            obj_state["dist"] = np.array(msg.d, float) if len(msg.d) else np.zeros(5)

        def on_obj_raw(self, msg):
            img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
            self.process_obj(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                             if img.ndim == 3 else img)

        def on_obj_jpg(self, msg):
            img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.process_obj(img)

        def process_obj(self, gray):
            if obj_state["K"] is None:
                return
            corners, ids, _ = det.detectMarkers(gray)
            if ids is None:
                return
            hit = [c[0] for c, i in zip(corners, ids.flatten())
                   if int(i) == obj_core.tag_id]
            if not hit:
                return
            r = obj_core.step(hit[0], obj_state["K"], obj_state["dist"])
            if r is None:
                return
            tvec, rvec, err = r
            if obj_pub is not None:
                # contract: position = tag centre in CAMERA OPTICAL frame [m],
                # velocity = Rodrigues rvec (camera<-tag), mode = tag id.
                # Consumer (h12_control_node) composes with believed wrist FK +
                # hand-eye extrinsic. NOTHING here enters the estimator.
                for k in range(3):
                    obj_msg.position[k] = float(tvec[k])
                    obj_msg.velocity[k] = float(rvec[k])
                obj_msg.mode = obj_core.tag_id
                obj_pub.Write(obj_msg)
            obj_state["n"] += 1
            now = time.time()
            if now - obj_state["last"] > 2.0:
                obj_state["last"] = now
                hz = obj_state["n"] / max(now - obj_state["t0"], 1e-6)
                print(f"[obj] tag{obj_core.tag_id} {obj_state['n']} poses "
                      f"({hz:.1f}Hz) t=[{tvec[0]:+.3f} {tvec[1]:+.3f} "
                      f"{tvec[2]:+.3f}]m reproj={err:.2f}px "
                      f"rej={obj_core.n_rejected}", flush=True)

        def on_info(self, msg):
            latest["K"] = np.array(msg.k, float).reshape(3, 3)
            latest["dist"] = np.array(msg.d, float) if len(msg.d) else np.zeros(5)

        def on_raw(self, msg):
            img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
            self.process(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img)

        def on_jpg(self, msg):
            img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.process(img)

        def process(self, gray):
            ls = latest["ls"]
            if ls is None or latest["K"] is None:
                return
            corners, ids, _ = det.detectMarkers(gray)
            if ids is None:
                return
            detections = {int(i): c[0] for c, i in zip(corners, ids.flatten())}
            imu_quat = np.array(list(ls.imu_state.quaternion))
            waist = float(ls.motor_state[TORSO_MOTOR].q)
            r = core.step(detections, latest["K"], latest["dist"],
                          imu_quat, waist, dt=0.033)
            if r is None:
                return
            xy, yoff, err = r[0], r[1], r[2]
            if a.abs_world:
                # 2026-08-13 ABSOLUTE MODE: publish the robot's true pose in
                # MODEL-WORLD coords (table front-left top corner at
                # (0.45, +0.2975); bundle x+ into the slab = world x+,
                # bundle y+ across = world y-). Un-rotated table-frame pose
                # (r[3]) so IMU yaw_off never leaks into position. Pair with
                # est --aux-abs (latch bypass) or the est will re-relativize.
                p_tt = r[3]
                xy = np.array([0.45 + p_tt[0], 0.2975 - p_tt[1]])
            out_msg.position[0], out_msg.position[1] = float(xy[0]), float(xy[1])
            # ★ 2026-08-18 LIVE YAW SIDE-CHANNEL: position[2] was a dead 0.0 (v4
            # fuses xy only and provably never reads z). Carry the continuously
            # tracked IMU-vs-table yaw offset [rad] so the CONTROL NODE can slew
            # its --imu_yaw_offset_deg live (--yaw_fusion, default off) instead
            # of freezing the preflight value -- the warm-IMU drift (1.7-2.5
            # deg/min measured 08-17/18) made the static offset rot 2-4 deg by
            # dive time and steered every dive left. Estimator behavior with
            # this field: byte-identical.
            out_msg.position[2] = float(yoff)
            for k in range(3):
                out_msg.velocity[k] = 0.0
            out_msg.mode = 2                       # POSITION-ONLY (v4 contract)
            pub.Write(out_msg)
            stats["n"] += 1
            now = time.time()
            if now - stats["last"] > 5.0:
                stats["last"] = now
                hz = stats["n"] / max(now - stats["t0"], 1e-6)
                print(f"[tag] {stats['n']} anchors ({hz:.1f}Hz) tags={sorted(detections)} "
                      f"reproj={err:.2f}px yaw_off={math.degrees(yoff):+.1f}deg "
                      f"rej={core.n_rejected}", flush=True)

    rclpy.init()
    node = Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
