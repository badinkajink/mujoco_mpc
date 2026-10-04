"""One-step sim-gap of the planner plant on real H1-2 transitions.
For each recorded sample t: reset the MuJoCo plant to the recorded (pelvis pose from estimator
+ torso IMU, joints q/dq), apply torque for 10 ms (5 x 2 ms substeps) and compare to recorded t+1.
Two torque models:
  pd  : tau = clip(kp*(cmd-q) - kd*dq + tau_ff, tau_lim) recomputed every substep  (full twin)
  tau : the measured tau_est held for 10 ms                                      (rigid-body+contact only)
Plant: table moved 100 m away, free object parked far; pelvis z from foot-floor anchoring, pelvis lin vel from
zero mean foot velocity; only both-feet-loaded samples (gantry-hung / limp legs dropped).
Outputs data/simgap/<run>.npz with predictions on the sampled indices.
"""
import os, sys, json, time
import numpy as np, mujoco
from multiprocessing import Pool
from common import *

SIMG = os.path.join(DATA, "simgap"); os.makedirs(SIMG, exist_ok=True)
STRIDE = int(os.environ.get("MBRL_STRIDE", "2"))
# effective ankle-pitch zero offsets (rad) added to the recorded q before anchoring: sweep over 4 runs from 4 days gave
# L +1..+2 deg, R +4..+8 deg (out/ankle_offset_sweep.json); matches the known right-ankle zero error.
ANKLE_OFF = np.radians([float(x) for x in os.environ.get("MBRL_ANKLE_OFF", "0,0").split(",")])  # superseded by flatten_feet
_m = None; _d = None

def make_model():
    """planner plant with the study's scene edits: table moved away, planner-only / gripper geoms non-colliding."""
    m = load_model(); m.body_pos[m.body("table").id] += [100, 0, 0]
    for g in range(m.ngeom):
        bn = m.body(m.geom_bodyid[g]).name
        # planner-only virtual geoms, and the gripper/wrist collision shapes: in the recordings the real hands sit closer
        # than these shapes allow, so they produce phantom self-collision forces (arm error 1.5 -> 0.11 rad/s without them)
        if "keepaway" in (m.geom(g).name or "") or "gripper" in bn or "wrist" in bn:
            m.geom_contype[g] = 0; m.geom_conaffinity[g] = 0
    return m

def _init():
    global _m, _d
    _m = make_model(); _d = mujoco.MjData(_m)

def feet_loaded(r, idx, thr=5.0):
    """both legs carrying load: |tau| on knee OR ankle pitch > thr Nm on each leg (drops gantry-hung / limp-leg samples)."""
    kn = np.abs(r["tau"][idx][:, [3, 9]]); ap = np.abs(r["tau"][idx][:, [4, 10]])
    return (kn > thr).all(1) | (ap > thr).all(1)

_FOOT = None
_LAST_CORR = np.zeros(7)
Q_OFFSET = np.zeros(NJ)   # joint zero offsets (rad) under identification, added to recorded q
def _foot_verts(m):
    """world-frame lowest-point helper: (geom id, local vertices) for the two foot collision meshes."""
    out = []
    for bn in ("left_ankle_roll_link", "right_ankle_roll_link"):
        bid = m.body(bn).id
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == bid and m.geom_contype[g] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]; a = m.mesh_vertadr[mid]; n = m.mesh_vertnum[mid]
                out.append((g, bid, m.mesh_vert[a:a + n].copy()))
    return out

_ANKLE_DOF = {"left_ankle_roll_link": (10, 11), "right_ankle_roll_link": (16, 17)}
FLAT_MAX = np.radians(float(os.environ.get("MBRL_FLAT_MAX", "10")))

def flatten_feet(m, d):
    """Rotate each foot flat onto the floor through its own ankle pitch/roll (the sole is the foot body's -z face).
    The real feet are flat when loaded; the plant's are not (ankle zero offsets, sole geometry), and a foot on an edge
    produces a contact impulse at reset. Corrections above FLAT_MAX (true heel lift) are reverted.
    Returns the per-foot (pitch, roll) corrections in rad."""
    global _FOOT
    if _FOOT is None: _FOOT = _foot_verts(m)
    corr = np.zeros(4); jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
    for k, (_, bid, _) in enumerate(_FOOT):
        dp, dr = _ANKLE_DOF[m.body(bid).name]; q0 = d.qpos[[dp + 1, dr + 1]].copy(); tot = np.zeros(2)
        for _ in range(5):
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            z = d.xmat[bid].reshape(3, 3)[:, 2]; e = np.cross(z, [0, 0, 1.0])
            if np.linalg.norm(e) < 1e-5: break
            mujoco.mj_jacBody(m, d, jacp, jacr, bid)
            dlt = np.linalg.lstsq(jacr[:, [dp, dr]], e, rcond=None)[0]; tot += dlt
            d.qpos[dp + 1] += dlt[0]; d.qpos[dr + 1] += dlt[1]
        if np.abs(tot).max() > FLAT_MAX: d.qpos[[dp + 1, dr + 1]] = q0; tot[:] = np.nan
        corr[2 * k:2 * k + 2] = tot
    return corr

_KNEE_DOF = {"left_ankle_roll_link": 9, "right_ankle_roll_link": 15}
GAP_MAX = float(os.environ.get("MBRL_GAP_MAX", "0.0"))   # 0 = measure the gap only (knee closure is ill-conditioned near a straight leg)

def sole_z(m, d):
    mujoco.mj_kinematics(m, d)
    return np.array([((d.geom_xmat[g].reshape(3, 3) @ v.T).T[:, 2] + d.geom_xpos[g][2]).min() for g, _, v in _FOOT])

def close_sole_gap(m, d):
    """Both feet are loaded, so both soles are on the floor in reality; in the plant they differ by up to ~12 mm
    (leg kinematic calibration, grows with knee asymmetry). Bend the knee of the higher foot until its sole matches the
    lower one, re-flattening that foot. Returns (gap_before_m, knee_correction_rad, foot_index); gap > GAP_MAX left as is."""
    z = sole_z(m, d); gap = z[0] - z[1]
    if abs(gap) < 1e-4 or abs(gap) > GAP_MAX: return np.array([gap, 0.0, -1.0])
    k = 0 if gap > 0 else 1           # higher foot
    g, bid, v = _FOOT[k]; dk = _KNEE_DOF[m.body(bid).name]
    jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv)); tot = 0.0
    for _ in range(6):
        z = sole_z(m, d); e = z[k] - z[1 - k]
        if abs(e) < 2e-4: break
        mujoco.mj_comPos(m, d); mujoco.mj_jacBody(m, d, jacp, jacr, bid)
        dz = jacp[2, dk]
        if abs(dz) < 1e-3: break
        step = -e / dz; d.qpos[dk + 1] += step; tot += step
        flatten_feet(m, d)
    return np.array([gap, tot, float(k)])

def anchor_feet(m, d, tau=None):
    """Pelvis z so the lowest foot-mesh vertex touches the floor at the equilibrium depth for the given joint torques;
    pelvis linear + angular velocity from the stationary-feet constraint (least squares over both feet).
    Valid when both feet are loaded (the tasks never walk)."""
    global _FOOT
    if _FOOT is None: _FOOT = _foot_verts(m)
    d.qpos[2] = 0.0; mujoco.mj_kinematics(m, d)
    zmin = min(((d.geom_xmat[g].reshape(3, 3) @ v.T).T[:, 2] + d.geom_xpos[g][2]).min() for g, _, v in _FOOT)
    d.qpos[2] = -zmin
    if tau is not None:
        d.qfrc_applied[:] = 0; d.qfrc_applied[JV] = tau
        lo, hi = -zmin - 0.03, -zmin + 0.002
        for _ in range(14):
            mid = 0.5 * (lo + hi); d.qpos[2] = mid; mujoco.mj_forward(m, d)
            if d.qacc[2] > 0: lo = mid
            else: hi = mid
        d.qpos[2] = 0.5 * (lo + hi); d.qfrc_applied[:] = 0
    mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
    J = np.zeros((12, m.nv)); jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
    for k, (_, bid, _) in enumerate(_FOOT):
        mujoco.mj_jacBody(m, d, jacp, jacr, bid); J[6 * k:6 * k + 3] = jacp; J[6 * k + 3:6 * k + 6] = jacr
    # free-joint rotational dofs are in the body frame: J already accounts for that (mj_jac maps qvel -> world)
    rhs = -J[:, 6:] @ d.qvel[6:]
    d.qvel[0:6] = np.linalg.lstsq(J[:, :6], rhs, rcond=None)[0]
    mujoco.mj_forward(m, d)

def set_state(m, d, r, i):
    """Write recorded sample i into d: joints, pelvis orientation from the torso IMU through the waist, feet flattened
    and anchored on the floor, base velocity from stationary feet."""
    global _LAST_CORR
    q = r["q"][i].astype(float) + Q_OFFSET; dq = r["dq"][i].astype(float)
    d.qpos[:] = m.qpos0; d.qpos[0:3] = 0; d.qpos[3:7] = [1, 0, 0, 0]; d.qpos[JQ] = q
    mujoco.mj_kinematics(m, d)
    R_rel = d.site("imu").xmat.reshape(3, 3).copy()          # imu frame in pelvis frame (pelvis at identity)
    R_imu = np.zeros(9); bq = r["base_quat"][i].astype(float); mujoco.mju_quat2Mat(R_imu, bq / np.linalg.norm(bq))
    R_pel = R_imu.reshape(3, 3) @ R_rel.T
    pq = np.zeros(4); mujoco.mju_mat2Quat(pq, R_pel.reshape(9))
    gyro = r["gyro"][i].astype(float)
    w_pel = R_rel @ (gyro - np.array([0, 0, dq[12]]))      # initial guess; replaced by the stationary-feet solve
    d.qpos[3:7] = pq
    d.qpos[7 + 4] += ANKLE_OFF[0]; d.qpos[7 + 10] += ANKLE_OFF[1]
    d.qpos[34:37] = [50, 50, 0.5]   # park the free object far away
    d.qvel[:] = 0; d.qvel[3:6] = w_pel; d.qvel[JV] = dq
    corr = flatten_feet(m, d)
    knee = close_sole_gap(m, d)
    _LAST_CORR = np.concatenate([corr, knee])
    anchor_feet(m, d, r["tau"][i].astype(float))
    d.qacc_warmstart[:] = 0; d.time = 0
    return R_rel

def step10(m, d, mode, r, i, nsub, tau_lim):
    d.qfrc_applied[:] = 0
    for k in range(nsub):
        if mode == "pd":
            tau = pd_torque(r["kp"][i], r["kd"][i], r["cmd"][i], d.qpos[JQ], d.qvel[JV], r["tau_ff"][i], tau_lim)
        else:
            tau = np.clip(r["tau"][i].astype(float), -tau_lim, tau_lim)
        d.qfrc_applied[JV] = tau
        mujoco.mj_step(m, d)
    return d.qpos[JQ].copy(), d.qvel[JV].copy(), d.qvel[3:6].copy(), int(d.ncon)

def run_one(name):
    m, d = _m, _d
    with np.load(os.path.join(RUNS, name + ".npz")) as z:
        r = {k: z[k] for k in z.files}   # materialise: NpzFile re-reads on every access
    idx = r["idx"][::STRIDE]
    idx = idx[feet_loaded(r, idx)]
    n = len(idx)
    tau_lim = r["tau_lim"].astype(float); tau_lim[~np.isfinite(tau_lim)] = 1e9
    nsub = int(round(REC_DT / m.opt.timestep))
    out = {k: np.zeros((n, NJ), np.float32) for k in ("q_pd", "dq_pd", "q_tau", "dq_tau", "tau_pd0")}
    out["w_pd"] = np.zeros((n, 3), np.float32); out["w_tau"] = np.zeros((n, 3), np.float32)
    out["ncon"] = np.zeros(n, np.int16); out["tilt"] = np.zeros(n, np.float32); out["ok"] = np.ones(n, bool)
    out["w_rec0"] = np.zeros((n, 3), np.float32); out["w_rec1"] = np.zeros((n, 3), np.float32)
    out["foot_corr"] = np.zeros((n, 7), np.float32)   # L(p,r) R(p,r) corrections rad, sole gap m, knee corr rad, which foot
    for j, i in enumerate(idx):
        try:
            mujoco.mj_resetData(m, d)
            R_rel = set_state(m, d, r, i); out["foot_corr"][j] = _LAST_CORR
            out["tilt"][j] = tilt_deg(r["base_quat"][i].astype(float))
            out["tau_pd0"][j] = pd_torque(r["kp"][i], r["kd"][i], r["cmd"][i], r["q"][i], r["dq"][i], r["tau_ff"][i], tau_lim)
            out["w_rec0"][j] = d.qvel[3:6]
            q1, dq1, w1, nc = step10(m, d, "pd", r, i, nsub, tau_lim)
            out["q_pd"][j], out["dq_pd"][j], out["w_pd"][j], out["ncon"][j] = q1, dq1, w1, nc
            mujoco.mj_resetData(m, d); set_state(m, d, r, i)
            q1, dq1, w1, nc = step10(m, d, "tau", r, i, nsub, tau_lim)
            out["q_tau"][j], out["dq_tau"][j], out["w_tau"][j] = q1, dq1, w1
            # recorded next pelvis angvel for reference
            mujoco.mj_resetData(m, d); set_state(m, d, r, i + 1); out["w_rec1"][j] = d.qvel[3:6]
            if not (np.isfinite(out["dq_pd"][j]).all() and np.isfinite(out["dq_tau"][j]).all()):
                out["ok"][j] = False
        except Exception as e:
            out["ok"][j] = False
    np.savez(os.path.join(SIMG, name + ".npz"), idx=idx.astype(np.int32), **out)
    return name, n, int(out["ok"].sum())

if __name__ == "__main__":
    man = json.load(open(os.path.join(DATA, "manifest.json")))
    names = [r["run"] for r in man]
    if len(sys.argv) > 1: names = names[:int(sys.argv[1])]
    t0 = time.time()
    with Pool(int(os.environ.get("MBRL_WORKERS", "20")), initializer=_init) as p:
        for k, (name, n, ok) in enumerate(p.imap_unordered(run_one, names)):
            print(f"[{k+1}/{len(names)}] {name:45s} {n:6d} ok {ok:6d}  {time.time()-t0:6.0f}s", flush=True)
    print("done", time.time() - t0)
