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
ANKLE_OFF = np.radians([float(x) for x in os.environ.get("MBRL_ANKLE_OFF", "1,5").split(",")])
_m = None; _d = None

def _init():
    global _m, _d
    _m = load_model(); _m.body_pos[_m.body("table").id] += [100, 0, 0]
    for g in range(_m.ngeom):
        bn = _m.body(_m.geom_bodyid[g]).name
        # planner-only virtual geoms, and the gripper/wrist collision shapes: in the recordings the real hands sit closer
        # than these shapes allow, so they produce phantom self-collision forces (arm error 1.5 -> 0.11 rad/s without them)
        if "keepaway" in (_m.geom(g).name or "") or "gripper" in bn or "wrist" in bn:
            _m.geom_contype[g] = 0; _m.geom_conaffinity[g] = 0
    _d = mujoco.MjData(_m)

def feet_loaded(r, idx, thr=5.0):
    """both legs carrying load: |tau| on knee OR ankle pitch > thr Nm on each leg (drops gantry-hung / limp-leg samples)."""
    kn = np.abs(r["tau"][idx][:, [3, 9]]); ap = np.abs(r["tau"][idx][:, [4, 10]])
    return (kn > thr).all(1) | (ap > thr).all(1)

_FOOT = None
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

def anchor_feet(m, d, tau=None):
    """Pelvis z so the lowest foot-mesh vertex touches the floor; pelvis linear velocity so the mean foot velocity is zero.
    Valid when both feet are loaded (the tasks never walk)."""
    global _FOOT
    if _FOOT is None: _FOOT = _foot_verts(m)
    d.qpos[2] = 0.0; mujoco.mj_kinematics(m, d)
    zmin = min(((d.geom_xmat[g].reshape(3, 3) @ v.T).T[:, 2] + d.geom_xpos[g][2]).min() for g, _, v in _FOOT)
    d.qpos[2] = -zmin
    # equilibrium penetration: bisect pelvis z so the world vertical acceleration of the base is ~0
    # under the measured joint torques (the real feet carry the weight; sim soft contact needs some depth)
    if tau is not None:
        d.qfrc_applied[:] = 0; d.qfrc_applied[JV] = tau
        lo, hi = -zmin - 0.03, -zmin + 0.002
        for _ in range(14):
            mid = 0.5 * (lo + hi); d.qpos[2] = mid; mujoco.mj_forward(m, d)
            if d.qacc[2] > 0: lo = mid     # pushed up too hard -> less penetration
            else: hi = mid
        d.qpos[2] = 0.5 * (lo + hi); d.qfrc_applied[:] = 0
    d.qvel[0:3] = 0; mujoco.mj_forward(m, d)
    vel = np.zeros(6); vs = []
    for _, bid, _ in _FOOT:
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, vel, 0); vs.append(vel[3:6].copy())
    d.qvel[0:3] = -np.mean(vs, 0)

def set_state(m, d, r, i):
    q = r["q"][i].astype(float); dq = r["dq"][i].astype(float)
    # pelvis orientation / angular velocity from torso IMU through the waist
    d.qpos[:] = m.qpos0; d.qpos[JQ] = q
    mujoco.mj_kinematics(m, d)
    # site orientation relative to pelvis = site xmat when pelvis quat is identity (qpos0 has identity quat? enforce)
    d.qpos[0:3] = 0; d.qpos[3:7] = [1, 0, 0, 0]; mujoco.mj_kinematics(m, d)
    R_rel = d.site("imu").xmat.reshape(3, 3).copy()
    R_imu = np.zeros(9); bq = r["base_quat"][i].astype(float); mujoco.mju_quat2Mat(R_imu, bq / np.linalg.norm(bq))
    R_pel = R_imu.reshape(3, 3) @ R_rel.T
    pq = np.zeros(4); mujoco.mju_mat2Quat(pq, R_pel.reshape(9))
    gyro = r["gyro"][i].astype(float)
    w_pel = R_rel @ (gyro - np.array([0, 0, dq[12]]))
    d.qpos[0:3] = [0, 0, 0]; d.qpos[3:7] = pq; d.qpos[JQ] = q
    d.qpos[7 + 4] += ANKLE_OFF[0]; d.qpos[7 + 10] += ANKLE_OFF[1]
    d.qpos[34:37] = [50, 50, 0.5]   # park the free object far away
    d.qvel[:] = 0; d.qvel[3:6] = w_pel; d.qvel[JV] = dq
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
    for j, i in enumerate(idx):
        try:
            mujoco.mj_resetData(m, d)
            R_rel = set_state(m, d, r, i)
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
