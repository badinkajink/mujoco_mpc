"""One-step sim-gap of the planner plant on real H1-2 transitions.
For each recorded sample t: reset the MuJoCo plant to the recorded (pelvis pose from estimator
+ torso IMU, joints q/dq), apply torque for 10 ms (5 x 2 ms substeps) and compare to recorded t+1.
Two torque models:
  pd  : tau = clip(kp*(cmd-q) - kd*dq + tau_ff, tau_lim) recomputed every substep  (full twin)
  tau : the measured tau_est held for 10 ms                                      (rigid-body+contact only)
Outputs data/simgap/<run>.npz with predictions on the sampled indices.
"""
import os, sys, json, time
import numpy as np, mujoco
from multiprocessing import Pool
from common import *

SIMG = os.path.join(DATA, "simgap"); os.makedirs(SIMG, exist_ok=True)
STRIDE = int(os.environ.get("MBRL_STRIDE", "2"))
_m = None; _d = None

def _init():
    global _m, _d
    _m = load_model(); _d = mujoco.MjData(_m)

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
    d.qpos[0:3] = r["base_pos"][i]; d.qpos[3:7] = pq; d.qpos[JQ] = q
    d.qvel[:] = 0; d.qvel[0:3] = r["base_vel"][i, :3]; d.qvel[3:6] = w_pel; d.qvel[JV] = dq
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
    r = np.load(os.path.join(RUNS, name + ".npz"))
    idx = r["idx"][::STRIDE]
    bp = r["bp_ok"]
    idx = idx[bp[idx]]
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
