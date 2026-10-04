"""Nominal plant rollouts + finite-difference Jacobians of the 10 ms macro step (PD torque recomputed per substep)
along each rollout, for linearised multi-step residual training.
State tangent (66): [pos tangent 33 = base pos 3, base rotvec 3, joints 27 | vel 33]. Object dofs excluded.
Segments: train/val runs, both feet loaded + consistent start, T steps inside the run, plant rollout finite.
Writes data/jac/<chunk>.npz : run, i0, split, xn (S,T+1,67 = qpos34+qvel33), A (S,T,66,66 f32), ok."""
import os, sys, json, time
import numpy as np, mujoco
from multiprocessing import Pool
from common import *
import simgap

JAC = os.path.join(DATA, "jac"); os.makedirs(JAC, exist_ok=True)
T = int(os.environ.get("MBRL_T", "25")); NX = 66
# contact dynamics are non-smooth below ~1 mm: FD at the scale of the corrections we will apply (1 mm / 1 mrad, 10 mrad/s)
EPS_P = 1e-3; EPS_V = 1e-2
_m = _d = None

def _init():
    global _m, _d
    simgap._init(); _m, _d = simgap._m, simgap._d

def get_x(d):
    return np.concatenate([d.qpos[:34].copy(), d.qvel[:33].copy()])

def set_x(d, x):
    d.qpos[:34] = x[:34]; d.qvel[:33] = x[34:]

def perturb(m, d, x, k):
    set_x(d, x)
    if k < 33:
        dpos = np.zeros(m.nv); dpos[k] = EPS_P; mujoco.mj_integratePos(m, d.qpos, dpos, 1.0)
    else:
        d.qvel[k - 33] += EPS_V

def tangent_diff(m, x1, x0):
    """x1 - x0 in tangent coords (66)"""
    out = np.zeros(NX); dp = np.zeros(m.nv)
    q1 = np.zeros(m.nq); q0 = np.zeros(m.nq); q1[:34] = x1[:34]; q0[:34] = x0[:34]
    mujoco.mj_differentiatePos(m, dp, 1.0, q0, q1); out[:33] = dp[:33]; out[33:] = x1[34:] - x0[34:]
    return out

def macro(m, d, r, i, tl):
    d.qfrc_applied[:] = 0
    for _ in range(5):
        d.qfrc_applied[JV] = pd_torque(r["kp"][i], r["kd"][i], r["cmd"][i], d.qpos[JQ], d.qvel[JV], r["tau_ff"][i], tl)
        mujoco.mj_step(m, d)

def segment(m, d, r, i0, tl):
    mujoco.mj_resetData(m, d); simgap.set_state(m, d, r, i0)
    if abs(simgap._LAST_CORR[4]) >= 0.003: return None
    xn = np.zeros((T + 1, 67)); A = np.zeros((T, NX, NX), np.float32)
    xn[0] = get_x(d); ws = d.qacc_warmstart.copy(); ctrl0 = d.ctrl.copy()
    for t in range(T):
        i = i0 + t
        # nominal step
        set_x(d, xn[t]); d.qacc_warmstart[:] = ws; d.time = 0; macro(m, d, r, i, tl)
        xn[t + 1] = get_x(d); ws_next = d.qacc_warmstart.copy()
        if not np.isfinite(xn[t + 1]).all(): return None
        for k in range(NX):
            perturb(m, d, xn[t], k); d.qacc_warmstart[:] = ws; d.time = 0; macro(m, d, r, i, tl)
            A[t, :, k] = tangent_diff(m, get_x(d), xn[t + 1]) / (EPS_P if k < 33 else EPS_V)
        ws = ws_next
    if not np.isfinite(A).all() or np.abs(A).max() > 1e5: return None
    return xn, A

def work(args):
    name, starts, split = args
    m, d = _m, _d
    with np.load(os.path.join(RUNS, name + ".npz")) as z: r = {k: z[k] for k in z.files}
    tl = r["tau_lim"].astype(float); tl[~np.isfinite(tl)] = 1e9
    out = []
    for i0 in starts:
        res = segment(m, d, r, int(i0), tl)
        if res is not None: out.append((int(i0), res[0].astype(np.float32), res[1].astype(np.float32)))
    if out:
        np.savez(os.path.join(JAC, name + ".npz"), run=name, split=split, i0=np.array([o[0] for o in out]),
                 xn=np.stack([o[1] for o in out]), A=np.stack([o[2] for o in out]))
    return name, len(starts), len(out)

if __name__ == "__main__":
    man = json.load(open(os.path.join(DATA, "manifest.json")))
    stride = int(os.environ.get("MBRL_SEG_STRIDE", "100")); per_run = int(os.environ.get("MBRL_SEG_PER_RUN", "15"))
    jobs = []
    for mm in man:
        if mm["split"] not in ("train", "val"): continue
        with np.load(os.path.join(RUNS, mm["run"] + ".npz")) as z: r = {k: z[k] for k in ("idx", "tau", "t")}
        idx = r["idx"]; idx = idx[simgap.feet_loaded(r, idx) & (idx + T + 1 < len(r["t"]))]
        dt = np.diff(r["t"]); idx = idx[np.all(dt[idx[:, None] + np.arange(T)] < 0.012, 1)] if len(idx) else idx
        starts = idx[::stride][:per_run]
        if len(starts): jobs.append((mm["run"], starts, mm["split"]))
    if len(sys.argv) > 1: jobs = jobs[:int(sys.argv[1])]
    print("jobs", len(jobs), "segments requested", sum(len(j[1]) for j in jobs), flush=True)
    t0 = time.time(); tot = 0
    with Pool(int(os.environ.get("MBRL_WORKERS", "20")), initializer=_init) as p:
        for k, (name, n, ok) in enumerate(p.imap_unordered(work, jobs)):
            tot += ok; print(f"[{k+1}/{len(jobs)}] {name:45s} {ok:3d}/{n:3d}  total {tot}  {time.time()-t0:6.0f}s", flush=True)
