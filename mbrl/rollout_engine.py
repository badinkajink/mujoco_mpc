"""Parallel nonlinear rollouts of the planner plant on recorded segments, with a pluggable numpy residual MLP and
pluggable plant parameters. Memory-safe: each worker receives only its compact segments (rows i0..i0+T+1).

segment record: dict(q, dq, cmd, kp, kd, tau_ff, tau, base_quat, gyro  -> (T+2, 27|4|3), tau_lim (27), run, i0, tilt0)
rollout(seg, params, mlp) -> dict(q_sim (T,27) after each 10 ms step (after correction), dq_sim (T,27),
                                  dq_plant (T,27) = plant prediction BEFORE the correction, X (T,168) features at step start)
"""
import os, json
import numpy as np, mujoco
from multiprocessing import Pool
from common import *
import simgap
from residual_fit import gvec

KEYS = ("q", "dq", "cmd", "kp", "kd", "tau_ff", "tau", "base_quat", "gyro")
_m = _d = None

def _init():
    global _m, _d
    simgap._init(); _m, _d = simgap._m, simgap._d

def filter_consistent(eng, segs, T):
    """keep the segments whose nominal-plant reset is consistent (both soles within 3 mm) so parameter changes
    cannot shift the evaluation set"""
    outs = eng.run(segs, 1, {}); return [s for s, o in zip(segs, outs) if o is not None]

def compact_segments(split, T, n_max, seed=0, stride=50, require_consistent=True):
    """pick segment starts (loaded feet, no gap in time) from the runs of a split; consistency checked later in-worker"""
    man = [m for m in json.load(open(os.path.join(DATA, "manifest.json"))) if m["split"] == split]
    rng = np.random.RandomState(seed); out = []
    per = max(1, int(1.5 * n_max / len(man)) + 1)
    for mm in man:
        with np.load(os.path.join(RUNS, mm["run"] + ".npz")) as z:
            r = {k: z[k] for k in KEYS}; idx = z["idx"]; t = z["t"]; tl = z["tau_lim"]
        idx = idx[(idx + T + 2 < len(t))]
        if len(idx) == 0: continue
        idx = idx[simgap.feet_loaded(r, idx)]
        if len(idx) == 0: continue
        dt = np.diff(t); idx = idx[np.all(dt[idx[:, None] + np.arange(T + 1)] < 0.012, 1)]
        if len(idx) == 0: continue
        idx = idx[::stride]
        for i0 in rng.choice(idx, min(per, len(idx)), replace=False):
            i0 = int(i0); rec = {k: r[k][i0:i0 + T + 2].copy() for k in KEYS}
            rec["tau_lim"] = tl; rec["run"] = mm["run"]; rec["i0"] = i0; rec["tilt0"] = tilt_deg(r["base_quat"][i0].astype(float)); out.append(rec)
    rng.shuffle(out); return out[:n_max]

class NumpyMLP:
    """mirror of residual_fit.MLP (Linear+SiLU x3, Linear) with standardisation and clipping, numpy forward"""
    def __init__(self, state, mu, sd, ymu, ysd, clip):
        self.W = [state[f"net.{i}.weight"].cpu().numpy() if hasattr(state[f"net.{i}.weight"], "cpu") else state[f"net.{i}.weight"] for i in (0, 2, 4, 6)]
        self.b = [state[f"net.{i}.bias"].cpu().numpy() if hasattr(state[f"net.{i}.bias"], "cpu") else state[f"net.{i}.bias"] for i in (0, 2, 4, 6)]
        self.mu, self.sd, self.ymu, self.ysd, self.clip = mu, sd, ymu, ysd, clip
    def __call__(self, x):
        h = (x - self.mu) / self.sd
        for W, b in zip(self.W[:-1], self.b[:-1]):
            h = h @ W.T + b; h = h / (1 + np.exp(-h))
        y = (h @ self.W[-1].T + self.b[-1]) * self.ysd + self.ymu
        return np.clip(y, -self.clip, self.clip)

def apply_params(m, params):
    """params: dict of plant modifications. keys: damping_legs/arms (scale), armature_legs/arms, torque_legs/arms,
    q_offsets (27 rad added to recorded q), nsub, dt"""
    legs = np.arange(NJ) < 13
    m.dof_damping[JV] = 10.0 * np.where(legs, params.get("damping_legs", 1.0), params.get("damping_arms", 1.0))
    m.dof_armature[JV] = 0.1 * np.where(legs, params.get("armature_legs", 1.0), params.get("armature_arms", 1.0))
    m.opt.timestep = params.get("dt", 0.002)
    simgap.Q_OFFSET = np.asarray(params.get("q_offsets", np.zeros(NJ)), float)
    return np.where(legs, params.get("torque_legs", 1.0), params.get("torque_arms", 1.0)), int(params.get("nsub", 5))

def features(d, r, i, tl):
    mujoco.mj_kinematics(_m, d); bq = np.zeros(4); mujoco.mju_mat2Quat(bq, d.site("imu").xmat)
    q = d.qpos[JQ].copy(); dq = d.qvel[JV].copy()
    tau0 = pd_torque(r["kp"][i], r["kd"][i], r["cmd"][i], q, dq, r["tau_ff"][i], tl)
    return np.concatenate([q, dq, tau0, r["cmd"][i] - q, r["kp"][i] / 100.0, r["kd"][i] / 10.0, gvec(bq[None])[0], d.qvel[3:6]]).astype(np.float32)

def rollout(r, T, params, mlp=None, want_X=False):
    """inject = params.get("inject", "post"): post = dq += r after the step (study protocol);
    pre = dq += r before the step (what the planner hook Task::ModifyRolloutState can do)."""
    m, d = _m, _d; tsc, nsub = apply_params(m, params); pre = params.get("inject", "post") == "pre"
    tl = r["tau_lim"].astype(float); tl[~np.isfinite(tl)] = 1e9
    mujoco.mj_resetData(m, d); simgap.set_state(m, d, r, 0)
    if abs(simgap._LAST_CORR[4]) >= 0.003 and not params.get("no_skip", False): return None
    out = dict(q_sim=np.zeros((T, NJ), np.float32), dq_sim=np.zeros((T, NJ), np.float32), dq_plant=np.zeros((T, NJ), np.float32), q0_sim=d.qpos[JQ].copy())
    if want_X: out["X"] = np.zeros((T, 168), np.float32)
    for t in range(T):
        x = features(d, r, t, tl) if (mlp is not None or want_X) else None
        if want_X: out["X"][t] = x
        if mlp is not None and pre: d.qvel[JV] += mlp(x[None])[0]
        d.qfrc_applied[:] = 0
        for _ in range(nsub):
            d.qfrc_applied[JV] = tsc * pd_torque(r["kp"][t], r["kd"][t], r["cmd"][t], d.qpos[JQ], d.qvel[JV], r["tau_ff"][t], tl); mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all() or np.abs(d.qvel[JV]).max() > 50: return dict(out, diverged_at=t)
        out["dq_plant"][t] = d.qvel[JV]
        if mlp is not None and not pre: d.qvel[JV] += mlp(x[None])[0]
        out["q_sim"][t] = d.qpos[JQ]; out["dq_sim"][t] = d.qvel[JV]
    return out

def _work(args):
    segs, T, params, mlp, want_X = args
    return [rollout(r, T, params, mlp, want_X) for r in segs]

class Engine:
    def __init__(self, nw=8):
        self.p = Pool(nw, initializer=_init); self.nw = nw
    def run(self, segs, T, params, mlp=None, want_X=False):
        chunks = [segs[k::self.nw] for k in range(self.nw)]
        res = self.p.map(_work, [(c, T, params, mlp, want_X) for c in chunks])
        out = [None] * len(segs)
        for k, c in enumerate(res):
            for j, o in enumerate(c): out[k + j * self.nw] = o
        return out
    def close(self): self.p.close(); self.p.join()

def score(segs, outs, horizons=(10, 25, 50)):
    """displacement RMSE per group per horizon over non-diverged, consistent segments; and the diverged/skipped counts"""
    res = {"n": len(segs), "skipped": 0, "diverged": 0}
    E = {h: [] for h in horizons}
    for r, o in zip(segs, outs):
        if o is None: res["skipped"] += 1; continue
        if "diverged_at" in o: res["diverged"] += 1; continue
        for h in horizons:
            if h <= len(o["q_sim"]): E[h].append((o["q_sim"][h - 1] - o["q0_sim"]) - (r["q"][h] - r["q"][0]))
    for h in horizons:
        e = np.array(E[h]) if E[h] else np.zeros((0, NJ))
        res[f"{h*10}ms"] = {g: float(np.sqrt(np.mean(e[:, ids] ** 2))) if len(e) else None for g, ids in GROUPS.items()}
    return res
