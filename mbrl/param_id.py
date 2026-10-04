"""Parametric identification against the recordings, two stages.
Stage K (kinematic): joint zero offsets on hip pitch / knee (L,R) that minimise the left-right sole height gap after
   foot flattening over loaded samples (both feet are on the floor in reality). Pure kinematics, fast.
Stage D (dynamic): with stage-K offsets applied, per-group damping and armature scale and a per-group torque scale
   that minimise the one-step dq error (PD mode) over a fixed stratified sample set. Nelder-Mead over few params,
   samples evaluated in parallel.
Writes out/param_id.json."""
import os, sys, json, time
import numpy as np, mujoco
from multiprocessing import Pool
from scipy.optimize import minimize
from common import *
import simgap

NS = int(os.environ.get("MBRL_ID_SAMPLES", "4000"))
_m = _d = None

def pick_samples(split="train", n=NS, seed=0):
    man = [m for m in json.load(open(os.path.join(DATA, "manifest.json"))) if m["split"] == split]
    rng = np.random.RandomState(seed); per = max(2, n // len(man)); out = []
    for mm in man:
        with np.load(os.path.join(RUNS, mm["run"] + ".npz")) as z: r = {k: z[k] for k in ("idx", "tau")}
        idx = r["idx"]; idx = idx[simgap.feet_loaded(r, idx)]
        if len(idx) == 0: continue
        for i in rng.choice(idx, min(per, len(idx)), replace=False): out.append((mm["run"], int(i)))
    rng.shuffle(out); return out[:n]

KEYS = ("q", "dq", "cmd", "kp", "kd", "tau_ff", "tau", "base_quat", "gyro")

def compact(samples):
    """Per-sample two-row records (rows i and i+1, remapped to i=0) so a worker holds only what it steps.
    Loads each run once, keeps nothing else. Memory: ~5 KB per sample."""
    by_run = {}
    for name, i in samples: by_run.setdefault(name, []).append(i)
    out = []
    for name, idxs in by_run.items():
        with np.load(os.path.join(RUNS, name + ".npz")) as z:
            arrs = {k: z[k] for k in KEYS}; tl = z["tau_lim"]
        for i in idxs:
            rec = {k: arrs[k][i:i + 2].copy() for k in KEYS}; rec["tau_lim"] = tl; out.append(rec)
    return out

def _init():
    global _m, _d
    simgap._init(); _m, _d = simgap._m, simgap._d

OFF_JOINTS = [1, 3, 7, 9]   # L hip pitch, L knee, R hip pitch, R knee  (ankles are absorbed by flattening)

def apply_offsets(off):
    """offsets (rad) added to recorded q before set_state: monkey-patch via module global used in set_state"""
    simgap.Q_OFFSET = np.zeros(NJ); simgap.Q_OFFSET[OFF_JOINTS] = off

def _gap_eval(args):
    off, chunk = args; apply_offsets(np.asarray(off)); m, d = _m, _d; gaps = []
    for r in chunk:
        mujoco.mj_resetData(m, d); simgap.set_state(m, d, r, 0); gaps.append(simgap._LAST_CORR[4])
    return gaps

def _step_eval(args):
    off, scales, chunk = args; apply_offsets(np.asarray(off)); m, d = _m, _d
    dl, da, al, aa, tl_, ta = scales    # damping legs/arms, armature legs/arms, torque scale legs/arms (waist with legs)
    m.dof_damping[JV] = 10.0 * np.where(np.arange(NJ) < 13, dl, da); m.dof_armature[JV] = 0.1 * np.where(np.arange(NJ) < 13, al, aa)
    tsc = np.where(np.arange(NJ) < 13, tl_, ta); E = []
    for r in chunk:
        i = 0; tl = r["tau_lim"].astype(float); tl[~np.isfinite(tl)] = 1e9
        mujoco.mj_resetData(m, d); simgap.set_state(m, d, r, i)
        if abs(simgap._LAST_CORR[4]) >= 0.003: continue
        d.qfrc_applied[:] = 0
        for _ in range(5):
            d.qfrc_applied[JV] = tsc * pd_torque(r["kp"][i], r["kd"][i], r["cmd"][i], d.qpos[JQ], d.qvel[JV], r["tau_ff"][i], tl); mujoco.mj_step(m, d)
        E.append(d.qvel[JV] - r["dq"][i + 1])
    return E

def main():
    samples = compact(pick_samples()); print("samples", len(samples), flush=True)
    nw = int(os.environ.get("MBRL_WORKERS", "8")); chunks = [samples[k::nw] for k in range(nw)]
    res = {}
    with Pool(nw, initializer=_init) as p:
        def gap_rms(off):
            g = np.concatenate(p.map(_gap_eval, [(off, c) for c in chunks])); return float(np.sqrt(np.mean(np.square(g)))), g
        f0, g0 = gap_rms(np.zeros(4)); print(f"stage K: nominal sole-gap rms {f0*1000:.2f} mm, |gap|<3mm frac {np.mean(np.abs(g0)<0.003):.2f}", flush=True)
        t0 = time.time()
        rK = minimize(lambda x: gap_rms(np.radians(x))[0], np.zeros(4), method="Nelder-Mead", options=dict(xatol=0.05, fatol=1e-5, maxfev=150, initial_simplex=np.vstack([np.zeros(4), 2 * np.eye(4)])))
        fK, gK = gap_rms(np.radians(rK.x)); print(f"stage K: offsets deg (Lhip,Lknee,Rhip,Rknee) = {np.round(rK.x,2)}  gap rms {fK*1000:.2f} mm, consistent frac {np.mean(np.abs(gK)<0.003):.2f}  ({time.time()-t0:.0f}s, {rK.nfev} evals)", flush=True)
        res["stageK"] = dict(offsets_deg=rK.x.tolist(), gap_rms_mm_before=f0 * 1000, gap_rms_mm_after=fK * 1000, consistent_before=float(np.mean(np.abs(g0) < 0.003)), consistent_after=float(np.mean(np.abs(gK) < 0.003)))
        offK = np.radians(rK.x)
        def step_rms(scales, off):
            E = np.concatenate(p.map(_step_eval, [(off, scales, c) for c in chunks])); e = np.sqrt(np.mean(np.square(E), 0))
            return {g: float(np.sqrt(np.mean(e[ids] ** 2))) for g, ids in GROUPS.items()} | {"all": float(np.sqrt(np.mean(e ** 2))), "n": int(len(E))}
        base0 = step_rms(np.ones(6), np.zeros(4)); baseK = step_rms(np.ones(6), offK)
        print(f"stage D: one-step dq rms nominal {base0}\n         with stage-K offsets {baseK}", flush=True)
        t0 = time.time()
        rD = minimize(lambda x: step_rms(np.exp(x), offK)["all"], np.zeros(6), method="Nelder-Mead", options=dict(xatol=0.02, fatol=1e-5, maxfev=250, initial_simplex=np.vstack([np.zeros(6), 0.7 * np.eye(6)])))
        sc = np.exp(rD.x); fin = step_rms(sc, offK)
        print(f"stage D: scales damping L/A {sc[0]:.2f}/{sc[1]:.2f} armature L/A {sc[2]:.2f}/{sc[3]:.2f} torque L/A {sc[4]:.3f}/{sc[5]:.3f} -> {fin}  ({time.time()-t0:.0f}s, {rD.nfev} evals)", flush=True)
        res["stageD"] = dict(scales=dict(damping_legs=sc[0], damping_arms=sc[1], armature_legs=sc[2], armature_arms=sc[3], torque_legs=sc[4], torque_arms=sc[5]), one_step_nominal=base0, one_step_offsets=baseK, one_step_fitted=fin)
        # held-out check on test samples
    test = compact(pick_samples("test", 1500, seed=1))
    with Pool(nw, initializer=_init) as p:
        chunks = [test[k::nw] for k in range(nw)]
        def step_rms(scales, off):
            E = np.concatenate(p.map(_step_eval, [(off, scales, c) for c in chunks])); e = np.sqrt(np.mean(np.square(E), 0))
            return {g: float(np.sqrt(np.mean(e[ids] ** 2))) for g, ids in GROUPS.items()} | {"all": float(np.sqrt(np.mean(e ** 2))), "n": int(len(E))}
        res["test"] = dict(nominal=step_rms(np.ones(6), np.zeros(4)), offsets=step_rms(np.ones(6), offK), fitted=step_rms(sc, offK))
        print("test:", json.dumps(res["test"], indent=0), flush=True)
    json.dump(res, open(os.path.join(OUT, "param_id.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
