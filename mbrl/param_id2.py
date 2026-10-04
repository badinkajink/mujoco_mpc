"""Parametric identification v2 with a horizon objective (the one-step objective is degenerate: an immovable robot
wins). Objective = 250 ms open-loop rollout displacement RMSE (legs + arms, plant only) on fixed training segments,
parameters bounded in log-scale [1/3, 3]: damping legs/arms, armature legs/arms, torque scale legs/arms; plus
stage K joint zero offsets (hip pitch / knee L,R, bounded +-4 deg) scored by the same objective.
Nelder-Mead, parallel rollouts, memory-safe. Writes out/param_id2.json."""
import os, json, time
import numpy as np
from scipy.optimize import minimize
from common import *
import rollout_engine as RE

T = 25; NSEG = int(os.environ.get("MBRL_ID_SEGS", "400")); NW = int(os.environ.get("MBRL_WORKERS", "8"))
PARAMS0 = dict(nsub=int(os.environ.get("MBRL_NSUB", "5")), dt=float(os.environ.get("MBRL_DT", "0.002")))

def main():
    t0 = time.time(); segs = RE.compact_segments("train", T, NSEG, seed=3); test = RE.compact_segments("test", T, 300, seed=4)
    eng = RE.Engine(NW); res = {}
    def objective(params, S=segs):
        sc = RE.score(S, eng.run(S, T, dict(PARAMS0, **params)), horizons=(10, 25))
        if sc["250ms"]["legs"] is None: return 1.0, sc
        return sc["250ms"]["legs"] + sc["250ms"]["arms"] + 0.002 * sc["diverged"], sc
    f0, s0 = objective({}); print(f"nominal: {f0:.5f} {json.dumps(s0)}", flush=True); res["nominal"] = s0
    # stage D: six log-scales
    keys = ["damping_legs", "damping_arms", "armature_legs", "armature_arms", "torque_legs", "torque_arms"]
    def fD(x): return objective({k: float(np.exp(np.clip(v, -1.1, 1.1))) for k, v in zip(keys, x)})[0]
    rD = minimize(fD, np.zeros(6), method="Nelder-Mead", options=dict(xatol=0.03, fatol=1e-5, maxfev=160, initial_simplex=np.vstack([np.zeros(6), 0.5 * np.eye(6)])))
    pD = {k: float(np.exp(np.clip(v, -1.1, 1.1))) for k, v in zip(keys, rD.x)}; fDv, sD = objective(pD)
    print(f"stage D: {json.dumps({k: round(v,3) for k,v in pD.items()})} -> {fDv:.5f} {json.dumps(sD)}  ({time.time()-t0:.0f}s, {rD.nfev} evals)", flush=True)
    res["stageD"] = dict(params=pD, score=sD)
    # stage K: joint zero offsets on top of stage D
    OJ = [1, 3, 7, 9]
    def fK(x):
        off = np.zeros(NJ); off[OJ] = np.radians(np.clip(x, -4, 4)); return objective(dict(pD, q_offsets=off))[0]
    rK = minimize(fK, np.zeros(4), method="Nelder-Mead", options=dict(xatol=0.1, fatol=1e-5, maxfev=100, initial_simplex=np.vstack([np.zeros(4), 1.5 * np.eye(4)])))
    off = np.zeros(NJ); off[OJ] = np.radians(np.clip(rK.x, -4, 4)); fKv, sK = objective(dict(pD, q_offsets=off))
    print(f"stage K: offsets deg (Lhip,Lknee,Rhip,Rknee) {np.round(np.clip(rK.x,-4,4),2)} -> {fKv:.5f} {json.dumps(sK)}  ({time.time()-t0:.0f}s)", flush=True)
    res["stageK"] = dict(offsets_deg=np.clip(rK.x, -4, 4).tolist(), score=sK)
    # held-out
    res["test"] = dict(nominal=objective({}, test)[1], stageD=objective(pD, test)[1], stageDK=objective(dict(pD, q_offsets=off), test)[1])
    print("test:", json.dumps(res["test"]), flush=True)
    eng.close(); json.dump(res, open(os.path.join(OUT, "param_id2.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
