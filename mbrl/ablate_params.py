"""Quick parametric ablation of the plant on a few runs: joint damping / armature / frictionloss
-> one-step dq RMSE per group (tau mode and pd mode), vs hold-velocity."""
import sys, json, numpy as np, mujoco
from common import *
import simgap

def evaluate(m, names, stride=20, mode="tau"):
    d = mujoco.MjData(m); simgap._m, simgap._d = m, d
    errs = {g: [] for g in GROUPS}; hold = {g: [] for g in GROUPS}
    nsub = int(round(REC_DT / m.opt.timestep))
    for name in names:
        with np.load(RUNS + f"/{name}.npz") as z: r = {k: z[k] for k in z.files}
        idx = r["idx"][::stride]; idx = idx[r["bp_ok"][idx]]
        tl = r["tau_lim"].astype(float); tl[~np.isfinite(tl)] = 1e9
        for i in idx:
            mujoco.mj_resetData(m, d); simgap.set_state(m, d, r, i)
            q1, dq1, w1, nc = simgap.step10(m, d, mode, r, i, nsub, tl)
            e = dq1 - r["dq"][i + 1]; h = r["dq"][i] - r["dq"][i + 1]
            for g, ids in GROUPS.items(): errs[g].append(e[ids]); hold[g].append(h[ids])
    return {g: (float(np.sqrt(np.mean(np.square(errs[g])))), float(np.sqrt(np.mean(np.square(hold[g]))))) for g in GROUPS}

if __name__ == "__main__":
    names = ["11_nominal_1_20260912_202906", "25_icemplan_3_20260825_234802", "29_13_20260828_044749", "v3stand_20260830_060832"]
    base = load_model()
    configs = {
        "plant as-is (damp10 arm0.1 fl0.2)": dict(),
        "damp1": dict(damping=1.0), "damp0.1": dict(damping=0.1), "damp0": dict(damping=0.0),
        "damp1 arm0.01": dict(damping=1.0, armature=0.01), "damp0.1 arm0.01": dict(damping=0.1, armature=0.01),
        "damp0.1 arm0.01 fl0": dict(damping=0.1, armature=0.01, frictionloss=0.0),
        "damp0 arm0.01 fl0": dict(damping=0.0, armature=0.01, frictionloss=0.0),
    }
    res = {}
    for lab, cfg in configs.items():
        m = load_model()
        if "damping" in cfg: m.dof_damping[JV] = cfg["damping"]
        if "armature" in cfg: m.dof_armature[JV] = cfg["armature"]
        if "frictionloss" in cfg: m.dof_frictionloss[JV] = cfg["frictionloss"]
        for mode in ("tau", "pd"):
            r = evaluate(m, names, mode=mode); res[f"{lab} [{mode}]"] = r
            print(f"{lab:36s} [{mode}] " + "  ".join(f"{g}: {v[0]:.3f} (hold {v[1]:.3f})" for g, v in r.items()), flush=True)
    json.dump(res, open(OUT + "/ablate_params.json", "w"), indent=1)
