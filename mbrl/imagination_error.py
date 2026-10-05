"""Planner imagination error in closed loop: for each iCEM dump (nominal trajectory = the planner's predicted states
under its chosen plan, horizon x (nq+nv+na) at the agent step), compare the predicted joint positions over the next
100 / 250 / 500 ms with what the twin actually did (recorder npz, 100 Hz, same plant clock). Error = displacement
RMSE (rad) per joint group, like the offline rollout benchmark, so residual-on vs residual-off is directly comparable.
  python imagination_error.py <run_dir> [<run_dir> ...]      (run_dir holds dump/tick_*.bin and <rec>/<rec>.npz)"""
import os, sys, glob, struct, json
import numpy as np
from common import *

def read_dump(path):
    with open(path, "rb") as f:
        hdr = struct.unpack("<6i", f.read(24)); ntraj, horizon, dim, nq, nv, nelite = hdr
        t0 = struct.unpack("<d", f.read(8))[0]
        f.seek(4 * ntraj, 1)
        f.seek(ntraj * (8 + 8 * horizon * dim), 1)
        ret = struct.unpack("<d", f.read(8))[0]
        nominal = np.frombuffer(f.read(8 * horizon * dim), dtype="<f8").reshape(horizon, dim)
    return t0, nominal, nq, nv

def score_run(run_dir, dt_agent=0.01, hors=(10, 25, 50)):
    recs = glob.glob(os.path.join(run_dir, "*", "*.npz"))
    if recs:                                   # recorder, 100 Hz
        z = np.load(recs[0]); t = z["t"]; q = z["q"]
    elif os.path.exists(os.path.join(run_dir, "run.npz")):   # twin trace (~23 Hz): resample to 100 Hz
        z = np.load(os.path.join(run_dir, "run.npz"), allow_pickle=True); tt = z["t"]; qq = z["q"]
        t = np.arange(tt[0], tt[-1], 0.01); q = np.stack([np.interp(t, tt, qq[:, j]) for j in range(NJ)], 1)
    else: return None
    E = {h: [] for h in hors}; n_dump = 0; t_off = None
    for f in sorted(glob.glob(os.path.join(run_dir, "dump", "tick_*.bin"))):
        t0, nom, nq, nv = read_dump(f); qn = nom[:, 7:7 + NJ]
        # clock alignment: the dump's t0 is planner time; find the recorder sample closest to t0 (one-off offset
        # estimated on the first dump by matching q)
        if t_off is None:
            cand = np.arange(len(t)); err = np.abs(q[cand] - qn[0]).mean(1); i = int(np.argmin(err)); t_off = t[i] - t0
        i0 = int(np.searchsorted(t, t0 + t_off))
        if i0 + max(hors) + 1 >= len(t) or i0 < 1: continue
        if np.abs(q[i0] - qn[0]).mean() > 0.05: continue           # misaligned / robot already down: skip
        n_dump += 1
        for h in hors:
            if h < len(qn): E[h].append((qn[h] - qn[0]) - (q[i0 + h] - q[i0]))
    out = {"n_dumps": n_dump}
    for h in hors:
        e = np.array(E[h]) if E[h] else np.zeros((0, NJ))
        out[f"{h*10}ms"] = {g: float(np.sqrt(np.mean(e[:, ids] ** 2))) if len(e) else None for g, ids in GROUPS.items()}
    return out

if __name__ == "__main__":
    res = {}
    for d in sys.argv[1:]:
        r = score_run(d); res[os.path.basename(d.rstrip("/"))] = r
        print(os.path.basename(d.rstrip("/")), json.dumps(r))
    json.dump(res, open(os.path.join(OUT, "imagination_error.json"), "w"), indent=1)
