"""Scan ~/Desktop/h12/logs/*/*.npz real-robot recordings -> per-run float32 transition files.
Keeps every 100 Hz sample whose row t and t+1 are both finite and 8-12 ms apart.
Writes data/runs/<run>.npz and data/manifest.json (run, n, strategy, split)."""
import glob, os, json, sys, hashlib
import numpy as np
from common import *

def main(limit=None):
    files = sorted(glob.glob(os.path.join(H12, "logs/*/*.npz")))
    if limit: files = files[:limit]
    man = []
    for f in files:
        try:
            z = np.load(f, allow_pickle=True)
        except Exception as e:
            print("skip", f, e); continue
        need = ("t", "q", "dq", "cmd", "tau", "base_quat", "base_pos", "base_vel", "kp", "kd", "tau_ff", "gyro", "accel")
        if any(k not in z.files for k in need) or z["q"].shape[1] != NJ or len(z["t"]) < 200:
            continue
        t = z["t"]; dt = np.diff(t)
        ok = np.isfinite(z["q"]).all(1) & np.isfinite(z["dq"]).all(1) & np.isfinite(z["cmd"]).all(1) \
             & np.isfinite(z["tau"]).all(1) & np.isfinite(z["base_quat"]).all(1) & np.isfinite(z["kp"]).all(1) \
             & np.isfinite(z["kd"]).all(1) & np.isfinite(z["tau_ff"]).all(1) & np.isfinite(z["gyro"]).all(1) \
             & np.isfinite(z["accel"]).all(1)
        ok &= np.linalg.norm(z["base_quat"], axis=1) > 0.5
        good = ok[:-1] & ok[1:] & (dt > 0.008) & (dt < 0.012)
        idx = np.where(good)[0]
        if len(idx) < 100: continue
        name = os.path.basename(f)[:-4]
        bp_ok = np.isfinite(z["base_pos"]).all(1) & np.isfinite(z["base_vel"][:, :3]).all(1)
        tau_lim = z["tau_lim"] if "tau_lim" in z.files else np.full(NJ, np.inf)
        out = dict(idx=idx.astype(np.int32), t=t.astype(np.float64),
                   q=z["q"].astype(np.float32), dq=z["dq"].astype(np.float32), cmd=z["cmd"].astype(np.float32),
                   tau=z["tau"].astype(np.float32), kp=z["kp"].astype(np.float32), kd=z["kd"].astype(np.float32),
                   tau_ff=z["tau_ff"].astype(np.float32), base_quat=z["base_quat"].astype(np.float32),
                   base_pos=z["base_pos"].astype(np.float32), base_vel=z["base_vel"].astype(np.float32),
                   gyro=z["gyro"].astype(np.float32), accel=z["accel"].astype(np.float32),
                   bp_ok=bp_ok, tau_lim=np.asarray(tau_lim, np.float32))
        np.savez(os.path.join(RUNS, name + ".npz"), **out)
        strat = name.split("_")[0]
        h = int(hashlib.md5(name.encode()).hexdigest(), 16) % 10
        split = "test" if h == 0 else ("val" if h == 1 else "train")
        man.append(dict(run=name, n=int(len(idx)), n_bp=int((bp_ok[idx] & bp_ok[idx + 1]).sum()), strategy=strat, split=split, src=f))
        print(f"{name:50s} {len(idx):7d} bp {int((bp_ok[idx]).sum()):7d} {split}")
    json.dump(man, open(os.path.join(DATA, "manifest.json"), "w"), indent=1)
    print("runs", len(man), "transitions", sum(r["n"] for r in man), "with base pose", sum(r["n_bp"] for r in man))

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
