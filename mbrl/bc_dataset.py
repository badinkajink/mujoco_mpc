"""Build the behaviour-cloning dataset for planner distillation from twin bench runs that kept their 100 Hz
recorder npz (lowstate q/dq/imu + the node's lowcmd cmd/tau_ff/kp/kd). Phase + keyframe index come from node.log
([BIO t=.. phase=..] and [STRAT n:.. | kf k/9]); node t = recorder t - t(first kp>1). Samples after (fall - 2 s) dropped.
usage: python bc_dataset.py 'glob1' 'glob2' ...   (run DIRS)   -> out_bc/bc_data.npz"""
import sys, os, re, glob, hashlib, numpy as np
from bc_common import obs_vec, phase_id, NOBS
H12 = os.path.expanduser("~/Desktop/h12")
dirs = sorted(set(d for g in sys.argv[1:] for d in glob.glob(os.path.join(H12, g))))
X, Y, RUN, PH, T = [], [], [], [], []
for ri, d in enumerate(dirs):
    npz = glob.glob(os.path.join(d, "*", "*.npz"))
    if not npz or not os.path.exists(os.path.join(d, "node.log")): continue
    z = np.load(npz[0], allow_pickle=True); z = {k: z[k] for k in ("t", "q", "dq", "cmd", "tau_ff", "kp", "base_quat", "gyro")}
    on = z["kp"][:, 0] > 1
    if on.sum() < 500: continue
    t_eng = z["t"][np.argmax(on)]; tn = z["t"] - t_eng          # node time
    log = open(os.path.join(d, "node.log"), errors="ignore").read()
    ev = []                                                    # (t, phase, kf) timeline
    kf = 1
    for m in re.finditer(r"kf (\d)/9\]|\[BIO +t=([0-9.]+) phase=([A-Za-z_0-9]+)", log):
        if m.group(1): kf = int(m.group(1))
        else: ev.append((float(m.group(2)), phase_id(m.group(3)), kf))
    if not ev: continue
    ev_t = np.array([e[0] for e in ev]); ev_p = np.array([e[1] for e in ev]); ev_k = np.array([e[2] for e in ev])
    # phase start times
    starts = {}; last = None
    for e in ev:
        if e[1] != last: starts[len(starts)] = e; last = e[1]
    tw = open(os.path.join(d, "twin.log"), errors="ignore").read() if os.path.exists(os.path.join(d, "twin.log")) else ""
    mf = re.search(r"FALLEN for 8s \(since t=([0-9.]+)s", tw); me = re.search(r"engaged at sim t=([0-9.]+)s", tw)
    t_end = (float(mf.group(1)) - float(me.group(1)) - 2.0) if (mf and me) else 1e9   # node time of fall - 2 s
    idx = np.where(on & (tn >= 0.3) & (tn < t_end) & np.isfinite(z["cmd"]).all(1) & np.isfinite(z["tau_ff"]).all(1))[0]
    j = np.searchsorted(ev_t, tn[idx], side="right") - 1; j = np.clip(j, 0, len(ev) - 1)
    # phase start time for each sample: walk the phase-change list
    chg_t = [ev[0][0]]; chg_p = [ev[0][1]]
    for e in ev[1:]:
        if e[1] != chg_p[-1]: chg_t.append(e[0]); chg_p.append(e[1])
    chg_t = np.array(chg_t); jj = np.clip(np.searchsorted(chg_t, tn[idx], side="right") - 1, 0, len(chg_t) - 1)
    for n, i in enumerate(idx):
        X.append(obs_vec(z["q"][i], z["dq"][i], z["base_quat"][i], z["gyro"][i], ev_p[j[n]], ev_k[j[n]], tn[i] - chg_t[jj[n]],
                         z["cmd"][i - 1] if i > 0 else None))
        Y.append(np.concatenate([z["cmd"][i] - z["q"][i], z["tau_ff"][i]]).astype(np.float32)); RUN.append(ri); PH.append(ev_p[j[n]]); T.append(tn[i])
    print(f"{os.path.basename(d):45s} n={len(idx):6d} phases={np.bincount(ev_p[j], minlength=4)} t_end={t_end if t_end<1e8 else 'none'}")
X = np.array(X); Y = np.array(Y); RUN = np.array(RUN); PH = np.array(PH); T = np.array(T)
split = np.array([int(hashlib.md5(os.path.basename(dirs[r]).encode()).hexdigest(), 16) % 100 < 15 for r in RUN])  # 15 % of runs = val
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_bc"); os.makedirs(OUT, exist_ok=True)
np.savez_compressed(os.path.join(OUT, "bc_data.npz"), X=X, Y=Y, run=RUN, phase=PH, t=T, val=split, dirs=np.array(dirs))
print("total", X.shape, Y.shape, "val frac", split.mean().round(3), "runs", len(set(RUN)), "phase counts", np.bincount(PH, minlength=4))
