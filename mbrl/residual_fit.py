"""Learned residual on top of the planner plant (PD-command mode).
   target  y = dq_rec(t+1) - dq_sim_pd(t+1)              (27)
   inputs  x = [q, dq, tau_cmd0, cmd-q, kp, kd, g_pelvis(3), w_pelvis(3)]   (planner-known quantities only)
Models: ridge (linear) and MLP (torch). Split by run (manifest). Metrics: one-step RMSE per group on test runs,
and multi-step closed-loop rollout error (plant +/- residual) on held-out 0.5 s segments."""
import os, json, glob, time, sys
import numpy as np, mujoco, torch, torch.nn as nn
from common import *
import simgap

SIMG = os.path.join(DATA, "simgap")
MAXTRAIN = int(os.environ.get("MBRL_MAXTRAIN", "4000000"))

def gvec(bq):
    """gravity direction in the IMU frame (z-down unit) from the torso quat (w,x,y,z), vectorised."""
    w, x, y, z = bq.T
    # third row of R^T applied to [0,0,-1] -> -R[:,2]... use R^T g_world with g_world=[0,0,-1]
    R20 = 2 * (x * z - w * y); R21 = 2 * (y * z + w * x); R22 = 1 - 2 * (x * x + y * y)
    return -np.stack([R20, R21, R22], 1)

def features(r, s):
    idx = s["idx"]; ok = s["ok"]
    q = r["q"][idx]; dq = r["dq"][idx]; cmd = r["cmd"][idx]; kp = r["kp"][idx]; kd = r["kd"][idx]
    X = np.concatenate([q, dq, s["tau_pd0"], cmd - q, kp / 100.0, kd / 10.0, gvec(r["base_quat"][idx]), s["w_rec0"]], 1).astype(np.float32)
    y = (r["dq"][idx + 1] - s["dq_pd"]).astype(np.float32)
    ez = (r["dq"][idx + 1] - dq).astype(np.float32)
    return X[ok], y[ok], ez[ok], idx[ok]

def load_split(split, maxn=None):
    man = [m for m in json.load(open(os.path.join(DATA, "manifest.json"))) if m["split"] == split]
    Xs, ys, ezs, runs = [], [], [], []
    for m in man:
        f = os.path.join(SIMG, m["run"] + ".npz")
        if not os.path.exists(f): continue
        with np.load(os.path.join(RUNS, m["run"] + ".npz")) as z: r = {k: z[k] for k in ("q", "dq", "cmd", "kp", "kd", "base_quat")}
        s = np.load(f); X, y, ez, idx = features(r, s)
        Xs.append(X); ys.append(y); ezs.append(ez); runs.append(np.full(len(X), m["run"]))
    X = np.concatenate(Xs); y = np.concatenate(ys); ez = np.concatenate(ezs); runs = np.concatenate(runs)
    if maxn and len(X) > maxn:
        sel = np.random.RandomState(0).choice(len(X), maxn, replace=False); X, y, ez, runs = X[sel], y[sel], ez[sel], runs[sel]
    return X, y, ez, runs

def grp_rmse(e):
    pj = np.sqrt(np.mean(e.astype(np.float64) ** 2, 0))
    return {g: float(np.sqrt(np.mean(pj[ids] ** 2))) for g, ids in GROUPS.items()} | {"all": float(np.sqrt(np.mean(pj ** 2)))}

class MLP(nn.Module):
    def __init__(self, nin, nout, h=512, nl=3):
        super().__init__()
        L = []; d = nin
        for _ in range(nl): L += [nn.Linear(d, h), nn.SiLU()]; d = h
        L += [nn.Linear(d, nout)]; self.net = nn.Sequential(*L)
    def forward(self, x): return self.net(x)

class Residual:
    """standardised ridge or MLP wrapper with numpy predict."""
    def __init__(self, kind): self.kind = kind
    def fit(self, X, y, Xv=None, yv=None, epochs=12, log=print):
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6; self.ymu = y.mean(0); self.ysd = y.std(0) + 1e-6
        Xn = (X - self.mu) / self.sd; yn = (y - self.ymu) / self.ysd
        if self.kind == "ridge":
            A = np.c_[Xn, np.ones(len(Xn))].astype(np.float64); lam = 1e-2 * len(A)
            self.W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ yn.astype(np.float64)).astype(np.float32)
            return
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = MLP(X.shape[1], y.shape[1]).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), 1e-3, weight_decay=1e-5)
        Xt = torch.tensor(Xn, device=dev); yt = torch.tensor(yn, device=dev); n = len(Xt); bs = 4096
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * ((n + bs - 1) // bs))
        for ep in range(epochs):
            perm = torch.randperm(n, device=dev); tot = 0.0
            for i in range(0, n, bs):
                b = perm[i:i + bs]; loss = nn.functional.smooth_l1_loss(self.net(Xt[b]), yt[b], beta=0.5)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(b)
            msg = f"  ep {ep} train {tot / n:.4f}"
            if Xv is not None: msg += f"  val rmse(all) {grp_rmse(yv - self.predict(Xv))['all']:.4f}"
            log(msg)
        self.net.eval()
    def predict(self, X):
        Xn = ((X - self.mu) / self.sd).astype(np.float32)
        if self.kind == "ridge": yn = np.c_[Xn, np.ones(len(Xn))] @ self.W
        else:
            with torch.no_grad():
                dev = next(self.net.parameters()).device; outs = []
                for i in range(0, len(Xn), 65536): outs.append(self.net(torch.tensor(Xn[i:i + 65536], device=dev)).cpu().numpy())
                yn = np.concatenate(outs)
        return yn * self.ysd + self.ymu

def rollout_eval(models, nseg=300, T=50, seed=0, log=print):
    """closed-loop T-step (10 ms) rollouts on test runs from recorded initial states, PD torque from recorded commands.
    Residual correction: after each 10 ms plant step, dq += f(x_t). Error = joint position RMSE vs recorded, by horizon."""
    simgap._init(); m, d = simgap._m, simgap._d
    man = [mm for mm in json.load(open(os.path.join(DATA, "manifest.json"))) if mm["split"] == "test"]
    rng = np.random.RandomState(seed); segs = []
    for mm in man:
        with np.load(os.path.join(RUNS, mm["run"] + ".npz")) as z: r = {k: z[k] for k in z.files}
        good = r["idx"]; bp = r["bp_ok"]
        cand = good[(good + T + 1 < len(r["t"])) & bp[good]]
        cand = cand[np.all(np.diff(r["t"])[cand[:, None] + np.arange(T)] < 0.012, 1)] if len(cand) else cand
        for i in rng.choice(cand, min(len(cand), max(1, nseg // len(man)) + 1), replace=False): segs.append((r, int(i)))
    segs = segs[:nseg]; log(f"rollout segments {len(segs)} x {T} steps")
    names = ["plant"] + list(models); hor = [10, 25, 50]
    err = {nm: np.zeros((len(segs), T, NJ), np.float32) for nm in names}
    nsub = int(round(REC_DT / m.opt.timestep))
    for si, (r, i0) in enumerate(segs):
        tl = r["tau_lim"].astype(float); tl[~np.isfinite(tl)] = 1e9
        for nm in names:
            mujoco.mj_resetData(m, d); simgap.set_state(m, d, r, i0)
            for k in range(T):
                i = i0 + k
                if nm != "plant":
                    bq = np.zeros(4)
                    mujoco.mj_kinematics(m, d); mujoco.mju_mat2Quat(bq, d.site("imu").xmat)
                    q = d.qpos[JQ].copy(); dq = d.qvel[JV].copy()
                    tau0 = pd_torque(r["kp"][i], r["kd"][i], r["cmd"][i], q, dq, r["tau_ff"][i], tl)
                    x = np.concatenate([q, dq, tau0, r["cmd"][i] - q, r["kp"][i] / 100.0, r["kd"][i] / 10.0, gvec(bq[None])[0], d.qvel[3:6]]).astype(np.float32)[None]
                for s_ in range(nsub):
                    d.qfrc_applied[JV] = pd_torque(r["kp"][i], r["kd"][i], r["cmd"][i], d.qpos[JQ], d.qvel[JV], r["tau_ff"][i], tl); mujoco.mj_step(m, d)
                if nm != "plant": d.qvel[JV] += models[nm].predict(x)[0]
                err[nm][si, k] = d.qpos[JQ] - r["q"][i + 1]
    res = {}
    for nm in names:
        res[nm] = {f"{h*10}ms": {g: float(np.sqrt(np.mean(err[nm][:, h - 1][:, ids] ** 2))) for g, ids in GROUPS.items()} for h in hor}
    np.savez(os.path.join(OUT, "rollout_err.npz"), **{f"err_{nm}": err[nm] for nm in names})
    return res

def main():
    log_f = open(os.path.join(OUT, "residual_fit.log"), "a")
    def log(s): print(s, flush=True); log_f.write(s + "\n"); log_f.flush()
    t0 = time.time()
    Xtr, ytr, eztr, _ = load_split("train", MAXTRAIN); Xv, yv, _, _ = load_split("val", 500000); Xte, yte, ezte, runs_te = load_split("test")
    log(f"train {Xtr.shape} val {Xv.shape} test {Xte.shape}  load {time.time()-t0:.0f}s")
    res = {"n_train": int(len(Xtr)), "n_test": int(len(Xte)), "one_step_test": {"hold_dq": grp_rmse(ezte), "plant_pd": grp_rmse(yte)}}
    models = {}
    for kind in ("ridge", "mlp"):
        t1 = time.time(); mdl = Residual(kind); mdl.fit(Xtr, ytr, Xv, yv, log=log); models[kind] = mdl
        res["one_step_test"][f"plant_pd+{kind}"] = grp_rmse(yte - mdl.predict(Xte)); log(f"{kind} fit {time.time()-t1:.0f}s  {res['one_step_test'][f'plant_pd+{kind}']}")
    # per-strategy one-step on test
    per = {}
    for st in sorted(set(s.split("_")[0] for s in runs_te)):
        mk = np.array([s.split("_")[0] == st for s in runs_te])
        per[st] = {"n": int(mk.sum()), "hold": grp_rmse(ezte[mk])["all"], "plant": grp_rmse(yte[mk])["all"], "plant+mlp": grp_rmse((yte - models["mlp"].predict(Xte))[mk])["all"]}
    res["per_strategy_test"] = per
    torch.save(models["mlp"].net.state_dict(), os.path.join(OUT, "residual_mlp.pt"))
    np.savez(os.path.join(OUT, "residual_norm.npz"), mu=models["mlp"].mu, sd=models["mlp"].sd, ymu=models["mlp"].ymu, ysd=models["mlp"].ysd, W_ridge=models["ridge"].W)
    json.dump(res, open(os.path.join(OUT, "residual_summary.json"), "w"), indent=1)
    log("one-step done; rollouts...")
    # rollout: move mlp to cpu for per-step inference
    models["mlp"].net = models["mlp"].net.cpu()
    res["rollout_test"] = rollout_eval({"ridge": models["ridge"], "mlp": models["mlp"]}, log=log)
    json.dump(res, open(os.path.join(OUT, "residual_summary.json"), "w"), indent=1)
    log(json.dumps(res["rollout_test"], indent=1)); log(f"total {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
