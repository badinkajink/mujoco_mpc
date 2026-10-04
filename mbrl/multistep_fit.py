"""Multi-step residual training through the linearised plant.
Along each nominal plant rollout x̂_t (t=0..T) with macro-step Jacobians A_t (66x66, tangent coords), a residual
r_θ(features(x̂_t + e_t)) is added to the joint velocities after each step:
    e_{t+1} = A_t e_t + S r_θ(...)         e_0 = 0,   S injects into the 27 joint-velocity rows (33+6 .. 33+33)
Loss = mean over t=1..T of || (x̂_t + e_t) - x_t^rec ||² on joint positions (weighted) and joint velocities,
i.e. the same closed-loop displacement error the rollout benchmark measures. Backprop runs through the recursion
(A_t constant). Warm start from the one-step MLP. Evaluated with residual_fit.rollout_eval on the true nonlinear plant."""
import os, json, glob, time
import numpy as np, torch, torch.nn as nn
from common import *
import residual_fit as RF

JAC = os.path.join(DATA, "jac")
dev = "cuda" if torch.cuda.is_available() else "cpu"

def load_jac(split):
    """Preallocated arrays (single copy in RAM; Jacobians ~2 GB f32 for ~5k segments). Batches go to the GPU per step."""
    files = []
    for f in sorted(glob.glob(os.path.join(JAC, "*.npz"))):
        with np.load(f, allow_pickle=True) as z:
            if str(z["split"]) == split: files.append((f, len(z["i0"]), z["A"].shape[1]))
    S = sum(n for _, n, _ in files); T = files[0][2]
    B = dict(xn=np.zeros((S, T + 1, 67), np.float32), A=np.zeros((S, T, 66, 66), np.float32),
             q=np.zeros((S, T + 1, NJ), np.float32), dq=np.zeros((S, T + 1, NJ), np.float32), cmd=np.zeros((S, T + 1, NJ), np.float32),
             kp=np.zeros((S, T + 1, NJ), np.float32), kd=np.zeros((S, T + 1, NJ), np.float32), ff=np.zeros((S, T + 1, NJ), np.float32),
             g=np.zeros((S, T + 1, 3), np.float32), tl=np.zeros((S, NJ), np.float32))
    k = 0
    for f, n, _ in files:
        with np.load(f, allow_pickle=True) as z:
            name = str(z["run"]); i0s = z["i0"]; B["xn"][k:k + n] = z["xn"]; B["A"][k:k + n] = z["A"]
        with np.load(os.path.join(RUNS, name + ".npz")) as rr: r = {kk: rr[kk] for kk in ("q", "dq", "cmd", "kp", "kd", "tau_ff", "base_quat", "tau_lim")}
        tl = r["tau_lim"].astype(np.float32); tl[~np.isfinite(tl)] = 1e9
        for s_, i0 in enumerate(i0s):
            ii = int(i0) + np.arange(T + 1); j = k + s_
            B["q"][j] = r["q"][ii]; B["dq"][j] = r["dq"][ii]; B["cmd"][j] = r["cmd"][ii]; B["kp"][j] = r["kp"][ii]; B["kd"][j] = r["kd"][ii]
            B["ff"][j] = r["tau_ff"][ii]; B["g"][j] = RF.gvec(r["base_quat"][ii]); B["tl"][j] = tl
        k += n
    return B

def to_torch(B): return {k: torch.from_numpy(v) for k, v in B.items()}

class MS(nn.Module):
    def __init__(self, base: RF.Residual):
        super().__init__()
        self.net = base.net; self.register_buffer("mu", torch.tensor(base.mu)); self.register_buffer("sd", torch.tensor(base.sd))
        self.register_buffer("ymu", torch.tensor(base.ymu)); self.register_buffer("ysd", torch.tensor(base.ysd))
        self.register_buffer("clip", torch.tensor(base.clip, dtype=torch.float32))
    def residual(self, x):
        y = self.net((x - self.mu) / self.sd) * self.ysd + self.ymu
        return torch.max(torch.min(y, self.clip), -self.clip)

def unroll(model, B, T, wq=1.0, wv=0.01):
    """B: batch dict on device. Returns loss and per-horizon joint displacement rmse (for logging)."""
    n = B["xn"].shape[0]; e = torch.zeros(n, 66, device=dev)
    q0_sim = B["xn"][:, 0, 7:34]; q0_rec = B["q"][:, 0]
    loss = 0.0; disp = []
    for t in range(T):
        xn = B["xn"][:, t]
        q = xn[:, 7:34] + e[:, 6:33]; dq = xn[:, 40:67] + e[:, 39:66]; w = xn[:, 37:40] + e[:, 36:39]
        tau0 = torch.clamp(B["kp"][:, t] * (B["cmd"][:, t] - q) - B["kd"][:, t] * dq + B["ff"][:, t], -B["tl"], B["tl"])
        x = torch.cat([q, dq, tau0, B["cmd"][:, t] - q, B["kp"][:, t] / 100.0, B["kd"][:, t] / 10.0, B["g"][:, t], w], 1)
        r = model.residual(x)
        e = torch.bmm(B["A"][:, t], e.unsqueeze(2)).squeeze(2)
        e = torch.cat([e[:, :39], e[:, 39:66] + r], 1)
        q1 = B["xn"][:, t + 1, 7:34] + e[:, 6:33]; dq1 = B["xn"][:, t + 1, 40:67] + e[:, 39:66]
        dq_err = dq1 - B["dq"][:, t + 1]; disp_err = (q1 - q0_sim) - (B["q"][:, t + 1] - q0_rec)
        loss = loss + wq * (disp_err ** 2).mean() + wv * (dq_err ** 2).mean()
        disp.append(disp_err.detach())
    disp = torch.stack(disp, 1)   # n,T,27
    return loss / T, disp

def main(epochs=int(os.environ.get("MBRL_MS_EPOCHS", "30")), bs=256, lr=3e-4):
    log_f = open(os.path.join(OUT, "multistep_fit.log"), "a")
    def log(s): print(s, flush=True); log_f.write(s + "\n"); log_f.flush()
    t0 = time.time(); Btr = to_torch(load_jac("train")); Bva = {k: v.to(dev) for k, v in to_torch(load_jac("val")).items()}
    n = Btr["xn"].shape[0]; T = Btr["A"].shape[1]; log(f"segments train {n} val {Bva['xn'].shape[0]}  load {time.time()-t0:.0f}s  RAM for A {Btr['A'].numel()*4/1e9:.1f} GB")
    # warm start from the one-step MLP
    base = RF.Residual("mlp"); base.net = RF.MLP(168, 27); base.net.load_state_dict(torch.load(os.path.join(OUT, "residual_mlp.pt")))
    nz = np.load(os.path.join(OUT, "residual_norm.npz")); base.mu, base.sd, base.ymu, base.ysd = nz["mu"], nz["sd"], nz["ymu"], nz["ysd"]
    base.clip = np.percentile(np.abs(nz["ysd"]) * 4, 50) * np.ones(27, np.float32) if "clip" not in nz.files else nz["clip"]
    model = MS(base).to(dev); opt = torch.optim.AdamW(model.net.parameters(), lr, weight_decay=1e-5)
    def evaluate(B):
        model.eval()
        with torch.no_grad(): loss, disp = unroll(model, B, T)
        model.train()
        return float(loss), {h: float(torch.sqrt((disp[:, h - 1, :12] ** 2).mean())) for h in (10, 25)}, {h: float(torch.sqrt((disp[:, h - 1, 13:] ** 2).mean())) for h in (10, 25)}
    with torch.no_grad():   # plant alone and one-step model, for reference
        class Zero(nn.Module):
            def residual(self, x): return torch.zeros(x.shape[0], 27, device=dev)
        model_z = Zero(); lz, dz = unroll(model_z, Bva, T); log(f"val plant-only: legs 100/250ms {torch.sqrt((dz[:,9,:12]**2).mean()):.4f}/{torch.sqrt((dz[:,24,:12]**2).mean()):.4f} arms {torch.sqrt((dz[:,9,13:]**2).mean()):.4f}/{torch.sqrt((dz[:,24,13:]**2).mean()):.4f}")
    l, lg, ar = evaluate(Bva); log(f"val one-step MLP warm start: loss {l:.5f} legs {lg} arms {ar}")
    best = 1e9
    for ep in range(epochs):
        perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; B = {k: v[idx].to(dev) for k, v in Btr.items()}
            loss, _ = unroll(model, B, T); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.net.parameters(), 1.0); opt.step(); tot += float(loss) * len(idx)
        l, lg, ar = evaluate(Bva); log(f"ep {ep} train {tot/n:.5f} val {l:.5f} legs {lg} arms {ar}")
        if l < best: best = l; torch.save(model.net.state_dict(), os.path.join(OUT, "residual_mlp_multistep.pt"))
    log(f"done {time.time()-t0:.0f}s best val {best:.5f}")

if __name__ == "__main__":
    main()
