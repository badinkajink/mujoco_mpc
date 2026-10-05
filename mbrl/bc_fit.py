"""Behaviour cloning of the iCEM planner (planner distillation, Stage 3a). GPU MLP on out_bc/bc_data.npz.
env: MBRL_BC_H (256) MBRL_BC_NL (3) MBRL_BC_EPOCHS (40) MBRL_BC_TAG MBRL_BC_PHASES (e.g. "0" = stand only) MBRL_BC_WTAU (0.1)
saves out_bc/bc_<tag>.npz (numpy weights + norms) for the python evaluator."""
import os, numpy as np, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out_bc")
H = int(os.environ.get("MBRL_BC_H", 256)); NL = int(os.environ.get("MBRL_BC_NL", 3)); EP = int(os.environ.get("MBRL_BC_EPOCHS", 40))
TAG = os.environ.get("MBRL_BC_TAG", "v1"); WTAU = float(os.environ.get("MBRL_BC_WTAU", 0.1))
z = np.load(os.path.join(OUT, "bc_data.npz")); X, Y, val, ph = z["X"], z["Y"], z["val"], z["phase"]
if os.environ.get("MBRL_BC_PHASES"):
    keep = np.isin(ph, [int(p) for p in os.environ["MBRL_BC_PHASES"].split(",")]); X, Y, val = X[keep], Y[keep], val[keep]
MASK = np.ones(X.shape[1], np.float32)
for blk in [b for b in os.environ.get("MBRL_BC_MASK", "").split(",") if b]:
    MASK[{"dq": slice(27, 54), "gyro": slice(57, 60), "g": slice(54, 57), "t": slice(73, 74)}[blk]] = 0
X = X * MASK
NOISE = np.zeros(X.shape[1], np.float32); nz = float(os.environ.get("MBRL_BC_NOISE", 0))
NOISE[:27] = np.radians(0.5) * nz; NOISE[27:54] = 0.1 * nz; NOISE[54:57] = 0.02 * nz; NOISE[57:60] = 0.05 * nz
NOISE *= MASK
mu, sd = X[~val].mean(0), X[~val].std(0) + 1e-6; ymu, ysd = Y[~val].mean(0), Y[~val].std(0) + 1e-6
w = np.ones(54, np.float32); w[27:] = WTAU                      # tau_ff is a weaker target
dev = "cuda" if torch.cuda.is_available() else "cpu"
Xt = torch.tensor((X[~val] - mu) / sd, device=dev); Yt = torch.tensor((Y[~val] - ymu) / ysd, device=dev)
Xv = torch.tensor((X[val] - mu) / sd, device=dev); Yv = torch.tensor((Y[val] - ymu) / ysd, device=dev); wt = torch.tensor(w, device=dev); nzt = torch.tensor(NOISE / sd, device=dev)
layers = []; nin = X.shape[1]
for _ in range(NL): layers += [nn.Linear(nin, H), nn.ELU()]; nin = H
net = nn.Sequential(*layers, nn.Linear(nin, 54)).to(dev)
opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=float(os.environ.get("MBRL_BC_WD", 1e-5))); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP)
B = 4096; n = len(Xt)
for ep in range(EP):
    net.train(); perm = torch.randperm(n, device=dev); tl = 0
    for i in range(0, n, B):
        b = perm[i:i + B]; xb = Xt[b] + torch.randn_like(Xt[b]) * nzt; loss = (((net(xb) - Yt[b]) ** 2) * wt).mean(); opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(b)
    sched.step(); net.eval()
    with torch.no_grad():
        pv = net(Xv); e = (pv - Yv) * torch.tensor(ysd, device=dev)
        cmd_deg = torch.sqrt((e[:, :27] ** 2).mean()).item() * 57.3; tau = torch.sqrt((e[:, 27:] ** 2).mean()).item()
    if ep % 5 == 0 or ep == EP - 1: print(f"ep {ep:3d} train {tl / n:.4f}  val cmd RMSE {cmd_deg:.2f} deg  tau_ff RMSE {tau:.2f} Nm", flush=True)
# baselines: predict mean, and "hold" (cmd=q -> delta 0)
e0 = Y[val][:, :27]; print("val cmd RMSE if predicting 0 (hold):", np.degrees(np.sqrt((e0 ** 2).mean())).round(2), " mean-predictor:", np.degrees(np.sqrt(((e0 - ymu[:27]) ** 2).mean())).round(2))
W = {f"W{i}": l.weight.detach().cpu().numpy() for i, l in enumerate(m for m in net if isinstance(m, nn.Linear))}
W.update({f"b{i}": l.bias.detach().cpu().numpy() for i, l in enumerate(m for m in net if isinstance(m, nn.Linear))})
np.savez(os.path.join(OUT, f"bc_{TAG}.npz"), mask=MASK, mu=mu, sd=sd, ymu=ymu, ysd=ysd, nl=NL + 1, **W); print("saved", f"bc_{TAG}.npz")
