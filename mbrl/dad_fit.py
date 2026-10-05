"""Data-as-Demonstrator (Venkatraman et al. 2015) training of the velocity residual: iterate
  1. roll the plant + current residual out on training segments (nonlinear, parallel),
  2. at every visited state record the one-step target  y = dq_rec(t+1) - dq_plant(t+1 | visited state),
  3. aggregate with the original one-step data and fine-tune the MLP,
so the residual is trained on the state distribution its own closed-loop rollouts produce. Validation = rollout
displacement error on held-out val segments; the best iterate is kept. Writes out/residual_mlp_dad.pt + dad_summary.json."""
import os, json, time
import numpy as np, torch
from common import *
import residual_fit as RF, rollout_engine as RE

T = int(os.environ.get("MBRL_T", "25")); ITERS = int(os.environ.get("MBRL_DAD_ITERS", "5"))
NSEG = int(os.environ.get("MBRL_DAD_SEGS", "3000")); NW = int(os.environ.get("MBRL_WORKERS", "8"))
PARAMS = dict(nsub=int(os.environ.get("MBRL_NSUB", "5")), dt=float(os.environ.get("MBRL_DT", "0.002")), inject=os.environ.get("MBRL_INJECT", "post"), cmd_noise=float(os.environ.get("MBRL_CMD_NOISE", "0")))
TAG = os.environ.get("MBRL_TAG", "dad")

def to_numpy_mlp(res: RF.Residual):
    return RE.NumpyMLP({k: v.detach().cpu().numpy() for k, v in res.net.state_dict().items()}, res.mu, res.sd, res.ymu, res.ysd, res.clip)

def main():
    seed = int(os.environ.get("MBRL_SEED", "0")); torch.manual_seed(seed); np.random.seed(seed)
    log_f = open(os.path.join(OUT, f"{TAG}_fit.log"), "a")
    def log(s): print(s, flush=True); log_f.write(s + "\n"); log_f.flush()
    t0 = time.time()
    # base one-step data (consistent samples) and warm-started model
    Xtr, ytr, _, _ = RF.load_split("train", int(os.environ.get("MBRL_DAD_BASE", "600000"))); Xv, yv, _, _ = RF.load_split("val", 100000)
    base = RF.Residual("mlp"); base.fit(Xtr, ytr, Xv, yv, epochs=8, log=lambda s: None)
    log(f"one-step base: val rmse {RF.grp_rmse(yv - base.predict(Xv))}  ({time.time()-t0:.0f}s)")
    segs_tr = RE.compact_segments("train", T, NSEG, seed=seed); segs_va = RE.compact_segments("val", T, 500, seed=1)
    log(f"segments train {len(segs_tr)} val {len(segs_va)}  plant params {PARAMS}")
    eng = RE.Engine(NW)
    plant_val = RE.score(segs_va, eng.run(segs_va, T, PARAMS)); log(f"val plant-only: {json.dumps(plant_val)}")
    best = (1e9, None); hist = []
    Xagg, yagg = [Xtr], [ytr]
    for it in range(ITERS + 1):
        mlp = to_numpy_mlp(base)
        sv = RE.score(segs_va, eng.run(segs_va, T, PARAMS, mlp)); crit = sv["250ms"]["legs"] + sv["250ms"]["arms"] + 0.01 * sv["diverged"]
        log(f"iter {it}: val rollouts {json.dumps(sv)}")
        hist.append(sv)
        if crit < best[0]: best = (crit, {k: v.detach().cpu().clone() for k, v in base.net.state_dict().items()}); torch.save(base.net.state_dict(), os.path.join(OUT, f"residual_mlp_{TAG}.pt")); np.savez(os.path.join(OUT, f"residual_norm_{TAG}.npz"), mu=base.mu, sd=base.sd, ymu=base.ymu, ysd=base.ysd, clip=base.clip)
        if it == ITERS: break
        outs = eng.run(segs_tr, T, PARAMS, mlp, want_X=True); Xn, yn = [], []
        for r, o in zip(segs_tr, outs):
            if o is None or "diverged_at" in o: continue
            Xn.append(o["X"]); yn.append(r["dq"][1:T + 1] - o["dq_plant"] + (mlp(o["X"]) if PARAMS["inject"] == "pre" else 0.0))
        Xn = np.concatenate(Xn); yn = np.concatenate(yn); Xagg.append(Xn); yagg.append(yn)
        # fine-tune on the aggregated on-rollout data mixed with an equal-size slice of the base one-step data
        Xo = np.concatenate(Xagg[1:]); yo = np.concatenate(yagg[1:]); nb = min(len(Xtr), 2 * len(Xo))
        sel = np.random.RandomState(it).choice(len(Xtr), nb, replace=False)
        Xa = np.concatenate([Xtr[sel], Xo]); ya = np.concatenate([ytr[sel], yo])
        log(f"  aggregated {len(Xn)} on-rollout samples (on-rollout total {len(Xo)}, mixed batch {len(Xa)}); fine-tuning")
        base.fit_more(Xa, ya, epochs=4, log=lambda s: None); del Xa, ya, Xo, yo
    eng.close()
    json.dump(dict(plant=plant_val, iters=hist, params=PARAMS), open(os.path.join(OUT, f"{TAG}_summary.json"), "w"), indent=1)
    log(f"done {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
