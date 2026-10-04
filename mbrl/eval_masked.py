"""Evaluate a trained residual on val rollouts with a per-group mask (legs/waist/arms on/off) at the planner step."""
import os, json, sys, numpy as np, torch
from common import *
import rollout_engine as RE, residual_fit as RF
tag = sys.argv[1]; T = 25
st = torch.load(os.path.join(OUT, f"residual_mlp_{tag}.pt"), map_location="cpu"); nz = np.load(os.path.join(OUT, f"residual_norm_{tag}.npz"))
params = dict(nsub=int(os.environ.get("MBRL_NSUB", "1")), dt=float(os.environ.get("MBRL_DT", "0.01")), inject=os.environ.get("MBRL_INJECT", "pre"))
eng = RE.Engine(8); segs = RE.filter_consistent(eng, RE.compact_segments("val", T, 800, seed=1), T)[:400]
print("segments", len(segs), params)
def mk(mask):
    m = RE.NumpyMLP({k: v.numpy() for k, v in st.items()}, nz["mu"], nz["sd"], nz["ymu"], nz["ysd"], nz["clip"] * mask); return m
res = {}
for name, mask in (("plant", None), ("all", np.ones(NJ)), ("arms+waist", (np.arange(NJ) >= 12).astype(float)), ("arms", (np.arange(NJ) >= 13).astype(float))):
    sc = RE.score(segs, eng.run(segs, T, params, None if mask is None else mk(mask)), horizons=(10, 25))
    res[name] = sc; print(f"{name:11s} div {sc['diverged']}  " + "  ".join(f"{h}: legs {sc[h]['legs']:.4f} waist {sc[h]['waist']:.4f} arms {sc[h]['arms']:.4f}" for h in ("100ms", "250ms")), flush=True)
eng.close(); json.dump(res, open(os.path.join(OUT, f"masked_{tag}.json"), "w"), indent=1)
