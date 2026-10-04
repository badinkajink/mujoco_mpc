import json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
res = json.load(open(os.path.join(OUT, "residual_summary.json")))
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
o = res["one_step_test"]; names = ["hold_dq", "plant_pd", "plant_pd+ridge", "plant_pd+mlp"]; labs = ["hold velocity", "plant (twin)", "plant + ridge", "plant + MLP"]
cols = ["#999", "#d9480f", "#2b8a3e", "#1864ab"]; x = np.arange(3); w = 0.2
for k, (nm, lab, c) in enumerate(zip(names, labs, cols)):
    ax[0].bar(x + (k - 1.5) * w, [o[nm][g] for g in ("legs", "waist", "arms")], w, label=lab, color=c)
ax[0].set_xticks(x); ax[0].set_xticklabels(["legs", "waist", "arms"]); ax[0].set_ylabel("one-step RMSE dq (rad/s)"); ax[0].set_title(f"One-step, held-out runs ({res['n_test']} transitions)"); ax[0].legend(); ax[0].grid(axis="y", alpha=.3)
ro = res["rollout_test"]["upright"]; hs = [100, 250, 500]
for nm, lab, c in zip(["plant", "ridge", "mlp"], ["plant (twin)", "plant + ridge", "plant + MLP"], cols[1:]):
    for g, ls in (("legs", "-"), ("arms", "--")):
        ax[1].plot(hs, [ro[nm][f"{h}ms"][g] for h in hs], ls, marker="o", color=c, label=f"{lab} {g}")
ax[1].set_xlabel("horizon (ms)"); ax[1].set_ylabel("joint displacement RMSE (rad)"); ax[1].set_title(f"Closed-loop rollouts from upright starts ({ro['n_segments']} segments)"); ax[1].legend(fontsize=8, ncol=2); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_residual.png"), dpi=130); print("saved")
