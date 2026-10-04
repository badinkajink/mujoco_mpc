"""Aggregate one-step sim-gap: per-joint RMSE of the next-step joint velocity for
  zoh : dq1 = dq0                      (trivial predictor, the scale bar)
  pd  : plant + reconstructed PD torque (the twin as deployed)
  tau : plant + measured tau_est        (rigid body + contact model only)
split by joint group and by phase (upright <12 deg tilt vs leaning). Writes out/simgap_summary.json + figs."""
import os, json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

SIMG = os.path.join(DATA, "simgap")

def load_all(split=None):
    man = {r["run"]: r for r in json.load(open(os.path.join(DATA, "manifest.json")))}
    acc = {k: [] for k in ("e_zoh", "e_pd", "e_tau", "eq_zoh", "eq_pd", "eq_tau", "tilt", "run", "strat", "dq0", "dtau")}
    for f in sorted(glob.glob(os.path.join(SIMG, "*.npz"))):
        name = os.path.basename(f)[:-4]
        if split and man[name]["split"] != split: continue
        s = np.load(f); r = np.load(os.path.join(RUNS, name + ".npz"))
        idx = s["idx"]; ok = s["ok"]
        dq0 = r["dq"][idx]; dq1 = r["dq"][idx + 1]; q0 = r["q"][idx]; q1 = r["q"][idx + 1]
        acc["e_zoh"].append((dq0 - dq1)[ok]); acc["e_pd"].append((s["dq_pd"] - dq1)[ok]); acc["e_tau"].append((s["dq_tau"] - dq1)[ok])
        acc["eq_zoh"].append((q0 + REC_DT * dq0 - q1)[ok]); acc["eq_pd"].append((s["q_pd"] - q1)[ok]); acc["eq_tau"].append((s["q_tau"] - q1)[ok])
        acc["tilt"].append(s["tilt"][ok]); acc["dq0"].append(dq0[ok]); acc["dtau"].append((s["tau_pd0"] - r["tau"][idx])[ok])
        acc["run"].append(np.full(ok.sum(), name)); acc["strat"].append(np.full(ok.sum(), man[name]["strategy"]))
    return {k: np.concatenate(v) for k, v in acc.items()}

def rmse(e, mask=None):
    if mask is not None: e = e[mask]
    return np.sqrt(np.mean(e.astype(np.float64) ** 2, axis=0))

def main():
    A = load_all()
    n = len(A["tilt"]); print("samples", n)
    lean = A["tilt"] > 12
    summ = {"n": int(n), "n_lean": int(lean.sum()), "per_joint": {}}
    for ph, mask in (("all", np.ones(n, bool)), ("upright", ~lean), ("lean", lean)):
        summ["per_joint"][ph] = {k: rmse(A[k], mask).round(4).tolist() for k in ("e_zoh", "e_pd", "e_tau", "eq_zoh", "eq_pd", "eq_tau")}
    # group table
    rows = []
    for ph in ("all", "upright", "lean"):
        for g, ids in GROUPS.items():
            r = {k: float(np.sqrt(np.mean(np.square(summ["per_joint"][ph][k])[ids]))) for k in ("e_zoh", "e_pd", "e_tau")}
            rows.append(dict(phase=ph, group=g, **{k: round(v, 4) for k, v in r.items()}, ratio_pd_zoh=round(r["e_pd"] / r["e_zoh"], 3), ratio_tau_zoh=round(r["e_tau"] / r["e_zoh"], 3)))
    summ["groups"] = rows
    # torque-model gap: PD reconstruction vs measured tau_est
    summ["tau_pd_minus_est_rmse"] = rmse(A["dtau"]).round(3).tolist()
    summ["tau_pd_minus_est_mean"] = A["dtau"].mean(0).round(3).tolist()
    # per-strategy
    per = {}
    for st in sorted(set(A["strat"])):
        mk = A["strat"] == st
        per[st] = dict(n=int(mk.sum()), zoh=float(np.sqrt(np.mean(np.square(rmse(A["e_zoh"], mk))))), pd=float(np.sqrt(np.mean(np.square(rmse(A["e_pd"], mk))))), tau=float(np.sqrt(np.mean(np.square(rmse(A["e_tau"], mk))))))
    summ["per_strategy"] = per
    json.dump(summ, open(os.path.join(OUT, "simgap_summary.json"), "w"), indent=1)
    print(json.dumps(rows, indent=0)[:2000])
    # fig 1: per-joint RMSE bars
    fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    x = np.arange(NJ); w = 0.27
    for a, ph in zip(ax, ("upright", "lean")):
        pj = summ["per_joint"][ph]
        a.bar(x - w, pj["e_zoh"], w, label="hold dq (trivial)", color="#999")
        a.bar(x, pj["e_pd"], w, label="plant + PD cmd (twin)", color="#d9480f")
        a.bar(x + w, pj["e_tau"], w, label="plant + measured tau", color="#1864ab")
        a.set_ylabel(f"{ph}\nRMSE dq (rad/s), 10 ms"); a.grid(axis="y", alpha=.3)
    ax[0].legend(); ax[1].set_xticks(x); ax[1].set_xticklabels(SHORT, rotation=70, fontsize=8)
    fig.suptitle(f"One-step (10 ms) joint-velocity prediction error on {n} real H1-2 transitions, {len(set(A['run']))} runs")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_simgap_per_joint.png"), dpi=130)
    # fig 2: torque reconstruction gap
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.bar(x, summ["tau_pd_minus_est_rmse"], color="#5c940d"); ax.plot(x, summ["tau_pd_minus_est_mean"], "k.", label="mean (bias)")
    ax.set_xticks(x); ax.set_xticklabels(SHORT, rotation=70, fontsize=8); ax.set_ylabel("Nm"); ax.legend()
    ax.set_title("PD-reconstructed command torque minus measured tau_est (RMSE bars, bias dots)"); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_tau_recon_gap.png"), dpi=130)
    # fig 3: error vs tilt for groups
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    bins = np.arange(0, 50, 4); tc = 0.5 * (bins[1:] + bins[:-1])
    for a, (g, ids) in zip(ax, GROUPS.items()):
        for k, c, lab in (("e_zoh", "#999", "hold"), ("e_pd", "#d9480f", "PD twin"), ("e_tau", "#1864ab", "meas. tau")):
            y = [np.sqrt(np.mean(np.square(A[k][(A["tilt"] >= lo) & (A["tilt"] < hi)][:, ids]))) if ((A["tilt"] >= lo) & (A["tilt"] < hi)).sum() > 50 else np.nan for lo, hi in zip(bins[:-1], bins[1:])]
            a.plot(tc, y, "-o", color=c, label=lab, ms=3)
        a.set_title(g); a.set_xlabel("torso tilt (deg)"); a.grid(alpha=.3)
    ax[0].set_ylabel("RMSE dq (rad/s)"); ax[0].legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_simgap_vs_tilt.png"), dpi=130)
    print("saved figs")

if __name__ == "__main__":
    main()
