#!/usr/bin/env python3
"""Belief-error and wrench-cap sweeps for the UR5 wrench-controlled push.

One episode per (object, belief, cap, seed). Objects are what the PLANT has; the
belief is what the PLANNER's model has; the cap is the planner's wrench ceiling.
Results append to <out>/results.jsonl as they land (fsync'd), per-episode logs go
to <out>/logs/<tag>.npz, and re-running skips tags already present.

Run policy (memory: mjpc-sweep-cpu-budget): jobs x threads must stay under nproc
and everything runs under nice. Default 3 jobs x 6 rollout threads = 18 of 20.
"""
import argparse, json, os, sys, time, math, dataclasses
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OBJECTS = {  # name: (mass kg, mu)
    "light":    (0.2, 0.4),
    "nominal":  (0.5, 0.4),
    "heavyish": (1.0, 0.4),
    "slippery": (0.5, 0.2),
    "grippy":   (0.5, 0.8),
    "heavy":    (2.0, 0.4),
}
MASS_RATIOS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]   # belief m_hat / m, mu_hat = mu
MU_RATIOS = [0.5, 2.0]                                 # belief mu_hat / mu, m_hat = m
CAP_RATIOS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]           # F_max / F_slide(true), matched belief


def conditions(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    conds = []
    for name, (m, mu) in OBJECTS.items():
        if args.objects and name not in args.objects.split(","):
            continue
        for seed in seeds:
            if "A" in args.sweeps:
                for r in MASS_RATIOS:
                    conds.append(dict(tag=f"A_{name}_m{r:g}_s{seed}", obj=name, m=m, mu=mu,
                                      m_hat=m * r, mu_hat=mu, f_max=math.inf, seed=seed, sweep="A"))
                for r in MU_RATIOS:
                    conds.append(dict(tag=f"A_{name}_mu{r:g}_s{seed}", obj=name, m=m, mu=mu,
                                      m_hat=m, mu_hat=mu * r, f_max=math.inf, seed=seed, sweep="A"))
            if "B" in args.sweeps:
                for r in CAP_RATIOS:
                    conds.append(dict(tag=f"B_{name}_cap{r:g}_s{seed}", obj=name, m=m, mu=mu,
                                      m_hat=m, mu_hat=mu, f_max=r * mu * m * 9.81, seed=seed, sweep="B"))
    return conds


def run_one(cond, out, nthread, t_max):
    os.nice(10)
    import numpy as np
    import ur5_push as up
    obj = up.Obj(mass=cond["m"], mu=cond["mu"])
    bel = up.Obj(mass=cond["m_hat"], mu=cond["mu_hat"])
    cfg = up.Cfg(nthread=nthread, seed=cond["seed"], f_max=cond["f_max"], t_max=t_max)
    log = []
    t0 = time.time()
    res = up.run_episode(obj, bel, cfg, log=log)
    res["wall_s"] = round(time.time() - t0, 1)
    rec = dict(cond)
    rec["f_max"] = None if math.isinf(cond["f_max"]) else cond["f_max"]
    rec.update(res)
    rec["cfg"] = {k: (None if isinstance(v, float) and math.isinf(v) else v)
                  for k, v in dataclasses.asdict(cfg).items()}
    keys = ["t", "box_x", "box_y", "box_z", "tilt_deg", "err", "hand_x", "hand_y", "hand_z",
            "f_mag", "f_lp", "J"]
    arr = {k: np.array([r[k] for r in log]) for k in keys}
    arr["u"] = np.array([r["u"] for r in log])
    arr["f_box"] = np.array([r["f_box"] for r in log])
    np.savez_compressed(os.path.join(out, "logs", cond["tag"] + ".npz"), **arr)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sweeps", default="AB")
    ap.add_argument("--objects", default="")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--t_max", type=float, default=6.0)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    res_path = os.path.join(a.out, "results.jsonl")
    done = set()
    if os.path.exists(res_path):
        for line in open(res_path):
            try:
                done.add(json.loads(line)["tag"])
            except Exception:
                pass
    conds = [c for c in conditions(a) if c["tag"] not in done]
    print(f"{len(conds)} episodes to run ({len(done)} already done)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex, open(res_path, "a") as f:
        futs = {ex.submit(run_one, c, a.out, a.threads, a.t_max): c for c in conds}
        n = 0
        for fut in __import__("concurrent.futures").futures.as_completed(futs):
            c = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:
                rec = dict(c, outcome="error", error=repr(e))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
            n += 1
            print(f"  [{n}/{len(conds)} {time.time() - t0:5.0f}s] {c['tag']:32s} {rec.get('outcome'):8s} "
                  f"t={rec.get('t_done', -1):5.2f} err={rec.get('final_err', float('nan')):.3f} "
                  f"f_lp={rec.get('peak_f_lp', float('nan')):.1f} wall={rec.get('wall_s')}", flush=True)


if __name__ == "__main__":
    main()
