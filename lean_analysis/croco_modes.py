#!/usr/bin/env python3
"""Enumerate contact modes at one reach target and SAVE the pose for each.

sweep5_full.py already enumerates 2^5 contact subsets and scores them with the
static QP, but it keeps only the scalars -- the pose that earned the score is
thrown away when `load()` is called for the next subset.  crocoddyl needs the
pose: q* is the terminal target of the OCP and the contact subset is its contact
schedule, and they have to be the SAME solve or the plan is aiming at a pose that
was never certified.

So this is sweep5_full's inner loop with the qpos retained, written one file per
admissible mode, plus a manifest recording which solve produced which pose.

Admissibility is sweep5_full's, unchanged:
    IK ok (reach < 3 cm, feet pinned, no penetration, every site actually
    touching)  AND  base residual < 1 N  AND  max |tau|/tau_limit <= 1.
Ranking is sweep5_full's too: least normalized actuator effort, ties broken
toward FEWER contacts.
"""

import argparse
import itertools
import json
import os
import subprocess

import numpy as np

import contact_select as cs

SITES5 = ("elbow", "forearm", "palm", "hip", "torso")


def enumerate_modes(target, sites=SITES5, verbose=True):
    subsets = [s for k in range(len(sites) + 1)
               for s in itertools.combinations(sites, k)]
    out = []
    for sub in subsets:
        m, d = cs.load()
        P = cs.solve_ik(m, d, np.asarray(target, float), sub)
        r = cs.equilibrium_qp(m, d, sub)
        adm = bool(P["ok"] and r["base_residual"] < 1.0 and r["max_ratio"] <= 1.0)
        rec = dict(subset=list(sub), admissible=adm,
                   reach=float(P["reach"]), penetration=float(P["penetration"]),
                   placed=bool(P["all_placed"]),
                   base_res=float(r["base_residual"]),
                   max_ratio=float(r["max_ratio"]), effort=float(r["effort"]),
                   brace_force=float(r["brace_force"]),
                   n_arm=len([t for t in sub if t in cs.ARM_SITES]),
                   qpos=d.qpos.copy())
        out.append(rec)
        if verbose:
            tag = "OK " if adm else "-- "
            print(f"  {tag}{str(sub):44s} effort={r['effort']:7.3f} "
                  f"max|tau|/lim={r['max_ratio']:5.2f} reach={P['reach']*1000:6.1f}mm "
                  f"brace={r['brace_force']:6.1f}N", flush=True)
    return out


def rank(records):
    adm = [r for r in records if r["admissible"]]
    return sorted(adm, key=lambda r: (round(r["effort"], 4), len(r["subset"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, nargs=3,
                    default=[0.9047, -0.2348, 1.0982],
                    help="reach target; default is the session-12 target")
    ap.add_argument("--out", default="runs/2026-08-05_session12/croco")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"target {args.target}   sites {SITES5}   ({2**len(SITES5)} subsets)\n")
    recs = enumerate_modes(args.target)
    order = rank(recs)

    print(f"\n{len(order)} admissible of {len(recs)}")
    for i, r in enumerate(order[:8]):
        print(f"  {i}. {'+'.join(r['subset']) or 'legs_only':28s} "
              f"effort={r['effort']:.3f}  max_ratio={r['max_ratio']:.2f}")

    manifest = {"target": list(map(float, args.target)),
                "sites": list(SITES5),
                "model": os.path.basename(cs.MODEL),
                "brace_arm": cs.BRACE_ARM, "site_set": cs.SITE_SET,
                "seed_key": cs.SEED_KEY, "tau_basis": cs.TAU_BASIS,
                "commit": _git_head(), "modes": []}
    for r in recs:
        name = "+".join(r["subset"]) or "legs_only"
        entry = {k: v for k, v in r.items() if k != "qpos"}
        entry["name"] = name
        if r["admissible"]:
            f = f"q_{name}.txt"
            np.savetxt(os.path.join(args.out, f), r["qpos"])
            entry["qpos_file"] = f
        manifest["modes"].append(entry)
    manifest["ranked"] = ["+".join(r["subset"]) or "legs_only" for r in order]
    with open(os.path.join(args.out, "modes.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"\nwrote {args.out}/modes.json")


def _git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
