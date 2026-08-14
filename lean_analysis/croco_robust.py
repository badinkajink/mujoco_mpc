#!/usr/bin/env python3
"""Robustness of the crocoddyl MPC: what makes it survive, and how often.

WHY THIS EXISTS.  S15 ended with a controller that is real-time on the mean and
(after S16's OpenMP build) on the tail, and with one sentence it could not
defend: "every configuration falls in two or three of five" under 0.02 rad of
Gaussian noise on the initial joint angles.  S15 attributed that to the plan
being tracked from a single certified pose and left it.  It is not that.  It is
a cost-weight balance, and it is measurable.

THE DIAGNOSIS (S17).  The failing seeds do not fail at the brace.  They fail in
the feet-only APPROACH, 0.3-1.2 s in, and they fail BACKWARDS: the CoM walks off
the heel edge of the foot polygon and the robot sits down without the bracing arm
ever reaching the table.  Three measurements pin it:

  * The nominal stance has ~40 mm of rearward CoM margin.  `stand` puts the CoM
    at x = 0.227 and the heel edge at x = 0.185, so the maneuver STARTS one
    perturbation away from its own stability boundary -- before any noise.
  * The CoM barrier that is supposed to prevent this is priced at nothing.  It is
    a quadratic barrier at weight 1e3, so standing 17 mm outside the polygon
    costs 0.14, against ~25 for the landing cost pulling the arm down and ~5 for
    the reach.  The optimiser sells the support polygon and it is right to: the
    numbers say the polygon is worthless.
  * The same is true of the friction/wrench cones at weight 1e1.  The feet are
    RIGID 6D contacts in this OCP -- they can pull, and they can carry a CoP
    outside the sole -- and the cone barrier is the only term that says otherwise.
    A plan that leans on a wrench MuJoCo will not supply is a plan that falls.

So the fix is not a bigger horizon (measured: H = 100, i.e. re-solving the whole
remaining maneuver every 20 ms, still falls on the same seed) and not more DDP
iterations (measured: 2 and 5 iterations fall too, with the line search collapsed
to alpha = 0.002 -- the signature of a problem whose descent direction is being
spent on a constraint that is priced wrong).  It is the weights.

WHAT THIS MODULE DOES.  Three subcommands, all of which end in the same
verdict function so the columns are comparable:

  stress    one plan, N seeds x noise levels (and optional pushes) -> survival
  weights   a list of cost-weight configurations: re-plan, then stress each
  ablate    one cost group removed at a time: re-plan, then stress

`verdict` is the user's specification, not a solver metric: upright, hand on
target, torques inside the clamp basis, nothing driven through the table.  What
contact pattern got there is deliberately NOT part of it.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

import croco_bridge as cb          # first: sets RTLD_GLOBAL
import croco_replay as cr

HERE = os.path.dirname(os.path.abspath(__file__))

# The verdict thresholds, in one place because every table in the S17 docpage is
# scored with them.
REACH_TOL = 0.03        # m -- the study's own IK admissibility tolerance
PEN_TOL = 0.005         # m -- 5 mm into the wood; the soft contact allows some
TAU_TOL = 1.0           # measured |tau| / clamp-basis limit
STILL_TOL = 0.02        # m -- reach must not be drifting at the end


def verdict(res):
    """Did this replay do the task?  Returns (ok, reason).

    Deliberately physical and deliberately blind to the contact mode.  The user's
    specification for this study is: the robot stays up, the hand gets to the
    target and stays there, no joint is asked for more than the safety clamp
    allows, and nothing is driven through the table.  Which links end up on the
    wood is an outcome, not a requirement.
    """
    if res["fell"]:
        return False, "fell"
    if res["reach_err_at_brace_end"] > REACH_TOL:
        return False, "reach"
    if res["max_tau_ratio"] > TAU_TOL:
        return False, "torque"
    if -res["worst_penetration"] > PEN_TOL:
        return False, "penetration"
    if (res.get("reach_err_braced_std") or 0.0) > STILL_TOL:
        return False, "unsteady"
    return True, "ok"


# DISTURBANCE PROFILES.  A profile is how the robot ARRIVES, not a scalar noise
# level, because the two disturbances this maneuver actually sees are different
# objects.  `q` perturbs the 27 joint angles with the base pinned, which moves
# the FEET relative to the floor (12-25 mm of foot-corner spread at 0.02 rad) --
# it is S15's model and it is really a drop test.  `winch` is how this robot is
# instantiated: lowered on a hoist, so where it lands, which way it faces and how
# level it is all vary, and the whole body moves relative to the TABLE the
# certified landing spots are written on.  The winch numbers are one scale
# factor `w` on a fixed shape, so severity is a single axis.
def winch(w=1.0, q=0.02):
    return dict(q_noise=q, base_xy=0.020 * w, base_z=0.010 * w,
                base_yaw=0.050 * w, base_rp=0.020 * w)


PROFILES = {
    "nominal": dict(),
    "q0.02": dict(q_noise=0.02),
    "q0.05": dict(q_noise=0.05),
    "winch0.5": winch(0.5),
    "winch1": winch(1.0),
    "winch2": winch(2.0),
}
NOISY = {k: v for k, v in PROFILES.items() if v}


def one(tag, run_dir, seed=None, profile=(), push=0.0, horizon=35, iters=1,
        threads=12, tau_clamp=True, settle=0, dt_scale=1, video=None,
        cam="wide", push_at=0.35, push_dir=(-1, 0, 0), sense=None,
        mu_scale=1.0, table_shift=(0.0, 0.0, 0.0), schedule_shift=0):
    dist = dict(PROFILES[profile] if isinstance(profile, str) else profile)
    log, plan = cr.replay(tag, ctrl_mode="mpc", run_dir=run_dir,
                          dt_plan=cr.plan_dt(run_dir, tag),
                          mpc_horizon=horizon, mpc_iters=iters,
                          mpc_threads=threads, seed=seed,
                          push=push, push_at=push_at, push_dir=push_dir,
                          tau_clamp=tau_clamp, settle=settle,
                          mpc_dt_scale=dt_scale, sense=sense,
                          mu_scale=mu_scale, table_shift=table_shift,
                          schedule_shift=schedule_shift,
                          video=video, cam=cam, **dist)
    res = cr.summarise(log, plan, "mpc", verbose=False)
    ok, why = verdict(res)
    st = plan.get("settle") or {}
    return dict(seed=seed, profile=profile if isinstance(profile, str) else "",
                dist=dist, push=push, settle=settle, dt_scale=dt_scale,
                horizon=horizon, ok=ok, why=why,
                fell=res["fell"],
                reach_mm=1000 * res["reach_err_at_brace_end"],
                reach_sd_mm=1000 * (res["reach_err_braced_std"] or 0.0),
                tau=res["max_tau_ratio"],
                pen_mm=1000 * res["worst_penetration"],
                margin_mm=1000 * res["min_support_margin"],
                brace_N=res["brace_total_braced_mean"],
                settle_drift=st.get("drift_rad"),
                settle_margin_mm=1000 * st["min_support_margin"]
                if st else None,
                solve_ms=res.get("mpc_solve_ms", {}).get("mean"),
                p95_ms=res.get("mpc_solve_ms", {}).get("p95")), log


def stress(tag, run_dir, seeds=(1, 2, 3, 4, 5), profiles=("nominal", "q0.02"),
           pushes=(), verbose=True, **kw):
    """Every (profile, seed) cell.  The nominal profile is deterministic, so it
    is run once rather than once per seed."""
    rows = []
    for prof in profiles:
        for seed in (seeds if PROFILES.get(prof, prof) else (None,)):
            r, _ = one(tag, run_dir, seed=seed, profile=prof, **kw)
            rows.append(r)
            if verbose:
                print(f"    {prof:9s} seed {str(seed):>4s}  "
                      f"{'OK ' if r['ok'] else 'BAD'} {r['why']:12s} "
                      f"reach {r['reach_mm']:7.1f} mm  tau {r['tau']:.2f}  "
                      f"pen {r['pen_mm']:+6.1f} mm", flush=True)
    for push in pushes:
        r, _ = one(tag, run_dir, push=push, **kw)
        rows.append(r)
        if verbose:
            print(f"    push {push:5.0f} N       "
                  f"{'OK ' if r['ok'] else 'BAD'} {r['why']:12s} "
                  f"reach {r['reach_mm']:7.1f} mm  tau {r['tau']:.2f}", flush=True)
    return rows


def rate(rows, profile=None):
    sel = [r for r in rows if profile is None or r.get("profile") == profile]
    if not sel:
        return float("nan"), 0
    return sum(r["ok"] for r in sel) / len(sel), len(sel)


# ------------------------------------------------------------------- plan --
def plan_one(run_dir, tag, mode, weights, dt=0.02, n_approach=120, n_braced=80,
             iters=200, timeout=900, force=False, extra=()):
    """Solve one plan in a SUBPROCESS.  See croco_sweep for why: a cell that
    segfaults inside crocoddyl must not take the sweep with it."""
    out = os.path.join(run_dir, f"plan_{tag}.json")
    if os.path.exists(out) and not force:
        return True, 0.0
    cmd = [sys.executable, os.path.join(HERE, "croco_run.py"),
           "--mode", mode, "--tag", tag, "--dir", run_dir, "--dt", str(dt),
           "--n-approach", str(n_approach), "--n-braced", str(n_braced),
           "--iters", str(iters)]
    for k, v in weights.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    cmd += list(extra)
    t0 = time.time()
    with open(os.path.join(run_dir, f"plan_{tag}.log"), "w") as fh:
        try:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                           timeout=timeout, cwd=os.getcwd())
        except subprocess.TimeoutExpired:
            pass
    return os.path.exists(out), time.time() - t0


# --------------------------------------------------------------- commands --
def cmd_stress(args):
    rows = stress(args.tag, args.dir, seeds=args.seeds, profiles=args.profiles,
                  pushes=args.pushes, horizon=args.horizon, iters=args.iters,
                  threads=args.threads, tau_clamp=not args.no_tau_clamp,
                  settle=args.settle)
    for p in args.profiles:
        r, n = rate(rows, p)
        print(f"  {p:9s}: {r*100:3.0f}% of {n}")
    return dict(tag=args.tag, settle=args.settle, rows=rows)


# The configurations the weight study walks.  Each is a DELTA on the S15
# defaults, so a row reads as "what one knob is worth".  The order is the order
# they were tried, and the dead ends are kept: three of these make it WORSE, and
# which three is the result.
WEIGHT_SETS = [
    # -- generation 1: the two weights the diagnosis pointed at ---------------
    ("s15",            {}),
    ("com1e4",         {"w_com": 1e4}),
    ("com1e5",         {"w_com": 1e5}),
    ("com1e6",         {"w_com": 1e6}),
    ("margin50",       {"com_margin": 0.05}),
    ("cone1e2",        {"w_cone": 1e2}),
    ("cone1e3",        {"w_cone": 1e3}),
    ("com1e5_cone1e3", {"w_com": 1e5, "w_cone": 1e3}),
    # -- generation 2: an always-on CoM term, and CoP margin ------------------
    ("track1e3",       {"w_com_track": 1e3}),
    ("track1e4",       {"w_com_track": 1e4}),
    ("track3e4",       {"w_com_track": 3e4}),
    ("cop06",          {"cop_shrink": 0.6}),
    ("cop06_cone1e2",  {"cop_shrink": 0.6, "w_cone": 1e2}),
    ("track1e4_cop06", {"w_com_track": 1e4, "cop_shrink": 0.6}),
    ("minf20",         {"min_nforce": 20.0}),
]

def cmd_weights(args):
    sets = [(n, w) for n, w in WEIGHT_SETS
            if not args.only or n in args.only.split(",")]
    out = []
    for name, w in sets:
        tag = f"w_{name}"
        ok, secs = plan_one(args.dir, tag, args.mode, w, dt=args.dt,
                            n_approach=args.n_approach,
                            n_braced=args.n_braced, force=args.force)
        print(f"\n=== {name:16s} {w}  plan {'ok' if ok else 'FAILED'} "
              f"({secs:.0f}s)", flush=True)
        if not ok:
            out.append(dict(name=name, weights=w, planned=False))
            continue
        pl = json.load(open(os.path.join(args.dir, f"plan_{tag}.json")))
        rows = stress(tag, args.dir, seeds=args.seeds, profiles=args.profiles,
                      horizon=args.horizon, iters=args.iters,
                      threads=args.threads, tau_clamp=not args.no_tau_clamp,
                      settle=args.settle)
        r0, _ = rate(rows, "nominal")
        rn, n = rate(rows, args.profiles[-1])
        print(f"    -> nominal {'OK' if r0 else 'BAD'}, "
              f"{args.profiles[-1]}: {rn*100:.0f}% of {n}")
        out.append(dict(name=name, weights=w, planned=True,
                        plan_reach_mm=1000 * pl.get("reach_err", float("nan")),
                        plan_iters=pl["iters"], plan_seconds=pl["solve_seconds"],
                        plan_converged=pl["converged"],
                        plan_tau=pl["max_torque_ratio"], rows=rows,
                        survive=rn, survive_n=n))
        json.dump(dict(weights=out), open(args.out, "w"), indent=1)
    return dict(weights=out)


def cmd_matrix(args):
    """(plan variant) x (settle) x (disturbance profile), one survival each.

    The two fixes S17 finds act on different halves of the loop -- one is a cost
    term in the OCP, one is how the controller starts -- so the question that
    decides what to ship is whether they are redundant.  This is the table that
    answers it.
    """
    out = []
    grid = [(t, s_, h, n) for t in args.tags for s_ in args.settles
            for h in args.horizons for n in args.dt_scales]
    for tag, settle, H, scale in grid:
        for _ in (0,):
            print(f"\n=== {tag}  settle {settle}  H {H}  dt_scale {scale} "
                  f"(preview {H * scale * 0.02:.2f} s)", flush=True)
            rows = stress(tag, args.dir, seeds=args.seeds,
                          profiles=args.profiles, horizon=H,
                          iters=args.iters, threads=args.threads,
                          tau_clamp=not args.no_tau_clamp, settle=settle,
                          dt_scale=scale)
            rec = dict(tag=tag, settle=settle, horizon=H, dt_scale=scale,
                       preview_s=H * scale * 0.02, rows=rows, rate={},
                       solve_ms=float(np.median([r["solve_ms"] for r in rows
                                                 if r["solve_ms"]])),
                       p95_ms=float(np.median([r["p95_ms"] for r in rows
                                               if r["p95_ms"]])))
            for p in args.profiles:
                r, n = rate(rows, p)
                rec["rate"][p] = [r, n]
                print(f"    {p:9s}: {r*100:3.0f}% of {n}")
            print(f"    solve      {rec['solve_ms']:.1f} ms, "
                  f"p95 {rec['p95_ms']:.1f} ms")
            out.append(rec)
            json.dump(dict(matrix=out), open(args.out, "w"), indent=1)
    return dict(matrix=out)


# ------------------------------------------------------------- sim2real --- #
def est(p):
    """A state estimator of quality `p` metres of base position error.

    The shape is fixed and only the scale moves: orientation error in radians at
    the same number (a metre and a radian are not comparable, but 10 mm and
    10 mrad are both "a good estimator" and both "a bad one" together), base
    velocity a decade coarser, encoders at 0.5 mrad and their differentiated
    velocities two decades worse -- which is what differentiating an encoder
    over a 20 ms period actually gives you.
    """
    return dict(base_p=p, base_r=p, base_v=10 * p, base_w=10 * p,
                q=5e-4, v=5e-2)


# What a deployment would have to get right, one axis at a time.  Everything
# here perturbs the CONTROLLER's information or the PLANT's parameters, never
# the disturbance profile -- so a row is readable as "how accurate does this
# have to be", which is the number a hardware plan needs.
SIM2REAL = (
    [(f"est{int(p*1000)}mm", dict(sense=est(p))) for p in
     (0.002, 0.005, 0.010, 0.020)] +
    [("enc_only", dict(sense=dict(q=5e-4, v=5e-2)))] +
    [(f"mu{int(s*100)}", dict(mu_scale=s)) for s in (0.5, 0.75, 1.5)] +
    [(f"table_x{int(v*1000)}", dict(table_shift=(v, 0.0, 0.0)))
     for v in (0.010, 0.020, 0.040)] +
    [(f"table_y{int(v*1000)}", dict(table_shift=(0.0, v, 0.0)))
     for v in (0.020, 0.040)] +
    [(f"late{v}", dict(schedule_shift=v)) for v in (-10, -5, 5, 10)]
)


def cmd_sim2real(args):
    """Each axis alone, against the same disturbance profile and verdict.

    The question behind this is the user's: the cone costs look like they need
    force feedback and they do not (their wrench comes from the OCP's own KKT
    solve, not a sensor), so what DOES this loop need that a robot would have to
    supply?  Four candidates, and this prices them: the floating-base state
    estimate, the friction the plan assumed, where the table is, and when the
    contact schedule thinks touchdown happened.
    """
    out = [("baseline", {})] + list(SIM2REAL)
    res = []
    for name, kw in out:
        print(f"\n=== {name}  {kw}", flush=True)
        rows = stress(args.tag, args.dir, seeds=args.seeds,
                      profiles=args.profiles, horizon=args.horizon,
                      iters=args.iters, threads=args.threads,
                      tau_clamp=not args.no_tau_clamp, settle=args.settle,
                      **kw)
        rec = dict(name=name, kw={k: (list(v) if isinstance(v, tuple) else v)
                                  for k, v in kw.items()}, rows=rows, rate={})
        for p in args.profiles:
            r, n = rate(rows, p)
            rec["rate"][p] = [r, n]
            print(f"    {p:9s}: {r*100:3.0f}% of {n}")
        res.append(rec)
        json.dump(dict(sim2real=res), open(args.out, "w"), indent=1)
    return dict(sim2real=res)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--dir", default="runs/2026-08-13_session15/croco")
        p.add_argument("--tag", default="s13")
        p.add_argument("--horizon", type=int, default=35)
        p.add_argument("--iters", type=int, default=1)
        p.add_argument("--threads", type=int, default=12)
        p.add_argument("--no-tau-clamp", action="store_true",
                       help="let the position servo push a joint past the "
                            "clamp basis, as S13-S16 did.  The default clamps "
                            "the PLANT, so the safety limit is enforced rather "
                            "than checked afterwards")
        p.add_argument("--settle", type=int, default=0,
                       help="control periods of gravity-compensated hold at "
                            "the start pose before the maneuver advances")
        p.add_argument("--seeds", type=int, nargs="+",
                       default=[1, 2, 3, 4, 5])
        p.add_argument("--profiles", nargs="+",
                       default=["nominal", "q0.02"],
                       help=" ".join(sorted(PROFILES)))
        p.add_argument("--out", default=None)

    p = sub.add_parser("stress"); common(p)
    p.add_argument("--pushes", type=float, nargs="*", default=[])

    p = sub.add_parser("sim2real"); common(p)

    p = sub.add_parser("matrix"); common(p)
    p.add_argument("--tags", nargs="+", default=["w_s15", "w_track1e4"])
    p.add_argument("--settles", type=int, nargs="+", default=[0, 25])
    p.add_argument("--horizons", type=int, nargs="+", default=[35])
    p.add_argument("--dt-scales", type=int, nargs="+", default=[1])

    p = sub.add_parser("weights"); common(p)
    p.add_argument("--mode", default="elbow+forearm")
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--n-approach", type=int, default=120)
    p.add_argument("--n-braced", type=int, default=80)
    p.add_argument("--only", default=None)
    p.add_argument("--force", action="store_true")

    args = ap.parse_args()
    args.out = args.out or f"{args.cmd}.json"
    res = dict(stress=cmd_stress, weights=cmd_weights,
               matrix=cmd_matrix, sim2real=cmd_sim2real)[args.cmd](args)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
