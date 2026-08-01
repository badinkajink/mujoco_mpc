#!/usr/bin/env python3
"""Why the simulation timestep is 0.005 and --speed is the knob instead.

Coarsening <option timestep> looks like a cheap way to get a 50 Hz control loop.
It is not the same experiment as --speed=0.25, and this measures the two reasons
why -- one about the integrator, one about the benchmark's own scoring.

1. Integrator fidelity. With u=0 and no friction the horizontal momentum starts
   at zero and is conserved, so the centre of mass is an EXACT invariant. Any
   drift is integrator error, not chaos -- which matters, because on a chaotic
   system the trajectory itself diverges under any perturbation whatsoever and
   so cannot distinguish "wrong" from "different".

2. Detection fidelity. corridor_benchmark samples ncon and min_clearance once
   per mj_step. A coarser timestep samples the collision test less often, so
   short contact episodes fall between samples and a worse simulation scores
   better. This measures how long the disqualifying episodes actually last.

Usage:
  python3 mjpc/tasks/triple_pendulum_cartpole/benchmark/timestep_study.py
  python3 ... timestep_study.py --dumps 'renders/outcome_dist/*/r_*.csv'
"""
import argparse
import csv
import glob
import pathlib

import numpy as np
import mujoco


def default_xml():
    return str(pathlib.Path(__file__).resolve().parent.parent / "slalom.xml")


def total_energy(m, d):
    mujoco.mj_energyPos(m, d)
    mujoco.mj_energyVel(m, d)
    return d.energy[0] + d.energy[1]


def com_x(m, d):
    """Horizontal centre of mass -- the exact invariant (see docstring)."""
    mass = m.body_mass[1:]
    return float((mass * d.xipos[1:, 0]).sum() / mass.sum())


def fidelity(xml, seconds):
    print(f"=== integrator fidelity: u=0, undamped, no contact, {seconds:.0f} s ===")
    print("CoM_x is exactly conserved here, so its drift is integrator error.\n")
    print(f"{'integrator':>14} {'dt':>7} {'steps':>7} {'|dE| (J)':>10} "
          f"{'CoM_x drift (m)':>17} {'cart (m)':>10}")
    integrators = [("implicitfast", mujoco.mjtIntegrator.mjINT_IMPLICITFAST),
                   ("rk4", mujoco.mjtIntegrator.mjINT_RK4)]
    for name, integ in integrators:
        for dt in (0.005, 0.010, 0.020, 0.040):
            m = mujoco.MjModel.from_xml_path(xml)
            m.opt.timestep = dt
            m.opt.integrator = integ
            d = mujoco.MjData(m)
            # off the equilibrium, well clear of the disks, so this measures the
            # integrator on the pendulum rather than the contact solver
            d.qpos[:] = [0.0, 0.3, -0.2, 0.4]
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            e0, c0 = total_energy(m, d), com_x(m, d)
            n = int(round(seconds / dt))
            for _ in range(n):
                mujoco.mj_step(m, d)
            print(f"{name:>14} {dt:7.3f} {n:7d} {abs(total_energy(m, d)-e0):10.4f} "
                  f"{abs(com_x(m, d)-c0):17.4f} {d.qpos[0]:10.4f}")
    print("\nRead CoM_x, not |dE|. Energy drift saturates across implicitfast's "
          "timesteps (2.4 to\n3.7 J, all of them larger than the 1.9 J the "
          "system started with) so it cannot rank\nthem; it only separates the "
          "two integrators. CoM_x ranks everything: 0.088 m at\ndt=0.005 "
          "against 1.476 m at dt=0.020, on a course whose half-gap is 0.25 m.")


def detection(patterns):
    print("\n=== contact-episode length in recorded slalom rollouts ===")
    files = sorted(f for p in patterns for f in glob.glob(p))
    if not files:
        print("no dumps found; run a sweep with --dump first")
        return
    episodes, first = [], []
    for path in files:
        with open(path) as f:
            rows = list(csv.DictReader(l for l in f if not l.startswith("#")))
        overlap = [float(r["min_clearance"]) < 0.0 for r in rows
                   if "min_clearance" in r]
        run, seen = 0, False
        for v in overlap + [False]:
            if v:
                run += 1
            elif run:
                episodes.append(run)
                if not seen:
                    first.append(run)
                    seen = True
                run = 0
    if not episodes:
        print("no overlaps recorded")
        return
    print(f"{len(files)} rollouts, {len(episodes)} overlap episodes, "
          f"{len(first)} of them a run's first (the disqualifying one)\n")
    for name, arr in (("all episodes", np.array(episodes)),
                      ("first episode per run", np.array(first))):
        print(f"  {name}: median {np.median(arr):.0f} steps "
              f"({np.median(arr)*5:.0f} ms at dt=0.005)")
        for k in (1, 2, 4, 8):
            print(f"    <= {k:2d} steps ({k*5:3d} ms): "
                  f"{100*(arr <= k).mean():5.1f}%")
    fe = np.array(first)
    print(f"\n  A 0.020 timestep samples the collision test every 4th row. "
          f"{(fe <= 4).sum()}/{len(fe)} ({100*(fe <= 4).mean():.0f}%) of "
          f"disqualifying\n  episodes are <= 4 rows long, so a coarser step "
          f"walks over a large share of them\n  and reports a HIGHER success "
          f"rate for a worse simulation.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xml", default=default_xml())
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--dumps", nargs="+",
                   default=["renders/outcome_dist/*/r_*.csv",
                            "renders/slalom_gallery/dumps/*.csv"])
    a = p.parse_args()
    fidelity(a.xml, a.seconds)
    detection(a.dumps)
