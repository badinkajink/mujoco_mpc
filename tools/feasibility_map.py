#!/usr/bin/env python3
"""feasibility_map.py -- the (table x, table z) map: where does a LEGAL, LOAD-BEARING,
RECOVERABLE forearm brace exist?

Implements the sweep specified in docs/feasibility_map_design_2026-08-04.md. The
single-cell machinery already existed (tools/feasible_region.py, the Y_fa LP with
the ankle L1 diamond and sole hulls); this drives it across geometry and joins it
to a pose solve, which is what turns one reading into a map.

PER CELL, three questions, in order -- a cell must pass all three:

  1. LEGAL     does a brace POSE exist here? (brace_bow_solve.py)
               forearm forward (splay <= 10 deg, else `Brace Arm Plane` w300
               refuses it -- 2026-08-05), seated, flat, wrist clear, nothing else
               touching the slab, joints inside 0.9 x TAU_ESTOP.
  2. FEASIBLE  is that pose statically supportable?   Y_fa(feet + forearm)
  3. RECOVERABLE  once the arm LEAVES, can the feet alone hold the CoM?
               Y_fa(feet only), scored WITHOUT the brace vertex.

★ (3) is the half that actually decides the stand-back and the half a textbook
support-region map omits. Every gate we ran had a healthy BRACED polygon and most
still ended bowed at +17 deg, because release drops the robot into the feet-only
region and the pose was never inside it. Do NOT credit the brace vertex when
scoring recoverability: the load-gated brace vertex is exactly what makes
`Balance` read ~0 and LICENSES the forward CoM.

⚠ Feasible is not discoverable. A cell this map calls good can still fail in the
pipeline because the CEM sampler cannot find it (search width alone moved the
stand-back 5/20 -> 12/20 with nothing else changed). #29 must report both.

Usage:
    feasibility_map.py --x 0.80,0.86,0.91,0.96 --z 0.87,0.925,0.985,1.04
                       [--com-cap 0.12] [--restarts 12] [--jobs 4] --out map.json
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import mujoco
from mujoco import mjtObj

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feasible_region as FR

SOLVER = os.path.expanduser(
    "~/.claude/skills/mjpc-fk-analyze/scripts/brace_bow_solve.py")


def solve_cell(model, x, z, a, tmpdir):
    """Run the pose solver as a SUBPROCESS so the cell uses the exact CLI (and
    exact defaults) we would run by hand, and one bad cell cannot poison the rest."""
    jf = os.path.join(tmpdir, f"cell_{x:.3f}_{z:.3f}.json")
    cmd = [sys.executable, SOLVER, "--model", model,
           "--table-x", str(x), "--surface", str(z),
           "--restarts", str(a.restarts), "--iters", str(a.iters),
           "--max-splay", str(a.max_splay), "--brace-n", "0",
           "--quiet", "--json", jf]
    if a.com_cap is not None:
        cmd += ["--com-cap", str(a.com_cap)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=a.timeout)
        with open(jf) as f:
            return json.load(f)
    except Exception as e:                      # noqa: BLE001 - a failed cell IS data
        return {"error": f"{type(e).__name__}: {e}"}


def regions_for_pose(model, qpos, x, z):
    """Y_fa with the forearm, and Y_fa on the FEET ALONE (the post-release test)."""
    m = mujoco.MjModel.from_xml_path(model)
    d = mujoco.MjData(m)
    # place the table where this cell says before reading any geometry off it
    tb = m.geom_bodyid[mujoco.mj_name2id(m, mjtObj.mjOBJ_GEOM, "table_top_collision")]
    gt = mujoco.mj_name2id(m, mjtObj.mjOBJ_GEOM, "table_top_collision")
    m.body_pos[tb][0] = x + m.geom_size[gt][0]
    m.body_pos[tb][2] += z - (m.body_pos[tb][2] + m.geom_pos[gt][2] + m.geom_size[gt][2])
    d.qpos[:] = m.qpos0
    d.qpos[:len(qpos)] = qpos
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)

    feet = [p for nm in ("left_ankle_roll_link", "right_ankle_roll_link")
            for p in FR.sole_hull(m, d, nm)]
    fore = list(FR.forearm_patch(m, d))
    amid = 0.5 * (d.xpos[mujoco.mj_name2id(m, mjtObj.mjOBJ_BODY, "left_ankle_roll_link")][0] +
                  d.xpos[mujoco.mj_name2id(m, mjtObj.mjOBJ_BODY, "right_ankle_roll_link")][0])

    out = {}
    for tag, pts in (("braced", feet + fore), ("feet_only", feet)):
        P = np.array(pts)
        V = FR.region(m, d, P, [np.array([0, 0, 1.0])] * len(P),
                      use_actuation=True, ankle_diamond=True)
        out[tag] = (None if len(V) == 0
                    else dict(fwd=float(V[:, 0].max() - amid),
                              back=float(V[:, 0].min() - amid),
                              lat=[float(V[:, 1].min()), float(V[:, 1].max())]))
    out["ankle_mid"] = float(amid)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..",
        "build/mjpc/tasks/humanoid_bench/lean/Lean_H12_Magpie.xml"))
    ap.add_argument("--x", default="0.80,0.85,0.91,0.96,1.01",
                    help="table NEAR-EDGE x. 0.85 and <=0.19-from-edge are the "
                         "known-BAD cells the map must reproduce (3/3 collapse).")
    ap.add_argument("--z", default="0.87,0.925,0.985,1.04",
                    help="slab surface height. Real table adjusts 0.57-1.23 and "
                         "is MOVED TO THE ROBOT, so both axes are genuinely free.")
    ap.add_argument("--com-cap", type=float, default=None)
    ap.add_argument("--max-splay", type=float, default=10.0)
    ap.add_argument("--restarts", type=int, default=12)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default="feasibility_map.json")
    a = ap.parse_args()

    model = os.path.abspath(a.model)
    # ⚠ MJPC task XMLs resolve <mesh file=...> relative to the CWD, not the XML.
    # Run from the model's own directory or every load dies on a missing STL.
    os.chdir(os.path.dirname(model))
    model = os.path.basename(model)
    xs = [float(s) for s in a.x.split(",")]
    zs = [float(s) for s in a.z.split(",")]
    cells = [(x, z) for z in zs for x in xs]
    print(f"model {model}\n{len(cells)} cells = {len(xs)} x-positions x {len(zs)} heights"
          f"   splay cap {a.max_splay} deg"
          f"   com cap {a.com_cap if a.com_cap is not None else 'none'}"
          f"   {a.jobs} jobs")

    tmpdir = tempfile.mkdtemp(prefix="feasmap_")
    results = []

    def run(cell):
        x, z = cell
        rec = solve_cell(model, x, z, a, tmpdir)
        rec.update(table_x=x, table_z=z)
        if "error" not in rec and rec.get("qpos"):
            try:
                rec["regions"] = regions_for_pose(model, rec["qpos"], x, z)
            except Exception as e:              # noqa: BLE001
                rec["regions"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"  cell x={x:.2f} z={z:.3f}  "
              + ("ERROR " + rec["error"] if "error" in rec else
                 f"pose {rec['npass']}/{rec['ntest']}  splay {rec.get('splay', float('nan')):+.1f}  "
                 f"CoM {rec['com']*1000:+.0f} mm"), flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for r in ex.map(run, cells):
            results.append(r)

    # ---- verdict per cell --------------------------------------------------
    for r in results:
        reg = r.get("regions") or {}
        fo = reg.get("feet_only") if isinstance(reg, dict) else None
        legal = ("error" not in r) and r.get("npass") == r.get("ntest")
        r["legal"] = bool(legal)
        # ⛔ 2026-08-05: DO NOT read a failed cell as "this geometry is bad".
        # The first sweep ran at --restarts 12 and 23/30 cells failed
        # `forearm_seated` -- i.e. the optimiser never pushed the arm down, not
        # that it couldn't. The SAME cell (0.50/0.985) scored 14/14 seated at
        # -1.8 mm with --restarts 24. An under-converged sweep is a false-negative
        # machine, so the failing CHECKS are reported, never just the count.
        r["fails"] = [k for k, ok in (r.get("verdict") or {}).items() if not ok]
        r["recoverable"] = bool(
            legal and fo and r["com"] <= fo["fwd"])
        r["feet_only_fwd"] = None if not fo else fo["fwd"]
        r["braced_fwd"] = None if not (isinstance(reg, dict) and reg.get("braced")) \
            else reg["braced"]["fwd"]

    with open(a.out, "w") as f:
        json.dump(dict(model=model, x=xs, z=zs, com_cap=a.com_cap,
                       max_splay=a.max_splay, cells=results), f, indent=1)

    print(f"\n=== MAP  (legal = pose passes every solver check; "
          f"recoverable = CoM inside FEET-ONLY Y_fa) ===")
    # report BOTH coordinates: the sweep axis is the NEAR EDGE, but the historical
    # known-bad/known-good cells (0.85 collapses 3/3, 0.91 works) are quoted in
    # table BODY x, and confusing the two would invalidate the validation.
    mm_ = mujoco.MjModel.from_xml_path(model)
    gt_ = mujoco.mj_name2id(mm_, mjtObj.mjOBJ_GEOM, "table_top_collision")
    half = float(mm_.geom_size[gt_][0])
    print(f"{'edge_x':>7}{'body_x':>8}{'z':>7}{'pose':>8}{'splay':>8}{'CoM mm':>8}"
          f"{'braced':>9}{'feetonly':>9}{'margin':>8}  verdict")
    for r in results:
        if "error" in r:
            print(f"{r['table_x']:>7.2f}{r['table_x']+half:>8.2f}{r['table_z']:>7.3f}"
                  f"   SOLVER ERROR: {r['error'][:50]}")
            continue
        fo, br = r["feet_only_fwd"], r["braced_fwd"]
        mar = None if fo is None else fo - r["com"]
        vs = ("RECOVERABLE" if r["recoverable"] else
              ("legal, NOT recoverable" if r["legal"] else "no legal pose"))
        print(f"{r['table_x']:>7.2f}{r['table_x']+half:>8.2f}{r['table_z']:>7.3f}"
              f"{r['npass']:>5}/{r['ntest']:<2}{r.get('splay', float('nan')):>8.1f}"
              f"{r['com']*1000:>8.0f}"
              f"{(br*1000 if br is not None else float('nan')):>9.0f}"
              f"{(fo*1000 if fo is not None else float('nan')):>9.0f}"
              f"{(mar*1000 if mar is not None else float('nan')):>8.0f}  {vs}")
    # convergence audit BEFORE any geometry claim: if `forearm_seated` dominates
    # the failures, the sweep is under-converged and its verdicts mean nothing.
    seat = sum(1 for r in results if "forearm_seated" in r.get("fails", []))
    if seat > 0.25 * len(results):
        print(f"\n⛔ UNDER-CONVERGED: {seat}/{len(results)} cells failed "
              f"`forearm_seated` at --restarts {a.restarts}. That is the OPTIMISER "
              f"not seating the arm, NOT the geometry refusing it. Re-run with more "
              f"restarts before reading ANY cell as infeasible.")
    fails = {}
    for r in results:
        for k in r.get("fails", []):
            fails[k] = fails.get(k, 0) + 1
    if fails:
        print("failing checks: " + ", ".join(
            f"{k} {v}/{len(results)}" for k, v in
            sorted(fails.items(), key=lambda kv: -kv[1])))
    ok = [r for r in results if r.get("recoverable")]
    print(f"\n{len(ok)}/{len(results)} cells RECOVERABLE")
    if ok:
        b = max(ok, key=lambda r: r["feet_only_fwd"] - r["com"])
        print(f"widest margin: x={b['table_x']:.2f} z={b['table_z']:.3f}  "
              f"{(b['feet_only_fwd']-b['com'])*1000:.0f} mm  "
              f"(needs >= 35 mm for the real +3.5 cm ZMP-vs-CoM gap)")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
