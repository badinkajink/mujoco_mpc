#!/usr/bin/env python3
# Copyright 2026 — HAMS h12_lowerbody task tooling (max_playground branch).
# Licensed under the Apache License, Version 2.0 (same terms as the repo).

"""Cost-weight search + residual ablation for the "H1-2 Lowerbody" stand task.

Two things, sharing one parallel MPC-rollout evaluator:

  1. SEARCH  — cross-entropy-method (CEM) search over the task's <user> cost
     weights (task.xml <sensor> block), in log-weight space, clipped to each
     term's XML slider bounds.
  2. ABLATE  — set each term's weight to 0 in turn (from a reference config)
     and re-score, to answer "is this residual even needed?". A term whose
     removal does not significantly worsen the held-out score is redundant.

Every candidate is scored by rolling out the iLQG planner against local MuJoCo
physics (the cartpole-demo idiom: set_state -> planner_step -> get_action ->
mj_step) while shoving the robot, and measuring a FIXED held-out metric —
uprightness + kinematic height minus a torque penalty — NEVER the tuned cost
itself (scoring a cost by itself is circular: zero weights would win).

Scale (32-core box): closed-loop MPC rollout is expensive (~1600 iLQG solves
per 4 s episode). We (a) run several capped agent_server workers in parallel
(--workers x --server-threads ~ ncores), and (b) rank search candidates on a
cheaper --search-episodes budget while reserving the full --episodes=N=25 for
the EVALUATION of baseline / best / ablations. Common random numbers: all
candidates in an iteration — and all configs in the ablation — see identical
push/init seeds, so comparisons are low-variance.

Run INSIDE the hams_ros container (the robot model is bind-mounted from
CL_Assets there; on the host the include files are 0-byte mount placeholders).
One-time setup (see this file's git-adjacent notes / the session that built it):

  /home/code/h12_sim_scripts/rebuild_mjpc.sh        # builds agent_server
  pip install grpcio grpcio-tools -c <(printf 'numpy==1.26.4\\nmujoco==3.2.3\\n')
  cd python && mkdir -p mujoco_mpc/proto && cp ../mjpc/grpc/agent.proto mujoco_mpc/proto/ \\
    && touch mujoco_mpc/proto/__init__.py \\
    && python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. mujoco_mpc/proto/agent.proto

  PYTHONPATH=/home/code/mujoco_mpc/python python tune_weights.py --mode both

Smoke test (~1-2 min):

  PYTHONPATH=... python tune_weights.py --mode search --iterations 1 \\
    --population 2 --search-episodes 1 --horizon 2 --planner-steps 1 --workers 2

Outputs (under --outdir, default runs/<timestamp>/ next to this file; the
submodule mount is rw so results persist on the host):
  config.json        resolved settings + term names/defaults/bounds
  search_log.jsonl   one row per candidate + per-iteration CEM summary
  best_weights.json  best weight dict so far (updated live)
  ablation.json      per-config N-episode stats + delta-vs-reference + verdict
  eval.json          N-episode stats for baseline (defaults) and best
  user_lines.xml     paste-ready <user .../> lines with the best weights
  server_*.log       per-worker agent_server stdout/stderr

--write-xml additionally patches the best weights into the source task.xml in
place (second field of each user="..." attribute) — review the diff before
committing; it edits the submodule working tree.
"""

import argparse
import json
import pathlib
import re
import subprocess
import time
from concurrent import futures

import mujoco
import numpy as np

from mujoco_mpc import agent as agent_lib

TASK_ID = "H1-2 Lowerbody"

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent                      # mjpc/tasks/h12_lowerbody
REPO_ROOT = TASK_DIR.parents[2]                   # mujoco_mpc checkout
BUILD_TASK_DIR = REPO_ROOT / "build" / "mjpc" / "tasks" / "h12_lowerbody"

# --- Held-out scoring metric (per control step, in ~[-0.1, 1]) -------------
# score = UPRIGHT_W * clip(pelvis_up_z, 0, 1)            uprightness
#       + HEIGHT_W  * clip(rel_height / NOMINAL_H, 0, 1) kinematic height
#       - EFFORT_PENALTY * mean((actuator_force / capacity)^2)
# rel_height = pelvis z above the lower foot body (same kinematic-not-world-z
# philosophy as HeightResidual). An episode that falls scores 0 for its
# remaining steps, so survival dominates; effort breaks ties toward calm,
# low-torque stands.
UPRIGHT_W = 0.6
HEIGHT_W = 0.4
EFFORT_PENALTY = 0.1
NOMINAL_H = 0.98          # 1.03 m pelvis qpos0 minus ~0.05 m foot-body height
FALL_UP_Z = 0.3           # pelvis tilted past ~72 deg -> fallen
FALL_REL_H = 0.55         # pelvis below ~half stance height -> fallen


def parse_args():
  p = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--mode", choices=["search", "ablate", "both", "eval"],
                 default="both")
  p.add_argument("--eval-configs", type=pathlib.Path, default=None,
                 help="mode=eval: JSON {name: {term: weight,...}, ...}; each "
                      "config is scored at N=--episodes with shared CRN seeds "
                      "and compared (paired) to the first config listed")
  p.add_argument("--task-xml", type=pathlib.Path, default=None,
                 help="task.xml to load (default: source dir if readable, "
                      "else the build-staged copy)")
  p.add_argument("--server-binary", type=pathlib.Path,
                 default=REPO_ROOT / "build" / "bin" / "agent_server")
  p.add_argument("--outdir", type=pathlib.Path, default=None,
                 help="output dir (default: runs/<timestamp>/ next to script)")
  p.add_argument("--terms", type=str, default="all",
                 help="comma-separated cost-term names to tune/ablate "
                      "(default: all user terms)")
  # parallelism
  p.add_argument("--workers", type=int, default=6,
                 help="parallel evaluator processes, each its own agent_server")
  p.add_argument("--cores-per-worker", type=int, default=0,
                 help="CPU cores pinned per worker via affinity (0=auto: "
                      "(ncpu-2)//workers). Each agent_server inherits the pin, "
                      "so its iLQG pool (planner_threads = seen_cores - 3) and "
                      "the workers stop fighting over the same cores. "
                      "mjpc's --mjpc_workers flag does NOT size the iLQG pool "
                      "(only sched_getaffinity does), which is why we pin.")
  # search
  p.add_argument("--iterations", type=int, default=6)
  p.add_argument("--population", type=int, default=25)
  p.add_argument("--search-episodes", type=int, default=8,
                 help="episodes per candidate during CEM ranking (cheaper than "
                      "the N=25 final eval; CRN keeps ranking stable)")
  p.add_argument("--elite-frac", type=float, default=0.3)
  p.add_argument("--sigma0", type=float, default=0.8,
                 help="initial CEM std in log-weight space (0.8 ~ x2.2 spread)")
  p.add_argument("--sigma-min", type=float, default=0.05)
  p.add_argument("--min-weight-frac", type=float, default=1e-4,
                 help="lower weight clamp as a fraction of each term's slider "
                      "max (upper clamp is the slider max itself)")
  # evaluation / ablation
  p.add_argument("--episodes", type=int, default=25,
                 help="N: episodes for the final evaluation of baseline / best "
                      "/ each ablation config (the benchmarking sample size)")
  p.add_argument("--ablate-ref", choices=["default", "best"], default="default",
                 help="reference config each term is zeroed from (default: the "
                      "committed XML weights — answers 'is it needed as-is?')")
  # episode dynamics
  p.add_argument("--horizon", type=float, default=3.0,
                 help="episode length, seconds")
  p.add_argument("--ctrl-dt", type=float, default=0.01,
                 help="replan/control period, seconds")
  p.add_argument("--planner-steps", type=int, default=1,
                 help="planner_step calls per control step (iLQG warm-starts, "
                      "so 1 is usually enough for standing)")
  p.add_argument("--warmup-steps", type=int, default=20,
                 help="planner_step calls before each episode starts")
  p.add_argument("--push-vel", type=float, default=0.5,
                 help="mean push magnitude, m/s of base linear velocity "
                      "(0 disables pushes)")
  p.add_argument("--push-interval", type=float, default=1.5,
                 help="mean seconds between pushes (first at ~1 s)")
  p.add_argument("--init-qvel-noise", type=float, default=0.02,
                 help="std of initial qvel perturbation, all DOFs")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--write-xml", action="store_true",
                 help="patch the best weights into the source task.xml")
  return p.parse_args()


def resolve_task_xml(task_xml_arg):
  """Source task dir when its bind-mounted includes are real, else build."""
  if task_xml_arg is not None:
    return task_xml_arg
  for d in (TASK_DIR, BUILD_TASK_DIR):
    inc = d / "h1_2_magpie.xml"
    if inc.is_file() and inc.stat().st_size > 0:
      return d / "task.xml"
  raise SystemExit(
      "No readable robot model: h1_2_magpie.xml is empty/missing in both "
      f"{TASK_DIR} and {BUILD_TASK_DIR}. Run inside hams_ros (assets are "
      "bind-mounted from CL_Assets there) or run rebuild_mjpc.sh.")


def cost_terms(model):
  """(name, default_weight, slider_hi) per <user> cost sensor (model's first
  sensors; sensor_user rows are [norm, weight, lo, hi, params...])."""
  terms = []
  for i in range(model.nsensor):
    if model.sensor_type[i] != mujoco.mjtSensor.mjSENS_USER:
      break
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
    terms.append((name, float(model.sensor_user[i][1]),
                  float(model.sensor_user[i][3])))
  if not terms:
    raise SystemExit("No <user> cost sensors found in the model.")
  return terms


# --------------------------------------------------------------------------
# Worker-process state + rollout. Each ProcessPool worker holds one persistent
# agent_server (own port) and its own MjModel/MjData, created in the initializer
# (after fork, so no gRPC state is inherited from the parent).
# --------------------------------------------------------------------------
_W = {}


def _worker_init(task_xml, server_binary, counter, lock, cores_per_worker,
                 ncpu, eval_params, log_dir):
  import os
  # Claim a disjoint core block. Setting THIS process's affinity before the
  # Agent is spawned makes the agent_server inherit it through fork+exec, so
  # its iLQG pool sizes to len(block)-3 and no two workers share cores.
  with lock:
    idx = counter.value
    counter.value += 1
  start = idx * cores_per_worker
  cores = set(range(start, min(start + cores_per_worker, ncpu)))
  if cores:
    os.sched_setaffinity(0, cores)

  wid = os.getpid()
  model = mujoco.MjModel.from_xml_path(task_xml)
  log = open(pathlib.Path(log_dir) / f"server_w{idx}_{wid}.log", "w")
  agent = agent_lib.Agent(
      task_id=TASK_ID, model=None, server_binary_path=server_binary,
      subprocess_kwargs={"stdout": log, "stderr": subprocess.STDOUT})
  limited = model.actuator_ctrllimited.astype(bool)
  maxabs = np.abs(model.actuator_ctrlrange).max(axis=1)
  capacity = np.where(limited, np.maximum(maxabs, 1e-9), 1.0)
  ep = dict(eval_params)
  ep["n_sub"] = max(1, int(round(ep["ctrl_dt"] / model.opt.timestep)))
  ep["n_ctrl"] = int(round(ep["horizon"] / ep["ctrl_dt"]))
  _W.update(model=model, data=mujoco.MjData(model), agent=agent,
            capacity=capacity, ep=ep)


def _episode(weights, ep_seed):
  """One pushed stand episode; returns (score, aux). Reads worker globals."""
  a, m, d, cap, ep = (_W["agent"], _W["model"], _W["data"],
                      _W["capacity"], _W["ep"])
  rng = np.random.default_rng(ep_seed)

  a.reset()
  a.set_cost_weights(weights, reset_to_defaults=True)
  mujoco.mj_resetData(m, d)                       # qpos0 = default stand
  d.qvel[:] = rng.normal(0.0, ep["init_qvel_noise"], m.nv)
  mujoco.mj_forward(m, d)

  # Push schedule drawn up front: same ep_seed -> same pushes (CRN).
  pushes = []
  if ep["push_vel"] > 0:
    t = 1.0
    while t < ep["horizon"]:
      ang = rng.uniform(0, 2 * np.pi)
      mag = ep["push_vel"] * rng.uniform(0.7, 1.3)
      pushes.append((t, mag * np.cos(ang), mag * np.sin(ang)))
      t += ep["push_interval"] * rng.uniform(0.7, 1.3)
  n_pushes = len(pushes)

  _set_state(a, d)
  for _ in range(ep["warmup_steps"]):
    a.planner_step()

  score = 0.0
  effort_sum = 0.0
  survived = ep["n_ctrl"]
  for step in range(ep["n_ctrl"]):
    while pushes and d.time >= pushes[0][0]:
      _, vx, vy = pushes.pop(0)
      d.qvel[0] += vx
      d.qvel[1] += vy

    _set_state(a, d)
    for _ in range(ep["planner_steps"]):
      a.planner_step()
    d.ctrl = a.get_action()
    for _ in range(ep["n_sub"]):
      mujoco.mj_step(m, d)

    up_z = d.sensor("pelvis_up").data[2]
    rel_h = d.sensor("pelvis_pos").data[2] - min(
        d.sensor("left_foot_pos").data[2], d.sensor("right_foot_pos").data[2])
    effort = float(np.mean((d.actuator_force / cap) ** 2))
    effort_sum += effort
    score += (UPRIGHT_W * np.clip(up_z, 0.0, 1.0)
              + HEIGHT_W * np.clip(rel_h / NOMINAL_H, 0.0, 1.0)
              - EFFORT_PENALTY * effort)

    if up_z < FALL_UP_Z or rel_h < FALL_REL_H:
      survived = step + 1
      break

  score /= ep["n_ctrl"]
  aux = {"survived_s": round(survived * ep["ctrl_dt"], 3),
         "fell": survived < ep["n_ctrl"],
         "mean_effort": round(effort_sum / max(survived, 1), 5),
         "pushes_applied": n_pushes - len(pushes)}
  return float(score), aux


def _set_state(a, d):
  a.set_state(time=d.time, qpos=d.qpos, qvel=d.qvel, act=d.act,
              mocap_pos=d.mocap_pos, mocap_quat=d.mocap_quat,
              userdata=d.userdata)


def _worker_eval(job):
  """job = (tag, weights, ep_seeds) -> (tag, scores, auxes).

  A single episode that raises (transient gRPC error) or diverges (physics NaN)
  is recorded as a worst-case fall rather than propagating — one bad episode
  must not kill a multi-hour run. Such episodes are flagged aux['error'].
  """
  tag, weights, ep_seeds = job
  scores, auxes = [], []
  for s in ep_seeds:
    try:
      sc, ax = _episode(weights, s)
      if not np.isfinite(sc):
        sc, ax = 0.0, {"survived_s": 0.0, "fell": True, "mean_effort": 0.0,
                       "pushes_applied": 0, "error": "nonfinite"}
    except Exception as exc:  # noqa: BLE001 - resilience over a long run
      sc, ax = 0.0, {"survived_s": 0.0, "fell": True, "mean_effort": 0.0,
                     "pushes_applied": 0, "error": repr(exc)[:200]}
    scores.append(sc)
    auxes.append(ax)
  return tag, scores, auxes


def _worker_verify(_):
  """Return (server_nq, sorted server cost-term names) for a consistency check."""
  a = _W["agent"]
  return len(a.get_state().qpos), sorted(a.get_cost_weights())


def summarize(scores, auxes):
  scores = np.asarray(scores, float)
  n = len(scores)
  return {
      "n": n,
      "mean": float(scores.mean()),
      "std": float(scores.std(ddof=1)) if n > 1 else 0.0,
      "sem": float(scores.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
      "min": float(scores.min()),
      "max": float(scores.max()),
      "fall_rate": float(np.mean([a["fell"] for a in auxes])),
      "mean_survived_s": float(np.mean([a["survived_s"] for a in auxes])),
      "mean_effort": float(np.mean([a["mean_effort"] for a in auxes])),
  }


def run_search(pool, args, tuned, defaults, bounds, outdir):
  lo_b, hi_b = bounds
  mu = np.clip(np.log([defaults[n] for n in tuned]), lo_b, hi_b)
  sigma = np.full(len(tuned), args.sigma0)
  log_f = open(outdir / "search_log.jsonl", "a", buffering=1)
  best = {"score": -np.inf, "weights": dict(defaults), "iteration": -1}

  print(f"SEARCH: {len(tuned)} terms | {args.iterations} iters x "
        f"{args.population} candidates x {args.search_episodes} eps")
  for it in range(args.iterations):
    ep_seeds = [[args.seed, it, e] for e in range(args.search_episodes)]  # CRN
    jobs = []
    thetas = []
    for c in range(args.population):
      rng = np.random.default_rng([args.seed, it, 7919 + c])
      theta = mu if c == 0 else np.clip(
          mu + sigma * rng.standard_normal(len(tuned)), lo_b, hi_b)
      thetas.append(theta)
      weights = dict(defaults)
      weights.update({n: float(np.exp(t)) for n, t in zip(tuned, theta)})
      jobs.append((c, weights, ep_seeds))

    t0 = time.monotonic()
    scores = [None] * args.population
    for tag, sc, ax in pool.map(_worker_eval, jobs):
      weights = jobs[tag][1]
      scores[tag] = float(np.mean(sc))
      log_f.write(json.dumps({
          "kind": "candidate", "iteration": it, "candidate": tag,
          "score": scores[tag], "episode_scores": sc,
          "fell": sum(a["fell"] for a in ax), "weights": weights}) + "\n")
      if scores[tag] > best["score"]:
        best = {"score": scores[tag], "weights": weights, "iteration": it}
        (outdir / "best_weights.json").write_text(json.dumps(best, indent=2))

    order = np.argsort(scores)[::-1]
    n_elite = max(1, min(args.population,
                         int(round(args.elite_frac * args.population))))
    elite = np.array([thetas[i] for i in order[:n_elite]])
    mu = np.clip(elite.mean(axis=0), lo_b, hi_b)
    sigma = np.maximum(elite.std(axis=0), args.sigma_min)
    log_f.write(json.dumps({
        "kind": "iteration", "iteration": it,
        "best_score": float(np.max(scores)), "mean_score": float(np.mean(scores)),
        "wall_s": round(time.monotonic() - t0, 1),
        "mu": {n: float(np.exp(v)) for n, v in zip(tuned, mu)}}) + "\n")
    print(f"  it {it}: iter-best {np.max(scores):.5f} mean {np.mean(scores):.5f} "
          f"overall {best['score']:.5f}  ({time.monotonic()-t0:.0f}s)")

  log_f.close()
  return best


def run_ablation(pool, args, tuned, reference, ref_name, outdir):
  """Zero each tuned term from `reference`; N-episode eval with shared seeds."""
  ep_seeds = [[args.seed, 424242, e] for e in range(args.episodes)]  # CRN
  jobs = [("__reference__", dict(reference), ep_seeds)]
  for term in tuned:
    cfg = dict(reference)
    cfg[term] = 0.0
    jobs.append((f"ablate::{term}", cfg, ep_seeds))

  print(f"ABLATION from '{ref_name}': {len(tuned)} terms x N={args.episodes} "
        f"episodes (+reference)")
  t0 = time.monotonic()
  stats, raw = {}, {}
  for tag, sc, ax in pool.map(_worker_eval, jobs):
    stats[tag] = summarize(sc, ax)
    raw[tag] = np.asarray(sc, float)
  ref = stats["__reference__"]
  ref_sc = raw["__reference__"]

  rows = []
  for term in tuned:
    s = stats[f"ablate::{term}"]
    # Paired comparison: reference and this ablation saw identical CRN seeds,
    # so score the per-episode difference (far tighter than unpaired). d>0 =>
    # removing the term hurt that episode => the term earns its keep.
    d = ref_sc - raw[f"ablate::{term}"]
    delta = float(d.mean())
    sem = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    z = delta / sem if sem > 0 else 0.0
    if delta <= 0 or z < 1.0:
      verdict = "REDUNDANT"          # removing it did not (significantly) hurt
    elif z < 2.0:
      verdict = "weak"
    else:
      verdict = "needed"
    rows.append({"term": term, "weight_in_ref": reference[term],
                 "ablated_mean": s["mean"], "delta": delta, "delta_z": z,
                 "verdict": verdict, "ablated_fall_rate": s["fall_rate"],
                 "delta_fall_rate": s["fall_rate"] - ref["fall_rate"],
                 "ablated_stats": s})

  rows.sort(key=lambda r: r["delta"])   # most-redundant first
  out = {"reference_name": ref_name, "reference_weights": reference,
         "reference_stats": ref, "episodes": args.episodes, "terms": rows}
  (outdir / "ablation.json").write_text(json.dumps(out, indent=2))
  print(f"  ablation done ({time.monotonic()-t0:.0f}s). "
        f"reference mean {ref['mean']:.5f} +/- {ref['sem']:.5f} "
        f"(fall {ref['fall_rate']:.2f})")
  return out


def evaluate(pool, args, tag, weights, seed_tag):
  ep_seeds = [[args.seed, seed_tag, e] for e in range(args.episodes)]
  (_, sc, ax), = pool.map(_worker_eval, [(tag, weights, ep_seeds)])
  return summarize(sc, ax)


def run_named_eval(pool, args, configs, outdir):
  """Score each named weight-config at N=args.episodes with shared CRN seeds;
  report each and its paired delta vs the FIRST config listed. Used to verify a
  hand-built config (e.g. tuned weights with redundant terms zeroed) as a whole,
  since one-at-a-time ablation does not capture interaction effects."""
  ep_seeds = [[args.seed, 314159, e] for e in range(args.episodes)]  # CRN
  jobs = [(name, dict(w), ep_seeds) for name, w in configs.items()]
  print(f"EVAL: {len(configs)} configs x N={args.episodes} episodes (shared CRN)")
  t0 = time.monotonic()
  stats, raw = {}, {}
  for tag, sc, ax in pool.map(_worker_eval, jobs):
    stats[tag] = summarize(sc, ax)
    raw[tag] = np.asarray(sc, float)
  (outdir / "eval_configs.json").write_text(json.dumps(
      {"episodes": args.episodes, "configs": configs, "stats": stats}, indent=2))

  base = list(configs)[0]
  print(f"  ({time.monotonic()-t0:.0f}s)   [d_vs_base = config - '{base}', paired; "
        f"z>0 => better]")
  print(f"  {'config':<22}{'mean':>9}{'sem':>8}{'fall':>7}{'d_vs_base':>11}{'z':>7}")
  for n in sorted(configs, key=lambda k: -stats[k]["mean"]):
    d = raw[n] - raw[base]
    dm = float(d.mean())
    ds = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    z = dm / ds if ds > 0 else 0.0
    print(f"  {n:<22}{stats[n]['mean']:>9.4f}{stats[n]['sem']:>8.4f}"
          f"{stats[n]['fall_rate']:>7.2f}{dm:>+11.4f}{z:>7.1f}")
  return stats


def emit_user_lines(task_xml_path, weights, out_path, write_in_place):
  text = task_xml_path.read_text()
  lines = []
  for name, w in weights.items():
    # group3 spans the closing quote through the tag close so group(0) is the
    # full self-closing <user .../> element (paste-ready).
    pat = re.compile(
        r'(<user\s+name="%s"[^>]*\buser=")([^"]*)("[^>]*/?>)' % re.escape(name))

    def sub(mt, w=w):
      fields = mt.group(2).split()
      fields[1] = "%.6g" % w
      return mt.group(1) + " ".join(fields) + mt.group(3)

    text, n = pat.subn(sub, text)
    if n != 1:
      print(f"  WARNING: {'no' if n == 0 else 'multiple'} <user> match for "
            f"'{name}'; skipped in XML output")
      continue
    lines.append(pat.search(text).group(0))

  out_path.write_text("\n".join(lines) + "\n")
  if write_in_place:
    task_xml_path.write_text(text)
    print(f"  patched weights into {task_xml_path}")
  return lines


def main():
  args = parse_args()
  task_xml = resolve_task_xml(args.task_xml)
  outdir = args.outdir or SCRIPT_DIR / "runs" / time.strftime("%Y%m%d-%H%M%S")
  outdir.mkdir(parents=True, exist_ok=True)
  import os
  ncpu = os.cpu_count() or 1
  cpw = args.cores_per_worker or max(1, (ncpu - 2) // args.workers)
  print(f"model:  {task_xml}")
  print(f"server: {args.server_binary}  ({args.workers} workers x {cpw} cores "
        f"= iLQG pool ~{max(1, cpw - 3)} threads each, of {ncpu} cores)")
  print(f"outdir: {outdir}")
  if not args.server_binary.is_file():
    raise SystemExit("agent_server not found — run rebuild_mjpc.sh first.")

  model = mujoco.MjModel.from_xml_path(str(task_xml))
  terms = cost_terms(model)
  defaults = {name: w for name, w, _ in terms}
  hi = {name: h for name, _, h in terms}
  if args.terms == "all":
    tuned = [name for name, _, _ in terms]
  else:
    tuned = [t.strip() for t in args.terms.split(",") if t.strip()]
    unknown = set(tuned) - set(defaults)
    if unknown:
      raise SystemExit(f"Unknown terms {sorted(unknown)}; have "
                       f"{[n for n, _, _ in terms]}")
  lo_b = np.log([max(args.min_weight_frac * hi[n], 1e-9) for n in tuned])
  hi_b = np.log([hi[n] for n in tuned])

  eval_params = {k: getattr(args, k) for k in
                 ("horizon", "ctrl_dt", "planner_steps", "warmup_steps",
                  "push_vel", "push_interval", "init_qvel_noise")}
  (outdir / "config.json").write_text(json.dumps({
      "args": {k: str(v) if isinstance(v, pathlib.Path) else v
               for k, v in vars(args).items()},
      "task_xml": str(task_xml), "tuned_terms": tuned, "defaults": defaults,
      "slider_hi": hi, "metric": {"upright_w": UPRIGHT_W, "height_w": HEIGHT_W,
                                  "effort_penalty": EFFORT_PENALTY,
                                  "nominal_h": NOMINAL_H}}, indent=2))

  import multiprocessing as mp
  mgr = mp.Manager()
  counter = mgr.Value("i", 0)
  lock = mgr.Lock()
  pool = futures.ProcessPoolExecutor(
      max_workers=args.workers, initializer=_worker_init,
      initargs=(str(task_xml), str(args.server_binary), counter, lock, cpw,
                ncpu, eval_params, str(outdir)))
  best = None
  try:
    # Consistency check via a worker (server loads its own registered task; we
    # never ship the 118 MB local model — it blows the 40 MB gRPC cap).
    nqs = list(pool.map(_worker_verify, range(args.workers)))
    server_nq, server_terms = nqs[0]
    if server_nq != model.nq:
      raise SystemExit(f"server nq={server_nq} != local nq={model.nq} — "
                       "agent_server task and local task.xml out of sync "
                       "(rerun rebuild_mjpc.sh to re-stage assets).")
    if set(server_terms) != set(defaults):
      print(f"  WARNING: server terms != local XML terms")

    if args.mode == "eval":
      configs = json.loads(args.eval_configs.read_text())
      run_named_eval(pool, args, configs, outdir)

    if args.mode in ("search", "both"):
      best = run_search(pool, args, tuned, defaults, (lo_b, hi_b), outdir)

    if args.mode in ("search", "both"):
      # Final N=25 evaluation of the honest comparison points.
      eval_out = {"baseline_default": evaluate(
          pool, args, "baseline", dict(defaults), 111)}
      if best is not None:
        eval_out["best"] = evaluate(pool, args, "best", best["weights"], 222)
        eval_out["best_weights"] = best["weights"]
      (outdir / "eval.json").write_text(json.dumps(eval_out, indent=2))
      b = eval_out["baseline_default"]
      print(f"EVAL N={args.episodes}: baseline {b['mean']:.5f} +/- {b['sem']:.5f} "
            f"(fall {b['fall_rate']:.2f})")
      if best is not None:
        e = eval_out["best"]
        print(f"           best     {e['mean']:.5f} +/- {e['sem']:.5f} "
              f"(fall {e['fall_rate']:.2f})")

    if args.mode in ("ablate", "both"):
      if args.ablate_ref == "best" and best is not None:
        reference, ref_name = best["weights"], "best-found"
      else:
        reference, ref_name = dict(defaults), "default-XML"
      abl = run_ablation(pool, args, tuned, reference, ref_name, outdir)
      print("\n  residual ablation (most-redundant first):")
      print(f"  {'term':<20}{'w':>8}{'ablated':>10}{'delta':>9}"
            f"{'z':>7}  verdict")
      for r in abl["terms"]:
        print(f"  {r['term']:<20}{r['weight_in_ref']:>8.3g}"
              f"{r['ablated_mean']:>10.4f}{r['delta']:>9.4f}"
              f"{r['delta_z']:>7.1f}  {r['verdict']}")
  except KeyboardInterrupt:
    print("\ninterrupted — finalizing with results so far")
  finally:
    pool.shutdown(wait=False, cancel_futures=True)

  # Paste-ready <user> lines for the best weights (source task.xml is truth).
  if best is not None:
    lines = emit_user_lines(TASK_DIR / "task.xml",
                            {n: best["weights"][n] for n in tuned},
                            outdir / "user_lines.xml", args.write_xml)
    print(f"\nbest score {best['score']:.5f} (iteration {best['iteration']})")
    print(f"  {'term':<22}{'default':>10}{'best':>10}")
    for n in tuned:
      print(f"  {n:<22}{defaults[n]:>10.4g}{best['weights'][n]:>10.4g}")


if __name__ == "__main__":
  main()
