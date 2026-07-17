#!/usr/bin/env python3
# Copyright 2026 — HAMS h12_lowerbody task tooling (max_playground branch).
# Licensed under the Apache License, Version 2.0 (same terms as the repo).

"""Cost-weight search for the "H1-2 Lowerbody" stand task.

Cross-entropy-method (CEM) search over the task's <user> cost-term weights
(task.xml <sensor> block). Each candidate weight vector is scored by rolling
out the iLQG planner against local MuJoCo physics (the cartpole demo idiom:
set_state -> planner_step -> get_action -> mj_step) while shoving the robot,
and measuring a FIXED held-out metric — uprightness + kinematic height minus
a torque penalty — never the tuned cost itself (scoring a cost by itself is
circular: zero weights would win).

Run INSIDE the hams_ros container (the task's robot model is bind-mounted
from CL_Assets there; on the host those include files are 0-byte mount
placeholders and this script falls back to the build-staged copy):

  docker exec -it hams_ros bash
  /home/code/h12_sim_scripts/rebuild_mjpc.sh        # agent_server is a default target
  pip install grpcio grpcio-tools                   # one-time
  cd /home/code/mujoco_mpc/python && python setup.py install   # one-time (upstream flow)
  python /home/code/mujoco_mpc/mjpc/tasks/h12_lowerbody/tuning/tune_weights.py

Smoke test (~1 min — one candidate at the XML defaults, one short episode):

  python tune_weights.py --iterations 1 --population 1 --episodes 1 --horizon 2

Outputs (under --outdir, default runs/<timestamp>/ next to this script — the
submodule mount is rw, so results persist on the host):
  config.json           resolved settings + term names/defaults/bounds
  log.jsonl             one row per candidate + per-iteration summary
  best_weights.json     best weight dict so far (updated live)
  user_lines.xml        ready-to-paste <user .../> lines with the best weights
  agent_server.log      server stdout/stderr

--write-xml additionally patches the weights into the source task.xml in
place (second field of each user="..." attribute). Review the diff before
committing — this edits the submodule working tree.

Fairness/variance notes: candidates within one iteration see identical push
sequences (common random numbers, rng seeded by [seed, iteration, episode]),
and the CEM mean is re-evaluated as candidate 0 of every iteration so the
incumbent stays comparable. Evaluation is serial; iLQG already fans out
across cores server-side, so parallel candidates would mostly fight for CPU.

Runtime scales as iterations x population x episodes x horizon x
planner_steps. Benchmarking uses N=25 episodes per candidate (the default),
which averages out push/init-noise variance but makes the full search an
overnight job (8 iters x 12 candidates x 25 eps x 4 s ~ 9600 planner-driven
episode-seconds). For a quick pass, drop --episodes first — CEM tolerates
noisy scores better than tiny populations tolerate noise.
"""

import argparse
import json
import pathlib
import re
import subprocess
import time

import mujoco
import numpy as np

from mujoco_mpc import agent as agent_lib

TASK_ID = "H1-2 Lowerbody"

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent                      # mjpc/tasks/h12_lowerbody
REPO_ROOT = TASK_DIR.parents[2]                   # mujoco_mpc checkout
BUILD_TASK_DIR = REPO_ROOT / "build" / "mjpc" / "tasks" / "h12_lowerbody"

# --- Held-out scoring metric (per control step, in [~-0.1, 1]) -------------
# score = UPRIGHT_W * clip(pelvis_up_z, 0, 1)            uprightness
#       + HEIGHT_W  * clip(rel_height / NOMINAL_H, 0, 1) kinematic height
#       - EFFORT_PENALTY * mean((actuator_force / capacity)^2)
# rel_height = pelvis z above the lower foot body (same kinematic-not-world-z
# philosophy as HeightResidual; NOMINAL_H is the default-stance value).
# An episode that falls scores 0 for its remaining steps, so survival
# dominates; the effort term breaks ties toward calm, low-torque stands.
UPRIGHT_W = 0.6
HEIGHT_W = 0.4
EFFORT_PENALTY = 0.1
NOMINAL_H = 0.98          # 1.03 m pelvis qpos0 minus ~0.05 m foot-body height
FALL_UP_Z = 0.3           # pelvis tilted past ~72 deg -> fallen
FALL_REL_H = 0.55         # pelvis below ~half stance height -> fallen


def parse_args():
  p = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--task-xml", type=pathlib.Path, default=None,
                 help="task.xml to load (default: source dir if its includes "
                      "are readable, else the build-staged copy)")
  p.add_argument("--server-binary", type=pathlib.Path,
                 default=REPO_ROOT / "build" / "bin" / "agent_server",
                 help="agent_server binary (default: %(default)s)")
  p.add_argument("--outdir", type=pathlib.Path, default=None,
                 help="output directory (default: runs/<timestamp>/ here)")
  p.add_argument("--terms", type=str, default="all",
                 help="comma-separated cost-term names to tune (default: all "
                      "user terms; untuned terms stay at their XML defaults)")
  # search
  p.add_argument("--iterations", type=int, default=8)
  p.add_argument("--population", type=int, default=12)
  p.add_argument("--elite-frac", type=float, default=0.3)
  p.add_argument("--sigma0", type=float, default=0.8,
                 help="initial CEM std in log-weight space (0.8 ~ x2.2 spread)")
  p.add_argument("--sigma-min", type=float, default=0.05,
                 help="CEM std floor (log space), keeps exploration alive")
  p.add_argument("--min-weight-frac", type=float, default=1e-4,
                 help="lower weight clamp as a fraction of each term's XML "
                      "slider max (upper clamp is the slider max itself)")
  # evaluation
  p.add_argument("--episodes", type=int, default=25,
                 help="episodes per candidate (averaged) — the benchmarking "
                      "sample size N; default 25")
  p.add_argument("--horizon", type=float, default=4.0,
                 help="episode length, seconds")
  p.add_argument("--ctrl-dt", type=float, default=0.01,
                 help="replan/control period, seconds (task agent_timestep)")
  p.add_argument("--planner-steps", type=int, default=4,
                 help="planner_step calls per control step")
  p.add_argument("--warmup-steps", type=int, default=10,
                 help="planner_step calls before each episode starts")
  p.add_argument("--push-vel", type=float, default=0.5,
                 help="mean push magnitude, m/s added to base linear velocity "
                      "(0 disables pushes)")
  p.add_argument("--push-interval", type=float, default=1.5,
                 help="mean seconds between pushes (first at ~1 s)")
  p.add_argument("--init-qvel-noise", type=float, default=0.02,
                 help="std of initial qvel perturbation, all DOFs")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--write-xml", action="store_true",
                 help="patch the best weights into the source task.xml")
  return p.parse_args()


def resolve_task_xml(args):
  """Source task dir when its bind-mounted includes are real, else build."""
  if args.task_xml is not None:
    return args.task_xml
  for d in (TASK_DIR, BUILD_TASK_DIR):
    inc = d / "h1_2_magpie.xml"
    if inc.is_file() and inc.stat().st_size > 0:
      return d / "task.xml"
  raise SystemExit(
      "No readable robot model: h1_2_magpie.xml is empty/missing in both "
      f"{TASK_DIR} and {BUILD_TASK_DIR}. Run inside hams_ros (assets are "
      "bind-mounted from CL_Assets there) or run rebuild_mjpc.sh to stage "
      "the build copy.")


def cost_terms(model):
  """(name, default_weight, slider_hi) for each <user> cost sensor.

  mjpc requires the user sensors to be the model's first sensors;
  sensor_user rows are [norm, weight, lo, hi, params...] (task.xml comment).
  """
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


class Evaluator:
  """Rolls out one candidate weight dict and returns its score."""

  def __init__(self, agent, model, args):
    self.agent = agent
    self.model = model
    self.data = mujoco.MjData(model)
    self.args = args
    self.n_ctrl = int(round(args.horizon / args.ctrl_dt))
    self.n_sub = max(1, int(round(args.ctrl_dt / model.opt.timestep)))
    # Torque capacity per actuator, JointTorquesResidual's guard: unlimited
    # actuators divide by 1.
    limited = model.actuator_ctrllimited.astype(bool)
    maxabs = np.abs(model.actuator_ctrlrange).max(axis=1)
    self.capacity = np.where(limited, np.maximum(maxabs, 1e-9), 1.0)

  def episode(self, weights, ep_seed):
    """One pushed stand episode; returns (score, aux dict)."""
    a, m, d = self.agent, self.model, self.data
    rng = np.random.default_rng(ep_seed)

    a.reset()  # also resets cost weights to XML defaults server-side
    a.set_cost_weights(weights, reset_to_defaults=True)

    mujoco.mj_resetData(m, d)  # qpos0 = default stand, pelvis at 1.03 m
    d.qvel[:] = rng.normal(0.0, self.args.init_qvel_noise, m.nv)
    mujoco.mj_forward(m, d)

    # Push schedule: horizontal base-velocity kicks in random directions,
    # drawn up front (common random numbers: same ep_seed -> same pushes).
    pushes = []
    if self.args.push_vel > 0:
      t = 1.0
      while t < self.args.horizon:
        ang = rng.uniform(0, 2 * np.pi)
        mag = self.args.push_vel * rng.uniform(0.7, 1.3)
        pushes.append((t, mag * np.cos(ang), mag * np.sin(ang)))
        t += self.args.push_interval * rng.uniform(0.7, 1.3)
    n_pushes = len(pushes)

    self._set_state()
    for _ in range(self.args.warmup_steps):
      a.planner_step()

    score = 0.0
    effort_sum = 0.0
    survived = self.n_ctrl
    for step in range(self.n_ctrl):
      while pushes and d.time >= pushes[0][0]:
        _, vx, vy = pushes.pop(0)
        d.qvel[0] += vx
        d.qvel[1] += vy

      self._set_state()
      for _ in range(self.args.planner_steps):
        a.planner_step()
      d.ctrl = a.get_action()
      for _ in range(self.n_sub):
        mujoco.mj_step(m, d)

      up_z = d.sensor("pelvis_up").data[2]
      rel_h = d.sensor("pelvis_pos").data[2] - min(
          d.sensor("left_foot_pos").data[2],
          d.sensor("right_foot_pos").data[2])
      effort = float(np.mean((d.actuator_force / self.capacity) ** 2))
      effort_sum += effort
      score += (UPRIGHT_W * np.clip(up_z, 0.0, 1.0)
                + HEIGHT_W * np.clip(rel_h / NOMINAL_H, 0.0, 1.0)
                - EFFORT_PENALTY * effort)

      if up_z < FALL_UP_Z or rel_h < FALL_REL_H:
        survived = step + 1
        break  # remaining steps contribute 0

    score /= self.n_ctrl  # early termination = lost score
    aux = {"survived_s": round(survived * self.args.ctrl_dt, 3),
           "fell": survived < self.n_ctrl,
           "mean_effort": round(effort_sum / max(survived, 1), 5),
           "pushes_applied": n_pushes - len(pushes)}
    return float(score), aux

  def _set_state(self):
    d = self.data
    self.agent.set_state(time=d.time, qpos=d.qpos, qvel=d.qvel, act=d.act,
                         mocap_pos=d.mocap_pos, mocap_quat=d.mocap_quat,
                         userdata=d.userdata)

  def candidate(self, weights, iteration, seed):
    """Mean score over episodes; per-episode seeds shared across candidates
    of the same iteration (common random numbers)."""
    scores, auxes = [], []
    for ep in range(self.args.episodes):
      s, aux = self.episode(weights, ep_seed=[seed, iteration, ep])
      scores.append(s)
      auxes.append(aux)
    return float(np.mean(scores)), scores, auxes


def emit_user_lines(task_xml_path, weights, out_path, write_in_place):
  """Rewrite the weight (2nd field of user="...") for each tuned term.

  Works on the XML text so norms/bounds/params and formatting survive
  untouched. Returns the rewritten <user .../> lines.
  """
  text = task_xml_path.read_text()
  lines = []
  for name, w in weights.items():
    pat = re.compile(
        r'(<user\s+name="%s"[^>]*\buser=")([^"]*)(")' % re.escape(name))

    def sub(mt, w=w):
      fields = mt.group(2).split()
      fields[1] = "%.6g" % w
      return mt.group(1) + " ".join(fields) + mt.group(3)

    text, n = pat.subn(sub, text)
    if n != 1:
      print(f"  WARNING: {'no' if n == 0 else 'multiple'} <user> match for "
            f"'{name}' in {task_xml_path}; skipped in XML output")
      continue
    lines.append(pat.search(text).group(0))

  out_path.write_text("\n".join(lines) + "\n")
  if write_in_place:
    task_xml_path.write_text(text)
    print(f"  patched weights into {task_xml_path}")
  return lines


def main():
  args = parse_args()
  task_xml = resolve_task_xml(args)
  outdir = args.outdir or SCRIPT_DIR / "runs" / time.strftime("%Y%m%d-%H%M%S")
  outdir.mkdir(parents=True, exist_ok=True)

  print(f"model:  {task_xml}")
  print(f"server: {args.server_binary}")
  print(f"outdir: {outdir}")
  if not args.server_binary.is_file():
    raise SystemExit("agent_server not found — run rebuild_mjpc.sh first.")

  model = mujoco.MjModel.from_xml_path(str(task_xml))
  terms = cost_terms(model)
  defaults = {name: w for name, w, _ in terms}

  if args.terms == "all":
    tuned = [name for name, _, _ in terms]
  else:
    tuned = [t.strip() for t in args.terms.split(",") if t.strip()]
    unknown = set(tuned) - set(defaults)
    if unknown:
      raise SystemExit(f"Unknown terms {sorted(unknown)}; "
                       f"available: {[n for n, _, _ in terms]}")

  hi = {name: h for name, _, h in terms}
  lo_b = np.log([max(args.min_weight_frac * hi[n], 1e-9) for n in tuned])
  hi_b = np.log([hi[n] for n in tuned])
  mu = np.clip(np.log([defaults[n] for n in tuned]), lo_b, hi_b)
  sigma = np.full(len(tuned), args.sigma0)

  (outdir / "config.json").write_text(json.dumps({
      "args": {k: str(v) if isinstance(v, pathlib.Path) else v
               for k, v in vars(args).items()},
      "task_xml": str(task_xml), "tuned_terms": tuned, "defaults": defaults,
      "slider_hi": hi, "metric": {"upright_w": UPRIGHT_W, "height_w": HEIGHT_W,
                                  "effort_penalty": EFFORT_PENALTY},
  }, indent=2))

  log_f = open(outdir / "log.jsonl", "a", buffering=1)
  server_log = open(outdir / "agent_server.log", "w")

  best = {"score": -np.inf, "weights": dict(defaults), "iteration": -1}
  with agent_lib.Agent(
      task_id=TASK_ID, model=model,
      server_binary_path=str(args.server_binary),
      subprocess_kwargs={"stdout": server_log,
                         "stderr": subprocess.STDOUT}) as agent:
    server_terms = set(agent.get_cost_weights())
    if server_terms != set(defaults):
      print(f"  WARNING: server terms {sorted(server_terms)} != local XML "
            f"terms — task registration and model out of sync?")

    ev = Evaluator(agent, model, args)
    print(f"tuning {len(tuned)}/{len(terms)} terms | "
          f"{args.iterations} iters x {args.population} candidates x "
          f"{args.episodes} eps x {args.horizon}s")

    try:
      for it in range(args.iterations):
        thetas, scores = [], []
        for c in range(args.population):
          # Candidate 0 is the current CEM mean (the incumbent); the rest
          # are log-space Gaussian samples clipped to the slider bounds.
          rng = np.random.default_rng([args.seed, it, 7919 + c])
          theta = mu if c == 0 else np.clip(
              mu + sigma * rng.standard_normal(len(tuned)), lo_b, hi_b)
          weights = dict(defaults)
          weights.update({n: float(np.exp(t)) for n, t in zip(tuned, theta)})

          t0 = time.monotonic()
          score, ep_scores, auxes = ev.candidate(weights, it, args.seed)
          thetas.append(theta)
          scores.append(score)
          log_f.write(json.dumps({
              "kind": "candidate", "iteration": it, "candidate": c,
              "score": score, "episode_scores": ep_scores, "episodes": auxes,
              "weights": weights, "wall_s": round(time.monotonic() - t0, 1),
          }) + "\n")
          marker = ""
          if score > best["score"]:
            best = {"score": score, "weights": weights, "iteration": it}
            (outdir / "best_weights.json").write_text(
                json.dumps(best, indent=2))
            marker = "  <-- best"
          print(f"  it {it:2d} cand {c:2d}  score {score:8.5f}  "
                f"fell {sum(a['fell'] for a in auxes)}/{len(auxes)}{marker}")

        # CEM update in log space, std floored to keep exploring.
        order = np.argsort(scores)[::-1]
        n_elite = max(1, min(args.population,
                             int(round(args.elite_frac * args.population))))
        elite = np.array([thetas[i] for i in order[:n_elite]])
        mu = np.clip(elite.mean(axis=0), lo_b, hi_b)
        sigma = np.maximum(elite.std(axis=0), args.sigma_min)
        log_f.write(json.dumps({
            "kind": "iteration", "iteration": it,
            "best_score": float(np.max(scores)),
            "mean_score": float(np.mean(scores)),
            "mu": {n: float(np.exp(v)) for n, v in zip(tuned, mu)},
            "sigma_log": {n: float(v) for n, v in zip(tuned, sigma)},
        }) + "\n")
        print(f"  == it {it}: best {np.max(scores):.5f} "
              f"mean {np.mean(scores):.5f} overall {best['score']:.5f}")
    except KeyboardInterrupt:
      print("\ninterrupted — finalizing with best-so-far")

  log_f.close()
  server_log.close()

  print(f"\nbest score {best['score']:.5f} (iteration {best['iteration']})")
  print(f"{'term':<22} {'default':>10} {'best':>10}")
  for n in tuned:
    print(f"{n:<22} {defaults[n]:>10.4g} {best['weights'][n]:>10.4g}")

  # Ready-to-paste <user> lines (and optional in-place task.xml patch) are
  # rewritten against the SOURCE task.xml, the committed source of truth.
  lines = emit_user_lines(TASK_DIR / "task.xml",
                          {n: best["weights"][n] for n in tuned},
                          outdir / "user_lines.xml", args.write_xml)
  print(f"\n{outdir / 'user_lines.xml'}:")
  for ln in lines:
    print(f"  {ln}")


if __name__ == "__main__":
  main()
