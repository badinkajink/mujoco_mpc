# Model-based RL for the H1-2 lean-and-brace controller: research plan

Written 2026-10-04 on branch `mbrl`. Grounded in live documentation fetched the same day
(MuJoCo, Unitree, NVIDIA; URLs at the end) and in what already exists in this tree.

## 1. Where we stand

The deployed controller is MuJoCo MPC with sampling (iCEM) over a *hand-built* MuJoCo plant,
`Lean_H12_Magpie.xml`. That is already the "planning" half of model-based RL. What is missing,
and what makes this a research contribution rather than engineering, is the **learning** half:

1. a model that is corrected from real data (the twin is known to be off: forward-mass offset
   ~8 cm, ankle-zero 4-8 deg, actuator chatter, pad/slab contact creep, heat-dependent torque);
2. a policy or value function learned *with* that model so the robot does not need 20-30
   plans/s of CPU at run time, and so the dive, graze and brace can be committed without an
   operator hand.

Everything below is sequenced so each stage produces a publishable measurement on its own.

## 2. Thesis

> A sampling MPC on a physics plant plus a small learned residual, trained from a few hours
> of real-robot logs, closes most of the sim-to-real gap for whole-body contact tasks, and the
> same corrected plant is good enough to distil an MPC policy that runs without online planning.

Falsifiable at every stage: if the residual does not cut one-step and 0.5 s rollout error on
held-out runs, stage 1 fails and we stop there.

## 3. Stages

### Stage 0: data and sim-gap measurement (started tonight, see `REPORT_*.md`)

Already in hand: 493 real-robot recordings, 6.1 M transitions at 100 Hz, each with joint
q/dq/tau_est, the full PD command (target, kp, kd, tau_ff), torso IMU quat/gyro/accel and the
estimator base pose. Tools: `build_dataset.py`, `simgap.py`, `analyze_simgap.py`.

Metric: one-step (10 ms) prediction error of the plant vs the recording, against the trivial
"hold velocity" predictor, split arms / legs / waist and upright / leaning. Two torque models:
PD-reconstructed command (the twin as deployed) and measured tau_est (isolates rigid-body +
contact error from actuator error).

### Stage 1: learned residual dynamics (started tonight, `residual_fit.py`)

Target: residual of the plant's next joint velocity. Inputs: only planner-known quantities
(q, dq, commanded torque, gains, gravity direction, pelvis rate). Models: ridge then a 3x512 MLP.
Judge on held-out *runs*, not held-out samples, with one-step RMSE and closed-loop 0.1 / 0.25 /
0.5 s rollouts (the MPC horizon is 1.0 s). Deliverable: a table "plant / plant+ridge /
plant+MLP / hold" per joint group and per strategy.

Second pass, if the residual helps: **parametric** correction first (armature, frictionloss,
damping, per-joint torque scale, pelvis mass offset) with the MuJoCo 3.5+ `mujoco.sysid`
toolbox. The Unitree mjlab community found the shipped armature/friction defaults give 30 %
overshoot in sim vs 2 % on the real R1 (`unitree_rl_mjlab#51`); an H1-2 equivalent is a clean
result on its own. Parametric fixes transfer into the planner XML directly; the MLP needs a
plugin or a post-step hook in `mjpc`.

### Stage 2: residual inside the planner

Two ways to use the learned residual at 20-30 plans/s:
- **Online**: a per-step velocity correction applied inside the rollout (`mjpc` rollouts call
  `mj_step`; a `user` callback or a tiny actuator plugin can add it). Cost: one MLP eval per
  rollout step; at 27 x 3 x 512 this is ~1 M flops, fine on CPU for 64 samples x 100 steps.
- **Offline**: distil the residual into plant parameters (Stage 1 second pass) and ship XML.

Measure: twin bench parity (does the honest twin with the corrected plant reproduce the real
seat/collapse statistics), then real-robot A/B on strat 25 with the fixed window protocol.

### Stage 3: policy from the model (the "RL" in MBRL)

Options, in order of risk:
1. **MPC distillation (DAgger)**: run iCEM on the corrected plant from thousands of perturbed
   starts, record (obs, action), train an MLP policy, iterate with on-policy relabelling. This is
   the MuZero/TD-MPC-lite route that keeps the planner as the teacher.
2. **Dyna-style model-based PPO**: model-free PPO on the corrected plant with the learned
   residual in the loop, GPU batched. Needs MJX/Warp; **blocker**: MuJoCo Warp does not support
   the `noslip` solver our `*_real_ns` plants use, and has no autodiff. Use the pyramidal plant
   for training, verify on the noslip twin.
3. **Model-based value expansion** on top of 1 or 2.

Measure: success rate on the lean/brace ladder in the honest twin with no operator assist, then
on hardware; CPU load at run time (the stand needs < 25 plans/s today).

## 4. What the ecosystem offers (fetched 2026-10-04)

| Need | Status | Source |
|---|---|---|
| GPU batched MuJoCo | MuJoCo 3.14 / `mujoco-warp` 3.14; Warp officially released in 3.5 (Feb 2026); no autodiff; no `noslip`, no implicitfast midpoint | mujoco changelog, mujoco_warp README |
| System ID | `mujoco.sysid` Python toolbox since 3.5; maintainers advise in-air leg ID for floating-base humanoids | mujoco discussion #3238 |
| Humanoid RL envs | Playground: H1 and G1 joystick envs, **no H1-2**; Isaac Lab: H1 and G1, no H1-2; mjlab: `Unitree-H1_2-Flat` 27 DoF with deploy yaml | mujoco_playground, IsaacLab, unitree_rl_mjlab |
| Learned dynamics in MJPC | None; "residual" there means cost residual. Direct/batch estimators exist | mujoco_mpc README, PR #402 |
| World models | GR00T Dreams and Cosmos 3 are pixel-space, data generators or action-conditioned video; not a state-space model for control | GR00T-Dreams, Cosmos docs |
| Real actuator ID recipe | mjlab#51: step + sine sweeps, fit armature/frictionloss per joint group | unitree_rl_mjlab#51 |
| Low-level state | `rt/lowstate` hg: q, dq, ddq, tau_est, temps, IMU; no foot force; 500 Hz in SDK examples | H1_developer docs |

## 5. Risks

- Base pose comes from our own estimator; leg one-step error is contaminated by it. Arms are
  the clean signal. Report both.
- Table height varied per run while the plant has one table; brace-phase error partly reflects
  that. Stage 1 features exclude the table on purpose.
- Heat: motor torque constant drifts with winding temperature (logged since 09-03). A residual
  that ignores temperature will show run-to-run bias; check per-run residual bias vs temp.
- No hardware tonight. Everything here is offline; nothing in this branch changes the live tree.

## 6. Sources

- https://mujoco.readthedocs.io/en/stable/changelog.html
- https://github.com/google-deepmind/mujoco_warp
- https://mujoco.readthedocs.io/en/stable/mjx.html
- https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/sysid/README.md
- https://github.com/google-deepmind/mujoco/discussions/3238
- https://github.com/google-deepmind/mujoco_playground
- https://github.com/google-deepmind/mujoco_mpc/issues/389
- https://github.com/unitreerobotics/unitree_rl_mjlab (and issues #51, #12, #25)
- https://github.com/unitreerobotics/unitree_rl_gym , https://github.com/unitreerobotics/unitree_rl_lab
- https://github.com/unitreerobotics/unitree_mujoco , https://github.com/unitreerobotics/unitree_ros/issues/130
- https://support.unitree.com/home/en/H1_developer (Basic Services Interface, joint order)
- https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html
- https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html
- https://github.com/NVIDIA/GR00T-Dreams , https://arxiv.org/abs/2505.12705
- https://github.com/NVIDIA/Cosmos/blob/main/docs/reference/models.md
- https://github.com/isaac-sim/IsaacLab/discussions/7749
