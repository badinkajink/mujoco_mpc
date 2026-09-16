# brace_payload — a wrong payload belief while braced (strategy 11)

Started 2026-09-14, branch `wxie/brace-payload` (off `wxie/planner-ablation`
tip 42b423d3, which is `icra2026` 20f20f0e + the lean_bench tooling). The
humanoid half of the wrench-prior direction
(`docs/wrench_prior/20260914-wrench_prior_direction.html`).

Question: the H1-2 grasps an object at full reach while braced on the table
and carries it back. If the planner's belief about the object's mass is wrong
(the VLM prior), what happens to the carry — fall, stalled ladder, or nothing?

What was added (all default-off = byte-identical):

| Where | What |
|---|---|
| `lean.h` | strategy slot 11 = `h12_brace_payload` |
| `strategies/h12_brace_payload.json` | strat 27 with `timeout_advance: true` on rungs 2–8 (in the headless bench rung 8 misses its 0.08 tolerance and the time-limit path resets the ladder to stand_up mid-brace → backward fall at t=83 s, seed 0) |
| `Lean_H12_Magpie.xml` | numerics `brace_force_target_add` (N, added to every positive per-keyframe target) and `brace_force_max` (already read by lean.cc, never declared) |
| `lean.cc` | the `brace_force_target_add` term in the Brace Force residual and its phase snapshot |
| `lean_bench.cc` | `--payload_true M --payload_belief M --payload_phase 5 --payload_ramp 0.3 --bft_add X --bft_add_until 8`; plant mass ramps in on `right_magpie_gripper` at rung 5, planner mass at once; summary gains `payload_*`, `max_phase`, `peak_brace_carry`, `min_pelvis_carry`, `max_tilt_carry` |
| `lean_bench.cc` | **bug fix**: `mj_setConst(model, data)` leaves `data->qpos` at qpos0; `--plant_mass_scale` runs before this fix started the plant at base x = 0.000 instead of 0.189 with the seed perturbation and the object pose wiped (measured with `--qpos_out`). Now a scratch mjData. Affects the planner-ablation mismatch ladder (campaign 17). |

Run:

```bash
python3 sweep.py --out runs/sweep1 --arms P,M --jobs 2     # 4x4 grid x 3 seeds x 2 arms = 96 runs, ~3 min each
python3 analyze.py --run runs/sweep1
```

Arm P: belief sets the planner's payload AND preloads the brace by
K_LEVER·m̂·g (K_LEVER 2.0: hand 0.48 m and brace 0.24 m ahead of the toe).
Arm M: belief sets the planner's payload only.

## sweep2 (2026-09-15): the release mechanism, the belief-3 column, 6 seeds, the interval baseline

Added to the bench (default-off, byte-identical otherwise):

| Where | What |
|---|---|
| `lean_bench.cc` | log columns `com_edge_belief`, `cop_edge_belief`, `brace_belief`: the plant's state evaluated on the planner's model (same qpos/qvel/ctrl, `mj_forward` on a scratch mjData of the planner copy, then the task's `ComputeMetrics` and the left-arm table load) |
| `lean_bench.cc` | `--plan_out f.csv --plan_out_from_phase 8 --plan_out_stride 5`: after every plan iteration in phases ≥ 8, the planner's nominal trajectory (`BestTrajectory()`), every 5th step of the horizon, evaluated on the planner model (`*_belief`) and on the plant model (`*_true`) |
| `lean_bench.cc` | `--scenario_masses 0,2,4 --scenario_agg mean|max|min`: switches `agent_planner` to 8 (iCEM-DR) and, from the attach on, scores every candidate over one model copy per listed mass on the payload body; before the attach the set is {0} = plain iCEM |
| `planners/icem_dr/planner.{h,cc}` | scenario mode: `scenario_body_`, `scenario_mass_`, `scenario_agg_` (0 min = the DR default, 1 mean, 2 max); the ensemble is the nominal model plus the listed masses (`mj_setConst` re-run), no randomization |
| `sweep.py` | arms `D` (= M + plan dump), `Smean`/`Smax`/`Smin` (scenario, belief 0); `--trues`, `--beliefs`, `--scenario_masses`; every job runs through `~/.claude/bin/resguard.sh run` (systemd scope + watchdog; a closed launch gate waits and retries) |
| `run_sweep2.sh` | the five stages, `./run_sweep2.sh runs/sweep2 [stage ...]`; 2 jobs × 6 threads = 7.4 GB RSS, 14 threads, nothing else heavy alongside (the 2026-09-15 reboot was a third `lean_bench`) |
| `analyze_plan.py` | the release table (`runs/sweep2/release_table.json`) and `fig_release_belief.png`, `fig_release_plan.png` |

Stage 1 result (12 runs, 11 attached): the 4 kg belief moves the planner's
upper-body CoM 4.1–5.7 cm ahead of the true one (mean 5.0 cm) and leaves its
brace force within 2–5 N of the truth; the planner regulates the believed CoM,
so the true CoM sits 5 cm nearer the heels for the whole carry. Two correct-
model runs fell at the release (m0/b0 seed 0: the push-off overshoots the heels,
predicted by the planner's own nominal a second ahead; m4/b4 seed 1: the gripper
catches the table top for 3 s). Believed-0 column with sweep1: 1/27; believed-4:
7/27.

Stages 2–5 result (2026-09-16, 108 runs; merged tables in `runs/sweep2/summary_merged.json`,
`python3 analyze2.py`): by belief error, under-belief 0/59, +1 kg 0/25, +2 kg 3/22,
+3 kg 7/15, +4 kg 3/10, correct 5/38 (four in the 4 kg/4 kg cell); by column, believed
0–2 kg 2/101, 3 kg 5/29, 4 kg 11/39. Scenario MPC over {0,2,4} kg: mean 0/11 fell,
max 0/12, min 1/12; mean and max are 3× the rollouts, 4 s slower attach→release, and
the max aggregate ran out of the 130 s budget in 5/12 runs (342 N brace on 4 kg
carries vs 233 N under the 0 kg belief). Pre-attach floor 23/192.
