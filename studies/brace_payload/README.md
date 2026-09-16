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
