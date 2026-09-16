# Experiment pages

One row per page. Pages are **local files in this repo**; the local path is the
canonical copy and the artifact column carries the claude.ai URL only when a
page was published there as well.

| Date | Page | Local file | Artifact URL |
|---|---|---|---|
| 2026-09-11 | Sampling-planner selection for the braced lean at the robot's plan rate (CEM / iCEM / MPPI / predictive sampling on the deploy joint gains, 5–167 plans/s, latency raw and compensated, plant-side mass/kp/friction mismatch ladders at 12 seeds with the σ link, commanded-target grid, renders; 2332 deploy-plant runs; supersedes the XML-plant version of the same page) | `docs/lean/20260911-planner_ablation.html` | https://claude.ai/code/artifact/6184e1b4-8fe2-421b-867a-e1c41cc4f77e |
| 2026-09-14 | Wrench prompting as a physical-parameter prior for contact MPC (assessment of the 2025 reject against OmniVIC / Exp-Force / CHORD, the proposed interface, E0–E4 program; first sim measurement on the UR5+MAGPIE wrench-actuated push: mass/friction belief error ×0.1–10 and predicted-contact-force caps 0.5–3 × μmg, 6 objects × 3 seeds, 270 episodes; Gemini prior probe on the 2025 photos) | `docs/wrench_prior/20260914-wrench_prior_direction.html` | https://claude.ai/code/artifact/a38de1f5-3bad-4add-a6a0-bcd27d0ca6b2 |
| 2026-09-14 (+2026-09-15) | Payload belief on the braced retrieval carry (strategy 11 = strat 27 + timeout_advance; true × believed payload 0–4 kg, 3 seeds, preload vs model-only arms, 96 lean_bench runs; every post-attach fall is a backward fall at release under a 4 kg belief; the mj_setConst qpos0 bug in --plant_mass_scale; 2026-09-15: 12 instrumented runs — the belief acts through a 5 cm phantom CoM offset, the brace force is unmoved, and the release push-off falls with a correct model too) | `docs/brace_payload/20260914-brace_payload_belief.html` | https://claude.ai/code/artifact/98401b8a-53a6-4556-90e1-f236fe199206 |

Pages from the table-height line live on branch `wxie/table-height` and are
indexed there (`docs/experiments/INDEX.md` on that branch).

## Supporting code

| Page | Harness | Study scripts |
|---|---|---|
| Payload belief on the braced retrieval carry (2026-09-14, +15) | `mjpc/lean_bench.cc` (`--payload_true/--payload_belief/--payload_phase/--bft_add`, `SetConstScratch`; `--plan_out`, belief columns, `--scenario_masses/--scenario_agg`), `planners/icem_dr` scenario mode, `lean.cc`/`Lean_H12_Magpie.xml` (`brace_force_target_add`, `brace_force_max`), `lean.h` slot 11, `strategies/h12_brace_payload.json` | `studies/brace_payload/{sweep,analyze,analyze_plan}.py`, `run_sweep2.sh`, `README.md`; results `studies/brace_payload/runs/{sweep1,sweep2}/` |
| Wrench prompting as a physical-parameter prior (2026-09-14) | Python predictive sampling over `mujoco.rollout` on the MJPC `UR5 Magpie` model (`build_cmake/mjpc/tasks/ur5/task_magpie.xml`), plant vs planner from one MjSpec, plate touch sensors as the object-force channel | `studies/wrench_prior/{ur5_push,sweep,analyze,make_tables,e0_probe}.py`, `README.md`; results `studies/wrench_prior/runs/sweep1/`, VLM probe `runs/e0_gemini_probe.json` |
| Sampling-planner selection (2026-09-11) | `mjpc/lean_bench.cc` (`--numeric`, `--state_out`, `--gains deploy`, `--log_hz`), planner knobs in `mjpc/planners/{sampling,mppi,cross_entropy,icem}/`, `agent_allocate_active_only` in `mjpc/agent.cc` | `studies/planner_ablation/{arms,sweep,analyze,paper_figs,persistence,report,tables,tables_rate,make_page,publish_prep}.py`, `score_rates.sh`, `campaign*.sh`, `HANDOFF.md`; paper figures in `studies/planner_ablation/paper_figs/` |
