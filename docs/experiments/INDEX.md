# Experiment pages

One row per page. Pages are **local files in this repo**; the local path is the
canonical copy and the artifact column carries the claude.ai URL only when a
page was published there as well.

| Date | Page | Local file | Artifact URL |
|---|---|---|---|
| 2026-09-11 | Sampling-planner selection for the braced lean at the robot's plan rate (CEM / iCEM / MPPI / predictive sampling on the deploy joint gains, 5–167 plans/s, latency raw and compensated, plant-side mass/kp/friction mismatch ladders at 12 seeds with the σ link, commanded-target grid, renders; 2332 deploy-plant runs; supersedes the XML-plant version of the same page) | `docs/lean/20260911-planner_ablation.html` | https://claude.ai/code/artifact/6184e1b4-8fe2-421b-867a-e1c41cc4f77e |
| 2026-09-14 | Wrench prompting as a physical-parameter prior for contact MPC (assessment of the 2025 reject against OmniVIC / Exp-Force / CHORD, the proposed interface, E0–E4 program; first sim measurement on the UR5+MAGPIE wrench-actuated push: mass/friction belief error ×0.1–10 and predicted-contact-force caps 0.5–3 × μmg, 6 objects × 3 seeds, 270 episodes; Gemini prior probe on the 2025 photos) | `docs/wrench_prior/20260914-wrench_prior_direction.html` | https://claude.ai/code/artifact/a38de1f5-3bad-4add-a6a0-bcd27d0ca6b2 |

Pages from the table-height line live on branch `wxie/table-height` and are
indexed there (`docs/experiments/INDEX.md` on that branch).

## Supporting code

| Page | Harness | Study scripts |
|---|---|---|
| Wrench prompting as a physical-parameter prior (2026-09-14) | Python predictive sampling over `mujoco.rollout` on the MJPC `UR5 Magpie` model (`build_cmake/mjpc/tasks/ur5/task_magpie.xml`), plant vs planner from one MjSpec, plate touch sensors as the object-force channel | `studies/wrench_prior/{ur5_push,sweep,analyze,make_tables,e0_probe}.py`, `README.md`; results `studies/wrench_prior/runs/sweep1/`, VLM probe `runs/e0_gemini_probe.json` |
| Sampling-planner selection (2026-09-11) | `mjpc/lean_bench.cc` (`--numeric`, `--state_out`, `--gains deploy`, `--log_hz`), planner knobs in `mjpc/planners/{sampling,mppi,cross_entropy,icem}/`, `agent_allocate_active_only` in `mjpc/agent.cc` | `studies/planner_ablation/{arms,sweep,analyze,paper_figs,persistence,report,tables,tables_rate,make_page,publish_prep}.py`, `score_rates.sh`, `campaign*.sh`, `HANDOFF.md`; paper figures in `studies/planner_ablation/paper_figs/` |
