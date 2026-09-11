# Experiment pages

One row per page. Pages are **local files in this repo**; the local path is the
canonical copy and the artifact column carries the claude.ai URL only when a
page was published there as well.

| Date | Page | Local file | Artifact URL |
|---|---|---|---|
| 2026-09-11 | Sampling-planner ablation on the braced lean (iCEM / CEM / MPPI / predictive sampling, 132 runs) | `docs/lean/20260911-planner_ablation.html` | https://claude.ai/code/artifact/6184e1b4-8fe2-421b-867a-e1c41cc4f77e |

Pages from the table-height line live on branch `wxie/table-height` and are
indexed there (`docs/experiments/INDEX.md` on that branch).

## Supporting code

| Page | Harness | Study scripts |
|---|---|---|
| Sampling-planner ablation (2026-09-11) | `mjpc/lean_bench.cc` (`--numeric`, `--state_out`), planner knobs in `mjpc/planners/{sampling,mppi,cross_entropy,icem}/`, `agent_allocate_active_only` in `mjpc/agent.cc` | `studies/planner_ablation/{arms,sweep,analyze,make_figs,tables,tables_rate,make_page}.py`, `queue*.sh` |
