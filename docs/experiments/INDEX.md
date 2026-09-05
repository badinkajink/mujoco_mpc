# Experiment pages

One row per page. Pages are **local files in this repo** — the user's standing
rule is that nothing lives only on claude.ai, so the local path is the canonical
copy and the artifact column is empty unless a page was explicitly published.

| Date | Page | Local file | Artifact URL |
|---|---|---|---|
| 2026-09-04 | Table height generalisation for the braced lean controller | `docs/lean/20260904-table_height_generalization.html` | — (local only, by request) |
| 2026-08-26 | What the lean schedule costs | `docs/lean/2026-08-26_schedule_cost.html` | — |

## Supporting code

| Page | Harness | Study scripts |
|---|---|---|
| Table height generalisation | `mjpc/lean_bench.cc` (CMake target `lean_bench`) | `studies/table_height/{sweep,analyze,render_video,make_page}.py` |
| Lean schedule cost | `mjpc/lean_bench.cc` | `studies/lean_sched/` |

Raw run outputs (`studies/*/runs/`) are gitignored: they are large and
regenerable. The figures and `summary.json` / `agg.json` a page depends on are
committed alongside the page under `docs/lean/media/`.
