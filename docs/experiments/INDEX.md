# Experiment pages

One row per page. Pages are **local files in this repo** — the user's standing
rule is that nothing lives only on claude.ai, so the local path is the canonical
copy and the artifact column is empty unless a page was explicitly published.

| Date | Page | Local file | Artifact URL |
|---|---|---|---|
| 2026-09-05 | Pitch and reach gates on the braced-lean height window | `docs/lean/20260905-height_window_gates.html` | — (local only, by request) |
| 2026-09-05 | Brace posture retargeting across table heights | `docs/lean/20260905-brace_posture_retarget.html` | — (local only, by request) |
| 2026-09-04 | Table height generalisation for the braced lean controller | `docs/lean/20260904-table_height_generalization.html` | — (local only, by request) |
| 2026-08-26 | What the lean schedule costs | `docs/lean/2026-08-26_schedule_cost.html` | — |

## Supporting code

| Page | Harness | Study scripts |
|---|---|---|
| Height window gates | `mjpc/lean_bench.cc` (`--numeric`) | `studies/table_height/{probe_pitch,probe_reachset,probe_basin,analyze_gates,sweep_basin,make_page_gates}.py`, `publish_gates.sh` |
| Brace posture retargeting | `mjpc/lean_bench.cc` (`--pose_track`, `--numeric`) | `studies/table_height/{retarget,probe_ik,probe_armreach,probe_static,analyze_pose,render_pose,sweep_ab,make_page_pose}.py` |
| Table height generalisation | `mjpc/lean_bench.cc` (CMake target `lean_bench`) | `studies/table_height/{sweep,analyze,render_video,make_page}.py` |
| Lean schedule cost | `mjpc/lean_bench.cc` | `studies/lean_sched/` |

Raw run outputs (`studies/*/runs/`) are gitignored: they are large and
regenerable. The figures and `summary.json` / `agg.json` a page depends on are
committed alongside the page under `docs/lean/media/`.
