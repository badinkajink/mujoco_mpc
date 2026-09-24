# brace_friction — arm–table and foot–floor friction on the braced lean

Why the simulator does not reproduce the real robot's brace-slip → foot-slip →
collapse. Page: `../../docs/lean/20260924-brace_friction.html`.

## The two findings

1. `lean_bench --plant_friction_scale` scales `geom_friction` only, and the brace
   pads meet the slab through declared `<contact><pair>` elements whose
   `pair_friction` it never touches. Every "plant friction ×0.6/×0.4" row of the
   planner ablation ran with arm–table friction at μ = 1.0.
2. MuJoCo's regularised friction makes a loaded pad creep in proportion to the
   shear it carries (3–80 mm/s) rather than stick and break away, so the plant
   has no event matching the hardware failure.

## Files

| File | What it does |
|---|---|
| `sweep.py` | one `lean_bench` run per (friction cell, seed), through `resguard.sh run`; resumable, fsync per row |
| `run_batches.sh` | the batch queue in priority order |
| `analyze.py` | scores a batch's `*.contact.csv` into `scored.jsonl` + a per-cell table |
| `replay_ablation.py` | reconstructs contact forces for the 2026-09 planner-ablation corpus from its `--state_out` tracks |
| `page_numbers.py` | every number the page quotes, computed from the records |
| `figs.py` | the page figures |
| `render.py` | MP4 / stills with shear arrows and slip trails drawn on the robot |
| `make_page.py` | builds the HTML page |

## lean_bench flags added for this study

`--plant_table_mu` / `--plant_foot_mu` / `--planner_table_mu` / `--planner_foot_mu`
(sliding friction per surface, plant and planner separately, pairs included; the
planner copy is synced before these apply, so it keeps the XML's 1.0 unless told),
`--plant_table_stiff 1` (slab contacts → `solref 0.01`, `solimp 0.95 0.99 0.001`),
`--plant_table_dz` (plant slab offset from the planner's),
`--contact_out f.csv --contact_hz 500` (per-step contact record).

## Recipe

```bash
cmake --build build_cmake --target lean_bench -j 6
./run_batches.sh                                    # ~80 s per run, 2 jobs
./analyze.py runs/b1_sd02
./replay_ablation.py --dirs gains_spp15,mismatch_spp15_mu0.6,mismatch_spp15_mu0.4 \
    --arms all --out runs/replay_ablation.jsonl --tracks runs/replay_tracks
./figs.py --out ../../docs/lean/media/brace_friction --chain b1_sd02:t0.6_f0.3_s2
MUJOCO_GL=egl ./render.py --runs runs/b1_sd02 --tags t1_f1_s0,t0.6_f0.3_s0 \
    --t0 18 --out ../../docs/lean/media/brace_friction/vid_mu_compare.mp4
./make_page.py
```

Cells are `table_mu:foot_mu[:planner_table_mu:planner_foot_mu]`.
