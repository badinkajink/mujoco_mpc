# Filmstrips for Algorithm 1: one CEM planning iteration, rollout by rollout

Talk assets (2026-09-21) and the tooling that makes them. One planning iteration of the
sampling-based MPC in the lean paper (Algorithm 1: Center, Spread, rollouts + scoring,
Fold) is rendered from what the CEM planner actually held at one plan tick of
`lean_bench`: the centre spline and its rollout, the sampled Σ and ε(k), all N candidate
rollouts with their J(k), the elite set, and the folded spline rolled out from the same x₀.

Two pages, two moments:

| page | moment | primary set |
|---|---|---|
| `docs/lean/20260921-alg1_filmstrips.html` — https://claude.ai/artifact/Pu3Nv9GpTYfahMHZjqTaN6 | lean onset, t = 12.5–13.5 s (0.5–1.5 s into the lean rung; the transition the lean-onset falls come from) | `figs/cem_stdmin05_s1/t13.5_*` (σ floor 0.05, wide fan), `figs/cem_s1/t12.5_*` (deployed σ) |
| `docs/lean/20260921-alg1_brace_contact.html` — https://claude.ai/artifact/LVNXY3AwP3EGQxfyQm1dpH | brace contact, t = 24.0–24.5 s (the forearm pad landing, deployed settings) | `figs/cem_stdmin01_s10_dense/t24.27_*`, `figs/cem_stdmin01_s9_dense/t24.5_*` |

Every figure exists as PNG + PDF under `figs/<run>/`, with a `clean/` copy without title
text for slides. `figs/` and `runs/` are gitignored (`studies/*/figs/`, `studies/*/runs/`);
the pages' 1800 px JPEGs live in `docs/lean/media/alg1_filmstrips/` and are tracked.

## How to generate such an analysis

Five steps. Each is one script; each script's docstring has its own usage line.

### 1. Build the bench

```bash
cd mujoco_mpc && cmake --build build_cmake --target lean_bench -j 2
```

`mjpc/lean_bench.cc` has the dump hook: `--samples_out DIR --samples_at t1,t2,...`. At
the first plan tick at or after each listed plant time it writes one directory,
`DIR/tickNN_tT/`, and touches nothing else (flag absent = byte-identical bench). CEM only
(`agent_planner=5`); other planners ignore the flag. What is in a tick directory:

| file | content |
|---|---|
| `meta.json` | t_plan, t_state, phase and name, N, n_elite, horizon steps, dt, nu, nq, knots, std_min in effect, spp, seed, strategy, gains |
| `center.csv` | θ̄: the resampled nominal spline, one row per knot (`knot, t, u0..u26`) |
| `sigma.csv` | per parameter: `sigma_used` (the std actually sampled at this tick = max(refit std, std_min), read before the iteration overwrites it) and `sigma_next` (the refit the iteration produced) |
| `noise.csv` | ε(k): one row per (k, knot) |
| `candidates.csv` | θ(k) = θ̄ + ε(k), one row per (k, knot) |
| `scores.csv` | per k: rank, elite flag, total_return J(k) |
| `rollouts.csv` | 101 horizon steps × (k = −1 θ̄'s rollout, −2 the fold's, 0..N−1 the candidates): `t, cost, q0..q40` |
| `fold.csv` | θ after the fold (the elite mean), one row per knot |

The fold rollout is computed after `PlanIteration` on a scratch `Trajectory` with the
planner's own state and model, so the planner's internals are untouched.

### 2. Run the bench with dumps

The deployed setting is strategy 25, `--gains deploy`, `--spp 15` (33 plans/s),
`--numeric agent_planner=5` (CEM: the robot runs CEM, not the XML's iCEM), std_min 0.01.
One run is 3.7 GB RSS and ~48 s wall for 30 s of sim at 6 threads; go through resguard.

```bash
cd studies/alg1_filmstrips
~/.claude/bin/resguard.sh run --mem 5G --cpu 700 -- ../../build_cmake/bin/lean_bench \
  --task "Lean H12 Magpie" --strategy 25 --seed 1 --total_time 30 --threads 6 --spp 15 \
  --gains deploy --numeric agent_allocate_active_only=1 --numeric agent_planner=5 \
  [--numeric std_min=0.05] --out runs/<run>/metrics.csv --qpos_out runs/<run>/qpos.csv \
  --samples_out runs/<run>/samples --samples_at 5,12.5,13.5,16,20,25,29
```

When the moment of interest is an event whose time you do not know in advance (a
contact, a rung transition), dump densely and pick afterwards:

```bash
./run_contact_dumps.sh "0.01:1 0.01:9 0.01:10 0.02:0" 19 29 30   # std_min:seed pairs, from, to, total_time
```

writes `runs/cem_stdmin<SS>_s<SEED>_dense/` with a dump every 0.25 s (41 ticks, ~6 MB
each), two runs at a time. Planner noise is not seed-reproducible (thread timing re-rolls
it), so a run that braced once may not brace again; dump several seeds.

### 3. Survey the ticks

Two cheap surveys before rendering anything.

```bash
MUJOCO_GL=egl ./render_alg1.py --tick runs/<run>/samples/tickNN_tT --out /tmp/x --only X
```

prints the tick's numbers as JSON: J of θ̄ / best / worst / fold, and the end-of-horizon
spread of the pelvis and both hands across the N rollouts against θ̄'s rollout. Use it to
find the tick where the fan is widest.

```bash
MUJOCO_GL=egl ./contact_stats.py runs/<run>/samples --from 23 --to 26 [--json out.json]
```

walks every rollout of every tick in the window through `mj_forward` and prints, per
tick, how many candidates land the forearm pad inside the horizon (gap between the lowest
point of `left_forearm_pad` and the table face ≤ 3 mm), how many of them are elites, the
J range of landers against hoverers, and whether θ̄ and the fold land. The tick you want
is the one where the count is between 1 and N−1: some rollouts found the contact and some
did not, so the sort has something to sort.

Any other event can be surveyed the same way: `pad_gap_track` in `contact_stats.py` is
the template — a function of one qpos row that returns the quantity, applied along
`rollouts.csv`.

### 4. Render

```bash
MUJOCO_GL=egl ./render_alg1.py --tick runs/<run>/samples/tickNN_tT --out figs/<run>/<label> [--clean] [--contact]
```

| suffix | algorithm line | what it shows |
|---|---|---|
| `A_center` | 1  θ̄ ← Center(θ) | six frames of θ̄'s rollout (faint grey = x₀), its running cost, the spline θ̄ for eight joints |
| `B_spread` | 3–4  ε(k) ~ N(0,Σ), θ(k) ← θ̄ + ε(k) | Σ as a 3 knots × 27 joints heatmap of the std sampled at this tick; the N perturbed splines around θ̄ |
| `C3_fan_strip` | 5 | all N rollouts on one image, as a filmstrip at h = 0.25, 0.5, 0.75, 1.0 s, tinted by J(k) |
| `C1_rollouts_fan` | 5–6 | the fan at h = 1 s beside the sorted J(k) bars |
| `C2_rollouts_strips` | 5–6 | best / two mid-ranked / worst as filmstrips with running cost against θ̄'s |
| `D_fold` | 8  θ ← Fold | elites in the J bars, elite fan with the fold's rollout (red outline), elite splines → mean |
| `D2_fold_strips` | 8 | θ̄'s rollout over θ's rollout, same x₀ |
| `E_overview` | all | 2 × 2 summary |
| `G1_contact` (`--contact`) | 5–8 | detail camera on the pad at h = 1 s with pad and hand traces; pad clearance along every rollout; J bars with landers solid, hoverers hatched, elites shaded |
| `G2_contact_strip` (`--contact`) | 5 | the detail camera as a filmstrip, with the count of landed rollouts per frame |
| `F_sigma_compare` (`--compare A,B --labels ...`) | 5 | the fan at h = 1 s for several dumps side by side |

Conventions: viridis by rank, yellow = lowest J; θ̄ black, the fold red, elites shaded
orange, dropped candidates grey. Camera: side view of the sagittal plane (azimuth 90,
elevation −8, lookat (0.40, 0, 0.85), distance 2.7 m); the `--contact` detail camera sits
1.7 m from the pad at elevation −16. Ghost silhouettes are the robot re-coloured light grey,
masked by the segmentation render and tinted; θ̄ and the fold are 2–3 px outlines; traces
are body positions projected through the scene's own frustum (`Scene.project`, checked
against the target ball's segmentation centroid to 0.5 px). `--clean` drops all title text.
`--frames`, `--box`, `--fanbox`, `--dbox`, `--azimuth`, `--elevation`, `--lookat`,
`--distance` are the layout knobs. Each figure is one function `fig_<x>` in
`render_alg1.py`; a new figure type is a new function that takes the loaded tick and a
`Scene` (`draw_fan`, `filmstrip_row`, `step_line`, `Scene.body_path`, `Scene.pad_gap`
are the building blocks).

### 5. Page

```bash
./make_page.py contact      # or: filmstrips, or no argument for both
```

Each page is a config block in `make_page.py`: sections of (run, tick prefix, suffix,
caption). It downsizes the PNGs to 1800 px JPEGs under `docs/lean/media/alg1_filmstrips/`
and writes `docs/lean/20260921-alg1_<page>.html`. The numbers in the captions are typed
in from steps 3–4; the page is the record of which figure came from which run and tick.
Republish to the existing artifact URL (see `docs/experiments/INDEX.md`), never a new one.

## What the two pages found

**Lean onset (page 1).** At the deployed σ the 20 rollouts already end 25 mm apart at the
pelvis (54 max) and up to 98 mm at the hands after 1 s; J 78.7–123.1 at t = 12.5 s. At
σ floor 0.05 (top of the CEM basin) the fan is 103 / 217 mm and the sort visibly separates
rollouts that bow into the brace from ones that hang back; the fold's J (180.7) is below
every sample (best 192.4). A σ 0.03 run in its lean-onset fall shows 20/20 rollouts
toppling — a collapsed fan. Two things the pictures showed: the third knot sits at
h = 1.0 s and under the hold acts on the last 10 ms only, so nothing selects on it and it
random-walks under the refit (largest entry of Σ at every tick); and the running cost is
sawtoothed for the first 0.5 s and smooth after the second knot (not investigated).

**Brace contact (page 2).** With the deployed settings the pad is held 23–43 mm above the
slab through rung 1 with 0/20 rollouts landing it; the landing appears in the rollouts
within one or two ticks of the rung-2 transition at 24.0 s, and the plant loads the pad
0.4–0.9 s later. Seed 10 at 24.27 s: 4/20 land, and the three lowest-J candidates are
three of them (J 1540–1576 against 1603–1606 for every hoverer); θ̄ hovers at +17 mm, the
fold descends to +6 mm; one tick later all 20 land. Seed 9 at 24.51 s: 9/20 land, 5 of the
6 elites are landers, θ̄ hovers at +3 mm and the fold lands at h = 0.61 s. A σ 0.02 run
found the contact inside rung 1, 1.5 s before the schedule asked for it.

**About the bench (found while looking for the contact tick).** In the planner-ablation
runs on the deploy plant at 33 plans/s (`studies/planner_ablation/runs/gains_spp15`),
shipped CEM is scored complete in 10/12 seeds but the forearm carries load in only 3 of
those 10 (seeds 1, 9, 10: 117–147 N); the other seven hold the pad 10–30 mm up and walk
through the rungs on 12.0 s intervals, because the rung's contact gate accepts a believed
pad gap under `brace_contact_zmargin` = 80 mm (`lean.cc`). At std_min 0.02–0.05 every
completing seed braces (157–365 N); PS-hold 10/12, MPPI 11/12, iCEM 7/12. Completion and
brace are different counts at the deployed floor. Next measurements: re-score the ablation
with `f_forearm` > 20 N sustained 1 s in rungs 1–2 next to completion; check the robot logs
for measured brace force before the reach rung fired; `run_contact_dumps.sh "0.02:0 0.02:1
0.02:2"` to count first-landing ticks at σ 0.02.

## Files

| file | role |
|---|---|
| `render_alg1.py` | figures A–G from one tick directory; `--only X` prints the tick's numbers |
| `contact_stats.py` | per-tick, per-rollout pad landing survey over a window of dumps |
| `run_contact_dumps.sh` | dense dumps (every 0.25 s) for several (std_min, seed) pairs, two at a time through resguard |
| `make_page.py` | the two contact-sheet pages and their media |
| `runs/<run>/` | `metrics.csv`, `qpos.csv`, `bench.log`, `samples/tickNN_tT/`, `contact_*.json` (gitignored) |
| `figs/<run>/` | PNG + PDF, `clean/` without titles (gitignored) |
| `../../mjpc/lean_bench.cc` | the `--samples_out / --samples_at` dump (CEM only, default off) |
