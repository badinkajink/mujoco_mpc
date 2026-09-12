# Planner ablation — handoff (written 2026-09-11 15:05 MDT, mid-campaign)

Branch `wxie/planner-ablation` off `icra2026` 20f20f0e. Everything below is on disk
in this repo; raw runs are gitignored under `studies/planner_ablation/runs/` on
this box. Read this before touching anything: the story changed twice on
2026-09-11 and the page (`docs/lean/20260911-planner_ablation.html`) still tells
the overnight version (§3 below says what it gets wrong).

## 1. Ground truth from the user (2026-09-11)

- **The real robot runs CEM (planner 5) on strategy 25 at 33 plans/s, "100%".**
  Not iCEM (the XML's `agent_planner=7`), and not 45–67 Hz (what the XML/deploy
  comments say). CEM is the reference arm; 33 Hz is the reference rate.
- The bench at 33 Hz does **not** reproduce the robot: CEM k=6 σ=0.01 is **1/6**
  there (five backward falls 2–3 s into the lean rung). Explaining that gap is
  the open problem; the deploy-gains test (§4) is the first candidate.

## 2. What is established (all at the compiled 0.985 m slab, strategy 25, 6 threads)

Plan rate: `--spp` counts 2 ms **plant** steps. spp 3 = 167 plans/s (every
overnight number), 6 = 83, 10 = 50, 15 = 33 Hz (the robot). Memory
`lean-bench-plan-rate` has the details.

### 2a. 167 plans/s (overnight, 3 seeds/arm, `runs/wave*`, scored `runs/summary.json`)
- Shipped numerics: iCEM 3/3, CEM 3/3, PS 0/3, MPPI 0/3 (PS/MPPI collapse onto the
  slab in the stand rung). Cause: PS/MPPI noise = `sampling_exploration` ×
  ½ ctrlrange = 0.03–0.36 rad/joint; CEM/iCEM noise = `std_min` = 0.01 rad, and
  the CEM elite variance collapses to that floor in 0.3 s (measured, `cem_std_mean`).
- Noise at 0.01 rad in the same units (`sampling_noise_raw=1`): PS 3/3 (cubic),
  2/3 (zero-order); MPPI 9/9 at λ = 0.1/1/10. Every elite-mean arm 39/39; argmin
  arms 12/15 (three backward stand-back falls, all 20-sample zero-order arms).
- PS noise band 0.004–0.03 rad works; 0.10 fails (reach gate never met); shipped
  fails (stand). iCEM std_min 0.03 3/3, 0.10 1/3.
- No iCEM component matters: α=0, keep=0, both (=CEM), frozen variance, k=2, k=10
  all 3/3; k=20 (no selection) 0/3 in 6 s; α=0.95 2/3 (first-knot std 0.31σ).
- Budget/horizon: N=8/k=2 3/3, N=40/k=12 3/3, PS N=40 3/3, horizon 0.5 s 3/3,
  0.3 s 0/3.

### 2b. Plan-rate campaign (2026-09-11 afternoon, 6 seeds/arm, per-plan logging)
Dirs `runs/rate_spp{15,10,6,3}`, scored `runs/summary_rate_spp{15,10,6,3}.json`
via `./score_rates.sh` (later dirs override earlier for the same arm+seed).

**33 Hz (`rate_spp15`, done, 108 runs):**

| arm | done | executed step/plan in lean, mrad | how it fails |
|---|---|---|---|
| icem | 1/6 | 3.0 | backward at lean onset t=14–15 |
| cem (k=6, σ=0.01) | 1/6 | 6.5 | same |
| cem_ne10 | 0/6 | 4.9 | same |
| cem_ne20 | 0/6 | 4.3 | stand + lean |
| mppi λ=1 | 1/6 | 8.3 | lean onset |
| mppi λ=0.1 | 3/6 | 8.5 | lean onset |
| mppi λ=10 | 2/6 | 8.8 | lean onset |
| ps_raw01_zero | 1/6 | 8.4 | lean onset |
| ps_raw01_cubic | 0/6 | 9.3 | lean onset |
| **cem_ne1** | **4/6** | 9.8 | lean onset ×2 |
| **cem_ne2** | **4/6** | 9.8 | lean onset ×2 |
| **cem_stdmin02** | **4/6** | 10.8 | lean onset ×2 |
| cem_stdmin03 | 3/6 | 16.7 | lean onset ×3 |
| ps_raw02_cubic | 1/6 | 17.9 | stand ×2, lean, reach |
| ps_raw03_cubic | 1/6 | 24.9 | stand ×2, lean ×2 |
| cem_stdmin05 | 0/6 | 27.8 | lean onset |
| ps_raw05_cubic | 0/6 | 40.6 | stand ×4 |
| cem_stdmin10 | 0/6 | 52.7 | lean/reach |

So at 33 Hz there is a **window on the executed step per plan, ~8–15 mrad RMS**:
below it the plan cannot follow the lean ramp (CoM retreats ~4 cm at the onset,
CoP saturates at the heel, backward fall at t=14–15 s); above it the servo noise
topples the stand. Shipped CEM sits below the window; k=1–2 or σ=0.02 moves it in.
**"Elite averaging keeps the step small" is the wrong direction at 33 Hz** — the
sentence I proposed on the overnight data is refuted by this wave.

**50 Hz (`rate_spp10`, done):** cem 4/6, icem 4/6, mppi λ=1 3/6, ps_cubic 1/6,
ps_zero 5/6, cem_stdmin03 3/6, ps_raw03_cubic 0/6. n=6 intervals overlap.

**83 Hz, 167 Hz seeds 3–5:** queued in `campaign2.sh` after the gains test.

### 2c. The lean-onset mechanism (from the traces, 33 Hz)
On entering rung 1 the CoM first moves back ~4 cm as the trunk starts to bow
(`com_beyond_foot_edge` −0.15 → −0.20), the CoP pins at the heel (−0.20); a
planner moving the targets ~10 mrad per 30 ms recovers by t=13.5 s, one moving
3–7 mrad does not and the robot is on its back at 14.5 s. `analyze.py` now
records `lean_onset_min_com_early` (min com_beyond in the first 1.2 s of the
lean; completing median −0.174, lean-onset falls median −0.244, 44/61 falls below
the completing p10 of −0.222) and `lean_onset_cop_sat_frac`. These are the dense
signals for §5.

## 2d. Deploy plant, 33 Hz: the spline is a component (cubic is upstream MJPC's sampling-planner default, 734bda29; CEM/iCEM hard-code the hold) (`runs/gains_spp15`, complete 258/258 at 17:18)
Spline × update rule at σ = 0.01 rad, 6 seeds:

| update rule | cubic | hold |
|---|---|---|
| argmin (PS) | 0/6 | 6/6 |
| softmax λ = 0.1 (MPPI) | 0/6 | 6/6 |
| softmax λ = 1 (MPPI) | 1/6 | 4/6 (2 stalls) |
| elite mean k = 6 (CEM) | 1/6 | 5/6 |
| iCEM (AR(1) α 0.7 + 2 kept elites) | **6/6** | 6/6 |

Shipped PS (cubic, 0.03–0.36 rad) 0/6 and MPPI 0/6 fall in the stand within
10–45 s; the one-knob "exploration 0.01" versions (0.003–0.03 rad, cubic) 0/6.
Every cubic failure is the backward drift at the lean onset: `com_beyond_foot_edge`
−0.13 → −0.28 m over 1.2 s, CoP pinned at the heel (−0.21), ICP behind the foot
by 14.0 s, fall at 14.4–14.9 s (traces of ps_raw01_cubic s1, cem_cubic s2; the one
hold-CEM failure, cem s0, is the same trace). The executed step per plan does
not separate cubic from hold (settled stand 6–12 s: PS cubic 7.3 vs hold 3.9 mrad,
CEM 6.3 vs 5.6, iCEM 2.6 vs 2.7; cem_ne2 at 9.9 completes 6/6).
500 Hz probe (`runs/probe500`, PS seed 1, cubic vs hold): the cubic command moves
within the plan interval at 70–90 mrad/s RMS over joints (2–2.5 mrad per 30 ms)
plus a 6–10 mrad jump at each plan; the hold has no within-plan motion and a
1.9 mrad jump in the stand. The ramp is not undone by the next jump
(corr +0.01…+0.03), so it is not a sawtooth. Working mechanism: with a hold
the perturbation of the first knot is executed for 0.33 s in the rollout (hold
knots at t, t+0.33, t+0.67; cubic knots at t, t+0.5, t+1.0), so the immediate
action is selected on; with a cubic and white knot noise the
first-knot perturbation is transient and the immediate action follows a curve
set by knots chosen for 0.5–1 s ahead. iCEM's colored noise ties the first
knot to the later ones, which would explain its immunity; batch Y
(`icem_a0_cubic` white noise + memory, `icem_keep0_cubic` colored noise, no
memory, and their hold versions) splits that, queued at 33 Hz in campaign8.
The knot-spacing test (6 knots, or horizon 0.5 s with 3 knots, cubic PS) is
the next falsifier: if the cubic failure is the transient first knot, denser
knots should rescue it.
Update-rule reading at the hold and 0.01 rad: every rule completes; the σ
basin is CEM 0.005–0.05 all ≥ 4/6, PS-hold 0.01–0.02, MPPI-hold 0.01–0.02;
k = 1, 2, 6, 10 all ≥ 5/6, k = 20 (no selection) 0/6; N = 8 hurts CEM (2/6)
and PS (2/6), not MPPI (4/6).

## 2e. 167 Hz on the deploy plant (`runs/gains_spp3`, 78 runs, 20:21) and the spline mechanism (batch Y, 33 Hz)
167 plans/s, 6 seeds: cem 6/6, icem 6/6, shipped ps 0/6, shipped mppi 0/6 (both
fall or collapse in the stand), ps_raw01_zero 6/6, ps_raw01_cubic 4/6,
ps_scaled01_cubic 6/6, mppi_raw01_zero_l1 6/6, mppi_raw01_cubic_l1 2/6,
mppi_raw01_cubic_l0.1 5/6, mppi_scaled01_cubic 5/6, cem_cubic 6/6, icem_cubic
4/6. So the noise-units failure is rate-independent and the cubic failure is a
rate × spline interaction (cubic rows: 4–6/6 at 167, 0–1/6 at 33; iCEM 6/6 at
both).
Batch Y at 33 Hz (6 seeds): icem_a0_cubic (white knot noise, elite memory kept)
1/6; icem_keep0_cubic (AR(1) α 0.7 noise, no memory) 6/6; icem_a0 (hold, white)
6/6; icem_keep0 (hold, colored) 6/6; ps_raw01_cubic_k6 (cubic, 6 knots 0.2 s
apart) 0/6; ps_raw01_zero_k6 (hold, 6 knots, first knot held 0.167 s) 2/6.
Reading: the colored noise is what makes iCEM immune to the cubic, not the
memory; denser knots do not rescue the white-noise cubic; and the hold itself
fails when its first knot is held for 0.167 s instead of 0.33 s. One statement
covers all six: the sampled perturbation of the immediate action has to persist
for roughly 0.3 s of the rollout to be selected on — a hold does it by
construction (0.33 s at 3 knots), AR(1) noise does it by correlating the knots
(ρ = 0.7 between knots 0.5 s apart), white-noise cubic (decorrelated at the next
knot, 0.5 s, and moving from t = 0) and a 6-knot hold (0.167 s) do not. Batch Z (20:49): ps hold 2 knots 5/6, ps cubic 2 knots **6/6**, icem_a03_cubic
5/6, icem_a095_cubic 0/6 (all six fall in the initial stand-up at 4.4–5.0 s:
the cold-started AR(1) at α 0.95 gives the first knot std 0.31 σ = 3 mrad, an
amplitude failure, not persistence). `persistence.py` computes the correlation
time τ_p of δu(τ) with δu(0) over the horizon with MJPC's interpolation/slope
rules: cubic-6 0.11 (0/6, 0/6), hold-6 0.17 (2/6), cubic-3 white 0.28 (0/6 PS,
1/6 CEM, 1/6 iCEM-a0, 1/6 MPPI), linear-3 0.32 (3/6), hold-3 0.34 (6/6, 5/6,
6/6, 4/6), cubic-3 AR0.3 0.45 (5/6), hold-2 0.50 (5/6), cubic-2 0.62 (6/6),
cubic-3 AR0.7 0.66 (6/6, 6/6), hold-3 AR0.7 0.65 (6/6, 6/6). **A step at
≈ 0.3 s across four planners — `fig_persist`.** The threshold is presumably
the plan interval's (≈ 10 intervals at 33 Hz): at 167 Hz cubic-3 completes
4–6/6. Prediction for the 50 Hz block: cubic-3 (0.28 s vs a 0.2 s threshold)
completes ≥ 4/6 (`ps_raw01_cubic`, `cem_cubic`, `mppi_raw01_cubic_l1` in
`runs/gains_spp10`). Batch W (23:11): 4-knot cubic 0/6, 4-knot linear 0/6, 5-knot hold 1/6,
**4-knot hold 5/6** — the last one breaks the τ_p ordering (0.25 s completes,
cubic-3 at 0.28 fails). The statistic that orders all 20 arms is **t80 = the lag
at which the autocorrelation of δu falls below 0.8** (`persistence.t_below`):
≤ 0.20 s → 0–2/6 (cubic-6 0.08, cubic-4 0.13, linear-4 0.15, hold-6 0.17,
cubic-3 0.19, hold-5 0.20), ≥ 0.22 s → 5–6/6 (linear-3 0.22 is the edge at
3/6; AR0.3 cubic 0.22 5/6; hold-4 0.25 5/6; AR0.7 cubic 0.28 6/6; hold-3 0.34;
cubic-2 0.43; hold-2 0.50), α 0.95 cubic 0.34 → 0/6 for amplitude. Reading:
selection needs the perturbation still applied at ≈ 0.2 s ≈ 7 plan intervals,
not a long total correlation. `fig_persist` uses t80. 50 Hz block (23:01):
cubic-3 ps 4/6, cem 3/6, mppi λ1 4/6, λ0.1 3/6 — graded with rate as
predicted, not a threshold measurement; the W arms at 50/167 Hz would be.

## 3. What the published page gets wrong now
`docs/lean/20260911-planner_ablation.html` (artifact 6184e1b4…) was written on the
overnight data. Its 167 Hz sections stand. Its §"At the deploy node's plan rate"
and the Result paragraph say iCEM 2/3 at 33 Hz and "the elite mean's smaller
step survives more often" — superseded by §2b (n=6: 1/6, and the step is too
*small*). Its title-level framing "what separates iCEM from PS" should become
"CEM at the robot's plan rate". Regenerate after the campaign:
`./score_rates.sh && ./paper_figs.py`, then rewrite `page_body.html` §"deploy
rate" and Result, `./make_page.py … --out ../../docs/lean/20260911-planner_ablation.html`,
copy `paper_figs/*.png` to `docs/lean/media/planner_ablation/`, republish to the
same artifact URL (listed in `docs/experiments/INDEX.md`).

## 3b. RESULT of the deploy-gains test (final, 36/36, 15:08 MDT; `runs/summary_gains_spp15.json`)
`--gains deploy` at 33 Hz, 6 seeds:

| arm | complete | fell | stalled | t_complete med (s) | onset margin med (m) | brace peak med (N) | seated_frac med |
|---|---|---|---|---|---|---|---|
| cem (k=6, σ 0.01) | 5/6 | 1 (s0, lean onset) | 0 | 41.6 | −0.155 | 50 | 0.03 |
| icem | 6/6 | 0 | 0 | 44.2 | −0.157 | 119 | – |
| cem_ne2 | 6/6 | 0 | 0 | 44.1 | −0.152 | 144 | – |
| cem_stdmin02 | 5/6 | 1 (s1) | 0 | 44.3 | −0.146 | 222 | – |
| mppi_raw01_zero_l1 | 4/6 | 0 | 2 (s2 at rung 6, s4 at rung 5) | 49.3 | −0.139 | 154 | 0.46 |
| ps_raw01_cubic | 0/6 | 5 (4 at rung 1, 1 at rung 2) | 1 (rung 2) | – | −0.143 | 41 | – |

**The bench-vs-robot gap was the gain table.** With the robot's joint PD the
bench reproduces the robot (CEM works at 33 Hz) and predictive sampling still
fails at the same noise and rate. MPPI's two failures are stalls in the
stand-back rungs, never falls, and MPPI rests on the table more (seated_frac
0.43–0.63 vs CEM 0.00–0.31; brace peak 88–211 N vs 2–120 N) — a candidate
differentiator from the task/sensor channels; check it on the full grid.
Consequences:
- Every earlier bench number (§2a, §2b) is on the XML gains and is a
  *different plant*; the 33 Hz window in §2b is a property of that plant. The
  decisive rows are being re-taken on `--gains deploy` by `campaign3.sh` (§4).
- Mechanism to check on the traces: with kp 90 the bracing arm swings forward
  faster, so the CoM does not retreat at the lean onset (compare
  `lean_onset_min_com_early` between `runs/rate_spp15` and `runs/gains_spp15`
  for the same arm+seed).
- Consider adding the deploy KP/KV to the XML actuators (or an `<include>`)
  so the planner model and the bench match the robot by default — coordinate
  with Allen; `PatchActuators` in the node would then be a no-op.

## 3c. FLAG: the task XML and the robot run different joint gains
Verified 2026-09-11 by loading `build_cmake/mjpc/tasks/humanoid_bench/lean/Lean_H12_Magpie.xml`
with `mujoco` and comparing `actuator_gainprm`/`biasprm` with `KP[]/KV[]` in
`mjpc/deploy/h12_control_node.cc` (the table `PatchActuators` writes into the
planner and latency models at node start). KV is identical everywhere. KP:

| joint | XML kp | deploy kp |
|---|---|---|
| ankle roll (both) | 200 | 80 |
| shoulder pitch (both) | 40 | 90 |
| shoulder roll (both) | 40 | 60 |
| shoulder yaw (both) | 40 | 40 |
| elbow (both) | 40 | 90 |
| wrist roll/pitch/yaw (both) | 40 | 15 |
| legs, hips, knees, ankle pitch, torso | = | = |

The deploy node's comments date the arm raises (2026-08-08 shoulder 30→60,
2026-08-21 elbow 40→90, 2026-08-22 shoulder pitch 60→90) and the ankle-roll
value; the XML was never updated to match. **Anything that runs the planner
without going through `h12_control_node` is on the XML plant**: the `mjpc` GUI,
`lean_bench` without `--gains deploy`, any headless scorer that loads the task
XML directly. The RoboCasa twin is on the deploy gains (the plant PD comes from
lowcmd kp/kd, the planner is patched by the node). The bench shows the two
plants differ in outcome at 33 Hz (shipped CEM 1/6 on XML vs 5/6 on deploy),
so a sim-only evaluation on the XML gains at the robot's plan rate does not
predict the robot. Told the user 2026-09-11; they will check whether the
table-height evals (student, `wxie/table-height`) went through the node or the
XML. Fix options: (a) put the deploy table in the XML actuators (Allen's file);
(b) an `<include>` of a gains file both the node and the XML read; (c) at
minimum, `--gains deploy` in every bench script and a loud note in the task
README.

## 4. In flight right now (`campaign4.sh`, log `runs/campaign4.log`; campaign3.sh killed 15:22 so batch X could go in first)
`campaign2.sh` was killed after `gains_spp15` finished (its 83 Hz and 167 Hz
rows would have been on XML gains). `campaign4.sh` (same list as campaign3.sh
with batch X added at 33/50/167 Hz; waits for campaign3's 33 Hz sweep, which
kept running) re-takes everything on
`--gains deploy`, in priority order, each sweep resumable (`--out` same dir):
1. `runs/gains_spp15` (33 Hz): batches R,A,S,K,T,N, 6 seeds — 150 new runs
   (~2.5 h). This is the 33 Hz slice of the §5 basin study plus the shipped
   ps/mppi and the k/λ/N axes. Log `runs/gains_spp15b.log`.
2. `runs/gains_spp3` (167 Hz): R + A (shipped four + matched PS/MPPI), 6 seeds,
   42 runs (~2.5 h). Log `runs/gains_spp3.log`.
3. `runs/gains_spp10` (50 Hz): R,S,K,T, 6 seeds, 138 runs (~3 h).
4. `runs/gains_spp6` (83 Hz): R, 6 seeds, 30 runs (~0.8 h).
5. `runs/gains_spp3` (167 Hz): S at seeds 0–2, 36 runs (~2 h). Log `runs/gains_spp3b.log`.
Batches (arms.py): S = σ ∈ {0.005, 0.01, 0.02, 0.03, 0.05} × {cem std_min, ps
raw cubic, mppi raw zero λ=1}; K = cem n_elite {1,2,10,20}; T = mppi λ {0.1,10};
N = {8, 40} trajectories for cem (k=N×0.3), ps cubic, mppi λ=1.
`sweep.py` resume was broken (seed stored as a string by the summary parse, so
`(arm, seed)` never matched) — fixed 15:05; the summary no longer overwrites
`seed`/`wall_s`. Check with `cat runs/campaign3.log`; score each dir with
`./analyze.py --runs runs/gains_sppN --out runs/summary_gains_sppN.json`.
`score_rates.sh` still points at the XML-gains dirs — repoint it at
`runs/gains_spp{3,6,10,15}` before regenerating `paper_figs.py`.
Box policy: 2 jobs × 6 threads, CPUQuota 550%, nice 15 while the user is at the
desk. **Never `pkill -f <script>` from the tool shell — it matches the tool's own
command line and kills the shell (exit 144); use the PID.**

## 5. Proposed next experiment: configuration sensitivity ("basin") per planner
Motivation (user, 2026-09-11): if the signals are dense and separable, characterise
each planner's sensitivity to its configuration; a wider admissible basin is
itself an argument for CEM.

**Common axes (same units for every planner, so the basins are comparable):**
σ ∈ {0.005, 0.01, 0.02, 0.03, 0.05} rad (`std_min` for CEM/iCEM, raw
`sampling_exploration` for PS/MPPI) × plan rate ∈ {33, 50, 167} Hz (spp 15, 10, 3)
× N ∈ {20} (add {8, 40} at 33 Hz only). Planner-specific: CEM k ∈ {1, 2, 6, 10};
MPPI λ ∈ {0.1, 1, 10}; PS spline ∈ {zero, cubic}. Three planners (CEM k=6,
PS argmin cubic, MPPI λ=1) × 5 σ × 3 rates × 6 seeds = 270 runs ≈ 6 h (33/50 Hz
runs are 1–2 min, 167 Hz 5–6 min, 2 jobs); the k and λ axes at 33 and 50 Hz add
~2 h. All of it runs on `--gains deploy` (campaign3.sh, §4).

**Outcomes, from coarse to dense (all already in `analyze.py` per run):**
1. `outcome` ∈ {complete, stalled, collapsed, fell}, `max_phase`, `t_complete`.
2. `lean_onset_min_com_early` (m, continuous, predicts the 33 Hz failure),
   `lean_onset_cop_sat_frac`.
3. `jitter_phase[rung]` (executed step per plan, mrad RMS; log per plan with
   `--log_per_plan`), `cost_phase_mean[rung]`, `brace_peak_N`,
   `brace_median_reach_N`, `reach_min_mm`, `seated_frac`, `rung_dur[rung]`.

**Analysis:** per planner, a (σ × rate) heat map of completion fraction with the
admissible region = cells at ≥ 4/6 (state the rule before running); basin size =
number/area of admissible cells; basin width in σ at the robot's rate. For the
dense channels, fit `lean_onset_min_com_early` and the executed step as
functions of (σ, rate, k) — they should collapse onto one curve in the executed
step per plan (the window in §2b), which would be the paper's mechanism figure:
one x-axis (step per plan), every planner and configuration on it, the window
marked. Sensitivity = the slope of completion (or of onset margin) against log σ
at the robot's rate; a flat top = wide basin.

**Figures to produce (in `paper_figs.py`, extend):** `fig_basin` (3 heat maps
side by side, sequential single-hue, admissible cells outlined), `fig_window`
(already there: completion vs step/plan; extend to all rates with the rate as
marker fill), `fig_rate` (already there).

## 6. Paper figures (`paper_figs.py`, output `paper_figs/`, PDF+PNG, IEEE column width)
Target set for the paper (user 2026-09-11: "fewer and denser"): three figures.
1. **fig_ladder** — one row per configuration from shipped PS/MPPI to iCEM,
   three switch columns (noise std, spline, update rule; the cell that changed
   from the row above is set in ink), completion at 33 plans/s (filled) and
   167 (hollow) with Wilson 95 %. Answers "which component".
2. **fig_basin** — σ × plan-rate completion grid per update rule (cells
   ≥ 2/3 outlined = the basin), plus CEM k × rate, MPPI λ × rate, N × rule at
   33 Hz. Answers "how sensitive is each rule to its configuration". Basin
   counts are written to `paper_figs/basin.json`.
3. **fig_rate** (completion vs plans/s per rule) with **fig_window_all**
   (completion and the lean-onset CoM margin against the executed step per
   plan, every arm at every rate) as the mechanism panel, if the deploy data
   collapse onto the step axis; if not, fig_rate alone.
Older candidates kept in the script: fig_elites, fig_floor, fig_step,
fig_shipped, fig_window (33 Hz only). `./score_rates.sh && ./paper_figs.py`
regenerates everything from `runs/summary_deploy_spp{3,6,10,15}.json`;
`--gains xml` draws the XML-plant set into `paper_figs_xml/` for the record.
Palette: update rule → colour (elite mean #2a78d6, argmin #eb6834, softmax
#1baf7a; gray #9a9a96); serif 8 pt; text in ink tokens, never a series colour.

## 7. Files
- Harness: `mjpc/lean_bench.cc` (`--numeric`, `--state_out`, `--log_hz`, `--gains
  deploy`), knobs in `mjpc/planners/{sampling,mppi,cross_entropy,icem}/planner.cc`
  and `mjpc/agent.cc` (`agent_allocate_active_only`), numerics block in
  `Lean_H12_Magpie.xml` before the first `residual_`.
- Study: `arms.py` (44 arms, batches A–E, R, L), `sweep.py` (`--spp --gains
  --log_per_plan`), `analyze.py`, `score_rates.sh`, `paper_figs.py`,
  `make_figs.py`/`tables.py`/`tables_rate.py`/`make_page.py`/`page_body.html`
  (the report page), `queue*.sh`, `campaign_rate.sh`, `campaign2.sh`.
- Memory files: `planner-noise-convention`, `lean-bench-plan-rate`,
  `mjpc-sweep-cpu-budget` (double-launch and pkill traps).
