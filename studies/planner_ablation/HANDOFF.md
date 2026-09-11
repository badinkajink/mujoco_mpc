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

## 4. In flight right now (`campaign2.sh`, log `runs/campaign2.log`)
1. **`runs/gains_spp15`** — `lean_bench --gains deploy`: the deploy node's KP/KV
   table (`h12_control_node.cc` KP[]/KV[]: arm kp 90/60/40/90/15, ankle roll 80;
   XML has arm 40, ankle roll 200) applied to plant AND planner model, at 33 Hz,
   6 seeds: cem, icem, ps_raw01_cubic, cem_ne2, cem_stdmin02, mppi λ=1. **This is
   the test of whether the bench-vs-robot gap is the gain table.** If cem comes
   back ≥5/6, every future bench run should use `--gains deploy` and the 167 Hz
   rows should be re-run with it. If not, next candidates: gravity feed-forward
   (`--gravity_ff` is 0 for the lean per the node's comments, so probably not),
   the estimator/latency compensation, the real actuator dynamics. Ask the user
   for a real-robot lowstate/lowcmd log of a strategy-25 lean: the executed
   target step per plan (this study's metric) can be computed from it directly
   and compared with the bench's 6.5 mrad.
2. `runs/rate_spp6` (83 Hz, batch R, 6 seeds), 3. `runs/rate_spp3` seeds 3–5.
Check with `cat runs/campaign2.log`; each sweep is resumable (`--out` same dir).
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
~2 h. Use `--gains deploy` if §4 says so.

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

## 6. Paper figures drafted so far (`paper_figs/`, PDF+PNG, IEEE column width)
`fig_rate` completion vs plans/s per update rule (6 seeds where available);
`fig_window` completion vs executed step per plan at 33 Hz, all arms;
`fig_elites` completion + step vs k at 33 Hz; `fig_floor` completion vs σ at 167
and 33 Hz, CEM vs PS; `fig_step` step per rung at 33 Hz; `fig_shipped` shipped vs
matched noise at 167 Hz. Palette: update rule → colour (elite mean #2a78d6,
argmin #eb6834, softmax #1baf7a), Wilson 95% bars, serif 8 pt. Regenerate with
`./score_rates.sh && ./paper_figs.py` after every wave.

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
