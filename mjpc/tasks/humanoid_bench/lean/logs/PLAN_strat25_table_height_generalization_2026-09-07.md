# Strat 25 across table heights — what must change, what generalizes

Session "h12" · 2026-09-07 · status: research done, Tier-0 ready to run
Baseline frozen at surface **0.985 / near edge 0.450** (world), snapshot
`logs/config_snapshots/cfg_0985_preheightsweep_20260907_052355/`.

---

## 1. What is actually height-dependent — audited, not guessed

| Layer | Height-dependent? | Why / evidence |
|---|---|---|
| `lean.cc` residuals | **NO** | every table quantity is read at runtime from the compiled `table_top_collision` geom (`data->geom_xpos` + `model->geom_size`): brace target, near-edge x, face z, side-swing via point, Body-Table-Clearance barrier. No recompile for a height change. |
| `reach_target_table` (strat 25's targeting rung) | **NO** | convention is `[depth in from near edge, lateral, height above face]` — table frame, auto-follows the slab (lean.cc:771-795). |
| numerics: `brace_target_inset`, `brace_lat_inset`, `table_clear_margin`, `brace_press_depth`, `reach_arm_hgate`, `brace_seat_proxy`, `servo_max_*` | **NO** | all relative offsets from the slab. |
| tag bundle / table anchor (real robot) | **NO** | tag offsets are surveyed in the TABLE frame; the bundle rides up with the table. ⚠ but the anchor is xy+yaw only (mode 2), so the *planner model's* table z is still an operator step. |
| planner table: body pos z + **8 leg geoms** | **YES** | `surface = body_z + 0.445`. Legs must span floor→slab underside or they hang in air — they are COLLISION geoms, so every clearance verdict silently changes. |
| twin table (`h1_mujoco/.../scene_handless_magpie_table.xml`) | **YES, and it is a separate file with a different convention** | twin: `surface = geom_z + 0.055`, body at z=0. `lean_set_table.py` did **not** touch it → silent parity break. Closed this session (§4). |
| `object_anchor` / `target` mocap z | YES | z = surface + 0.05. |
| `reach_target` numeric | partly | supplies x,y during brace phases; its z is overridden by `table_face_z + 0.30`. |
| **`forearm_brace_lean` keyframe** | **YES — this is the parameter** | base z 0.964, bow +25.9°, whole arm chain solved for 0.985. |
| `forearm_brace_release`, `standback_r1/r2/r3` | derived | built off lean by the documented 0.60 / 0.38 / 0.18 delta-lerp. |
| `stand_up`, `standback_r4` | NO | they are stands. |
| cost weights | **unknown** | the 1.09 campaign needed Pelvis Forward 100→300 / Posture 80→40 to fight the jackknife. This is the risk to the clean claim. |

**Candidate generalization claim:** the whole strategy is parameterized by
**one keyframe + two scalars (surface H, near-edge x)** — no code, no weights.
Tiers 0-3 below are the test of that claim.

---

## 2. Fresh offline measurement (this session)

`reach_vs_table_height.py`, feet-anchored at the MEASURED stance
(`phase_snaps/0_stand_up.npy`), walking the frozen stand → `forearm_brace_lean` line:

| quantity | value |
|---|---|
| surfaces the FROZEN ladder can brace-land on | **1.090 (s=0, upright) … 0.980 (s=1, the keyframe)** |
| forward reach past toes, peak | **683 mm at surface ≈ 0.987** |
| reach at 1.000 / 0.955 | 675 mm / 511 mm |
| CoM leaves the support polygon (brace becomes load-bearing) | below ≈ **0.955** |

⇒ with literally nothing changed, the strategy has geometric support only over
**≈0.98–1.09 m**, and the deployed 0.985 sits essentially at the reach optimum.
Below 0.98 the frozen pose can only get lower by *sinking* — the documented drape
mode, not a deeper bow.

---

## 3. Prior evidence — already paid for, do not re-pay

- **2026-07-29:** frozen keyframes at raised heights = **0/3, collapse 5-10 s** after the
  switch (vs 1/2 for the control). Height-only change does **not** transfer.
- **2026-07-31 2-D map** (`brace_hd_sweep_newtable.csv`): arm-only seat pocket
  **h 0.815-0.955 × edge 0.26-0.30**, widest at h=0.850. Standing closer than edge 0.26
  puts the crouched hip/knee INSIDE the table at every height ("H" wall).
- **2026-08-07 bow solve at edge 0.45:** load-bearing island **1.06-1.09**
  (real brace N + ~7× the torso clearance + a 21.8° bow instead of 34°).
- **2026-08-10:** raising to 1.09 at the old distance demanded **+21 cm of pelvis travel**;
  table + ladder had to move **−0.213 m** to be stance-anchored. Then the **JACKKNIFE**
  (planner hip-folds instead of ankle-dorsiflexing) killed the commit.

⇒ **height and distance are coupled.** Freezing the distance will give a narrow band —
that is the honest Tier-0 answer, and edge(H) is the interesting second axis.

---

## 4. Tooling state

| tool | covers | status |
|---|---|---|
| `lean_set_table.py --surface H --edge X` | planner `lean.xml` body z + 8 legs, source AND build tree; optional object/target/reach_target | existed; **validated** — dry-run at 0.985/0.45 reproduces the live file exactly |
| `lean_set_table_twin.py --surface H --edge X` | twin scene body x, both top geoms, 4 leg geoms (pos z + half-height) | **NEW this session** (skill `mjpc-fk-analyze/scripts/`); dry-run at 0.985/0.45 reproduces the live file exactly |
| `parity_audit.py <planner> <twin>` | name-matched geom diff | existing — run after EVERY height change |
| `brace_bow_solve.py --surface H --advance A` | re-solve `forearm_brace_lean`, 14-point verdict + reach + brace N + joint ledger | existing; ~2-4 min/cell |
| `brace_ladder.py --from-npy 0_stand_up.npy --to-key ... --emit` | attainable approach rungs, stance-anchored | existing |
| `xml_set_keyframe.py` | install a solved qpos | existing |
| `ownsim_bench.py` / `realchain_bench.sh` | twin validation | existing; ⚠ ownsim has **no seed control** → unpaired comparisons |

---

## 5. The ladder

**Tier 0 — zero change (the honest generalization measurement).**
Move only the table (planner + twin, parity-verified). Frozen keyframes, weights,
distance. Heights **0.90 / 0.94 / 0.985 (control) / 1.03 / 1.07**.
Deliverable: **the no-change band**. Prediction from §2/§3: passes at 0.985, fails
≥1.03 (arm cannot get low enough to seat) and ≤0.94 (bow saturates, pelvis over the slab).

**Tier 1 — one re-solved pose.**
Per height: `brace_bow_solve --surface H --advance A(H)` → `xml_set_keyframe` →
regenerate release/standback by the 0.60/0.38/0.18 lerp → same gate. Nothing else moves.
Deliverable: **does one keyframe recover the band?** If yes, that is the claim.

**Tier 2 — re-anchor the distance.** Only for heights Tier 1 fails. Expected failure is
pelvis-travel (the 1.09 lesson); the fix is edge(H) so the solved ladder is stance-anchored.
Claim becomes "parameterized by (H, edge(H))" with edge a 1-D function.

**Tier 3 — weights.** Only if the jackknife returns. Levers already identified: staged
approach appA/appB, Pelvis Forward 100→300, Posture 80→40. If Tier 3 is needed the
generalization is **not** clean and the paper must say so.

---

## 6. Instrument and guardrails

- **Stressed 5-run gates**, not 20-run benches (`feedback_stress_gates_not_20run_benches`).
  ownsim first, realchain for finalists only. State the power before any long bench.
- **Interleave heights** run-by-run — never block one height then the next, or thermal /
  foot-creep drift aliases onto the height axis.
- **Score with the force+posture brace definition** (pad ≥30 N while bow +10..+50° for ≥3 s),
  never the geometric seat metric — it ranked phantoms highest on 2026-08-10.
- **Precision metric** for strat 25 = |gripper tip − rung-2 `reach_target_table` point|,
  the paper's 2.8±1.3 cm transverse number. The deliverable curve is **precision vs height**.
- `bench_preflight.sh --kill` **before every run** (orphan estimators poisoned ~9 runs on
  09-06; tell = twin z <0.90 at t=40, and plan rate goes UP not down).
- `--plan_threads` ≤ 8. 12 physical cores; 21 threads is the 09-05 CPU-starvation confound.
- `parity_audit.py` after every height change. Wire check incl. `enx*` before any twin bench.
- Restore the snapshot at the end of the campaign.
