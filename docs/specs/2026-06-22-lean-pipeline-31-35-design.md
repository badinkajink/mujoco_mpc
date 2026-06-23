# Lean Pipeline 31–35 — Design Spec

**Date:** 2026-06-22
**Status:** Brainstorming complete, approved for planning. No code written yet.
**Related memory:** `project_lean_pipeline_31_35`, `project_reach_strat21`, `project_squatter_strat18`, `project_h12_mpc2real_paper`

---

## 1. Goal

Build the leaning-project pipeline: the robot **stands, reaches toward an object on a table in front, leans (counterbalanced) to extend its reach, then plants its other forearm on the table to brace** while the reaching arm retrieves the object. Both feet stay planted throughout. The end goal is to feed the target from a **vision** stack and drive approach from **nav2**; the gripper open/close is owned by the separate `magpie_msgs` controller (out of scope here — MJPC only *positions* the reaching hand).

This is the contact-rich "lean-over-a-surface-to-retrieve" pipeline that is the core of the mpc2real paper.

## 2. Numbering convention

- **`<30` = "pre-lean" / testing strategies** — **hardcoded** target (model `reach_target` numeric) + **hardcoded** L/R hand (`reach_hand` 1/2). All tuning and real-robot testing of each stage happens here.
- **`>30` = the lean project** — identical cost/tuning, but **runtime** target (vision/nav2 writes `data->mocap_pos`) + **auto-pick nearest arm** (`reach_hand=0`).

Each `>30` strategy is paired 1:1 with a `<30` pre-lean twin and is **generated** from it — never hand-edited.

## 3. Strategy roster

| Lean (`>30`) | Stage | Pre-lean twin (`<30`) | Base status |
|---|---|---|---|
| **31** | stand | **6** `h12_simple_stand` | ✅ real-validated |
| **32** | nearest arm extends toward target (upright) | **21** `h12_simple_reach` | twin-only, needs real tuning |
| **33** | lean further toward target via the other arm's counterweight | **16** `h12_simple_counterbalance` | twin-only, needs real tuning |
| **34** | other arm plants **forearm** to brace (no palm) | **22** `h12_simple_forearm_brace` *(NEW)* | brand new |
| **35** | all four auto-sequenced with transition flags | — (assembled from 31–34) | — |

The `residual_Strategy` slider max in the task XMLs is raised **21 → 35**.

## 4. Per-stage behavior

Cumulative-prefix ladder (same proven pattern as the legacy 0–5): strategy N runs phases `0..N` then holds its final phase (`success_sustain = 9999`). Strategy 35 gives every phase a finite sustain so it flows through and holds the brace.

- **31 — stand.** Phase `stand_up` only (= strat 6 block). Lean depth: none.
- **32 — reach.** `stand_up → reach_to_target`. The **nearest arm** (auto / hardcoded) extends to the target; body stays upright (strat 21's R=0.50 m spherical clamp, `Pelvis Tilt 100`, no torso pitch). Lean depth: arm only.
- **33 — counterbalanced lean.** `… → counterbalance_lean`. The reaching arm pulls toward the target; the **other arm + torso swing back as a counterweight**, enlarging the safe-lean envelope so the body leans further toward the target with the CoM still over the feet. Lean depth: counterweight-extended.
- **34 — forearm brace.** `… → forearm_brace_lean`. The **other (counterweighting) arm comes down and plants its forearm on the near table edge**; now supported, the lean reaches its deepest and *holds* while the reaching arm stays on the object. Lean depth: braced / deepest.

## 5. Cost-term policy — the strat-6 "standing core"

The validated stand (strat 6) is the single source of truth for the **standing/lower-body behavior** of every strategy. Terms are classified into three buckets (classification grounded in a 2026-06-21 diff of strats 6/21/16):

**A. LEG-CORE — locked to strat 6 everywhere (31–35 + pre-leans).** Injected identically by the generator:
`Foot Left Up 2000`, `Foot Right Up 2000`, `Foot Stability 15`, `Hip Yaw L/R 20`, `Hip Roll L/R 20`, `Lateral Center 150`, `Symmetry 200`, `Angular Momentum 5`, `Balance 2.5`, `CoM Vel. 10`, `Velocity 0.625`, `Joint Vel. 0.01`, `Joint Vel. Limit 5`, `Waist Yaw 30`, `Control 0.05`, `Posture 12`. (`Symmetry` penalizes only legs, so it does not fight asymmetric arm/counterweight motion.)

**B. STANDING-HEIGHT terms — strat-6 default, relax only at deep brace.** `Base Height 450`, `Height 100`, `Knees Straight 40`. Held at strat-6 values for 31/32/33; a *small, explicit, twin-justified* relaxation is permitted only at stage 34 as the pelvis lowers onto the forearm.

**C. LEAN KNOBS + ARM/TASK — per-stage, the actual work.** `Pelvis Tilt` (100 upright → 50 lean → brace value), `Torso Forward Tilt` (0 → 15 → deeper), `Reaching Hand Dist`, `Object Dist`, `Brace Pos`, `Brace Force`, `Contact`.

**Consequence / motivation:** strat 21 already matches the strat-6 leg-core verbatim. Strat 16 **zeroed the entire leg-core** (ad-hoc stance) — almost certainly why it is shaky and untested on real. Re-anchoring 16 to the strat-6 core is both what we want *and* a likely real-robot robustness fix.

**Invariant:** leg-core is never silently overridden. Any relaxation (bucket B at stage 34) is a named exception, minimal, and validated on the twin before it ships.

## 6. Targeting & arm selection (shared plumbing)

All of 32/33/34 need the same two capabilities, which **strat 21 already implements** and strat 16 currently does not:

1. **Single target source = `data->mocap_pos`.** Vision/nav2 writes it for `>30`; the `reach_target` numeric injects it for the pre-lean twins (both flow through `data->mocap_pos`). We unify on this and **drop the static `object_pos` path** (the documented strat-21 bug) so there is one target.
2. **Auto-arm via `reach_hand`** (`0` = nearest reaches; `1/2` = forced side for pre-lean tuning). The **bracing/counterweight arm is defined as the *other* arm** (opposite of the reach pick).

Required generalizations in `lean.cc`:
- **Counterbalance branch** (currently a hardcoded foot-anchored target + fixed **left** arm) → read the mocap target and use the `reach_hand` pick, mirroring for the chosen reaching side.
- **Forearm-brace residual** (`Brace Pos` / residual 12, currently hardcoded **left** hand) → drive the **non-reaching** arm's forearm to the brace point.

## 7. Stage 34 — forearm brace (no palm)

- **Why no palm:** the magpie grippers occupy the palm, so a flat-hand plant is physically impossible (it worked when handless). Stage 34 therefore **deletes the legacy `arm_plant → lean_forward` palm beats** and goes **33 → `forearm_brace_lean` directly**.
- **Bracing arm:** the **non-reaching** arm. Its **forearm** (wrist→elbow segment) rests on the table.
- **Brace surface:** the existing modeled `table` (near edge ≈ 0.4 m ahead, top ≈ 0.77 m). The forearm plants on the **near edge**; the reaching arm goes for the object farther out (~object at x≈1.5).
- **Transition timing ("as slowly as possible"):** a **~15 s** `target_ramp_sec` on the `forearm_brace_lean` phase, gated behind a firm `success_sustain` on 33 so the counterbalanced lean is fully settled before the forearm descends. These are the "transition flags."

## 8. Sync mechanism — single-source generator (approach ①)

- **Hand-edited sources only:** the four pre-lean files (`h12_simple_stand` / `_reach` / `_counterbalance` / `_forearm_brace`), each carrying **only** its arm/task terms + named lean-knob overrides. The strat-6 leg-core is the standing source.
- **Generator** `_gen_lean_pipeline.py` (sibling of the existing `_gen_simple_strategies.py`): for each strategy it emits `{strat-6 leg-core} + {stage overrides}`, concatenates the tuned phase blocks into the 31–35 ladder, and stamps the `>30` config (runtime mocap target + `reach_hand=0` auto-arm). 35 is just the full concatenation with finite sustains.
- **Build wiring:** a CMake custom command that runs the generator on every `ninja`, beside `copy_model_resources`, so tuning a pre-lean (or strat 6) and rebuilding **automatically** updates its twin and 35. Drift is structurally impossible.
- **Optional guard:** a validator that asserts every generated strategy's leg-core == strat 6 (build fails on drift).

## 9. Changes required (summary)

- **New JSON:** `h12_simple_forearm_brace.json` (strat 22) + its `h12_hands_*` mirror.
- **Refactor pre-lean JSONs** (21, 16) to override-only form + re-anchor 16's leg-core to strat 6.
- **`lean.cc`:** generalize counterbalance + forearm-brace branches to the mocap target + `reach_hand` auto-arm (reaching arm) and the *other* arm (bracing/counterweight); add the direct 33→34 transition with the 15 s ramp.
- **`lean.h`:** register strategy names at indices 22 and 31–35 (base + Hands + Magpie share the base list).
- **Model — collision presence:** the body/legs/feet/wrist already carry efficient `class="collision"` proxies (21 of them — capsules/spheres/boxes — incl. a forearm capsule + end sphere at the wrist, model lines 273–274) and the table has `table_top_collision`. The only phantom part is the **magpie gripper** (the mass-only geom I added is `contype=0/conaffinity=0`) → add a simple collision proxy (capsule/box, not the full mesh) so the whole arm has presence. Ensure the forearm proxy spans the brace contact point. Use `contype`/`conaffinity` masking so the robot collides with the **world** (table/floor) with *controlled* self-collision. Do **not** flip the ~50 detailed visual meshes to full-mesh collision (real-time planner cost + spurious self-contacts). Add/verify a right-arm-mirrored `forearm_brace_lean` keyframe. **Validate planning rate stays adequate** after enabling the new contacts.
- **Task XMLs:** raise `residual_Strategy` slider max to 35; add `reach_target`/`reach_hand` numerics for the pre-lean twins.
- **Generator + CMake hook.**

## 10. Open implementation items & risks

- **Collision presence vs planner cost:** stage 34 needs a real forearm↔table contact. Good news — the wrist/forearm already has a collision capsule and the table is solid, so the main gap is the phantom **gripper**. Risk: every contact added slows the real-time planner (the project's recurring under-planning failure mode) and can introduce noisy self-contacts → mitigate with `contype`/`conaffinity` masking (robot↔world, scoped self-collision) and gate on planning rate, not just correctness.
- **Right-arm brace keyframe:** the existing `forearm_brace_lean` was authored for the left arm; auto-arm needs a mirrored pose.
- **Counterbalance re-anchor:** restoring strat-16's leg-core may change its behavior vs the twin-tuned version; re-tune on the twin after re-anchoring.
- **Generated-file location:** the loader reads strategies from the **source** `strategies/` dir; decide whether the generator writes there (git-tracked) or into the build tree (and adjust the load path). Planning decision.
- **None of 21/16/22 is real-validated** — promotion up the ladder is gated on real bench, harness-first.

## 11. Bring-up & test order

1. Tune + real-test each pre-lean: 6 ✅ → **21** → **16** (after re-anchor) → **22** (new), each on the twin then harness.
2. Promote up the ladder **31 → 32 → 33 → 34 → 35**, harness-first for anything that leans/braces, with the real table placed to match the model.
3. Wire vision → `mocap_pos` and nav2 approach last, against the validated `>30` strategies.

## 12. Non-goals

- No gripper open/close (owned by `magpie_msgs`).
- No leg lifting / stepping (both feet planted; that's the stumble strategy's domain).
- Cleanup/retirement of the stale legacy 0–5 pipeline is **deferred** (user: "not now").
