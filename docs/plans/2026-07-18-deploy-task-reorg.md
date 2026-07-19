# Reorganization plan: mjpc/deploy + the H1-2 task twins

**Date:** 2026-07-18 · **Branch:** all work on fork `upper-body-handover` (current tip `df0283b` = the HAMS submodule pin) · **Owner:** max

## 0. Decisions this plan implements

Agreed in the 2026-07-18 review session:

1. **Task twins:** extract a shared core; `lean` and `stabilize` remain two tasks (names,
   slot tables, XML contracts unchanged).
2. **Deploy layer:** full restructure — one shared main skeleton, canonical gain tables,
   `deploy_common.cc` split into responsibility files.
3. **Dead code:** keep every registered task, strategy JSON, slot table, and residual
   path (including config-disabled ones — they are research assets). Remove only
   zero-consumer artifacts ("truly dead code").
4. **Comments:** current-truth only. Fix everything that contradicts the code; migrate
   dated war stories/A-B history to `HISTORY.md` files with pointers from the code.
5. **Process:** four staged PRs onto fork `upper-body-handover`; each PR is verified by
   running the RoboCasa sim bringup **before and after** the change and comparing.

## 1. Current state (one paragraph)

The only chain bringup runs (sim and real) is: `estimator_node` → `mjpc_split_core`
(compiled from `mjpc/deploy/h12_split_controller.cc` + `deploy_common.cc`; task
**"Lean H12 Magpie Split"**, strategy 6) → optional `mjpc_debug_visualizer`.
`deploy_common.cc` is 2,095 lines holding ~9 responsibilities; the four mains
(`h12_control_node`, `h12_lower_body_controller`, `h12_upper_body_controller`,
`h12_split_controller`) duplicate gain tables, start-pose arrays, and ~180 lines of
flag text each. `stabilize.cc` (4,653 ln) was forked from `lean.cc` (4,544 ln) on
2026-06-30 and they are still **~76% byte-identical** (≈3,550 lines after class
rename), with fixes hand-ported in both directions and already missed each way.
Roughly ten comment sites still describe the **H2 torque clamp removed on
2026-07-16** as an active safety backstop.

## 2. Invariants — the checklist every stage must respect

| # | Invariant | Where it bites |
|---|---|---|
| I1 | **Strategy slot numbers are cross-repo ABI.** `--strategy` indexes `GetStrategyNames()`; h1_bringup YAMLs, both launchers, and the mains' `==26` lockstand branch hardcode them. Slot 22 and 26 already **mean different things** in lean vs stabilize. Never renumber or compact. | tasks, mains, core_ws |
| I2 | **Task `Name()` strings are ABI** ("Stabilize H12 Magpie", "Lean H12 Magpie", "Lean H12 Magpie Split", "Upper H12 Magpie") — matched via `GetTaskIdByName` from CLI defaults, launchers, YAMLs. | tasks.cc, core_ws |
| I3 | **Residual emission order is locked to each task XML's `<user>` sensor list** (46 terms in Stabilize_H12_Magpie.xml). Strategy-JSON weight maps are keyed by sensor *name*; parameter indices are positional and append-only (lean: Cmd 7-11; stabilize: ArmPlan 3-18, Cmd 19-23, FunnelArm 24). Never reorder emission or parameters. | task refactor |
| I4 | **Deploy sources are compiled by path from two build systems**: fork `mjpc/CMakeLists.txt` (3 targets, `MJPC_BUILD_DEPLOY`) and `core_ws/src/h12_deploy_mjpc/CMakeLists.txt` (3 targets). Every file add/rename must update both, and the submodule pin dance applies (commit+push fork → bump pin → wipe `core_ws/build/h12_deploy_mjpc` → rebuild at matching `MJPC_REF`). | stages 2-4 |
| I5 | **Toolchain/ABI:** colcon cores must use clang-13 + ld.lld-13 (libmjpc.a is clang-13 LTO) and compile against the build tree's `_deps` MuJoCo headers, never pip headers (the 2026-07 `PatchActuators`/`actuator_gear` corruption). | any file move |
| I6 | **Binary names are ABI**: `mjpc_lowerbody_core` / `mjpc_fullbody_core` / `mjpc_split_core` (colcon) and the fork-side names. Launchers and taskset/affinity comments reference them. Keep all four targets and names. | stage 2 |
| I7 | **Strategy JSONs resolve via compile-time `SOURCE_DIR` path** (`kLean/kStabilizeStrategyFilePath`); task XMLs resolve via `MJPC_TASKS_DIR`. Do not move `strategies/` dirs or task XML files in this reorg. `lean/lean.xml` is included by Stabilize's XML via `../lean/` — the two task dirs are filesystem-coupled. **`mjpc/deploy/helper_scripts/` is external ABI too**: `estimator_node.py`/`debug_visualizer_node.py` runpy `base_estimator_node.py`/`plan_visualizer.py` by hardcoded path — exempt from the restructure. | stages 3-4 |
| I8 | **Model XML bytes are the contract**: `h1_2_base/` is a deliberate anti-drift vendoring; the staging order in `mjpc/tasks/CMakeLists.txt` (derive nu=12/nu=15 models *before* the magpie patch) keeps deploy models gripperless. No XML "cleanup". | all |
| I9 | **Runtime invariants inside deploy_common** (preserve verbatim in any split): overrides→`PatchActuators`→`Agent::Initialize` ordering, `PatchActuators` on *both* planner and latency models; eq_data dual-write (g_model + const_cast g_agent_model); pause subscriber seeded from `pause_upper_init` *before* `InitChannel`; `BestTrajectory()` read only at the post-`PlanIteration` point; `tick0` never reset at handover; plant-time vs wall-time dual clock; `SetUpperLocked` before planner thread start; cores never rclcpp nodes (CycloneDDS soname collision). | stage 3 |
| I10 | **Cost/forcer coherence**: the stepping family works only because `Residual` (cost), `ModifyControl` (swing forcer), and `TransitionLocked` (latches/governors) share clock, `SwingBell`, name-token gates, and `v_des`. Refactor must make them consume *one* implementation, never re-derived copies. | stage 4 |
| I11 | **Cross-component magic numbers**: `kImuOffset` == estimator `IMU_OFFSET`; KP/KV == model actuator classes == twin PD; `TAU_ESTOP` == safety-layer estop table; the upper main's arm kp=40 is a **deliberate** divergence (P6.2 gate byte-parity) — never "unify" it. | stage 2 |
| I12 | **Behavior preservation**: divergent twin features (stabilize's stand-hardening terms, lean's brace/jump/split-lock) stay task-only behind hooks. Converging behavior is a *tuning* decision for a later, separate pass — not part of this reorg. | stage 4 |

## 3. Verification recipe (every PR, before and after)

Baseline is captured **once at `df0283b` before stage 1** and re-captured after each PR.

```bash
# Terminal 1 — sim first
docker/scripts/docker_run.sh robocasa --headless
# Terminal 2 — build + bringup
docker/scripts/docker_run.sh ros
# inside: after a fork change run the MJPC rebuild first
/home/code/h12_sim_scripts/rebuild_mjpc.sh          # --install if assets/proto changed
# wipe stale package build whenever CMake source lists changed:
rm -rf /home/code/core_ws/build/h12_deploy_mjpc
colcon build --symlink-install    # (launcher normally does this)
ros2 launch h1_bringup h1_sim_bringup.launch.py use_skills:=false use_nav:=false use_rviz:=false
```

**Band release (blocker found in review):** `drop_band: false` in `mjpc_sim.yaml`
claims the sim auto-releases the elastic band, but `--band-auto-release` does not
exist in this branch's `h12_mujoco.py` — a naive run stands **tethered**. After the
core reaches POLICY phase, release explicitly:
`ros2 service call /elastic_band/toggle std_srvs/srv/Trigger "{}"` (or set
`drop_band: true` for verification runs). Note the band toggle is stateful.

Pass criteria, compared before vs after:

1. Split launcher completes the frame-task **handshake** (log line) and the core
   reaches POLICY phase; no safe-hold, no watchdog trip.
2. Robot **stands ≥ 60 s** at strategy 6 *with the band released*; planner status
   lines show comparable plans/s and cost (no order-of-magnitude drift).
3. `ros2 topic hz /lowstate` healthy; `rt/mjpc/plan` publishing (visualizer ghost OK).
4. **Stages 3-4 additionally:** switch to strategy **23 (trot)** and confirm
   stepping starts and survives ≥ 20 s — the stepping family exercises the
   "MUST match" cost/forcer code that stages 3-4 touch.
   **Caveat (review finding):** the core's stdin live-switch is *unreachable* in the
   launched flow — `split_body_controller.py` Popens without `stdin=`, so under
   `ros2 launch` the stdin thread EOFs and dies right after printing "live switch
   ready". Stage 2's companion adds `stdin=PIPE` + a `switch_strategy` service on
   the launcher (same pattern as its pause bridge); until then, boot at
   `strategy: 23` via a one-off yaml edit (stepping strategies carry a stand lead-in).
5. Flag-surface check (stage 2): `<core> --help` diffed against the pre-refactor
   capture — identical flags & defaults except the documented additions.
6. Both build sites compile: colcon package **and** fork cmake with
   `-DMJPC_BUILD_DEPLOY=ON` (needs unitree_install; at minimum configure-check).
   **Do the deploy-ON configure in a scratch build dir** — never flip options in the
   shared `container_cache/mjpc_build` tree (it is configured deploy-OFF and its
   `_deps` are `.git`-less; reconfigure churn there poisons every consumer).

Baseline capture also records: the actual plant dt (lowstate hz vs `/clock` — the
yaml's "MUST equal RoboCasa's timestep" claim has no enforcing knob on this branch)
and `cat container_cache/mjpc_build/.mjpc_ref` (the cache is already at `730d81e`
≠ pin `df0283b`; builds currently rely on delta-touch alone).

Caches: `container_cache/mjpc_build` has previously caused mixed-compiler links —
after any file rename/split, verify the hydrated tree picked up the change
(`launch_ros.sh` delta-touch needs the fork commit to be visible to
`.git/modules/mujoco_mpc`).

---

## 4. Stage 1 (PR 1) — comment truth-sweep, truly-dead deletions, HISTORY migration

Lowest risk; no code motion, no behavior change. Every item below was found by the
2026-07-18 mapping pass; **re-verify each against the code at fix time** (line
numbers drift).

### 4.1 Safety-relevant lies (the H2-clamp class) — highest priority

The torque clamp was removed 2026-07-16 (`deploy_common.cc` ~1753: monitor only;
banner prints "torque-budget clamp OFF"). Fix every site that still claims emitted
torque is capped at 0.9×TAU_ESTOP:

- `deploy_common.h`: file header ("two remaining .cc files" → four mains, list them);
  `kClampRatio` comment (now: parity tighten + over-budget telemetry only);
  align `align_ki`/anti-stiction note ("that SAME clamp is the backstop" → align_i_max
  is the only bound; PD torque **can** reach the estop).
- `deploy_common.cc`: file header's "four defensive fixes" H2 entry; the ~1739
  shift-before-clamp ordering rationale (ordering harmless, rationale obsolete);
  also the header's reference to nonexistent `mjpc_deploy_lowerbody_controller.cpp`.
- All four mains: "settled values … plan threads 12 … torque-budget clamp" header
  sentence; `TAU_ESTOP` table comment ("basis of the H2 clamp"); `--align_ki` /
  `--align_i_max` help (lower + split); `--frc_parity` help claiming a task turns
  parity on (no task does — `lean.cc` deliberately leaves `deploy_frc_parity` unset).
- Rename-adjacent (defer actual identifier renames to stage 2/3): note in comments
  that `kClampRatio`/`m_clamp_count` are budget-*monitor* terms.

### 4.2 Wrong-task / wrong-range help and banners

- `h12_split_controller.cc` `--strategy` help: currently the **Stabilize** slot table
  with "lean slots are absent; nu=12" — the binary's default task is the whole-body
  **Lean Split** (nu=27). Rewrite with the lean table, and **document the slot-26
  collision**: `--strategy 26` selects the lockstand *align pose* in `main()` while
  lean slot 26 loads `h12_simple_jump`. Same file `--frc_parity` help (lean XMLs do
  ship the numeric; launcher forces 1).
- `h12_lower_body_controller.cc` `--strategy` help: add 25 (straighten) and 26
  (lockstand) to the slot list.
- `h12_upper_body_controller.cc` `--straighten_start` help: drop "pair with
  --strategy 25" (the upper main has no strategy flag).
- `deploy_common.cc` stdin prompt "strategy number 0-20" (×2): print the actual
  range from the loaded task's `GetStrategyNames().size()`.
- `--plan_threads` help in lower/upper/split: "0 = compiled kPlanThreads (12)" →
  0 = AUTO (hw−6); copy the corrected text already in `h12_control_node.cc`.
- `stabilize.h`: slot-table intro ("only slot 6 is real today" — seven slots are
  real: 6/20/22/23/24/25/26); the copy-pasted "Slider layout (Lean H12)" 6-phase
  block above `GetStrategyNames`.
- `stabilize.cc`: file banner "Residuals for humanoid lean task … Residual(0)..(12)"
  → stabilize, current term count; `[lean-residual]` fprintf tag and
  "Lean residual length" `mju_warning` → say stabilize; kStepHeight comment still
  quoting 0.022 (now 0.06); the 2026-06-08 "ramp is OFF" revert note → mark
  superseded by the 2026-06-16 re-enable directly below; "actuator order in
  lean.cc::kJointVelLimit" → this file; `arm_aware_plan` "default 1 = on" → note the
  shipped XML pins it 0; the strat-16/19/21/33 lean-slot comments on the (stabilize-
  unreachable) reach/brace branches → mark "lean numbering; inert in stabilize".
- **Misplaced section banners** (functions were inserted between banner and body):
  stabilize.cc ComputeMetrics banner sits above `PlannerNumericOverrides`;
  ModifyControl banner above `ModifyRolloutState`; same pair in lean.cc. Move
  banners to their functions.
- `lean.cc`: 13-residual banner (2025-10 era) → regenerate from the current term
  list; `ComputeMetrics` "right arm always braces" convention note → describe the
  `reach_hand` numeric + auto-pick (the shipped Magpie config reaches with the
  RIGHT); replace drifted "line ~N" cross-references with function/section names.
- `stabilize.h` ~192: claim that lean fails to propagate `cmd_active_` → fixed in
  lean 2026-07-12; delete the stale contrast.
- Both task XMLs: the "These MUST stay the LAST two `<user>` sensors" comment is
  stale in `Lean_H12_Magpie.xml` (~804) and `Stabilize_H12_Magpie.xml` (~856) —
  Foot Slip follows in both; stabilize adds three more terms after. Comment-only
  edit (parsing unaffected; I8 concerns `h1_2_base/`, not task-scene comments).
- `_gen_stabilize_model.py`: FOOT_SPHERES rationale block argues for a config that
  is now `False` — compress to current truth + history pointer.
- Legacy-task headers (kept per decision #3, but docs go current-truth):
  `push.cc`/`avoid.cc` "Residuals for humanoid stand task" headers; `ur5.cc` doc
  block describing a never-implemented shadow-hand cost; `avoid.cc` "63 capacitive
  sensors" (count is model-dependent); humanoid_bench `Readme.md` Punch-task section
  (task not ported) → mark not-present-in-this-fork. *(Optional, zero-risk: fix
  copy-pasted `model=` name attrs in Stand_G1/Walk_H12/task_inspire XMLs.)*
- `README_EMBED.md` ("Finalized 2026-06-01", one node, deleted `--warmup_sec` flag,
  pre-HAMS paths): replace with a new **`mjpc/deploy/README.md`** covering the four
  mains, the two build routes (fork `MJPC_BUILD_DEPLOY` vs colcon), the topic/port
  map (full → `rt/safety/lowcmd_in`; lower+split legs → `rt/safety/lowcmd_lower_in`;
  upper+split arms → `rt/safety/lowcmd_upper_in`; pause `rt/mjpc/pause_upperbody`;
  plan `rt/mjpc/plan`; gRPC 10000/10001 fork-builds only), and the flag-diet
  constants. Delete README_EMBED.md (git preserves it) **and retarget its two
  dangling pointers**: `mjpc/CMakeLists.txt:~282` and `h12_control_node.cc:~4`.
- `helper_scripts/Command_Sheet_h12.html` (pre-HAMS `~/Desktop/h12/...` run flows,
  referenced by `mjpc_real.yaml:~6`): annotate as historical or HISTORY-migrate;
  do not delete (it documents the strat-24 WASD teleop recipe).

### 4.3 HISTORY migration (current-truth policy)

Create **`mjpc/deploy/HISTORY.md`** and **`mjpc/tasks/humanoid_bench/HISTORY.md`**.
Move (verbatim, with dates) the war stories; leave a one-line pointer at each site:

- Deploy: flag-diet note; H2-clamp add→remove arc; plan-threads 12→AUTO starvation
  story; align/straighten/handover evolution; start-pose provenance essays (the
  30-line blocks above `kLowerStartPose`/`kLockstandStartPose`).
- Tasks: catch-step v1–v5 catalog; stand_recover real-robot rejection; ankle-kp
  softening rejection; stand spline-5 rejection; jump override rejection; Tier-C
  exploration left-off note; frc_parity twin A/B "NO BETTER"; walk-ceiling /
  cmd-propagation bug arc; FOOT_SPHERES revert.
- Keep **in code**: every "MUST match" coherence note, thread-safety contracts,
  ordering invariants, numeric-equality contracts (I9–I11) — those are current truth.

### 4.4 Truly-dead deletions (per decision #3 — conservative)

Zero-consumer artifacts only; every registered task/strategy/residual stays:

- `lean/lean_simple_gripper.cc` — unbuilt 2025-10 stash; cannot compile (references
  removed members); duplicate symbols if it ever did.
- `stabilize/strategies/stabilize_simple_stand.json.bak`,
  `h1_2_modified_magpie.xml.patch.bak`, `.prepad.bak` — unreferenced snapshots.
- `avoid/utils.h` (0 bytes) + its `#include` in avoid.h + CMake listing.
- Dead decls in **both** task headers: `LeanMode` enum, `kContactStableTime`,
  `kAscentTargetRampSeconds`/`kDescentTargetRampSeconds` (never referenced).
- `deploy_common.h`: `kUseTwinTime` (referenced by nothing).
- `stabilize.cc`: the `/home/the2xman/Desktop/h12/.cursor/debug-*.log` agent-log
  block (file I/O under the transition lock, another machine's home dir).
- Commented-out FOREARM BRACING blocks in both task .cc files (several referenced
  locals — `left_reaches`, `bracing_palm`, the elbow-sensor locals — no longer exist
  in live code; cannot be re-enabled by uncommenting; git preserves them).
- `MotionStrategy::Clear()` — never called and UB if it ever were (indexes the
  vector it just cleared). Keep `SaveStrategy`/`to_json` (harmless upstream API).
- `mjpc/deploy/helper_scripts/__pycache__/` (untracked detritus) + gitignore entry.
- Kept deliberately (flagged, not deleted): `h12_table_lean_reach*.json` orphan
  strategies, G1 XMLs + `g1.xml.patch`, `utility/utility_functions.*`, the legacy
  chains (`controller_launcher.py`, `mjpc_fullbody_core`, `h12_upper_body_controller`),
  and all inherited GUI tasks.

### 4.5 Companion commit in HAMS (same-day, HAMS `upper-body-handover`)

- Delete dead `core_ws/src/h12_deploy_mjpc/config/controller.yaml` (nothing loads
  it; documents flags deleted from the fork).
- `package.xml` description + CLAUDE.md §4 row: the live entry points are
  `mjpc_split_core` + `split_body_controller.py` (+ estimator/visualizer).
- `mjpc_sim.yaml`: compress the four-generation `gravity_ff` comment stack to the
  surviving decision (sim 1.0 / real 0.85) + HISTORY pointer; fix "controller_
  launcher.py hardcodes --plan_topic" refs (it's the split launcher); fix
  "stabilize's only real slot" note.  `mjpc_real.yaml`: same two fixes.
- `mjpc_sim.yaml` header (~4-7): "torque-budget clamp … compiled into the core" —
  the same H2-clamp lie as §4.1, HAMS-side.
- `mjpc_sim.yaml` ghost sim flags: comments cite `--sim-dt` (~22), `--rigid-feet`
  (~134), `--truth-sportstate` (~165), `--band-auto-release` (~197) — none exist in
  this branch's `h12_mujoco.py`. Fix the comments; the `--band-auto-release` one is
  a live verification hazard (see §3) and the `twin_dt`-must-match-plant claim has
  no enforcing knob — say so.
- "Split core = a copy of the legs-only Stabilize lower-body core" is false in
  `split_body_controller.py:~321` and `h12_deploy_mjpc/CMakeLists.txt:~169`
  (it is the whole-body nu=27 Lean Split main publishing BOTH channels; only the
  scaffolding was copied).
- `controller_launcher.py:~110`: claims the full-body core has an empty-default
  `--plan_topic` flag — it has no such flag at all (that's the stage-2 fix).
- `h1_real_desktop_bringup.launch.py` CPU-affinity comment: names the retired
  lowerbody chain; it pins the split controller.
- `controller_launcher.py` header: note the fullbody path is broken until stage 2
  (unknown `--plan_topic`).  `rebuild_mjpc.sh` comment: agent_server copy matters to
  python clients, not to h12_deploy_mjpc (which relinks libmjpc.a).
- `split_body_controller.py`: delete the commented-out `--straighten_start` line
  (stage 2 makes it a declared param if wanted).

---

## 5. Stage 2 (PR 2) — deploy dedupe: shared tables, flag manifest, thin mains

Target: each main becomes a ~40-60-line variant descriptor; every shared numeric
lives in exactly one header. Binary names/targets unchanged (I6).

### 5.1 New shared headers in `mjpc/deploy/`

- **`h12_gain_tables.h`** — the canonical 27-row `KP/KV/TAU_ESTOP/TAU_LIMIT/
  FRC_LIMIT/JOINT_NAMES` (from the identical full/split copies), plus slice
  accessors (legs = rows 0-11, upper = rows 12-26) and the upper node's
  **deliberate** kp-40 arm override as an explicit named table with the P6.2
  rationale (I11). Lower main uses the leg slice; the numeric-equality invariants
  become structural instead of comment-enforced.
- **`h12_start_poses.h`** — `kLowerStartPose` / `kLockstandStartPose` (byte-identical
  copies today) + one-line provenance + HISTORY pointer. Add
  `AlignPoseForStrategy(int strategy)` so the duplicated `==26` pick in lower/split
  `main()` collapses — **fenced to the leg-owning variants only** (review finding:
  these are leg poses in motor rows 0-11; on the upper main `motor_offset=12` would
  interpret them as shoulder/elbow targets. The helper takes the variant and returns
  nothing for full/upper, and must never move into the uniform skeleton).
- **`deploy_flags.inc`** — an x-macro flag manifest: each main `#define`s its
  node-specific defaults (`H12_DEFAULT_TASK`, `H12_DEFAULT_STRATEGY` — upper pins 0
  to preserve today's hardcoded value, `H12_DEFAULT_BAD_ORIENT` 0.0 full / 0.9
  others, `H12_DEFAULT_GRPC_PORT` 10000/10001, …) then includes the manifest, which
  `ABSL_FLAG`s the shared set with those defaults. **Help text is overridable
  per-node too** (`H12_HELP_<NAME>` macros defaulting to the canonical string) —
  review found several help strings are load-bearing and legitimately differ:
  `--strategy` (three different slot tables), `--task` (the split main documents the
  eq-lock model contract), `--arm_aware` vs the upper's reversed complement,
  `--grpc_port` on upper (goal-ingest seam, not just monitor), `--bad_orient_rad`
  scope, `--frc_parity` numbers. Simplest split: those six stay declared in the
  mains next to the node-unique flags (split's `--pause_upper_*`); everything else
  goes through the manifest. The ankle-calibration group is **excluded from the
  upper main** (all its application sites are `moff == 0`-gated → silently inert).
- `DefaultDomainId()` moves to `deploy_common.h` (byte-identical ×4 today).

### 5.2 Flag-surface changes (deliberate, documented)

- **All four mains get the full common manifest** (minus the upper's ankle-group
  exclusion) — this gives `h12_control_node` `--plan_topic`/`--plan_hz` (+ align
  group), which **fixes the launcher-unreachable `mjpc_fullbody_core`** (both
  launchers pass `--plan_topic` unconditionally; absl aborts on unknown flags
  today). Review verified both features are fully data-driven through `NodeConfig`
  (plan publish gates on `!cfg.plan_pub_topic.empty()`; align without an explicit
  pose falls back to the model's `stand` keyframe rows `moff..moff+nu-1`, coherent
  on every node). Default-off → no behavior change unless passed. Document the F5
  slot-26 collision in the full main's align help (`--align_start --strategy 26`
  picks the lockstand pose while lean slot 26 = jump).
- Upper main's `--leg_aware` is renamed to the canonical `--arm_aware` (same
  `cfg.arm_aware` field; the binary is fork-dev-only, no launcher passes it) — noted
  in its help that the complement is reversed for the upper node.
- `--help` snapshots per binary are captured pre-change and diffed post-change
  (verification item 5).

### 5.2b Companion (HAMS side): strategy-switch service

`split_body_controller.py` Popens the core with `stdin=PIPE` and subscribes to a
`mjpc_deploy/switch_strategy` **std_msgs/Int32 topic** (a Trigger service can't
carry the slot number; a one-shot topic fits fire-and-forget) that writes
`"N\n"` to the child (`data < 0` sends a bare ENTER for the align/straighten
gate) — making the core's live strategy switch reachable under `ros2 launch`
for the first time (today the stdin thread EOFs immediately; see §3.4).

### 5.3 Build-system dedupe

- colcon: an `add_mjpc_core(<target> <main.cc>)` function replaces the three
  near-identical stanzas (include dirs, --start-group link set, lld-13, rpath,
  install).
- fork: a `foreach` over the deploy targets; **add the missing
  `h12_split_controller` target** to `MJPC_BUILD_DEPLOY` so a fork-only build can
  produce the binary production actually runs (today it exists only via colcon).
  **Fold the gRPC gating foreach** (`mjpc/CMakeLists.txt` ~383-390, a separate
  target-name list) into the same loop, and decide deliberately whether the fork
  split target gets `H12_NODE_GRPC` — if yes, a fork-built split core behaves
  differently from the production colcon binary at the same SHA (ties into F7).
- After this stage the deploy source list is stated **once per repo** (2 places,
  down from 6; verified nothing else compiles deploy sources) — prerequisite for
  stage 3's file split.

### 5.4 `main()` skeleton

A shared `BuildNodeConfig(const NodeVariant&)` + `RunDeployNode` call; each main
supplies a `NodeVariant` (telemetry enum, lowcmd topic(s), table slices,
motor_offset/upper_count/comp_motor_offset, extra cfg fields). The four mains keep
their files (targets need distinct sources) but shrink to the variant definition.

---

## 6. Stage 3 (PR 3) — split `deploy_common.cc` + explicit bring-up phase enum

Highest-risk deploy stage; do after stage 2's CMake dedupe so the source-list edit
is 2 places. Proposed decomposition (all in `mjpc/deploy/`):

| New file | Contents (from deploy_common.cc regions) |
|---|---|
| `deploy_net.{h,cc}` | NIC auto-pin, DDS channel setup, `Crc32Core` (exported — also used by the orchestrator's command build), lowstate/sportstate subscribers, publishers, pause-toggle subscriber |
| `deploy_model.{h,cc}` | task lookup, `LoadModel`, `PlannerNumericOverrides` application, `PatchActuators`, latency-model copy |
| `deploy_state.{h,cc}` | `StateData`/`RobotState`, `fill_state`, `QuatRot` (its only user — not net code), IMU/ankle calibration, pelvis-from-IMU reconstruction (`kImuOffset` contract note) |
| `deploy_choreo.{h,cc}` | warmup/ramp/hold/policy-blend, live-switch settle/blend, Phase-A align, straighten funnel, split arm-release — driven by an explicit `enum class BringupPhase { WARMUP, ALIGN_DRAG, ALIGN_HOLD, HANDOVER_SETTLE, HANDOVER_BLEND, POLICY, SWITCH_SETTLE, SWITCH_BLEND, DAMPED }` replacing the ~10-boolean ladder |
| `deploy_telemetry.{h,cc}` | status lines, B0 report, cost dump, `AppendPlanJson` |
| `deploy_grpc.{h,cc}` | `NodeAgentService` + server start/stop, `#ifdef H12_NODE_GRPC` preserved exactly |
| `deploy_runtime.h` | `NodeRuntime` struct — see scoping note below; only the `mjcb_sensor` trio + signal/`mju_error` bridge remain process-global |

**NodeRuntime scoping (review finding — this is 10× the state first named):** the
struct must absorb not just the anon-namespace globals and function-local statics
(bad-orient latch, plan-rate statics, sensor-adr cache, `PatchActuators`'
`narrated`), but the **captured-lambda and loop-local state** that crosses the new
file boundaries: `fill_state`'s captures (state holder, calib offsets,
`eq_motor`/`eq_weld`, arm_aware, moff/coff), `emit_safe_hold`'s (state, publisher,
cfg), `predict_forward`'s (scratch mjData, EWMA inputs), the planner thread's
counters (read by telemetry for plans/s), and the ~40 choreography/metrics locals
(`align_*`, `handover_*`, `straighten_*`, `upper_release_*`, `last_cmd_q`,
`ewma_comp`, `fd_*`, `tick0`, `m_*`). `tick0`/twin-time land here with the
"never reset at handover" contract attached (I9). `g_upper_paused`'s pause-callback
becomes a capturing lambda holding `NodeRuntime*` (`InitChannel` takes
`std::function`; today's lambda is captureless). Keep `g_model` the single
process-global (it triple-serves sensor callback, latency rollouts, telemetry) —
never duplicate the pointer into NodeRuntime. Two threads the table must own:
the **stdin live-switch thread** (writes switch/align atomics, calls
`SetParamByName`) → deploy_choreo; the **planner thread** → stays with the
orchestrator. The ~300-400-line `RunDeployNode` target is achievable only with
NodeRuntime designed for this from the start.

`RunDeployNode` stays in `deploy_common.cc` as the orchestrator.
Identifier renames land here: `kClampRatio`→`kBudgetRatio`, `m_clamp_count`→
over-budget monitor naming (the last clamp-era vocabulary goes).

**Phase-enum migration rule:** build the enum as a *derived view* first (asserting
it agrees with the existing booleans for one verification run), then delete the
booleans. The dual clock stays: choreography on plant time, align/handover/physical
on wall time; `tick0` untouched at handover (I9).

**Execution record (2026-07-18):** stage 3 landed in the reduced-risk order:
**3a** extracted the genuinely self-contained units (`deploy_net`, `deploy_model`,
`deploy_state` types+QuatRot, `deploy_telemetry` AppendPlanJson, header-only
`deploy_grpc`) and applied the clamp-era renames (`kClampRatio`→`kBudgetRatio`,
`m_clamp_count`→`m_over_budget_count`); **3b** introduced `BringupPhase` as the
derived view, printed in every status line (each verification run exercises it).
The remaining stage-3 work — NodeRuntime absorbing the loop/lambda state,
`deploy_choreo.cc`, and inverting the enum to drive the ladder — is deferred to a
**3c** pass after stage 4, with the derived view as its correctness oracle: 3c
must reproduce the exact phase sequence 3b's labels log today. Rationale: having
read all 2,138 lines, the lambda-capture web (fill_state/emit_safe_hold/
predict_forward + ~40 loop locals) makes the full NodeRuntime move the riskiest
single step of the reorg; the derived view extracts most of the readability value
at zero behavioral risk and turns the final move into a diff against logged truth.

**Explicitly unchanged (flagged follow-up F2):** safe-hold and the stale watchdog
publish only on `cfg.lowcmd_topic` — the split core's upper channel goes silent, not
damped, on stale state. Changing that is a safety-semantics decision, not a
refactor.

---

## 7. Stage 4 (PR 4, likely split into 4a/4b/4c) — lean/stabilize shared core

Goal: one implementation of the ~3,550 shared lines; both tasks keep their names,
slot tables, XMLs, JSONs, parameter layouts, and **current behavior** (I12).

**Build site (review finding):** task sources compile **only into libmjpc** — the
colcon package links the prebuilt `libmjpc.a` and never compiles task files. So
`h12_common/` is added to the libmjpc source list in `mjpc/CMakeLists.txt` (one
place), staged automatically by `copy_resources` (whole-tree copy — no staging
edits), and reaches the container via `rebuild_mjpc.sh` + colcon relink. The
six-stanza worry (I4) does not apply to this stage.

**Ordering premise (verified in review):** the first 43 `<user>` sensor names in
`Stabilize_H12_Magpie.xml` and `Lean_H12_Magpie.xml` are identical and in identical
relative order (dims differ only via `model->nu`); stabilize's extra terms (Stance
Width, Foot Flat, CoM Cap) are appended strictly after. Shared emitters therefore
need no per-task ordering parameters.

New directory `mjpc/tasks/humanoid_bench/h12_common/`:

| New unit | Contents |
|---|---|
| `h12_plan_snapshot.h` | **`PlanSnapshot` struct** copied wholesale into `ResidualLocked` — replaces the 17-arg ctor + 7 post-hoc field assignments (the pattern that caused the lean `cmd_active_` rollout bug). Every rollout-visible field lives here. |
| `h12_phase_machine.{h,cc}` | MotionStrategy plumbing (load-on-change, clamp, phase advance/scrub), weight-ramp trio (`SnapshotXmlDefaultWeights`/`PrepareNextPhaseWeights`/`ApplyRampedWeights`), target-pose ramp, `SnapshotEffectiveScales`/`MarkNewlyAppearedContacts` |
| `h12_balance.{h,cc}` | the byte-identical support-polygon/capture-point block (stabilize 1135-1440 == lean 1394-1699 today), `project_triangle` (single copy), edge amplifier |
| `h12_gait.{h,cc}` | `SwingBell`, gait clock, **one** signed capture-danger function (today ×5 hand-synced copies), v_des resolution chain, step-and-settle pulse, settle governor, Cartesian gait/step-place, drive FSM, catch latch — the I10 coherence contract becomes structural: cost, `ModifyControl`, and `TransitionLocked` all call these. The emitters are not independent: the gait clock computes per-call values (`g_amp`, swing bumps, capture excursions) consumed by several later terms — extract a **`GaitState` struct computed once per Residual/ModifyControl call and threaded through the emitters** |
| `h12_task_debug.{h,cc}` | the env-gated diagnostics block (gate reads `H12_TASK_DEBUG`, still honoring `LEAN_DEBUG` for muscle memory), tagged with the actual task name |
| `h12_task_core.{h,cc}` | base `ResidualFn`/task scaffolding tying the above together, with virtual hooks for task-specific residual terms, `TransitionLocked` extensions, `PlannerNumericOverrides`, `ModifyControl`/`ModifyRolloutState` |

Rules:

- **Each task's `Residual()` remains the order-driver** calling shared term
  emitters — emission order stays exactly its XML's `<user>` order (I3). No shared
  "emit everything" loop.
- **Name-token dispatch → enum at load:** derive `PhaseKind {STAND, STUMBLE, TROT,
  WALK, DRIVE, STRAIGHTEN, JUMP, LOCKSTAND, …}` once per strategy load in
  `TransitionLocked`, store in the snapshot; the substring tests
  (`"trot"`/`"drive"`/…) happen in one function. JSON phase names unchanged.
- **Divergent features stay task-only** (I12): stabilize keeps Foot Flat / CoM Cap /
  Stance Width / ankle tax / catch-march / funnel-arm / ARM_PLAN; lean keeps
  TableBraceForce / jump / split `SetUpperLocked` / reach seam. The lean-pipeline
  branches move into shared code (lean exercises them; stabilize's JSONs never
  trigger them — nothing is deleted, per decision #3).
- Per-task remains: `GetStrategyNames`, parameter-index constants (diverged layouts,
  I3), XML paths, `strategies/` dirs, the two `kLeanStrategyFilePath`-style
  constants (I7).
- Suggested sub-PRs: **4a** introduce `PlanSnapshot` in both tasks (mechanical,
  twin-verified); **4b** extract leaf helpers bottom-up (capture-danger, yaw,
  triangle, SwingBell, balance block, debug block); **4c** phase machine +
  `TransitionLocked` skeleton + headers onto the base class.
- Thread-safety contract carries over verbatim: `TransitionLocked` writes under the
  transition lock before workers fan out; `ModifyRolloutState` touches only
  per-mjData fields with restore hygiene.

**Execution record (2026-07-19):** **4a landed** (`7e84691`): `PlanSnapshot` via a
shared base struct (`h12_common/h12_plan_snapshot.h`) mixed into both `ResidualFn`s
by inheritance (member names unchanged → zero `.cc` edits); both `ResidualLocked`s
are now one wholesale struct assignment. **4b-1 landed** (`6c53b62`):
`h12_common/h12_gait.h` — shared `SwingBell` (both twins' local copies deleted) and
`CaptureExcursionFrom` replacing the 5-per-file hand-synced signed-excursion copies
(Residual cost, recovery tier, DC-EMA, catch latch, drive gates), arithmetic order
preserved bit-identically. Both gated on the RoboCasa stand (z≈1.01, ~52 plans/s).
**4b-2 analysis complete, extraction pending:** the twins' balance/support-polygon
blocks re-anchor at `project_triangle` (stabilize ~1144, lean ~1432, ~316 lines) and
diff to ~21 lines, ALL of which is stabilize's F3 `back_balance_boost` — whose
numeric defaults to 1.0 (OFF) when absent, and lean's XML has no such numeric, so a
shared function WITH the F3 lookup is behavior-identical for lean (the divergence
unifies without an I12 violation). The extraction needs a context struct for the
enclosing-scope inputs (`capture_point`, `is_leg_lift_stage_early`,
`any_arm_contact`, `brace_contact_force`, `foot_left_pos`/`foot_right_pos`,
`bracing_hand`, `edge_smooth`/`bscale`/`fwd_scale`/`cp_dx`, residual+counter) plus
the `PlanSnapshot` fields it reads. Remaining after 4b-2: 4b-3 (weight-ramp trio +
LEAN_DEBUG block), then 4c (phase machine + `h12_task_core` base class), then the
deferred 3c.
- **No function-local statics in shared code** (review rule): both tasks are
  instantiated in *every* process (`GetTasks()` — deploy nodes and the GUI alike),
  so a static moved into `h12_common/` becomes cross-task shared state. Per-task
  state goes through the task instance / `PlanSnapshot`, never a shared static.
  Corollaries verified safe today: stabilize's `s_cap_*`/`s_trim_x` file statics
  stay task-side (the shared balance emitter takes the — already-trimmed — capture
  point as an argument; the DC-blind recovery tier is stabilize-only per I12).
  Include DAG stays acyclic as long as the leaf units (`h12_gait`, `h12_balance`)
  take plain structs and never include task headers.

---

## 8. Flagged follow-ups (each needs its own decision — NOT in this reorg)

| # | Item |
|---|---|
| F1 | `LoadStrategy` failure is unhandled: **28 of 36** `Lean_H12_Hands` slots have no JSON (8 resolve: 1, 3, 4, 31-35) → empty-vector deref (UB) if selected. Make load failure fail-safe (keep previous keyframes / fall back to stand) — small fix, big footgun. |
| F2 | Split core's **upper channel is never safe-held** (silent, not damped, on stale-state/fatal). Safety-semantics decision. |
| F3 | Stabilize GUI slider range 0-24 excludes deploy slots 25/26 (SetParamByName bypasses the range). Bump the numeric or document the lockout. |
| F4 | Strategy-JSON resolution via compile-time `SOURCE_DIR` vs XMLs via `MJPC_TASKS_DIR` — unify on `MJPC_TASKS_DIR` so containers don't need the fork source mounted at the configure-time path. Runtime behavior change. |
| F5 | Split core's `--strategy 26` = lockstand-align on a lean-task binary (lean 26 = jump). Consider name-based align selection. |
| F6 | Behavior convergence between the twins (port stand-hardening to lean, phantom-brace fix to stabilize's remnants, …) — a tuning pass with twin A/B, after stage 4. |
| F7 | gRPC availability matrix: no colcon core compiles `H12_NODE_GRPC`, yet all parse `--grpc_port`; split core can never have the monitor. Wire it into colcon or drop the flag from container docs. |
| F8 | `gen_lean_pipeline` regenerates committed JSONs **into the source tree** each build; hands ladder silently frozen (inputs deleted). Redirect to the binary dir or stop committing output. |
| F9 | Legacy chains (`controller_launcher.py`+lowerbody yaml sections, `mjpc_fullbody_core`, `h12_upper_body_controller`) kept for now per decision #3 — revisit after the handover branch stabilizes. |
| F10 | `AppendPlanJson` hand-rolled serializer → nlohmann once the colcon include path can see `_deps/json-src`. |
| F11 | **The §3.4 trot criterion is unachievable via live-switch in RoboCasa** (found during the stage-3 gate, 2026-07-19): a live switch into 23 keeps the *startup* strategy's planner numerics (stand's spline-3 — documented as unable to represent the gait), and the robot falls backward ~4-15 s after the switch on **both** stage-2 and stage-3 binaries (A/B-neutral → not a regression; stage-3 gate met on behavior preservation). A real trot gate needs boot-at-`strategy: 23` (proper `PlannerNumericOverrides`) or the fork twin at RTF≈1. Alternatively make the deploy node re-apply per-strategy numerics on a live switch — a behavior change, decide separately. |

## 9. Mechanics per stage

1. Branch off fork `upper-body-handover`; PR back into it. Commit messages follow
   the existing `deploy:`/`stabilize:`/`lean:` prefixes.
2. After merge: bump the HAMS submodule pin on HAMS `upper-body-handover`
   (never leave the pin on an unpushed commit), wipe
   `core_ws/build/h12_deploy_mjpc`, rebuild the ros image at the matching
   `MJPC_REF` before anyone runs the real robot. At each image rebake also bump the
   two stale ref constants the pin dance doesn't touch: `docker/RosDockerfile`'s
   `ARG MJPC_REF` default (currently `c7fc99f`, two pins behind — a bare
   `docker compose build` bakes the wrong seed) and `launch_ros.sh`'s `SEED_REF`
   fallback (`9f3cb648`, staler).
3. Run the §3 recipe before (baseline) and after; attach the two run logs to the PR.
4. Real-robot use of any post-reorg pin: stand re-validation first
   (`start_position_verified` flow) — same bar as any deploy change.
