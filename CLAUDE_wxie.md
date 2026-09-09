# CLAUDE_wxie.md — wxie's lean work

Branch **`wxie/table-height`**, cut from `icra2026` at `3e876d70` on 2026-09-05.
`CLAUDE.md` holds the facts that are true for anyone touching this repo; this
file holds what is specific to our work. Neither repeats the other — if a fact
appears in both, one of them is stale.

## 0. The question this branch is answering

**Two agents are on this branch. Read and append to §0b before starting anything.**

**What is the minimal change that makes table-height generalisation work?**

Minimal is counted in NUMBERS TOUCHED — an existing model numeric beats a new
weight, a weight beats a new residual, a new residual beats moving the robot. A
result counts only if it widens the window on **3 seeds per height at a fixed
thread count** and costs no completions at the compiled 0.985 m.

**Read `docs/lean/20260906-table_height_window_bounds.md` first.** It is the
current, self-contained account: the window, all ten refuted levers with their
kill conditions, the high-end localisation, and what is still open. This section
is only the pointer and the standing corrections.

**Answer so far: twelve arms run, none widens the window.** The high end is a
**shoulder-height deficit of about 90 mm**: the brace posture does not lift with
the table, so the left shoulder sits 0.405-0.419 m above the face at 0.985 m and
0.250-0.321 m at 1.085 m, and the forearm pad never gets above the top face at
all. Every lever tried acts horizontally — stance, aim, caps, base height, reach
posture — and the short axis is vertical. The next candidate is to command the
brace keyframe's base z per slab; the retarget only asks for +18 mm and takes the
rest out of trunk pitch, which a posture cost cannot command.

Refuted by direct test, not inference: the stance shift (`--stance_shift_x
0.063`, bench-only) put the feet where the brace keyframes assume them and the pad
did not move; `brace_target_slab` 1 moved the aim 80 mm onto the slab and bought
27 mm of pad advance with no change in clearance. Both are free at 0.985 m.

⚠ **Superseded.** Earlier revisions of this file and of `STATUS.md` say the low
end is bounded by `standback_pitch_release` with a predicted lower edge at
0.958 m. The pitch requirement is real (47.1° at 0.785 m, 36.1° at 0.885 m) but
**no low-height run ever reaches the release rung**, so the gate cannot be the
blocker. The low end is a loss of balance during the lean itself. `STATUS.md` is
generated and still carries the old text in its `GATES` block; the 2026-09-06 page
is authoritative where they disagree.

⚠ Strategy 25's ladder has **no manipulation rung**: stand, brace, reach,
release, stand back up. Rung 2 IS the task, so opening its tolerance is a
diagnostic and never a fix.

Numbers live in `studies/table_height/STATUS.md` (generated) and on the pages
`docs/lean/20260906-table_height_window_bounds.md` (current),
`docs/lean/20260905-height_window_gates.html` and
`docs/lean/20260905-brace_posture_retarget.html`. The repo-level correction —
that the rung-2 tolerance is NOT dead on a `reach_target_table` rung — is in
`CLAUDE.md` §1c.

## 0a. Tools this line has built

| Script | What it answers |
|---|---|
| `retarget.py` | the offline twin of `brace_pose_track`: re-solve the brace keyframes for a slab |
| `probe_ik.py` | what posture each slab requires, and how many dimensions the family uses |
| `probe_armreach.py` | trunk frozen, which slabs the bracing arm can seat on at all |
| `probe_static.py` | whether those poses are statically holdable on the feet |
| `probe_advance.py`, `analyze_gate.py` | what the rung-2 gate saw, replayed from the qpos dumps |
| `analyze_lead.py` | forward base travel against the line `Brace Reach Lead` draws |
| `probe_pitch.py` | the pitch each slab requires, against the release gate |
| `probe_reachset.py` | which rung-2 waypoints the arm can hold with the brace seated |
| `probe_basin.py` | where `reach_arm_q` aims, against a per-slab solve |
| `analyze_gates.py` | the two-gate figures |
| `probe_base_split.py` | how much of the brace retarget lives in the floating base (the part a posture cost cannot command) |
| `analyze_pitch.py` | the six-arm comparison: shipped, pose_track, the pitch-track gain ladder, the tilt cap |
| `probe_stance.py` | where the robot stands and how far the pad gets over the slab, against what the keyframes assume |
| `probe_bracetgt.py` | where `Brace Pos` aims the pad, against where the pad actually gets (`--slab_inset` for a `brace_target_slab` run) |
| `probe_padheight.py` | whether the arm can get the pad above the face at all, and whether it holds it there |
| `sweep_ab.py`, `sweep_lead.py`, `sweep_mode2.py`, `sweep_tol.py`, `sweep_basin.py`, `sweep_pitch.py`, `sweep_tilt.py`, `sweep_stance.py`, `sweep_bracetgt.py` | the paired arms |
| `publish_pose.sh`, `publish_gates.sh` | figures, videos, page and the STATUS section, idempotent |

Everything replays from `--qpos_out`; none of it costs sim time.

## 0b. Running work log — APPEND ONLY, newest last

Two agents are working this branch (Claude Code and codex). **Claim a line here
before you start it**, and write the result under the same entry when it lands.
An entry with no result is work in flight, not work done. Timestamps are local
(America/Denver). Keep entries short: what was claimed, what was measured, where
the artifact is.

### 2026-09-08 14:xx — CLAUDE — cmpc table-height skunkworks  [IN FLIGHT]

**Claimed.** The whole crocoddyl side of the table-height question, i.e.
everything under `Humanoid_Simulation/crocoddyl_mpc`. Codex: do not start work in
that repo without saying so here first.

User's brief, three parts: (1) is crocoddyl-mpc robust to table heights, as a
comparison and a motivator; (2) map the *stably reachable* target set per height
rather than the single shipped target — "a hemisphere of stably (leanable and)
reachable poses"; (3) revisit the brace pose keyframes, since cmpc certifies
several viable contact modes and nothing says the shipped one is right off
0.985 m.

**Established so far (no numbers yet, just wiring facts).**

- The cmpc replay model `Lean_H12_Magpie.xml` carries the table top face at
  **0.985 m** — the same compiled face as the MJPC study — so the two studies are
  asking about the same slab and heights are directly comparable.
  (`table` body at `pos=1.04 0 0.54`, `table_top` geom `pos=0 0 0.39`,
  half-extent `0.055` → 0.54+0.39+0.055 = 0.985.)
- Near edge is `body_x − 0.59 = 0.450`. The cmpc seed keyframe
  `forearm_brace_reach` stands the ankles at **x = 0.2534**, i.e. **197 mm**
  behind the near edge — which is where the MJPC brace keyframes assume the feet
  (−199…−203 mm), NOT where MJPC's `home` reset puts them (−268 mm). The 63 mm
  stance disagreement recorded on the MJPC side does not exist here.
- Every table query in `contact_select.py` (`table_top_z`, `table_x_range`,
  `table_y_range`, the IK collision rows, the narrowphase placement test) reads
  the `table_top_collision` geom out of `d.geom_xpos` rather than a constant, so
  moving the table BODY moves the face, the near edge, the legs and the
  narrowphase together. A height knob is therefore one hook in `cs.load()` and
  every downstream study (`croco_modes`, `croco_stance`, `croco_grid`) inherits
  it for free.
- Certification is static and cheap: IK + equilibrium QP per contact subset, no
  MPC solve, no sim time. A (height × target) map is affordable in a way the
  MJPC sweep is not.

**Environment that has to be right or nothing runs** (from `crocoddyl_mpc/CLAUDE.md`):
`LEAN_TASK_DIR` must be **absolute** or the mesh paths resolve relative and the
model fails to open; `~/miniconda3/envs/croco/bin/python`, never base (crocoddyl
segfaults in base); never export `LD_PRELOAD`; `--dt 0.02` on any `croco_run`.

**Next, in order.** (a) add a `TABLE_H` env knob to `contact_select.load()`,
default unset = byte-identical; (b) 1-D height sweep at the shipped target to
see whether cmpc certifies off 0.985 at all; (c) (height × target) grid for the
reachable-set map; (d) if a height certifies, export its `q*` as a candidate
MJPC brace keyframe — that is the direct attack on MJPC open item 1
(`brace_base_z_gain`), with a solved per-height pose instead of a linear guess.

**Not claimed, still free for codex:** everything in `mujoco_mpc` — in
particular MJPC open items 2 (standoff sweep at 0.785/0.885 m), 3 (locate the
upper edge to 5 mm at 1.040/1.045), 4 (the never-run 0.905–0.935 band).

## 1. Branch discipline

- **Work on `wxie/table-height`, never on `icra2026` directly.** Allen owns
  `lean.cc` and commits to `icra2026` most days; the first session of this study
  edited `icra2026`'s working tree and would have lost everything to his next
  push.
- Rebase onto `icra2026` rather than merging, so our commits stay a readable
  strip on top of his.
- Our edits to files Allen owns (`lean.cc`, `lean.h`, the lean XMLs) are
  deliberately **additive and default-off**: `residual_Table H` defaults to 0,
  which reproduces the compiled model byte for byte. That is what makes them
  cheap to rebase and safe to propose upstream.

## 2. What this branch adds

| Path | What |
|---|---|
| `mjpc/tasks/humanoid_bench/lean/lean.h` | `kLeanTableHeightParameterIndex = 7`, `BenchPhase*` accessors, `table_h_applied_` |
| `mjpc/tasks/humanoid_bench/lean/lean.cc` | the `Table H` block at the top of `TransitionLocked` |
| `mjpc/tasks/humanoid_bench/lean/*.xml` | `residual_Table H` numeric, appended at the TAIL |
| `mjpc/lean_bench.cc`, `mjpc/CMakeLists.txt` | the headless bench and its target |
| `studies/table_height/` | sweep / analyze / render / page / status scripts |
| `docs/lean/2026*.html`, `docs/lean/media/`, `docs/experiments/INDEX.md` | the doc pages and their figures |

### `residual_Table H` — the one interface we added

MJPC task parameter **index 7**, on every `lean/*.xml` that declares
`residual_Reach Z`. Value is the absolute world z of the slab's physical top
face, in metres. **0 = OFF = the compiled model = byte-identical.** Non-zero
makes `lean::TransitionLocked` rewrite the table body's z, restretch the four
cosmetic legs floor-to-underside, and shift the free object + `target` mocap by
the same delta, so the manipulation task stays fixed **in the table frame**.
Compiled face is **0.985 m**.

Appended at the tail so indices 0-6 (Height Goal / Strategy / Phase / Reach
Active/X/Y/Z), which `lean.h` hardcodes, are untouched. `deploy_common.cc`
resolves parameters by name, so nothing downstream shifts.

**The asymmetry this exposes is the whole experiment.** Task-space terms already
track the slab because they derive from the `table_surface_pos` framepos and the
compiled `table_top` half-extents: Brace Pos, the brace-force proximity gate,
Hip / Leg / Body-Table Clearance, the `reach_target_table` rungs. Fixed constants
fitted at one height do not track: `com_cap_fwd` 0.145, `pelvis_cap_fwd` 0.13,
`lean_nominal_x` 0.06, `brace_erect_target` 0.38, `brace_lead_x0` 0.24.

## 3. Running it

```bash
ninja -C build_cmake -j4 lean_bench

# one run
build_cmake/bin/lean_bench --task "Lean H12 Magpie" --strategy 25 \
  --table_h 0.985 --seed 0 --total_time 75 --threads 6 --spp 3 \
  --out r.csv --qpos_out r.qpos.csv

# a sweep, then figures, page and status
studies/table_height/sweep.py   --out studies/table_height/runs/X --seeds 3
studies/table_height/analyze.py --runs studies/table_height/runs/X --out studies/table_height/figs
studies/table_height/write_analysis.py --figs studies/table_height/figs --out studies/table_height/figs/analysis.html
studies/table_height/make_page.py --figs studies/table_height/figs --figs_rel media/th \
    --media docs/lean/media/th --media_rel media/th \
    --analysis studies/table_height/figs/analysis.html --out docs/lean/<date>-<name>.html
studies/table_height/write_status.py --figs studies/table_height/figs \
    --long studies/table_height/figs_long --ab studies/table_height/runs/bracehold \
    --out studies/table_height/STATUS.md
```

`studies/table_height/finish.sh` chains all of that unattended after a sweep.

### Run policy on this box — not optional

MJPC is CPU-bound and a parallel sweep has stuttered this desktop twice.
**Serial, `--threads 6`, under `systemd-run --user --scope -p CPUQuota=700%`.**
`sweep.py` enforces it and refuses to start if the 1-min load is already above
`nproc/2`.

⚠ **Set `SWEEP_MEM_MAX`.** `lean_bench` at these settings needs about **9.9 GB
resident**, and `sweep.py`'s default `MemoryMax=6G` silently pages the difference
to swap — which is why runs used to take 5-10 min and why they SIGKILL outright
once swap is full (`rc=-9` at 6-8 s, before the sim starts). At `SWEEP_MEM_MAX=11G`
a run takes **200-360 s**. Keep the cap under `MemAvailable`: its job is to make
`lean_bench` die before the desktop does. `nice` alone does not
protect the compositor when the contention is thread count.

### Rendering needs a display

`render_video.py` replays qpos through the same model. EGL and OSMesa both fail
here; glfw works but needs a real `$DISPLAY`, and an agent shell inherits an
empty one (the failure reads `Renderer has no attribute _mjr_context`). The
script now falls back to the X socket it finds, which on this box is **`:1`**,
not `:0`. It also re-applies the table height, because the model on disk always
shows the nominal slab and a naive replay draws the robot bracing on air.

## 4. Current results

**`studies/table_height/STATUS.md` is the single source of truth for numbers,
and it is generated — do not hand-edit it.** Regenerate with `write_status.py`
after any new sweep. The doc page (`docs/lean/`) is likewise generated from
`agg.json` / `summary.json`, so prose and figures cannot drift apart.

No result numbers are repeated in this file on purpose: they would drift the
first time a sweep is re-run. STATUS.md carries the window, the per-height
outcome counts, the failure modes and the open questions; the page carries the
figures and the scored hypotheses.

## 5. Invariants for our own code

- **Contact summing has a trap** (table legs on the floor -> the table's own
  weight counted as brace load). It is a property of the model, so it is
  documented once, in `CLAUDE.md` §1a. `lean_bench.cc` already excludes body 0
  and the free `object`; do not undo that.
- **Seating is defined by newtons, not geometry.** `pad_clear <= 5 mm` alone
  scores a forearm hanging outboard of a too-high slab as 100% seated while it
  carries 0 N. `analyze.py` requires contact; the geometric measure is kept
  beside it as `at_face_fraction` because the gap between them is the diagnostic.
- **Load statistics go over the seated window, not the whole brace phase.** Most
  of the phase is approach, and a whole-phase median reads 0 N while the forearm
  peaks at 234 N.
- **>= 3 seeds, and hold `--threads` fixed across a comparison** — see the
  nondeterminism section in `CLAUDE.md`.

## 6. Writing the pages

Global style rules live in `~/.claude/CLAUDE.md` and are not repeated here. Two
that bite most often: the title is a descriptive noun phrase, and the filename is
date-prefixed (`20260904-table_height_generalization.html`). Every page is a
**local file** — the user is explicit that nothing is to live only on claude.ai —
and `docs/experiments/INDEX.md` carries one row per page.
