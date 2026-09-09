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

**RESULT 1 — every height the MJPC study fails at certifies statically in
crocoddyl.** `crocoddyl_mpc/studies/height_reach.py`, run
`studies/runs/2026-09-08_height_reach/facerel`. Static IK + equilibrium QP (real
friction cones, real actuator limits, feet pinned), curated 8-mode ladder, at
the shipped target held 113 mm above the face:

| face | best mode | effort | max tau/lim | brace N | support margin | pad clear | base pitch |
|---|---|---|---|---|---|---|---|
| 0.785 | elbow+wrist | 0.657 | 0.43 | 45.5 | 79 mm | −1 mm | −45.2° |
| 0.885 | forearm | 0.695 | 0.37 | 64.0 | 85 mm | −2 mm | −32.6° |
| 0.985 | elbow+wrist | 0.561 | 0.43 | 35.7 | 87 mm | 0 mm | −24.2° |
| 1.035 | elbow+wrist | 0.560 | 0.46 | 14.4 | 74 mm | −1 mm | −19.5° |
| 1.050 | elbow+forearm+palm | 0.549 | 0.46 | 28.7 | 85 mm | −2 mm | −17.9° |
| 1.060 | elbow+forearm+palm | 0.536 | 0.46 | 30.4 | 86 mm | −2 mm | −16.9° |
| 1.085 | elbow+forearm | 0.570 | 0.47 | 34.5 | 86 mm | −2 mm | −15.1° |

**8 of 8 modes admissible at all seven heights**, including `elbow+forearm` --
the posture MJPC ships -- with both pads seated (gaps −1.5 to −2.9 mm) and peak
torque under half the clamp basis everywhere. A statically valid, torque-
feasible, balance-feasible braced pose therefore EXISTS at 0.785 and at 1.085 m.
The MJPC height window is not a limit of the robot or of the maneuver.

The brace also matters MORE at the tall slab, which is the motivator: legs-only
support margin decays 79 → 35 mm from 0.785 to 1.085 m while the braced margin
holds 74–87 mm across the whole range.

**RESULT 2 — the certified pose does NOT preserve the brace geometry, and the
change it asks for is small at the high end and large at the low end.** Joint-
space delta of the certified `elbow+forearm` pose from Allen's shipped
`forearm_brace_reach` keyframe, per height (`ef/` run). Read the deltas BETWEEN
rows, not the absolute ones -- the certification differs from the keyframe by
−62 mm of x and +22° of right_shoulder_pitch even at 0.985, where the keyframe
was authored:

| face | base dz | base dpitch | rms joint | three biggest joints |
|---|---|---|---|---|
| 0.785 | −58.5 mm | +19.8° | 0.199 rad | l_hip_pitch −32.1°, l_shoulder_roll +25.5°, r_hip_pitch −23.9° |
| 0.885 | −47.7 | +9.3 | 0.143 | l_hip_pitch −22.4, r_hip_pitch −18.1, l_shoulder_roll +18.0 |
| 0.985 | −12.6 | −2.2 | 0.096 | r_shoulder_pitch +22.1, l_shoulder_roll +14.3, r_shoulder_roll +4.6 |
| 1.035 | −3.9 | −6.6 | 0.101 | r_shoulder_pitch +22.4, l_shoulder_roll +14.9, l_elbow +11.1 |
| 1.085 | +2.0 | −10.7 | 0.124 | r_shoulder_pitch +22.6, l_elbow +19.6, l_shoulder_roll +15.8 |

Relative to its own 0.985 solution, the tall slab asks for **+14.6 mm of base z
and −8.5° of pitch**. MJPC's `retarget.solve()` -- a damped-least-squares
retarget, a completely different tool -- asks for **+18 mm and −8.7°**. Two
independent solvers agree to 3 mm and 0.2°.

**CORRECTION to §0 and to the 2026-09-08 report.** §0 says "the next candidate
is to command the brace keyframe's base z per slab; the retarget only asks for
+18 mm and takes the rest out of trunk pitch", with the implication that +18 mm
is too timid against a measured ~90 mm shoulder deficit. That inference is
wrong. The ~90 mm figure is the gap between the ROLLOUTS' shoulder height and
the shoulder height that would reproduce the 0.985 geometry, and reproducing the
0.985 geometry is not what the tall slab requires -- the certified pose bows 9°
less and gets there. +18 mm and −8.7° is the right answer, and MJPC's
`brace_pose_track` already delivers it. **MJPC open item 1 (`brace_base_z_gain`)
is therefore predicted dead on arrival**: it commands a quantity the retarget
already commands correctly. Codex, do not spend runs on it without reading this
first.

**What that leaves.** The pose is right, applying it does not help (pose_track
is refuted on the MJPC side, 0/3 at 1.085 and it costs a completion at 1.035),
so the failure is in the TRANSIT or in the fact that MJPC applies the keyframe
as one soft cost among many rather than as a contact schedule. That is the
question the crocoddyl dynamic test is now pointed at, and it is the honest
comparison: gradient planner with prescribed contacts vs sampling planner with
cost-shaped contacts, same robot, same slab.

**Caveat, stated plainly.** All of the above is STATIC and the feet are PINNED
at the seed stance. It proves the pose exists and is holdable; it does not prove
the robot can get there. `croco_modes` is a conservative screen, not a workspace
bound (see memory `cmpc-legs-only-is-real`). The dynamic replay is what upgrades
this.

**Secondary finding, directive (c) / brace-posture question.** `elbow+forearm`
is the ranked pick at exactly one of seven heights (1.085). At 0.885 m its
effort is **1.651 against 0.695 for `forearm` alone** -- the shipped two-contact
posture is 2.4x more expensive than a single-contact brace at the height where
MJPC falls forward. The winning posture changes with the slab, which is the
thing neither study has ever let vary.

**TRAP, cost me a run and would cost anyone else one.** Do NOT run a crocoddyl
height study at the paper's nominal reach target **x = 0.9047**. It is the one
target on the whole ladder where the CMPC brace does not survive a hold --
measured 2026-08-22: elbow+forearm at contact Kp = 50 falls at 10.2 s while
legs_only holds indefinitely, because the certified brace force there is 51.5 N
against 80.9 N at 1.05 and the declared contact is a light touch. Bracing at the
near target is harmful in EVERY contact mode (least-effort mode selection only
halves the fall rate 2/2 → 1/2). I ran it and got pelvis 0.076 m -- the robot on
the floor -- and it is not a height result, it is that pathology reproduced at
every height. `brace_vs_stand.NOMINAL2_X = 1.06` is the settled condition and is
now `height_dynamic.DEFAULT_TX`. Control at 0.985 m / x = 1.06: braced run
upright at pelvis 0.952 with 9.5 mm of reach error and 143.9 N through the
table, standing run upright with 32.7 mm of error.

The load path is the one `croco-brace-load-path` records, unchanged at this
target: the force arrives through `left_shoulder_yaw_link` (the "elbow" site's
body) and `left_wrist_yaw_link` (the wrist pad). The forearm pad -- the geom the
MJPC study measures clearance against and names its brace after -- carries
nothing. **That is a live hypothesis for the MJPC high-end failure and codex
should not assume otherwise: MJPC may be measuring the wrong pad.**

**Also needed or nothing runs:** `CL_ASSETS_DIR` must point at the CL_Assets
checkout (`$PWD/../CL_Assets`) or `croco_run` exits rc=1 in 0.4 s with "no H1-2
magpie URDF found" and `solve_plans` reports zero plans without erroring.

**RESULT 3 — the brace load leaves the forearm pad at the tall slab, and the
report's "the pad never reaches the wood" is wrong.** `studies/table_height/
probe_padset.py`, replay only, no new runs: logged qpos evaluated with
**`mj_forward`** (not `mj_kinematics`) so MuJoCo's own narrowphase decides what
touched, over the brace rungs, table contacts only, body 0 and the free object
skipped.

Peak normal force and the fraction of brace-rung frames in contact, median
across shipped seeds:

| face | `left_forearm_pad` | `left_wrist_roll_link` (geom46, unnamed mesh) | `left_gripper_jaw_a` | `left_wrist_pad` |
|---|---|---|---|---|
| 0.985 | **52 N, 19%** | 0 N, 0% | 14 N, 6% | 12 N, 3% |
| 1.035 | **89 N, 50%** | 0 N, 0% | 27 N, 18% | 0 N, 0% |
| 1.050 | 58 N, 49% | 47 N, <1% | 27 N, 40% | 0 N, 0% |
| 1.060 | 73 N, 54% | 66 N, <1% | 26 N, 44% | 43 N, 0% |
| 1.085 | **5 N, 76%** | **51 N, 52%** | 26 N, 35% | 32 N, 24% |

At 1.085 m the forearm pad is in contact for **76% of the brace-rung frames and
carries 5 N**, while an unnamed mesh geom on `left_wrist_roll_link` carries 51 N
for half of them. The arm is braced; it is braced on the wrist housing. Two
statements in the 2026-09-08 report have to be withdrawn: "the pad never reaches
the wood" (it does, most of the time) and the 0 N median forearm force at 1.085
read as no contact (it is contact carrying nothing).

**Why this may be the whole high-end story.** `lean.cc:5583` — the
brace-contact-gated advance — scans for contacts on **`left_forearm_pad` only**.
The escape hatch, `wrist_brace_gate` (lean.cc:5595, "a WRIST-on-rail brace never
triggers the `left_forearm_pad` contact scan above, so the reach rungs could only
advance by blind TIMEOUT"), is a numeric that **ships at 0 = OFF**. So at the
tall slab the robot is loaded through the wrist, the gate is looking at a pad
that is touching but unloaded, and the rung advances on a timer. That is a
one-numeric change by the branch's own minimality metric, and it is now the top
MJPC candidate — ahead of `brace_base_z_gain`, which Result 2 predicts dead.

**Caveat, and it is a real one.** n = 2 shipped runs with qpos dumps at 1.085 m
and n = 1 at 1.050, 1.060 and 0.885. The 0.985 and 1.035 rows are n = 2. This
is enough to reverse a claim I made from a clearance metric on the same runs; it
is not enough to certify `wrist_brace_gate` as the fix. The measurement that
would settle it: set `wrist_brace_gate` to the ~30 N the wrist actually carries
and run 3 seeds at 1.050, 1.060 and 1.085, with the 0.985 control. **Codex: this
is unclaimed and is the highest-value MJPC run on the board.** Say so here before
you start it.

Two earlier metrics were wrong before this one was right, and both errors are
worth not repeating: a purely geometric clearance test scores an arm hanging
BESIDE the table as having reached the wood (median closest approach −748 mm at
0.885 m, an arm on the floor next to a slab it never touched), and restricting
to the slab footprint then scores the arm swinging UNDER the overhang as 16 mm
of penetration. Only the narrowphase distinguishes these.

**RESULT 4 — crocoddyl completes at 0.885 m, where MJPC is 0/6.** Dynamic, not
static: `crocoddyl_mpc/studies/height_dynamic.py`, plan + closed-loop MuJoCo,
25 s episodes, 3 seeds, contact Kp = 50, reach target x = 1.06 held 113 mm above
the face. Partial — the sweep is still running at the time of writing.

| face | MJPC (shipped) | CMPC braced, upright and loaded | CMPC brace force | CMPC reach err | CMPC standing reach err |
|---|---|---|---|---|---|
| 0.785 | 0/6 | **1/3** | 146 N | 15 mm | 44 mm (3/3 upright) |
| 0.885 | **0/6** | **3/3** | 150–157 N | 12–15 mm | 39–41 mm (3/3 upright) |
| 0.985 | 6/6 | 1/1 so far | 144 N | 10 mm | 33 mm |

At 0.885 m the gradient planner establishes and holds the brace on every seed and
lands the hand 12–15 mm from target, against 39–41 mm standing — so the brace is
buying accuracy there, not just surviving. The sampling planner completes none of
six at the same slab. The two differ in what a contact mode IS (MJPC's is three
cost weights over a contact-implicit planner; CMPC's is a contact schedule baked
into the action models, and no weight can add a contact), which makes this a
planner-architecture result rather than a robot limit — consistent with Result 1.

**RESULT 5 — the reachable set, and why a tall table is where bracing matters
most.** `height_reach.py` over 5 faces x 12 reach distances (`xmap` run, 60
cells, 371 s). The reach target keeps its 113 mm offset above the face and its
shipped y; x runs 0.75 to 1.30 m. Legs-only is enumerated in every cell as the
control, so each cell says both "can it be reached" and "does the brace buy
anything".

Unbraced reach limit — the largest x at which `legs_only` still certifies:

| face | unbraced limit | braced limit | what the brace buys |
|---|---|---|---|
| 0.785 | 1.10 m | 1.30 m | 200 mm |
| 0.885 | 1.10 | 1.30 | 200 |
| 0.985 | 1.15 | 1.30 | 150 |
| 1.035 | 1.15 | 1.25 | 100 |
| 1.085 | **1.00** | 1.25 | **250** |

At 1.085 m the unbraced envelope collapses by 150 mm while the braced envelope
barely moves, so the brace is worth 250 mm there against 150 mm at the compiled
height. The support-margin gain says the same thing: braced minus legs-only
margin runs from ~0 at x = 0.75-0.90 to +200 to +260 mm at x = 1.20-1.30, at
every height. **A taller table is the case where bracing earns the most, which
is the opposite of how the MJPC window reads.**

The near end has its own bound and it is not reach. At x = 0.75-0.85 on the low
slabs the ranked pick is `legs_only` -- the brace never wins -- and the margin
gain is 0 or slightly negative (−6 mm at 0.90/0.785, −9 mm at 0.90/0.885). That
is the same near-target pathology that put the robot on the floor in the trap
above, now visible statically. At 1.085 m / x = 0.75 nothing certifies at all
(0 of 8 modes).

**And the winning posture moves distally as the slab rises.** Low tables rank
`elbow+wrist` and `elbow+forearm`; at 1.085 m the ranked pick is `forearm+palm`
across x = 0.95-1.15. That is the same conclusion the MJPC narrowphase reached
from the other direction in Result 3 -- at the tall slab the load leaves the
forearm and goes to the wrist and hand. Two independent methods, one static and
one a replay of real rollouts, agree that the tall-slab brace should be distal.

**CLAIMED AND RUNNING — the wrist gate.** I am running the Result 3 experiment
rather than leaving it: `studies/table_height/sweep_wristgate.py`, arm `wg30`,
heights 1.085 / 1.050 / 0.985, 3 seeds, `--threads 6`, `SWEEP_MEM_MAX=11G`.
`wrist_brace_gate` is now declared in `Lean_H12_Magpie.xml` at **data="0.0" =
OFF = byte-identical** (it was implemented in lean.cc on 2026-09-01 and this
model simply never declared it, so the wrist path could not be armed at all);
the arm moves it to 30, the value Allen already ships on
`Lean_H12_Magpie_battery_hip.xml`. Kill condition, stated before the runs: the
arm is refuted unless it completes some height above 1.035 m on 2 of 3 seeds AND
holds 3/3 at the 0.985 m control. **Codex: take open items 2, 3 and 4 instead.**

**Bookkeeping defect in my own harness, noted so a count is not misread.**
`height_dynamic.py` skips cached episodes but does not add them back to
`summary.json`, so a re-run reports only the episodes it actually flew. The CSVs
on disk are the truth; count those, not `len(summary["episodes"])`.

**RESULT 6 — THE WRIST GATE IS REFUTED, and it moved the failure one rung.**
`runs/wristgate/wg30`, `--numeric wrist_brace_gate=30`, 3 seeds, `--threads 6`.

| face | complete | phase entries |
|---|---|---|
| 1.085 | **0/3** | rung 2 at 24.2 / 25.4 / 27.1 s, rung 3 never |
| 1.050 | **0/3** | rung 2 at 24.0 / 24.0 / 25.6 s, rung 3 never |
| 0.985 | **3/3** | full ladder, t_complete 40.5 / 46.9 / 49.9 s |

Kill condition met on the failing side, so the arm is refuted: nothing above
1.035 m completes. The control holds 3/3, so declaring the numeric and arming it
costs nothing at the compiled height.

It is not a null result, though. **Every seed now reaches rung 2 at 1.085 m
(3/3), against 3 of 6 shipped** (`ab/off` 1/3, `seeded` 2/3), and the entries
come 1–2 s later, which is the gate doing exactly what it is for: holding the
lean rung until the wrist is verified instead of advancing on a timer. The brace
contact gate was a real defect and arming the wrist path fixes it. It is not the
thing that bounds the window.

**So the high end is a REACH bound on the reaching arm, not a brace bound on the
bracing arm.** Nothing gets past rung 2 at any height above 1.035, in any arm
ever run, and rung 2's gate is not the brace at all -- it is
`target_distance_tolerance` on the RIGHT gripper jaw tip against a point built
from the slab, `near edge + 0.55, centre − 0.04, face + 0.15`, tolerance 0.07
(CLAUDE.md 1c; perfect separation over 21 runs). At a 1.085 m face that point
sits at **1.235 m absolute**, against a shoulder that holds ~1.40 m. The window
is the set of slabs where the reaching hand can get within 70 mm of a target
that rises with the table.

That reframes the high end, and I then checked the obvious follow-up and it is
dead too. **`probe_reachset.py --tol 0.07` (the gate's own tolerance): the
rung-2 waypoint is reachable at EVERY height and at every cell of the grid**,
0.30-0.75 m in from the near edge by 0.00-0.30 m above the face, with both brace
pads seated to 0.1 mm and the feet planted. At 1.085 m the SHIPPED cell
(0.550, 0.150) solves with a tip error of **0.0 mm**. So lowering `rtt[2]` in
the strategy JSON cannot be the fix -- the arm can already reach where it is
being sent. Do not spend runs on it.

**Which closes the argument.** At 1.085 m: a statically valid, torque- and
balance-feasible braced pose exists (Result 1); the keyframe change it needs is
+15 mm and −9 deg and `brace_pose_track` already delivers it (Result 2); the
brace does establish and carry ~50 N through the wrist (Result 3); arming the
wrist gate gets 3 of 3 seeds to rung 2 (Result 6); and the rung-2 waypoint is
kinematically reachable from the braced pose with 0.0 mm of error (this probe).
Every static, kinematic and contact precondition is satisfied and no run
completes. **What is left is the planner.** The MJPC height window is a property
of the sampling search, not of the robot, the pose, the brace or the gates --
and Result 7 is the constructive proof, because the same robot on the same slab
does all seven heights under a gradient planner with an explicit contact
schedule.

Caveat on the probe, from its own docstring: feet planted and pads seated is a
strict SUBSET of what the controller must do, so a reachable cell is necessary
and not sufficient. It cannot show the arm can get there dynamically while
staying inside its cost caps. It can and does show that no amount of cost tuning
makes an unreachable cell reachable -- and none of these cells is unreachable.

**RESULT 7 — crocoddyl generalises across the whole range. This is the answer to
the session's question.** `height_dynamic.py`, complete: 7 faces x 3 seeds x
{braced, standing}, 25 s episodes, plan + closed-loop MuJoCo, reach target
x = 1.06 held 113 mm above the face, contact Kp = 50.

| face | MJPC shipped | CMPC braced, upright and loaded | CMPC reach error |
|---|---|---|---|
| 0.785 | 0/6 | 1/3 | 15 mm (the one that held) |
| 0.885 | 0/6 | **3/3** | 12–15 mm |
| 0.985 | 6/6 | **3/3** | 5–6 mm |
| 1.035 | 5/6 | **3/3** | 3–13 mm |
| 1.050 | 0/3 | **3/3** | 2–3 mm |
| 1.060 | 0/3 | **3/3** | 3–7 mm |
| 1.085 | 0/6 | **3/3** | 1–3 mm |

**0.885 through 1.085 m, 3 of 3 seeds at every height, hand within 1–15 mm.**
The tall slabs are the most accurate cells in the sweep, not the least. The only
weak height is 0.785 m at 1/3, the balance-limited low end, which is where the
static map also says the brace stops being worth taking.

The two planners are running the same robot, the same slab and the same target
offset, so the difference is what a contact mode IS: MJPC's is three cost
weights over a contact-implicit planner and the contact has to be discovered;
CMPC's is a contact schedule baked into the action models and no weight can add
or remove one. Table-height generalisation is available on this machine today
with a gradient planner and an explicit schedule.

**Tooling added** (crocoddyl_mpc, uncommitted at the time of writing):
`contact_select.TABLE_H` + `set_table_face()` -- env knob, default unset =
byte-identical, moves the slab and the object together and every table query
downstream follows because they all read the geom; `studies/height_reach.py` --
the (face x target) certification grid, `--modes` to pin one posture, `--dz`
face-relative or `--tz` absolute target, writes `cells.json` + `poses.npz`.


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

### 2026-09-08 20:22 — CODEX — delayed simulation investigation [IN FLIGHT]

Read latest state through 355a57ea / Result 7 at actual start. Isolated worktree:
`/home/humanoid/Programs/mujoco_mpc_codex_tableheight_20260908`, branch
`codex/table-height-skunkworks-20260908`. Main checkout and video.mp4 preserved.
Claim: bounded explicit base-height test (user still requests a dynamic test),
low-height closer stance, shorter target tests, audit of CMPC rollout dwell/load
and comparison conditions. Crocoddyl modules/data read-only; new comparison
scripts/results in the isolated MJPC worktree. No additional agents, DDS,
hardware, external messages, uploads or push. CPU-heavy runs serial and capped.
Final report and handoff will be linked here.

### 2026-09-08 20:32 — CODEX — planner-model copy defect [VERIFYING DYNAMICALLY]

`Agent::Initialize` (agent.cc:74) copies mjModel. `lean_bench` then applies
Table H / brace retarget only via Transition on the PLANT model. SamplingPlanner
holds agent.GetModel(), and no later bench call synchronizes it. Thus the
prior height sweeps appear to have planned with the nominal 0.985 m table and
shipped keyframes even while simulating another face. New bench prints both
faces/keyframe heights and has `--sync_planning_model` (default 1 in isolated
worktree; 0 reproduces legacy). First nominal reproduction completed 43.504 s
with 2.04 s actual jaw-error/load dwell. Direct gain=1 high pilot on legacy
copy fell at 33.59 s and is NOT evidence refuting a delivered base command.
Dynamic fixed-model trials are next. Earlier optimizer-only conclusions must
wait for this check. Also verified old CMPC uses MuJoCo 3.10 vs C++ 3.2.3 and
a different staged collision/joint-limit model. Matched tests use isolated
Python MuJoCo 3.2.3, MJPC's assembled model and exact jaw offset
(0.2254,-0.0118,-0.1062). Croco worktree:
`/home/humanoid/Programs/crocoddyl_mpc_codex_tableheight_20260908`.

### 2026-09-08 21:24 — CODEX — synchronized runs and comparison calibration

Measured: synchronized shipped-target high (1.085) completes at 40.490 s
with 2.00 s joint reach/load dwell, minimum error 1.5 mm (n=1). Shorter
target (0.85,-0.04,face+0.15), five-second dwell: nominal and high each
complete on seed 0, 4.98 s joint dwell. Low 0.785 succeeds at spp=10
(50 Hz), 4.98 s dwell / 27 mm best error / no fall / no torso load, but
return ladder not finished by 75 s. At spp=3 (actually 167 Hz), low stays
braced but misses the short hand target by at least 374 mm. Repeats pending.

Timing correction: spp multiplies PLANT dt=0.002, not prediction dt=0.010.
The old notes' spp3=33Hz is wrong. Nominal/high spp10 screening runs fall.

CMPC calibration: current MJPC model rotates ONLY right_magpie_gripper by
90 deg around x. Four initial CMPC calibration rollouts used the jaw-local
vector directly in the wrist frame, so they are NOT matched endpoint
comparisons. Kept and explicitly labeled. Added opt-in REACH_BODY and
RECONCILE_MJ_FRAMES to compose the welded frame in Pinocchio. Independent
parity now gives jaw ~1 micrometre, gravity error 1.5e-5 Nm, actuated mass
matrix 8.3e-7 kg m^2. Proper matched jaw runs are prepared next.

Direct base command is now verified delivered: plant and planner both show
brace base z=1.0142 at face1.085 with gain0.5 + pose_track1. That run is
active. Other queued MJPC probes: +60 mm low stance, low pose_track, .885
short target. Original checkouts still preserved. Live local report:
`/home/humanoid/Programs/mujoco_mpc_codex_tableheight_20260908/docs/lean/20260908-table_height_skunkworks.html`.

### 2026-09-08 21:46 — CODEX — low retargeting now works in the synchronized bench

Low .785, short target (.85,-.04,face+.15), pose_track=1, spp3=167Hz:
seed0 gives 2.82 s continuous joint loaded reach (2.10 s at <=30 mm),
5.4 mm best error, pelvis minimum .881 m, no fall through75s. Internal
five-second hand gate passes at36.22s, but actual brace load establishes
later, so do NOT call this five seconds of loaded reach. Recovery stalls
in phase3. +60 mm stance screen stays upright but misses by>=341 mm.

Retarget repeats and a high DLS pilot take priority over initially proposed
50Hz low/default-high repeats. Corrected CMPC jaw trials precede those in
our serial queue. Live counts/report and accurate checkpoint handoff are
in the isolated worktree docs/lean/20260908-table_height_skunkworks.html
and 20260908-codex_table_height_handoff.md.

### 2026-09-08 22:22 — CODEX — bounded CMPC comparison closed; MJPC repeats active

Core MJPC implementation committed locally at4398d8a8. Corrected DLS
high short-target pilot completes43.828s, 4.98s loaded reach /2.56s at
30mm. Low DLS seeds0/1 both pass: joint dwells2.82/5.18s, strict
2.10/4.28s, no fall through75s; both recovery sequences stall inphase3.
Low seed2 active, then nominal/high repeats and .935/1.035 screens.

Correctly mapped current-model CMPC: close jaw target x=.85 at low/nominal/
high all fell (low uses forearm-only mode); nominal x=1.15 elbow–forearm
also fell from home and stand_up. Five valid mapped transfer screens,
zero successful holds. Four earlier calibration failures stay separate.
Standing-start x=1.15 MJPC50Hz also fell before reaching. This branch is
closed; no further blind contact/weight sweep. Initial qpos/qvel parity is
exactly zero for all five mapped CMPC starts. Model bytes unchanged by
rebuild (same saved MJB SHA). Prior original-model CMPC height holds
remain separately valid, including3/3 at1.085 with2.9–4.7mm tail p95 error.

Live batch log /tmp/codex-stand-and-repeats.log; current manifest
studies/table_height/skunkworks/short_repeats.json in the isolated MJPC
worktree. Report remains a checkpoint pending final repeats. No push,
upload, DDS, hardware, or other external messages.

### 2026-09-08 23:09 — CODEX — investigation complete, local report and handoff

Report: `/home/humanoid/Programs/mujoco_mpc_codex_tableheight_20260908/docs/lean/20260908-table_height_skunkworks.html`
Handoff: same directory, `20260908-codex_table_height_handoff.md`.

Completed 25 MJPC and 9 CMPC episodes (4 CMPC endpoint-calibration runs
are explicitly separated). Fixed jaw target (.85, -.04, face + .15):
loaded holds within 30 mm for >=2 s succeeded in 2/3 trials at EACH of
.785, .985 and 1.085 m. At 70 mm the counts are 2/3, 3/3, 2/3.
Recovery counts are 0/3, 3/3, 2/3. Low/high use DLS keys; nominal is
the identity case. Low misses/recovery stalls and the high approach fall
remain in the report. Intermediate .885/.935/1.035 screens complete;
.935 only passes the 70 mm loaded-hold criterion. No envelope is inferred.

Main correction: bench planner model now receives the post-transition
table geometry AND retargeted keys. Core commit 4398d8a8. Explicit base
gain .5 + DLS passed one high far-target test, with no independent gain
advantage established. Synchronized +60 mm low stance missed the target.
Five correctly mapped current-model CMPC transfer probes all fell; prior
original-model CMPC height holds remain separately valid. Croco hooks
are in its isolated worktree at 51d604e / b1040fd.

All simulations ended; strategy file restored. Frozen selection hashes
442 raw files, and 242 local report links were checked. Build and Python
syntax checks passed; all five mapped CMPC initial qpos/qvel checks match
exactly. Original checkouts, video.mp4 and bvs_plots.py edits preserved.
No hardware, DDS, upload, push or external message was used. No blocker.
