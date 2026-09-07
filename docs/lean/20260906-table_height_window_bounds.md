# Table-height generalisation in the braced lean: what bounds the window

Branch `wxie/table-height`, cut from `icra2026` at `3e876d70`. Numbers below were
measured on 2026-09-05 and 2026-09-06 with `mjpc/lean_bench.cc`, 3 seeds per cell
at `--threads 6 --spp 3 --total_time 75`. Every table in this document is
reproducible from committed scripts; the commands are in §8.

## 1. State

The braced lean completes at slab faces **0.985 m and 1.035 m** and fails at
**0.785 m, 0.885 m, 1.050 m, 1.060 m and 1.085 m**. The upper edge is between
1.035 m and 1.050 m. Twelve arms have been run against that baseline —
eleven candidate levers and one deliberate diagnostic — and **none has widened the
window**. Two are disqualified for costing completions where the controller
already works: `brace_pitch_gain 100` drops 0.985 m to 2/3, and
`brace_pose_track 1` drops 1.035 m to 1/3.

**The high end is a shoulder-height deficit of about 90 mm.** The brace posture
does not lift with the table: measured over the brace rungs on upright frames, the
left shoulder sits 0.405–0.419 m above the slab face at 0.985 m and 0.250–0.321 m
at 1.085 m, dropping by the full 100 mm the slab rose. The robot holds the same
absolute posture and lets the table come up into its working volume. The pad's
clearance budget goes with it, from a maximum of +79 mm above the face at 0.985 m
to −12…+1 mm at 1.085 m, so **the pad never gets over the top face at all** and
carries 0.0–2.8 N through every brace rung of every arm (§4a).

That is why every lever tried has been null. Stance, aim, forward caps, base
height and reach posture all act on where the pad goes *horizontally*; the axis
that is short is vertical, and no term in the shipped cost commands it. The two
things that could supply the height — base z and trunk pitch — are asked for only
+18 mm and −8.7° by the per-slab retarget, and trunk pitch is the component a
posture cost cannot command at all (§6a).

## 2. The task, the bench, and the run policy

**Task.** `Lean H12 Magpie`, strategy slot 25 (`h12_brace_targeting.json`), a
9-rung phase ladder:

| rung | name | `reach_target_table` | tolerance |
|---|---|---|---|
| 0 | `stand_up` | — | 0.30 |
| 1 | `forearm_brace_lean` | — | 0.40 |
| 2 | `forearm_brace_lean` | [0.55, 0.04, 0.15] | **0.07** |
| 3 | `forearm_brace_release` | — | 0.30 |
| 4–7 | `standback_r1..r4` | — | 0.30 |
| 8 | `stand_up` | — | 0.30 |

There is **no manipulation rung**. Rung 2 is the task: put the right gripper jaw
tip within 70 mm of `(near_edge + 0.55, table_ctr_y − 0.04, face + 0.15)` while
the left forearm and wrist pads brace on the slab. Opening that tolerance is a
diagnostic and can never be a fix.

`Foot Left Up` and `Foot Right Up` carry weight **2000 on every one of the nine
rungs**, and `Right Foot Lift` is 0.0. The ladder has no stepping rung.

**Table height** is MJPC task parameter index 7, `residual_Table H`, the absolute
world z of the slab's top face. `0 = OFF = the compiled model, byte-identical`;
non-zero makes `lean::TransitionLocked` move the slab, restretch the cosmetic
legs, and shift the free object and `target` mocap by the same delta, so the
manipulation target stays fixed in the table frame. Compiled face is 0.985 m.

**Reset.** `lean_bench.cc:150`, `app.cc:423` and `deploy_common.cc:899` all reset
to the `home` keyframe, so where `home` puts the robot is a property of the
shipped configuration, not of the bench.

**Run policy, not optional.** MJPC is CPU-bound and a parallel sweep has hard-frozen
this workstation twice. Serial, `--threads 6`, under
`systemd-run --user --scope -p CPUQuota=700% -p MemoryMax=6G`; `sweep.py`
enforces it and refuses to start if the 1-minute load is already above `nproc/2`.
Budget 5–10 min per run. Three runs were SIGKILLed by the 6 GB cap during the
pitch sweep and had to be re-run serially.

**The planner is not deterministic.** Sampling/CEM, 20 rollouts, `spp 3`. The same
input 3.9 s apart gives different trajectories, and changing the thread count
re-rolls the RNG stream. Hold `--threads` fixed across every arm of a comparison
and never conclude anything from n=1.

## 3. The baseline window

3 seeds per height, shipped controller, nothing changed:

| face (m) | complete | failure mode | seated | forearm peak | CoM past toes |
|---|---|---|---|---|---|
| 0.785 | 0/3 | drapes on the slab | 13% | 191 N | +625 mm |
| 0.885 | 0/3 | falls forward (torso 106°) | 11% | 143 N | +371 mm |
| 0.985 | 3/3 | — | 32% | 234 N | +185 mm |
| 1.035 | 2/3 – 3/3 | marginal | 23% | 179 N | +34 mm |
| 1.085 | 0/3 | falls backward | 0% | 0 N | −85 mm |

Three distinct failure modes, not one. At 0.785 m the robot bows over the slab and
puts 398 N through the torso against 37 N on the forearm; it does not fall because
the table holds it up. At 1.085 m it never gets support at all.

1.035 m completed 3/3 at `--total_time 75` but only 1/2 at 140 s. Treat the upper
edge of the window as marginal.

**Seating is defined in newtons, not geometry.** A `pad_clear ≤ 5 mm` test scores
1.085 m as "100% at face level" while the pad carries 0 N, because the forearm
hangs outboard of the wood at face height. `analyze.py` requires contact; the
geometric measure is kept beside it as `at_face_fraction` because the gap between
the two is the diagnostic.

## 4. What bounds the high end

### 4a. The pad never gets above the slab's top face

Over the brace rungs, restricted to frames with an upright pelvis (z > 0.55 m,
because a toppling robot swings the pad through arbitrary heights), from the
logged qpos (`probe_padheight.py`). Three seeds per cell, three arms:

| face | max pad clearance | median pad clearance | shoulder above face |
|---|---|---|---|
| 0.985 m | +0.073 … +0.081 m | +0.042 … +0.063 m | **+0.405 … +0.419 m** |
| 1.085 m | **−0.012 … +0.001 m** | −0.011 … −0.021 m | **+0.250 … +0.321 m** |

(One 1.085 m seed under `brace_target_slab` touched +0.045 m; the other eight
1.085 m runs across three arms did not clear the face at any instant.)

Median forearm load through the brace rungs at 1.085 m is 0.0–2.8 N in every run
of every arm.

**The shoulder is pinned at a fixed absolute height and the clearance falls
linearly with the slab.** Across every arm that has dumps:

| face (m) | shoulder above face | shoulder, absolute | max pad clearance |
|---|---|---|---|
| 0.885 | +0.466 … +0.501 | ~1.37 m | +0.141 … +0.182 |
| 0.985 | +0.406 … +0.418 | ~1.40 m | +0.073 … +0.083 |
| 1.035 | +0.356 … +0.362 | ~1.40 m | +0.030 … +0.031 |
| 1.085 | +0.253 … +0.315 | ~1.40 m | −0.012 … +0.001 |

The slope of clearance against slab height is **−0.87**: the arm recovers 13% of
each millimetre the table rises and the posture supplies the rest of nothing.
Extrapolated to zero clearance, the upper edge is **1.07 m**.

**The 1.07 m prediction was tested and is refuted.** Three seeds each at 1.050 m
and 1.060 m, shipped controller, nothing changed: **0/3 at both**, all six falling
at 29–34 s. The upper edge of the window is therefore between **1.035 m and
1.050 m** — 50 mm above the compiled face, not 85 mm.

The clearance line itself survived the test exactly. Interpolated from 1.035 and
1.085 it predicts +0.019 m at 1.050 and +0.012 m at 1.060; measured, +0.019 and
+0.009. What is refuted is that positive clearance is *sufficient*. It is not:

| face | max pad clear | peak forearm load | outcome |
|---|---|---|---|
| 1.035 m | +0.030 m | 137–189 N | 2/3–3/3 |
| 1.050 m | +0.019 m | 93–113 N | **0/3** |
| 1.060 m | +0.009 m | 89–94 N | **0/3** |
| 1.085 m | −0.006 m | 0–3 N | 0/3 |

There are three regimes, not two. Above 1.085 m the pad never reaches the wood and
carries nothing. Between 1.050 and 1.060 m it does seat and load, to roughly half
the force the working heights carry, and the robot still topples. The brace needs
a clearance *margin* of roughly 25–30 mm to build the force that holds it, not
merely a non-negative number, so the fix in §7 item 1 has to buy back tens of
millimetres rather than just cross zero.

**The two ends are different defects.** At 0.885 m the pad has +164 mm of
clearance — more than at the compiled height — and still completes 0/3. Height is
abundant there; §5's balance failure is what stops it. The low end is not this
mechanism with the sign reversed.

### 4a-ii. Forward reach, the symptom rather than the cause

Best forward reach of the `left_forearm_pad` centre past the slab's near edge,
over the brace rungs (`probe_stance.py`):

| face (m) | shipped | + `brace_pose_track 1` | + `brace_pitch_track` (3 seeds) |
|---|---|---|---|
| 0.985 | +0.140 | +0.143 | +0.152, +0.107, +0.115 |
| 1.035 | +0.054 | +0.035 | +0.030, +0.035, +0.030 |
| 1.085 | **−0.066** | **−0.067** | **−0.085, −0.084, −0.084** |

The `forearm_brace_lean` keyframe puts the pad at **+0.110 m**. The series is
monotone in slab height and crosses zero between 1.035 m and 1.085 m, which is
where completions stop. At 1.085 m the pad stops short of the wood and 13–23 mm
below the face, so it carries 0.00 N in every run of every arm, and the run ends
with the CoM **1.07–1.14 m behind the forward foot edge**.

### 4a-bis. Stance and aim are both refuted, by direct test

**Stance.** `--stance_shift_x 0.063` (bench-only, §7) starts the robot 63 mm
closer, landing the feet at −0.199 to −0.203 m from the near edge, which is where
all three brace keyframes assume them. The pad did not move:

| 1.085 m | feet from edge | pad past edge | pad clear |
|---|---|---|---|
| shipped | −0.268 m | −0.066 m | −0.013 m |
| shift, seed 0 | −0.203 m | −0.069 m | −0.018 m |
| shift, seed 1 | −0.199 m | −0.077 m | −0.019 m |
| shift, seed 2 | — | never entered a brace rung | — |

0/3 at 1.085 m, no seed reaching rung 2, against 0/3 shipped with two seeds
reaching it. The feet moved exactly as commanded and the pad stayed at the same
offset from the slab.

**Aim.** `brace_target_slab` ships **0**, so `Brace Pos` aims the pad with the
legacy expression at `lean.cc:952`, `torso_x + 0.4*(table_centre_x − torso_x)` — a
convex combination of the *torso* and the table_top geom centre, which tracks the
body rather than the slab. Measured over the brace rungs, past the near edge
(`probe_bracetgt.py`):

| face (m) | torso | target | pad | pad − target |
|---|---|---|---|---|
| 0.985 | −0.235 | **+0.095** | +0.140 | +0.045 |
| 1.035 | −0.339 | **+0.033** | +0.054 | +0.021 |
| 1.085 | −0.444 | **−0.030** | −0.066 | −0.036 |

At 1.085 m the aim point is 30 mm in front of the near edge, off the wood, and
with `brace_press_depth` −0.044 under `brace_target_face` 1 it sits 44 mm above
the face — so the gradient pulls the pad into the slab's front face rather than
over the edge. The stance shift moved that target onto the slab in all three seeds
(+0.021 to +0.089) and the pad still did not follow, opening a 98–185 mm gap to
its own target. At this height the wood sets the pad's x.

### 4b. The robot never moves its feet

Median foot midpoint over the brace rungs, relative to the slab's near edge:

| face (m) | 0.785 | 0.885 | 0.985 | 1.035 | 1.085 |
|---|---|---|---|---|---|
| feet from edge (m) | −0.270 | −0.262 | −0.265 | −0.266 | −0.268 |

Flat to within 8 mm across the whole sweep. The feet stay where `home` puts them.

### 4c. The two keyframes disagree by 63 mm

Measured at the compiled face:

| keyframe | base x | feet x | feet from edge | pad from edge |
|---|---|---|---|---|
| `home` | +0.190 | +0.190 | **−0.260** | — |
| `forearm_brace_lean` | +0.212 | +0.253 | **−0.197** | +0.110 |
| `forearm_brace_reach` | +0.212 | +0.253 | −0.197 | +0.117 |
| `forearm_brace_release` | +0.212 | +0.253 | −0.197 | +0.147 |

The brace pose assumes a stance **63 mm further forward** than the reset provides,
with the feet 41 mm ahead of the pelvis rather than under it. Every rung pins both
feet at weight 2000, so the ladder cannot close the gap. This disagreement is
real and worth fixing on its own terms, but §4a-bis shows it is **not** what
bounds the height window: closing it changes nothing at 1.085 m.

### 4d. What the high end is not

| candidate | measurement | verdict |
|---|---|---|
| arm reach | rung-2 waypoint solves to <1 mm at every height with the pads seated and feet planted (`probe_reachset.py`) | not a reach limit |
| `Base Height` | delivered pelvis z tracks the brace keyframe to **+1 to +2 mm** at 1.085 m, ±25 mm everywhere | not binding |
| `com_cap_fwd` (0.145) | peak CoM excursion at 1.085 m is **+0.009 m** | not binding |
| `pelvis_cap_fwd` (0.13) | peak −0.047 to −0.059 m against the keyframe's −0.046 | not binding |
| `brace_lead_x0` (0.24) | base x reaches 0.207 m | not binding |
| release pitch gate | 11.4° of margin at 1.085 m | not binding |
| `stance_off_x` (0.13) | declared in `Lean_H12_Magpie.xml:1110`, read only by `beginning.cc:3703` and `stabilize.cc:2857`, **never by `lean.cc`** | dead numeric for this task |

## 5. What bounds the low end

**Correction to the previous write-up.** `STATUS.md` and `CLAUDE_wxie.md` §0 say
the low end is bounded by `standback_pitch_release` (0.50 rad = 28.6°), with a
predicted lower edge at 0.958 m. The pitch *requirement* is real geometry —
seating both pads needs 36.1° of base pitch at 0.885 m and 47.1° at 0.785 m — but
it is not what the runs die on. Phases reached, shipped controller, 3 seeds
(`0` = `stand_up`, `1`/`2` = the two lean rungs, `3` = release):

| face (m) | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 0.785 | `01` | `01` | `01` |
| 0.885 | `0` | `012` | `01` |

**No run at either low height ever reaches the release rung**, so the gate that
sits on it cannot be what stops them. At 0.885 m one seed never leaves `stand_up`.
Do not spend a sweep on `standback_pitch_release=0.70`.

What actually happens at the low end is a loss of balance during the lean itself:
the robot bows past the point where the feet can hold it, drapes onto the slab
(0.785 m) or falls forward (0.885 m). Base pitch measured over the brace rungs at
0.785 m reaches 29.9° shipped and 59.7° with `brace_pose_track 1` (seed 0, the
only seed with a qpos dump in the `ab` arms) — the pose tracking makes it bow
harder and it still fails.

The pitch requirement per height, for reference (`probe_pitch.py`):

| face (m) | 0.785 | 0.885 | 0.985 | 1.035 | 1.085 |
|---|---|---|---|---|---|
| required base pitch | 47.1° | 36.1° | 25.9° | 21.3° | 17.2° |

## 6. Levers tried and refuted

Ranked by the minimality rule in force: an existing model numeric beats a new
weight, a weight beats a new residual, a new residual beats moving the robot. A
result counts only if it widens the window on 3 seeds per height at fixed thread
count **and costs no completions at 0.985 m**.

| # | lever | numbers touched | what it changed mechanically | outcome |
|---|---|---|---|---|
| 1 | `brace_com_hold` 0.05 | 1 numeric | mean survival at 1.085 m 32.6 s → 47.8 s, non-overlapping sets | 0/3, buys time only |
| 2 | `Brace Reach Lead` 400 | 1 weight | forward base travel followed the commanded line | window unmoved |
| 3 | rung-2 tolerance 0.07 → 0.20 | 1 number (**diagnostic**) | ladder ran end to end at 1.085 m | proves the gate is the discriminator, not a fix |
| 4 | `brace_pose_track` 1 | 1 numeric | re-solves the brace keyframes per slab by damped least squares, seating both pads to 0.1 mm offline at every height | window unmoved, **costs 1.035 m** (2/3 → 1/3) |
| 5 | `brace_pose_track` 2 | 1 numeric | same, reaching arm included | 0/2 at 0.885 m |
| 6 | `reach_arm_posture` | 1 numeric | offline aim error 303/200/98/82/105 mm → 0–6 mm | neutral; disqualified when paired with #4 |
| 7 | `pelvis_tilt_max_deg` 50 | 1 numeric | caps the bow | 3/3 at 0.985 m, 0/3 at both low heights |
| 8 | `brace_pitch_track` 1 | 2 numerics (new residual term, default-off) | see §6a | window unmoved |
| 9 | `brace_pitch_gain` 20 | 1 numeric | pitch 12.1° → 14.2° at 1.085 m | **all 3 seeds fall at ~14.3 s**, never reach rung 2 |
| 10 | `brace_pitch_gain` 100 | 1 numeric | pitch 12.1° → 19.8° at 1.085 m, within 2.6° of the 17.2° target | **falls at ~14.1 s; costs 0.985 m (3/3 → 2/3)** |
| 11 | `--stance_shift_x` 0.063 | 1 number (bench-only) | feet land at −0.199…−0.203 m from the edge, where the brace keyframes assume them | pad unmoved (§4a-bis); 0/3 at 1.085 m, 0.985 m held |
| 12 | `brace_target_slab` 1 | 1 numeric | aim moves from 30 mm in front of the near edge to 50 mm onto the slab | 0/3 at 1.085 m and 0.785 m; **3/3 at 0.985 m** |

Outcome matrix, 3 seeds per cell:

| face | shipped | `pose_track 1` | `pitch_track` g1 | g20 | g100 | `tilt_max 50` | `stance 63` | `target_slab` |
|---|---|---|---|---|---|---|---|---|
| 0.785 | 0/3 | 0/3 | 0/3 | — | — | 0/3 | — | 0/3 |
| 0.885 | 0/3 | 0/3 | 0/3 | — | — | 0/3 | — | — |
| 0.985 | 3/3 | 3/3 | 3/3 | — | 2/3 | 3/3 | 2/2 | 3/3 |
| 1.035 | 2/3 | 1/3 | 0/3 | — | — | — | — | — |
| 1.085 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | — | 0/3 | 0/3 |

### 6b. What `brace_target_slab` did change

It is free at the compiled height (3/3, and marginally faster: 40.7/44.8/41.0 s
against 44.2/45.1/47.8 s shipped) and it moves the aim from 30 mm in front of the
near edge to 50 mm onto the slab. Two consequences, neither a fix:

- **At 1.085 m the ladder now reliably reaches rung 2** — all three seeds enter it
  at 24.0–24.3 s, against 1 of 3 shipped — and survives 33.8–47.3 s. The pad
  advances from −0.066 to −0.039/−0.043/−0.076 m past the edge, roughly a third of
  the 80 mm the aim moved, and still stops 89–126 mm short of its own target
  because the wood is in the way. Clearance is unchanged.
- **At 0.785 m it is worse.** Two of three seeds now fall (30.5 s, 17.6 s) where
  all three shipped seeds stalled to the 75 s cap. Median torso load does drop to
  0.0 N in all three, but that is because they fall before the drape develops, not
  because it resolved: median forearm load is 0.0 N in 3/3, the same as the
  shipped 6/6. H2 as written was not a sound test — peak torso force at 0.785 m
  ranges from 237 N to 12.8 kN across shipped seeds, so the drape has to be scored
  on forearm load, which never leaves zero.

### 6a. Why `brace_pose_track` cannot work, and what `brace_pitch_track` was

`brace_pose_track` re-solves the brace keyframes for the actual slab and seats
both pads to 0.1 mm offline. It changes nothing in the runs because the largest
component of that retarget lives in the **floating base**, which a posture cost
cannot command (`probe_base_split.py`):

```
 face | retarget delta:  base_x   base_z   pitch | joint |dq| max/rms | pads with base PINNED (mm above face)
 0.785 |  -0.052  -0.047  +21.3 | 0.494/0.149 | forearm +204.6  wrist +275.3
 0.885 |  -0.025  -0.027  +10.2 | 0.270/0.080 | forearm  +98.8  wrist +137.2
 0.985 |  +0.000  +0.000   +0.0 | 0.000/0.000 | forearm   +2.1  wrist   +9.4
 1.035 |  +0.001  +0.011   -4.6 | 0.139/0.043 | forearm  -38.9  wrist  -45.8
 1.085 |  -0.005  +0.018   -8.7 | 0.302/0.086 | forearm  -75.0  wrist  -94.3
```

Reapplying only the joint half leaves the pad 75 mm below the face at 1.085 m and
205 mm above it at 0.785 m.

Base pitch also has no gradient on the lean rungs in the shipped controller: the
`Pelvis Tilt` residual is a one-sided dead band that is free from 0–60° whenever
an arm is in contact, `Torso Forward Tilt` is yaw-only, and `Brace Erect` is gated
to `forearm_brace_release` with weight 0.0 on the lean rungs.

`brace_pitch_track` (`lean.cc` ~line 1692, gated by a model numeric, `0 = OFF =
byte-identical`) replaces that dead band on `forearm_brace_lean` rungs with a
signed error against the keyframe's own pitch, scaled by `brace_pitch_gain`. It
works as designed — median base pitch at 1.085 m is monotone in gain: 12.1°
(shipped), 10.5° (`pose_track`), 8.1° (gain 1), 14.2° (gain 20), 19.8° (gain 100),
against a required 17.2°. Gain 1 was too weak by construction: a 7° error at gain 1
contributes 2.9 against a rung-2 total cost of ~1740, or 0.17%. At gains that do
bite, the robot commits to the bow before it has the pad over the slab and falls
two seconds into the lean rung.

## 7. What is still open

Ranked. Each entry names the measurement that settles it.

1. **Command the shoulder height at the tall slab.** This is the only candidate
   whose axis matches §4a. The measurement says the shoulder needs about 90 mm it
   is not getting, and the two sources are base z and trunk pitch. `Base Height`
   is a live residual whose delivered value tracks its command to 2 mm, so the
   command is what is wrong — the brace keyframe. The cheapest test is to raise
   the brace keyframe's base z directly, per slab, rather than through the
   retarget that only asks for +18 mm: add a numeric `brace_base_z_gain` scaling
   `(face − 0.985)` into the keyframe's base z under the existing
   `brace_pose_track` machinery, which already writes `model->key_qpos`
   (`lean.cc:4380`). One numeric, default 0 = byte-identical. **Confirmed** if
   1.085 m max pad clearance goes positive and rung-2 forearm load leaves zero.
   **Killed** if 0.985 m loses a completion, or if the shoulder rises and the pad
   does not follow — which would mean the arm posture, not the base, holds it.
2. **Where between 1.035 m and 1.050 m the edge sits, and what the force threshold
   is.** Both questions were opened by the refuted 1.07 m prediction (§4a). Three
   seeds at 1.040 m and 1.045 m locates the edge to 5 mm and, with the clearance
   and peak-force numbers already tabulated, says whether completion tracks a
   clearance threshold (~25 mm) or a force threshold (~130 N). That distinction
   decides how much height item 1 has to buy. 6 runs, ~35 min.
   *(Item 2 as previously written — whether the low end is the same defect with
   the sign reversed — was answered on 2026-09-06 and is now in §4a: it is not.
   At 0.885 m the pad has +164 mm of clearance and still completes 0/3.)*
3. **The 0.905–0.935 m band has never been run under any arm.** Three seeds at each,
   shipped controller, 6 runs (~35 min at the corrected memory cap). Bounds the
   lower edge to 50 mm and tests whether §5's balance story predicts it.
4. **A stepping rung.** `Foot Left Up`/`Foot Right Up` at 2000 on all nine rungs is
   what makes the 63 mm keyframe disagreement structural. Dropping the weight on
   rung 1 only, with `Right Foot Lift` re-enabled, is a 2-number change to the
   strategy JSON and needs no rebuild. Low priority: §4a-bis shows stance does not
   bound the window, so this fixes a real defect that is not the one in the way.
5. **`brace_target_slab` is worth proposing upstream on its own merits**, separately
   from this study. It is free at the compiled height, it makes the brace aim track
   the slab instead of the torso, and it takes rung-2 entry at 1.085 m from 1 of 3
   to 3 of 3. It does not widen the window, so it is not an answer to this
   question, but leaving the aim tied to the body is a latent sim-to-real defect
   of exactly the kind the ground-truth rule exists to prevent.

Not planned: `brace_erect_target` and the remaining brace-geometry constants
fitted at 0.985 m — §4a says the pad is not near the slab at 1.085 m, so a term
that shapes contact once seated cannot fire. `brace_press_depth` is a special case
and is **already fixed**: it ships at **−0.044** under `brace_target_face` 1, i.e.
the press target sits 44 mm *above* the face, not the 60 mm below that its own
comments in `lean.cc` still describe. The z half of the buried-target defect has
been repaired; the x half was `brace_target_slab`, tested here.

## 8. Reproducing every number here

```bash
cd mujoco_mpc
ninja -C build_cmake -j4 lean_bench
S=studies/table_height

# one run
build_cmake/bin/lean_bench --task "Lean H12 Magpie" --strategy 25 \
  --table_h 1.085 --seed 0 --total_time 75 --threads 6 --spp 3 \
  --out r.csv --qpos_out r.qpos.csv

# §4a-4c  stance, pad reach, keyframe gap        (no sim time, replays qpos)
$S/probe_stance.py --runs shipped=$S/runs/ab/off pose1=$S/runs/ab/on \
    pitch=$S/runs/pitch/pitch g100=$S/runs/pitch/g100 --json $S/figs/stance.json

# §5  pitch requirement per slab                 (no sim time)
$S/probe_pitch.py --json $S/figs/pitch.json
# §6a base/joint split of the retarget           (no sim time)
$S/probe_base_split.py
# §4d rung-2 reach set with the pads seated      (no sim time)
$S/probe_reachset.py --json $S/figs/reachset.json

# §4a  pad height and shoulder height          (no sim time)
$S/probe_padheight.py --runs shipped=$S/runs/ab/off slab=$S/runs/bracetgt/slab
# §4a-bis  where Brace Pos aims, vs where the pad gets   (no sim time)
$S/probe_bracetgt.py --runs shipped=$S/runs/ab/off       # legacy aim
$S/probe_bracetgt.py --runs slab=$S/runs/bracetgt/slab --slab_inset 0.05

# §6  the arms.  SWEEP_MEM_MAX matters: lean_bench needs ~9.9 GB at these
# settings and the 6G default silently pages the difference to swap.
$S/sweep_ab.py       --out $S/runs/ab        # pose_track off vs on, one binary
$S/sweep_pitch.py    --out $S/runs/pitch     # brace_pitch_track, gain ladder
$S/sweep_tilt.py     --out $S/runs/tilt      # pelvis_tilt_max_deg 50
SWEEP_MEM_MAX=11G $S/sweep_stance.py   --out $S/runs/stance    # --stance_shift_x
SWEEP_MEM_MAX=11G $S/sweep_bracetgt.py --out $S/runs/bracetgt  # brace_target_slab
$S/analyze_pitch.py                          # the six-arm comparison + figures
```

`lean_bench` takes `--numeric <name>=<value>` for any numeric that already exists
in the model (it prints `[bench] <name> = <value>` and exits 2 with
`no such numeric` otherwise) and `--pose_track <0|1|2>`. That is how every arm in
§6 was run from one binary, which is what makes the comparisons paired.

Raw runs under `studies/table_height/runs/` are gitignored and regenerable.

## 9. Traps that have already cost time

- **`brace_force` in `lean::ComputeMetrics` reads `right_contact[0]`** while the
  model braces with `left_forearm_pad` / `left_wrist_pad`. It reports 0.00 N
  through a real brace in which the left forearm carried 97–168 N. Not yet sent to
  Allen. Does not affect the controller, only the metric and the deploy monitor.
- **`reach_err` / `reach_tgt_*` are gated on `kf.name == "reach_to_target"`**, a
  rung name strategy 25 never uses, so the whole family is nan for any braced
  ladder. Same status.
- **Contact summing counts the table's own weight** when the table legs touch the
  floor. `lean_bench.cc` excludes body 0 and the free `object`; do not undo that.
- **Load statistics must be taken over the seated window**, not the whole brace
  phase. Most of the phase is approach, and a whole-phase median reads 0 N while
  the forearm peaks at 234 N.
- **`render_video.py` needs a real `$DISPLAY`.** EGL and OSMesa both fail on this
  box; glfw works on `:1`, not `:0`. It also re-applies the table height, because
  the model on disk always shows the nominal slab and a naive replay draws the
  robot bracing on air.
- **`pgrep -f` matches its own command line.** A wait loop spun for 75 minutes
  instead of running the next sweep. Use `pgrep -f "[c]hain\.sh"`.

## 10. Branch discipline

Work on `wxie/table-height`, never on `icra2026` — Allen owns `lean.cc` and
commits to `icra2026` most days. Rebase onto it rather than merging. Every edit to
a file Allen owns is additive and default-off (`residual_Table H`,
`brace_pose_track`, `brace_pitch_track`, `pelvis_tilt_max_deg` all default to 0 and
reproduce the compiled model byte for byte), which is what makes the branch cheap
to rebase and safe to propose upstream. Nothing has been pushed.
