# H1-2 task (lean/stabilize) tuning & decision history

Dated tuning history, A/B verdicts, and post-mortems migrated out of the
lean/stabilize task code comments (2026-07-18 comment policy: code carries
current-truth invariants only; chronology lives here). Verbatim at migration time.


## Phase-ramp duration: 3.0s trial rejected

*Migrated 2026-07-18 from `stabilize.h ~111 (kPhaseRampSeconds)`:*

```
// Tried bumping to 3.0 to slow the arm swing during 2→3 — backfired:
// with MPC horizon 1.0s the lean_forward gradient stayed weak for ~2s
// while Height (head wants to stay high, weight 35 effective in
// arm_contact_or_lean) was full strength — body settled into a slight
// backward bend as the cheap local optimum. If arm swing is still too
// fast at 1.5s, a surgical fix (Brace Hand Velocity residual active
// only during arm_plant) is preferable to slowing every cost ramp.
```

## Asymmetric target ramp: ascent 3.0s rationale (constant deleted as dead)

*Migrated 2026-07-18 from `stabilize.h ~120 (deleted constant kAscentTargetRampSeconds)`:*

```
// STAND-UP-only target-pose ramp duration (see the asymmetric target ramp
// in stabilize.cc). Deliberately LONGER than kPhaseRampSeconds: at the live
// ~60/s plan rate, straightening the legs from a crouch over only 1.5s
// still launches the body backward on the second cycle (squat fell ~28s).
// Spreading the leg extension over 3s lets the sampler keep the capture
// point under the feet the whole way up. Only used when a phase transition
// moves the target pose CLOSER to home (standing up); descents still snap.
static constexpr mjtNum kAscentTargetRampSeconds = 3.0;
```

## Asymmetric target ramp: descent 0.6s rationale (constant deleted as dead)

*Migrated 2026-07-18 from `stabilize.h ~129 (deleted constant kDescentTargetRampSeconds)`:*

```
// Crouch-DOWN target-pose ramp duration. Much SHORTER than the ascent ramp:
// a pure snap folds the legs so fast the upper body overshoots into a
// forward pitch (the recurring squat descent fall), but the slow ascent
// ramp on a descent lets the robot catch the target and kills the
// stabilising spring (forward pitch at ~2.6s). 0.6s threads the needle: the
// target still leads the robot (spring preserved) yet the fold is spread
// over ~0.6s instead of instantaneous, capping the overshoot. So short that
// single-phase pose strategies (which settle for seconds) are unaffected.
static constexpr mjtNum kDescentTargetRampSeconds = 0.6;
```

## Stale lean.cc contrast: cmd_active_ rollout propagation (lean was fixed 2026-07-12; contrast deleted)

*Migrated 2026-07-18 from `stabilize.h ~193 (cmd_active_ member comment)`:*

```
// ★ cmd_active_ + cmd_vdes_world_ ARE propagated into every rollout copy by
// ResidualLocked (unlike lean.cc, which propagates only drive_gait_amp_/
// drive_yaw_des_ -- so ITS rollout residuals silently see cmd_active_=false
// and cost the gait as an IN-PLACE trot while ModifyControl drives the swing
// FORWARD at the governed v_des). That disagreement makes the sampler fight
// the open-loop walk drive on every rollout, which is the same signature as
// the documented "free-twin ~6s walk ceiling". Propagating them is what the
// "MUST match ModifyControl" comments in the residual actually require.
```

## Copy-pasted LEAN slider layout + 2026-05-26 leg-lift design note

*Migrated 2026-07-18 from `stabilize.h ~336 (above GetStrategyNames)`:*

```
// Slider layout (Lean H12) — user's 6-phase decomposition:
//   0  stand            — stand_up
//   1  arm_extend       — stand → arm_extend_standing (arm out, body upright)
//   2  lean_no_brace    — stand → extend → lean_with_arm_no_brace
//   3  brace_hand_lean  — stand → extend → stabilize → arm_plant → lean_forward
//   4  forearm_brace    — above + forearm_brace_lean (hand+elbow on table)
//   5  full_pipeline    — identical to slot 4 now: ends in a HELD two-foot
//                         braced stabilize (DEFAULT).
//
// DESIGN (2026-05-26): the leg-lift phase (leg_lift_arm_plant) is DROPPED
// permanently. BOTH feet stay stable on the ground through EVERY phase of
// the pipeline. The only lower-body motion allowed is WBC-driven foot
// re-placement / hip twist IN SERVICE OF the brace (to hold balance while
// reaching/leaning) — never lifting a leg off the floor. No strategy JSON
// contains leg_lift_arm_plant anymore, so slot 5 == slot 4.
//
// Each slot is a literal truncation of the index-5 pipeline with the
// last phase forced indefinite (sustain/time_limit = 9999).
```

## Leg-lift drop chronology (ITER 36 / DESIGN 2026-05-26)

*Migrated 2026-07-18 from `stabilize.cc ~175 (Residual, leg-lift stage detection)`:*

```
// ITER 36 (2026-05-18): is_leg_lift detection is PHASE-NAME based (keyed on
// the active keyframe name), not contact-count based.
//
// DESIGN (2026-05-26): the leg-lift phase is DROPPED permanently — both feet
// stay grounded through the whole pipeline (see the lean.h header). NO
// strategy JSON contains "leg_lift_arm_plant" / "deep_reach" anymore, so
// is_leg_lift_stage_early is now ALWAYS FALSE and every `is_leg_lift_stage`
// branch below is DORMANT (kept as vestigial history, not removed) — the
// grounded branch (both feet anchored) always runs. Any future single-
// support move must come from a WBC/balance decision, not this hard gate.
```

## Target-pose ramp revert (2026-06-08) and gated re-enable (2026-06-16)

*Migrated 2026-07-18 from `stabilize.cc ~241 + ~264 + ~275 (Residual, posture target ramp)`:*

```
// REVERTED 2026-06-08: the squat target-pose ramp (asymmetric stand-up/crouch
// ramp) is OFF — restored to the original instantaneous SNAP. The ramp ran in
// this hot residual path for EVERY strategy (incl. single-phase crouch/stand/
// arms) and was suspected of regressing the live crouch (one-knee-locked slow
// creep) even though headless held. Snap = exactly the accepted behaviour.
// The prev_posture_key_id_ plumbing (lean.h, SnapshotEffectiveScales) is left
// in place but UNUSED; re-enable the ramp here only behind a per-strategy gate
// that provably never touches single-phase strategies. See [[project_squat_strategy]].
[header of the re-enabled block:]
// ----- GENERAL target-pose ramp (re-enabled 2026-06-16, behind the per-strategy
//       gate the 2026-06-08 revert note above requires) --------------------- //
[closing clause:]
// -- the unconditional version
// that ran for EVERY residual call of EVERY strategy is what got reverted.
[related sentence removed from stabilize.h num_phases_: "This is the per-strategy gate the 2026-06-08 revert note (stabilize.cc) said the ramp needed."]
```

## Brace Y-clamp test-14 revert (superseded by the 2026-05-20 shoulder-pinned Y)

*Migrated 2026-07-18 from `stabilize.cc ~754 (Residual, bracing position)`:*

```
// Bracing position calculation. Reverted Y-clamp (was test 14) → back
// to bracing_hand[1] (test 12 state). User confirmed test 14 introduced
// chaotic early-phase behaviour. Y free means no restoring force on
// lateral position; the eventual ~60s slip seen in test 12 is the
// known trade-off for accepting this baseline.
```

## Forced static counterweight investigated and rejected (2026-06-23)

*Migrated 2026-07-18 from `stabilize.cc ~812 (counterbalance_standing reach target)`:*

```
// INVESTIGATED 2026-06-23: a deliberate static counterweight (free arm swung
// back +/- knee bend) was trialed to add forward-push margin, but EVERY forced
// posture override (arm-back >= 0.30 rad OR knee-bend 0.18) toppled it BACKWARD
// during lean establishment -- the planner's EMERGENT counterbalance is already
// optimal and a forced pose disrupts it. Kept the validated emergent behavior
// (twin + GUI 3/3). The forward-push fragility of this FREE-STANDING no-brace
// lean is inherent (planted feet, no step); the push-robust paths are BRACING
// on the surface (strat 33/34) or STEPPING (strat 20), not strat 16.
```

## Height x0.35 during arm contact: TEST #8 chronology

*Migrated 2026-07-18 from `stabilize.cc ~1035 (Height residual)`:*

```
// TEST #8 (2026-05-17): Restore Height ×0.35 during arm contact
// (test 4's working setting). Combined with the foot-z anchors added
// in test 6 (Right Foot Lift penalises lift; Left Leg Anchor enforces
// both left knee height AND left foot on ground), the squat is the
// stable solution but now with feet anchored. Tests 6 (×0.7) and 7
// (×1.0) both failed because the planner can't stand-and-bend with
// kp_ankle=20 — it either tips forward (×0.7) or backward (×1.0).
// The squat IS the natural solution given the weak PD.
```

## Capture-point horizon 0.45s preview trial reverted (2026-05-26)

*Migrated 2026-07-18 from `stabilize.cc ~1112 (capture point)`:*

```
// Horizon kept at 0.3 s. A 0.45 s preview was trialed (2026-05-26) to fight the
// forward-velocity overshoot at brace commit, but a 10-run trace showed it did
// NOT improve the hold rate (still 8/10) and produced lower-quality holds (one
// barely-leaning +2 deg, one drifting -8 deg lateral), so it was reverted.
```

## Pelvis tilt: Round 7 / Round 8 directional form / bisection revert to symmetric

*Migrated 2026-07-18 from `stabilize.cc ~1531 (pelvis tilt residual)`:*

```
// Phase-dependent residual that mixes two sensors:
//  • pelvis_up[2]      = cos(tilt magnitude) — symmetric in direction
//  • pelvis_forward[2] = -sin(pitch angle)   — DIRECTIONAL (forward = -,
//    backward = +)
// Round 7 fix used `pelvis_up[2] - target` everywhere, which is symmetric
// for any tilt direction. During lean_forward (target 0.85 = 32° tilt),
// backward tilt achieved the same target as forward tilt, and MPC chose
// backward to avoid the Hip-Clearance penalty on forward pelvis travel —
// the user saw the robot stand → plant hand → tip BACKWARD → fall on its
// back. Round 8 fix: switch the lean-phase branch to pelvis_forward[2]
// so the target -sin(32°)=-0.530 is only met by forward pitch; backward
// pitch gives residual +1.060 (cost ~9.6/step ≈ 640/horizon — overwhelming
// deterrent vs the ~3 Hip-Clearance penalty MPC was previously avoiding).
// Upright phases keep pelvis_up because the home pose is at pelvis_up=1.0
// so the residual is already 0 there and roll is also penalized.
// BISECTION TEST #2 (2026-05-17): reverted to symmetric residual
// ALWAYS. Pre-R8 form. The directional `pelvis_forward[2] − (−0.530)`
// in lean_forward was forcing 32° pitch regardless of CoM state; with
// weak ankle PD (kp=20) this over-committed the lean and let the planner
// exploit the pelvis-table exclude. Symmetric is permissive: any tilt
// (forward or backward) costs the same — MPC picks based on other terms.
```

## Arm lock on counterbalance_standing: rejected 2026-06-25, re-enabled 2026-06-26 with explicit counterweight

*Migrated 2026-07-18 from `stabilize.cc ~1693 (non-reaching arm pin)`:*

```
// 2026-06-25: TRIED extending this lock to counterbalance_standing (strat 16) so
// 16's right arm would also hold forward instead of swinging back to -0.8 m. A/B on
// the deploy twin (gravcomp 0.85, seed 0) REJECTED it: WITH lock 16 fell at 14.3 s,
// WITHOUT lock at 23.5 s — the lock makes 16 topple ~9 s SOONER. 16's free-standing
// lean USES the reaching arm as part of its emergent counterbalance; locking it
// steals that DOF and it falls backward earlier (the documented "forcing the arm
// topples 16" finding). So the lock stayed reach_to_target (strat 21) ONLY.
// 2026-06-26: RE-ENABLED for counterbalance_standing (16) — the prior rejection
// locked the right arm with NO replacement counterweight. Now the
// counterbalance_standing keyframe drives the LEFT (non-reaching) arm fully BACK
// (qpos[20]=+1.3); brace_hold pins it there as an EXPLICIT counterweight, so the
// right arm can hold forward AND the body stays balanced (the two arms counterweight
// each other -> CoM centered -> ankle unloaded -> both extend). User's figure-skater
// insight; supplies exactly the balance DOF the bare lock removed.
```

## Ankle-action tax: 2026-07-11 measured history (rails, hip/ankle price ratio)

*Migrated 2026-07-18 from `stabilize.cc ~1787 (Control residual, ankle tax)`:*

```
// ANKLE-ACTION TAX (2026-07-11, hip-strategy rebalance): the free-standing cost
// set prices a hip correction ~50x above an ankle one (AngMom + linear Pelvis
// Tilt + the 3.5x hip-pitch Posture anchor all tax the hip throw; NOTHING taxes
// ankle action while the foot stays flat), so the sampler does ALL fore-aft
// correction on the ankle channel -- exactly the joint with the per-power-on
// zero-lottery error and the 60/54 Nm rail (real stand parks forward, hunts,
// rails LankP at 80-89%). Multiply ONLY the ankle rows (nu-idx 4/5 L, 10/11 R)
// of the Control residual: sustained ankle targets get expensive, correction
// authority shifts to the hips. kCosh is ~quadratic near 0 -> effective ankle
// control weight scales ~gain^2. <numeric name="ankle_ctrl_gain"> 1.0 = OFF
// (byte-identical). Free-standing STAND-family only: brace tasks 0-5 unchanged,
// trot/stumble keep free ankles for stepping.
[NOTE: the 'ankle rows (nu-idx 4/5 L, 10/11 R)' claim was stale -- the code taxes pitch rows {4, 10} only; fixed in place.]
```

## kStepHeight comment before the 0.022->0.06 bump

*Migrated 2026-07-18 from `stabilize.cc ~2549 (Cartesian gait block)`:*

```
// Cartesian step height TARGET: kStepHeight (0.022) scaled by the trot-window
// swing scale (default 1 outside the window). At swing_scale 2.5 the target is
// ~0.055 m -- between the quadruped trot (0.03) and Unitree humanoid walk
// (0.08) clearance, sized for the bigger H1-2 foot.
```

## Staggered stance tested and reverted (2026-06-18)

*Migrated 2026-07-18 from `stabilize.cc ~2655 (Step Place, kStagL/kStagR)`:*

```
// NOTE (2026-06-18): a STAGGERED/braced stance (L foot fore, R foot aft, via a
// staggered stumble_march keyframe + kStagger here) was TESTED to ~double the
// fore-aft support polygon (0.26->0.46 m). It improved the static baseline (4/4)
// but did NOT improve fwd/back PUSH survival and slightly hurt lateral -> REVERTED.
// Root cause: fore-aft recovery is reaction-BANDWIDTH limited (the planner doesn't
// exploit the bigger base / use the ankle headroom fast enough at spline 5), not
// support-geometry limited. The remaining principled lever is the HIP/arm
// angular-momentum strategy (ankle->HIP->step hierarchy) — unbuilt. The foot is
// already reference-grade fore-aft (G1 0.17 m / H1 0.20 m) so do NOT lengthen it.
```

## Foot Slip R7 stand anti-shuffle arc (2026-07-10)

*Migrated 2026-07-18 from `stabilize.cc ~2695 (Foot Slip, non-gait branch)`:*

```
// R7 (2026-07-10, stand anti-shuffle): Foot Slip was gait-only; the free
// STAND wrote 0 here, so sliding a planted foot was a FREE extra DoF the
// sampler used to null CoM error every correction cycle -- the measured
// real-robot shuffling (research doc 2026-07-09 §2.7/§3). Write the real
// tangential speed of BOTH feet (double support: both are stance). Cost
// stays 0 for every strategy whose JSON does not set "Foot Slip" (XML
// default weight 0) -> placeholders/lean phases byte-identical; the stand
// JSON opts in at 25 (same weight the stumble/trot/walk JSONs use).
```

## ITER 26 equality-weld removal (2026-05-18)

*Migrated 2026-07-18 from `stabilize.cc ~2897 (TransitionLocked header)`:*

```
// -------- Transition for humanoid_bench lean task -------- //
// ------------------------------------------------------------ //
//
// ITER 26 (2026-05-18): removed the iter-23 equality-weld toggle. Single-
// point/SE3 pins on the foot felt unnatural (sway-spring behaviour); foot
// anchoring is now done by real physics (gravcomp 0.97 → 0.90 gives ~49 N
// of net body weight on each foot, enough friction to hold against the
// soft cost gradients without artificial pins).
```

## DC-washout 2026-07-11 real A/B failure signature

*Migrated 2026-07-18 from `stabilize.cc ~2906 (DC-blind baseline intro)`:*

```
// ---- DC-BLIND HIP RECOVERY baseline (2026-07-11) ------------------------- //
// With an ankle zero error the robot PARKS off-vertical (+4..6 deg fwd = cap
// excursion ~0.09 m), so an ABSOLUTE-excursion recover tier sees the park as a
// permanent "falling" signal and throws counter-momentum continuously -- the
// 07-11 real A/B: backward overshoot (CoM_margin -0.165) -> fwd flail -> crouch
// jam. Fix: EMA the MEASURED excursion here (real state, once per plan) and let
// the recovery tier react only to the deviation FROM that baseline (escapes),
// never the standing offset. stand_recover_washout_sec = EMA tau; 0 = off.
```

## T1 trim symmetric-cap post-mortem (C2pure live)

*Migrated 2026-07-18 from `stabilize.cc ~2940 (T1 trim caps)`:*

```
// T1 trim: lean park_dc -> Balance fore-aft bias. C2pure live: symmetric
// ±stand_trim_max let bring-up wind trim_x to -0.08 and leave it there
// after park_dc~0 -> permanent "hold CoM forward" (operator push from front).
// Asymmetric: +side still uses stand_trim_max; -side uses stand_trim_neg_max.
```

## Excursion deadband: the 07-08 gain-300 A/B failure

*Migrated 2026-07-18 from `stabilize.cc ~2421 (stand recover deadband)`:*

```
// EXCURSION DEADBAND (2026-07-11): the 07-08 gain-300 A/B failed because
// g_cap_ex is NOT ~0 at the nominal stance (XML:484 post-mortem) -- the
// gain made a PERSISTENT angmom target -> continuous hip pumping + fwd
// drift. Only the excursion BEYOND the deadband commands counter-momentum;
// inside it the target is exactly 0, so the calm stand is byte-identical
// to gain 0. stumble keeps its tuned catch-march path unchanged.
```

## Catch latch v5.2: why catch_march_thresh is its own numeric

*Migrated 2026-07-18 from `stabilize.cc ~3124 (catch-march latch threshold)`:*

```
// march latch threshold is its OWN numeric (v5.2): reusing catch_full
// coupled the latch to the legacy COST-side overlay band (trig..full)
// -- lowering it to 0.07 for the latch collapsed that band and sent
// the cost side into a full march on ANY 0.07+ danger while the freeze
// only played backward => fwd 0.5 fell (half-machinery again).
// catch_trig/catch_full stay at their validated 0.12/0.24.
```

## Catch latch v5.1 backward-only rationale + v5.4 dominance-condition revert (measured matrices)

*Migrated 2026-07-18 from `stabilize.cc ~3146 (catch-march latch condition)`:*

```
// BACKWARD-DOMINANT ONLY (v5.1, 2026-07-03): the march latches solely
// on a backward capture escape (-ex). Backward is the one axis whose
// support polygon is structurally deficient (heel lever 0.035 m vs toe
// 0.115 m, lateral hip load/unload bulletproof); fwd/lateral already
// recover QUIETLY (hip-throw + capture-lateral + weight-shift), and the
// 8-cell matrix showed an omni latch REGRESSES them (fwd 0.5 RECOVER
// 6.4deg -> DRIFT 29deg; left 0.3 peak 2.2 -> 24.8deg) -- the march's
// own motion becomes the disturbance on axes that don't need a step.
// NOTE (v5.4 REVERTED): a backward-DOMINANCE condition ((-ex) >= |ey|)
// was tried to silence marches on lateral pushes -- it VETOED genuine
// backward latches instead (|ey| = zc*ty tilt-amplified sway routinely
// exceeds the barely-crossing -ex ~ 0.07; back 0.3 dropped 3/3 -> 1/3
// with legs ~16deg = march never fired) and did not change the lateral
// peaks it targeted. Backward-threshold-only is the validated form.
```

## TEST #16 spawn-range change (TransitionLocked copy)

*Migrated 2026-07-18 from `stabilize.cc ~3321 (TransitionLocked target respawn)`:*

```
// TEST #16 (2026-05-18): target x range 1.4-1.6 → 1.2-1.4. Closer to
// robot so static reach is within natural-posture range; combined with
// kp_ankle 20→40 should let stand-and-lean be stable.
```

## TEST #16 spawn-range change (ResetLocked copy)

*Migrated 2026-07-18 from `stabilize.cc ~3937 (ResetLocked target spawn)`:*

```
// TEST #16 (2026-05-18): target x range 1.4-1.6 → 1.2-1.4 (matches
// the same fix in TransitionLocked above). Missing this caused the
// FIRST target on every reset to spawn far at [1.4, 1.6] — robot
// couldn't reach without losing balance, and user saw the tipping.
```

## Catch-march v5 vs v1-v4 scripted single-step coin flip

*Migrated 2026-07-18 from `stabilize.cc ~4051 (ModifyControl amplitude)`:*

```
// amplitude: TROT = continuous forced-march arm-ramp (matches the residual
// `arm`, kArmSec 2.0). STUMBLE quiet stand (v5, 2026-07-03) = a CATCH-MARCH
// episode: when TransitionLocked latched danger > catch_full, run THIS SAME
// clock-driven march (the twin-validated 9/9 in-place trot -- clock, cost,
// and freeze coherent by construction) for catch_step_sec with an ease-in/
// out envelope, then hand back to the quiet stand. v1-v4 tried a scripted
// SINGLE catch-step instead and hit a ~1-in-6 coin flip: breaking symmetric
// double support, unloading, swinging far, and stabilising the stance leg
// in 0.3 s FROM ZERO RHYTHM is the hard problem; the march always has
// rhythm, a push only modulates its placement. "Push -> stumble (march) ->
// settle -> keep standing."
```

## Planner numeric tuning history: 36-trajectory twin tuning, stand-override rejection (2026-07-10), and the 36->17 deploy-real re-budget

*Migrated 2026-07-18 from `stabilize.cc ~4312 (PlannerNumericOverrides)`:*

```
// "Stumble" is the only stepping strategy: its gait clock oscillates the legs,
// which the stand-tuned sampling_spline_points=3 cannot represent (the swing
// foot never lifts). spline=5 is the twin-validated sweet spot -- spline=8
// over-actuates and destabilises even a plain stand (2026-06-18). exploration
// 0.05 == the XML default, pinned here so stumble is unaffected if that
// default ever changes. Keyed by NAME so both the Lean_H12 and Lean_H12_Hands
// stumble slots match. Adding a future strategy that needs a different planner
// bandwidth is a one-line edit HERE (fork side) -- the deploy node stays
// strategy-agnostic and never changes.
// TROT (slot 23, capture-point footstep controller, stabilize::ModifyControl): like
// stumble needs spline 5 for the swing, but ALSO needs MORE rollouts to balance
// single-support around the forced swing. Twin hold-rate: 16 trajectories = 1/5
// (sampler runs out of balancing budget); 36 = 5/5 at the full ~5cm lift. The
// marginality was a SAMPLING-BUDGET limit, not a controller flaw. NOTE: 36 ~=
// 2.25x compute -> verify real-time on the deploy CPU (or use a GPU/more cores);
// on the lockstep twin it's free. Keyed by NAME (Lean_H12 + Hands trot slots).
// STAND (slot 6) planner-resolution override: TESTED AND REJECTED 2026-07-10.
// Theory (research doc h12/stabilize_stand_robustness_research_2026-07-09.md):
// the task-default 3 knots over the 1.0s horizon = 0.495s ZERO-ORDER-HOLD
// blocks (CEM is hard-wired kZeroSpline, cross_entropy/planner.h:148) = 1.65x
// the pendulum time constant -> can't represent small fast corrections ->
// overcorrection sway. Twin A/B (3x30s, plan-hz 80, flat-foot keyframes):
//   spline 3 / elite 6 (default): fwd pk-pk 2.4deg, shuffle 0.04m  <- BEST
//   spline 5 / elite 3:           fwd pk-pk 3.2deg, shuffle 0.15m
//   spline 5 / elite 6:           fwd pk-pk 3.8deg
// At 10 rollouts the extra knot dimensions cost more sampling variance than
// the resolution buys -- the same lesson as the 2026-06-18 "spline 8
// destabilises even a plain stand" note above. NO override for the stand.
// The knot-coarseness theory may still hold on REAL (23-60Hz replan makes a
// committed ZOH block live 2-4x longer than on the 80Hz twin) but that must
// be tested on real WITH a rollout bump (e.g. spline 5 + trajectories 36 like
// the trot below), not shipped twin-regressed.
// ===== STEPPING STRATEGIES (walk 22, trot 23, drive 24): the DEPLOY-REAL budget =====
// spline 5: the stand-tuned 3 knots cannot represent the leg oscillation.
//
// trajectories 17 (WAS 36 -- and 36 was WRONG on the real robot). 36 came from
// the LOCKSTEP twin (16 traj = 1/5 hold, 36 = 5/5), but the lockstep twin is
// STRUCTURALLY BLIND to plan rate: it waits for physics, so rollouts are free.
// On hardware they are not. CEM schedules trajectories + 1 jobs (the nominal
// rollout rides along) and blocks on ALL of them, so one plan iteration costs
// ceil((N+1)/threads) thread-WAVES. 36 traj on a 12-thread pool = 4 waves =
// 27-30 plans/s MEASURED on the robot -- far below the 50-100 Hz band every
// real legged sampling-MPC deployment needs (CMU's Go1 whole-body MPPI runs 30
// samples at 100 Hz on CPU and says outright that limited compute is much
// better spent reaching a ~100 Hz policy than on more samples; Unitree's own
// H1-2 RL deploys decide at 50 Hz). Re-measured at 18 traj / 18 threads: 45-52
// plans/s = in band. 17 and not 18 because the nominal is scheduled alongside:
// 17 + 1 = 18 jobs = exactly ONE wave on the auto-sized 18-thread pool, where
// 18 + 1 = 19 spills a second wave the whole pool then waits on for a single
// straggler. The deploy node prints a PLAN-RATE WARNING when (traj+1) > threads.
//
// Rate beats samples here because the swing is FORCED open-loop by
// stabilize::ModifyControl (rate-independent) -- the sampler only has to
// BALANCE, and the stance-leg weight shift is exactly what starves first.
```

## R2 sphere-sole rationale (spheres later reverted; FOOT_SPHERES=False today)

*Migrated 2026-07-18 from `_gen_stabilize_model.py ~24 (FOOT_SPHERES)`:*

```
# R2 (2026-07-04): sole = 4 corner SPHERES per foot (G1 Menagerie pattern,
# H1-2 dimensions), mesh sole demoted to visual. The mesh foot gives a
# NON-DETERMINISTIC support polygon (L vs R resolve different contact sets =
# the one-knee-crouch signature; G1 abandoned mesh feet for exactly this).
# Geometry-ONLY change: spheres inherit the validated condim 4 + friction
# 1/0.06 (G1's 0.6/condim3 are separate parity knobs, NOT taken). Corner
# placement measured from the mesh sole AABB (body frame: x -0.085..0.173,
# y +-0.042, sole plane z=-0.045), corners ~2 cm inside the edges; sphere
# bottoms sit exactly on the mesh sole plane (z center -0.040, r 0.005) so
# the standing height is unchanged. Set False to revert to the mesh sole.
```

## Squat ascent target-pose ramp (3.0 s) rationale

*Migrated 2026-07-18 from `mjpc/tasks/humanoid_bench/lean/lean.h — ResidualFn::kAscentTargetRampSeconds (deleted constant; described an asymmetric stand-up target ramp that never existed in this file — lean.cc's ramp uses target_ramp_sec/kPhaseRampSeconds)`:*

```
// STAND-UP-only target-pose ramp duration (see the asymmetric target ramp
// in lean.cc). Deliberately LONGER than kPhaseRampSeconds: at the live
// ~60/s plan rate, straightening the legs from a crouch over only 1.5s
// still launches the body backward on the second cycle (squat fell ~28s).
// Spreading the leg extension over 3s lets the sampler keep the capture
// point under the feet the whole way up. Only used when a phase transition
// moves the target pose CLOSER to home (standing up); descents still snap.
static constexpr mjtNum kAscentTargetRampSeconds = 3.0;
```

## Squat descent target-pose ramp (0.6 s) rationale

*Migrated 2026-07-18 from `mjpc/tasks/humanoid_bench/lean/lean.h — ResidualFn::kDescentTargetRampSeconds (deleted constant; same never-implemented asymmetric ramp)`:*

```
// Crouch-DOWN target-pose ramp duration. Much SHORTER than the ascent ramp:
// a pure snap folds the legs so fast the upper body overshoots into a
// forward pitch (the recurring squat descent fall), but the slow ascent
// ramp on a descent lets the robot catch the target and kills the
// stabilising spring (forward pitch at ~2.6s). 0.6s threads the needle: the
// target still leads the robot (spring preserved) yet the fold is spread
// over ~0.6s instead of instantaneous, capping the overshoot. So short that
// single-phase pose strategies (which settle for seconds) are unaffected.
static constexpr mjtNum kDescentTargetRampSeconds = 0.6;
```

## kPhaseRampSeconds 1.5->3.0 A/B rejection (arm-swing slowdown backfired)

*Migrated 2026-07-18 from `mjpc/tasks/humanoid_bench/lean/lean.h — ResidualFn::kPhaseRampSeconds comment (trimmed; constant and current rule kept in code with HISTORY pointer)`:*

```
// Tried bumping to 3.0 to slow the arm swing during 2→3 — backfired:
// with MPC horizon 1.0s the lean_forward gradient stayed weak for ~2s
// while Height (head wants to stay high, weight 35 effective in
// arm_contact_or_lean) was full strength — body settled into a slight
// backward bend as the cheap local optimum. If arm swing is still too
// fast at 1.5s, a surgical fix (Brace Hand Velocity residual active
// only during arm_plant) is preferable to slowing every cost ramp.
```

## 2026-07-12 bugfix: rollout snapshots missing the governed cmd state (free-twin ~6 s walk ceiling)

*Migrated 2026-07-18 from `mjpc/tasks/humanoid_bench/lean/lean.h — lean::ResidualLocked, cmd_active_/cmd_vdes_world_ snapshot propagation (condensed to the surviving MUST-carry invariant + pointer)`:*

```
// ★ BUGFIX 2026-07-12: propagate the GOVERNED COMMAND too. Without this every
// rollout residual sees cmd_active_=false and therefore takes the legacy
// trot_des_vel numeric path (v_des = 0) -- i.e. it costs the gait as an
// IN-PLACE trot -- while lean::ModifyControl (which reads the CANONICAL
// residual_) drives the swing FORWARD at the governed v_des. Cost and swing
// then disagree on every sampled trajectory and the sampler spends the whole
// plan cancelling the walk drive. That is the exact opposite of what the
// "MUST match lean::ModifyControl" comments in Residual() require, and it is
// the prime suspect for the free-twin ~6 s walk ceiling.
```

## 2026-07-12 faithful-twin actuator-authority A/B: parity is not the cure

*Migrated 2026-07-18 from `mjpc/tasks/humanoid_bench/lean/lean.cc — lean::PlannerNumericOverrides, deploy_frc_parity block (rewritten to post-clamp-removal current truth; NOTE the 'clamped (= the REAL deploy)' row describes the pre-2026-07-16 deploy — the H2 emit clamp has since been removed)`:*

```
// *** deploy_frc_parity is deliberately NOT set here -- it stays 0 (OFF). ***
// The ACTUATOR-AUTHORITY finding is real, but the parity fix is NOT the cure, and the
// twin said so before the robot did (authority_ab.py, faithful twin, n=5, 2026-07-12):
//     arm                          HELD    ankleP pinned at its budget
//     phantom (the twin as it was)  5/5     0%   <- the twin plant has INFINITE actuator torque
//     clamped (= the REAL deploy)   3-4/5  42%   <- the H2 clamp ALONE costs 1-2/5
//     parity  (planner told truth)  2/5    38%   <- NO BETTER; the knee went to its STOP
// Telling the sampler its real budget did not make it discover a hip/step solution -- it
// just strutted harder. Releasing the Hip Roll pin (30->0) and softening the capture
// over-step (2.2->1.4) did nothing either (3/5 each; `clamped` alone scored 3/5 then 4/5
// on IDENTICAL config, so +/-1/5 is CEM noise, not signal). The keyframe is innocent too:
// its static single-support ankle load is 19.2 Nm against a 48.6 Nm budget.
// What IS solid: the stance ankle sits pinned at EXACTLY its 48.6 Nm budget for 34-73% of
// upright ticks while the planner asks for 44-99 deg of ankle travel when only 35 deg is
// buyable. The trot's ankle DEMAND is structurally ~1.5-3x the H1-2 safety envelope. That
// is the wall; no cost lever tested moves it.
// The flag survives as a REAL-ROBOT A/B lever (--frc_parity=1). Do NOT promote it to a
// default on the strength of the mechanism alone -- hardware has to say it helps.
```

## 2026-07-11 real-trot phantom-authority observation (original deploy_frc_parity comment)

*Migrated 2026-07-18 from `mjpc/tasks/humanoid_bench/lean/Lean_H12_Magpie.xml — deploy_frc_parity <numeric> comment (rewritten; original described the since-removed H2 emit clamp and falsely claimed PlannerNumericOverrides enables parity for trot/drive)`:*

```
<!-- deploy_frc_parity (2026-07-11): ACTUATOR-AUTHORITY PARITY switch, read by the
     DEPLOY node (deploy_common PatchActuators), NOT by the planner itself. 1 =>
     tighten every planner actuator forcerange to the torque the node can actually
     emit (0.9 x tau_estop = the H2 command clamp). The stock model hands the planner
     ankle +/-75 Nm and torso +/-200 Nm while the node emits at most 48.6 / 36.0 Nm,
     so the sampler plans single-support balance on 1.5-5.6x PHANTOM authority; on the
     REAL trot (2026-07-11, plan rate healthy 45-52/s) the stance ankle railed at
     exactly 48.6 and the torso at exactly 36.0 for 6 s while the stance knee locked
     to a passive prop against a weight-200 anti-strut cost = the weak-ankle crutch.
     DEFAULT 0 here (= every strategy byte-identical); Task::PlannerNumericOverrides
     turns it ON for the STEPPING strategies only (trot 23 / drive 24), so nothing
     real-validated (stand/crouch/reach/lean/stumble) changes. CLI kill: --frc_parity=0. -->
```
