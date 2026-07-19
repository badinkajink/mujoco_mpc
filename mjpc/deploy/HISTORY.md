# Deploy-layer tuning & decision history

Dated tuning history, A/B verdicts, and post-mortems migrated out of the
deploy-layer code comments (2026-07-18 comment policy: code carries current-truth
invariants only; chronology lives here). Each entry is the verbatim comment text
at migration time; the git history of the source files has the full context.


## 2026-07-02 flag diet + 2026-07-03 plan-flag re-add chronology

*Migrated 2026-07-18 from `mjpc/deploy/deploy_common.h (top-of-file header)`:*

```
// FLAG DIET (2026-07-02, HAMS_integration): the ~20 flags that were passed
// IDENTICALLY in every documented invocation (Command_Sheet_h12.html A1/A2/B1/
// B2/B-Stabilize, run_realchain.sh, tab launchers) are now compiled-in
// constants below; the flags that provably differ between plants/missions
// survive in the thin mains (task, strategy, gravity_ff, twin_dt,
// sportstate_topic, IMU/ankle calibration, network_interface, domain_id,
// grpc_port, arm_aware). Deleted dead paths: --sync_plan / --plan_rate_hz (all
// runs used the async planner thread), --arm_ramp_sec (only live when
// start_ramp_sec==0, which never happened), --require_sportstate=false debug
// mode (the base estimator / OptiTrack always publishes sportmodestate now),
// --execute_best (never used).
// NO TUNED VALUE CHANGED: only how it is supplied (compiled default vs CLI).
// RE-ADDED 2026-07-03: --plan_trajectories + --plan_threads. Cutting them was
// a diet mistake -- they are the R2 plan-rate sweep levers on REAL hardware
// (samples-per-plan vs replan-rate: trot's 36 traj @ 12 threads plans only
// ~28 Hz = the 06-29 starvation diagnosis; one-wave rule: traj <= threads).
// Both default 0 = the compiled/task value, so a bare invocation is unchanged.
```

## Planner thread-pool starvation diagnosis: hard 12 -> AUTO sizing

*Migrated 2026-07-18 from `mjpc/deploy/deploy_common.h (kPlanThreads)`:*

```
// WAS a hard 12 ("leaves cores for twin/safety"), which silently starved every STEPPING
// strategy on the real robot: CEM schedules num_trajectory + 1 jobs (the nominal rollout
// rides along, cross_entropy/planner.cc:469) and WAITS for all of them, so a plan iteration
// costs ceil((N+1)/threads) thread-WAVES. Trot's 36 trajectories on 12 threads = 4 waves =
// 27-30 plans/s measured on real -- below the 50-100 Hz band every real deployment needs
// (CMU Go1 MPPI: 30 samples @ 100 Hz, CPU-only, "limited computing is much better spent on
// achieving a ~100 Hz policy than additional sample evaluations"; Unitree's own H1-2 RL
// deploys decide at 50 Hz). The open-loop swing (lean::ModifyControl) is rate-INDEPENDENT
// so the legs still alternated -- but the stance-leg weight shift is SAMPLER-owned, so it
// starved: feet never unloaded. AUTO-sizing gives 18 on the dev laptop (24 hw threads),
// which measured 45-52 plans/s on the real robot -- in band.
```

## Actuator-authority parity diagnosis: 2026-07-11 real-trot ankle/torso rail under the then-active H2 clamp

*Migrated 2026-07-18 from `mjpc/deploy/deploy_common.h (NodeConfig::frc_parity)`:*

```
  // ---- ACTUATOR-AUTHORITY PARITY (2026-07-11): the planner must not plan with torque
  // the node will never emit. The H2 clamp bounds the EMITTED command to
  // kClampRatio * tau_estop, but the PLANNER model's forceranges were left at the MJCF
  // defaults for legs/torso -- so the sampler plans single-support balance believing it
  // has authority it does not have:
  //     joint    planner sees   node actually emits   over-estimate
  //     ankleP     +/-75 Nm          48.6 Nm             1.54x
  //     ankleR     +/-75 Nm          32.4 Nm             2.31x
  //     torso     +/-200 Nm          36.0 Nm             5.56x
  //     hipY      +/-200 Nm          54.0 Nm             3.70x
  // REAL 2026-07-11 (strat 23, plan rate healthy at 45-52/s): the stance ankle pitch railed
  // at EXACTLY 48.6 Nm and the torso at EXACTLY 36.0 Nm for 6 s while the stance knee locked
  // to -0.05 rad (a passive prop) AGAINST a weight-200 anti-strut cost -- the classic
  // weak-ankle crutch. The plan was valid in the planner's model and unexecutable in the
  // node's. Same bug CLASS as the phantom-table parity bug that faked the "forward walk needs
  // RL" verdict: the planner was solving the wrong physics.
```

## Anti-stiction headroom/backstop reasoning under the removed H2 emit clamp

*Migrated 2026-07-18 from `mjpc/deploy/deploy_common.h (align_ki anti-stiction comment)`:*

```
  // loose. The head-room is real -- the H2 clamp permits |tgt - q| up to
  // (0.9*tau_estop - |tau_ff| - kv*|dq|)/kp = 0.36..1.35 rad on these joints -- and that SAME
  // clamp is the backstop: the emitted torque still cannot exceed 0.9 x the safety estop, so
  // this pushes hard but never leaves the safety envelope. 0 = off (pure PD, may stall short).
```

## 2026-07-02 dedupe merge of the two original mains + the four rclcpp-audit fixes port (H1/H2/M4/M5)

*Migrated 2026-07-18 from `mjpc/deploy/deploy_common.cc (file header)`:*

```
// Shared core of the H1-2 MJPC deploy nodes -- see deploy_common.h for the
// architecture note and the flag-diet rationale. This file is the line-for-line
// merge of the formerly-duplicated h12_control_node.cc / h12_lower_body_controller.cc
// (HAMS_integration 2026-07-02), parameterized by NodeConfig:
//   - kNU / gain tables       -> cfg.nu + cfg.kp/kv/tau_estop/tau_limit/frc_limit
//   - arm-aware machinery     -> gated on cfg.upper_count > 0 (legs-only node)
//   - status-line variant     -> cfg.telemetry
// plus the four defensive fixes ported from the HAMS rclcpp rewrite (audit
// H1/H2/M4/M5, see mjpc_deploy_lowerbody_controller.cpp):
//   H1  input-freshness watchdog: state older than kStaleSec -> damping safe-hold
//   H2  torque clamp on the FULL budget tau_ff + KP*e + KV*dq (was KP*e only)
//   M4  tau%%estop telemetry graded against TAU_ESTOP (was mislabeled TAU_LIMIT)
//   M5  mju_error -> emit safe-hold BEFORE terminating (was a bare exit)
```

## Shift-after-clamp ordering post-mortem (kp*off hole in the estop guarantee)

*Migrated 2026-07-18 from `mjpc/deploy/deploy_common.cc (ankle zero-offset ordering comment, ~line 1745)`:*

```
    // fill_state. Applied BEFORE the torque clamp so the clamp bounds the delta the motor PD
    // actually sees (encoder frame) -- the old shift-after-clamp order left a kp*off hole in
    // the estop guarantee.
```

## start-pose provenance

*Migrated 2026-07-18 from `mjpc/deploy/h12_lower_body_controller.cc (kLowerStartPose comment block, pre-edit lines 32-54)`:*

```
// START POSE (--align_start): the stance the node drags the legs into BEFORE
// handing the robot to MJPC. R^12, radians, in DDS motor order (rows 0..11 of
// lowcmd -- the same rows this node publishes to rt/safety/lowcmd_lower_in).
//
// PROVENANCE (2026-07-13). Derived from the only long good stand on record,
// logs/stand_cost_3_20260711_175521: averaging the MEASURED pose over its 164 s
// of stable standing (70 s -> 234 s after hand-off) gives
//     { 0.011,-0.276, 0.173, 0.370,-0.154,-0.133,
//      -0.006,-0.269,-0.108, 0.371,-0.233, 0.221 }
// i.e. feet 0.549 m apart, knees bent 21.2/21.3 deg, pelvis 1.017 m. Every joint
// of that lands within 5.8 deg of the model's `stand` keyframe, so the keyframe
// was already right; the run just confirms it on hardware.
//
// SHIPPED below == the 'stand_up'/'stand' keyframe legs EXACTLY (2026-07-14, user
// request): the align target now equals what strategy-6 Posture pulls toward, so
// there is NO pose discontinuity at handover. (Previously knee 0.37 / hip_pitch
// -0.27 / ankle_pitch -0.21 were taken from a measured good-stand HOLD -- droop-
// aware -- which differed from the keyframe by up to 7 deg and caused a small jump
// at handover.) Under load the robot droops FROM this commanded pose; the align
// exits on SETTLED (not q==target), so it rests slightly off and reports a residual
// -- expected. Edit freely; this is the knob.
//
//                          feet ~0.516 m apart, knees ~20 deg bent (== keyframe)
```

## start-pose provenance

*Migrated 2026-07-18 from `mjpc/deploy/h12_lower_body_controller.cc (kLockstandStartPose comment block, pre-edit lines 70-75)`:*

```
// LOCKSTAND (strategy 26) align target = the 'lockstand' keyframe legs: LOCKED knee
// + WIDE stance. Used ONLY when --strategy 26, so the bring-up places the feet apart
// and the knees straight BEFORE handover (a balance hold cannot widen PLANTED feet --
// they must start wide). Matches the own-sim-validated pose (held 3/3), so the robot
// lands at lockstand's target instead of straightening under load after handover.
//                          feet ~0.635 m apart, knees ~4.6 deg (locked strut)
```

## start-pose provenance

*Migrated 2026-07-18 from `mjpc/deploy/h12_split_controller.cc (kLowerStartPose comment block, pre-edit lines 38-60; text identical to the lower controller's block)`:*

```
// START POSE (--align_start): the stance the node drags the legs into BEFORE
// handing the robot to MJPC. R^12, radians, in DDS motor order (rows 0..11 of
// lowcmd -- the same rows this node publishes to rt/safety/lowcmd_lower_in).
//
// PROVENANCE (2026-07-13). Derived from the only long good stand on record,
// logs/stand_cost_3_20260711_175521: averaging the MEASURED pose over its 164 s
// of stable standing (70 s -> 234 s after hand-off) gives
//     { 0.011,-0.276, 0.173, 0.370,-0.154,-0.133,
//      -0.006,-0.269,-0.108, 0.371,-0.233, 0.221 }
// i.e. feet 0.549 m apart, knees bent 21.2/21.3 deg, pelvis 1.017 m. Every joint
// of that lands within 5.8 deg of the model's `stand` keyframe, so the keyframe
// was already right; the run just confirms it on hardware.
//
// SHIPPED below == the 'stand_up'/'stand' keyframe legs EXACTLY (2026-07-14, user
// request): the align target now equals what strategy-6 Posture pulls toward, so
// there is NO pose discontinuity at handover. (Previously knee 0.37 / hip_pitch
// -0.27 / ankle_pitch -0.21 were taken from a measured good-stand HOLD -- droop-
// aware -- which differed from the keyframe by up to 7 deg and caused a small jump
// at handover.) Under load the robot droops FROM this commanded pose; the align
// exits on SETTLED (not q==target), so it rests slightly off and reports a residual
// -- expected. Edit freely; this is the knob.
//
//                          feet ~0.516 m apart, knees ~20 deg bent (== keyframe)
```

## start-pose provenance

*Migrated 2026-07-18 from `mjpc/deploy/h12_split_controller.cc (kLockstandStartPose comment block, pre-edit lines 76-81; text identical to the lower controller's block)`:*

```
// LOCKSTAND (strategy 26) align target = the 'lockstand' keyframe legs: LOCKED knee
// + WIDE stance. Used ONLY when --strategy 26, so the bring-up places the feet apart
// and the knees straight BEFORE handover (a balance hold cannot widen PLANTED feet --
// they must start wide). Matches the own-sim-validated pose (held 3/3), so the robot
// lands at lockstand's target instead of straightening under load after handover.
//                          feet ~0.635 m apart, knees ~4.6 deg (locked strut)
```

## frc-parity clamp-era telemetry

*Migrated 2026-07-18 from `mjpc/deploy/h12_lower_body_controller.cc --frc_parity help (dated clamp-era fragment, also present verbatim in the pre-edit h12_split_controller.cc copy of the flag)`:*

```
The stabilize planner model ships the ankle at +/-75 Nm while the node emits at most 48.6 -- a 1.54x overestimate on the joint that OWNS fore-aft balance, so the sampler buys sway correction with ankle torque that the clamp then eats (real stand: LankP railed at 48.2/48.6, clamp 6.9%).
```

## estimator tick_dt / --sim-dt plant timestep parity

*Migrated 2026-07-18 from `core_ws/src/h1_bringup/config/mjpc_sim.yaml (h12_deploy_mjpc_estimator.tick_dt)`:*

```
tick_dt: 0.005   # follows --sim-dt 0.005 (plant timestep parity, 2026-07-04)
```

## planner-rate parity / superseded 3-thread band-aid (2026-07-04)

*Migrated 2026-07-18 from `core_ws/src/h1_bringup/config/mjpc_sim.yaml (mjpc_deploy_splitbody_controller.plan_threads)`:*

```
    # PLANNER-RATE PARITY (2026-07-04, wobble root cause R1): the CEM planner
    # free-runs on wall time (~55 iters/s at 12 threads); at RoboCasa's ~0.25x
    # RTF that is ~4x more iterations per PLANT-second than twin/real -- the
    # elite-variance sampler collapses to its 0.01 rad floor on the near-static
    # state and can't see recovery candidates until the tilt re-spreads the
    # elites = the measured 5-7 plant-s wobble limit cycle. 3 threads ~= 14
    # iters/s wall ~= the twin's iterations-per-plant-second. RoboCasa BENCH
    # yamls only; twin/real chains keep --plan_threads 12.
```

## gravity_ff generation 1: band-drop stand A/B 2026-07-03 (kept 0.85)

*Migrated 2026-07-18 from `core_ws/src/h1_bringup/config/mjpc_sim.yaml (mjpc_deploy_splitbody_controller.gravity_ff)`:*

```
    # BENCHED 2026-07-03 (A/B on the band-drop stand test): 0 collapses into a
    # knee-pegged crouch (z 0.64->0.51, legs under-driven -- unlike our twin,
    # this plant has NO leg gravcomp; only the two hands); 0.85 holds height
    # (z 0.91-0.99) all the way to the drop. Keep 0.85 here.
```

## gravity_ff surviving generation: ballast + rigid feet take 12 (moved to 1.0)

*Migrated 2026-07-18 from `core_ws/src/h1_bringup/config/mjpc_sim.yaml (mjpc_deploy_splitbody_controller.gravity_ff)`:*

```
    # gravity_ff A/B, WITH the 0.67 kg back_equipment ballast added AND rigid feet
    # (--rigid-feet, default ON in h12_mujoco.py): rigid feet ELIMINATED the topple
    # (foot-slip was the dominant driver). Residual then = a vertical SINK; take 12
    # showed gff 1.0 holds height (z~0.92 for 40s) vs gff 0.85 sinking to z 0.64.
    # So 1.0 here (full grav-comp) with the feet planted + mass matched. Remaining
    # 5-12deg tilt wobble is NOT a stand yet -- next parity lever is the global
    # integrator (RoboCasa Euler vs planner/twin implicitfast; the task XML warns
    # Euler diverges on biped balance). REAL keeps 0.85 (mjpc_real.yaml).
```

## gravity_ff generation: ballast takes 9-10, contact/integrator root cause (kept 0.85)

*Migrated 2026-07-18 from `core_ws/src/h1_bringup/config/mjpc_sim.yaml (mjpc_deploy_splitbody_controller.gravity_ff)`:*

```
    # gravity_ff A/B, WITH the 0.67 kg back_equipment ballast added (takes 9-10,
    # 2026-07-03): neither 0.85 (fwd+lat fold) nor 0 (stronger LATERAL fold +
    # knee collapse) stands. The ballast is a correct parity fix but NOT the
    # dominant gap -- the fold is a lateral foot-slip, root-caused (opt dump of
    # the assembled robosuite model) to the CONTACT/INTEGRATOR mismatch:
    # RoboCasa runs impratio=20 / integrator=Euler / cone=elliptic vs the twin+
    # planner's 100 / implicitfast / pyramidal. gravity_ff is a minor amplifier;
    # keep 0.85 (matches real) until the contact model is addressed.
```

## gravity_ff generation: harness takes 2026-07-03, twin free-stand gap reproduced (kept 0.85)

*Migrated 2026-07-18 from `core_ws/src/h1_bringup/config/mjpc_sim.yaml (mjpc_deploy_splitbody_controller.gravity_ff)`:*

```
    # gravity_ff A/B (harness takes, 2026-07-03): 0 = knee-pegged collapse;
    # 0.85 = slow policy-commanded knee fold once the blend starts (z 1.03->0.87,
    # low knee torque -- the fold is planned, not saturated); 1.0 = height holds
    # better (z~0.97) but the tilt fight is WORSE (posture assist 80-100 Nm).
    # Neither frees the stand: the divergence starts at blend onset (twin ~5.2s)
    # in every take = the documented post-parity twin structural free-stand gap,
    # faithfully reproduced on this plant. Keep the validated 0.85.
```

## twin_dt 0.005 selection: T2-vs-T3b dt-sensitivity bench (2026-07-04)

*Migrated 2026-07-18 from `core_ws/src/h1_bringup/config/mjpc_sim.yaml (mjpc_deploy_splitbody_controller.twin_dt)`:*

```
    # 0.005 (2026-07-04, T2-vs-T3b): the balance bench now runs the plant at
    # the TWIN's dt via --sim-dt 0.005 -- the deploy-chain stand is
    # dt-sensitive (twin@0.002 hangs, twin@0.005 stands even at RTF 0.5).
```

## sportstate twin-parity A/B: ground-truth sportstate did not stop the fall (2026-07-06)

*Migrated 2026-07-18 from `core_ws/src/h1_bringup/config/mjpc_sim.yaml (mjpc_deploy_splitbody_controller.sportstate_topic)`:*

```
    # ★ TWIN-PARITY A/B RESULT (2026-07-06): tested rt/sportmodestate (GROUND TRUTH,
    # plant --truth-sportstate, VERIFIED 191 Hz + controller reading it) -> did NOT
    # stop the fall. The LATERAL fold became a SAGITTAL forward->backward divergence
    # (plane-floor had already fixed lateral). So the ESTIMATOR was NOT the killer.
    # REVERTED to rt/sportmodestate_est = real-parity + the 30 ms LPF that damps the
    # raw-truth base velocity the planner over-reacts to. Re-flip to rt/sportmodestate
    # (+ plant --truth-sportstate) only to A/B raw truth again.
```

## sportstate real-parity source rationale: EKF vs raw truth AngMom-5 A/B (2026-07-05)

*Migrated 2026-07-18 from `core_ws/src/h1_bringup/config/mjpc_sim.yaml (mjpc_deploy_splitbody_controller.sportstate_topic)`:*

```
    # REAL-PARITY STATE SOURCE (2026-07-05): read the RW-EKF estimate on
    # rt/sportmodestate_est — the SAME signal path the real robot uses (the
    # bringup already launches the estimator). Twin A/B at AngMom 5 showed the
    # EKF-filtered velocity gives a QUIETER stand (tilt 3-5° steady vs 5-6°
    # then fall) than raw truth: the earlier --truth-sportstate override was a
    # self-inflicted parity gap (truth = full-bandwidth velocity the planner
    # over-reacts to; the 30 ms LPF matches real). The EKF's leg-odometry xy
    # wanders ~0.3 m under sim foot micro-slip, but for STANDING xy position is
    # irrelevant (balance = orientation + base velocity + CoM). Set this back to
    # rt/sportmodestate (+ sim --truth-sportstate) only to A/B raw truth again.
```

## elastic-band release ownership: removed sim slack-triggered auto-release doctrine

*Migrated 2026-07-18 from `core_ws/src/h1_bringup/config/mjpc_sim.yaml (mjpc_deploy_splitbody_controller.drop_band)`:*

```
    # Band release is owned by the SIM's slack-triggered auto-release
    # (h12_mujoco --band-auto-release: |band force| <= 30 N sustained 2 sim-s,
    # the twin harness "handoff-timing" doctrine -- a self-supporting robot
    # leaves the band slack). Keep the launcher's TIMER release OFF: the
    # /elastic_band/toggle service is a stateful TOGGLE, so firing it after
    # the sim already auto-released would turn the band back ON. (Timer
    # releases were also benched dishonest: they fire mid-sway and dump the
    # robot from whatever posture the tether wound it into.)
```
