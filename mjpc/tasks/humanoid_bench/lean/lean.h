#ifndef MJPC_TASKS_HUMANOID_BENCH_LEAN_LEAN_H_
#define MJPC_TASKS_HUMANOID_BENCH_LEAN_LEAN_H_

#include <map>
#include <atomic>
#include <memory>
#include <random>
#include <limits>
#include <string>
#include <vector>

#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include "mjpc/tasks/humanoid/interact/contact_keyframe.h"
#include "mjpc/tasks/humanoid/interact/motion_strategy.h"
#include "mujoco/mujoco.h"

namespace mjpc {

// ★ 2026-08-24 STRAT 27 GRASP GATE — process-global handshake between the lean
// task's Transition (which decides WHEN to close, from believed tip-to-target
// distance + dwell on the straddle rung) and the deploy layer (which carries
// the command to the magpie gripper over DDS rt/grasp_gate and returns the
// relay's verdict via rt/grasp_ack). Globals, not task members, because the
// deploy code only sees the abstract Task. Binaries without the deploy plumb
// (ownsim, twin bench) leave the ack at 0 and the gate falls through on
// grasp_ack_timeout => close is assumed => byte-identical ladder shape.
//   g_grasp_gate_cmd: 0 = idle, 1 = CLOSE requested (task sets; deploy clears
//                     on publish is NOT done -- deploy just mirrors it out).
//   g_grasp_ack:      0 = none, 1 = closed-on-object (advance), -1 = closed
//                     EMPTY (aperture check failed -> retry/recover).
inline std::atomic<int> g_grasp_gate_cmd{0};
inline std::atomic<int> g_grasp_ack{0};

// ★ 2026-08-24 STRAT 27 OBJECT SERVO BUS. The gripper-cam tag bridge publishes
// the object's pose IN THE CAMERA OPTICAL FRAME on DDS rt/object_tag; the deploy
// layer (--object_servo) drops the raw translation here and bumps g_object_seq.
// The TASK does the geometry (compose with believed wrist FK + the hand-eye
// extrinsic, convert to a world offset, slew-limit) ONCE PER PLAN in
// TransitionLocked -- never per-rollout -- exactly like the T1 reference trim.
// Kept as separate scalars rather than a locked struct: a torn read can only mix
// components of two consecutive 30 Hz detections of a slowly-moving object
// (sub-mm), and the slew limiter downstream swallows that anyway.
// g_object_seq is the FRESHNESS signal: the task watches it change rather than
// comparing a callback wall-clock against plant time (the two-clock trap).
inline std::atomic<double> g_object_cam_x{0.0};
inline std::atomic<double> g_object_cam_y{0.0};
inline std::atomic<double> g_object_cam_z{0.0};
inline std::atomic<unsigned long long> g_object_seq{0};

constexpr int kLeanStrategyParameterIndex = 1;

// Manual-phase override. -1 (default) = auto-advance through keyframes
// based on success_sustain_time + target_distance_tolerance. 0..N-1 =
// hold at that keyframe index regardless of progress. Lets the user
// scrub through the loaded strategy's phases without reloading (which
// would reset to keyframe 0 and snap the body back through stand_up).
constexpr int kLeanPhaseParameterIndex = 2;

// LIVE reach target (vision/nav integration, 2026-07-02). These are gRPC-settable
// task parameters (SetTaskParameters, same path as Strategy/Phase) that let an
// external ROS2 bridge in core_ws drive the reach target at runtime instead of
// the hardcoded `reach_target` model numeric. "Reach Active" != 0 switches the
// task from the static numeric to the live (X,Y,Z) point. The X/Y/Z are expected
// in the MJPC PLANNER-WORLD frame: the bridge converts the perception PoseStamped
// (published in the robot `pelvis` frame by h12_skills after graspgen) into MJPC
// world using the MPC node's own reported base pose (GetState) BEFORE setting
// them, so the two systems' differing world frames (Unitree-IMU vs FAST-LIO
// camera_init) are reconciled via the common robot-body frame. Appended AFTER
// Phase so the existing positional indices (Height Goal 0 / Strategy 1 / Phase 2)
// are unchanged; guarded everywhere by a parameters.size() check so models that
// don't declare these fall back to the legacy numeric path.
constexpr int kLeanReachActiveParameterIndex = 3;
constexpr int kLeanReachXParameterIndex = 4;
constexpr int kLeanReachYParameterIndex = 5;
constexpr int kLeanReachZParameterIndex = 6;

// ★ 2026-09-04 TABLE HEIGHT (generalisation study). Absolute world z of the
// table's PHYSICAL TOP FACE, in metres. 0 (default) = OFF = use whatever the
// compiled model says, i.e. byte-identical to every run before this parameter
// existed. Non-zero = TransitionLocked rewrites model->body_pos[table] so the
// face lands exactly there, and shifts the object/target bodies by the same
// delta so the manipulation task stays fixed IN THE TABLE FRAME.
//
// WHY A PARAMETER AND NOT A MODEL VARIANT. Almost every table-dependent term in
// lean.cc already derives its geometry from the `table_surface_pos` framepos and
// the compiled `table_top` geom half-extents (Brace Pos, the brace force gate,
// Hip/Leg/Body-Table Clearance, the reach_target_table rungs), so the height is
// an INPUT the costs can consume, not a constant baked into them. Exposing it as
// parameter index 7 makes a sweep a --table_h flag rather than 9 XML forks, and
// lets the GUI slider show the degradation live.
//
// Appended AFTER Reach Z so indices 0-6 are untouched; guarded everywhere by a
// parameters.size() check so models that do not declare it fall through.
constexpr int kLeanTableHeightParameterIndex = 7;

constexpr char kLeanStrategyFilePath[] =
    SOURCE_DIR "/mjpc/tasks/humanoid_bench/lean/strategies/";

class lean : public Task {
 public:
  std::string Name() const override = 0;

  std::string XmlPath() const override = 0;

  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const lean *task,
                        const mjpc::humanoid::ContactKeyframe& kf =
                            mjpc::humanoid::ContactKeyframe(),
                        mjtNum keyframe_start_time = 0.0,
                        mjtNum prev_reach_scale = 0.0,
                        mjtNum prev_brace_pos_scale = 0.0,
                        mjtNum prev_posture_scale = 1.0,
                        mjtNum prev_brace_force_target = 0.0,
                        int prev_posture_key_id = 0,
                        int num_phases = 1,
                        const bool* contact_pair_is_new = nullptr)
        : mjpc::BaseResidualFn(task),
          residual_keyframe_(kf),
          keyframe_start_time_(keyframe_start_time),
          prev_phase_reach_scale_(prev_reach_scale),
          prev_phase_brace_pos_scale_(prev_brace_pos_scale),
          prev_phase_posture_scale_(prev_posture_scale),
          prev_phase_brace_force_target_(prev_brace_force_target),
          prev_posture_key_id_(prev_posture_key_id),
          num_phases_(num_phases) {
      for (int i = 0; i < 5; ++i) {
        contact_pair_is_new_[i] =
            contact_pair_is_new ? contact_pair_is_new[i] : false;
      }
    }

    void Residual(const mjModel *model, const mjData *data,
                  double *residual) const override;

    // Phase-transition ramp duration: the reach + brace cost scales smoothly
    // interpolate from their previous-phase values to the new-phase values
    // over this many seconds after each keyframe advance. 1.5s gives the
    // robot time to absorb the new gradient instead of being shoved forward.
    // Tried bumping to 3.0 to slow the arm swing into the brace — backfired:
    // with MPC horizon 1.0s the lean gradient stayed weak for ~2s while Height
    // (head wants to stay high, weight 35 effective under any_arm_contact) was
    // full strength — body settled into a slight backward bend as the cheap
    // local optimum. If arm swing is still too fast at 1.5s, a surgical fix (a
    // Brace Hand Velocity residual gated to the brace phase) is preferable to
    // slowing every cost ramp.
    static constexpr mjtNum kPhaseRampSeconds = 1.5;

    // STAND-UP-only target-pose ramp duration (see the asymmetric target ramp
    // in lean.cc). Deliberately LONGER than kPhaseRampSeconds: at the live
    // ~60/s plan rate, straightening the legs from a crouch over only 1.5s
    // still launches the body backward on the second cycle (squat fell ~28s).
    // Spreading the leg extension over 3s lets the sampler keep the capture
    // point under the feet the whole way up. Only used when a phase transition
    // moves the target pose CLOSER to home (standing up); descents still snap.
    static constexpr mjtNum kAscentTargetRampSeconds = 3.0;

    // Crouch-DOWN target-pose ramp duration. Much SHORTER than the ascent ramp:
    // a pure snap folds the legs so fast the upper body overshoots into a
    // forward pitch (the recurring squat descent fall), but the slow ascent
    // ramp on a descent lets the robot catch the target and kills the
    // stabilising spring (forward pitch at ~2.6s). 0.6s threads the needle: the
    // target still leads the robot (spring preserved) yet the fold is spread
    // over ~0.6s instead of instantaneous, capping the overshoot. So short that
    // single-phase pose strategies (which settle for seconds) are unaffected.
    static constexpr mjtNum kDescentTargetRampSeconds = 0.6;

   protected:
    mjpc::humanoid::ContactKeyframe residual_keyframe_;

    // ----- Phase-transition state -----------------------------------------
    // `keyframe_start_time_`: wall time at which the current keyframe became
    // active (set in TransitionLocked). The residual uses `data->time -
    // keyframe_start_time_` to compute how far through the ramp we are.
    // `prev_phase_*_scale_`: the scales that were in effect just before the
    // last transition. Together they let Residual() lerp smoothly into the
    // new phase's scales, which is the WBC-style smooth handoff the robot
    // needs to avoid lurching when a contact cost switches on.
    mjtNum keyframe_start_time_ = 0.0;
    // `foot_pin_x_`: the Foot Stability anchor's x, captured from the MEASURED
    // released stance in TransitionLocked (single-threaded, real state) and only
    // READ by the residual, which runs in parallel rollout threads. NaN = not yet
    // pinned => the residual keeps the hardcoded home. See lean.cc.
    mjtNum foot_pin_x_ = std::numeric_limits<mjtNum>::quiet_NaN();
    mjtNum prev_phase_reach_scale_ = 0.0;
    mjtNum prev_phase_brace_pos_scale_ = 0.0;
    // Posture scale starts at 1.0 (no boost) and ramps to 3.0 during stand_up.
    mjtNum prev_phase_posture_scale_ = 1.0;
    // ITER 28: previous phase's brace_force_target value, used to smoothstep
    // the brace force demand across phase boundaries so MPC doesn't see a
    // step change (which would plan an impulsive arm slam into the table).
    mjtNum prev_phase_brace_force_target_ = 0.0;

    // Previous phase's posture keyframe id (model <key> index), captured at
    // every transition (SnapshotEffectiveScales) so Residual() can ramp the
    // TARGET pose from it to the current keyframe over kPhaseRampSeconds —
    // parallels prev_phase_posture_scale_ but for the pose itself, not its
    // weight. 0 = home on cold start. Only matters when consecutive phases name
    // DIFFERENT keyframes (cyclic squat); pipeline phases all resolve to home.
    int prev_posture_key_id_ = 0;

    // Number of phases (keyframes) in the active strategy; set in TransitionLocked
    // from motion_strategy_.GetKeyframesCount(). The target-pose ramp in Residual()
    // is GATED on num_phases_ > 1 so single-phase strategies (stand/crouch/arms)
    // never enter the ramp branch -- byte-identical to before. This is the
    // per-strategy gate the 2026-06-08 revert note (lean.cc) said the ramp needed.
    int num_phases_ = 1;

    // Per-contact-pair "is new this phase" flags. true when a contact pair
    // went from inactive (body1=-1) in the previous keyframe to active in
    // the current one — i.e. a brand-new target that just appeared. Used
    // by ContactResidual to multiply each newly-appeared pair's residual
    // by smoothstep(t_in_phase / kPhaseRampSeconds) so the cost grows
    // from 0 to full strength over the same 1.5s window as the weights.
    // Without this, the planner sees the new contact target's gradient
    // instantly and slams the body toward it (the 2→3 hand-slam-into-
    // table failure mode). Pairs that were continuously active across
    // the transition keep factor 1.0 throughout.
    bool contact_pair_is_new_[5] = {false, false, false, false, false};


   private:
    friend class lean;

    static constexpr double kHandDistThreshold = 0.0;
    static constexpr double kContactStableTime = 0.0;
    static constexpr double kContactForceThreshold = 0.0;

    void ContactResidual(const mjModel *model, const mjData *data,
                         double *residual, int *counter) const;
  };

  lean() : residual_(this), current_strategy_(-1) {
    target_position_ = {1.5, 0.0, 0.83};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_x(1.4, 1.6);
    std::uniform_real_distribution<> dis_y(-0.3, 0.3);
    target_position_ = {dis_x(gen), dis_y(gen), 0.83};
  }

  void TransitionLocked(mjModel *model, mjData *data) override;

  void ResetLocked(const mjModel *model) override;

  // ---- headless bench accessors (lean_bench.cc) ------------------------- //
  // The phase index/name are what a "did the ladder actually advance" call is
  // made of, and they live behind motion_strategy_. Read-only, main-thread.
  int BenchPhaseIndex() const {
    return motion_strategy_.GetCurrentKeyframeIndex();
  }
  int BenchPhaseCount() const { return motion_strategy_.GetKeyframesCount(); }
  std::string BenchPhaseName() const {
    return motion_strategy_.HasKeyframes()
               ? motion_strategy_.GetCurrentKeyframe().name
               : std::string("none");
  }

  // Populate phase-aware monitoring metrics (reach, CoP, ICP, brace force,
  // saturation, etc.) for the Research GUI / headless analyzer. Reads the
  // current keyframe + sensor stack; safe to call from the gRPC poll loop.
  // See task.h ComputeMetrics for contract.
  void ComputeMetrics(const mjModel *model, const mjData *data,
                      std::map<std::string, double> *metrics,
                      std::string *phase_name) const override;

  // No PlannerNumericOverrides override: the surviving lean strategies all run
  // at the model's own planner bandwidth, so the base Task implementation
  // (returns {}) is exactly right. deploy_common.cc still calls it through the
  // base-class interface.

  // ★ 2026-08-10 phase-scheduled CEM variance floor (see Task::LiveStdMinOverride
  // and the std_min_state_gated numeric). Written in TransitionLocked
  // (single-threaded), read by CrossEntropyPlanner::Rollouts on the plan thread
  // -> atomic. -1 = no override (gate off / other strategies) = byte-identical.
  double LiveStdMinOverride() const override { return std_min_live_.load(); }
  mutable std::atomic<double> std_min_live_{-1.0};
  // ★ v4 ramp state (TransitionLocked-only, single-threaded): the v3 STEP
  // 0.01->0.05 at lean entry killed 20/20 runs at t=70-90 -- 5x noise injected
  // at the exact commit moment. The floor now smoothsteps between phase targets
  // over `std_min_ramp_sec` so the first seconds of each transition inherit the
  // previous phase's crispness.
  double std_min_target_ = -1.0;
  double std_min_ramp_from_ = -1.0;
  double std_min_ramp_t0_ = 0.0;

  // Slider layout (Lean H12). Live roster -- everything else is padding:
  //    6  h12_simple_stand          — stand_up (DEFAULT, and the deploy default)
  //   21  h12_simple_reach          — reach_to_target
  //   22  h12_simple_forearm_brace  — forearm_brace_lean
  //   33  h12_simple_grasp          — reach_to_target (grasp bench; grasp.h patches 21)
  //   34  h12_mission_brace_grasp   — 4-phase retrieval mission
  //                                   (stand_up -> forearm_brace_lean ->
  //                                    reach_to_target -> stand_up)
  // stand_up / reach_to_target / forearm_brace_lean are the ONLY phase names any
  // live strategy uses; lean.cc's phase gates recognise nothing else.
  //
  // DESIGN (2026-05-26): the leg-lift phase is DROPPED permanently. BOTH feet
  // stay stable on the ground through EVERY phase. The only lower-body motion
  // allowed is WBC-driven foot re-placement / hip twist IN SERVICE OF the brace
  // (to hold balance while reaching/leaning) — never lifting a leg off the floor.
  //
  // STRATEGY IS A POSITIONAL INDEX, NOT A NAME. lean.cc rounds
  // parameters[kLeanStrategyParameterIndex] to an int and subscripts this
  // vector; an out-of-range value is CLAMPED to the last entry, not rejected.
  // So the roster keeps all 36 slots even though only a few are live: that is
  // what makes `--strategy 6` (deploy default, deploy_common.h:130) and
  // `--strategy 22` keep meaning what every run note says they mean, and what
  // keeps grasp.h's `names[21]` patch landing on a slot that exists.
  //
  // Retired slots are padded with "h12_simple_stand" -- the same idiom already
  // used here for 27-30 and used throughout stabilize.h. A padded slot MUST
  // name a file that exists: LoadStrategy's failure return is ignored by
  // lean.cc, which has already cleared the keyframes, and GetCurrentKeyframe()
  // then indexes an empty vector (undefined behaviour). A mistyped strategy
  // number must land on a robot that stands, not on garbage.
  virtual std::vector<std::string> GetStrategyNames() const {
    const std::string kPad = "h12_simple_stand";
    std::vector<std::string> names(36, kPad);
    names[6]  = "h12_simple_stand";         // stand: mission phase 0, deploy default
    names[7]  = "h12_newstand";             // dedicated stand for estimator foot-anchor captures (BRACE)
    names[8]  = "h12_contact_implicit";     // contact-implicit brace discovery (spec 2026-08-30)
    // ★ 2026-09-03 strat 9: SERVO TARGET SWEEP (session "retrieve"). Strat 29
    //  kf0-4 (dive, hover, approach, servo) with the servo rung as a 5 s HOLD
    //  (grasp_close=false, no retract rung) -> release -> standback. No kf5:
    //  the old retract pulled the pelvis back 8-21 cm while braced (29_43/46/52).
    //  Grid point = numerics target_col_x / target_col_y (also shift servo_nominal).
    names[9]  = "h12_brace_servo_sweep";
    names[21] = "h12_simple_reach";         // plain reach bench; Grasp overrides this slot
    names[22] = "h12_simple_forearm_brace"; // brace: mission phase 1
    // ★ 2026-08-17 RECOVERY-ONLY ladder: byte-identical to 22 except the
    // forearm_brace_reach rung's "Reaching Hand Dist" weight is 0 — the
    // post-brace right-arm extension toward the OBJECT is the retrieval
    // mission's payload, not part of the recovery; with it off the ladder
    // goes brace -> consolidate press -> release/standback directly (user:
    // "i want it to not do that extra reaching ... initiate the recovery").
    names[23] = "h12_recovery_only";        // recovery-only brace (22 minus reach)
    // ★ 2026-08-20 strat 24: strat 23 with the vestigial forearm_brace_reach
    // keyframe DELETED (not just neutered). Sequence: stand_up ->
    // forearm_brace_lean -> forearm_brace_release -> standback_r1..r4 ->
    // stand_up. Removes the phase-2 CoM-lunge that sources the "torso hits
    // table" lurch; the reach braced NO harder than lean in the good runs
    // (22/28: ~9 Nm shoulder both) and is a mission leftover, not a recovery
    // need. Carries B (Brace Roll Level 200) on the same lean rung.
    names[24] = "h12_recovery_noreach";     // strat 23 minus the reach keyframe
    // ★ 2026-08-22 strat 25: TARGETING bench. Starts as a byte-copy of strat
    // 24 (the locked recovery config); target-hover phases get inserted
    // between forearm_brace_lean and forearm_brace_release as the design
    // lands. Deliberately independent of 22/23.
    names[25] = "h12_brace_targeting";      // strat 24 + right-arm target hovers
    // ★ 2026-08-24 strat 27: BRACED RETRIEVAL (design:
    // docs/strat27_retrieval_design_2026-08-24.md). Fork of strat 25 with the
    // grasp phases inserted between forearm_brace_lean and release: acquire
    // hover -> descend -> pre-grasp -> straddle (node-side close gate fires
    // here) -> lift -> retract -> tuck -> unchanged recovery. Targets are
    // reach_target_table rungs, servo-updated live from the gripper-cam
    // tag30 channel (rt/object_tag); JSON values = nominal placeholders.
    names[27] = "h12_brace_retrieval";      // strat 25 + grasp/lift/tuck phases
    // ★ 2026-08-26 strat 28: VISION RETRIEVAL. Strat 25's ramp/sustain numbers
    // on the shared phases (stand/dive/release/standback); the middle is the
    // user's vision architecture: arm FORWARD ~45cm to an easy acquire pose
    // (wrist cam sees tag30) -> visual-SERVO rungs ("servo": true, corrections
    // from rt/object_tag) with a TILTED approach ("reach_pitch_deg": ~40 --
    // the tilt drops the grasp centre below the wrist, reaching the
    // table-height block at TRUE B3 with NO squat) -> grasp gate -> retract ->
    // strat 25 recovery. Servo rungs skip the basin lock (lean.cc).
    names[28] = "h12_brace_vision_retrieval";
    // ★ 2026-08-27 strat 29: SIDE GRASP (design:
    // project_strat29_side_grasp_design). Fork of 28 corrected for a SIDEWAYS
    // grasp — jaws close on the block's ±Y faces (lateral), not top/bottom.
    // Body = strat 25's GENTLE reach (Reaching Hand Dist ~300-500, not 28's
    // 800-1800 that over-drove the body forward); grasp = 28's tilt/gate but
    // the reach ABOVE the block then a ~30deg pitch-down diagonal descent onto
    // the sides. Targeting = head-cam TRACK-then-FREEZE (real robot); servo
    // REFINE only at the hover (kf2), descent rungs open-loop. com_gate +
    // reach-arm table-clearance guard extended to this slot. Twin targets true
    // B3; closure is real-only (blob can't close in sim).
    names[29] = "h12_brace_side_grasp";
    // ★ 2026-08-28 strat 30: BATTERY GRASP (session "battery"). Byte-fork of
    // strat 29 (same fixed-frame side grasp, kf0-4 identical) EXCEPT the retract:
    // where 29 arcs the block wide-right off the slab, 30 lifts the battery
    // STRAIGHT UP out of its bay to a standup pose directly above it (kf5-7 hold
    // the grasp column x0.55/y0.16 and raise z 0.13->0.26->0.32, pitch 0). Run
    // WITHOUT H12_S29_GUARD (=0) so the reach-arm exemption stays unconditional
    // and the arm may lift straight up over the slab -- the guard is the ARC
    // enforcer and is wrong for a vertical lift. Twin scene:
    // scene_handless_magpie_table_battery.xml. Recovery rungs (release/standback)
    // inherited from 29 unchanged (shared open wall, owned by session "retrieve").
    names[30] = "h12_brace_battery_grasp";  // strat 29 + straight-up lift (no arc)
    // ★ 2026-08-28 strat 31: REAL BATTERY RAIL brace + grasp (session "battery").
    // Runs on --task "Lean H12 Battery" (Lean_H12_Magpie_battery.xml): LEFT WRIST
    // braces the near rail (0.8065), RIGHT arm reaches the module on the pack top
    // (0.870), lifts it STRAIGHT UP. reach_target_table in RAIL frame (depth from
    // rail near edge 0.3629, lateral, height above rail face). Seed brace keyframe
    // = forearm_brace_lean (wrist-on-rail pose). brace_wrist=1 in the model.
    names[31] = "h12_brace_battery_rail";
    // ★ 2026-08-29 strat 32: SPAWN-BRACED test -- kf0 IS the wrist-on-rail lean
    // (twin --spawn-key), isolating HOLD+REACH from the descent the planner can't do.
    names[32] = "h12_brace_battery_hold";
    names[33] = "h12_simple_grasp";         // grasp-reach bench: mission phase 2 source
    names[34] = "h12_mission_brace_grasp";
    names[35] = "h12_brace_battery_hip";   // 2026-08-29 toes-under-slab HIP-PRESS + wrist brace (task "Lean H12 Battery Hip") // 4-phase retrieval mission (see docs/plans)
    return names;
  }

  // Live per-phase weight blending --------------------------------------- //
  // Per-phase keyframes in the strategy JSON carry a `weight: { name: val }`
  // map. On phase advance we snapshot the live cost weights, compute the new
  // phase's targets (JSON override OR XML default for missing keys), and ramp
  // weight[] from snapshot → target over kPhaseRampSeconds using the same
  // smoothstep curve the residual uses for reach/brace/posture scales.
  // This lets the user isolate behaviours from the strategy file alone:
  // setting "Brace Pos": 0 in a phase silences brace cost without recompiling.
  // Missing keys preserve XML defaults so existing strategies (empty `{}`)
  // keep their old behaviour.
  void ApplyRampedWeights(const mjModel *model, const mjData *data);

 private:
  void SnapshotXmlDefaultWeights(const mjModel *model);
  void PrepareNextPhaseWeights(const mjpc::humanoid::ContactKeyframe &kf);
  void SnapshotCurrentWeightsAsPrev();

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    // Copy the phase-transition timing state along with the keyframe so
    // freshly-spawned residuals (one per rollout thread) see the same ramp
    // progress as the canonical residual_.
    auto rfn = std::make_unique<ResidualFn>(
        this, residual_.residual_keyframe_,
        residual_.keyframe_start_time_,
        residual_.prev_phase_reach_scale_,
        residual_.prev_phase_brace_pos_scale_,
        residual_.prev_phase_posture_scale_,
        residual_.prev_phase_brace_force_target_,
        residual_.prev_posture_key_id_,
        residual_.num_phases_,
        residual_.contact_pair_is_new_);
    return rfn;
  }

  ResidualFn *InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
  std::array<double, 3> target_position_;
  mjpc::humanoid::MotionStrategy motion_strategy_;
  int current_strategy_;

  // ★ 2026-09-04 Table-height parameter state. The `Table H` value last written
  // into model->body_pos, so the write (and the object/target shift that rides
  // with it) happens once per CHANGE instead of once per step -- a per-step
  // write would fight the free object's own dynamics. -1 = never applied.
  double table_h_applied_ = -1.0;

  // Weight-ramp state (parallel to ResidualFn::prev_phase_*_scale_):
  //   xml_default_weights_  -- per-residual default from sensor user data,
  //                            snapshot once in ResetLocked. Used as the
  //                            fallback when a phase's JSON weight map
  //                            doesn't include a particular residual name.
  //   prev_phase_weights_   -- weight[] snapshot at the start of the current
  //                            ramp. Captured mid-ramp so successive phase
  //                            advances blend smoothly through whatever the
  //                            rollouts were actually seeing.
  //   next_phase_weights_   -- target weight[] for the current phase.
  std::vector<double> xml_default_weights_;
  std::vector<double> prev_phase_weights_;
  std::vector<double> next_phase_weights_;
};

class Lean_H12 : public lean {
 public:
  std::string Name() const override { return "Lean H12"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/lean/Lean_H12.xml");
  }
};

// Identical to Lean_H12 (same strategies/costs/weights, nu=27, nq=41) but the
// model carries the two fixed magpie grippers as ~0.506 kg mass-only bodies at
// the wrists. Load with --task "Lean H12 Magpie" when the grippers are mounted;
// fall back to "Lean H12" when they're removed. The grippers are NOT actuated
// here -- open/close is owned by the separate magpie_msgs controller.
class Lean_H12_Magpie : public lean {
 public:
  std::string Name() const override { return "Lean H12 Magpie"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/lean/Lean_H12_Magpie.xml");
  }
};

// ★ 2026-08-28 BATTERY variant (session "battery"). Same robot + costs as
// Lean_H12_Magpie, but the environment is the REAL ARPA battery workcell
// (lean_battery.xml: low pack top 0.870 + a near-side brace RAIL at 0.8065) and
// the brace is the LEFT WRIST on the rail (brace_wrist=1) rather than the forearm
// on a table -- the pack has no flat span for a forearm brace. Drive with
// --task "Lean H12 Battery" + strategy 31 (h12_brace_battery_rail).
class Lean_H12_Battery : public lean {
 public:
  std::string Name() const override { return "Lean H12 Battery"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/lean/Lean_H12_Magpie_battery.xml");
  }
};

// --task "Lean H12 Battery Hip" + strategy 35 (h12_brace_battery_hip): workcell 0.30 m
// closer, toes under the slab, hips press the slab edge, wrist brace beside the hip.
class Lean_H12_BatteryHip : public lean {
 public:
  std::string Name() const override { return "Lean H12 Battery Hip"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/lean/Lean_H12_Magpie_battery_hip.xml");
  }
};

// --task "Lean H12 Magpie Discover" + strategy 8 (h12_contact_implicit): the FLAT
// magpie table brought 0.12 m CLOSER (workcell -0.12 x, lean_discover.xml) so the
// forearm's balance-limited settling point lands ON reachable wood. Same robot/costs
// as Lean_H12_Magpie; only the table x is shifted, in BOTH this planner model and the
// twin (scene_close_real.xml) for parity -- so the Brace-Pos target follows the table.
// Contact-implicit brace discovery, closer-stance parity build (spec 2026-08-30).
class Lean_H12_MagpieDiscover : public lean {
 public:
  std::string Name() const override { return "Lean H12 Magpie Discover"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/lean/Lean_H12_Magpie_discover.xml");
  }
};

}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_BENCH_LEAN_LEAN_H_
