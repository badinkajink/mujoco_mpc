#ifndef MJPC_TASKS_HUMANOID_BENCH_LEAN_LEAN_H_
#define MJPC_TASKS_HUMANOID_BENCH_LEAN_LEAN_H_

#include <atomic>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include "mjpc/tasks/humanoid/interact/contact_keyframe.h"
#include "mjpc/tasks/humanoid/interact/motion_strategy.h"
#include "mjpc/tasks/humanoid_bench/h12_common/h12_plan_snapshot.h"
#include "mujoco/mujoco.h"

namespace mjpc {

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

// LIVE cmd_vel teleop (WASD/gamepad/Nav2, 2026-07-03). gRPC-settable via
// SetTaskParameters like the Reach seam above. Vx/Vy are BODY-frame m/s;
// the TransitionLocked governor clamps/slews/rotates them before the trot
// sees anything. Seq is a client heartbeat counter: unchanged > 1 s means
// the client died -> watchdog zeroes the command.
constexpr int kLeanCmdActiveParameterIndex = 7;
constexpr int kLeanCmdVxParameterIndex = 8;
constexpr int kLeanCmdVyParameterIndex = 9;
constexpr int kLeanCmdSeqParameterIndex = 10;
constexpr int kLeanCmdWzParameterIndex = 11;   // V3 yaw-rate (drive strat only)

constexpr char kLeanStrategyFilePath[] =
    SOURCE_DIR "/mjpc/tasks/humanoid_bench/lean/strategies/";

class lean : public Task {
 public:
  std::string Name() const override = 0;

  std::string XmlPath() const override = 0;

  // Per-plan ROLLOUT-VISIBLE state (stage 4a): the twins' shared snapshot
  // (h12_common/h12_plan_snapshot.h). Lean has no task-specific extras; the
  // named subclass keeps symmetry with stabilize::PlanSnapshot. ResidualLocked
  // copies this WHOLESALE into every rollout residual -- add a rollout-visible
  // field to the base and it propagates automatically.
  struct PlanSnapshot : mjpc::h12::PlanSnapshotBase {};

  class ResidualFn : public mjpc::BaseResidualFn, public PlanSnapshot {
   public:
    explicit ResidualFn(const lean *task) : mjpc::BaseResidualFn(task) {}
    ResidualFn(const lean *task, const PlanSnapshot &snap)
        : mjpc::BaseResidualFn(task), PlanSnapshot(snap) {}

    void Residual(const mjModel *model, const mjData *data,
                  double *residual) const override;

    // Phase-transition ramp duration: the reach + brace cost scales smoothly
    // interpolate from their previous-phase values to the new-phase values
    // over this many seconds after each keyframe advance. 1.5s gives the
    // robot time to absorb the new gradient instead of being shoved forward.
    // Raising it is NOT a free fix for abrupt arm swings — a 3.0s trial
    // backfired (history: see mjpc/tasks/humanoid_bench/HISTORY.md).
    static constexpr mjtNum kPhaseRampSeconds = 1.5;

   protected:
    // (Rollout-visible per-plan state lives in the PlanSnapshot base above;
    //  everything below is CANONICAL-ONLY bookkeeping, never read from a
    //  rollout copy and deliberately not propagated: the cmd governor's
    //  slew/watchdog state and the drive FSM latch bookkeeping.)
    double cmd_filt_[2] = {0.0, 0.0};
    bool   cmd_starved_ = false;   // log-once latch for the heartbeat watchdog
    double cmd_last_seq_ = -1.0;
    double cmd_seq_time_ = -1.0;
    double cmd_prev_time_ = -1.0;
    double cmd_settle_until_ = -1.0;
    double cmd_wz_ = 0.0;          // governed yaw-rate [rad/s] (TransitionLocked only)
    bool   drive_walk_ = false;
    double drive_idle_since_ = -1.0;
    double drive_ramp_prev_ = -1.0;

   private:
    friend class lean;

    static constexpr double kHandDistThreshold = 0.0;
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

  // Populate phase-aware monitoring metrics (reach, CoP, ICP, brace force,
  // saturation, etc.) for the Research GUI / headless analyzer. Reads the
  // current keyframe + sensor stack; safe to call from the gRPC poll loop.
  // See task.h ComputeMetrics for contract.
  void ComputeMetrics(const mjModel *model, const mjData *data,
                      std::map<std::string, double> *metrics,
                      std::string *phase_name) const override;

  // Per-strategy planner model-numeric overrides (e.g. sampling_spline_points).
  // See task.h PlannerNumericOverrides for the contract. Keyed by strategy NAME
  // (GetStrategyNames()[strategy]) so the override follows the strategy across
  // the Lean_H12 / Lean_H12_Hands model variants.
  std::map<std::string, double> PlannerNumericOverrides(
      int strategy) const override;

  // Open-loop channel freeze for the TROT strategy (phase name contains "trot").
  // Hard-writes the swing-leg actuator channels (hip_pitch/knee/ankle_pitch of
  // whichever foot is in its swing half-cycle) to the scripted Tier-B fold the
  // sampler keeps refusing in the cost, so the foot lifts open-loop and the
  // sampler must balance the stance leg + upper body AROUND the forced swing.
  // Blended in by the gait bump so the stance leg stays planner-controlled and
  // the swing/stance handoff is smooth. is_trot-gated -> all other strategies
  // (incl. strat 20 "stumble_march") are byte-identical (default no-op). See
  // Task::ModifyControl + the gait clock in ResidualFn::Residual.
  void ModifyControl(const mjModel* model, const double* qpos,
                     const double* qvel, double time,
                     double* ctrl) const override;

  // ---- SPLIT-BODY upper-body pause lock (2026-07-17) --------------------- //
  // The whole-body split deploy core (mjpc_split_core) calls SetUpperLocked
  // when its upper-channel pause toggle flips. While locked, the frame_task IK
  // owns the physical arms, so this task must stop planning them:
  //   - ModifyRolloutState activates the per-data upper-joint equality locks
  //     (data->eq_active; the model ships them active="false") so every rollout
  //     holds torso+arms at the eq_data target -- which the deploy node
  //     retargets to the MEASURED pose each tick, exactly the legs-only node's
  //     arm_aware arrangement -- and restores them inactive when unlocked
  //     (worker mjData persists across rollouts; see task.h hygiene note).
  //   - ModifyControl pins ctrl rows 12..26 to the same eq target so the
  //     sampler's arm channels cannot fight the locks and the executed action's
  //     arm rows equal the measured pose.
  // Requires a model that DEFINES the 15 upper-joint equality constraints
  // (Split_H12_Magpie.xml); on models without them (plain Lean_H12*) the lock
  // silently degrades to "arms still planned free" -- the deploy node warns.
  void SetUpperLocked(bool locked) override {
    if (locked) upper_locked_touched_.store(true, std::memory_order_relaxed);
    upper_locked_.store(locked, std::memory_order_release);
  }
  void ModifyRolloutState(const mjModel* model, mjData* data) const override;

  // Slider layout (Lean H12) — user's 6-phase decomposition:
  //   0  stand            — stand_up
  //   1  arm_extend       — stand → arm_extend_standing (arm out, body upright)
  //   2  lean_no_brace    — stand → extend → lean_with_arm_no_brace
  //   3  brace_hand_lean  — stand → extend → lean → arm_plant → lean_forward
  //   4  forearm_brace    — above + forearm_brace_lean (hand+elbow on table)
  //   5  full_pipeline    — identical to slot 4 now: ends in a HELD two-foot
  //                         braced lean (DEFAULT).
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
  virtual std::vector<std::string> GetStrategyNames() const {
    return {// 0-5: the lean pipeline (unchanged; default Strategy=5)
            "h12_pipeline_stand",
            "h12_pipeline_arm_extend",
            "h12_pipeline_lean_no_brace",
            "h12_pipeline_brace_hand_lean",
            "h12_pipeline_forearm_brace",
            "h12_pipeline_full_pipeline",
            // 6+: standalone single-skill tasks for incremental sim2real
            // bring-up. Each is one indefinite phase. Pose tasks (crouch,
            // arms_*, lean_*, torso_twist) name a model <key> keyframe that
            // the Posture/Control costs track (see lean.cc pose-library
            // block). Slide the Strategy parameter to select; raise the
            // residual_Strategy slider max in Lean_H12.xml accordingly.
            "h12_simple_stand",          // 6  balance hold at home pose
            "h12_simple_reach_forward",  // 7  left arm forward, body squared
            "h12_simple_crouch",         // 8  symmetric squat, torso upright
            "h12_simple_arms_sideways",  // 9  bilateral lateral raise (T)
            "h12_simple_arms_forward",   // 10 bilateral front raise
            "h12_simple_arms_overhead",  // 11 both arms overhead
            "h12_simple_single_arm_raise",  // 12 right arm to the side
            "h12_simple_lean_left",      // 13 lateral weight-shift onto L
            "h12_simple_lean_right",     // 14 lateral weight-shift onto R
            "h12_simple_torso_twist",    // 15 waist yaw to the left
            "h12_simple_counterbalance", // 16 left-arm forward reach + emergent
                                         //    right-arm/torso counterweight
            "h12_simple_squat",          // 17 cyclic squat: crouch(5s)->stand(5s)->loop
                                         //    2-phase strategy; phase machine wraps
                                         //    via NextKeyframe() modulo (motion_strategy.cc)
            "h12_simple_squatter",       // 18 native squatter: 2-phase stand_up<->crouch
                                         //    auto-cycle reusing strat 6 (stand) + strat 8
                                         //    (crouch) weight blocks VERBATIM. The MJPC-side
                                         //    twin of the node's --strategy 18 sequencer
                                         //    (h12_control_node.cc), so the planner/monitor/
                                         //    analyzer can drive the squat natively.
            "h12_simple_jab",            // 19 standing boxing jab: 3-phase
                                         //    stand_up -> jab_guard -> jab_extend auto-cycle. A
                                         //    stand_up LEAD-IN (gentle 3s ramp from home, like the
                                         //    stand/squatter strategies) settles the legs/balance
                                         //    FIRST, THEN the arms raise to guard, THEN the RIGHT
                                         //    arm punches straight forward (shoulder_pitch -1.0,
                                         //    elbow straightens) and retracts. The lead-in is what
                                         //    gives the real robot a proper home->stand->guard
                                         //    transition (raising arms straight from home spun it).
                                         //    WBC base = strat 6 (stand) weights, Posture 60 for
                                         //    punch authority; asymmetric arms are free (Symmetry
                                         //    penalizes only legs). Slow, deliberate cadence.
            "h12_simple_stumble",        // 20 stumble: gait-clock STEPPING. A
                                         //    stand_up LEAD-IN settles balance, THEN a continuous
                                         //    gait clock alternates the feet (antiphase, ~1.6 Hz,
                                         //    duty 0.65, 6 cm step) so the robot STEPS IN PLACE to
                                         //    stay up ("stumble") and WALKS when a desired CoM
                                         //    velocity is commanded (nav-package hook in lean.cc).
                                         //    Unlike every other lean strategy (both feet planted,
                                         //    ankle-only balance) it RELOCATES the support polygon
                                         //    under the CoM -> escapes the weak-ankle ceiling. New
                                         //    Gait + Step Place cost terms, name-gated; Symmetry/
                                         //    Lateral Center/Knees OFF (stepping breaks the static-
                                         //    stance symmetry on purpose).
            "h12_simple_reach",          // 21 reach-to-target: standalone REACH
                                         //    primitive. A stand_up LEAD-IN (reach_stand)
                                         //    settles balance, THEN the nearer hand reaches an
                                         //    EXTERNAL target (object_pos mocap = the `reach_target`
                                         //    numeric, or a vision/nav input) while the body stays
                                         //    upright and the feet planted -- no lean, no brace. The
                                         //    target is auto-clamped to a balance-safe workspace box
                                         //    (lean.cc reach_to_target): in-reach => hand on target,
                                         //    out-of-reach => fully-extended arm, still standing
                                         //    (the beyond-reach regime is LEAN's job). The reusable
                                         //    base primitive for pick/retrieve; the lean pipeline is
                                         //    this reach + brace + whole-body pitch for FAR targets.
            "h12_simple_forearm_brace",  // 22  pre-lean forearm brace
            "h12_simple_trot",  "h12_simple_drive", "h12_straighten", "h12_simple_jump", // 23 trot (leg-lift test vehicle), 24 WSS teleop drive (stand<->trot FSM), 25 pre-stand STRAIGHTEN/bring-up (wide-basin drive-to-upright), 26 JUMP (one-shot in-place hop: time-scheduled crouch->push->flight->absorb->stand, flip-pattern residual schedule; Magpie-primary)
            "h12_simple_stand", "h12_simple_stand", "h12_simple_stand", "h12_simple_stand", // 27-30 reserved
            "h12_lean_stand",            // 31
            "h12_lean_reach",            // 32
            "h12_lean_counterbalance",   // 33
            "h12_lean_brace",            // 34
            "h12_lean_full"};            // 35
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

  // SPLIT-BODY upper lock state (see SetUpperLocked above). Written by the
  // deploy node's 200 Hz control loop on a toggle edge, read lock-free by every
  // rollout worker in ModifyRolloutState/ModifyControl. `touched_` is one-way:
  // once the lock has ever been active, the unlocked path must keep RESTORING
  // eq_active=0 on worker mjData (they persist across rollouts); a pristine
  // process (never locked) early-outs and is byte-identical to before.
  std::atomic<bool> upper_locked_{false};
  std::atomic<bool> upper_locked_touched_{false};

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    // Wholesale copy of the canonical residual_'s PlanSnapshot (stage 4a):
    // keyframe/ramp state, straighten seed, the governed command, and the
    // drive FSM outputs propagate in ONE struct assignment. Fields added to
    // the snapshot propagate automatically (the old per-field list is the
    // code shape that produced the 2026-07-12 walk-ceiling forgot-to-copy
    // bug; history: see mjpc/tasks/humanoid_bench/HISTORY.md).
    return std::make_unique<ResidualFn>(
        this, static_cast<const PlanSnapshot &>(residual_));
  }

  ResidualFn *InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
  std::array<double, 3> target_position_;
  mjpc::humanoid::MotionStrategy motion_strategy_;
  int current_strategy_;

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

// Lean_H12_Magpie + the 15 upper-body (torso + two arms) joint equality
// constraints defined INACTIVE (Split_H12_Magpie.xml wraps the Magpie model).
// This is the model the whole-body SPLIT deploy core (mjpc_split_core) must
// load: while its upper channel is active the locks stay off and the planner
// drives all 27 joints exactly like "Lean H12 Magpie"; when the upper channel
// is PAUSED (frame_task IK owns the arms) the deploy core activates the locks
// per rollout-data and retargets them to the measured arm pose each tick
// (lean::SetUpperLocked / ModifyRolloutState / ModifyControl), so the planner
// balances the legs AROUND the real arms instead of planning arm motion it
// cannot execute. Same strategies/costs/weights as the Magpie task.
class Lean_H12_Magpie_Split : public lean {
 public:
  std::string Name() const override { return "Lean H12 Magpie Split"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/lean/Split_H12_Magpie.xml");
  }
};

class Lean_H12_Hands : public lean {
 public:
  std::string Name() const override { return "Lean H12 Hands"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/lean/Lean_H12_Hands.xml");
  }

  // Mirrors Lean_H12::GetStrategyNames slot-for-slot. NOTE: the Hands model's
  // Posture cost (dim 27) does not reach the right arm (qpos[39:46] — the
  // 12-DOF dexterous hands shift it), so right-arm pose tasks are partial on
  // this variant until the Hands Posture dim is widened. Lower-body + left-arm
  // poses (stand/crouch/lean_*/arms_forward-left) track correctly.
  std::vector<std::string> GetStrategyNames() const override {
    return {"h12_hands_pipeline_stand",
            "h12_hands_pipeline_arm_extend",
            "h12_hands_pipeline_lean_no_brace",
            "h12_hands_pipeline_brace_hand_lean",
            "h12_hands_pipeline_forearm_brace",
            "h12_hands_pipeline_full_pipeline",
            "h12_hands_simple_stand",
            "h12_hands_simple_reach_forward",
            "h12_hands_simple_crouch",
            "h12_hands_simple_arms_sideways",
            "h12_hands_simple_arms_forward",
            "h12_hands_simple_arms_overhead",
            "h12_hands_simple_single_arm_raise",
            "h12_hands_simple_lean_left",
            "h12_hands_simple_lean_right",
            "h12_hands_simple_torso_twist",
            "h12_hands_simple_counterbalance",  // 16 (mirrors Lean_H12 slot 16)
            "h12_hands_simple_squat",   // 17 cyclic squat (mirrors Lean_H12 slot 17)
            "h12_hands_simple_squatter",  // 18 native squatter (mirrors Lean_H12 slot 18)
            "h12_hands_simple_jab",     // 19 standing boxing jab (mirrors Lean_H12 slot 19).
                                        //    Right-arm punch tracks only partially on the
                                        //    Hands model (Posture dim 27 misses the right arm).
            "h12_hands_simple_stumble",  // 20 gait-clock stepping (mirrors Lean_H12 slot 20).
            "h12_hands_simple_reach",    // 21 reach-to-target (mirrors Lean_H12 slot 21).
            "h12_hands_simple_forearm_brace",  // 22  pre-lean forearm brace
            "h12_hands_simple_trot", "h12_hands_simple_drive", "h12_hands_simple_stand", "h12_hands_simple_stand", // 23 trot (leg-lift test vehicle), 24 WSS teleop drive (mirrors base slot 24), 25-26 reserved
            "h12_hands_simple_stand", "h12_hands_simple_stand", "h12_hands_simple_stand", "h12_hands_simple_stand", // 27-30 reserved
            "h12_hands_lean_stand",            // 31
            "h12_hands_lean_reach",            // 32
            "h12_hands_lean_counterbalance",   // 33
            "h12_hands_lean_brace",            // 34
            "h12_hands_lean_full"};            // 35
  }
};

}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_BENCH_LEAN_LEAN_H_
