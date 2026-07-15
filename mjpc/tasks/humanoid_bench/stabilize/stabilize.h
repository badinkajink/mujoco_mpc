#ifndef MJPC_TASKS_HUMANOID_BENCH_STABILIZE_STABILIZE_H_
#define MJPC_TASKS_HUMANOID_BENCH_STABILIZE_STABILIZE_H_

#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include "mjpc/tasks/humanoid/interact/contact_keyframe.h"
#include "mjpc/tasks/humanoid/interact/motion_strategy.h"
#include "mujoco/mujoco.h"

namespace mjpc {

constexpr int kStabilizeStrategyParameterIndex = 1;

// Manual-phase override. -1 (default) = auto-advance through keyframes
// based on success_sustain_time + target_distance_tolerance. 0..N-1 =
// hold at that keyframe index regardless of progress. Lets the user
// scrub through the loaded strategy's phases without reloading (which
// would reset to keyframe 0 and snap the body back through stand_up).
constexpr int kStabilizePhaseParameterIndex = 2;

// ARM_PLAN mode-2 preview seam (2026-07-10): gRPC-settable plan = trigger +
// duration + 14 joint goals (motor idx 13..26, L sh_p/sh_r/sh_y/elbow/wr_r/
// wr_p/wr_y then R same). XML order after residual_Phase fixes the indices.
constexpr int kStabilizeArmPlanActiveParameterIndex = 3;
constexpr int kStabilizeArmPlanSecParameterIndex = 4;
constexpr int kStabilizeArmGoalParameterIndex0 = 5;   // J0..J13 -> 5..18

// LIVE cmd_vel teleop (WSS port, 2026-07-12) — the drive (strat 24) seam.
// gRPC-settable via SetTaskParameters exactly like the arm-plan block above.
// Vx/Vy are BODY-frame m/s; the TransitionLocked governor clamps/slews/rotates
// them before the trot sees anything. Seq is a client heartbeat counter:
// unchanged for > 1 s means the client died -> the watchdog zeroes the command.
// APPENDED after Arm Goal J13 (18) on purpose: indices 0..18 keep their meaning,
// so the perfected stand (strat 6) reads exactly the parameters it always did.
constexpr int kStabilizeCmdActiveParameterIndex = 19;
constexpr int kStabilizeCmdVxParameterIndex = 20;
constexpr int kStabilizeCmdVyParameterIndex = 21;
constexpr int kStabilizeCmdSeqParameterIndex = 22;
constexpr int kStabilizeCmdWzParameterIndex = 23;   // yaw-rate (drive strat only)

constexpr char kStabilizeStrategyFilePath[] =
    SOURCE_DIR "/mjpc/tasks/humanoid_bench/stabilize/strategies/";

class stabilize : public Task {
 public:
  std::string Name() const override = 0;

  std::string XmlPath() const override = 0;

  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const stabilize *task,
                        const mjpc::humanoid::ContactKeyframe& kf =
                            mjpc::humanoid::ContactKeyframe(),
                        mjtNum keyframe_start_time = 0.0,
                        mjtNum prev_reach_scale = 0.0,
                        mjtNum prev_brace_pos_scale = 0.0,
                        mjtNum prev_posture_scale = 1.0,
                        mjtNum prev_brace_force_target = 0.0,
                        int prev_posture_key_id = 0,
                        int num_phases = 1,
                        const bool* contact_pair_is_new = nullptr,
                        mjtNum catch_ep_t0 = -1.0e9,
                        bool catch_ep_left = false)
        : mjpc::BaseResidualFn(task),
          residual_keyframe_(kf),
          keyframe_start_time_(keyframe_start_time),
          catch_ep_t0_(catch_ep_t0),
          catch_ep_left_(catch_ep_left),
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
    // Tried bumping to 3.0 to slow the arm swing during 2→3 — backfired:
    // with MPC horizon 1.0s the lean_forward gradient stayed weak for ~2s
    // while Height (head wants to stay high, weight 35 effective in
    // arm_contact_or_lean) was full strength — body settled into a slight
    // backward bend as the cheap local optimum. If arm swing is still too
    // fast at 1.5s, a surgical fix (Brace Hand Velocity residual active
    // only during arm_plant) is preferable to slowing every cost ramp.
    static constexpr mjtNum kPhaseRampSeconds = 1.5;

    // STAND-UP-only target-pose ramp duration (see the asymmetric target ramp
    // in stabilize.cc). Deliberately LONGER than kPhaseRampSeconds: at the live
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

    enum LeanMode {
      kModeReach = 0,
      kModeRetrieve,
      kNumMode
    };

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

    // FORCED CATCH-STEP episode (strategy 20 quiet stand, 2026-07-03). The
    // cost-side catch-step alone never lifts a foot: a sampling planner
    // cannot DISCOVER a contact-breaking swing (every short-horizon rollout
    // prefers both feet planted; trace-proven on the backward push). The trot
    // already solves this with the stabilize::ModifyControl channel-freeze;
    // these members arm the SAME scripted swing as a one-shot episode:
    // latched in TransitionLocked (real plant state, once per plan) when the
    // capture excursion escapes catch_full, executed open-loop by
    // ModifyControl for catch_step_sec, then the planner owns the legs again.
    // t0 <= -1e8 means "no episode ever". v4 (2026-07-03): ALSO propagated
    // into every rollout residual copy (ResidualLocked ctor args) so the
    // COST side expects the scripted swing -- same foot, same window -- in
    // every sampled trajectory. Without this the rollout residuals re-derive
    // their own foot pick from diverged rollout states and the Foot-Up cost
    // fights the freeze in half the samples (the v1-v3 coin-flip catches).
    // This mirrors WHY the trot works: its cost gait + freeze share a clock.
    mjtNum catch_ep_t0_ = -1.0e9;   // plant time the active episode began
    bool catch_ep_left_ = false;    // true = left foot swings
    // R3 (2026-07-04): latch PERSISTENCE. Real Unitree deploys exclude base
    // lin-vel from the loop entirely (unobservable on HW); our danger's tau*vx
    // term rides the noisy EKF estimate, and quiet-stand sway historically
    // exceeded 0.06 on real. Require the crossing to be SUSTAINED
    // catch_persist_sec before latching a march (sway spikes are brief; a
    // genuine backward fall stays above threshold). Canonical-only member.
    mjtNum catch_cross_t0_ = -1.0e9;  // when -ex first exceeded the threshold

    // ----- LIVE cmd_vel teleop (WSS port, 2026-07-12) ---------------------
    // Written ONLY by the TransitionLocked governor (once per plan, under the
    // transition lock, before the rollout workers fan out); read by Residual()
    // and by stabilize::ModifyControl. cmd_active_=false => both readers take
    // the legacy trot_des_vel numeric path => byte-identical to the validated
    // static-numeric configs, so a process that never gets a command behaves
    // exactly as before.
    //
    // ★ cmd_active_ + cmd_vdes_world_ ARE propagated into every rollout copy by
    // ResidualLocked (unlike lean.cc, which propagates only drive_gait_amp_/
    // drive_yaw_des_ -- so ITS rollout residuals silently see cmd_active_=false
    // and cost the gait as an IN-PLACE trot while ModifyControl drives the swing
    // FORWARD at the governed v_des). That disagreement makes the sampler fight
    // the open-loop walk drive on every rollout, which is the same signature as
    // the documented "free-twin ~6s walk ceiling". Propagating them is what the
    // "MUST match ModifyControl" comments in the residual actually require.
    bool   cmd_active_ = false;
    double cmd_vdes_world_[2] = {0.0, 0.0};  // governed v_des, WORLD frame
    // --- canonical-only bookkeeping (never read from a rollout copy) ---
    double cmd_filt_[2] = {0.0, 0.0};        // slew-limited BODY-frame command
    bool   cmd_starved_ = false;             // log-once latch, heartbeat watchdog
    double cmd_last_seq_ = -1.0;
    double cmd_seq_time_ = -1.0;
    double cmd_prev_time_ = -1.0;
    double cmd_settle_until_ = -1.0;         // settle-through-zero dwell
    double cmd_wz_ = 0.0;                    // governed yaw-rate [rad/s]

    // ----- WSS drive FSM (strat 24 stand<->trot teleop) --------------------
    // drive_gait_amp_ (0..1): gait-enable multiplier. 0 => feet plant => the
    // balance-gated stand (still steps to catch a push); 1 => full trot.
    // Written by the TransitionLocked latch, propagated into the plan snapshot
    // by ResidualLocked so the COST gates g_amp exactly the way ModifyControl
    // does. drive_yaw_des_ = integrated desired WORLD heading [rad] (Body Yaw
    // reference). The latch bookkeeping below stays canonical-only.
    double drive_gait_amp_ = 0.0;
    double drive_yaw_des_ = 0.0;
    bool   drive_walk_ = false;
    double drive_idle_since_ = -1.0;
    double drive_ramp_prev_ = -1.0;

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
    // per-strategy gate the 2026-06-08 revert note (stabilize.cc) said the ramp needed.
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
    friend class stabilize;

    static constexpr double kHandDistThreshold = 0.0;
    static constexpr double kContactStableTime = 0.0;
    static constexpr double kContactForceThreshold = 0.0;

    void ContactResidual(const mjModel *model, const mjData *data,
                         double *residual, int *counter) const;
  };

  stabilize() : residual_(this), current_strategy_(-1) {
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

  // ARM_PLAN mode-2 rollout injection: per-worker eq-disable + qfrc PD toward
  // the min-jerk segment latched in TransitionLocked, so every rollout
  // physically replays the commanded future arm motion (preview). Pristine
  // process (plan never armed) = first-branch return = byte-identical.
  void ModifyRolloutState(const mjModel* model, mjData* data) const override;

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
  virtual std::vector<std::string> GetStrategyNames() const {
    // Lower-body STABILIZE strategy slots. Only slot 6 (stand) is real today;
    // 0-5 are placeholders reserved for future lower-body skills (weight-shift,
    // crouch, brace-step). Slot 6 == stand keeps parity with the
    // h12_lower_body_controller --strategy 6 default.
    return {
        "stabilize_placeholder",     // 0
        "stabilize_placeholder",     // 1
        "stabilize_placeholder",     // 2
        "stabilize_placeholder",     // 3
        "stabilize_placeholder",     // 4
        "stabilize_placeholder",     // 5
        "stabilize_simple_stand",    // 6  legs-only balance hold at home (REAL)
        "stabilize_placeholder",     // 7
        "stabilize_placeholder",     // 8
        "stabilize_placeholder",     // 9
        "stabilize_placeholder",     // 10
        "stabilize_placeholder",     // 11
        "stabilize_placeholder",     // 12
        "stabilize_placeholder",     // 13
        "stabilize_placeholder",     // 14
        "stabilize_placeholder",     // 15
        "stabilize_placeholder",     // 16
        "stabilize_placeholder",     // 17
        "stabilize_placeholder",     // 18
        "stabilize_placeholder",     // 19
        "stabilize_simple_stumble",  // 20  balance-gated march + catch-march (push recovery)
        "stabilize_placeholder",     // 21
        "stabilize_simple_walk",     // 22  forward walk = trot + a baked v_des (walk_des_vel_x)
        "stabilize_simple_trot",     // 23  capture-point in-place trot
        "stabilize_simple_drive",    // 24  WSS teleop drive: stand<->trot FSM on live cmd_vel
        "stabilize_placeholder",     // 25  reserved (lean slot 25 = straighten; kept clear here)
        "stabilize_lockstand",       // 26  locked-knee wide-stance balance hold (strut stand)
    };
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
        residual_.contact_pair_is_new_,
        residual_.catch_ep_t0_,
        residual_.catch_ep_left_);
    // WSS drive: propagate the FSM gait-enable + heading into this plan's
    // snapshot so the rollout COST gates g_amp (and aims Body Yaw) exactly the
    // way stabilize::ModifyControl does. Non-drive strategies leave these 0.
    rfn->drive_gait_amp_ = residual_.drive_gait_amp_;
    rfn->drive_yaw_des_ = residual_.drive_yaw_des_;
    // ★ AND the governed command itself -- without this the rollouts cost an
    // IN-PLACE trot (cmd_active_=false -> v_des from the static numerics = 0)
    // while ModifyControl forces the swing FORWARD, so the sampler spends every
    // rollout cancelling the walk drive. See the note on cmd_active_ above.
    rfn->cmd_active_ = residual_.cmd_active_;
    rfn->cmd_vdes_world_[0] = residual_.cmd_vdes_world_[0];
    rfn->cmd_vdes_world_[1] = residual_.cmd_vdes_world_[1];
    return rfn;
  }

  ResidualFn *InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
  std::array<double, 3> target_position_;
  mjpc::humanoid::MotionStrategy motion_strategy_;
  int current_strategy_;

  // ARM_PLAN mode-2 state. Written ONLY in TransitionLocked (once per plan,
  // under the transition lock, before rollout workers fan out -- same
  // guarantee the F1-A shared-eq_data write relies on); read from
  // ModifyRolloutState on the worker threads. arm_plan_touched_ stays true
  // for the process lifetime once a plan has armed: reused worker mjData may
  // carry stale eq_active/qfrc_applied, so the restore path must keep running.
  bool arm_plan_active_ = false;
  bool arm_plan_touched_ = false;
  double arm_plan_t0_ = 0.0;
  double arm_plan_T_ = 1.0;
  double arm_plan_q0_[14] = {0};
  double arm_plan_qg_[14] = {0};

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

class Stabilize_H12_Magpie : public stabilize {
 public:
  std::string Name() const override { return "Stabilize H12 Magpie"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/stabilize/Stabilize_H12_Magpie.xml");
  }
};

}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_BENCH_STABILIZE_STABILIZE_H_
