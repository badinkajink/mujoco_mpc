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
#include "mjpc/tasks/humanoid_bench/h12_common/h12_plan_snapshot.h"
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
// STRAIGHTEN funnel arm (2026-07-15): 1 (default) = the live-seed funnel seeds +
// its target_ramp_sec glide clock starts at strategy load (GUI/twin behaviour).
// 0 = DISARMED: the task re-pins the seed to the live pose and freezes the phase
// clocks every tick. The deploy node sets 0 at boot under --straighten_start and
// 1 when the ENTER hand-over reaches full planner authority, so the glide runs
// WHILE the planner rises instead of burning down during the operator hold.
constexpr int kStabilizeFunnelArmParameterIndex = 24;

constexpr char kStabilizeStrategyFilePath[] =
    SOURCE_DIR "/mjpc/tasks/humanoid_bench/stabilize/strategies/";

class stabilize : public Task {
 public:
  std::string Name() const override = 0;

  std::string XmlPath() const override = 0;

  // Per-plan ROLLOUT-VISIBLE state (stage 4a): the twins' shared snapshot
  // (h12_common/h12_plan_snapshot.h) + stabilize's extras. ResidualLocked
  // copies this WHOLESALE into every rollout residual -- add a rollout-visible
  // field HERE (or in the base) and it propagates automatically.
  struct PlanSnapshot : mjpc::h12::PlanSnapshotBase {
    // FORCED CATCH-STEP episode (strategy 20): latched in TransitionLocked
    // (real plant state, once per plan) when the capture excursion escapes
    // catch_full; executed open-loop by ModifyControl for catch_step_sec.
    // In the snapshot (v4) so the COST side expects the scripted swing --
    // same foot, same window -- in every sampled trajectory; without it the
    // Foot-Up cost fights the freeze in half the samples. t0 <= -1e8 = "no
    // episode ever". (history: see mjpc/tasks/humanoid_bench/HISTORY.md)
    mjtNum catch_ep_t0_ = -1.0e9;   // plant time the active episode began
    bool catch_ep_left_ = false;    // true = left foot swings
    // STRAIGHTEN (strat 25) foot anchor: while strat 25 is active the "Foot
    // Stability" residual anchors to THESE captured world positions instead of
    // the world-home constants (which the real estimator's drifting odometric
    // frame invalidates). Seeded at strategy load, re-pinned while the funnel
    // is deploy-held. Default = the home constants (benign if never seeded).
    // In the snapshot so rollout copies cost the same anchor as the canonical.
    double straighten_foot_anchor_[4] = {0.2196, -0.163, 0.2196, 0.163};  // Rx,Ry,Lx,Ly
  };

  class ResidualFn : public mjpc::BaseResidualFn, public PlanSnapshot {
   public:
    explicit ResidualFn(const stabilize *task) : mjpc::BaseResidualFn(task) {}
    ResidualFn(const stabilize *task, const PlanSnapshot &snap)
        : mjpc::BaseResidualFn(task), PlanSnapshot(snap) {}

    void Residual(const mjModel *model, const mjData *data,
                  double *residual) const override;

    // Phase-transition ramp duration: the reach + brace cost scales smoothly
    // interpolate from their previous-phase values to the new-phase values
    // over this many seconds after each keyframe advance. 1.5s gives the
    // robot time to absorb the new gradient instead of being shoved forward.
    // Raising it was tried and rejected (history: see
    // mjpc/tasks/humanoid_bench/HISTORY.md).
    static constexpr mjtNum kPhaseRampSeconds = 1.5;

   protected:
    // (Rollout-visible per-plan state lives in the PlanSnapshot bases above;
    //  everything below is CANONICAL-ONLY bookkeeping, never read from a
    //  rollout copy and deliberately not propagated.)

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
    // exactly as before. (cmd_active_ + cmd_vdes_world_ live in the
    // PlanSnapshot base -- propagated to every rollout copy.)
    // --- canonical-only bookkeeping (never read from a rollout copy) ---
    double cmd_filt_[2] = {0.0, 0.0};        // slew-limited BODY-frame command
    bool   cmd_starved_ = false;             // log-once latch, heartbeat watchdog
    double cmd_last_seq_ = -1.0;
    double cmd_seq_time_ = -1.0;
    double cmd_prev_time_ = -1.0;
    double cmd_settle_until_ = -1.0;         // settle-through-zero dwell
    double cmd_wz_ = 0.0;                    // governed yaw-rate [rad/s]

    // ----- WSS drive FSM (strat 24 stand<->trot teleop) --------------------
    // drive_gait_amp_/drive_yaw_des_ (the FSM OUTPUTS the cost gates on) live
    // in the PlanSnapshot base; the latch bookkeeping here is canonical-only.
    bool   drive_walk_ = false;
    double drive_idle_since_ = -1.0;
    double drive_ramp_prev_ = -1.0;

   private:
    friend class stabilize;

    static constexpr double kHandDistThreshold = 0.0;
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
  // model variants.
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

  virtual std::vector<std::string> GetStrategyNames() const {
    // Lower-body STABILIZE strategy slots. Seven slots are real today:
    // 6 (stand), 20 (stumble), 22 (walk), 23 (trot), 24 (drive),
    // 25 (straighten), 26 (lockstand). Slots 0-5, 7-19 and 21 are placeholders
    // kept for numbering parity with the lean task's slot layout (slot 6 ==
    // stand also matches the h12_lower_body_controller --strategy 6 default).
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
        "stabilize_straighten",      // 25  pre-stand slump recovery (legs-only port of lean strat 25)
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
    // Wholesale copy of the canonical residual_'s PlanSnapshot (stage 4a):
    // every rollout-visible field -- keyframe/ramp state, catch episode,
    // straighten seed + foot anchor, the governed command, the drive FSM
    // outputs -- propagates in ONE struct assignment. Fields added to the
    // snapshot propagate automatically (the old per-field list is the code
    // shape that produced the walk-ceiling forgot-to-copy bug).
    return std::make_unique<ResidualFn>(
        this, static_cast<const PlanSnapshot &>(residual_));
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
  // STRAIGHTEN funnel deploy-hold latch (log-once edges; see TransitionLocked)
  bool straighten_funnel_pinned_ = false;
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
