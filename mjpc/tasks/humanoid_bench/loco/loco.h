#ifndef MJPC_TASKS_HUMANOID_BENCH_LOCO_LOCO_H_
#define MJPC_TASKS_HUMANOID_BENCH_LOCO_LOCO_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include "mjpc/tasks/humanoid/interact/contact_keyframe.h"
#include "mjpc/tasks/humanoid/interact/motion_strategy.h"
#include "mujoco/mujoco.h"

namespace mjpc {

constexpr int kLocoStrategyParameterIndex = 1;

// Manual-phase override. -1 (default) = auto-advance through keyframes
// based on success_sustain_time + target_distance_tolerance. 0..N-1 =
// hold at that keyframe index regardless of progress. Lets the user
// scrub through the loaded strategy's phases without reloading (which
// would reset to keyframe 0 and snap the body back through stand_up).
constexpr int kLocoPhaseParameterIndex = 2;

// LIVE cmd_vel teleop (WASD/gamepad/Nav2, 2026-07-03). gRPC-settable via
// SetTaskParameters like the Reach seam above. Vx/Vy are BODY-frame m/s;
// the TransitionLocked governor clamps/slews/rotates them before the trot
// sees anything. Seq is a client heartbeat counter: unchanged > 1 s means
// the client died -> watchdog zeroes the command.
// Loco (2026-10-14): the Reach Active/X/Y/Z params that used to sit at 3..6 in
// the Lean/Beginning task are gone (no reach target in this task), so the Cmd
// block follows Phase directly. The bridge sets these by NAME, so the shift is
// invisible to wss_cmd_bridge.py / wasd_teleop.py.
constexpr int kLocoCmdActiveParameterIndex = 3;
constexpr int kLocoCmdVxParameterIndex = 4;
constexpr int kLocoCmdVyParameterIndex = 5;
constexpr int kLocoCmdSeqParameterIndex = 6;
constexpr int kLocoCmdWzParameterIndex = 7;   // V3 yaw-rate (drive strat only)

constexpr char kLocoStrategyFilePath[] =
    SOURCE_DIR "/mjpc/tasks/humanoid_bench/loco/strategies/";

class loco : public Task {
 public:
  std::string Name() const override = 0;

  std::string XmlPath() const override = 0;

  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const loco *task,
                        const mjpc::humanoid::ContactKeyframe& kf =
                            mjpc::humanoid::ContactKeyframe(),
                        mjtNum keyframe_start_time = 0.0,
                        mjtNum prev_posture_scale = 1.0,
                        int prev_posture_key_id = 0,
                        int num_phases = 1,
                        const mjtNum* straighten_start_qpos = nullptr,
                        double straighten_start_tilt = 0.0,
                        bool straighten_seeded = false)
        : mjpc::BaseResidualFn(task),
          residual_keyframe_(kf),
          keyframe_start_time_(keyframe_start_time),
          prev_phase_posture_scale_(prev_posture_scale),
          prev_posture_key_id_(prev_posture_key_id),
          num_phases_(num_phases),
          straighten_start_tilt_(straighten_start_tilt),
          straighten_seeded_(straighten_seeded) {
      for (int i = 0; i < 64; ++i) {
        straighten_start_qpos_[i] =
            straighten_start_qpos ? straighten_start_qpos[i] : 0.0;
      }
    }

    void Residual(const mjModel *model, const mjData *data,
                  double *residual) const override;

    // Phase-transition ramp duration: the posture scale, the per-phase cost
    // weights and (for multi-phase strategies) the posture target smoothly
    // interpolate from their previous-phase values to the new-phase values
    // over this many seconds after each keyframe advance / live switch.
    static constexpr mjtNum kPhaseRampSeconds = 1.5;

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
    // ----- Live teleop command (cmd_vel seam, 2026-07-03) ------------------
    // Written ONLY by the TransitionLocked governor; read by Residual()
    // (via the per-plan-iteration ResidualFn snapshot copy) and by
    // loco::ModifyControl (live, friend access -- same benign unlocked
    // double-read class as keyframe_start_time_). cmd_active_=false =>
    // both readers take the legacy numeric path, byte-identical.
    bool   cmd_active_ = false;
    double cmd_vdes_world_[2] = {0.0, 0.0};  // governed v_des, WORLD frame
    // governor state (BODY-frame slewed command + bookkeeping)
    double cmd_filt_[2] = {0.0, 0.0};
    bool   cmd_starved_ = false;   // log-once latch for the heartbeat watchdog
    double cmd_last_seq_ = -1.0;
    double cmd_seq_time_ = -1.0;
    double cmd_prev_time_ = -1.0;
    double cmd_settle_until_ = -1.0;
    // ----- WSS drive FSM (strat 24 stand<->trot teleop, 2026-07-07) --------
    // drive_gait_amp_ (0..1): gait-enable multiplier for the drive strategy.
    // 0 => feet planted (a real stand; strat-20 balance-gating still catches a
    // push), 1 => full get_rz trot. Written by the TransitionLocked latch on
    // residual_ and COPIED into the per-plan snapshot in ResidualLocked (the
    // ctor doesn't take it) so the rollout COST sees the same amplitude that
    // loco::ModifyControl drives open-loop -> cost/swing agree. The latch
    // bookkeeping (drive_walk_/idle_since_/ramp_prev_) lives on residual_ only.
    double drive_gait_amp_ = 0.0;
    bool   drive_walk_ = false;
    double drive_idle_since_ = -1.0;
    double drive_ramp_prev_ = -1.0;
    // V3 yaw (2026-07-07): cmd_wz_ = governed yaw-rate [rad/s] (TransitionLocked
    // only); drive_yaw_des_ = integrated desired WORLD heading [rad], copied to
    // the snapshot so Residual can point reach_dir (Body Yaw target) at it.
    double cmd_wz_ = 0.0;
    double drive_yaw_des_ = 0.0;

    // Foot Stability anchor for the STEPPING keyframes (trot/walk/drive), 2026-07-20.
    // Rx,Ry,Lx,Ly. The residual's kRight/kLeftFootHomeXY constants live in the
    // ODOMETRIC frame, which drifts (~13 m measured on real) -- so the cost ends up
    // dragging the feet toward a point metres away and the robot "keeps trying to
    // lean backward". Same disease the lower-body stand fixed 2026-07-18 (cost
    // 225 -> 0.00 on real). Re-pinned to the MEASURED stance by TransitionLocked at
    // stepping-keyframe entry, and again whenever the drive FSM is idle (standing).
    // NOT re-pinned mid-gait: while stepping the feet are SUPPOSED to travel, so a
    // per-tick re-pin would zero out an 8-weight cost the walk was tuned with.
    // Defaults == the old constants => inert until TransitionLocked pins it.
    double drive_foot_anchor_[4] = {0.2196, -0.163, 0.2196, 0.163};

    // ===== R7 CONTACT-TRIGGERED GAIT + FROZEN PLACEMENT (2026-09-03) ======
    // WHY. The gait clock is open-loop -- ph = fmod(t*cadence,1) -- so touchdown
    // carries an uncorrected phase error of up to +/-T_swing (0.36 s at cadence
    // 1.1 / duty 0.60) on EVERY step. Early touchdown => do_leg keeps position-
    // driving a LOADED leg toward a placement target; late touchdown => the bell
    // returns to 0 with the foot still airborne and the script hands a FLOATING
    // leg to the planner. Both read as a hitch at the step boundary. The
    // reference DCM controller (The5439Workshop capture-point walker, and every
    // DCM/ZMP walker since Englsberger) switches stance on CONTACT with a
    // debounce, so its phase error is structurally zero.
    //
    // ARCHITECTURE. loco::ModifyControl is const and gets only
    // (model, qpos, qvel, time, ctrl) -- NO mjData, so it cannot see contacts.
    // Therefore every quantity below is LATCHED in TransitionLocked (which has
    // mjData, runs once per plan on the REAL state), copied into the per-plan
    // snapshot by ResidualLocked, and read by BOTH ResidualFn::Residual and
    // loco::ModifyControl -- exactly the drive_gait_amp_ pattern above.
    // Rollouts inherit the latch and march the nominal clock from it, which is
    // what makes cost and open-loop swing agree over the horizon.
    //
    // ALL OF IT IS NUMERIC-GATED AND DEFAULT-OFF (trot_contact_switch,
    // trot_freeze_place, trot_ground_seek, trot_cop_anchor, trot_dcm_target all
    // default 0) => byte-identical to the pre-R7 gait until switched on.

    // Phase-locked-loop correction [cycles] added to the raw clock. Updated on
    // each debounced touchdown by the CLAMPED phase error, so the clock
    // converges onto the real contact rhythm in 1-2 steps instead of snapping
    // (a hard snap would discontinuously jump the CONTRALATERAL leg, which is
    // antiphase on the same clock, straight through its swing).
    double gait_phase_offset_ = 0.0;
    // Debounce + swing bookkeeping, per leg [0]=L [1]=R.
    double gait_td_time_[2] = {-1.0, -1.0};   // last accepted touchdown time [s]
    bool   gait_in_swing_[2] = {false, false};// leg is inside its scripted swing
    double gait_swing_t0_[2] = {-1.0, -1.0};  // swing-start time [s]
    // Ground-seek overrun [s]: how long past its nominal swing end a leg has
    // been airborne without an accepted touchdown. The reference walker's
    // sin_adapt() keeps driving the foot DOWN when tau>1; our SwingBell returns
    // to exactly 0 and stops, so a foot over ground 1 cm lower than expected
    // just hangs there. This is that missing search-for-ground.
    double gait_overrun_[2] = {0.0, 0.0};
    // ★ R7b LIFTOFF INTERLOCK (2026-09-05). A touchdown only counts if the foot
    // ACTUALLY LEFT THE GROUND first. Without this the swing window is decided by
    // the CLOCK alone, so a strategy that is standing still -- drive while idle
    // (drive_gait_amp_ = 0, feet planted) or the balance-gated stumble march --
    // still toggles gait_in_swing_ every cycle while BOTH feet stay loaded, and
    // the touchdown test fires immediately on phantom contact, winding the phase
    // PLL with garbage. Drive spends most of its life idle, so the offset would be
    // junk the instant a walk command arrives. Mirrors the reference walker's
    // `contact_lifted` flag: set when the foot unloads, and required before the
    // next touchdown is accepted. Self-gating -- a planted foot never unloads, so
    // an idle stand contributes exactly zero PLL updates.
    bool   gait_lifted_[2] = {false, false};
    // Frozen per-step placement [Lx,Ly,Rx,Ry]. Recomputing step_x from live
    // qvel every tick means the swing foot chases a MOVING setpoint, and a
    // smoothstep ramp toward a moving setpoint is not monotone -- the foot
    // reverses mid-swing whenever v fluctuates (which it does, hard, during the
    // weight shift). Latched once at swing start, then merely tracked.
    double step_frozen_[4] = {0.0, 0.0, 0.0, 0.0};
    bool   step_frozen_ok_[2] = {false, false};
    // Filtered CoM acceleration [m/s^2], world xy -- the CoP anchor. The DCM law
    // p = xi - xi_dot/omega - K_xi(xi - xi_des) reduces, at K_xi = -1 and with
    // the LIPM relation xdd = omega^2 (x - p_cop), to
    //        p = (x - a/omega^2) + tau*(v - v_des)
    // i.e. step to the MEASURED CoP plus the velocity-error capture offset. We
    // already have the second term exactly; the first is currently a hard-coded
    // constant (stance_off_x = 0.13), which is a gait-phase-locked placement
    // error of a/omega^2 (~cm at the accelerations the weight shift produces).
    // Differentiated from the subtree CoM velocity sensor and low-passed --
    // the raw derivative is far too noisy to place a foot with.
    double com_acc_filt_[2] = {0.0, 0.0};
    double com_vel_prev_[2] = {0.0, 0.0};
    double com_acc_prev_time_ = -1.0;
    // ---- ★ CAPTURE-POINT DIVERGENCE RATE (catch_rate_lead input) ---------
    // MEASURED 2026-09-07: lowering catch_trig/catch_full barely lengthens the
    // window in which the catch-step is at full amplitude -- 0.31 s at
    // 0.12/0.24 vs 0.50 s at 0.07/0.11. That is not a tuning failure, it is the
    // inverted pendulum: the capture point diverges with time constant
    // tau = sqrt(z/g) ~ 0.30 s, so halving the threshold buys only
    // tau*ln(2) ~ 0.2 s. A humanoid swing needs 0.3-0.5 s, and the trigger
    // cannot fire below the gait's own wobble (quiet-stand danger peaks 0.057).
    // Level alone therefore CANNOT buy step time. The rate can: a fatal
    // disturbance ramps `danger` monotonically for ~1 s before it crosses any
    // level, while an absorbed one peaks ~0.07 and turns over.
    // Low-passed here, once per real control tick (rollouts see it frozen),
    // exactly as com_acc_filt_ is -- a raw per-rollout derivative would be
    // noise.
    double danger_prev_ = 0.0;
    double danger_rate_ = 0.0;
    double danger_prev_time_ = -1.0;
    // ★ CATCH PHASE RESET (stumble_phase_reset). catch_armed_ is the rising-edge
    // latch on danger > catch_trig; catch_reset_time_ debounces it so a danger
    // signal hovering at the threshold cannot re-slam the gait clock every tick.
    bool catch_armed_ = false;
    double catch_reset_time_ = -1.0;
    // ★ SUSTAINED-BIAS ARM (catch_bias_trig / catch_bias_dwell). bias_since_ is
    // the time `danger` first rose above the LOW bias threshold and stayed there;
    // catch_bias_on_ latches once it has held for the dwell. Computed once per
    // real tick in TransitionLocked and copied into every rollout snapshot, like
    // danger_rate_ -- a rollout must not re-derive a latch from its own clock.
    bool catch_bias_on_ = false;
    double bias_since_ = -1.0;
    // Time the catch gate last went from closed to open. Latched unconditionally
    // (not only when the phase reset is enabled) because the UNWEIGHT pulse
    // needs the onset instant too.
    double catch_onset_time_ = -1.0;
    // ★ STANCE SEPARATION (stumble_sep_hold). Lateral ankle-to-ankle distance in
    // the BASE-YAW frame, latched once per real tick in TransitionLocked.
    // ModifyControl has no mjData and cannot read foot sensors, so the measured
    // separation has to be carried across like com_acc_filt_ / step_frozen_.
    // -1 = not yet measured.
    double stance_sep_y_ = -1.0;
    // ★ COMMITTED STEP PAIR (stumble_step_pair). A catch is armed for a measured
    // 0.36-0.9 s while ONE gait cycle is 1/cadence = 0.91 s, so only one foot
    // ever gets a swing window and the stance ends up translated by one leg
    // instead of both. These hold the catch alive for a whole cycle once a step
    // has actually been committed, so the trailing foot follows and the pair
    // ends up back at its nominal separation.
    double catch_hold_until_ = -1.0;
    double catch_hold_recov_ = 0.0;
    // Per-step exponential DCM target [m], world xy: the closed-form LIPM
    // solution x_target = p0 + (xi0 - p0) e^{omega T} with
    // p0 = (xi_des - xi0 e^{omega T})/(1 - e^{omega T}). Latched at swing start.
    // This is the one term that says where the CoM will BE at the end of the
    // step; without it the placement script is memoryless. e^{omega T} ~ 3.2 for
    // us (omega 3.21, T_swing 0.364), so an unmanaged placement error is
    // amplified 3.2x per step -- the quantitative case for freezing a target.
    double dcm_target_[2] = {0.0, 0.0};
    bool   dcm_target_ok_ = false;
    // ===== COMMIT-THEN-TRACK stance CoM reference (2026-10-14, `trot_com_commit`) =====
    // WHY. The swing foot is forced open-loop (ModifyControl) but the stance-side
    // weight shift was only the soft "Lateral Center" rock the sampler had to
    // re-discover every plan; at the honest 50 plans/s it never does ("feet cycle,
    // never unload", 10-14 twin: ankle pinned 46-54 Nm, 0/8 forward bursts). The
    // capture-point walker commits a stance plan per step and TRACKS it. Latched in
    // TransitionLocked at the swing-start edge (R7 latch), copied to the snapshot,
    // read by Residual: lateral target = stance-foot y (+ inward offset) for the
    // whole step instead of midfoot + bump. Default numeric 0 => byte-identical.
    double com_ref_p_[2] = {0.0, 0.0};   // stance foot xy at the edge
    double com_ref_xi_[2] = {0.0, 0.0};  // DCM xy at the edge (for a later x_ref)
    double com_ref_t0_ = -1.0;           // edge time
    double com_ref_omega_ = 3.13;        // sqrt(g/z) at the edge
    int    com_ref_stance_ = -1;         // 0 = LEFT foot is stance, 1 = RIGHT is stance
    bool   com_ref_ok_ = false;

    // Posture scale starts at 1.0 (no boost) and ramps to 3.0 during stand_up.
    mjtNum prev_phase_posture_scale_ = 1.0;

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
    // per-strategy gate the 2026-06-08 revert note (loco.cc) said the ramp needed.
    int num_phases_ = 1;

    // ----- STRAIGHTEN (strategy 25) live-seed min-jerk ramp state (C3) --------
    // The pre-stand bring-up strategy must drive the body to upright+centered
    // from ANY near-standing release config. A STATIC strong upright/posture cost
    // slams the correction and OVERSHOOTS (twin: a +10deg lean is flung backward
    // past vertical to ~130deg). The fix (plan §0/§6-C3, "the ramped reference IS
    // the funnel"): capture the release pose ONCE at phase entry and ramp the
    // upright + posture TARGETS from it to the nominal along a min-jerk (smoothstep)
    // over target_ramp_sec, so the instantaneous error — and thus the corrective
    // force — stays small the whole way up. Captured on the TRUE agent state in
    // TransitionLocked (never per-rollout), then propagated to every rollout
    // residual via the ResidualLocked ctor (parallel to prev_posture_key_id_).
    //   straighten_start_qpos_  : full qpos at straighten entry (posture ramp FROM)
    //   straighten_start_tilt_  : pelvis tilt angle [rad] at entry (upright ramp FROM)
    //   straighten_seeded_      : true once captured (else fall back to static target)
    mjtNum straighten_start_qpos_[64] = {0};
    double straighten_start_tilt_ = 0.0;
    bool straighten_seeded_ = false;

   private:
    friend class loco;
  };

  loco() : residual_(this), current_strategy_(-1) {}

  void TransitionLocked(mjModel *model, mjData *data) override;

  void ResetLocked(const mjModel *model) override;

  // Populate monitoring metrics (CoM, CoP, ICP, saturation, feet) for the
  // Research GUI / headless analyzer. Reads the
  // current keyframe + sensor stack; safe to call from the gRPC poll loop.
  // See task.h ComputeMetrics for contract.
  void ComputeMetrics(const mjModel *model, const mjData *data,
                      std::map<std::string, double> *metrics,
                      std::string *phase_name) const override;

  // Per-strategy planner model-numeric overrides (e.g. sampling_spline_points).
  // See task.h PlannerNumericOverrides for the contract. Keyed by strategy NAME
  // (GetStrategyNames()[strategy]).
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

  // Loco task (2026-10-14): locomotion-only fork of the Beginning snapshot.
  // SLOT NUMBERS ARE PRESERVED from Beginning/Lean so every bench script,
  // handoff and memory entry ("strat 20", "strat 24") keeps meaning. Retired
  // slots are padded with h12_simple_stand (same idiom Lean uses).
  //   6  stand        -- balance hold at stand_up (deploy default)
  //  20  stumble      -- gait-clock stepping / push recovery (stumble_march)
  //  23  trot         -- in-place trot, leg-lift test vehicle (stumble_trot)
  //  24  drive        -- WSS teleop: idle stand <-> trot FSM on live cmd_vel
  //                      (stumble_trot_drive; residual_Cmd Active/Vx/Vy/Wz/Seq)
  //  25  straighten   -- pre-stand bring-up, wide-basin drive-to-upright
  virtual std::vector<std::string> GetStrategyNames() const {
    const std::string kPad = "h12_simple_stand";
    std::vector<std::string> names(26, kPad);
    names[6]  = "h12_simple_stand";
    names[20] = "h12_simple_stumble";
    names[23] = "h12_simple_trot";
    names[24] = "h12_simple_drive";
    names[25] = "h12_straighten";
    return names;
  }

  // Live per-phase weight blending --------------------------------------- //
  // Per-phase keyframes in the strategy JSON carry a `weight: { name: val }`
  // map. On phase advance we snapshot the live cost weights, compute the new
  // phase's targets (JSON override OR XML default for missing keys), and ramp
  // weight[] from snapshot → target over kPhaseRampSeconds using the same
  // smoothstep curve the residual uses for the posture scale.
  // This lets the user isolate behaviours from the strategy file alone:
  // setting "Gait": 0 in a phase silences that cost without recompiling.
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
        residual_.prev_phase_posture_scale_,
        residual_.prev_posture_key_id_,
        residual_.num_phases_,
        residual_.straighten_start_qpos_,
        residual_.straighten_start_tilt_,
        residual_.straighten_seeded_);
    // WSS drive: propagate the FSM gait-enable into this plan's snapshot so the
    // rollout cost gates g_amp the same way ModifyControl does (cost/swing
    // agreement). Every non-drive strategy leaves this 0 (default path).
    rfn->drive_gait_amp_ = residual_.drive_gait_amp_;
    rfn->drive_yaw_des_ = residual_.drive_yaw_des_;   // V3 yaw heading target
    for (int i = 0; i < 4; i++)                       // measured Foot Stability anchor
      rfn->drive_foot_anchor_[i] = residual_.drive_foot_anchor_[i];
    // ★ BUGFIX 2026-07-12: propagate the GOVERNED COMMAND too. Without this every
    // rollout residual sees cmd_active_=false and therefore takes the legacy
    // trot_des_vel numeric path (v_des = 0) -- i.e. it costs the gait as an
    // IN-PLACE trot -- while loco::ModifyControl (which reads the CANONICAL
    // residual_) drives the swing FORWARD at the governed v_des. Cost and swing
    // then disagree on every sampled trajectory and the sampler spends the whole
    // plan cancelling the walk drive. That is the exact opposite of what the
    // "MUST match loco::ModifyControl" comments in Residual() require, and it is
    // the prime suspect for the free-twin ~6 s walk ceiling.
    rfn->cmd_active_ = residual_.cmd_active_;
    rfn->cmd_vdes_world_[0] = residual_.cmd_vdes_world_[0];
    rfn->cmd_vdes_world_[1] = residual_.cmd_vdes_world_[1];
    // ★ R7 (2026-09-03): propagate the contact-latched gait state. Same reason
    // as the cmd_active_ bugfix directly above -- without this the rollout cost
    // would run the RAW open-loop clock and the un-frozen placement while
    // loco::ModifyControl (reading the canonical residual_) runs the
    // contact-corrected clock and the frozen target, so cost and swing would
    // disagree on every sampled trajectory. All of it is inert while the R7
    // numerics are 0 (offset stays 0, frozen_ok stays false).
    rfn->gait_phase_offset_ = residual_.gait_phase_offset_;
    for (int i = 0; i < 2; i++) {
      rfn->gait_td_time_[i]   = residual_.gait_td_time_[i];
      rfn->gait_in_swing_[i]  = residual_.gait_in_swing_[i];
      rfn->gait_swing_t0_[i]  = residual_.gait_swing_t0_[i];
      rfn->gait_overrun_[i]   = residual_.gait_overrun_[i];
      rfn->gait_lifted_[i]    = residual_.gait_lifted_[i];   // R7b interlock
      rfn->step_frozen_ok_[i] = residual_.step_frozen_ok_[i];
      rfn->com_acc_filt_[i]   = residual_.com_acc_filt_[i];
      if (i == 0) {
        rfn->danger_prev_ = residual_.danger_prev_;
        rfn->danger_rate_ = residual_.danger_rate_;
        rfn->danger_prev_time_ = residual_.danger_prev_time_;
        rfn->catch_armed_ = residual_.catch_armed_;
        rfn->catch_reset_time_ = residual_.catch_reset_time_;
        rfn->catch_bias_on_ = residual_.catch_bias_on_;
        rfn->bias_since_ = residual_.bias_since_;
        rfn->catch_onset_time_ = residual_.catch_onset_time_;
        rfn->stance_sep_y_ = residual_.stance_sep_y_;
        rfn->catch_hold_until_ = residual_.catch_hold_until_;
        rfn->catch_hold_recov_ = residual_.catch_hold_recov_;
      }
      rfn->dcm_target_[i]     = residual_.dcm_target_[i];
    }
    for (int i = 0; i < 4; i++) rfn->step_frozen_[i] = residual_.step_frozen_[i];
    rfn->dcm_target_ok_ = residual_.dcm_target_ok_;
    for (int i = 0; i < 2; i++) {
      rfn->com_ref_p_[i]  = residual_.com_ref_p_[i];
      rfn->com_ref_xi_[i] = residual_.com_ref_xi_[i];
    }
    rfn->com_ref_t0_ = residual_.com_ref_t0_;
    rfn->com_ref_omega_ = residual_.com_ref_omega_;
    rfn->com_ref_stance_ = residual_.com_ref_stance_;
    rfn->com_ref_ok_ = residual_.com_ref_ok_;
    return rfn;
  }

  ResidualFn *InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
  mjpc::humanoid::MotionStrategy motion_strategy_;
  int current_strategy_;

  // Weight-ramp state (parallel to ResidualFn::prev_phase_posture_scale_):
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


// Whole-body (nu=27, nq=34) H1-2 with the two fixed magpie grippers as
// ~0.506 kg mass-only bodies at the wrists. Load with --task "Loco H12 Magpie".
// The grippers are NOT actuated here -- open/close is owned by the separate
// magpie_msgs controller. No table, no object, no mocap target: the scene is
// the robot on a flat floor.
class Loco_H12_Magpie : public loco {
 public:
  std::string Name() const override { return "Loco H12 Magpie"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/loco/Loco_H12_Magpie.xml");
  }
};


}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_BENCH_LOCO_LOCO_H_
