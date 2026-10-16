#include "mjpc/tasks/humanoid_bench/loco/loco.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

#include "mujoco/mujoco.h"
#include "mjpc/tasks/humanoid/interact/contact_keyframe.h"

namespace mjpc {

// T1 REFERENCE TRIM v2 -- ported from stabilize.cc (commit 1708253) 2026-07-20.
// Written by TransitionLocked (real state, once per plan) and read by the
// rollout workers in Residual (benign double read; changes on a seconds
// timescale, same pattern as stabilize's s_cap_ex_dc).
//
// WHY THE STAND NEEDS IT (2026-07-20 real): with the IMU offset finally
// measured honest (pitch 0.0 / roll 1.3) the stand held 83 s with base_z
// sd 0.005 -- but still walked a slow FORWARD ramp (lean +0.2 deg at t=4 ->
// +6.8 deg at t=72) that loaded LankP to its 44.8 Nm useful cap (49.3% of
// ticks budget-clamped, 29.3 deg tracking error) until the left leg gave up.
// That residual bias is ankle-zero + CoM-model error: it is NOT constant
// (both 07-20 recordings show the solved ankle zeros drifting 4-8 deg WITHIN
// one 50 s run), so no XML constant can cancel it. A leaky integral does not
// need to know the bias -- it nulls whatever steady park it observes.
//
// Recipe (Stephens IROS'07 integral CoP/posture; Caron ICRA'19 leaky DCM
// integral) with the four v1 defects already fixed upstream: leak, support
// frame, nominal 0, quiet gate. See stand_trim_* in Loco_H12_Magpie.xml.
// stand_trim_tau = 0 (the shipped default) forces both to 0 => byte-identical.
static double s_trim_x = 0.0, s_trim_y = 0.0;

// Phase-ramp duration resolver (defined with the weight-ramp helpers below):
// `phase_ramp_sec` numeric when > 0, else the compiled fallback.
static double LeanPhaseRampSeconds(const mjModel *model, double fallback);

namespace {
// Swing clearance bell 0->1->0 over swing progress s in [0,1]. Cubic-Bezier
// (MuJoCo Playground gait.get_rz shape): smoothstep b(t)=3t^2-2t^3 of a 0->1->0
// triangle. dcl/ds = 0 at s=0, 0.5, 1 => zero swing-JOINT velocity at liftoff
// AND touchdown (soft, quiet landing). Replaces sin(pi*s), whose endpoint slope
// is +-pi => max foot velocity exactly at touchdown = the sport-mode slap.
// Peak and area match sin closely (apex 1.0 at s=0.5) so step CLEARANCE is
// preserved; only the endpoint velocity changes.
inline double SwingBell(double s) {
  s = s < 0.0 ? 0.0 : (s > 1.0 ? 1.0 : s);
  double t = (s <= 0.5) ? 2.0 * s : 2.0 - 2.0 * s;   // triangle ramp 0->1->0
  return t * t * (3.0 - 2.0 * t);                     // smoothstep of the ramp
}

// ============================================================================
// ★ R7 ONE GAIT CLOCK, TWO READERS (2026-09-03).
// The clock was duplicated verbatim in ResidualFn::Residual and in
// loco::ModifyControl, with a "MUST match" comment holding them together by
// hand. R7 adds a contact-derived phase OFFSET to it, which is exactly the kind
// of change that silently desynchronises two hand-copied blocks -- so the clock
// is hoisted here and both sites call it. Keeping the reads (stumble_cadence,
// trot_duty) inside the helper means neither caller can look up a different
// knob than the other either.
//
//   ph_l = frac((t - offset/cad) * cad)   ph_r = ph_l + 0.5 (antiphase)
//
// `offset` is in CYCLES and comes from the contact PLL (see
// ResidualFn::gait_phase_offset_). offset == 0 reproduces the pre-R7 clock
// bit-for-bit, which is what keeps trot_contact_switch=0 byte-identical.
// ============================================================================
struct GaitClock {
  double ph_l, ph_r;   // per-leg phase in [0,1); swing iff ph >= duty
  double duty, cad;    // resolved duty ratio and cadence [Hz]
};

inline GaitClock GaitPhases(const mjModel *model, double time, double offset,
                            bool is_trot) {
  constexpr double kCadenceHz = 1.1, kDutyRatio = 0.60;
  GaitClock g;
  int cad_id = mj_name2id(model, mjOBJ_NUMERIC, "stumble_cadence");
  g.cad = (cad_id >= 0) ? model->numeric_data[model->numeric_adr[cad_id]]
                        : kCadenceHz;
  // duty: TROT may raise it (trot_duty, is_trot-gated) for more double support
  // during slow forward walk; strat 20 keeps kDutyRatio. Both readers resolve it
  // identically here so cost and swing can never disagree about the duty.
  // A value <= 0 means "unset" and falls back to kDutyRatio, so declaring the
  // numeric with the file's usual data="0" default is byte-identical to not
  // declaring it at all. Without this guard a 0 would mean "always swinging"
  // and would destroy the gait silently. Clamped to 0.95 because duty >= 1.0
  // divides by zero in the (1 - kDuty) swing-progress denominators below.
  int dty_id = mj_name2id(model, mjOBJ_NUMERIC, "trot_duty");
  double duty_raw = (is_trot && dty_id >= 0)
                        ? model->numeric_data[model->numeric_adr[dty_id]] : 0.0;
  g.duty = (duty_raw > 1e-6) ? mju_min(duty_raw, 0.95) : kDutyRatio;
  double ph = std::fmod(time * g.cad + offset, 1.0);
  if (ph < 0.0) ph += 1.0;                     // fmod is sign-preserving
  g.ph_l = ph;
  g.ph_r = std::fmod(ph + 0.5, 1.0);
  return g;
}

// Ground-seek extension [rad] for a swing leg that is past its nominal swing end
// with no accepted touchdown. The reference capture-point walker's sin_adapt()
// adds z_offset = -(tau-1) for tau>1, i.e. it keeps driving the foot DOWN until
// it finds the floor; SwingBell instead returns to exactly 0 and holds, so a
// foot over ground even a centimetre lower than expected hangs in the air and
// the planner has to catch the stall. Linear in overrun time, saturating, so a
// genuinely missing floor does not command an unbounded lunge.
inline double GroundSeek(double overrun_s, double rate, double max_rad) {
  if (overrun_s <= 0.0 || rate <= 0.0) return 0.0;
  double e = overrun_s * rate;
  return (e > max_rad) ? max_rad : e;
}

// ============================================================================
// STANCE-SIDE LATERAL WEIGHT-SHIFT ("ROCK") — trot_rock_ff, default 0 = OFF.
//
// WHY (2026-07-20, async-twin diagnosis of the real drive-24 Q/E estop):
// this gait already forces the SWING leg open-loop in loco::ModifyControl, so
// the lift is RATE-INDEPENDENT. But the other half of a step -- shifting the
// weight OFF the foot that is about to swing -- is only a COST TARGET (the
// "Lateral Center" rock at the subcom_y residual, whose own comment says "the
// planner finds the hip-roll / ankle to achieve it"). That half is SAMPLER-
// owned, and it is exactly what starves when the plan rate drops: the node's
// own PLAN-RATE WARNING predicts it verbatim -- "the stance-leg weight shift
// is SAMPLER-owned and starves first (feet cycle, never unload)".
// MEASURED on the async twin (strat 24, yaw 0.3, est real):
//     28 Hz (real rate): foot unload 2-7%,  z_min 0.75, knee 1.0  -> FELL 0/3
//     200 Hz (lockstep): foot unload 25.8%, z_min 0.99, knee 0.92 -> HELD 3/3
// i.e. at the real rate it cannot commit to a step -- it shuffles and SAGS,
// the same knee~0.95 / z~0.957 crouch the real robot froze in on 2026-07-13.
// So: give the weight shift the same open-loop treatment the swing already has.
//
// WHY THE HIP (not the ankle): the ankle is the documented saturating joint
// (useful torque capped ~44.8 Nm) -- adding lateral demand there is exactly
// the wrong direction. The hip has the authority (FK 0.79 m/rad). And a twin
// A/B already showed that merely PERMITTING the hip strategy (Hip Roll 30->0)
// does NOT make the sampler discover it -- so FORCING it, the way the swing is
// forced, is the untried move.
//
// SIGNS: right foot sits at -y, left at +y. +hip_roll moves that leg's foot
// +y (0.79 m/rad, both legs) -- so on a PLANTED foot the same +hip_roll moves
// the PELVIS -y. To unload the LEFT (swinging) foot the body must go -y, onto
// the right stance foot => +delta on the RIGHT hip_roll. Mirrored for the
// other half-cycle. In double support both bumps are 0 => both deltas are 0,
// so a quiet stand is untouched even with the numeric ON.
//
// ★ ONE FORMULA, TWO READERS. loco::ModifyControl adds these deltas to the
// stance hip_roll, and ResidualFn::Residual shifts the "Hip Roll L/R" cost
// TARGET by the SAME deltas. That cost pins hip_roll to the keyframe at
// weight 30, so driving the joint without moving its target would be a
// cost-vs-feedforward fight -- the same class of bug as the walk->stop
// faceplant (cost kept the catch-step, ModifyControl quit). Both callers go
// through here so they cannot drift.
//
// rock_m = commanded pelvis lateral shift at full bump [m]. 0 => both 0 =>
// byte-identical for every strategy and every existing run.
// ============================================================================
// ★ PHASE LEAD (trot_rock_lead, cycle fractions, default 0). You cannot unload a
// foot AFTER it is supposed to have left the ground. With duty 0.60 the swing
// bump is 0 at liftoff (ph=0.60) and peaks mid-swing (ph=0.80), so a rock driven
// by the RAW bump is exactly zero when the unload is needed and maximal when the
// foot is already airborne -- measured 2026-07-20 (lead 0: unload 3.8%->9.8%,
// z_min unmoved at 0.75 = the sag survived). The COST can absorb that because it
// optimizes over a 1 s horizon and can anticipate; an instantaneous FEEDFORWARD
// cannot. lead shifts the bump EARLIER so the weight is already off the foot at
// liftoff: lead = (1-duty)/2 = 0.20 puts the peak exactly at liftoff; 0.20-0.30
// is the sane band. Both readers get their bump from HERE, so cost and
// feedforward cannot disagree about the timing either.
inline void TrotRockHipRoll(double rock_m, double lead, double g_amp,
                            double ph_l, double ph_r, double duty,
                            double *droll_l, double *droll_r) {
  *droll_l = 0.0;
  *droll_r = 0.0;
  if (rock_m <= 1e-6 || ph_l < 0.0 || duty >= 1.0) return;
  constexpr double kFootPerHipRoll = 0.79;   // m of foot-y per rad (FK-measured)
  constexpr double kMaxRoll = 0.25;          // rad, same clamp as the swing dHipR
  auto led_bump = [&](double ph) {
    double p = std::fmod(ph + lead, 1.0);
    if (p < 0.0) p += 1.0;
    return (p < duty) ? 0.0 : SwingBell((p - duty) / (1.0 - duty));
  };
  const double gb_l = led_bump(ph_l), gb_r = led_bump(ph_r);
  const double k = rock_m / kFootPerHipRoll;
  // LEFT about to swing (gb_l > 0) -> RIGHT is stance -> pelvis -y -> +R.
  // RIGHT about to swing (gb_r > 0) -> LEFT is stance -> pelvis +y -> -L.
  double dr = +k * g_amp * gb_l;
  double dl = -k * g_amp * gb_r;
  *droll_r = dr < -kMaxRoll ? -kMaxRoll : (dr > kMaxRoll ? kMaxRoll : dr);
  *droll_l = dl < -kMaxRoll ? -kMaxRoll : (dl > kMaxRoll ? kMaxRoll : dl);
}

// Target (post-ramp) POSTURE scale for each named phase. Posture is boosted
// during stand_up because the audit-spec PD gains (ankle kp=20, knee kp=200)
// aren't stiff enough on their own to pull a drifted knee back to extension --
// the Posture cost has to do it via MPC. straighten (strat 25 phase 0) and the
// stepping keyframes (stumble_*) run at x1.0: straighten's phase-0 retune
// (Posture 35 / Balance 40) was tuned against x1.0 -- do NOT boost it to x3
// without re-running the slump suite; the gait keyframes hold a soft stance
// the swing fold modulates each cycle.
inline double PhasePostureScale(const std::string& name) {
  if (name == "stand_up") return 3.0;
  return 1.0;
}
}  // namespace

// ------------------ Residuals for the Loco task ------------------
//   Term order MUST match the <user> sensors in Loco_H12_Magpie.xml:
//     Height, CoM Vel., Joint Vel., Balance, Pelvis Tilt, Foot Right Up,
//     Foot Left Up, Waist Yaw, Hip Yaw L, Hip Roll L, Hip Yaw R, Hip Roll R,
//     Posture, Velocity, Control, Foot Stability, Joint Vel. Limit, Body Yaw,
//     Knees Straight, Symmetry, Base Height, Angular Momentum, Lateral
//     Center, Gait, Step Place, Foot Slip.
//   Parameters: 0 Height Goal, 1 Strategy, 2 Phase, 3..7 Cmd Active/Vx/Vy/Seq/Wz.
// ------------------------------------------------------------------
void loco::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                double *residual) const {
  double const height_goal = parameters_[0];
  int counter = 0;

  // ---- Phase-aware POSTURE scale (smoothstep across phase / strategy switches)
  // Smooth time-ramp from the previous phase's scale to the new one: the linear
  // progression alpha = clamp(t/T, 0, 1) goes through a smoothstep
  // s(a) = a^2 (3 - 2a) before interpolating, so the cost gradient changes
  // gradually (C1) instead of as a step the planner reads as an impulse.
  double target_posture_scale = PhasePostureScale(residual_keyframe_.name);
  double time_in_phase = mju_max(0.0, data->time - keyframe_start_time_);
  double phase_ramp_seconds = kPhaseRampSeconds;
  // `phase_ramp_sec` numeric: 0 = OFF = the compiled 1.5 s. >0 overrides the
  // cost-surface + weight ramp for EVERY transition (handover-transient knob,
  // live-tunable so it can be swept without a rebuild).
  {
    int pr_id = mj_name2id(model, mjOBJ_NUMERIC, "phase_ramp_sec");
    double pr = (pr_id >= 0) ? model->numeric_data[model->numeric_adr[pr_id]] : 0.0;
    if (pr > 1e-9) phase_ramp_seconds = pr;
  }
  double alpha_lin     = mju_min(time_in_phase / phase_ramp_seconds, 1.0);
  double alpha         = alpha_lin * alpha_lin * (3.0 - 2.0 * alpha_lin);
  double phase_posture_scale =
      prev_phase_posture_scale_ + alpha * (target_posture_scale -
                                           prev_phase_posture_scale_);

  // ----- Pose-library target keyframe ------------------------------------ //
  // A standalone "simple task" strategy (stand / crouch / arms_sideways / …)
  // names its phase after a model <key> keyframe. When a keyframe with that
  // exact name exists, the Posture + Control costs track THAT pose instead of
  // the home keyframe — a selectable pose-library player. mj_name2id returns -1 when no keyframe matches, so every existing
  // pipeline phase ("stand_up", "arm_plant", …) falls back to key 0 (home)
  // and behaves exactly as before. The lookup is one hash probe per Residual
  // call — negligible next to the ~10 std::string compares PhaseTargetScales
  // already does.
  // REVERTED 2026-06-08: the squat target-pose ramp (asymmetric stand-up/crouch
  // ramp) is OFF — restored to the original instantaneous SNAP. The ramp ran in
  // this hot residual path for EVERY strategy (incl. single-phase crouch/stand/
  // arms) and was suspected of regressing the live crouch (one-knee-locked slow
  // creep) even though headless held. Snap = exactly the accepted behaviour.
  // The prev_posture_key_id_ plumbing (loco.h, SnapshotEffectiveScales) is left
  // in place but UNUSED; re-enable the ramp here only behind a per-strategy gate
  // that provably never touches single-phase strategies. See [[project_squat_strategy]].
  int posture_key_id =
      mj_name2id(model, mjOBJ_KEY, residual_keyframe_.name.c_str());
  if (posture_key_id < 0) posture_key_id = 0;  // home
  const mjtNum *posture_target = model->key_qpos + posture_key_id * model->nq;
  // one-shot forensic: the hand-off snap survived 6 cost fixes -- verify LIVE which
  // keyframe the residual actually pulls toward (assumed 'stand' knee=0.35 z=0.98)
  static int dbg_last_key = -999;
  if (posture_key_id != dbg_last_key) {
    dbg_last_key = posture_key_id;
    std::fprintf(stderr,
                 "[loco-residual] posture keyframe: name='%s' id=%d z=%.3f kneeL=%.3f kneeR=%.3f\n",
                 residual_keyframe_.name.c_str(), posture_key_id,
                 posture_target[2], posture_target[7 + 3], posture_target[7 + 9]);
  }

  // ----- GENERAL target-pose ramp (re-enabled 2026-06-16, behind the per-strategy
  //       gate the 2026-06-08 revert note above requires) --------------------- //
  // On a phase transition, ease the posture target from the keyframe we LEFT
  // (prev_posture_key_id_, captured at every transition) into the new one over a
  // smoothstep window -- the "blend tasks, don't switch them" idea the weight ramp
  // already uses, applied to the pose itself. Because it lives in the planner's OWN
  // cost, the MJPC GUI, the twin and the real robot glide identically -- unlike a
  // node-side, plant-only command blend, this opens no sim2real gap. Ramp duration
  // = the entered phase's JSON target_ramp_sec when >=0, else kPhaseRampSeconds.
  //   GATED num_phases_ > 1: single-phase strategies (stand/crouch/arms) never take
  //   this branch -> byte-identical. WINDOW-GATED: only runs while alpha<1 (the
  //   ~ramp_dur right after a transition), into a stack buffer, so steady state and
  //   every snap phase (target_ramp_sec==0) pay nothing -- the unconditional version
  //   that ran for EVERY residual call of EVERY strategy is what got reverted.
  // STRAIGHTEN (strat 25): is this the pre-stand bring-up strategy? Used here
  // (posture ramp FROM the live release pose) and below (upright-target ramp).
  const bool is_straighten =
      residual_keyframe_.name.rfind("straighten", 0) == 0;
  mjtNum ramped_posture_target[64];
  // STRAIGHTEN live-seed posture ramp (C3): glide the posture target from the
  // captured release qpos -> the centered `straighten` keyframe over
  // target_ramp_sec (min-jerk). Unlike the keyframe->keyframe ramp below, the
  // FROM pose is the robot's ACTUAL measured pose at engage, so it straightens
  // "from any launch configuration" (splayed legs / off torso) without a slam.
  if (is_straighten && straighten_seeded_ && model->nq <= 64) {
    double ramp_dur = (residual_keyframe_.target_ramp_sec >= 0.0)
                          ? residual_keyframe_.target_ramp_sec
                          : kPhaseRampSeconds;
    if (ramp_dur > 1e-9 && time_in_phase < ramp_dur) {
      double ra_lin = mju_min(time_in_phase / ramp_dur, 1.0);
      double ra = ra_lin * ra_lin * (3.0 - 2.0 * ra_lin);  // min-jerk smoothstep
      for (int i = 0; i < model->nq; i++) {
        ramped_posture_target[i] =
            straighten_start_qpos_[i] +
            ra * (posture_target[i] - straighten_start_qpos_[i]);
      }
      posture_target = ramped_posture_target;
    }
  } else if (num_phases_ > 1 &&
      prev_posture_key_id_ != posture_key_id && model->nq <= 64) {
    // Pose-target ramp for multi-phase strategies only (straighten -> stand).
    // Single-phase strats snap on a cold load -- prev_posture_key_id_ ==
    // posture_key_id there -> this branch is skipped.
    double ramp_dur = (residual_keyframe_.target_ramp_sec >= 0.0)
                          ? residual_keyframe_.target_ramp_sec
                          : kPhaseRampSeconds;
    if (ramp_dur > 1e-9 && time_in_phase < ramp_dur) {
      double ra_lin = mju_min(time_in_phase / ramp_dur, 1.0);
      double ra = ra_lin * ra_lin * (3.0 - 2.0 * ra_lin);  // smoothstep
      const mjtNum *from_target =
          model->key_qpos + prev_posture_key_id_ * model->nq;
      for (int i = 0; i < model->nq; i++) {
        ramped_posture_target[i] =
            from_target[i] + ra * (posture_target[i] - from_target[i]);
      }
      posture_target = ramped_posture_target;
    }
  }

  // ---- POSTURE LEG-CHAIN LEVELLER (posture_leg_level, 0 = OFF) ------------ //
  // ★★ 2026-07-27: THE STANDING ORDER TO LEAN FORWARD.
  // A flat-footed, upright-pelvis pose requires the leg PITCH chain to close:
  //     hip_pitch + knee + ankle_pitch == 0
  // MEASURED on the compiled model (build tree), left leg:
  //     stand / stand_up           -0.360 +0.550 -0.190 = +0.0000  ->  0.00 deg
  //     stumble_march / stumble_trot -0.250 +0.350 -0.210 = -0.1100  -> +6.30 deg FWD
  //     stumble_trot_drive         -0.140 +0.350 -0.210 = -0.0000  ->  0.00 deg
  // So with both soles flat on the floor, strat 21 / 16 are told -- at every
  // rollout, for the whole run -- to pitch the pelvis 6.3 deg FORWARD. Posture
  // weight is 12.0 in strat 21 and 12.0 in strat 6: IDENTICAL. The forward pull was
  // never in a weight, which is why ~50 cost-re-weighting runs could not remove it.
  //
  // ★ THE IDENTICAL BUG WAS ALREADY FOUND AND FIXED ONCE, WITH THE IDENTICAL NUMBER.
  // 2026-07-20 (DRIVE-24): "idle-tips-fwd keyframe lean FIXED hipP -0.25 -> -0.14
  // (drive-only)". That is exactly -(knee + ankleP) = -(0.350 - 0.210) = -0.140, the
  // value that closes the chain -- and `stumble_trot_drive` carries it today while its
  // unfixed siblings `stumble_march` and `stumble_trot` still carry -0.250. The fix
  // was simply never propagated.
  //
  // ★ CORROBORATION, independent: across n=30 genuinely-upright lean stages in the
  // 78-run corpus the median-of-run-max forward tilt is 6.3 deg, matching the
  // keyframe's built-in +6.30 deg to two decimals. That IS the observed fixed point.
  //
  // Target-side only -- no weight changes, no base-z change (with hipP -0.14 at base
  // z 1.020 the sole sits at z +0.0059 vs stand's +0.0056). Provably a NO-OP for every
  // keyframe whose chain already closes, which includes stand/stand_up/
  // stumble_trot_drive -- so strategy 6 stays byte-identical even with this numeric
  // ON. It also levels straighten (-0.080), deliberately in scope and not separately gated.
  // qpos leg layout: L hipY7 hipP8 hipR9 knee10 ankP11 ankR12 | R hipY13 hipP14 ...
  mjtNum posture_level_buf[64];
  {
    int pll_id = mj_name2id(model, mjOBJ_NUMERIC, "posture_leg_level");
    double pll = (pll_id >= 0)
        ? model->numeric_data[model->numeric_adr[pll_id]] : 0.0;
    if (pll > 0.5 && model->nq <= 64) {
      mju_copy(posture_level_buf, posture_target, model->nq);
      posture_level_buf[8]  = -(posture_level_buf[10] + posture_level_buf[11]);
      posture_level_buf[14] = -(posture_level_buf[16] + posture_level_buf[17]);
      posture_target = posture_level_buf;   // function scope: outlives every use below
    }
  }

  // ==================== STUMBLE: gait clock + swing-leg JOINT reference ===== //
  // Strategy 20 "stumble" stepping. (Full rationale at the Gait/Step Place
  // residual terms near the end of this function.) The gait CLOCK and the swing-
  // leg JOINT reference are computed HERE -- before the Posture cost reads
  // posture_target -- so the swing leg's hip/knee/ankle TARGETS fold up each
  // cycle and the strong, well-behaved Posture cost DRIVES the foot off the
  // ground. That turns stepping into a TRACKING problem (which sampling MPC
  // solves trivially, like every pose strategy tracking a keyframe) instead of a
  // SEARCH problem -- the planner will not spontaneously cross the cost barrier
  // of a half-formed step out of the stable-stand local minimum at calm
  // exploration. The Cartesian Gait/Step Place terms at the END of the function
  // refine foot height + xy placement, reusing g_amp/g_bump_* computed here.
  // NAME-GATED on "stumble": every other strategy (0-19) skips this block and
  // its two end-of-function residual terms default to weight 0 -> byte-identical.
  mjtNum stumble_posture_target[64];  // private swing-leg reference (stumble only)
  const bool is_stumble = (residual_keyframe_.name.rfind("stumble", 0) == 0);
  // TROT (strat 23, phase "stumble_trot"): the open-loop channel-freeze leg-lift
  // test vehicle. is_stumble subset; drives a CONTINUOUS forced march in the cost
  // (below) to match loco::ModifyControl, which hard-writes the swing leg in ctrl.
  const bool is_trot =
      is_stumble && (residual_keyframe_.name.find("trot") != std::string::npos);
  // WSS DRIVE (strat 24): a trot whose gait amplitude is command-latched (idle
  // => plant feet => stand). is_drive implies is_trot (keyframe "stumble_trot_
  // drive"), so all the velocity/step machinery below works; only g_amp differs.
  const bool is_drive =
      is_stumble && (residual_keyframe_.name.find("drive") != std::string::npos);
  // gait parameters. ★ R7 (2026-09-03): cadence + duty moved into the shared
  // GaitPhases helper (top of file) so the residual and loco::ModifyControl
  // cannot resolve different values; step height became a live numeric. The
  // tuning history they carried is kept here because it is the reason the
  // defaults are what they are:
  //   cadence 1.1 Hz  -- 0.8->1.1 FOOT-LIFT CLUSTER 2026-06-24 (Unitree H1-2
  //                      rl_gym uses 1.25). steps/s per foot. Slower -> the swing
  //                      fits in fewer spline knots. Stand-tolerable bandwidth:
  //                      spline 8 destabilises even a plain stand, spline<=5
  //                      holds (2026-06-18). WATCH: faster cadence may re-stress
  //                      the spline-5 bandwidth -> back off to 0.9 if it does.
  //   duty    0.60    -- 0.70->0.60 cluster: more SWING time (40% vs 30%) so the
  //                      foot has longer to clear. (more double support = steadier)
  //   height  0.06 m  -- 0.022->0.06 cluster: swing-foot peak Cartesian clearance.
  //                      ★ R7 raises the DEFAULT to 0.08 = Unitree H1-2 walk
  //                      clearance. This is only safe alongside contact-triggered
  //                      touchdown + ground-seek: higher clearance with a
  //                      free-running clock makes the landing problem WORSE, not
  //                      better, because the foot has further to fall in the same
  //                      fixed swing window. Live-tunable via stumble_step_height.
  constexpr double kStepHeightDflt = 0.08;
  int sth_id = mj_name2id(model, mjOBJ_NUMERIC, "stumble_step_height");
  const double kStepHeight = (sth_id >= 0)
      ? model->numeric_data[model->numeric_adr[sth_id]] : kStepHeightDflt;
  constexpr double kAmpRampSec = 4.5;   // ease the gait in over the first 4.5 s of the phase
  // desired CoM velocity [m/s] world x/y -- the NAV/walk command. Read from the
  // trot_des_vel_x/y numerics ONLY for the trot (is_trot); every other stumble
  // strategy (e.g. strat 20 march-in-place) keeps 0 so it steps in place. Feeds
  // BOTH the cost-side Step-Place foot target (below) AND the CoM-Vel velocity-
  // tracking residual -- without a nonzero target the cost says "stand still" and
  // the sampler CANCELS the open-loop walk drive (loco::ModifyControl), so foot
  // placement alone can never move the body. This is the propulsion connection.
  double kDesVelX = 0.0, kDesVelY = 0.0;   // NAV HOOK (trot only)
  if (is_trot) {
    int dvx_id = mj_name2id(model, mjOBJ_NUMERIC, "trot_des_vel_x");
    int dvy_id = mj_name2id(model, mjOBJ_NUMERIC, "trot_des_vel_y");
    if (dvx_id >= 0) kDesVelX = model->numeric_data[model->numeric_adr[dvx_id]];
    if (dvy_id >= 0) kDesVelY = model->numeric_data[model->numeric_adr[dvy_id]];
    // LIVE teleop override (2026-07-03; mirrored in loco::ModifyControl --
    // MUST match): the governed world-frame v_des from the TransitionLocked
    // cmd_vel governor replaces the static numerics when a client is active.
    // Step-pulse + R6 settle-governor below still apply on top.
    if (cmd_active_) {
      kDesVelX = cmd_vdes_world_[0];
      kDesVelY = cmd_vdes_world_[1];
    }
    // STEP-AND-SETTLE pulse: walk for trot_step_walk s, then SETTLE (v_des=0) for
    // the rest of trot_step_period s. v_des=0 gates OFF every walk term -> reverts
    // to the validated-ROBUST in-place trot (the recovery is the thing that already
    // works), so continuous walk's bistable topple is decomposed into discrete
    // forward steps each caught by the in-place trot. trot_step_period<=0 =>
    // continuous (no pulse) = byte-identical. MUST match loco::ModifyControl's pulse
    // (same data->time -> deterministic agreement between cost and open-loop swing).
    int tp_id = mj_name2id(model, mjOBJ_NUMERIC, "trot_step_period");
    int tw_id = mj_name2id(model, mjOBJ_NUMERIC, "trot_step_walk");
    double Tp = (tp_id >= 0) ? model->numeric_data[model->numeric_adr[tp_id]] : 0.0;
    double Tw = (tw_id >= 0) ? model->numeric_data[model->numeric_adr[tw_id]] : 0.0;
    if (Tp > 1e-6 && std::fmod(mju_max(0.0, data->time), Tp) >= Tw) {
      kDesVelX = 0.0; kDesVelY = 0.0;   // settle window -> robust in-place trot
    }
    // R6 SETTLE GOVERNOR -- MUST match loco::ModifyControl's governor (same
    // qpos/qvel inputs -> deterministic cost/swing agreement, exactly the
    // step-pulse pattern above). |v - v_des|*tau beyond trot_settle_thresh ->
    // v_des fades to 0 (all walk terms gate off = auto-settle to the validated
    // in-place trot) until the capture error re-enters the band. Default 0=off.
    // drive (strat 24) reads drive_settle_thresh so it is ON for drive without
    // touching strat 23 (its trot_settle_thresh stays 0 = byte-identical).
    const char *st_name = is_drive ? "drive_settle_thresh" : "trot_settle_thresh";
    int st_id = mj_name2id(model, mjOBJ_NUMERIC, st_name);
    double kSettle = (st_id >= 0)
        ? model->numeric_data[model->numeric_adr[st_id]] : 0.0;
    if (kSettle > 1e-6 && (kDesVelX != 0.0 || kDesVelY != 0.0)) {
      double zg = mju_max(0.5, data->qpos[2]);
      double tg = mju_sqrt(zg / 9.81);
      double gex = (data->qvel[0] - kDesVelX) * tg;
      double gey = (data->qvel[1] - kDesVelY) * tg;
      double gerr = mju_sqrt(gex * gex + gey * gey);
      double gg = mju_max(0.0, mju_min(1.0,
          (1.5 * kSettle - gerr) / (0.5 * kSettle)));
      gg = gg * gg * (3.0 - 2.0 * gg);
      kDesVelX *= gg; kDesVelY *= gg;
    }
  }
  // swing-leg JOINT-space lift offsets [rad] at full bump (added to the keyframe
  // target): fold the leg up so the foot clears ground. hip flexes back, knee
  // bends, ankle holds the sole level. This is what BOOTSTRAPS the step; the
  // leg hip_pitch/knee entries are additionally leg-gain-amplified by the
  // Posture cost, so even a modest Posture weight tracks the swing crisply.
  // FOOT-LIFT Tier B (2026-06-24): SCRIPTED swing fold RAISED now that Tier A
  // releases the competing costs (Foot-Up/Base-Height/Balance off the swing leg).
  // This joint-space arc IS the imposed swing reference tracked by the position
  // servos -- the lift driver. Was a gentle 0.11/0.20/0.08 (~1.5cm, fought by the
  // old costs); now folds the leg for a real ~5-6cm clearance. Live-scalable via
  // the stumble_swing_scale numeric (applied to the march fold below).
  constexpr double kSwingHip   = 0.35;  // hip_pitch flexes back -> thigh lifts
  constexpr double kSwingKnee  = 0.70;  // knee bends -> shank folds up (the clearance)
  constexpr double kSwingAnk   = 0.12;  // ankle_pitch -> toe clears (Foot-Up released on swing)
  // ---- BALANCE-GATED stepping (2026-06-19; signed-danger gate 2026-06-20) ----
  // STAND STILL (bent-knee, g_amp=0) and ramp the gait in ONLY while balance is
  // genuinely being lost; ease back to a still stand once recovered. "Losing
  // balance" = the SIGNED capture-point danger (computed once below) crossing
  // catch_trig -- the SAME quantity the catch-step + hip/arm tier use. The old
  // gate keyed on ABSOLUTE base tilt (deadband 5.7deg) + ABSOLUTE |vx|, which the
  // real-robot bring-up settle (leans ~8.7deg) and recovery motion both exceed ->
  // it MARCHED while settling ("never stood still") and kept marching through the
  // recovery ("kept twisting legs"); the twin settles ~3-4deg so it never showed
  // it (live-only). Signed danger is recovery-aware: a settling/recovering lean is
  // NOT treated as imbalance. Toggle off with model numeric
  // stumble_balance_gated=0 to restore the old always-on march (A/B, no rebuild).
  constexpr double kArmSec = 2.0;     // arm the gate this long after engage (calm bring-up)
  double g_amp = 0.0, g_bump_l = 0.0, g_bump_r = 0.0;
  // Stance-width regulator hand-off (see the block after the catch-step). When
  // a corrective step is scheduled these tell the Step Place residual to aim
  // that foot at the NOMINAL half-width instead of kStanceOffY, so the cost
  // that carries the foot and the decision to lift it cannot disagree.
  bool reg_l = false, reg_r = false;
  double reg_half = 0.0;
  // signed hip-pitch PLACEMENT offsets for the catch-step swing posture
  // (0 = the old pure-lift fold). See catch_place_hip below.
  double catch_dhip_l = 0.0, catch_dhip_r = 0.0;
  // PRE-LOADED lateral weight transfer [m], signed toward the STANCE foot.
  // See the catch_prep block: a fore/aft step cannot begin until the CoM is
  // over one foot, and that transfer does not fit inside the step's own window.
  double prep_rock = 0.0;
  // Raw gait phases + duty, hoisted to function scope so the ROCK (see
  // TrotRockHipRoll) can build its own PHASE-LED bump far below, from the same
  // clock loco::ModifyControl uses. -1 = clock never ran => rock stays 0.
  double rock_ph_l = -1.0, rock_ph_r = -1.0, rock_duty = 0.60;
  // Capture-point excursion (CoM heading vs equilibrium), hoisted to function
  // scope so BOTH the catch-step (below) AND the hip/arm angular-momentum
  // recovery tier (Centroidal angular momentum term, far below) read the same
  // signal -- the ankle->HIP->step hierarchy keyed off one quantity.
  double g_cap_ex = 0.0, g_cap_ey = 0.0;
  // TROT-window step scales (function scope so the Cartesian Gait/Step-Place
  // block far below sees them). Both default 1.0 (== quiet stand / push-recovery
  // BYTE-IDENTICAL) and only become >1 inside the trot starter window. Per the
  // MJPC-quadruped + Unitree-RL-walk references: a stable in-place step needs the
  // foot-CLEARANCE term to DOMINATE (quadruped Gait 2.0 = highest; Unitree
  // feet_swing_height = strongest), NOT a boosted joint reference the balance-
  // cautious planner ignores. trot_swing_scale raises the swing-foot height TARGET
  // (joint ref + Cartesian arc); trot_gait_wscale boosts the Gait foot-clearance
  // residual = an effective weight bump (25 -> ~25*scale^2) so foot-tracking wins
  // over the cheap lateral rock. Quiet stand keeps Gait 25 (residual ~0 when
  // planted anyway), so the validated 9/10 stand + push-recovery are untouched.
  double trot_swing_scale = 1.0, trot_gait_wscale = 1.0;
  if (is_stumble) {
    // gait CLOCK always runs; only the AMPLITUDE g_amp decides whether it steps.
    // Cadence is a live numeric (stumble_cadence) so the operator can dial the
    // step rate -- gentler/fewer steps for the trot starter -- without a rebuild.
    // ★ R7: resolved inside GaitPhases below (gc.cad), so the lookup that used to
    // sit here is gone; both readers now get cadence from the one helper.
    // stumble_swing_scale (default 1) multiplies the swing-leg lift DURING the
    // trot starter only (swing_mult below), so the operator can dial step HEIGHT
    // up to clear visibility live -- the balance-cautious planner damps the
    // reference, so the realized lift is a fraction of it. The catch-step keeps
    // mult=1 (validated push-recovery untouched).
    int ss_id = mj_name2id(model, mjOBJ_NUMERIC, "stumble_swing_scale");
    double kSwingScale = (ss_id >= 0)
        ? model->numeric_data[model->numeric_adr[ss_id]] : 1.0;
    // stumble_gait_boost (default 1): trot-window multiplier on the Gait foot-
    // clearance RESIDUAL -> effective weight 25*boost^2 (boost 2.6 -> ~170,
    // making foot-tracking the DOMINANT term per the references). Live-tunable.
    int gb_id = mj_name2id(model, mjOBJ_NUMERIC, "stumble_gait_boost");
    double kGaitBoost = (gb_id >= 0)
        ? model->numeric_data[model->numeric_adr[gb_id]] : 1.0;
    // ★ R7: duty + phases now come from the SHARED GaitPhases helper (see the
    // "ONE GAIT CLOCK, TWO READERS" block at the top of this file), so this site
    // and loco::ModifyControl cannot resolve a different cadence/duty or a
    // different phase offset. gait_phase_offset_ is the contact PLL correction,
    // latched in TransitionLocked; it is 0 unless trot_contact_switch is on, and
    // offset 0 reproduces the old fmod(t*kCad) clock bit-for-bit.
    const GaitClock gc = GaitPhases(model, data->time, gait_phase_offset_, is_trot);
    const double kDuty = gc.duty;
    double ph_l = gc.ph_l;                                  // L foot phase
    double ph_r = gc.ph_r;                                  // R antiphase
    g_bump_l = (ph_l < kDuty) ? 0.0
        : SwingBell((ph_l - kDuty) / (1.0 - kDuty));
    g_bump_r = (ph_r < kDuty) ? 0.0
        : SwingBell((ph_r - kDuty) / (1.0 - kDuty));
    rock_ph_l = ph_l; rock_ph_r = ph_r; rock_duty = kDuty;   // for the ROCK
    double const *cvel = SensorByName(model, data, "waist_lower_subcomvel");
    double const *flp  = SensorByName(model, data, "foot_left_pos");
    double const *frp  = SensorByName(model, data, "foot_right_pos");
    // --- SIGNED CAPTURE-POINT DANGER (computed ONCE; drives the whole
    //     ankle->HIP->step hierarchy: the march amplitude g_amp, the catch-step
    //     foot choice, and the hip/arm angular-momentum recovery tier far below
    //     via g_cap_ex/ey). e = z*tilt_dir + tau*com_vel (meters from upright):
    //   * base TILT (torso_up) = a LAG-FREE CoM-displacement proxy (IMU-direct;
    //     the HW velocity estimate lags ~30 ms), measured RELATIVE to the steady
    //     ~3-4deg forward lean (lean_nominal_x) so the catch is SYMMETRIC about
    //     equilibrium (a back push immediately gives ex<0), and
    //   * com_vel * sqrt(z/g) = the capture-point velocity LEAD.
    //   danger = |SIGNED capture excursion| = sqrt(ex^2 + (z*ty)^2). SIGNED so a
    //   RECOVERING velocity SHRINKS danger (a settling lean already moving back is
    //   NOT "losing balance"); the LATERAL axis keeps tilt only (z*ty) and DROPS
    //   lateral velocity vy = the gait weight-shift ROCK, which would self-trigger.
    double const *tup = SensorByName(model, data, "torso_up");
    int pid = mj_name2id(model, mjOBJ_BODY, "pelvis");
    if (pid < 0) pid = 1;
    const mjtNum *com = data->subtree_com + 3 * pid;
    double zc  = mju_max(0.5, com[2]);
    double tau = mju_sqrt(zc / 9.81);
    int lnx_id = mj_name2id(model, mjOBJ_NUMERIC, "lean_nominal_x");
    double kLocoX = (lnx_id >= 0)
        ? model->numeric_data[model->numeric_adr[lnx_id]] : 0.06;
    // ★ STALE CALIBRATION (measured 2026-09-08, 24 quiet stands, spread 0.0010).
    // lean_nominal_x is meant to BE the resting torso_up[0] so that `danger` is
    // ~0 at equilibrium and the catch is symmetric. It is 0.06 ("~sin(3-4deg)");
    // the stand actually rests at 0.0197 (1.13 deg). Almost certainly stale since
    // stance_off_x 0.13 -> 0.06 on 2026-09-06 changed the resting posture.
    // Consequences, both measured:
    //  * quiet-stand danger idles at 0.041 -- 4.4x its own 0.0093 jitter -- so the
    //    trigger floor is pinned at 0.057 by a CONSTANT, not by noise, and the
    //    catch can never be armed early enough to matter.
    //  * the catch is ASYMMETRIC: from rest, backward needs d|e| = 0.079 to reach
    //    catch_trig while forward needs 0.161. A 2x bias nobody intended.
    // Scoped to strat 20 (is_stumble && !is_trot) because the correct value is a
    // property of the KEYFRAME's resting posture, and trot/drive stand in a
    // different pose -- one shared number cannot be right for both. -1 = "use
    // lean_nominal_x" = byte-identical.
    if (is_stumble && !is_trot) {
      double so = GetNumberOrDefault(-1.0, model, "stumble_lean_nominal");
      if (so >= 0.0) kLocoX = so;
    }
    double tx = (tup ? tup[0] : 0.0) - kLocoX, ty = tup ? tup[1] : 0.0;
    double vx = cvel ? cvel[0] : 0.0, vy = cvel ? cvel[1] : 0.0;
    double ex = zc * tx + tau * vx;                      // signed fore-aft capture
    double ey = zc * ty + tau * vy;                      //   (full, for foot choice)
    g_cap_ex = ex; g_cap_ey = ey;         // share with the hip/arm recovery tier
    double ey_pos = zc * ty;                             // lateral: tilt only (rock-immune)
    double danger = mju_sqrt(ex * ex + ey_pos * ey_pos);
    // ★ RATE LEAD (catch_rate_lead, default 0 = level-only, byte-identical).
    // danger_eff = danger + k * tau * d(danger)/dt : one lead-time of
    // divergence, so a disturbance that is GROWING arms the step while it is
    // still small, and one that has peaked and is settling does not. Only the
    // GATE uses it -- a settling lean must not be handed a catch-step.
    const double kRL = GetNumberOrDefault(0.0, model, "catch_rate_lead");
    if (kRL > 0.0) danger += kRL * tau * mju_max(0.0, danger_rate_);
    int ct_id = mj_name2id(model, mjOBJ_NUMERIC, "catch_trig");
    int cf_id = mj_name2id(model, mjOBJ_NUMERIC, "catch_full");
    double kCatchTrig = (ct_id >= 0)
        ? model->numeric_data[model->numeric_adr[ct_id]] : 0.085;
    double kCatchFull = (cf_id >= 0)
        ? model->numeric_data[model->numeric_adr[cf_id]] : 0.16;
    // ★ SUSTAINED-BIAS ARM: once the latch is set (see TransitionLocked (a2b)),
    // SHIFT the whole ramp down to the bias threshold, keeping its WIDTH. Not
    // "trigger at the low value and still ramp to catch_full" -- that would arm
    // early and then deliver almost no amplitude for most of the extra time,
    // which is the worst of both. Shifting means recov reaches 1 at
    // bias_trig + (catch_full - catch_trig), so the step gets FULL authority
    // during the seconds the early detection just bought. Latch is false at the
    // default catch_bias_trig = 0 => byte-identical.
    if (catch_bias_on_) {
      const double width = mju_max(1e-3, kCatchFull - kCatchTrig);
      kCatchTrig = GetNumberOrDefault(kCatchTrig, model, "catch_bias_trig");
      kCatchFull = kCatchTrig + width;
    }
    // recov: 0 below catch_trig (STAND STILL), smoothstep to 1 at catch_full.
    double recov = mju_min(1.0, mju_max(0.0,
        (danger - kCatchTrig) / mju_max(1e-3, kCatchFull - kCatchTrig)));
    recov = recov * recov * (3.0 - 2.0 * recov);         // smoothstep
    // ★ COMMITTED STEP PAIR: keep the amplitude up for a whole gait cycle after
    // a catch fires, so the TRAILING foot gets its swing window too. Without it
    // the arm window (0.36-0.9 s) is shorter than one cycle (0.91 s) and only
    // one foot can ever move -- which is why the stance translates by one leg.
    if (catch_hold_until_ > 0.0 && data->time < catch_hold_until_)
      recov = mju_max(recov, catch_hold_recov_);
    double arm = mju_min(time_in_phase / kArmSec, 1.0);
    arm = arm * arm * (3.0 - 2.0 * arm);                 // calm bring-up, no spurious step
    // --- AMPLITUDE: balance-gated (default) or legacy continuous march ---
    int bg_id = mj_name2id(model, mjOBJ_NUMERIC, "stumble_balance_gated");
    bool balance_gated =
        (bg_id < 0) || (model->numeric_data[model->numeric_adr[bg_id]] > 0.5);
    if (balance_gated) {
      g_amp = arm * recov;   // STAND STILL until the signed capture point escapes
    } else {
      double amp_r = mju_min(time_in_phase / kAmpRampSec, 1.0);
      g_amp = amp_r * amp_r * (3.0 - 2.0 * amp_r);       // legacy continuous march
    }
    if (is_trot) {
      // TROT: continuous forced march. The channel-freeze (loco::ModifyControl)
      // hard-writes the swing leg into ctrl regardless of the sampler, so make
      // the COST agree -- ramp the gait amplitude in (NO balance gate) so the
      // swing fold + Tier-A swing-release apply continuously and the Posture cost
      // EXPECTS the lifted swing leg (no spurious penalty fighting the freeze).
      g_amp = arm;
      // DRIVE: the stand<->trot latch scales the march. idle (drive_gait_amp_=0)
      // => g_amp = arm*recov = the strat-20 balance-gated stand (still, but
      // steps to catch a push); commanded => arm*1 = full trot. drive_gait_amp_
      // comes from the plan snapshot (set in TransitionLocked), so cost and
      // ModifyControl swing agree.
      if (is_drive) g_amp = arm * mju_max(drive_gait_amp_, recov);
    }
    // --- TROT STARTER (opt-in, strategy 20 only): a deliberate in-place march
    // over the window [trot_delay, trot_delay+trot_sec] after engage, then HAND
    // OFF to the quiet balance-gated stand above ("trot 10 s -> stand & stumble").
    // Lets the operator SEE balance-during-stepping at startup without waiting
    // for a push. trot_delay skips the node's bring-up ramp; the window eases in
    // AND out over kTrotEase so start/stop are smooth (no snap). g_amp =
    // max(gated, env) so a genuine push DURING the trot still drives a decisive
    // catch-step. The march reuses the same clock/swing as the catch-step
    // (g_bump_l/r below), so the arm-quiet velocity split (leg_gate = 1 - g_amp)
    // damps the upper body throughout. trot_sec<=0 => OFF => byte-identical to
    // the pre-trot gait (and every other strategy stays untouched, name-gated).
    int td_id = mj_name2id(model, mjOBJ_NUMERIC, "stumble_trot_delay");
    int tn_id = mj_name2id(model, mjOBJ_NUMERIC, "stumble_trot_sec");
    double kTrotDelay = (td_id >= 0)
        ? model->numeric_data[model->numeric_adr[td_id]] : 0.0;
    double kTrotSec = (tn_id >= 0)
        ? model->numeric_data[model->numeric_adr[tn_id]] : 0.0;
    double t_trot = time_in_phase - kTrotDelay;
    if (kTrotSec > 1e-3 && t_trot >= 0.0 && t_trot < kTrotSec) {
      constexpr double kTrotEase = 1.5;                  // s, smooth ease in AND out
      double rin  = mju_min(t_trot / kTrotEase, 1.0);
      double rout = mju_min((kTrotSec - t_trot) / kTrotEase, 1.0);
      double env  = mju_max(0.0, mju_min(rin, rout));
      env = env * env * (3.0 - 2.0 * env);               // smoothstep envelope
      g_amp = mju_max(g_amp, env);                       // force the march, keep push-step
      trot_swing_scale = kSwingScale;   // taller swing TARGET (joint ref + Cartesian arc)
      trot_gait_wscale = kGaitBoost;    // dominant foot-clearance cost (the lift driver)
    }
    // --- DECISIVE OMNIDIRECTIONAL catch-step: pick the foot toward the capture
    // point and force it to swing NOW (N-step capturability, Pratt/Koolen --
    // relocate the base under the falling CoM before the weak ankle saturates).
    // In gated mode g_amp already reflects danger (arm*recov); in legacy mode
    // SNAP it (arm-gated, so bring-up stays calm). fwd->trailing foot fwd,
    // back->front foot back, left/right->falling-side foot out.
    if (cvel && flp && frp && danger > kCatchTrig) {
      g_amp = mju_max(g_amp, arm * recov);
      bool stepL;
      if (std::fabs(ey) > std::fabs(ex)) {
        stepL = (ey > 0.0);                  // falling LEFT -> left foot out left
      } else {
        stepL = (ex > 0.0) ? (flp[0] <= frp[0])    // forward -> trailing(rear) foot
                           : (flp[0] >= frp[0]);   // back -> front foot
      }
      if (stepL) g_bump_l = mju_max(g_bump_l, arm * recov);
      else       g_bump_r = mju_max(g_bump_r, arm * recov);
      // ---- ★ MAKE THE CATCH-STEP ACTUALLY STEP (2026-09-07) --------------
      // The catch-step ARMS correctly -- it fires in 87 of 87 falls with a
      // 0.72 s median lead -- but replaying 94 runs found essentially no
      // recovery-scale steps: lift is fine (~5 cm realised against a 6 cm
      // command) while TRAVEL is 0.6-8 cm where relocating the support polygon
      // needs 15-30 cm. The two multipliers that make a step decisive already
      // exist and are deliberately withheld from it: the source note at
      // kSwingScale reads "The catch-step keeps mult=1 (validated push-recovery
      // untouched)", so `trot_swing_scale` (taller swing target) and
      // `trot_gait_wscale` (foot-clearance cost DOMINATES -- the reference
      // walkers' first lesson) apply ONLY inside the trot-starter window and
      // never during a real catch.
      // Scaled by `recov`, so a marginal wobble is untouched and only a genuine
      // capture-point escape gets the decisive step. Both default 1.0 = no-op.
      const double kCSS = GetNumberOrDefault(1.0, model, "catch_swing_scale");
      const double kCGB = GetNumberOrDefault(1.0, model, "catch_gait_boost");
      if (kCSS > 1.0)
        trot_swing_scale = mju_max(trot_swing_scale, 1.0 + (kCSS - 1.0) * recov);
      if (kCGB > 1.0)
        trot_gait_wscale = mju_max(trot_gait_wscale, 1.0 + (kCGB - 1.0) * recov);
      // ---- ★ DIRECTION-AWARE SWING POSTURE (catch_place_hip, default 0) ----
      // The swing fold below is a PURE SYMMETRIC LIFT: hip_pitch -= kSwingHip
      // regardless of which way the foot must travel. That is a FORWARD swing
      // shape, so a backward catch-step has the Posture cost pulling the thigh
      // forward while Step Place asks the foot to land behind -- the two costs
      // fight, and the foot goes nowhere.
      // loco::ModifyControl already solves this for the TROT with a signed
      // placement term, dHipP = clip(-step_x/0.80, +-0.45), added on the
      // placement ramp. That machinery is trot-only. This is the same term for
      // the stumble COST, so the posture target agrees with where Step Place is
      // sending the foot instead of opposing it.
      const double kCPH = GetNumberOrDefault(0.0, model, "catch_place_hip");
      if (kCPH > 0.0) {
        // same capture-point target the Step Place residual builds, recomputed
        // here because that block runs much later in the residual
        const double sox = GetNumberOrDefault(0.13, model, "stance_off_x");
        double offx = vx * tau;
        offx = mju_max(-0.30, mju_min(0.30, offx));
        const double tgt_x = com[0] + sox + offx;
        catch_dhip_l = mju_max(-0.45, mju_min(0.45, -(tgt_x - flp[0]) / 0.80)) * kCPH;
        catch_dhip_r = mju_max(-0.45, mju_min(0.45, -(tgt_x - frp[0]) / 0.80)) * kCPH;
      }
    }
    // ---- ★ PRE-LOADED WEIGHT TRANSFER (catch_prep_trig/_rock, default OFF) --
    // MEASURED 2026-09-07, the ~1.1 s before a fall (1.0 = CoM over one foot):
    //     fore/aft back 0.31   fore/aft fwd 0.68   |  LATERAL 11.3 and 13.9
    //     L/R load imbalance  54% / 37%            |  115% / 120%
    // Both feet stand at the same x (kStagL = kStagR = 0), so a FORE/AFT step
    // needs a LATERAL weight shift before either foot is free. A lateral push
    // supplies that shift for free -- which is exactly and only where steps
    // occur. Fore/aft never gets it, so the swing is commanded onto a foot that
    // is still carrying 300+ N and nothing lifts.
    // The existing rock (Lateral Center, lat_amp * g_amp * (g_bump_r-g_bump_l))
    // is commanded at the SAME instant as the swing, and with the capture point
    // diverging on tau = 0.30 s there is under 0.5 s to move the CoM ~0.18 m.
    // Measured outcome: commanded 0.175 m, ACHIEVED 0.06-0.13 m.
    // So start the transfer EARLIER, on the same rate-led danger but at a LOWER
    // threshold, toward the foot that the catch-step rule will keep planted --
    // predicted here with that identical rule so prep and step cannot disagree.
    // Both numerics default 0 => byte-identical.
    const double kPT = GetNumberOrDefault(0.0, model, "catch_prep_trig");
    const double kPR = GetNumberOrDefault(0.0, model, "catch_prep_rock");
    if (kPT > 0.0 && kPR > 0.0 && cvel && flp && frp && kCatchTrig > kPT) {
      double pp = (danger - kPT) / mju_max(1e-3, kCatchTrig - kPT);
      pp = mju_min(1.0, mju_max(0.0, pp));
      pp = pp * pp * (3.0 - 2.0 * pp);                  // smoothstep
      bool willL;                                       // SAME rule as the step
      if (std::fabs(ey) > std::fabs(ex)) {
        willL = (ey > 0.0);
      } else {
        willL = (ex > 0.0) ? (flp[0] <= frp[0]) : (flp[0] >= frp[0]);
      }
      // swing LEFT -> shift the CoM toward the RIGHT (-y) stance foot, matching
      // the sign of (g_bump_r - g_bump_l) that the step's own rock uses.
      prep_rock = (willL ? -1.0 : 1.0) * kPR * pp * arm;
    }
    // ---- ★ STANCE-WIDTH REGULATOR (stance_regulate, default 0 = OFF) ------
    // THE STAND CREEPS ITS OWN FEET APART. Measured 2026-09-07 on quiet 240 s
    // runs with ZERO disturbance: ankle-to-ankle 0.375 m -> 0.618 m, and
    // 96-100% of that travel happens while the foot carries 250-450 N -- the
    // LOADED foot slides. It is not slippery contact (mu=1.0, impratio 100),
    // and it is not the Step Place target: the `step_place_gated` arm removed
    // that command entirely during a still stand and the creep was unchanged.
    // The cause is statics -- both hips hold ~34 N*m of ABDUCTION to carry the
    // pelvis between splayed legs (L +34.5, R -33.0 measured), and the contact
    // creeps under that sustained lateral load. So it cannot be tuned away,
    // only CORRECTED, and nothing in this strategy ever brought the feet back:
    // the stance could only widen, from bring-up until the fall.
    //
    // Schedules a corrective step for the foot furthest from its nominal
    // half-width and points Step Place at that half-width. Amplitude ramps
    // from 0 at the tolerance edge so it fades in and out instead of
    // chattering at the boundary. GATED TO THE CALM STATE (danger <=
    // catch_trig): a real catch-step must always outrank housekeeping.
    const double kRegOn = GetNumberOrDefault(0.0, model, "stance_regulate");
    if (kRegOn > 0.5 && cvel && flp && frp && danger <= kCatchTrig) {
      const double kNomW = GetNumberOrDefault(0.38, model, "stance_nominal_w");
      const double kTol = mju_max(
          1e-3, GetNumberOrDefault(0.05, model, "stance_regulate_tol"));
      const double width = flp[1] - frp[1];
      const double err = mju_abs(width - kNomW);
      if (err > kTol) {
        reg_half = 0.5 * kNomW;
        // whichever foot is further from where it should be does the moving
        const double eL = mju_abs((flp[1] - com[1]) - reg_half);
        const double eR = mju_abs((frp[1] - com[1]) + reg_half);
        const double amp = arm * mju_min(1.0, (err - kTol) / kTol);
        if (eL >= eR) { reg_l = true; g_bump_l = mju_max(g_bump_l, amp); }
        else          { reg_r = true; g_bump_r = mju_max(g_bump_r, amp); }
        g_amp = mju_max(g_amp, amp);
      }
    }
    if (model->nq <= 64) {
      for (int i = 0; i < model->nq; i++)
        stumble_posture_target[i] = posture_target[i];
      // ---- ★ STAGGERED STANCE (stumble_stagger [rad], default 0 = OFF) -----
      // Fore/aft failure is CoP-GEOMETRY limited: at collapse the CoP sits at
      // 110% of the heel (backward) and 100% of the toe (forward) while the
      // ankle still holds 15-41% of its torque in reserve. So the binding
      // number is HEEL LENGTH BEHIND THE CoM, and the only way to raise it is
      // to lengthen the polygon. Measured offline from this keyframe:
      //     stagger 0.00 rad -> heel margin 0.059 m
      //     stagger 0.05 rad -> heel margin 0.109 m  (+85%)
      //     stagger 0.10 rad -> heel margin 0.145 m  (+146%)
      // Only the HIPS are biased here; the knee/ankle compensation that keeps
      // both soles flat is left to the planner, because Foot L/R Up (2000) and
      // Base Height (450) dominate Posture (12) and will settle the feet on the
      // floor far more reliably than a hand-solved pose -- an offline 2x2
      // Newton solve for that pose left one foot up to 18 mm off the ground.
      // NOTE the 2026-06-18 stagger trial was REVERTED as "no push improvement",
      // but that was scored on levels-survived, which cannot separate absorbing
      // from stepping, and before the CoP limit was known. Re-testing on the
      // continuous drift metric.
      const double kStag = GetNumberOrDefault(0.0, model, "stumble_stagger");
      if (kStag != 0.0) {
        stumble_posture_target[7 + 1] -= kStag;   // L hip_pitch: leg forward
        stumble_posture_target[7 + 7] += kStag;   // R hip_pitch: leg back
      }
      double ll = g_amp * g_bump_l * trot_swing_scale, lr = g_amp * g_bump_r * trot_swing_scale;  // lift per leg
      // L leg joints: qpos 7+1 hip_pitch, 7+3 knee, 7+4 ankle_pitch
      stumble_posture_target[7 + 1] -= kSwingHip  * ll;
      stumble_posture_target[7 + 1] += catch_dhip_l * ll;   // signed placement
      stumble_posture_target[7 + 3] += kSwingKnee * ll;
      stumble_posture_target[7 + 4] -= kSwingAnk  * ll;
      // R leg joints: qpos 7+7 hip_pitch, 7+9 knee, 7+10 ankle_pitch
      stumble_posture_target[7 + 7]  -= kSwingHip  * lr;
      stumble_posture_target[7 + 7]  += catch_dhip_r * lr;  // signed placement
      stumble_posture_target[7 + 9]  += kSwingKnee * lr;
      stumble_posture_target[7 + 10] -= kSwingAnk  * lr;
      posture_target = stumble_posture_target;
    }
  }

  // -------------- Residual terms (order == XML <user> sensors) -------------- //
  // ----- Height: head feet vertical error ----- //

  // feet sensor positions
  double *foot_right_pos = SensorByName(model, data, "foot_right_pos");
  double *foot_left_pos = SensorByName(model, data, "foot_left_pos");

  double *head_position = SensorByName(model, data, "head_position");
  double head_feet_error =
      head_position[2] - 0.5 * (foot_right_pos[2] + foot_left_pos[2]);
  // ===== STUMBLE swing-release masks (foot-lift fix 2026-06-24, Tier A) =====
  // During a foot's scheduled swing (g_amp*g_bump_foot -> 1) the "keep foot down /
  // upright / at height" costs are RELEASED off the swing leg / single-support
  // body so lifting is affordable -- the MJPC-quadruped per-gait weight drop,
  // applied per-tick. is_stumble-gated (swing_*/step_active = 0 for every other
  // strategy AND for a planted/calm stumble) -> calm stand + all strats byte-id.
  double swing_r = is_stumble ? mju_min(1.0, g_amp * g_bump_r) : 0.0;
  double swing_l = is_stumble ? mju_min(1.0, g_amp * g_bump_l) : 0.0;
  double step_active = mju_max(swing_r, swing_l);  // whole-body single-support activity
  double height_scale = 1.0 - 0.6 * step_active;  // release base-height anchor in single support (CoM dips)
  residual[counter++] = height_scale * (head_feet_error - height_goal);

  // ----- Balance: CoM-feet xy error ----- //

  // capture point
  double *com_velocity = SensorByName(model, data, "torso_subtreelinvel");

  // ----- CoM xy velocity tracking ----- //
  // Target 0 (stand still) normally; for the trot, target the commanded walk
  // velocity kDesVel* so this term DRIVES the forward push instead of fighting it.
  // THE propulsion cost: every reference sampling/RL walker (MJPC walk.cc,
  // Playground/Unitree) tracks CoM velocity -- foot placement alone cannot move
  // the body against the balancing sampler (verified: kw=0 placement just wanders
  // around rest). kDesVel*=0 for all non-trot -> byte-identical there. trot_vel_w
  // boosts ONLY the FORWARD (x) tracking authority for the trot so velocity
  // tracking out-votes the come-to-rest Balance term (MJPC walk.cc demotes Balance
  // / promotes the velocity term for locomotion). trot_lat_w does the SAME for the
  // LATERAL (y) tracking (target kDesVelY, usually 0) -- forward walk wanders
  // sideways because x-tracking was boosted 12x while y stayed at base 1, so the
  // sampler let lateral velocity drift; trot_lat_w damps it to walk STRAIGHT.
  // Both default 1.0 -> in-place trot + non-trot byte-identical.
  double vel_w = 1.0, lat_w = 1.0;
  if (is_trot) {
    int vw_id = mj_name2id(model, mjOBJ_NUMERIC, "trot_vel_w");
    if (vw_id >= 0) vel_w = model->numeric_data[model->numeric_adr[vw_id]];
    int lw_id = mj_name2id(model, mjOBJ_NUMERIC, "trot_lat_w");
    if (lw_id >= 0) lat_w = model->numeric_data[model->numeric_adr[lw_id]];
  }
  residual[counter + 0] = vel_w * (com_velocity[0] - kDesVelX);
  residual[counter + 1] = lat_w * (com_velocity[1] - kDesVelY);
  counter += 2;

  // ----- joint velocity ----- //
  mju_copy(residual + counter, data->qvel + 6, model->nu);
  if (is_stumble) {
    // Split the "Velocity" joint-velocity damping by joint group (the G1-
    // locomotion pattern: vel/pose-regularised ARMS, gait-driven LEGS). LEG/torso
    // damping (actuators 0..12) RELEASES as the gait amplitude ramps (1-g_amp) so
    // a catch-step is fast and unimpeded; ARM damping (13..26) is BOOSTED + always
    // on so the upper body stays quiet (no flail) even as the gait/balance perturb
    // it -- the user's "keep the arms at home, recover with legs/hips". When calm
    // (g_amp=0) the legs are fully damped (leg_gate=1) so the still-stand is
    // unchanged. is_stumble-gated => strat 0-19, 21 byte-identical.
    double leg_gate = 1.0 - g_amp;
    for (int i = 0; i < model->nu; i++)
      residual[counter + i] *= (i < 13) ? leg_gate : 1.6;
  }
  counter += model->nu;

  // ----- torso height ----- //
  double torso_height = SensorByName(model, data, "torso_position")[2];

  // ----- balance ----- //
  // capture point
  double *subcom = SensorByName(model, data, "torso_subcom");
  double *subcomvel = SensorByName(model, data, "torso_subcomvel");

  double capture_point[3];
  // Horizon kept at 0.3 s (a 0.45 s preview was trialed 2026-05-26 and reverted:
  // no hold-rate gain, lower-quality holds).
  mju_addScl(capture_point, subcom, subcomvel, 0.3, 3);
  capture_point[2] = 1.0e-3;

  // CoM fore-aft sim2real correction (2026-06-13). The REAL robot's CoM sits ahead of the
  // model's (un-sysid'd mass), so a model-centered CoM leaves the REAL CoM forward -> the
  // robot leans forward and the ankles range-limit trying to pull it back. `com_x_offset`
  // (model numeric, meters, default 0) biases this balance target: it tells the planner the
  // CoM is this much further FORWARD (world +x = robot-forward for the facing-+x stand), so
  // the planner holds the actual CoM that much BACK. RAISE it if the real robot leans forward;
  // the value that makes it stand upright == the real CoM forward offset (doubles as a sysid
  // measurement). XML-tunable -> no rebuild to change the value; default 0 = exact prior behavior.
  // 2026-07-20: com_x_offset now rides the SUPPORT frame, not world +x. It was
  // written when the stand faced +x and was harmless while shipped at 0.0, but
  // it goes LIVE this session and real headings ran -41..+15 deg tonight -- a
  // world-frame bias would be aimed that far off the robot's actual forward.
  // Same midFeetZUp rule as the trim below / the 2026-07-16 balance_frame fix;
  // byte-identical at heading 0 and while the numeric is 0.
  {
    int com_off_id = mj_name2id(model, mjOBJ_NUMERIC, "com_x_offset");
    double com_fwd_bias =
        (com_off_id >= 0) ? model->numeric_data[model->numeric_adr[com_off_id]] : 0.0;
    if (com_fwd_bias != 0.0) {
      double fwd[2] = {1.0, 0.0};
      double const *flf = SensorByName(model, data, "foot_left_forward");
      double const *frf = SensorByName(model, data, "foot_right_forward");
      if (flf && frf) {
        double fx = flf[0] + frf[0], fy = flf[1] + frf[1];
        double len = mju_sqrt(fx * fx + fy * fy);
        if (len > 1.0e-6) { fwd[0] = fx / len; fwd[1] = fy / len; }
      }
      capture_point[0] += com_fwd_bias * fwd[0];
      capture_point[1] += com_fwd_bias * fwd[1];
    }
  }

  // T1 REFERENCE TRIM v2 (ported 2026-07-20; see the s_trim_x file-scope note).
  // Applied along the SUPPORT frame -- the feet's own mean heading, the same
  // midFeetZUp rule as the 2026-07-16 balance_frame fix -- NOT world +x. This
  // matters on real: heading ran -41 deg .. +11 deg across the 07-20 runs, so a
  // world-frame correction would be aimed tens of degrees off the robot's
  // actual forward. (The com_x_offset block above is still world-frame; it is
  // shipped at 0.0 so nothing is live, but it carries the same latent defect.)
  //
  // Semantics mirror com_x_offset: a POSITIVE trim tells the planner the
  // capture point is further forward/left than measured, so the planner holds
  // the real CoM back/right. Both trims are exactly 0.0 unless stand_trim_tau
  // > 0, and the whole block is skipped in that case -> byte-identical.
  // com_y_offset rides the same support-frame axes: the static lateral sibling
  // of com_x_offset, for a sided park that survives an honest roll calibration
  // (lateral mass-model asymmetry, e.g. the one-sided magpie arm). Shipped 0.
  double com_y_bias = 0.0;
  {
    int cy_id = mj_name2id(model, mjOBJ_NUMERIC, "com_y_offset");
    if (cy_id >= 0) com_y_bias = model->numeric_data[model->numeric_adr[cy_id]];
  }
  // STATIC support-frame FORWARD bias (2026-07-22): the fore-aft sibling of
  // com_y_offset. MEASURED need on real (ankleidk npz): ZMP sits +3.5 cm ahead
  // of the model CoM at quiet stance = unmodeled forward mass the planner never
  // compensates -> chronic forward park -> tips. POSITIVE = planner holds the
  // real CoM BACK by this much (same semantics as trim_x). Support-frame, so it
  // stays aimed at the robot's true forward at any heading (runs today sat at
  // +20..+35 deg where the world-frame com_x_offset would be aimed wrong).
  // The leaky trim rides ON TOP for the residual/day-to-day part.
  double com_x_bias = 0.0;
  {
    int cx_id = mj_name2id(model, mjOBJ_NUMERIC, "com_x_offset_support");
    if (cx_id >= 0) com_x_bias = model->numeric_data[model->numeric_adr[cx_id]];
  }
  if (s_trim_x != 0.0 || s_trim_y != 0.0 || com_y_bias != 0.0 ||
      com_x_bias != 0.0) {
    double fwd[2] = {1.0, 0.0}, lat[2] = {0.0, 1.0};
    double const *flf = SensorByName(model, data, "foot_left_forward");
    double const *frf = SensorByName(model, data, "foot_right_forward");
    if (flf && frf) {
      double fx = flf[0] + frf[0], fy = flf[1] + frf[1];
      double len = mju_sqrt(fx * fx + fy * fy);
      if (len > 1.0e-6) {
        fwd[0] = fx / len; fwd[1] = fy / len;
        lat[0] = -fwd[1];  lat[1] = fwd[0];
      }
    }
    double lat_bias = s_trim_y + com_y_bias;
    double fwd_bias = s_trim_x + com_x_bias;
    capture_point[0] += fwd_bias * fwd[0] + lat_bias * lat[0];
    capture_point[1] += fwd_bias * fwd[1] + lat_bias * lat[1];
  }

  // project onto the support polygon = the {L_foot, R_foot} line (plus the
  // optional sole hull below). Inside -> pcp = capture_point -> residual 0.
  double pcp[3];
  {
    // foot-foot line projection (the only support polygon in this task).
    double axis[3], center[3], vec[3];
    mju_sub3(axis, foot_right_pos, foot_left_pos);
    axis[2] = 1.0e-3;
    double length = 0.5 * mju_normalize3(axis) - 0.05;
    mju_add3(center, foot_right_pos, foot_left_pos);
    mju_scl3(center, center, 0.5);
    // ---- ★ 2026-09-07: THE SUPPORT "POLYGON" HERE IS A LINE AT THE ANKLES ---
    // foot_left_pos / foot_right_pos are framepos of *_ankle_roll_link -- the
    // ANKLE JOINT, not the sole. This branch projects the capture point onto
    // the segment joining them, which has ZERO fore-aft extent, so Balance
    // pulls the CoM onto the ankle line and penalises every millimetre of
    // fore/aft excursion off it. But the real sole runs 0.177 m AHEAD of the
    // ankle and only 0.087 m behind it, i.e. the ankle sits 0.045 m BEHIND the
    // foot's centre.
    // Measured consequence (quiet stand, scratch_bench/stance_margin.py): the
    // CoM parks 0.045-0.076 m behind the foot-polygon centre -- exactly on the
    // ankle line, where this cost puts it -- leaving 0.088 m of heel margin
    // against 0.180 m of toe margin. Sustained-drift capacity follows that
    // geometry: ~43 N forward vs ~15 N backward.
    // balance_fore_bias shifts the projection target forward along the body
    // fore-aft axis; +0.045 puts it on the SOLE CENTRE instead of the ankle.
    // Default 0.0 = the old line, byte-identical, and stumble-gated besides.
    const double kFB = is_stumble
        ? GetNumberOrDefault(0.0, model, "balance_fore_bias") : 0.0;
    // ---- ★ SOLE HULL: give the projection its real FORE-AFT EXTENT --------
    // Shifting the target point alone (balance_fore_bias) was measured and did
    // NOT help (32-run 4-direction confirmation, pooled p=0.52). The deeper
    // defect is not WHERE the line sits but that it IS a line: the support
    // region has no fore-aft width at all, so Balance penalises the capture
    // point for being 1 mm ahead of the ankles even though the sole carries it
    // for another 0.177 m. Real support polygons are 2-D, and a capture point
    // INSIDE one costs nothing (that is the whole WBC premise, already honoured
    // by the arm-contact branches above via project_triangle / the 3-vertex
    // inside-test -- this branch is the odd one out).
    // balance_sole_fore / _aft (both default 0 = the old zero-width line, so
    // byte-identical when off) admit the capture point free of charge within
    // [-aft, +fore] of the ankle line; outside, it still projects to the edge.
    // Measured sole extents are +0.177 toe / -0.087 heel from the ankle.
    const double kSF = is_stumble
        ? GetNumberOrDefault(0.0, model, "balance_sole_fore") : 0.0;
    const double kSA = is_stumble
        ? GetNumberOrDefault(0.0, model, "balance_sole_aft") : 0.0;
    double perp[3] = {-axis[1], axis[0], 0.0};   // body front, yaws with stance
    if (kFB != 0.0 || kSF > 0.0 || kSA > 0.0) mju_normalize3(perp);
    if (kFB != 0.0) mju_addToScl3(center, perp, kFB);
    mju_sub3(vec, capture_point, center);
    double t = mju_dot3(vec, axis);
    t = mju_max(-length, mju_min(length, t));
    double fa = 0.0;
    if (kSF > 0.0 || kSA > 0.0) {
      fa = mju_dot3(vec, perp);
      fa = mju_max(-kSA, mju_min(kSF, fa));
    }
    mju_scl3(vec, axis, t);
    mju_add3(pcp, vec, center);
    if (fa != 0.0) mju_addToScl3(pcp, perp, fa);
    pcp[2] = 1.0e-3;
  }

  // is leaning - modified to be less strict than standing.
  // floored at 0.3 so balance never fully turns off when falling.
  double leaning =
      torso_height / mju_sqrt(torso_height * torso_height + 0.65 * 0.65) - 0.2;
  leaning = mju_max(leaning, 0.3);

  double cp_dx = capture_point[0] - pcp[0];
  double cp_dy = capture_point[1] - pcp[1];
  // Free-standing: penalize forward as strictly as backward (symmetric barrier --
  // there is no table brace to catch a forward fall; a one-sided discount would
  // let the CoM drift forward into nothing, the ~15 deg forward lean that topples).
  double dir_scale_x = 1.0;
  double dir_scale_y = 1.0;
  // STUMBLE Tier A: release the capture/balance barrier during single support so
  // the capture point may legitimately leave the (shrinking) polygon to be caught
  // by the step. Partial (keep ~40%) -- a full release faceplants (see note below).
  double bal_rel = 1.0 - 0.6 * step_active;
  // TROT-WALK: demote the come-to-rest Balance barrier so the velocity-tracking
  // cost can drive the body forward (MJPC walk.cc gates Balance DOWN for
  // locomotion -- it keeps the CoM between the feet, not parked at rest). Demote
  // FORE-AFT (x) ONLY: the lateral (y) axis is the narrow biped's WEAK axis (a
  // global demotion produced metre-scale sideways drift) -- keep FULL lateral
  // balance. ONLY when actually walking (kDesVel != 0); in-place trot (v_des=0)
  // keeps full balance both axes. trot_bal_scale numeric (1=off).
  //
  // ASYMMETRIC BARRIER (2026-06-29): the demotion must NOT weaken the 10x
  // fall-catch (below). Previously the demoted eff_dx fed balance_excursion, so a
  // large forward lean read SMALL -> the amplifier never engaged -> trot-walk
  // pitched forward and faceplanted at ~13 s. Fix: drive balance_excursion (the
  // edge detector) from the UN-demoted eff_dx_full, and fade the demotion back to
  // 1.0 as the lean enters the fall-catch zone (walk_demote = bscale in the
  // walking regime, -> 1.0 at the edge). Non-trot / in-place trot: bscale==1 ->
  // walk_demote==1 -> eff_dx == eff_dx_full (byte-identical to prior residual).
  double bscale = 1.0;
  if (is_trot && (kDesVelX != 0.0 || kDesVelY != 0.0)) {
    int bs_id = mj_name2id(model, mjOBJ_NUMERIC, "trot_bal_scale");
    if (bs_id >= 0) bscale = model->numeric_data[model->numeric_adr[bs_id]];
  }
  // Defensive: keep walk_demote in [bscale,1] (sign-safe). Identity for the
  // documented config (trot_bal_scale in [0,1]); only caps a misconfigured numeric.
  bscale = mju_max(0.0, mju_min(1.0, bscale));
  // FULL (un-demoted) excursion drives the edge amplifier (the fall-catch barrier).
  double eff_dx_full = cp_dx * dir_scale_x * bal_rel;
  double eff_dy = cp_dy * dir_scale_y * bal_rel;
  double balance_excursion =
      mju_sqrt(eff_dx_full * eff_dx_full + eff_dy * eff_dy);
  constexpr double kEdgeInner = 0.05;       // m — amplifier still 1×
  constexpr double kEdgeOuter = 0.10;       // m — amplifier saturated
  constexpr double kEdgePeakAmplifier = 10.0;
  double edge_t =
      mju_min(1.0, mju_max(0.0, (balance_excursion - kEdgeInner) /
                                    (kEdgeOuter - kEdgeInner)));
  double edge_smooth = edge_t * edge_t * (3.0 - 2.0 * edge_t);
  // The 10x edge amplifier is the SUPPORT-POLYGON BARRIER (restoring authority that ramps up as the
  // capture point nears the foot edge) -- NOT a knife-edge bug. Removing it for free-standing (tried
  // 2026-06-05) deleted the balance restoring force -> faceplant in 3s. KEEP it. The correct cause-B
  // is the SYMMETRIC barrier (dir_scale_x = 1.0, above) WITH this amplifier intact.
  double edge_amplifier = 1.0 + (kEdgePeakAmplifier - 1.0) * edge_smooth;
  // Asymmetric demotion: bscale in the WALKING regime (edge_smooth~0), fading to
  // 1.0 (FULL barrier) as the lean enters the fall-catch zone (edge_smooth->1).
  double walk_demote = bscale + (1.0 - bscale) * edge_smooth;
  double eff_dx = eff_dx_full * walk_demote;

  // Per-axis residual: directional scale already includes balance_scale
  // for forward, 1.0 for backward/lateral. NO outer multiplication by
  // balance_scale (would double-count for forward, and incorrectly
  // relax backward/lateral).
  residual[counter + 0] = eff_dx * leaning * edge_amplifier;
  residual[counter + 1] = eff_dy * leaning * edge_amplifier;
  counter += 2;

  // ----- pelvis tilt (free-standing upright hold) ----- //
  double *pelvis_up = SensorByName(model, data, "pelvis_up");
  double pelvis_tilt_residual;
  {
    // FREE-STANDING upright: the legacy residual
    // pelvis_up[2]-1 == cos(tilt)-1 is QUADRATICALLY FLAT near vertical, so a
    // ~17deg counterbalance lean cost almost nothing and the planner parked
    // there (butt-back, arms-forward) on the decoupled twin -> metastable, then
    // toppled. Root-caused 2026-06-05. Replace with sin(tilt) (horizontal
    // magnitude of the pelvis up-vector), which is ~LINEAR near vertical (~6x
    // sharper at 17deg) -> pulls the torso erect. Scaled by the `upright_gain`
    // numeric so it is tunable/reversible WITHOUT a recompile:
    //   gain  > 0 : sharpened linear term * gain  (default 1.0 = the fix)
    //   gain <= 0 : exact legacy cos-flat behaviour (clean A/B baseline)
    double upright_gain = GetNumberOrDefault(1.0, model, "upright_gain");
    if (upright_gain <= 0.0) {
      pelvis_tilt_residual = pelvis_up[2] - 1.0;
    } else {
      double sin_tilt =
          mju_sqrt(mju_max(0.0, 1.0 - pelvis_up[2] * pelvis_up[2]));
      // STRAIGHTEN upright-target ramp (C3): drive the pelvis to vertical along a
      // min-jerk from the captured release tilt, not instantly. sin_target ramps
      // sin(start_tilt) -> 0 over target_ramp_sec; penalize only the tilt BEYOND
      // the moving setpoint, so the corrective force stays small and the body
      // glides erect instead of being FLUNG past vertical (the static-target
      // overshoot: a +10deg lean toppled backward to ~130deg on the twin). After
      // the ramp window sin_target=0 -> identical to the plain upright hold.
      double sin_target = 0.0;
      if (is_straighten && straighten_seeded_) {
        double ramp_dur = (residual_keyframe_.target_ramp_sec >= 0.0)
                              ? residual_keyframe_.target_ramp_sec
                              : kPhaseRampSeconds;
        if (ramp_dur > 1e-9 && time_in_phase < ramp_dur) {
          double ra_lin = mju_min(time_in_phase / ramp_dur, 1.0);
          double ra = ra_lin * ra_lin * (3.0 - 2.0 * ra_lin);
          sin_target = (1.0 - ra) * std::sin(straighten_start_tilt_);
        }
      }
      pelvis_tilt_residual = upright_gain * mju_max(0.0, sin_tilt - sin_target);
    }
  }
  residual[counter++] = (1.0 - 0.5 * step_active) * pelvis_tilt_residual;  // Tier A: relax torso-upright in single support

  // ----- foot up-vectors: prevent ankle roll ----- //
  double *foot_right_up = SensorByName(model, data, "foot_right_up");
  double *foot_left_up  = SensorByName(model, data, "foot_left_up");
  // STUMBLE Tier A (foot-lift fix): release the flat-foot penalty on the SWING
  // foot -- a swinging foot MUST plantarflex/fold (foot_up[2] < 1); the STANCE
  // foot keeps the full weight. THE single biggest anti-lift blocker (named by
  // 5/5 research agents + matches MJPC quadruped's step[foot]?h:0 self-gating).
  residual[counter++] = (1.0 - swing_r) * mju_abs(foot_right_up[2] - 1.0);
  residual[counter++] = (1.0 - swing_l) * mju_abs(foot_left_up[2]  - 1.0);

  // ----- waist yaw: stop planner from yawing torso to swing arm ----- //
  // torso_joint is the 13th actuated DOF (nu index 12), home = 0.
  // Confirmed from ResetLocked joint-order print: qpos index 19 = 7 + 12.
  residual[counter++] = data->qpos[7 + 12] - model->key_qpos[7 + 12];

  // ----- hip yaw + roll: stop the planner exploiting hip rotation to shuffle
  // the CoM (same exploitation mode as waist yaw). Targets are home (key_qpos =
  // facing front) so this drives the legs back to a square stance.
  // Nu indices: L_hip_yaw=0, L_hip_roll=2, R_hip_yaw=6, R_hip_roll=8.
  // Free-standing: unit squaring authority on both hips (the x2/x4/x12 brace and
  // leg-lift scalings of the lean task do not exist here).
  constexpr double hip_square_scale = 1.0;
  constexpr double stance_square_scale = 1.0;
  // ROCK (trot_rock_ff, default 0 = OFF): move the hip-ROLL cost TARGET by the
  // same stance-side weight-shift delta loco::ModifyControl drives, so the
  // weight-30 pin above asks for the rock instead of fighting it. Trot-gated and
  // zero in double support => every non-trot strategy (and trot with the numeric
  // at 0) is byte-identical. See TrotRockHipRoll for signs + the measurement.
  double rock_dl = 0.0, rock_dr = 0.0;
  if (is_trot) {
    TrotRockHipRoll(GetNumberOrDefault(0.0, model, "trot_rock_ff"),
                    GetNumberOrDefault(0.0, model, "trot_rock_lead"), g_amp,
                    rock_ph_l, rock_ph_r, rock_duty, &rock_dl, &rock_dr);
  }
  residual[counter++] = stance_square_scale * (data->qpos[7 + 0] - model->key_qpos[7 + 0]);
  residual[counter++] = stance_square_scale *
      (data->qpos[7 + 2] - (model->key_qpos[7 + 2] + rock_dl));
  residual[counter++] = hip_square_scale * (data->qpos[7 + 6] - model->key_qpos[7 + 6]);
  residual[counter++] = hip_square_scale *
      (data->qpos[7 + 8] - (model->key_qpos[7 + 8] + rock_dr));

  // ----- posture ----- //
  // Phase-scaled: ×3 during stand_up, ×1 for straighten and the stepping keyframes.
  // Why: the Posture cost is the ONLY signal that pulls knee + hip_pitch
  // back to extension. Hip yaw/roll, waist yaw, foot up have dedicated
  // residuals; knee + hip_pitch only get the general 27-dim Posture pull.
  // At weight 0.015 with phase_posture_scale=1 it's too weak — once a knee
  // drifts a few degrees, nothing pulls it back. During stand_up the boost
  // gives Posture 9× more effective cost (quadratic), keeping legs extended.
  mju_sub(&residual[counter], data->qpos + 7, posture_target + 7, model->nu);
  mju_scl(&residual[counter], &residual[counter], phase_posture_scale,
          model->nu);
  // ----- targeted knee + hip-pitch extension anchor (free-standing only) -----
  // The general 27-dim Posture pull is the ONLY signal holding knee + hip_pitch
  // to the keyframe (see comment above), and at its effective weight it is too
  // weak: over the 1s horizon the sampler slowly trades leg extension for a lower
  // CoM and the legs creep open until collapse (the ~30-90s stand/crouch sink,
  // 2026-06-06 research). Amplify ONLY the leg entries (hip_pitch nu-idx 1/7,
  // knee 3/9) inside the Posture residual so the legs hold their keyframe target
  // (straight for stand, 0.7 for crouch) WITHOUT over-constraining the 23 other
  // DOFs -- raising GLOBAL Posture did that and caused the asymmetric crouch.
  // Tunable: <numeric name="leg_extension_gain">
  // (1.0 = off / unchanged; sweep up if it still creeps, down if it over-stiffens).
  {
    double leg_gain = GetNumberOrDefault(2.5, model, "leg_extension_gain");
    // Bent-knee holds (crouch/squat) need a STRONGER symmetric leg anchor than
    // straight-knee holds (stand/arms). At the live plan rate the saddle-unstable
    // symmetric crouch breaks into an asymmetric one-leg PROP — one knee drives to
    // its -0.12 ctrl floor while the other over-flexes — unless the bilateral
    // knee+hip-pitch pull is stiff enough to outweigh the balance benefit of
    // propping. Gate on a bent TARGET knee (posture_target L/R knee, nu-idx 3/9 =
    // qpos 7+3 / 7+9, > 0.3 rad) so straight-knee strategies hit neither branch and
    // stay byte-identical to the accepted stand/arms (NO regression). Tunable.
    if (posture_target[7 + 3] > 0.3 || posture_target[7 + 9] > 0.3)
      leg_gain = GetNumberOrDefault(6.0, model, "crouch_leg_extension_gain");
    for (int li : {1, 3, 7, 9}) residual[counter + li] *= leg_gain;
  }
  counter += model->nu;

  // com vel
  double *waist_lower_subcomvel =
      SensorByName(model, data, "waist_lower_subcomvel");
  double *torso_velocity = SensorByName(model, data, "torso_velocity");
  double com_vel[2];
  mju_add(com_vel, waist_lower_subcomvel, torso_velocity, 2);
  mju_scl(com_vel, com_vel, 0.5, 2);

  // ----- move feet ----- //
  double *foot_right_vel = SensorByName(model, data, "foot_right_vel");
  double *foot_left_vel = SensorByName(model, data, "foot_left_vel");
  double move_feet[2];
  mju_copy(move_feet, com_vel, 2);
  mju_addToScl(move_feet, foot_right_vel, -0.5, 2);
  mju_addToScl(move_feet, foot_left_vel, -0.5, 2);

  mju_copy(&residual[counter], move_feet, 2);
  mju_scl(&residual[counter], &residual[counter], leaning, 2);
  counter += 2;

  // ----- control ----- //
  mju_sub(&residual[counter], data->ctrl, posture_target + 7,
          model->nu);  // because of pos control (tracks pose-library key)
  counter += model->nu;

  // ----- foot stability: restoring force toward home XY position ----- //
  // Position-based (not velocity-based) so there's a continuous gradient even
  // when the foot is stationary but displaced. Home positions taken from the
  // actual home-pose foot xipos (= body inertial frame, which is what the
  // `framepos objtype="body"` sensor returns — NOT the kinematic xpos). At
  // the H1-2 home keyframe the ankle_roll_link inertial offset places the
  // foot COM at x = 0.2196, not pelvis_x = 0.19. The old 0.19 constant
  // (copied from the pelvis qpos) made the residual +0.03 at home, adding
  // a continuous 3 cm backward pull on both feet.
  static constexpr double kRightFootHomeXY[2] = {0.2196, -0.163};
  static constexpr double kLeftFootHomeXY[2]  = {0.2196,  0.163};
  // Loco: both feet are always planted-anchored at unit scale (the lean task's
  // brace / leg-lift scalings and its 8 cm forward brace stance do not exist here).
  constexpr double kFootHomeX = kRightFootHomeXY[0];
  // ★ MEASURED FOOT ANCHOR for the STEPPING keyframes (trot/walk/drive), 2026-07-20.
  // kRight/kLeftFootHomeXY are ODOMETRIC-frame constants; that frame drifts (~13 m
  // measured on real), so this cost drags the feet toward a point metres away and
  // the robot "keeps trying to lean backward". The lower-body stand fixed exactly
  // this on 2026-07-18 (Foot Stability cost 225 -> 0.00 on real) but the full-body
  // LEAN task never got the port. drive_foot_anchor_ is re-pinned to the MEASURED
  // stance by TransitionLocked (stepping-keyframe entry + every idle tick), never
  // mid-gait. is_trot gate => STRAT 6 STAND AND EVERY NON-STEPPING STRATEGY ARE
  // BYTE-IDENTICAL (they keep kFootHomeX / the home constants), and the anchor's
  // own defaults equal the old constants so even the stepping path is inert until
  // TransitionLocked pins it.
  // ★ DRIVE-ONLY + OPT-IN (narrowed 2026-07-20 after a REAL-ROBOT FAULT: the
  // is_trot-gated, always-on version dragged the robot sideways at startup --
  // the cold-start pin only null-checked sensor POINTERS, so an uninitialised
  // estimate could pin both anchors near the origin and a weight-8 cost then
  // hauled both feet toward it). TROT 22/23 also excluded on their own merits:
  // measured stance x=0.030 vs the constant 0.2196 would move their anchor 19 cm
  // and they were twin-TUNED against the constant (a pp7 spot-check gave 4.1 s
  // vs a historical ~5.1 s, which this bench's non-deterministic CEM cannot
  // resolve at n=3). Default numeric 0 => strat 6 AND drive are byte-identical.
  const bool anchor_measured =
      is_drive && GetNumberOrDefault(0.0, model, "drive_foot_anchor") > 0.5;
  const double rf_ax = anchor_measured ? drive_foot_anchor_[0] : kFootHomeX;
  const double rf_ay = anchor_measured ? drive_foot_anchor_[1] : kRightFootHomeXY[1];
  const double lf_ax = anchor_measured ? drive_foot_anchor_[2] : kFootHomeX;
  const double lf_ay = anchor_measured ? drive_foot_anchor_[3] : kLeftFootHomeXY[1];
  residual[counter++] = foot_right_pos[0] - rf_ax;
  residual[counter++] = foot_right_pos[1] - rf_ay;
  residual[counter++] = foot_left_pos[0] - lf_ax;
  residual[counter++] = foot_left_pos[1] - lf_ay;

  // ----- Joint Velocity Limits ------------------------------------------ //
  // One-sided residual that's zero when |qvel| ≤ 0.85·ω_max and grows
  // linearly beyond. ω_max values are the per-joint velocity limits from
  // the H1-2 URDF (h1_2.urdf <limit velocity="..."/>). Stops MJPC from
  // planning trajectories that demand impossible joint speeds — important
  // for sim2real because a plan that exceeds ω_max will fail on hardware
  // regardless of how good the torque is. 0.85 leaves a 15% safety buffer.
  //
  // Index order = actuator order = the ctrlrange order in h1_2_pos.xml.
  // Same indices work for both H1-2 and H1-2-Hands because both variants
  // expose only these 27 actuated joints (fingers are unactuated in our
  // Hands MJCF).
  // DEPLOYED-ENVELOPE PARITY (2026-06-12): the binding limit on hardware is the
  // SAFETY LAYER's velocity estop (default_safety_full.yaml velocity_ratio x URDF,
  // h12_safety_layer/core/joint_limits.py), which latches and cuts ALL motors --
  // three runs in a row died to mid-recovery arm/leg swings the old URDF-only
  // numbers considered free (RshP estop 4.5 rad/s vs old threshold 0.85*9=7.65).
  // Each entry = min(URDF, estop): hips .20/.25/.20*23, knee .60*14, ankles
  // 3.0*9 (URDF binds), torso .15*23, sho_p/r .50*9, sho_y .50*20, elbow .30*20,
  // wrists .30*31.4. The 0.85 factor below keeps a 15% buffer UNDER the estop.
  static constexpr double kJointVelLimit[27] = {
      // L_hip_yaw, L_hip_pitch, L_hip_roll, L_knee, L_ank_p, L_ank_r
      4.6, 5.75, 4.6, 8.4, 9.0, 9.0,
      // R_hip_yaw, R_hip_pitch, R_hip_roll, R_knee, R_ank_p, R_ank_r
      4.6, 5.75, 4.6, 8.4, 9.0, 9.0,
      // torso
      3.45,
      // L_sho_p, L_sho_r, L_sho_y, L_elbow, L_wr_r, L_wr_p, L_wr_y
      4.5, 4.5, 10.0, 6.0, 9.42, 9.42, 9.42,
      // R_sho_p, R_sho_r, R_sho_y, R_elbow, R_wr_r, R_wr_p, R_wr_y
      4.5, 4.5, 10.0, 6.0, 9.42, 9.42, 9.42,
  };
  // Walk actuators (always 27 here) and look up each joint's qvel index via
  // jnt_dofadr — robust to the Hands variant inserting finger joints later
  // in the qvel layout. MuJoCo stores the actuator's transmission target as
  // actuator_trnid[2*i] (first slot is the joint id for joint-type
  // transmissions, which is all of ours).
  for (int i = 0; i < model->nu && i < 27; i++) {
    int jntid  = model->actuator_trnid[2 * i];
    int dofadr = model->jnt_dofadr[jntid];
    double abs_vel = std::abs(data->qvel[dofadr]);
    double threshold = 0.85 * kJointVelLimit[i];
    residual[counter++] = std::max(0.0, abs_vel - threshold);
  }

  // ----- Body Yaw (xy-projected) ---------------------------------------- //
  // The floating-root quaternion has no direct cost in any joint-level
  // residual, so the planner can rigidly yaw the body via foot-on-floor pivot
  // without paying (Waist Yaw + Hip Yaw target joint angles; whole-body
  // rotation leaves those at home). This residual is yaw-only: project
  // torso_forward onto the xy-plane and compare with the HEADING TARGET
  // drive_yaw_des_ -- the commanded heading for drive (integrated yaw-rate),
  // or the heading latched when the gait armed for the plain trot
  // (TransitionLocked). Pitch drops out because pitching shortens
  // torso_forward's xy length without changing its xy direction.
  // ★ STEPPING HEADING LOCK (trot 23 + drive 24, bugfix 2026-07-12): the lean
  // task aimed this at a RANDOMISED reach target (+/-11 deg of heading bias,
  // re-rolled per process), which made a trot-walk crab and gave the in-place
  // trot its lateral wander. Loco has no reach target: the heading target is
  // the only reference. Non-stepping strategies write 0 (their JSON weight is 0).
  {
    double body_yaw_residual = 0.0;
    if (is_trot) {
      double *torso_forward = SensorByName(model, data, "torso_forward");
      double tf_xy_len = mju_sqrt(torso_forward[0] * torso_forward[0] +
                                  torso_forward[1] * torso_forward[1]);
      double yaw_alignment = (tf_xy_len > 1e-6)
          ? (torso_forward[0] * std::cos(drive_yaw_des_) +
             torso_forward[1] * std::sin(drive_yaw_des_)) / tf_xy_len
          : 1.0;
      body_yaw_residual = 1.0 - yaw_alignment;
    }
    residual[counter++] = body_yaw_residual;
  }

  // ----- Knees Straight: STUMBLE bent-knee floor (anti-strut) ------------- //
  double left_knee_angle  = data->qpos[7 + 3];
  double right_knee_angle = data->qpos[7 + 9];
  if (is_stumble) {
    // STUMBLE: a BENT-KNEE FLOOR -- forbid
    // EITHER knee from diving toward the straight/locked stop (the strut), in
    // EVERY phase (calm OR stepping), so the one-leg passive prop is impossible
    // WHENEVER. One-sided max(0, floor - knee): a knee at the 0.35 march pose,
    // or folded UP for a swing, is FREE (>= floor); only a knee locking toward
    // the floor is penalised. UNLIKE the Symmetry term this is NOT g_amp-gated,
    // so even a sustained step cannot strut; and the swing leg rises ABOVE the
    // floor so the gait is untouched. Strength = the "Knees Straight" JSON
    // weight (0 = off, tunable live). Pairs with the (g_amp-gated) Symmetry term
    // above: Symmetry keeps L/R matched when calm, this floor blocks the lock
    // always. Non-stumble strategies fall through to the else (0,0) -> byte-id.
    constexpr double kKneeFloor = 0.15;  // rad; below this a knee is locking -> strut
    residual[counter++] = mju_max(0.0, kKneeFloor - left_knee_angle);
    residual[counter++] = mju_max(0.0, kKneeFloor - right_knee_angle);
  } else {
    residual[counter++] = 0.0;
    residual[counter++] = 0.0;
  }

  // ----- Leg LEFT/RIGHT symmetry (kill the one-knee strut) -------------- //
  // The recurring free-standing failure (stand/crouch/squat) is a one-leg
  // "strut": one knee locks near its lower stop as a passive support while the
  // other balances, then both sink and the robot topples (~70-160 s live). By
  // construction the strut is a large LEFT/RIGHT difference in the sagittal leg
  // joints (knee + hip-pitch). No existing term penalises that DIFFERENCE:
  // Posture penalises each leg's ABSOLUTE deviation from the keyframe, so a
  // matched bent stance and a strut can cost nearly the same in Posture while
  // the strut wins on Balance (offloading one leg to a free passive column is
  // genuinely lower-cost). This quadratic penalty on (L - R) makes the
  // asymmetric strut strictly more expensive than a symmetric stance WITHOUT
  // pinning the absolute pose: the legs may still flex *together* for balance,
  // and tiny micro-asymmetries stay nearly free (quadratic gradient -> 0 at 0)
  // so it does not over-constrain the lateral ankle/hip balance.
  //   knee_L = qpos[7+3], knee_R = qpos[7+9]; hipPitch_L = qpos[7+1],
  //   hipPitch_R = qpos[7+7] (verified vs body-tree joint order + Knees
  //   Straight above). Only the sagittal pair: ankle is the primary balance
  //   actuator (asymmetry there is normal), and hip-roll/yaw are mirror-signed
  //   (the widened stance is +/-0.12), so neither belongs in an (L - R) term.
  // With the XML default weight 0 the term is OFF for every strategy that does
  // not opt in via its JSON weight map ("Symmetry": w).
  {
    // STUMBLE (strat 20): RELEASE symmetry WHILE stepping. A catch-step is an
    // INTENTIONAL L/R asymmetry (one foot swings) so the symmetry term must not
    // fight it; when calm (g_amp=0) symmetry is FULL, making stumble's bent-knee
    // stand as strut-proof as the strat-6 stand (which runs Symmetry 200 and
    // never struts). g_amp is 0 for EVERY non-stumble strategy (set only inside
    // the is_stumble stepping block far above), so sym_gate == 1.0 there ->
    // stand/crouch/jab/reach stay byte-identical. This is why the user's strat-20
    // struts and strat-6 doesn't: stumble had Symmetry=0 (it conflicts with
    // stepping); gating restores it for the calm phase the march-gate now holds.
    double sym_gate = 1.0 - g_amp;
    residual[counter++] = sym_gate * (data->qpos[7 + 3] - data->qpos[7 + 9]);  // knee L-R
    residual[counter++] = sym_gate * (data->qpos[7 + 1] - data->qpos[7 + 7]);  // hipPitch L-R
    // anklePitch L-R: the SAGITTAL ankle. joint_forensics on the cold-start
    // backward fall (strat 19) showed the one-leg strut hides HERE — L/R
    // ankle_pitch diverged 13.9deg while knee diverged only 5.2deg. The
    // original term excluded "ankle" reasoning about the LATERAL ankle_roll
    // (asymmetry there is normal); ankle_PITCH is fore/aft and must stay
    // symmetric for a square stance, so penalising its (L-R) closes the
    // strut's last escape. Quadratic => tiny asymmetries stay ~free.
    residual[counter++] = sym_gate * (data->qpos[7 + 4] - data->qpos[7 + 10]); // anklePitch L-R
  }

  // ----- Base-height anchor (free-standing anti-sink) ------------------- //
  // After the strut is removed (Symmetry), the residual failure is a SYMMETRIC
  // CoM SINK: the planner lowers the base for stability (Balance rewards a low
  // CoM) with nothing holding standing height, so stand slowly crouches and
  // topples (~76s live), and the squat sinks instead of re-rising on its
  // ascent. Anchor the base height: penalise the base dropping BELOW the active
  // phase keyframe's base z (posture_target[2]) -- a CoM-height proxy that does
  // NOT freeze individual joints (the planner picks HOW to hold height), unlike
  // stiffening the knee posture (leg_extension_gain 5.5 OVER-stiffened and broke
  // stand in the headless gate). One-sided (max(0, target - z)): only the SINK
  // is penalised, never being tall, so it adds no cost while held. Scaled x10 so
  // a metre-deviation maps to a joint-angle-like residual (sane JSON weights).
  //   STAND: posture_target[2] = home key z = 1.028 = correct standing height.
  //   CROUCH/SQUAT: all keyframes share the nominal base z 1.028 (the real
  //   bent-leg height comes from FK, not qpos[2]), so this anchor is WRONG for
  //   them -> kept OFF via JSON weight 0 (crouch already holds; squat-ascent
  //   needs a per-phase FK-derived target, a follow-on).
  {
    double bh_err = posture_target[2] - data->qpos[2];
    residual[counter++] = 10.0 * mju_max(0.0, bh_err);   // one-sided anti-sink
  }

  // ----- Centroidal angular momentum (hip / Horak-Nashner strategy) ----- //
  // Regulating whole-body angular momentum about the CoM toward zero gives the
  // sampling planner the active whole-body ("hip") balance strategy that the
  // ankle (capture-point Balance) and the static L/R Symmetry term do not: it
  // penalises the TRANSIENT rotational mode that lets an incipient tip grow into
  // the one-leg passive-column "strut", so the planner instead nulls momentum
  // with a coordinated hip/trunk rotation and can RETRIEVE to symmetric when
  // nudged. Pairs with Balance (linear-momentum/ankle) and Symmetry (static
  // pose) -- it is the dynamic rotational complement, NOT a replacement.
  // SAMPLING-LEGAL: penalises only the STATE L_cm over the rollout horizon --
  // no feedback law, no Jacobian, no dL/dt. data->subtree_angmom is ALREADY
  // populated by the forward pass: mj_sensorVel calls mj_subtreeVel because the
  // model has <subtreelinvel> sensors (the Balance term above reads
  // torso_subtreelinvel), and that one call fills subtree_linvel AND
  // subtree_angmom for every body. So the read is O(0) and thread-safe
  // (read-only of this rollout thread's own mjData; NO mj_forward). "pelvis" is
  // the floating-base root body, so its subtree == the whole robot and
  // subtree_angmom[3*pelvis] == the centroidal angular momentum [Lx,Ly,Lz]
  // about the whole-body CoM (world frame, kg.m^2/s). Scaled x0.1 so a fall-rate
  // momentum (~10) maps to an O(1) residual. Carrier weight in the XML; the
  // strategy JSON key "Angular Momentum" sets it per phase.
  {
    int pelvis_id = mj_name2id(model, mjOBJ_BODY, "pelvis");
    if (pelvis_id < 0) pelvis_id = 1;  // floating-base root fallback (always exists)
    const mjtNum *angmom = data->subtree_angmom + 3 * pelvis_id;
    // HIP/ARM RECOVERY tier (Strategy 20): when the capture point excurses, target
    // COUNTER angular momentum (throw torso+arms OPPOSITE the fall) instead of zero
    // -> the flat-footed ankle->HIP->step INTERMEDIATE tier that hauls the CoM back
    // BEFORE a step is needed, WITHOUT rocking onto the toe/heel (the CoP/ankle
    // strategy we deliberately avoid). Zero target when calm == the original
    // regulate-to-zero (unchanged for every other strategy). Fore-aft capture
    // excursion -> pitch momentum (Ly); lateral -> roll (Lx); yaw (Lz) always ->0.
    // Gains are numerics (default 0 = OFF until tuned on the twin; SIGN found there).
    double Lx_tgt = 0.0, Ly_tgt = 0.0;
    if (is_stumble) {
      int rpg_id = mj_name2id(model, mjOBJ_NUMERIC, "recover_pitch_gain");
      int rrg_id = mj_name2id(model, mjOBJ_NUMERIC, "recover_roll_gain");
      double kRecPitch = (rpg_id >= 0)
          ? model->numeric_data[model->numeric_adr[rpg_id]] : 0.0;
      double kRecRoll = (rrg_id >= 0)
          ? model->numeric_data[model->numeric_adr[rrg_id]] : 0.0;
      Ly_tgt = kRecPitch * g_cap_ex;   // fore-aft fall -> pitch counter-momentum
      Lx_tgt = kRecRoll  * g_cap_ey;   // lateral fall  -> roll counter-momentum
    }
    // TROT-WALK PITCH/YAW CATCH (2026-06-29, the "active torso-pitch catch"): the
    // forward-walk failure is a forward-PITCH runaway -- angmom[1] is the lateral-
    // axis (pitch) centroidal angular momentum, so penalising it RESISTS the
    // topple. Crucially angmom is RATE-like: a steady walking lean has ~0 net pitch
    // momentum (it oscillates fore-aft and cancels) so this does NOT tax walking,
    // but an ACCELERATING topple spikes angmom[1] -> caught. angmom[2] (yaw) damps
    // the heading wander seen in walk. ACTIVE only for trot-WALK (is_trot &&
    // v_des!=0) via trot_angmom_w; 0 for EVERY other strategy (incl. in-place trot,
    // strat 20, stand/crouch/arms) so they stay byte-identical even though the XML
    // "Angular Momentum" weight is now a nonzero carrier (cost = w * Norm(0) = 0).
    double angmom_w = 0.0;
    if (is_trot && (kDesVelX != 0.0 || kDesVelY != 0.0)) {
      int aw_id = mj_name2id(model, mjOBJ_NUMERIC, "trot_angmom_w");
      angmom_w = (aw_id >= 0) ? model->numeric_data[model->numeric_adr[aw_id]] : 1.0;
    }
    // ---- ★ STUMBLE HIP/ARM MOMENTUM TIER (stumble_angmom_w, default 0 = OFF)
    // 2026-09-08. THE CARRIER WAS NEVER CONNECTED FOR STRATEGY 20. The
    // counter-momentum target above (recover_pitch_gain / recover_roll_gain) is
    // gated `is_stumble`, but the WEIGHT that multiplies it, angmom_w, was only
    // ever set for trot-WALK and for jump -- so for strat 20 the whole residual
    // was `0.0 * 0.1 * (angmom - Ltgt)` == 0 regardless of the gains. The XML
    // comment on recover_pitch_gain records a 2026-06-26 sweep ("tested +10
    // (2/6) and +30 (6/12) = NO robust catch-step fix ... was variance") -- that
    // sweep moved a number that could not reach the cost. It is an ACHIEVED-ZERO
    // null, exactly like the two void stagger tests of 2026-09-07, and it says
    // nothing about the mechanism. Treat the tier as UNTESTED.
    //
    // WHY THIS TIER AND NOT ANOTHER KNOB. The 2026-09-07 CoP measurement closed
    // the ankle/geometry tier: at fore/aft collapse the centre of pressure sits
    // at 110% of the heel (back) / 100% of the toe (fwd) while the ankle is at
    // only 59%/85% of its rating and never exceeds 90%. The GROUND-REACTION
    // authority is exhausted; the ACTUATORS are not. Taking moments about the
    // contact point,
    //     xddot_com = (g/z)*(x_com - x_cop)  -  Ldot_y / (m*z)
    // the first term is what saturates at the polygon edge and the second is
    // free: centroidal angular momentum RATE shifts the effective pivot OUTSIDE
    // the support polygon, which is the one thing left that does not need a step,
    // a weight transfer, or a different stance. This is the "ankle -> HIP ->
    // step" middle rung the header comments have described as unbuilt all along.
    // SIGN (to be confirmed on the twin, both signs are swept): a FORWARD fall is
    // ex > 0, needs xddot_com > 0, hence Ldot_y > 0, hence a POSITIVE pitch
    // target -- so recover_pitch_gain > 0, with Ly_tgt = k * ex as coded above.
    //
    // Scoped `is_stumble && !is_trot` so strategies 23/24 (in-place trot, which
    // are also is_stumble) keep angmom_w == trot_angmom_w exactly. Default 0.0
    // => byte-identical to the validated 2026-09-06 push-recovery until swept.
    if (is_stumble && !is_trot) {
      angmom_w = mju_max(angmom_w,
                         GetNumberOrDefault(0.0, model, "stumble_angmom_w"));
    }
    residual[counter++] = angmom_w * 0.1 * (angmom[0] - Lx_tgt);
    residual[counter++] = angmom_w * 0.1 * (angmom[1] - Ly_tgt);
    residual[counter++] = angmom_w * 0.1 * angmom[2];
  }

  // ----- Lateral CoM centering (the frontal-plane balance Balance OMITS) ----- //
  // ROOT CAUSE of the systematic rightward lean (2026-06-09 live trace): the
  // capture-point Balance term projects the capture point onto the support
  // polygon, which for free-standing is the {L_foot, R_foot} LINE. A lateral
  // CoM shift TOWARD a foot stays ON that line -> projects to itself -> residual
  // ~ 0 -> UNPENALISED. So Balance constrains fore-aft excursion and polygon
  // EXIT, but lateral CoM CENTERING between the feet is structurally free. A
  // sub-threshold sideways seed then drifts (unopposed) until the CoM sits over
  // one foot: that leg loads and its knee buckles while the unloaded leg locks
  // straight into the passive "strut" -- a positive feedback that the live trace
  // shows as roll +0.6 -> +3 deg with the right hip_roll torque 2.3x the left.
  // This term closes that gap: penalise the torso-subtree CoM's LATERAL (y)
  // offset from the foot midpoint (same CoM proxy + foot sensors Balance already
  // uses, all fetched above). Quadratic -> gradient 0 at centre, so it opposes a
  // SUSTAINED drift without fighting a legitimate transient lateral push-recovery
  // or the normal ankle/hip-roll micro-corrections. x10 so a ~metre maps to a
  // joint-angle-like residual (sane JSON weights, matches Base Height). XML
  // default weight 0 -> OFF unless a strategy opts in via JSON "Lateral Center".
  {
    double midfoot_y = 0.5 * (foot_left_pos[1] + foot_right_pos[1]);
    // STUMBLE: rock the CoM toward the STANCE foot as the other foot swings --
    // you cannot raise a LOADED foot, so the weight must shift off it first.
    // (g_bump_r - g_bump_l) is -bump when the LEFT foot swings (shift CoM to the
    // -y RIGHT stance foot) and +bump when the RIGHT swings; 0 in double support
    // (stay centred). This gait-synced weight-shift reference is what converts
    // the folded-swing-leg Posture into a REAL step; the planner finds the hip-
    // roll / ankle to achieve it. Only for stumble; every other strategy keeps
    // the plain centred target (and weight 0 unless it opts in). --- //
    // kLatShift scaled by the trot-window swing scale: a bigger STEP needs a
    // bigger weight-SHIFT to fully unload the swing foot (the measured one-sided
    // step came from too small a rock -- the left foot never unloaded, so it could
    // not lift). At swing_scale 2.5 the rock target is ~0.18 m, enough to take the
    // load off EITHER foot in turn. ==base 0.07 outside the trot window (quiet
    // stand + push-recovery byte-identical). Capped so it can't drive the CoM off
    // the support polygon. --- //
    constexpr double kLatShift = 0.07;   // base CoM lateral rock amplitude [m]
    double lat_amp = mju_min(0.20, kLatShift * trot_swing_scale);
    // prep_rock pre-positions the CoM over the stance foot BEFORE the swing is
    // commanded (0 unless catch_prep_* are set). Same sign convention as the
    // step's own rock, and the sum is capped at the same 0.20 m so the two
    // together can never drive the CoM off the support polygon.
    double rock_cmd = is_stumble
        ? mju_max(-0.20, mju_min(0.20,
              lat_amp * g_amp * (g_bump_r - g_bump_l) + prep_rock))
        : 0.0;
    double lat_tgt = midfoot_y + rock_cmd;
    // COMMIT-THEN-TRACK (trot_com_commit, 2026-10-14): once a step has been
    // committed, the lateral target is the STANCE foot's measured y (plus a small
    // inward offset toward the midline) for the WHOLE step, scaled by the gait
    // amplitude -- not a bump around the midfoot. The sampler then only has to
    // TRACK a fixed target (what it does well in the stand) rather than discover
    // the weight transfer. Flips to the other foot at the next swing-start edge.
    if (is_trot && com_ref_ok_ && com_ref_stance_ >= 0 &&
        GetNumberOrDefault(0.0, model, "trot_com_commit") > 0.5) {
      const double kIn = GetNumberOrDefault(0.02, model, "trot_com_commit_in");
      const double other_y = (com_ref_stance_ == 0) ? foot_right_pos[1]
                                                    : foot_left_pos[1];
      const double dir = (other_y > com_ref_p_[1]) ? 1.0 : -1.0;
      const double y_commit = com_ref_p_[1] + dir * kIn;
      const double a = mju_min(1.0, mju_max(0.0, g_amp));
      lat_tgt = (1.0 - a) * lat_tgt + a * y_commit;
    }
    residual[counter++] = 10.0 * (subcom[1] - lat_tgt);
  }

  // ============ STUMBLE: Cartesian Gait + Step Place refinement ============ //
  // Strategy 20 "stumble" stepping. The gait clock + swing-leg JOINT reference
  // that BOOTSTRAP the step (via the Posture cost) were computed near the TOP of
  // this function; here we add the two Cartesian refinement terms that ground
  // the motion in task space, reusing g_amp / g_bump_l / g_bump_r. Both are the
  // LAST two <user> sensors (Gait dim 2, Step Place dim 4), XML default weight 0,
  // and write 0 for every non-stumble strategy -> strategies 0-19 stay byte-
  // identical (zero residual AND zero weight => zero cost), exactly the Symmetry/
  // Angular Momentum/Lateral Center opt-in pattern above.
  //
  // WHY STEPPING (not more ankle gain): the documented H1-2 limiter is ankle
  // under-authority -- the ankle saturates once the CoP reaches the support-
  // polygon edge, so a large capture-point excursion is UNRECOVERABLE by ankle
  // torque (the ~56%-marginal stand / one-leg-strut ceiling every other lean
  // strategy hits). Stepping RELOCATES the support polygon under the falling CoM
  // (N-step capturability, Koolen/Pratt), extending the recoverable region far
  // beyond the ankle limit -- the gradient-free win for a weak-ankle robot.
  if (is_stumble) {
    // Cartesian step height TARGET: kStepHeight (0.022) scaled by the trot-window
    // swing scale (default 1 outside the window). At swing_scale 2.5 the target is
    // ~0.055 m -- between the quadruped trot (0.03) and Unitree humanoid walk
    // (0.08) clearance, sized for the bigger H1-2 foot.
    double amp = kStepHeight * g_amp * trot_swing_scale;   // ramped Cartesian step height [m]
    // --- Gait foot-height (dim 2): reinforce the SWING foot rising its bump in
    //     task space; STANCE foot untracked (residual 0). Ground reference = the
    //     lower (planted) foot, so it auto-calibrates with no hard-coded foot z.
    //     trot_gait_wscale boosts this residual in the trot window -> the foot-
    //     clearance cost DOMINATES (the references' #1 lesson), so the planner
    //     actually lifts the foot instead of only rocking. ==1 outside the window
    //     -> quiet stand + push-recovery catch-step byte-identical. --- //
    double ref_z = mju_min(foot_left_pos[2], foot_right_pos[2]);
    // CONTACT-SCHEDULE signal (the references' bilateral-alternation glue, e.g.
    // Unitree's `contact` reward): a foot must be UNLOADED during its swing phase.
    // Foot-height alone gave ONE-SIDED steps -- the rock unloaded only the lighter
    // foot; the other stayed planted (loaded) so its height term just paid a fixed
    // penalty. Penalising the swing foot's CONTACT FORCE forces the planner to
    // shift the CoM far enough to take the load off EITHER foot in turn. Per-foot
    // normal force summed from the live contacts; normalised by ~half body weight
    // (~500 N) to sit at the same ~0.05 scale as the height target. Trot-window
    // only (sched=0 when trot_gait_wscale==1) -> quiet stand + push-recovery
    // byte-identical. Costs ~ncon iterations, stumble + trot only. --- //
    double ff_l = 0.0, ff_r = 0.0;
    // STUMBLE Tier A: contact-schedule unload is ALWAYS-ON while stepping (was
    // trot-window-only) -- you cannot lift a LOADED foot, so penalise the swing
    // foot's contact force during its scheduled swing to force the weight-transfer.
    bool sched_on = (g_amp * mju_max(g_bump_l, g_bump_r) > 0.01) || (trot_gait_wscale > 1.01);
    if (sched_on) {
      int fl_body = mj_name2id(model, mjOBJ_BODY, "left_ankle_roll_link");
      int fr_body = mj_name2id(model, mjOBJ_BODY, "right_ankle_roll_link");
      for (int c = 0; c < data->ncon; c++) {
        const mjContact& con = data->contact[c];
        int ba = model->geom_bodyid[con.geom1], bb = model->geom_bodyid[con.geom2];
        mjtNum f6[6];
        mj_contactForce(model, data, c, f6);
        double fn = mju_abs(f6[0]);  // normal force (contact-frame x)
        if (ba == fl_body || bb == fl_body) ff_l += fn;
        if (ba == fr_body || bb == fr_body) ff_r += fn;
      }
    }
    constexpr double kSched = 0.12;  // 0.06->0.12 (Tier A): stronger swing-load penalty to force the unload
    double sched_l = sched_on ? kSched * g_amp * g_bump_l * mju_min(2.0, ff_l / 500.0) : 0.0;
    double sched_r = sched_on ? kSched * g_amp * g_bump_r * mju_min(2.0, ff_r / 500.0) : 0.0;
    // SUBTRACT the contact penalty so it REINFORCES the height deficit (a planted
    // swing foot has a NEGATIVE height residual -amp*g_bump; a loaded one makes it
    // MORE negative -> larger cost -> forces the unload). Adding them cancels (the
    // bug that left it one-sided). A correctly-lifted unloaded foot: height~0 and
    // ff~0 -> residual~0 (no penalty).
    // STUMBLE Tier A: ONE-SIDED clearance (MJPC quadruped Scramble pattern) -- only
    // a swing foot BELOW its scheduled height is penalised; OVER-clearing is FREE,
    // so the planner is never punished for lifting higher. -sched forces the unload.
    residual[counter++] = trot_gait_wscale *
        ((g_bump_l > 0.0) ? mju_min(0.0, (foot_left_pos[2]  - ref_z - amp * g_bump_l) - sched_l) : 0.0);
    residual[counter++] = trot_gait_wscale *
        ((g_bump_r > 0.0) ? mju_min(0.0, (foot_right_pos[2] - ref_z - amp * g_bump_r) - sched_r) : 0.0);

    // --- Step Place (dim 4): aim each SWING foot's xy at a capture-point /
    //     Raibert target: x_foot = com_xy + nominal_offset
    //                            + (com_vel - v_des)*sqrt(z/g) + v_des*T_stance.
    //     v_des=0, balanced -> nominal stance (march in place); a push -> target
    //     shifts into the push -> recovery STEP; v_des>0 -> feet step in the
    //     commanded direction -> walk (the nav hook). Stance foot untracked. --- //
    int pelvis_id = mj_name2id(model, mjOBJ_BODY, "pelvis");
    if (pelvis_id < 0) pelvis_id = 1;  // floating-base root fallback (always exists)
    const mjtNum *com_pos = data->subtree_com + 3 * pelvis_id;     // whole-body CoM xyz
    double const *com_vel = SensorByName(model, data, "waist_lower_subcomvel");
    double z_com     = mju_max(0.5, com_pos[2]);
    double tau_cap   = mju_sqrt(z_com / 9.81);     // sqrt(z/g): capture-point time const
    // CLAMP the capture offset to a physical step reach. Unclamped, an incipient
    // topple spikes com_vel -> the foot target explodes metres away -> the swing
    // leg FLINGS to chase it (seen as 0.5 m foot-height spikes that then caused
    // the fall). A real step can only reach ~+/-0.22 m, so clamp there: the term
    // still biases the step toward the capture point for recovery, but can never
    // command an unreachable lunge.
    constexpr double kStepReach = 0.30;   // max single-step xy reach [m] (0.22->0.30
                                          // 2026-06-18: bigger CATCH-STEP for push
                                          // recovery; forward shoves need a longer
                                          // reach to plant the foot ahead of the CoM)
    // PURE velocity-error catch (NO neutral feedforward). The Raibert neutral
    // +v_des*t_stance places the swing foot AHEAD, which pushes the body BACKWARD
    // near rest -- it only "maintains" a velocity the body already has, so without
    // an independent propulsion source it backfires (verified: it produced a steady
    // backward bias that only lost to the velocity-track cost at a destabilising
    // weight). Propulsion now comes from the CoM-Vel tracking cost; placement only
    // REGULATES, so error-catch alone is correct. v_des=0 (all non-trot, incl.
    // strat 20) => identical to before (the neutral term was already 0 there).
    double off_x = (com_vel[0] - kDesVelX) * tau_cap;
    double off_y = (com_vel[1] - kDesVelY) * tau_cap;
    off_x = mju_max(-kStepReach, mju_min(kStepReach, off_x));
    off_y = mju_max(-kStepReach, mju_min(kStepReach, off_y));
    // nominal foot offsets from the CoM at the standing stance (measured ~0.064 m
    // fore, +/-0.26 m lateral). Biased to 0.09 fore so the feet land slightly
    // AHEAD of the CoM -- countering the documented H1-2 forward CoM lean (~3-4 cm
    // ahead) that topples the static stand. CoM-relative so the stance translates
    // when walking.
    // Fore-aft nominal stance offset: feet planted this far AHEAD of the CoM.
    // +0.13 gave forward-push margin back when forward was the weak axis -- but it
    // also CANCELS the backward catch-step's negative off_x (the rear foot only
    // reaches TO the CoM, never behind it -> can't catch a backward fall). Now
    // that the omnidirectional catch-step makes forward solid, REBALANCE this
    // toward centre so backward gets margin too. Model numeric `stance_off_x`
    // (default 0.13) -> tune live, no rebuild.
    int sox_id = mj_name2id(model, mjOBJ_NUMERIC, "stance_off_x");
    double kStanceOffX = (sox_id >= 0)
        ? model->numeric_data[model->numeric_adr[sox_id]] : 0.13;
    // ---- ★ 2026-09-07: LATERAL STANCE TARGET WAS WIDER THAN THE STANCE -----
    // kStanceOffY is the HALF-width this residual pulls each foot to, so 0.26
    // commands a 0.52 m stance. The stumble_march keyframe stands at hip_roll
    // +/-0.12 = ~0.38 m measured. The residual is gated on g_bump (the
    // FREE-RUNNING gait clock), NOT on g_amp, so it is live in every cycle even
    // during a perfectly quiet balance-gated stand -- and it spends that
    // authority walking the feet toward a stance 14 cm wider than the one the
    // robot holds. Measured on a QUIET 240 s run with NO disturbance at all:
    // ankle-to-ankle 0.375 m -> 0.618 m, monotonic. The same creep is in every
    // push and drift run; it is not a catch-step artefact.
    // Model numeric `stance_off_y` (default 0.26 = previous behaviour).
    const double kStanceOffY = GetNumberOrDefault(0.26, model, "stance_off_y");
    // NOTE (2026-06-18): a STAGGERED/braced stance (L foot fore, R foot aft, via a
    // staggered stumble_march keyframe + kStagger here) was TESTED to ~double the
    // fore-aft support polygon (0.26->0.46 m). It improved the static baseline (4/4)
    // but did NOT improve fwd/back PUSH survival and slightly hurt lateral -> REVERTED.
    // Root cause: fore-aft recovery is reaction-BANDWIDTH limited (the planner doesn't
    // exploit the bigger base / use the ankle headroom fast enough at spline 5), not
    // support-geometry limited. The remaining principled lever is the HIP/arm
    // angular-momentum strategy (ankle->HIP->step hierarchy) — unbuilt. The foot is
    // already reference-grade fore-aft (G1 0.17 m / H1 0.20 m) so do NOT lengthen it.
    constexpr double kStagL = 0.0, kStagR = 0.0;   // stagger off (see note)

    // ---- ★ R7: CoP ANCHOR + FROZEN TARGET (both default OFF) -------------
    // MUST MATCH loco::ModifyControl's twin of this block -- the cost and
    // the open-loop swing have to aim the foot at the SAME point or the sampler
    // spends the plan fighting the script (the 2026-07-12 cmd_active_ bug class).
    //
    // (1) CoP anchor. kStanceOffX = 0.13 is a CONSTANT standing in for the DCM
    //     law's -a/omega^2 term; see the derivation in ModifyControl. Replacing
    //     the constant part of it with the measured CoM acceleration removes a
    //     gait-phase-locked placement error.
    // (2) Frozen target. Latched per leg at swing start so the foot tracks a
    //     FIXED point instead of a setpoint that moves under it every tick.
    double off_x_l = off_x, off_y_l = off_y;
    double off_x_r = off_x, off_y_r = off_y;
    const double kCopOnR = GetNumberOrDefault(0.0, model, "trot_cop_anchor");
    if (kCopOnR > 1e-6) {
      const double om2r = 9.81 / z_com;            // omega^2 = g/z
      double cdx = mju_max(-0.10, mju_min(0.10,
          -kCopOnR * com_acc_filt_[0] / om2r));
      double cdy = mju_max(-0.06, mju_min(0.06,
          -kCopOnR * com_acc_filt_[1] / om2r));
      off_x_l += cdx; off_x_r += cdx;
      off_y_l += cdy; off_y_r += cdy;
    }
    if (GetNumberOrDefault(0.0, model, "trot_freeze_place") > 0.5) {
      // step_frozen_ holds the ABSOLUTE placement offset latched at swing start,
      // in the same units/frame as off_x/off_y (a CoM-relative displacement), so
      // it substitutes for them directly.
      if (step_frozen_ok_[0]) { off_x_l = step_frozen_[0]; off_y_l = step_frozen_[1]; }
      if (step_frozen_ok_[1]) { off_x_r = step_frozen_[2]; off_y_r = step_frozen_[3]; }
    }

    // ---- ★ PLACEMENT AUTHORITY GATE (step_place_gated, default 0 = OFF) ----
    // As above: g_bump_* is the free-running gait clock, so without this gate
    // the placement target is commanded during a quiet stand as well as during
    // a step. step_place_gated = 1 scales the residual by g_amp -- the SAME
    // balance gate that already scales the swing height (amp = kStepHeight *
    // g_amp) and the swing posture fold -- so a still stand commands no
    // placement at all and only a real catch-step aims the foot. Default 0
    // reproduces the previous residual exactly (byte-identical), so this is
    // A/B-able against the deployed behaviour rather than a silent change.
    const bool sp_gated =
        GetNumberOrDefault(0.0, model, "step_place_gated") > 0.5;
    const double place_l = (g_bump_l > 0.0) ? (sp_gated ? g_amp : 1.0) : 0.0;
    const double place_r = (g_bump_r > 0.0) ? (sp_gated ? g_amp : 1.0) : 0.0;

    residual[counter++] = place_l *
        (foot_left_pos[0]  - (com_pos[0] + kStanceOffX + kStagL + off_x_l));
    residual[counter++] = place_l *
        (foot_left_pos[1]  - (com_pos[1] + (reg_l ? reg_half : kStanceOffY)
                              + (reg_l ? 0.0 : off_y_l)));
    residual[counter++] = place_r *
        (foot_right_pos[0] - (com_pos[0] + kStanceOffX + kStagR + off_x_r));
    residual[counter++] = place_r *
        (foot_right_pos[1] - (com_pos[1] - (reg_r ? reg_half : kStanceOffY)
                              + (reg_r ? 0.0 : off_y_r)));

    // --- Foot Slip (dim 2): penalize the STANCE foot SLIDING horizontally (gated
    //     on bump==0 = the planted foot). The reference MuJoCo Playground humanoid
    //     relies on this (G1 feet_slip cost): a sampling planner has no learned
    //     recovery reflex, so a stance foot that skids during the weight-shift
    //     builds the exact lateral momentum that topples a weak-ankle stepper --
    //     killing the slip at the source beats reacting to it via the capture
    //     point. (Source: Playground g1/joystick.py _cost_feet_slip.) --- //
    double const *vfl = SensorByName(model, data, "foot_left_velocity");
    double const *vfr = SensorByName(model, data, "foot_right_velocity");
    residual[counter++] =
        (g_bump_l > 0.0) ? 0.0 : mju_sqrt(vfl[0]*vfl[0] + vfl[1]*vfl[1]);
    residual[counter++] =
        (g_bump_r > 0.0) ? 0.0 : mju_sqrt(vfr[0]*vfr[0] + vfr[1]*vfr[1]);
  } else {
    residual[counter++] = 0.0;  // Gait L
    residual[counter++] = 0.0;  // Gait R
    residual[counter++] = 0.0;  // Step Place Lx
    residual[counter++] = 0.0;  // Step Place Ly
    residual[counter++] = 0.0;  // Step Place Rx
    residual[counter++] = 0.0;  // Step Place Ry
    residual[counter++] = 0.0;  // Foot Slip L
    residual[counter++] = 0.0;  // Foot Slip R
  }

  // sensor dim sanity check
  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  // Per-sensor diagnostic: prints once when the mismatch fires so we can see
  // exactly which residual is short. mju_warning is non-fatal (vs mju_error
  // which calls exit() and freezes headless agent_server with "Press Enter").
  if (user_sensor_dim != counter) {
    static bool printed = false;
    if (!printed) {
      printed = true;
      mju_warning(
          "Loco residual length %d != user_sensor_dim %d. Per-sensor:",
          counter, user_sensor_dim);
      for (int i = 0; i < model->nsensor; i++) {
        if (model->sensor_type[i] == mjSENS_USER) {
          const char *nm = mj_id2name(model, mjOBJ_SENSOR, i);
          mju_warning("  user sensor %d: %s dim=%d", i,
                      nm ? nm : "?", model->sensor_dim[i]);
        }
      }
    }
    // Pad with zeros so the planner doesn't crash; this is the fallback path
    // until the missing residual is tracked down.
    while (counter < user_sensor_dim) {
      residual[counter++] = 0.0;
    }
  }
}

// -------- Transition for the Loco task -------- //
// Runs once per plan on the REAL state: stand trim integrator, cmd_vel
// governor, drive stand<->trot FSM, R7 contact-triggered gait latch, strategy
// load / phase advance, and the per-phase weight ramp.
void loco::TransitionLocked(mjModel *model, mjData *data) {
  // ---- T1 REFERENCE TRIM v2 (ported from stabilize.cc 1708253, 2026-07-20) - //
  // Runs on the REAL state once per plan (never inside a rollout), so the
  // integrator sees measured physics, not the sampler's imagination.
  //
  // EXACT-NAME GATE: the stand keyframe only. The trim's whole premise is a
  // both-feet quiet park; a trot/walk/drive keyframe has no such park and its
  // twin tuning was never done against a moving reference. Same scoping rule
  // the 07-19 stumble anchor port used, and the reason a flag's verdict is
  // scoped to the strategy it was measured on.
  {
    const bool trim_strategy =
        (residual_.residual_keyframe_.name == "stand_up");
    int tt_id = mj_name2id(model, mjOBJ_NUMERIC, "stand_trim_tau");
    double ttau = (tt_id >= 0)
        ? model->numeric_data[model->numeric_adr[tt_id]] : 0.0;
    double const *tup = SensorByName(model, data, "torso_up");
    double const *cvel = SensorByName(model, data, "waist_lower_subcomvel");
    int pid = mj_name2id(model, mjOBJ_BODY, "pelvis");
    static double s_trim_t = -1.0;      // last tick time, for dt
    static double s2_ex = 0.0, s2_ey = 0.0;   // trim's own 4 s DC EMA
    if (trim_strategy && ttau > 0.0 && tup && cvel && pid >= 0) {
      auto Num = [&](const char *name, double dflt) {
        int id = mj_name2id(model, mjOBJ_NUMERIC, name);
        return (id >= 0) ? model->numeric_data[model->numeric_adr[id]] : dflt;
      };
      double tdelay   = Num("stand_trim_delay", 0.0);
      double tmax_pos = Num("stand_trim_max", 0.08);
      double tmax_neg = Num("stand_trim_neg_max", tmax_pos);
      double tleak    = Num("stand_trim_leak", 60.0);
      double tquiet   = Num("stand_trim_quiet", 0.03);
      double tlat_max = Num("stand_trim_lat_max", 0.02);
      double tnom     = Num("stand_trim_nominal_x", 0.0);
      const bool trim_armed = (tdelay <= 0.0) || (data->time >= tdelay);

      // SUPPORT frame from the feet's own mean heading (must match the axes the
      // Residual applies the trim along; degenerate feet -> world axes).
      double fwd[2] = {1.0, 0.0}, lat[2] = {0.0, 1.0};
      {
        double const *flf = SensorByName(model, data, "foot_left_forward");
        double const *frf = SensorByName(model, data, "foot_right_forward");
        if (flf && frf) {
          double fx = flf[0] + frf[0], fy = flf[1] + frf[1];
          double len = mju_sqrt(fx * fx + fy * fy);
          if (len > 1.0e-6) {
            fwd[0] = fx / len; fwd[1] = fy / len;
            lat[0] = -fwd[1];  lat[1] = fwd[0];
          }
        }
      }
      const mjtNum *com = data->subtree_com + 3 * pid;
      double zc = mju_max(0.5, com[2]);
      double tau_c = mju_sqrt(zc / 9.81);
      // capture excursion in the SUPPORT frame; nominal 0 = upright (v1's bug
      // was using the strat-20 lean nominal 0.06 here, which drove the stand's
      // CoM 6 cm toward the toe).
      double exf = zc * ((tup[0] * fwd[0] + tup[1] * fwd[1]) - tnom) +
                   tau_c * (cvel[0] * fwd[0] + cvel[1] * fwd[1]);
      double eyl = zc * (tup[0] * lat[0] + tup[1] * lat[1]) +
                   tau_c * (cvel[0] * lat[0] + cvel[1] * lat[1]);
      double dt = (s_trim_t >= 0.0) ? mju_max(0.0, data->time - s_trim_t) : 0.0;
      s_trim_t = data->time;
      double a2 = (dt > 0.0) ? mju_min(1.0, dt / 4.0) : 1.0;  // first call snaps
      s2_ex += a2 * (exf - s2_ex);
      s2_ey += a2 * (eyl - s2_ey);
      // QUIET gate: only a STEADY park integrates. During a push, a catch, or
      // an operator hand on the chest the instantaneous excursion leaves its DC
      // -> freeze. (v1 wound those transients into the reference = the 07-14
      // hunt. It also means an ASSISTED run must not be used to judge the trim:
      // the gate cannot see an external force, only the motion it suppresses.)
      const bool quiet = std::fabs(exf - s2_ex) < tquiet &&
                         std::fabs(eyl - s2_ey) < tquiet;
      if (trim_armed && quiet) {
        s_trim_x += (dt / ttau) * s2_ex;
        s_trim_y += (dt / ttau) * s2_ey;
      } else if (!trim_armed) {
        s_trim_x = 0.0; s_trim_y = 0.0;
      }
      // LEAK (Caron'19): decay toward 0. A true constant bias re-wins every
      // tick (steady state trim = park * tleak/ttau, residual park =
      // trim * ttau/tleak -- at the 60/15 defaults a 4 cm need leaves ~1 cm
      // park); transient garbage has no source and self-unwinds.
      if (tleak > 0.0 && dt > 0.0) {
        double decay = 1.0 - mju_min(1.0, dt / tleak);
        s_trim_x *= decay;
        s_trim_y *= decay;
      }
      s_trim_x = mju_min(tmax_pos, mju_max(-tmax_neg, s_trim_x));
      s_trim_y = mju_min(tlat_max, mju_max(-tlat_max, s_trim_y));
      static double last_print = -1.0e9;
      if (data->time - last_print > 5.0) {
        last_print = data->time;
        // stderr on purpose: agent_server/deploy stdout is BLOCK-buffered when
        // redirected, and the line vanishes if the process is killed unflushed.
        std::fprintf(stderr,
                     "[trim] t=%.1f park(fwd%+.3f lat%+.3f) m trim(x%+.3f "
                     "y%+.3f) m armed=%d quiet=%d\n",
                     data->time, s2_ex, s2_ey, s_trim_x, s_trim_y,
                     trim_armed ? 1 : 0, quiet ? 1 : 0);
      }
    } else {
      // OFF (or a non-stand strategy): hard-zero everything, including the EMA
      // state, so a later arm starts clean instead of inheriting a stale park.
      s_trim_x = 0.0; s_trim_y = 0.0;
      s2_ex = 0.0; s2_ey = 0.0; s_trim_t = -1.0;
    }
  }

  // ---- LIVE cmd_vel GOVERNOR (WASD/gamepad/Nav2 teleop, 2026-07-03) ------
  // Single safety authority for live velocity commands: heartbeat watchdog,
  // gait-arm lockout, envelope clamp, settle-through-zero on sign flips,
  // slew limit, then ONE body->world yaw rotation. The result is stored on
  // residual_ so BOTH v_des readers (Residual + loco::ModifyControl) consume
  // the SAME world vector -- the mirrored-agreement requirement holds by
  // construction. Inactive => members zeroed + cmd_active_=false => readers
  // take the legacy trot_des_vel numeric path, byte-identical to the
  // validated configs.
  {
    const bool cmd_on = (int)parameters.size() > kLocoCmdSeqParameterIndex &&
                        parameters[kLocoCmdActiveParameterIndex] > 0.5;
    if (cmd_on) {
      double now = data->time;
      double vx = parameters[kLocoCmdVxParameterIndex];  // BODY-frame m/s
      double vy = parameters[kLocoCmdVyParameterIndex];
      double seq = parameters[kLocoCmdSeqParameterIndex];
      if (seq != residual_.cmd_last_seq_) {
        residual_.cmd_last_seq_ = seq;
        residual_.cmd_seq_time_ = now;
      }
      // client-heartbeat watchdog: Seq frozen > 1 s = dead client -> stop.
      bool starved = (now - residual_.cmd_seq_time_) > 1.0;
      // gait-arm lockout: kArmSec(2.0) swing arm-in + margin. The R6 lesson:
      // acting on v_des during the bring-up transient topples the robot.
      bool locked = (now - residual_.keyframe_start_time_) < 6.0;
      if (starved || locked) { vx = 0.0; vy = 0.0; }
      // envelope = the twin-validated speeds (2026-07-02): fwd 0.15 8/8,
      // back 0.10 5/5. 0.20 is fast-marginal and deliberately excluded.
      vx = mju_min(0.15, mju_max(-0.10, vx));
      // V2 strafe (2026-07-07): the DRIVE strategy accepts lateral, clamped to
      // the (weak-axis) drive_vy_max envelope; EVERY other strategy keeps vy=0
      // (V0 byte-identical). drive_vy_max default 0.06; the twin campaign sets
      // it without a rebuild.
      {
        int strat_g = (int)std::round(parameters[kLocoStrategyParameterIndex]);
        auto snames_g = GetStrategyNames();
        bool drive_g = strat_g >= 0 && strat_g < (int)snames_g.size() &&
            snames_g[strat_g].find("drive") != std::string::npos;
        double vy_max = 0.0;
        if (drive_g) {
          int vym_id = mj_name2id(model, mjOBJ_NUMERIC, "drive_vy_max");
          vy_max = (vym_id >= 0)
              ? model->numeric_data[model->numeric_adr[vym_id]] : 0.06;
        }
        vy = mju_min(vy_max, mju_max(-vy_max, vy));
      }
      // V3 yaw-rate (2026-07-07): drive strategy only; clamp to drive_wz_max and
      // slew into cmd_wz_ (the FSM integrates it into a desired heading below).
      {
        int strat_w = (int)std::round(parameters[kLocoStrategyParameterIndex]);
        auto sn_w = GetStrategyNames();
        bool drive_w = strat_w >= 0 && strat_w < (int)sn_w.size() &&
            sn_w[strat_w].find("drive") != std::string::npos;
        double wz = 0.0;
        if (drive_w && (int)parameters.size() > kLocoCmdWzParameterIndex &&
            !starved && !locked) {
          int wzm = mj_name2id(model, mjOBJ_NUMERIC, "drive_wz_max");
          double wzmax = (wzm >= 0)
              ? model->numeric_data[model->numeric_adr[wzm]] : 0.3;
          wz = mju_min(wzmax,
                       mju_max(-wzmax, parameters[kLocoCmdWzParameterIndex]));
        }
        double dtw = (residual_.cmd_prev_time_ > 0.0)
                         ? mju_max(0.0, now - residual_.cmd_prev_time_) : 0.0;
        double dwz = 1.0 * dtw;   // slew 1 rad/s^2
        residual_.cmd_wz_ += mju_min(dwz, mju_max(-dwz, wz - residual_.cmd_wz_));
      }
      // settle-through-zero: a fwd<->back flip dwells at 0 for 1.5 s (~1.6
      // gait cycles) so the capture catch kills momentum before reversing.
      if (vx * residual_.cmd_filt_[0] < -1e-9)
        residual_.cmd_settle_until_ = now + 1.5;
      if (now < residual_.cmd_settle_until_) vx = 0.0;
      // slew 0.08 m/s^2 (0 -> 0.15 in ~1.9 s); gentler than the validated
      // campaigns' cold-start step changes.
      double dt = residual_.cmd_prev_time_ > 0.0
                      ? mju_max(0.0, now - residual_.cmd_prev_time_)
                      : 0.0;
      double dv = 0.08 * dt;
      double tgt[2] = {vx, vy};
      for (int a = 0; a < 2; a++) {
        double d = tgt[a] - residual_.cmd_filt_[a];
        residual_.cmd_filt_[a] += mju_min(dv, mju_max(-dv, d));
      }
      residual_.cmd_prev_time_ = now;
      // body->world (yaw only; identity at yaw 0 => twin-validated behavior
      // bit-unchanged). Also keeps "W = robot-forward" under IMU yaw drift.
      double qw = data->qpos[3], qx = data->qpos[4], qy = data->qpos[5],
             qz = data->qpos[6];
      double yaw = std::atan2(2.0 * (qw * qz + qx * qy),
                              1.0 - 2.0 * (qy * qy + qz * qz));
      double cy = std::cos(yaw), sy = std::sin(yaw);
      residual_.cmd_vdes_world_[0] =
          cy * residual_.cmd_filt_[0] - sy * residual_.cmd_filt_[1];
      residual_.cmd_vdes_world_[1] =
          sy * residual_.cmd_filt_[0] + cy * residual_.cmd_filt_[1];
      if (!residual_.cmd_active_)
        std::fprintf(stderr, "[loco] cmd governor: ACTIVE (vx=%.3f)\n", vx);
      else if (starved && !residual_.cmd_starved_)
        std::fprintf(stderr, "[loco] cmd governor: starved=1 -> zeroing\n");
      residual_.cmd_starved_ = starved;
      residual_.cmd_active_ = true;
    } else if (residual_.cmd_active_) {
      residual_.cmd_active_ = false;
      residual_.cmd_filt_[0] = residual_.cmd_filt_[1] = 0.0;
      residual_.cmd_vdes_world_[0] = residual_.cmd_vdes_world_[1] = 0.0;
      residual_.cmd_prev_time_ = -1.0;
      std::fprintf(stderr, "[loco] cmd governor: OFF -> legacy numerics\n");
    }
  }

  // strategy-based contact keyframe progression
  const auto kStrategyNames = GetStrategyNames();
  int requested_strategy =
      (int)std::round(parameters[kLocoStrategyParameterIndex]);
  requested_strategy = std::max(
      0, std::min(requested_strategy, (int)kStrategyNames.size() - 1));

  // ---- WSS DRIVE stand<->trot FSM (strat 24, 2026-07-07) -----------------
  // Gait-enable latch: no command -> drive_gait_amp_ ramps to 0 (feet plant =
  // real stand; the strat-20 balance-gate below still steps to catch a push);
  // any command -> ramps to 1 (full get_rz trot). Hysteresis (engage fast,
  // release after a dwell) avoids stand<->walk chatter at the boundary. Stored
  // on residual_; ResidualLocked copies it into the plan snapshot so the cost
  // agrees with ModifyControl. Drive-strategy only -> every other strategy is
  // byte-identical (drive_gait_amp_ stays 0, unread outside is_drive).
  {
    const bool is_drive_strat =
        kStrategyNames[requested_strategy].find("drive") != std::string::npos;
    if (is_drive_strat) {
      double now = data->time;
      // command magnitude: the governor's slewed body command when a client is
      // live; no client -> 0 (static trot_des_vel default 0 => idle => stand).
      double cvx = residual_.cmd_active_ ? residual_.cmd_filt_[0] : 0.0;
      double cvy = residual_.cmd_active_ ? residual_.cmd_filt_[1] : 0.0;
      double cwz = residual_.cmd_active_ ? residual_.cmd_wz_ : 0.0;
      // yaw contributes to engage (so Q/E alone spins in place = rotational
      // in-place trot); 0.15 converts rad/s to a translation-comparable scale.
      double m = std::sqrt(cvx * cvx + cvy * cvy) + 0.15 * std::fabs(cwz);
      constexpr double kEngage = 0.02, kRelease = 0.01, kReleaseDwell = 1.2;
      if (m > kEngage) {
        if (!residual_.drive_walk_)
          std::fprintf(stderr,
                       "[loco] drive: WALK (m=%.3f) -> gait ramping up\n", m);
        residual_.drive_walk_ = true;
        residual_.drive_idle_since_ = -1.0;
      } else if (m < kRelease) {
        if (residual_.drive_idle_since_ < 0.0) residual_.drive_idle_since_ = now;
        // R2 velocity-gated disengage (2026-07-08): don't release on the dwell
        // timer alone -- hold the gait until the base has actually SLOWED
        // (step-to-rest, the Unitree "track velocity to zero WHILE stepping"
        // recipe) so we never plant the feet with forward momentum -> faceplant.
        // kReleaseDwell is the MINIMUM dwell; then release once |base_vel| drops
        // below kReleaseSpeed, OR after kMaxReleaseDwell as a backstop (R1's
        // capture catch-step handles any residual velocity if we plant anyway).
        double bspd = std::sqrt(data->qvel[0] * data->qvel[0] +
                                data->qvel[1] * data->qvel[1]);
        constexpr double kReleaseSpeed = 0.08, kMaxReleaseDwell = 3.0;
        double idle_for = now - residual_.drive_idle_since_;
        if (residual_.drive_walk_ && idle_for > kReleaseDwell &&
            (bspd < kReleaseSpeed || idle_for > kMaxReleaseDwell)) {
          residual_.drive_walk_ = false;
          std::fprintf(stderr,
                       "[loco] drive: STAND -> gait ramping down (v=%.2f)\n",
                       bspd);
        }
      }
      // ramp drive_gait_amp_ toward the latch over kDriveRampSec.
      constexpr double kDriveRampSec = 1.5;
      double tgt = residual_.drive_walk_ ? 1.0 : 0.0;
      double dt = (residual_.drive_ramp_prev_ > 0.0)
                      ? std::max(0.0, now - residual_.drive_ramp_prev_)
                      : 0.0;
      double da = dt / kDriveRampSec;
      double d = tgt - residual_.drive_gait_amp_;
      residual_.drive_gait_amp_ += std::min(da, std::max(-da, d));
      residual_.drive_gait_amp_ =
          std::min(1.0, std::max(0.0, residual_.drive_gait_amp_));
      // ★ RE-PIN WHILE IDLE (2026-07-20): once the gait has fully ramped down the
      // robot is standing on both planted feet, so re-anchor Foot Stability to
      // where the feet ACTUALLY ended up. This is what makes the post-walk halt a
      // clean stand instead of one fighting a stale target -- the feet legitimately
      // moved during the walk, and without this the cost would drag them back to a
      // pre-walk (and odometrically drifting) point. NOT done mid-gait: while
      // stepping the feet are SUPPOSED to travel, and re-pinning every tick would
      // zero out the weight-8 cost the walk was tuned with.
      // ★ GUARDED (2026-07-20, after the real-robot sideways drag): opt-in via
      // the drive_foot_anchor numeric (default 0 = OFF), only after the strategy
      // has been live for kAnchorSettle so the estimator is initialised, and only
      // if the reading is a PLAUSIBLE stance. Anything outside those bounds is a
      // bad read, not a stance -- keep the previous anchor rather than pin to it.
      constexpr double kAnchorSettle = 3.0;   // s since keyframe entry
      const bool anchor_on =
          GetNumberOrDefault(0.0, model, "drive_foot_anchor") > 0.5;
      if (anchor_on && residual_.drive_gait_amp_ <= 1e-3 &&
          !residual_.drive_walk_ &&
          data->time - residual_.keyframe_start_time_ > kAnchorSettle) {
        double *frp = SensorByName(model, data, "foot_right_pos");
        double *flp = SensorByName(model, data, "foot_left_pos");
        if (frp && flp) {
          // right foot -y, left foot +y; half-stance 6..30 cm, |x| < 0.6 m.
          const bool sane =
              frp[1] < -0.06 && frp[1] > -0.30 && flp[1] > 0.06 && flp[1] < 0.30 &&
              std::fabs(frp[0]) < 0.6 && std::fabs(flp[0]) < 0.6;
          if (sane) {
            residual_.drive_foot_anchor_[0] = frp[0];
            residual_.drive_foot_anchor_[1] = frp[1];
            residual_.drive_foot_anchor_[2] = flp[0];
            residual_.drive_foot_anchor_[3] = flp[1];
          }
        }
      }
      // V3 heading integrator: idle + no yaw cmd -> track current heading (no
      // rotation demand); else integrate cmd_wz_ into the desired WORLD yaw.
      double qw = data->qpos[3], qx = data->qpos[4],
             qy = data->qpos[5], qz = data->qpos[6];
      double cur_yaw = std::atan2(2.0 * (qw * qz + qx * qy),
                                  1.0 - 2.0 * (qy * qy + qz * qz));
      // 2026-10-14 BUG FIX: a yaw command BELOW the gait engage threshold
      // (|wz| 0.12 -> m = 0.018 < kEngage) used to integrate into the heading
      // target while the feet stayed planted: Body Yaw 40 then twisted the
      // standing robot over (twin run `eslow`, fell with zero WALK events).
      // The heading only moves while the gait is latched ON.
      if (!residual_.drive_walk_)
        residual_.drive_yaw_des_ = cur_yaw;
      else
        residual_.drive_yaw_des_ += residual_.cmd_wz_ * dt;
      residual_.drive_ramp_prev_ = now;
    } else {
      residual_.drive_gait_amp_ = 0.0;
      residual_.drive_walk_ = false;
      residual_.drive_idle_since_ = -1.0;
      residual_.drive_ramp_prev_ = -1.0;
      residual_.cmd_wz_ = 0.0;
      // ★ HEADING LATCH for the non-drive trot (strat 23). BUGFIX 2026-07-12: it
      // has no yaw command, so the heading it should hold is simply the one it had
      // when the gait armed. TRACK the live yaw while the gait is still ramping in
      // (< kArmSec, the same 2 s the swing amplitude uses), then FREEZE it -- from
      // then on the Body Yaw lock above holds that heading instead of chasing the
      // randomized reach target. Without this drive_yaw_des_ stays 0 for the trot,
      // which is only ACCIDENTALLY right (the keyframe quat happens to be yaw 0)
      // and breaks silently the moment the robot starts from any other heading.
      const bool is_trot_strat =
          kStrategyNames[requested_strategy].find("trot") != std::string::npos;
      if (is_trot_strat) {
        double qw = data->qpos[3], qx = data->qpos[4],
               qy = data->qpos[5], qz = data->qpos[6];
        double cur_yaw = std::atan2(2.0 * (qw * qz + qx * qy),
                                    1.0 - 2.0 * (qy * qy + qz * qz));
        constexpr double kArmSec = 2.0;   // matches the gait amplitude ramp
        if (data->time - residual_.keyframe_start_time_ < kArmSec)
          residual_.drive_yaw_des_ = cur_yaw;   // track, then hold
      } else {
        residual_.drive_yaw_des_ = 0.0;
      }
    }
  }

  // ============ ★ R7 CONTACT-TRIGGERED GAIT LATCH (2026-09-03) ============
  // Everything the contact-corrected gait needs, computed ONCE per plan on the
  // REAL state. This is the ONLY place that can do it: loco::ModifyControl
  // is const and receives (model, qpos, qvel, time, ctrl) with NO mjData, so it
  // cannot see a single contact. Results are written onto residual_ and copied
  // into the per-plan snapshot by ResidualLocked, so the rollout cost and the
  // open-loop swing writer read identical values -- the same discipline the
  // 2026-07-12 cmd_active_ bugfix had to restore.
  //
  // Produces (each inert while its numeric is 0 => byte-identical to pre-R7):
  //   gait_phase_offset_  contact PLL correction  <- trot_contact_switch
  //   gait_overrun_       ground-seek overrun     <- trot_ground_seek
  //   step_frozen_        per-step frozen target  <- trot_freeze_place
  //   com_acc_filt_       CoP anchor input        <- trot_cop_anchor
  //   dcm_target_         exponential DCM target  <- trot_dcm_target
  {
    const std::string &r7name = residual_.residual_keyframe_.name;
    const bool r7_stumble = (r7name.rfind("stumble", 0) == 0);
    if (!r7_stumble) {
      // Not a stepping keyframe -> reset, so a later trot always starts clean
      // rather than inheriting a stale phase correction from a previous run.
      residual_.gait_phase_offset_ = 0.0;
      residual_.com_acc_prev_time_ = -1.0;
      residual_.dcm_target_ok_ = false;
      residual_.catch_armed_ = false;        // catch phase-reset latch
      residual_.catch_reset_time_ = -1.0;
      residual_.catch_bias_on_ = false;      // sustained-bias arm latch
      residual_.bias_since_ = -1.0;
      residual_.catch_onset_time_ = -1.0;
      residual_.stance_sep_y_ = -1.0;
      residual_.catch_hold_until_ = -1.0;
      residual_.catch_hold_recov_ = 0.0;
      for (int i = 0; i < 2; i++) {
        residual_.gait_td_time_[i] = -1.0;
        residual_.gait_in_swing_[i] = false;
        residual_.gait_swing_t0_[i] = -1.0;
        residual_.gait_overrun_[i] = 0.0;
        residual_.gait_lifted_[i] = false;   // R7b interlock re-arms on unload
        residual_.step_frozen_ok_[i] = false;
        residual_.com_acc_filt_[i] = 0.0;
      }
      residual_.com_ref_ok_ = false;
      residual_.com_ref_stance_ = -1;
    } else {
      const bool r7_trot = (r7name.find("trot") != std::string::npos);
      const double now = data->time;
      auto r7clip = [](double x, double lo, double hi) {
        return x < lo ? lo : (x > hi ? hi : x);
      };

      // ---- (a) filtered CoM acceleration (the CoP-anchor input) -----------
      // a = d/dt of the subtree CoM velocity sensor. The raw finite difference
      // is far too noisy to place a feet with -- hence the spike reject plus a
      // first-order low-pass whose time constant is a live numeric. This runs
      // unconditionally (it is cheap and stateful, so it must stay warm even
      // while trot_cop_anchor is 0, or switching the anchor on mid-run would
      // start it from a cold filter).
      double *r7cv = SensorByName(model, data, "waist_lower_subcomvel");
      if (r7cv) {
        double dt7 = (residual_.com_acc_prev_time_ > 0.0)
                         ? (now - residual_.com_acc_prev_time_) : 0.0;
        if (dt7 > 1e-4 && dt7 < 0.5) {
          const double lp = mju_max(1e-3,
              GetNumberOrDefault(0.08, model, "trot_cop_lp"));   // [s]
          const double a_lp = dt7 / (lp + dt7);
          for (int i = 0; i < 2; i++) {
            double raw = (r7cv[i] - residual_.com_vel_prev_[i]) / dt7;
            raw = r7clip(raw, -8.0, 8.0);        // reject differentiation spikes
            residual_.com_acc_filt_[i] += a_lp * (raw - residual_.com_acc_filt_[i]);
          }
        }
        residual_.com_vel_prev_[0] = r7cv[0];
        residual_.com_vel_prev_[1] = r7cv[1];
        residual_.com_acc_prev_time_ = now;
      }

      // ---- (a2) filtered CAPTURE-POINT DIVERGENCE RATE -------------------
      // Same danger the catch-step gate uses, differentiated and low-passed.
      // Runs unconditionally so the filter stays warm even while
      // catch_rate_lead is 0 (switching it on mid-run must not start cold).
      {
        double const *tup2 = SensorByName(model, data, "torso_up");
        int pid2 = mj_name2id(model, mjOBJ_BODY, "pelvis");
        if (pid2 < 0) pid2 = 1;
        const mjtNum *com2 = data->subtree_com + 3 * pid2;
        double zc2 = mju_max(0.5, com2[2]);
        double tau2 = mju_sqrt(zc2 / 9.81);
        double lnx2 = GetNumberOrDefault(0.06, model, "lean_nominal_x");
        {   // same stumble-scoped override as the residual (see there)
          double so2 = GetNumberOrDefault(-1.0, model, "stumble_lean_nominal");
          if (so2 >= 0.0 && !r7_trot) lnx2 = so2;
        }
        double vx2 = r7cv ? r7cv[0] : 0.0;
        double ex2 = zc2 * ((tup2 ? tup2[0] : 0.0) - lnx2) + tau2 * vx2;
        double ey2 = zc2 * (tup2 ? tup2[1] : 0.0);
        double dg2 = mju_sqrt(ex2 * ex2 + ey2 * ey2);
        double dtd = (residual_.danger_prev_time_ > 0.0)
                         ? (now - residual_.danger_prev_time_) : 0.0;
        if (dtd > 1e-4 && dtd < 0.5) {
          const double lpd = mju_max(1e-3,
              GetNumberOrDefault(0.10, model, "catch_rate_lp"));    // [s]
          const double a_d = dtd / (lpd + dtd);
          double raw = (dg2 - residual_.danger_prev_) / dtd;
          raw = r7clip(raw, -5.0, 5.0);
          residual_.danger_rate_ += a_d * (raw - residual_.danger_rate_);
        }
        residual_.danger_prev_ = dg2;
        residual_.danger_prev_time_ = now;

        // ---- ★ (a2a) MEASURED STANCE SEPARATION -----------------------------
        // Lateral ankle-to-ankle distance, projected on the BASE-YAW lateral
        // axis so a yawed stand (or a staggered one, where the feet also differ
        // in x) still reports the LATERAL separation and not the 2-D distance.
        // Feeds the swing-foot separation regulator in ModifyControl, which has
        // no mjData of its own. Runs unconditionally so the value is warm.
        {
          double const *flp3 = SensorByName(model, data, "foot_left_pos");
          double const *frp3 = SensorByName(model, data, "foot_right_pos");
          if (flp3 && frp3) {
            const double qw = data->qpos[3], qx = data->qpos[4];
            const double qy = data->qpos[5], qz = data->qpos[6];
            // base forward (x-axis of the base frame) projected on the ground
            double fx = 1.0 - 2.0 * (qy * qy + qz * qz);
            double fy = 2.0 * (qx * qy + qw * qz);
            const double fn = mju_sqrt(fx * fx + fy * fy);
            if (fn > 1e-6) { fx /= fn; fy /= fn; }
            else { fx = 1.0; fy = 0.0; }
            const double lx = -fy, ly = fx;          // base LEFT = +y of base
            residual_.stance_sep_y_ =
                (flp3[0] - frp3[0]) * lx + (flp3[1] - frp3[1]) * ly;
          }
        }

        // ---- ★ (a2b) SUSTAINED-BIAS ARM (catch_bias_trig, default 0 = OFF) --
        // THE TIME BUDGET IS THE WHOLE PROBLEM. Measured 2026-09-08 across 88
        // runs: `danger` stays above catch_trig for only 0.5-0.9 s before the
        // robot is down -- and that is true under the SLOW DRIFT rig too, not
        // just under an impulse. So the controller does not admit it is falling
        // until it is already falling, and every downstream tier inherits that
        // budget.
        // What the budget has to pay for: a fore/aft step cannot begin until the
        // CoM has moved laterally onto one foot (~0.19 m), and lateral CoM
        // acceleration is capped by keeping the CoP inside the polygon at
        // a = 0.19*g/z = 2.07 m/s^2, which covers 0.19 m in 0.43 s. There is no
        // way to buy that inside 0.3 s; the 2026-09-07 prep-rock measurement
        // (commanded 0.175 m, ACHIEVED 0.06-0.13 m) is exactly this bound, and
        // it is why catch_prep_* failed both alone and in combination. The
        // transfer was never wrong -- it was starved of time.
        // A slow lean is detectable SECONDS before a capture-point divergence.
        // So: also arm when danger has held above a much LOWER threshold
        // CONTINUOUSLY for catch_bias_dwell. The low threshold buys the time;
        // the dwell is what stops a quiet-stand wobble from marching the robot
        // (a bare low catch_trig has no such guard, which is why lowering it
        // alone was always a hold-rate risk). The fast capture-point path is
        // untouched and still fires first on a real shove.
        {
          const double kBT = GetNumberOrDefault(0.0, model, "catch_bias_trig");
          const double kBD = GetNumberOrDefault(0.40, model, "catch_bias_dwell");
          if (kBT > 0.0 && dg2 > kBT) {
            if (residual_.bias_since_ < 0.0) residual_.bias_since_ = now;
          } else {
            residual_.bias_since_ = -1.0;
          }
          residual_.catch_bias_on_ = (kBT > 0.0) &&
                                     (residual_.bias_since_ >= 0.0) &&
                                     (now - residual_.bias_since_ >= kBD);
        }

        // ---- ★ (a3) CATCH PHASE RESET (stumble_phase_reset, default 0 = OFF)
        // WHY. Wiring the open-loop swing writer for strat 20 (see
        // stumble_swing_force) made the foot actually lift -- measured clearance
        // 0.012 -> 0.041 m backward -- and immediately exposed the next
        // constraint: `danger` stays above catch_trig for only 0.5-0.9 s before
        // the robot is down, while a swing window is (1-duty)/cadence = 0.36 s
        // and opens on a FREE-RUNNING clock. Arming at a uniformly random phase
        // means an expected 0.28 s and a worst case of 0.55 s of pure WAITING
        // before the chosen foot is even allowed to leave the ground. That alone
        // can consume the entire budget, and it is why the catch looks like it
        // "sometimes" steps: it is a coin flip on clock phase, not on physics.
        //
        // So on the RISING EDGE of the catch gate, slew the gait clock so the
        // chosen catch foot is at exactly phase == duty, i.e. at swing start,
        // NOW. This is the same contact-triggered switching R7 built for the
        // trot (trot_contact_switch, which writes this identical field) -- strat
        // 20 simply never had a trigger wired to it.
        //
        // Foot choice replicates the residual's rule EXACTLY (fwd -> trailing
        // foot, back -> front foot, lateral -> falling-side foot), because the
        // cost's g_bump_l/g_bump_r and this clock must name the same leg or the
        // sampler is asked to lift one foot while the forcer drives the other.
        //
        // Edge-latched AND time-debounced: a danger signal hovering at the
        // threshold would otherwise re-slam the clock every tick and no swing
        // would ever complete. 0 = OFF = the free-running clock, byte-identical.
        {
          const double kPhR = GetNumberOrDefault(0.0, model, "stumble_phase_reset");
          double kCT2 = GetNumberOrDefault(0.12, model, "catch_trig");
          const double kDeb = GetNumberOrDefault(0.60, model, "stumble_phase_debounce");
          // arm on whichever gate is open: the fast capture-point one, or the
          // sustained-bias one latched just above (which fires far earlier).
          if (residual_.catch_bias_on_)
            kCT2 = GetNumberOrDefault(kCT2, model, "catch_bias_trig");
          const bool armed_now = (dg2 > kCT2);
          const bool fresh = (residual_.catch_reset_time_ < 0.0) ||
                             (now - residual_.catch_reset_time_ > kDeb);
          if (kPhR > 0.5 && armed_now && !residual_.catch_armed_ && fresh) {
            double const *flp2 = SensorByName(model, data, "foot_left_pos");
            double const *frp2 = SensorByName(model, data, "foot_right_pos");
            bool stepL;
            if (std::fabs(ey2) > std::fabs(ex2)) {
              stepL = (ey2 > 0.0);                       // falling LEFT
            } else if (flp2 && frp2) {
              stepL = (ex2 > 0.0) ? (flp2[0] <= frp2[0]) // fwd  -> trailing foot
                                  : (flp2[0] >= frp2[0]);// back -> front foot
            } else {
              stepL = (ex2 > 0.0);
            }
            const GaitClock gr = GaitPhases(model, now, 0.0, r7_trot);
            // ph_l = frac(now*cad + off), ph_r = ph_l + 0.5. Solve for the off
            // that puts the CHOSEN leg at swing start (ph == duty) right now.
            double want_l = stepL ? gr.duty : std::fmod(gr.duty + 0.5, 1.0);
            double off = std::fmod(want_l - now * gr.cad, 1.0);
            if (off < 0.0) off += 1.0;
            residual_.gait_phase_offset_ = off;
            residual_.catch_reset_time_ = now;
          }
          // Onset is latched whether or not the phase reset is enabled -- the
          // unweight pulse in ModifyControl keys off it independently.
          if (armed_now && !residual_.catch_armed_)
            residual_.catch_onset_time_ = now;
          residual_.catch_armed_ = armed_now;

          // ---- ★ COMMITTED STEP PAIR (stumble_step_pair, default 0 = OFF) --
          // MEASURED 2026-09-09: a lateral catch stays armed 0.36-0.9 s while
          // one gait cycle is 1/cadence = 0.91 s. So exactly one foot gets a
          // swing window, and the recorded consequence is that the stance
          // TRANSLATES BY ONE LEG -- separation 0.358 -> 0.401 m and
          // 0.338 -> 0.432 m in the two runs that stepped, against +0.002 m in
          // a run that did not. Every step in every run was the same foot. It
          // is not choosing one foot; it only has time for one.
          // So once the catch has fired, hold it armed for stumble_step_pair
          // gait cycles regardless of `danger` falling back, and hold the
          // amplitude at the value it reached, so the trailing foot takes its
          // own swing and the PAIR ends up displaced together.
          // NOTE both feet cannot leave the ground simultaneously without a
          // flight phase, during which there is no ground reaction and no
          // balance authority at all -- so this is a fast ALTERNATING pair
          // (A lands, B swings half a cycle later), not a hop.
          const double kPair = GetNumberOrDefault(0.0, model, "stumble_step_pair");
          if (kPair > 1e-6) {
            const GaitClock gp = GaitPhases(model, now, 0.0, r7_trot);
            const double cyc = (gp.cad > 1e-6) ? (1.0 / gp.cad) : 0.91;
            if (armed_now) {
              double rr = (dg2 - kCT2) /
                          mju_max(1e-3, GetNumberOrDefault(0.24, model, "catch_full") - kCT2);
              rr = mju_min(1.0, mju_max(0.0, rr));
              rr = rr * rr * (3.0 - 2.0 * rr);
              if (now >= residual_.catch_hold_until_) {   // fresh catch
                residual_.catch_hold_until_ = now + kPair * cyc;
                residual_.catch_hold_recov_ = rr;
              } else {                                    // still committed
                residual_.catch_hold_recov_ =
                    mju_max(residual_.catch_hold_recov_, rr);
              }
            }
          }
        }
      }

      // ---- (b) per-foot normal force -> contact ---------------------------
      // Same per-body contact-force summation the Gait residual already uses.
      // A FORCE threshold (not mere geom proximity) is what makes this a real
      // touchdown detector: a foot brushing the floor unloaded is not a stance.
      int r7fl = mj_name2id(model, mjOBJ_BODY, "left_ankle_roll_link");
      int r7fr = mj_name2id(model, mjOBJ_BODY, "right_ankle_roll_link");
      double r7fn[2] = {0.0, 0.0};
      for (int c = 0; c < data->ncon; c++) {
        const mjContact &con = data->contact[c];
        int ba = model->geom_bodyid[con.geom1], bb = model->geom_bodyid[con.geom2];
        mjtNum f6[6];
        mj_contactForce(model, data, c, f6);
        double fn = mju_abs(f6[0]);              // normal force (contact-frame x)
        if (ba == r7fl || bb == r7fl) r7fn[0] += fn;
        if (ba == r7fr || bb == r7fr) r7fn[1] += fn;
      }
      const double kTdF = GetNumberOrDefault(60.0, model, "trot_td_force");  // [N]
      const bool r7_contact[2] = {r7fn[0] > kTdF, r7fn[1] > kTdF};

      // ---- (c) clock, swing bookkeeping, ground-seek overrun --------------
      const GaitClock g7 = GaitPhases(model, now, residual_.gait_phase_offset_,
                                      r7_trot);
      const double r7ph[2] = {g7.ph_l, g7.ph_r};
      const double T_sw = (1.0 - g7.duty) / mju_max(1e-6, g7.cad);  // swing dur [s]
      const double kMinStep =
          GetNumberOrDefault(0.10, model, "trot_min_step");   // debounce [s]
      const bool kSwitchOn =
          GetNumberOrDefault(0.0, model, "trot_contact_switch") > 0.5;

      for (int i = 0; i < 2; i++) {
        const bool nominal_swing = (r7ph[i] >= g7.duty);
        // ★ R7b LIFTOFF INTERLOCK: arm the leg only once it has genuinely
        // unloaded. The swing WINDOW is a clock fact, but leaving the ground is a
        // physical one, and the two disagree exactly when the gait amplitude is 0
        // -- drive standing idle, or the balance-gated stumble march holding
        // still. There BOTH feet stay loaded through every nominal swing window,
        // so an un-interlocked touchdown test fires on phantom contact every
        // cycle and winds the PLL with noise. Cleared on touchdown below.
        if (!r7_contact[i]) residual_.gait_lifted_[i] = true;

        // LIFTOFF: the clock has just entered this leg's swing window.
        if (nominal_swing && !residual_.gait_in_swing_[i]) {
          residual_.gait_in_swing_[i] = true;
          residual_.gait_swing_t0_[i] = now;
          residual_.gait_overrun_[i] = 0.0;

          // ---- (d) FREEZE the placement for this step --------------------
          // Latch the target ONCE, here, instead of letting the swing chase a
          // setpoint recomputed from live qvel on every tick. Same quantity the
          // residual's off_x/off_y compute (a CoM-relative displacement), so it
          // substitutes for them directly on both readers.
          int pid7 = mj_name2id(model, mjOBJ_BODY, "pelvis");
          if (pid7 < 0) pid7 = 1;
          const mjtNum *com7 = data->subtree_com + 3 * pid7;
          const double z7 = mju_max(0.5, com7[2]);
          const double tau7 = mju_sqrt(z7 / 9.81);
          const double om7 = 1.0 / tau7;
          // v_des in the WORLD frame -- MUST match the residual's resolution
          // order (governed teleop command first, static numerics otherwise).
          double vd7[2] = {0.0, 0.0};
          if (r7_trot) {
            int dvx7 = mj_name2id(model, mjOBJ_NUMERIC, "trot_des_vel_x");
            int dvy7 = mj_name2id(model, mjOBJ_NUMERIC, "trot_des_vel_y");
            if (dvx7 >= 0) vd7[0] = model->numeric_data[model->numeric_adr[dvx7]];
            if (dvy7 >= 0) vd7[1] = model->numeric_data[model->numeric_adr[dvy7]];
            if (residual_.cmd_active_) {
              vd7[0] = residual_.cmd_vdes_world_[0];
              vd7[1] = residual_.cmd_vdes_world_[1];
            }
          }
          constexpr double kStepReach7 = 0.30;   // matches the residual clamp
          double frz[2];
          if (r7cv) {
            frz[0] = (r7cv[0] - vd7[0]) * tau7;
            frz[1] = (r7cv[1] - vd7[1]) * tau7;
          } else {
            frz[0] = frz[1] = 0.0;
          }

          // ---- (e) EXPONENTIAL DCM TARGET (trot_dcm_target, default OFF) --
          // Closed-form LIPM solution over the swing:
          //     xi0   = x + v/omega                (DCM now)
          //     xides = x + v_des/omega            (DCM we want to be riding)
          //     p0    = (xides - xi0 e^{omega T}) / (1 - e^{omega T})
          //     xtgt  = p0 + (xi0 - p0) e^{omega T}
          // xtgt is where the DCM will BE at touchdown, so planting the foot
          // there is the deadbeat DCM step. Without it the placement script is
          // memoryless -- and with e^{omega T} ~ 3.2 for us (omega 3.21,
          // T_swing 0.364 s) an unmanaged placement error is amplified 3.2x per
          // step, which is the quantitative reason a memoryless target drifts.
          // (Note: the reference 2D walker computes this in get_target_0 but
          // never consumes it downstream -- get_p drives its placement. We use
          // it as the deadbeat placement it is actually derived to be.)
          if (r7cv && GetNumberOrDefault(0.0, model, "trot_dcm_target") > 0.5) {
            const double eT = std::exp(om7 * T_sw);
            const double den = 1.0 - eT;                  // != 0 for T_sw > 0
            if (std::fabs(den) > 1e-6) {
              for (int a = 0; a < 2; a++) {
                const double xi0 = com7[a] + r7cv[a] * tau7;
                const double xid = com7[a] + vd7[a] * tau7;
                const double p0 = (xid - xi0 * eT) / den;
                residual_.dcm_target_[a] = p0 + (xi0 - p0) * eT;
              }
              residual_.dcm_target_ok_ = true;
              // placement offset = CoM -> predicted end-of-step DCM
              frz[0] = residual_.dcm_target_[0] - com7[0];
              frz[1] = residual_.dcm_target_[1] - com7[1];
            }
          }
          residual_.step_frozen_[2 * i]     = r7clip(frz[0], -kStepReach7, kStepReach7);
          residual_.step_frozen_[2 * i + 1] = r7clip(frz[1], -kStepReach7, kStepReach7);
          // COMMIT-THEN-TRACK (trot_com_commit, 2026-10-14): foot i starts its
          // swing -> the OTHER foot is the stance for this step. Latch its measured
          // xy + the DCM at the edge; the residual tracks a per-step reference
          // built from these instead of re-discovering the weight shift.
          if (GetNumberOrDefault(0.0, model, "trot_com_commit") > 0.5 && r7cv) {
            const double *stp = SensorByName(model, data,
                                             (i == 0) ? "foot_right_pos" : "foot_left_pos");
            if (stp) {
              residual_.com_ref_p_[0] = stp[0];
              residual_.com_ref_p_[1] = stp[1];
              residual_.com_ref_xi_[0] = com7[0] + r7cv[0] * tau7;
              residual_.com_ref_xi_[1] = com7[1] + r7cv[1] * tau7;
              residual_.com_ref_t0_ = now;
              residual_.com_ref_omega_ = om7;
              residual_.com_ref_stance_ = (i == 0) ? 1 : 0;
              residual_.com_ref_ok_ = true;
            }
          }
          residual_.step_frozen_ok_[i] = true;
        }

        // TOUCHDOWN: loaded foot, was swinging, and past the debounce. The
        // debounce is the reference walker's min_step_time -- without it a
        // single bouncing landing re-triggers the switch several times.
        const bool debounced =
            (residual_.gait_td_time_[i] < 0.0) ||
            (now - residual_.gait_td_time_[i] > kMinStep);
        // gait_lifted_ is the R7b interlock: this leg must have actually left the
        // ground since its last touchdown. A permanently-planted foot (idle drive,
        // held stumble) therefore never produces a touchdown event at all.
        if (r7_contact[i] && residual_.gait_in_swing_[i] &&
            residual_.gait_lifted_[i] && debounced) {
          residual_.gait_in_swing_[i] = false;
          residual_.gait_lifted_[i] = false;      // re-arm on the next unload
          residual_.gait_td_time_[i] = now;
          residual_.gait_overrun_[i] = 0.0;
          residual_.step_frozen_ok_[i] = false;

          // ---- (f) CONTACT PLL ----------------------------------------------
          // This leg has just landed, so by definition it should be at the START
          // of its stance: ph_i == 0. Steer the shared clock offset toward that
          // instead of SNAPPING it -- the contralateral leg is antiphase on the
          // SAME clock, so a hard snap would jump it discontinuously through its
          // own swing. Clamped per event and gained < 1, this converges in 1-2
          // steps while never moving either leg more than trot_phase_snap.
          if (kSwitchOn) {
            double err = r7ph[i];
            if (err > 0.5) err -= 1.0;             // wrap to (-0.5, 0.5]
            const double kSnap =
                GetNumberOrDefault(0.35, model, "trot_phase_snap");   // cycles
            const double kGain =
                GetNumberOrDefault(0.70, model, "trot_phase_gain");   // 0..1
            double corr = r7clip(-err * kGain, -kSnap, kSnap);
            residual_.gait_phase_offset_ =
                std::fmod(residual_.gait_phase_offset_ + corr, 1.0);
            if (residual_.gait_phase_offset_ < 0.0)
              residual_.gait_phase_offset_ += 1.0;
          }
        }

        // GROUND-SEEK OVERRUN: still flagged in swing past the nominal swing
        // duration => the floor is not where the clock assumed. Time-based (not
        // phase-based) so it keeps growing after the phase has wrapped.
        if (residual_.gait_in_swing_[i] && residual_.gait_swing_t0_[i] > 0.0) {
          residual_.gait_overrun_[i] =
              mju_max(0.0, (now - residual_.gait_swing_t0_[i]) - T_sw);
        } else {
          residual_.gait_overrun_[i] = 0.0;
        }
      }
    }
  }

  // Helper: snapshot the scales currently in effect (mid-ramp possible) so
  // the next phase ramps smoothly out of them. The lerp matches what
  // Residual() is doing, so prev_* always equals what the rollouts are
  // actually seeing at the moment of transition.
  auto SnapshotEffectiveScales = [&]() {
    double p_t = PhasePostureScale(residual_.residual_keyframe_.name);
    double dt = mju_max(0.0, data->time - residual_.keyframe_start_time_);
    double alpha_lin =
        mju_min(dt / LeanPhaseRampSeconds(model, ResidualFn::kPhaseRampSeconds),
                1.0);
    // Match the smoothstep used in Residual() so the snapshot equals what
    // the rollouts were actually seeing at the moment of transition.
    double alpha = alpha_lin * alpha_lin * (3.0 - 2.0 * alpha_lin);
    residual_.prev_phase_posture_scale_ =
        residual_.prev_phase_posture_scale_ +
        alpha * (p_t - residual_.prev_phase_posture_scale_);
    // Capture the posture keyframe id of the phase we are LEAVING so Residual()
    // ramps the target pose OUT of it into the new phase (parallels the scale
    // snapshots above). Transitions here fire post-settle (sustain >> ramp) so
    // the leaving keyframe == the effective target; mid-ramp scrubs accept a
    // small target discontinuity, same as the pre-existing scale snapshot.
    int leaving_key = mj_name2id(model, mjOBJ_KEY,
                                 residual_.residual_keyframe_.name.c_str());
    residual_.prev_posture_key_id_ = (leaving_key < 0) ? 0 : leaving_key;
  };

  bool cold_start = !motion_strategy_.HasKeyframes();
  if (cold_start || requested_strategy != current_strategy_) {
    // A live strategy switch (not the first load) eases into the new task:
    // snapshot the scales + weights the rollouts currently see, so the new
    // strategy ramps OUT of them over kPhaseRampSeconds instead of snapping.
    // Reuses the exact machinery the manual Phase scrubber uses, so a
    // task-select button press transitions calmly rather than lurching.
    if (!cold_start) {
      SnapshotEffectiveScales();
      SnapshotCurrentWeightsAsPrev();
    }
    current_strategy_ = requested_strategy;
    motion_strategy_.ClearKeyframes();
    motion_strategy_.LoadStrategy(kStrategyNames[current_strategy_],
                                  kLocoStrategyFilePath);
    // Bench contract (feedback_bench_task_strategy_pairing): every headless bench
    // asserts `[STRAT n:name]` in node.log to prove the task/strategy pairing it
    // scored. Beginning printed it on each debug tick; here once per load.
    printf("  [STRAT %d:%s | kf %d/%d]\n", current_strategy_,
           kStrategyNames[current_strategy_].c_str(), 1,
           motion_strategy_.GetKeyframesCount());
    fflush(stdout);
    // Gate the target-pose ramp (Residual): >1 phase => cyclic strategy gets the
    // smooth transitions; 1 => single-phase, the ramp stays disabled (byte-identical).
    residual_.num_phases_ = motion_strategy_.GetKeyframesCount();
    motion_strategy_.SetCurrentKeyframeStartTime(data->time);
    motion_strategy_.SetCurrentKeyframeSuccessStartTime(data->time);
    residual_.residual_keyframe_ = motion_strategy_.GetCurrentKeyframe();
    residual_.keyframe_start_time_ = data->time;
    // ★ Pin the Foot Stability anchor to the MEASURED stance on entry to a
    // STEPPING keyframe (trot/walk/drive). The residual's home constants are
    // odometric-frame and that frame drifts, so anchoring to where the feet
    // ACTUALLY are at hand-over is the whole fix (lower-body: cost 225 -> 0.00
    // on real, 2026-07-18). Feet are planted at entry, so this is a clean
    // capture. Non-stepping keyframes (incl. strat 6 stand) never touch it.
    // ⚠ NO COLD-START PIN. The first version pinned HERE, at strategy entry --
    // i.e. before the state estimate is initialised -- and only null-checked the
    // sensor pointers. On the real robot that DRAGGED IT SIDEWAYS at startup
    // (uninitialised read => both anchors near the origin => the weight-8 cost
    // hauls both feet toward it). The anchor now starts at the legacy constants
    // and is only ever pinned by the drive-idle path below, which is gated on
    // settle time AND plausibility. Left as a note so nobody re-adds it.
    // STRAIGHTEN (strat 25) live-seed (C3): capture the release pose ONCE here on
    // the TRUE agent state (never per-rollout), so Residual ramps the upright +
    // posture targets FROM it to nominal along a min-jerk — the "ramped reference
    // IS the funnel" that keeps the corrective force small and prevents the
    // static-target overshoot. Fires on cold boot into straighten AND on a live
    // switch into it (mid-mission re-straighten). Every other strategy leaves
    // straighten_seeded_ = false -> byte-identical.
    if (residual_.residual_keyframe_.name.rfind("straighten", 0) == 0) {
      int nq = mju_min(model->nq, 64);
      for (int i = 0; i < nq; i++)
        residual_.straighten_start_qpos_[i] = data->qpos[i];
      double *pu = SensorByName(model, data, "pelvis_up");
      double up_z = pu ? mju_max(-1.0, mju_min(1.0, pu[2])) : 1.0;
      residual_.straighten_start_tilt_ = std::acos(up_z);
      residual_.straighten_seeded_ = true;
      std::fprintf(stderr,
                   "[loco-straighten] seeded release: tilt=%.1fdeg base_z=%.3f\n",
                   residual_.straighten_start_tilt_ * 180.0 / M_PI,
                   data->qpos[2]);
    } else {
      residual_.straighten_seeded_ = false;
    }
    if (cold_start) {
      // First load: no history to ramp from. Posture scale starts at 1.0
      // ("no boost") and ramps to the phase target over kPhaseRampSeconds.
      residual_.prev_phase_posture_scale_      = 1.0;
      residual_.prev_posture_key_id_           = 0;  // home: gentle boot ramp
      PrepareNextPhaseWeights(residual_.residual_keyframe_);
      prev_phase_weights_ = next_phase_weights_;  // snap (no history)
    } else {
      // Live switch: prev_* scales + prev_phase_weights_ already hold the
      // current effective values (snapshotted above). Only set the new
      // targets; ApplyRampedWeights + the Residual() scale ramp carry the
      // transition over kPhaseRampSeconds (alpha keyed off keyframe_start_time_).
      PrepareNextPhaseWeights(residual_.residual_keyframe_);
    }
    ApplyRampedWeights(model, data);
    return;
  }

  // ----- Manual phase scrubber ---------------------------------------- //
  // residual_Phase parameter:
  //   -1  → auto-advance (existing behaviour, runs below)
  //   0..N-1 → hold at that keyframe index regardless of progress;
  //            auto-advance disabled. Body state is preserved across
  //            manual jumps — only the active keyframe changes, with
  //            weights smoothstep-ramping into the new phase (the same
  //            machinery the auto-advance uses).
  int requested_phase =
      (int)std::round(parameters[kLocoPhaseParameterIndex]);
  bool manual_phase_mode = (requested_phase >= 0);
  if (manual_phase_mode) {
    int n_kf = motion_strategy_.GetKeyframesCount();
    requested_phase = std::max(0, std::min(requested_phase, n_kf - 1));
    int current_kf_idx = motion_strategy_.GetCurrentKeyframeIndex();
    if (requested_phase != current_kf_idx) {
      // Manual phase jump — same snapshot+ramp dance as auto-advance, so
      // weights blend smoothly from the old phase's effective values into
      // the new phase's targets over kPhaseRampSeconds. Body qpos/qvel
      // are untouched; only the active keyframe (contact targets, brace
      // force target, weight overrides) changes.
      SnapshotEffectiveScales();
      SnapshotCurrentWeightsAsPrev();
      motion_strategy_.UpdateCurrentKeyframe(requested_phase);
      residual_.residual_keyframe_ = motion_strategy_.GetCurrentKeyframe();
      motion_strategy_.SetCurrentKeyframeStartTime(data->time);
      motion_strategy_.SetCurrentKeyframeSuccessStartTime(data->time);
      residual_.keyframe_start_time_ = data->time;
      PrepareNextPhaseWeights(residual_.residual_keyframe_);
    }
    // Skip auto-advance entirely while in manual mode.
  } else {
    const mjpc::humanoid::ContactKeyframe& current_kf =
        motion_strategy_.GetCurrentKeyframe();
    const double total_distance =
        motion_strategy_.CalculateTotalKeyframeDistance(
            data, mjpc::humanoid::ContactKeyframeErrorType::kNorm);

    // ---- STRAIGHTEN->stand basin-containment gate (2026-07-11) --------- //
    // Lean keyframes carry no contact pairs, so total_distance == 0 always
    // and the bare rule below would advance the straighten bring-up phase on
    // a PURE TIMER (Unitree-style blind ramp). For the handoff to the stand
    // phase we instead require the robot to actually BE inside the stand
    // basin on the TRUE agent state: upright (pelvis tilt), tall (base z,
    // i.e. unslumped knees), and quiescent (base speed) -- sustained
    // success_sustain_time like any other phase, re-armed on violation.
    // Name-gated on the straighten keyframe -> every other strategy's
    // advance logic is byte-identical.
    bool straighten_gate_ok = true;
    if (current_kf.name.rfind("straighten", 0) == 0) {
      double *pu = SensorByName(model, data, "pelvis_up");
      double up_z = pu ? mju_max(-1.0, mju_min(1.0, pu[2])) : 1.0;
      double tilt_deg = std::acos(up_z) * 180.0 / M_PI;
      double base_spd = std::sqrt(data->qvel[0] * data->qvel[0] +
                                  data->qvel[1] * data->qvel[1] +
                                  data->qvel[2] * data->qvel[2]);
      // TIGHTENED 2026-07-11 after the first REAL run: the old bar (z>=0.95,
      // no knee condition) released from a knee-0.76 crouch at t=1.6s -- phase 0
      // never actually fixed the pose, and the weak stand-phase weights
      // (Balance 2.5 / Posture 12) inherited an unfinished rise mid-bring-up.
      // The gate's contract is "the pose IS ideal": full height (1.00 vs the
      // 1.02 keyframe), legs extended (<=0.50 rad vs the 0.35 target; knee
      // encoders carry no per-power-on zero error, unlike ankles), upright and
      // quiet. Until then phase 0's high-authority weights (Balance 40 /
      // Posture 35 / Pelvis Tilt 150) keep owning the straightening.
      straighten_gate_ok = tilt_deg <= 3.0 && data->qpos[2] >= 1.00 &&
                           base_spd <= 0.15 &&
                           data->qpos[10] <= 0.50 &&   // left knee
                           data->qpos[16] <= 0.50;     // right knee
    }

    if (data->time - motion_strategy_.GetCurrentKeyframeStartTime() >
            current_kf.time_limit &&
        total_distance > current_kf.target_distance_tolerance) {
      // Time-limit reset (strategy restarts from keyframe 0). Save the
      // scales that were just in effect so the next ramp blends from them.
      SnapshotEffectiveScales();
      SnapshotCurrentWeightsAsPrev();
      motion_strategy_.Reset();
      residual_.residual_keyframe_ = motion_strategy_.GetCurrentKeyframe();
      motion_strategy_.SetCurrentKeyframeStartTime(data->time);
      motion_strategy_.SetCurrentKeyframeSuccessStartTime(data->time);
      residual_.keyframe_start_time_ = data->time;
      PrepareNextPhaseWeights(residual_.residual_keyframe_);
    } else if (straighten_gate_ok &&
               total_distance <= current_kf.target_distance_tolerance &&
               data->time -
                       motion_strategy_.GetCurrentKeyframeSuccessStartTime() >
                   current_kf.success_sustain_time) {
      // Normal phase advance — this is the path that fires after stand_up
      // succeeds. Snapshot first so the new ramp starts from the old scales.
      if (current_kf.name.rfind("straighten", 0) == 0) {
        std::fprintf(stderr,
                     "[loco-straighten] basin gate PASS (z=%.3f kneeL=%.2f "
                     "kneeR=%.2f) -> advancing to stand phase\n",
                     data->qpos[2], data->qpos[10], data->qpos[16]);
      }
      SnapshotEffectiveScales();
      SnapshotCurrentWeightsAsPrev();
      motion_strategy_.NextKeyframe();
      residual_.residual_keyframe_ = motion_strategy_.GetCurrentKeyframe();
      motion_strategy_.SetCurrentKeyframeStartTime(data->time);
      motion_strategy_.SetCurrentKeyframeSuccessStartTime(data->time);
      residual_.keyframe_start_time_ = data->time;
      PrepareNextPhaseWeights(residual_.residual_keyframe_);
    } else if (total_distance > current_kf.target_distance_tolerance ||
               !straighten_gate_ok) {
      // Re-arm the success clock: outside tolerance, or (straighten only)
      // the basin gate is violated -- sustain must be CONSECUTIVE.
      motion_strategy_.SetCurrentKeyframeSuccessStartTime(data->time);
    }
  }

  // Apply the smoothstep weight ramp every tick. This also propagates the
  // current target weights into weight[] when no phase advance fired, which
  // is what makes the live GUI cost sliders mirror the JSON values.
  ApplyRampedWeights(model, data);
}

// ---------------- Live per-phase weight blending --------------------- //
// All loops over weight[] must be bounded by `weight_names.size()` (== the
// actual number of user-sensor cost terms), NOT `weight.size()` — the Task
// base class resizes `weight` to `kMaxCostTerms` (128) regardless of the
// real term count, while `weight_names` is sized to `num_term`. Indexing
// `weight_names` past num_term reads uninitialised std::strings and
// segfaults when the JSON map then calls .find(garbage).

// Snapshot the XML default weights from sensor user data. Called once per
// Reset so the fallback for "JSON omitted this residual" stays in sync with
// the static XML weights (which Task::Reset itself populated into weight[]).
void loco::SnapshotXmlDefaultWeights(const mjModel *model) {
  const std::size_t n = weight_names.size();
  xml_default_weights_.assign(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) xml_default_weights_[i] = weight[i];
}

// Build next_phase_weights_ from this phase's JSON weight map. Missing keys
// fall back to the XML default snapshot. Unknown keys (typos) are silently
// skipped — same forgiving behaviour as interact.cc.
void loco::PrepareNextPhaseWeights(const mjpc::humanoid::ContactKeyframe &kf) {
  const std::size_t n = weight_names.size();
  if (next_phase_weights_.size() != n) next_phase_weights_.assign(n, 0.0);
  if (xml_default_weights_.size() != n) xml_default_weights_.assign(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    const std::string &name = weight_names[i];
    auto it = kf.weight.find(name);
    next_phase_weights_[i] =
        (it != kf.weight.end()) ? it->second : xml_default_weights_[i];
  }
}

void loco::SnapshotCurrentWeightsAsPrev() {
  const std::size_t n = weight_names.size();
  prev_phase_weights_.assign(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) prev_phase_weights_[i] = weight[i];
}

// Lerp weight[] from prev → next using the same smoothstep curve the
// residual uses for its phase scales, so the rollouts' cost surface evolves
// continuously across phase boundaries.
// 2026-07-28: shared resolver for the phase ramp so the WEIGHT ramp and the residual
// scale ramp cannot drift apart. `phase_ramp_sec` numeric: 0 = OFF = byte-identical.
static double LeanPhaseRampSeconds(const mjModel *model, double fallback) {
  int pr_id = mj_name2id(model, mjOBJ_NUMERIC, "phase_ramp_sec");
  double pr = (pr_id >= 0) ? model->numeric_data[model->numeric_adr[pr_id]] : 0.0;
  return (pr > 1e-9) ? pr : fallback;
}

void loco::ApplyRampedWeights(const mjModel *model, const mjData *data) {
  const std::size_t n = weight_names.size();
  if (prev_phase_weights_.size() != n || next_phase_weights_.size() != n) {
    return;
  }
  double dt = mju_max(0.0, data->time - residual_.keyframe_start_time_);
  double alpha_lin = mju_min(
      dt / LeanPhaseRampSeconds(model, ResidualFn::kPhaseRampSeconds), 1.0);
  double alpha = alpha_lin * alpha_lin * (3.0 - 2.0 * alpha_lin);
  for (std::size_t i = 0; i < n; ++i) {
    weight[i] = prev_phase_weights_[i] +
                alpha * (next_phase_weights_[i] - prev_phase_weights_[i]);
  }
}

void loco::ResetLocked(const mjModel *model) {
  // Capture the XML default weights AFTER Task::Reset has populated weight[]
  // from sensor user data. These act as the fallback for any residual the
  // current phase's JSON doesn't override, and as the prev/next initialisers
  // before the first phase transition. All three vectors are sized to
  // weight_names.size() (== num_term), NOT weight.size() (== kMaxCostTerms).
  SnapshotXmlDefaultWeights(model);
  prev_phase_weights_ = xml_default_weights_;
  next_phase_weights_ = xml_default_weights_;
  if (!residual_.residual_keyframe_.weight.empty()) {
    PrepareNextPhaseWeights(residual_.residual_keyframe_);
  }

}

// ============================================================================
// ModifyControl — CAPTURE-POINT FOOTSTEP CONTROLLER for the TROT strategy
// (slot 23, phase "stumble_trot"). Called by Task::ModifyControl after the
// policy fills ctrl and before mj_step, in EVERY rollout (NoisyRollout) AND on
// the executed action (CEM ActionFromPolicy). Open-loop drives ONLY the swing
// leg: lift for clearance mid-swing, then PLACE the foot at the capture point
//   xi = v * sqrt(z/g)   (inverted-pendulum "catch" spot)
// by touchdown, so single-support is a PLANNED CATCH, not an unguided fall. The
// sampler still owns the stance leg + torso (balance). A desired CoM velocity
// (numerics trot_des_vel_x/y, default 0) biases the step: 0 = step in place
// (trot/stumble), >0 = WALK forward -- ONE generalized controller. is_trot-
// gated: strat 20 + every other strategy byte-identical. ctrl[i] == joint
// target for qpos[7+i] (position-servo). FK-measured gains (twin model): +hip_
// pitch -> foot BACK (0.80/rad), +hip_roll -> foot +y (0.79/rad), both legs.
// ============================================================================
void loco::ModifyControl(const mjModel *model, const double *qpos,
                         const double *qvel, double time, double *ctrl) const {
  const std::string &kfname = residual_.residual_keyframe_.name;
  const bool is_trot = (kfname.rfind("stumble", 0) == 0) &&
                       (kfname.find("trot") != std::string::npos);
  const bool is_drive = is_trot &&
                        (kfname.find("drive") != std::string::npos);
  // ---- ★ STUMBLE CATCH-STEP FORCER (stumble_swing_force, default 0 = OFF) --
  // 2026-09-08. THIS EARLY RETURN IS WHY THE CATCH-STEP NEVER STEPS.
  // Everything below is the OPEN-LOOP SWING WRITER: it drives the swing leg's
  // hip/knee/ankle position targets directly (lift bell + Raibert placement
  // ramp) instead of hoping the sampler discovers them. Strategy 20's keyframe
  // is `stumble_march`, which does not contain "trot", so is_trot is false and
  // strat 20 has ALWAYS returned here -- its catch-step is COST-ONLY.
  //
  // The cost side already asks for the step: the residual computes the same
  // gait clock (GaitPhases with the same contact-PLL offset), the same swing
  // bells g_bump_l/g_bump_r, and gates them by g_amp = arm*recov -- so when
  // `danger` crosses catch_trig the cost genuinely wants an alternating march.
  // What it is asking AGAINST is Foot Right/Left Up at weight 2000 and Posture
  // at 12, and the sampler has to find the whole lift-swing-place trajectory
  // inside its horizon by random search. It essentially never does: replaying
  // 94 recorded runs on 2026-09-07 found 2 genuine steps, and all 2 were
  // lateral. This file's own drive-24 comment (a dozen lines below) records the
  // same failure from the other side -- "the sampler alone would not lift the
  // catch foot and forward momentum toppled the robot" -- and the fix there was
  // to keep THIS forcer alive through the disengage. Strat 20 never had it.
  //
  // So: run the forcer for stumble too, but ONLY while the catch is armed.
  // g_amp is multiplied by the same smoothstep `recov` the cost uses, so below
  // catch_trig g_amp is 0 and the function returns two lines later -- the quiet
  // stand, and every validated 2026-09-06 push-recovery number, is untouched at
  // the default 0.0. stumble_swing_force scales the forced amplitude (1.0 = the
  // full trot swing), so it is a strength knob, not just an on/off.
  const bool is_stumble_kf = (kfname.rfind("stumble", 0) == 0);
  const double kStumbleForce =
      (is_stumble_kf && !is_trot)
          ? GetNumberOrDefault(0.0, model, "stumble_swing_force") : 0.0;
  // ---- ★ UNWEIGHT PULSE (catch_unweight [rad], default 0 = OFF) -----------
  // WHY THIS EXISTS. Measured 2026-09-08, the fore/aft step is TIME-BOUND, and
  // the bound is on the requirement, not the budget:
  //   need  = 0.43 s lateral weight transfer + 0.36 s swing = 0.79 s
  //   have  = 0.40 s today; 0.65 s after the lean_nominal calibration fix at the
  //           lowest threshold that does not false-fire on a quiet stand.
  // The 0.43 s is irreducible for a CoP-driven transfer: moving the CoM by one
  // half-stance d at the polygon-bounded a = d*g/z takes sqrt(2d/a) = sqrt(2z/g)
  // = 0.43 s -- note d CANCELS, so narrowing or widening the stance cannot help.
  // Lowering the trigger cannot buy the last 0.14 s either: the floor is set by
  // the quiet-stand danger, not by choice.
  // So reduce the REQUIREMENT. A foot does not need the CoM fully over the other
  // foot; it needs its own load low enough to leave the ground. Flexing both
  // knees drops the CoM, and while it accelerates downward at `a` every foot
  // load falls by (1 - a/g). Halving the load halves the lateral transfer the
  // step needs -> 0.30 s instead of 0.43 s. Crucially this is knee flexion, a
  // ~0.15 s action, NOT pendulum-limited -- it is the one lever in this system
  // that is not tied to sqrt(z/g). It is also exactly how humans initiate a fast
  // step (the unweighting phase of an anticipatory postural adjustment).
  // Half-sine over catch_unweight_dur from the catch onset, so it is a transient
  // DIP, not a crouch: a sustained lower target would just stand lower and
  // unweight nothing (the unweighting is in the ACCELERATION, not the height).
  // Runs BEFORE the swing-forcer early return so it can be tested with the
  // forcer off. Default 0 => byte-identical.
  if (is_stumble_kf && !is_trot && residual_.catch_onset_time_ > 0.0) {
    const double kUW = GetNumberOrDefault(0.0, model, "catch_unweight");
    const double kUWD = mju_max(0.05,
        GetNumberOrDefault(0.20, model, "catch_unweight_dur"));
    const double te = time - residual_.catch_onset_time_;
    if (kUW > 1e-6 && te >= 0.0 && te < kUWD) {
      const double pw = std::sin(3.14159265358979 * te / kUWD);  // 0 -> 1 -> 0
      const int knee_ch[2] = {3, 9};                 // L knee, R knee
      for (int j = 0; j < 2; j++) {
        const int ch = knee_ch[j];
        if (ch >= model->nu) continue;
        double v = ctrl[ch] + kUW * pw;              // + = flex = shank folds up
        const double lo = model->actuator_ctrlrange[2 * ch];
        const double hi = model->actuator_ctrlrange[2 * ch + 1];
        ctrl[ch] = mju_max(lo, mju_min(hi, v));
      }
    }
  }

  if ((!is_trot && kStumbleForce <= 1e-6) || model->nu < 11) return;
  auto clip = [](double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
  };

  // continuous forced-march amplitude (matches the residual `arm`, kArmSec 2.0)
  constexpr double kArmSec = 2.0;
  double t_phase = mju_max(0.0, time - residual_.keyframe_start_time_);
  double g_amp = mju_min(t_phase / kArmSec, 1.0);
  g_amp = g_amp * g_amp * (3.0 - 2.0 * g_amp);
  // DRIVE (strat 24): the stand<->trot latch gates the OPEN-LOOP swing, BUT the
  // balance CATCH-STEP must survive a disengage. R1 fix (2026-07-08): gating by
  // drive_gait_amp_ ALONE quit the forcer the instant the walk latch ramped to 0
  // on release, WHILE the cost (residual line ~571, arm*max(drive_gait_amp_,
  // recov)) still wanted the catch -- so the sampler alone would not lift the
  // catch foot and forward momentum toppled the robot (walk->stop faceplant).
  // Mirror the cost: gate by max(drive_gait_amp_, recov), computing the SAME
  // signed capture-point danger as the residual from a base-quat tilt + base
  // qvel proxy (ModifyControl has no mjData/sensors, and the capture step below
  // already uses base qvel -- so this is consistent). Quiet idle: danger <
  // catch_trig -> recov 0 -> g_amp 0 -> early-return (idle stand byte-identical);
  // a push OR a come-to-rest tip -> recov > 0 -> the swing fires the come-to-rest
  // capture step (cap_g*tau*qvel, computed below), catching the fall.
  if (is_drive || !is_trot) {
    double zc = mju_max(0.5, qpos[2]);
    double tauc = mju_sqrt(zc / 9.81);
    // base up-axis (x,y) from the pelvis free-joint quat qpos[3:7]=(w,x,y,z);
    // matches the up_z = 1-2(x^2+y^2) convention used by the standing gate below.
    double qw = qpos[3], qxx = qpos[4], qyy = qpos[5], qzz = qpos[6];
    double up_x = 2.0 * (qxx * qzz + qw * qyy);
    double up_y = 2.0 * (qyy * qzz - qw * qxx);
    int lnx = mj_name2id(model, mjOBJ_NUMERIC, "lean_nominal_x");
    double kLocoX =
        (lnx >= 0) ? model->numeric_data[model->numeric_adr[lnx]] : 0.06;
    // MUST match the residual's stumble-scoped override or cost and swing writer
    // disagree about where equilibrium is.
    if (!is_trot) {
      double so3 = GetNumberOrDefault(-1.0, model, "stumble_lean_nominal");
      if (so3 >= 0.0) kLocoX = so3;
    }
    double tx = up_x - kLocoX, ty = up_y;             // tilt rel. steady lean
    double ex = zc * tx + tauc * qvel[0];             // signed fore-aft capture
    double eyp = zc * ty;                             // lateral: tilt only
    double danger = mju_sqrt(ex * ex + eyp * eyp);
    int ct = mj_name2id(model, mjOBJ_NUMERIC, "catch_trig");
    int cf = mj_name2id(model, mjOBJ_NUMERIC, "catch_full");
    double kCT = (ct >= 0) ? model->numeric_data[model->numeric_adr[ct]] : 0.085;
    double kCF = (cf >= 0) ? model->numeric_data[model->numeric_adr[cf]] : 0.16;
    // MUST mirror the residual's bias shift or the cost and the swing writer
    // disagree about when the catch is on -- the exact class of bug the
    // cmd_active_ fix of 2026-07-12 had to undo.
    if (residual_.catch_bias_on_) {
      const double width = mju_max(1e-3, kCF - kCT);
      kCT = GetNumberOrDefault(kCT, model, "catch_bias_trig");
      kCF = kCT + width;
    }
    double recov = mju_min(
        1.0, mju_max(0.0, (danger - kCT) / mju_max(1e-3, kCF - kCT)));
    recov = recov * recov * (3.0 - 2.0 * recov);      // smoothstep (matches cost)
    // mirror the residual's committed-pair hold, or the cost would keep asking
    // for the second step while this writer had already stopped driving it.
    if (residual_.catch_hold_until_ > 0.0 && time < residual_.catch_hold_until_)
      recov = mju_max(recov, residual_.catch_hold_recov_);
    // drive: the walk latch OR a catch keeps the swing alive.
    // stumble: the catch ALONE -- recov == 0 in a quiet stand, so g_amp falls to
    // 0 and the next line returns, leaving the validated still-stand exactly as
    // it was. kStumbleForce scales how hard the forced swing is driven.
    g_amp *= is_drive ? mju_max(residual_.drive_gait_amp_, recov)
                      : recov * kStumbleForce;
  }
  if (g_amp <= 1e-4) return;                     // still settling -> planner owns

  // ★ R7: gait clock from the SHARED GaitPhases helper -- the same call the
  // residual makes, with the same contact-PLL offset, so the cost and this
  // open-loop swing writer can no longer drift apart. (is_trot is true here by
  // the early return above, matching the residual's is_trot-gated duty lookup.)
  // ★ pass the REAL is_trot (was hard-coded true, correct only while this
  // function was unreachable for stumble): trot_duty is is_trot-gated inside
  // GaitPhases, so a nonzero trot_duty would otherwise give the stumble forcer a
  // different duty than the stumble COST -- cost and swing writer must resolve
  // the same clock or they fight. (trot_duty is 0 today, so this is a no-op now
  // and a trap disarmed for later.)
  const GaitClock gc = GaitPhases(model, time, residual_.gait_phase_offset_, is_trot);
  const double kCad = gc.cad, kDuty = gc.duty;
  double ph_l = gc.ph_l;
  double ph_r = gc.ph_r;

  // ---- CAPTURE POINT (instantaneous, inverted-pendulum) -------------------
  // base lin-vel (qvel[0:2], world) ~= CoM vel; step to xi = (v - v_des)*tau to
  // CATCH while tracking the desired velocity. v_des>0 -> net forward travel.
  double z = mju_max(0.5, qpos[2]);
  double tau = mju_sqrt(z / 9.81);
  int dvx_id = mj_name2id(model, mjOBJ_NUMERIC, "trot_des_vel_x");
  int dvy_id = mj_name2id(model, mjOBJ_NUMERIC, "trot_des_vel_y");
  double dvx = (dvx_id >= 0) ? model->numeric_data[model->numeric_adr[dvx_id]] : 0.0;
  double dvy = (dvy_id >= 0) ? model->numeric_data[model->numeric_adr[dvy_id]] : 0.0;
  // LIVE teleop override (2026-07-03; mirrored in the residual -- MUST match).
  if (residual_.cmd_active_) {
    dvx = residual_.cmd_vdes_world_[0];
    dvy = residual_.cmd_vdes_world_[1];
  }
  // The residual reads trot_des_vel_x/y ONLY when is_trot; here they were read
  // unconditionally, which was harmless while stumble could not reach this code.
  // Now it can, so mirror the cost exactly: strategy 20 steps IN PLACE, and a
  // stray nonzero trot_des_vel_* must never turn a catch-step into a walk.
  if (!is_trot) { dvx = 0.0; dvy = 0.0; }
  // STEP-AND-SETTLE pulse (MUST match the residual's pulse, same data time): walk
  // for trot_step_walk s, settle (v_des=0 -> robust in-place trot) the rest of
  // trot_step_period s. Tp<=0 => continuous (byte-identical).
  {
    int tp_id = mj_name2id(model, mjOBJ_NUMERIC, "trot_step_period");
    int tw_id = mj_name2id(model, mjOBJ_NUMERIC, "trot_step_walk");
    double Tp = (tp_id >= 0) ? model->numeric_data[model->numeric_adr[tp_id]] : 0.0;
    double Tw = (tw_id >= 0) ? model->numeric_data[model->numeric_adr[tw_id]] : 0.0;
    if (Tp > 1e-6 && std::fmod(mju_max(0.0, time), Tp) >= Tw) { dvx = 0.0; dvy = 0.0; }
  }
  // R6 SETTLE GOVERNOR (trot_settle_thresh [m], default 0 = off = byte-
  // identical): the REACTIVE version of the step-and-settle pulse. When the
  // velocity-error capture offset |v - v_des|*tau exceeds the band, fade v_des
  // to 0 -- all walk terms gate off and the controller auto-reverts to the
  // validated-robust in-place trot until the error re-enters the band. Fade is
  // smooth (1 below thresh -> 0 at 1.5x) so there is no command chatter at the
  // boundary. MUST match the residual-side governor (same qpos/qvel/time
  // inputs -> deterministic cost/swing agreement, the step-pulse pattern).
  {
    // drive-scoped (see the residual-side twin of this block): strat 24 reads
    // drive_settle_thresh, strat 23 keeps trot_settle_thresh=0 (byte-identical).
    const char *st_name = is_drive ? "drive_settle_thresh" : "trot_settle_thresh";
    int st_id = mj_name2id(model, mjOBJ_NUMERIC, st_name);
    double st = (st_id >= 0) ? model->numeric_data[model->numeric_adr[st_id]] : 0.0;
    if (st > 1e-6 && (dvx != 0.0 || dvy != 0.0)) {
      double gex = (qvel[0] - dvx) * tau, gey = (qvel[1] - dvy) * tau;
      double gerr = mju_sqrt(gex * gex + gey * gey);
      double gg = clip((1.5 * st - gerr) / (0.5 * st), 0.0, 1.0);
      gg = gg * gg * (3.0 - 2.0 * gg);
      dvx *= gg; dvy *= gg;
    }
  }
  // capture gain (live numeric, default 1.0 = deadbeat one-step capture): >1
  // over-steps (catches harder, kills velocity faster), <1 under-steps. LATERAL
  // gets its OWN gain (trot_cap_gain_lat, default = fore-aft gain) -- the sideways
  // axis is the narrow-base biped's weak axis and usually needs a harder catch.
  int cg_id = mj_name2id(model, mjOBJ_NUMERIC, "trot_cap_gain");
  double cap_g = (cg_id >= 0) ? model->numeric_data[model->numeric_adr[cg_id]] : 1.0;
  int cgl_id = mj_name2id(model, mjOBJ_NUMERIC, "trot_cap_gain_lat");
  double cap_gl = (cgl_id >= 0) ? model->numeric_data[model->numeric_adr[cgl_id]] : cap_g;
  // trot_walk_gain (kw) = gain on the Raibert NEUTRAL (velocity-maintaining) step
  // (T_st/2)*v_des; default 1.0 = the theoretical sustaining placement. v_des=0
  // makes this term vanish, so in-place trot is byte-identical regardless of kw.
  int wg_id = mj_name2id(model, mjOBJ_NUMERIC, "trot_walk_gain");
  double kw = (wg_id >= 0) ? model->numeric_data[model->numeric_adr[wg_id]] : 1.0;
  // STANDING GATE (MJPC humanoid-walk's core robustness trick, walk.cc:91-95):
  // scale the forward PROPULSION by an uprightness factor that -> 0 as the torso
  // tips, so a tipping robot STOPS pushing forward and the pure capture CATCH
  // (cap_g*v*tau, ungated) recovers it -- instead of driving itself into a
  // forward faceplant. up_z = torso up-axis z from the base quat (1=upright).
  double qx = qpos[4], qy = qpos[5];
  double up_z = 1.0 - 2.0 * (qx * qx + qy * qy);
  double standing = clip((up_z - 0.80) / (0.97 - 0.80), 0.0, 1.0);
  standing = standing * standing * (3.0 - 2.0 * standing);  // smoothstep
  // FOOT PLACEMENT = Raibert/DCM, NOT the come-to-rest capture point. Research
  // (MJPC walk.cc, Playground G1/H1, Unitree RL, Khadiv DCM) verdict: placing the
  // foot at the full capture point xi = v*tau = v/omega is a STOP target (~0.32*v
  // for us); the VELOCITY-SUSTAINING step is only (T_st/2)*v (~0.15*v) -- stepping
  // to the full capture point is ~2x too long, so every step bleeds forward
  // momentum and the CoM drops behind the advancing feet -> backward fall (our
  // 6-10 s symptom). Raibert:  foot_x = (T_st/2)*v_des + k*(v - v_des), k=cap_g*tau
  //   neutral term kw*(T_st/2)*v_des = velocity-MAINTAINING placement (feedfwd)
  //   error  term cap_g*tau*(v - v_des) = balance CATCH on the velocity error.
  // standing-gated: upright -> full Raibert (walks); tipping (standing->0) -> the
  // v_des terms drop and it collapses to the pure come-to-rest catch cap_g*tau*v
  // (= validated in-place trot recovery, no forward faceplant). v_des=0 makes
  // EVERY v_des term vanish -> byte-identical to the 9/9 in-place trot.
  double T_st = (kCad > 1e-6) ? (kDuty / kCad) : 0.5;   // single-support duration
  double step_x = clip(cap_g * tau * (qvel[0] - standing * dvx) +
                       kw * standing * dvx * (0.5 * T_st), -0.30, 0.30);  // m
  double step_y = clip(cap_gl * (qvel[1] - dvy) * tau, -0.12, 0.12);  // tight=stabler

  // ---- ★ R7 CoP ANCHOR (trot_cop_anchor, default 0 = OFF) ----------------
  // The Raibert error term above is already algebraically identical to the
  // reference DCM law's error term: cap_g*tau*(v - v_des) == (v - v_des)/omega
  // at cap_g = 1. What differs is the ANCHOR the offset is measured from. The
  // DCM law p = xi - xi_dot/omega - K_xi(xi - xi_des), at K_xi = -1 and with the
  // LIPM relation xdd = omega^2 (x - p_cop), collapses to
  //        p = (x - a/omega^2) + tau*(v - v_des),
  // so the anchor is the MEASURED instantaneous CoP, x - a/omega^2. Our cost
  // side anchors at the CoM plus a hard-coded stance_off_x = 0.13, which is that
  // term frozen at a constant. During the weight shift the CoM fore-aft accel is
  // NOT zero, so the correct anchor moves by a/omega^2 -- and because that
  // motion is gait-phase-locked it reads as periodic jiggle rather than noise.
  // Add the correction here as a placement DELTA (the cost side applies the same
  // delta to its Step Place target, so both readers stay consistent).
  // com_acc_filt_ is a LOW-PASSED derivative latched in TransitionLocked: the raw
  // finite difference of the CoM velocity sensor is far too noisy to place a
  // foot with, which is why this is gated off by default until it is benched.
  const double kCopOn = GetNumberOrDefault(0.0, model, "trot_cop_anchor");
  if (kCopOn > 1e-6) {
    const double om2 = 9.81 / mju_max(0.5, qpos[2]);   // omega^2 = g/z
    // -a/omega^2, clamped: a bad accel estimate must never command a lunge.
    double cop_dx = clip(-kCopOn * residual_.com_acc_filt_[0] / om2, -0.10, 0.10);
    double cop_dy = clip(-kCopOn * residual_.com_acc_filt_[1] / om2, -0.06, 0.06);
    step_x = clip(step_x + cop_dx, -0.30, 0.30);
    step_y = clip(step_y + cop_dy, -0.12, 0.12);
  }

  // ---- ★ R7 FROZEN PLACEMENT (trot_freeze_place, default 0 = OFF) --------
  // Recomputing step_x/step_y from live qvel on every tick makes the swing foot
  // chase a MOVING setpoint; combined with the smoothstep ramp `pl` below, the
  // commanded foot path is not monotone and reverses mid-swing whenever v
  // fluctuates. The reference walker freezes its target at leg switch
  // (get_target_0) and only tracks it thereafter. step_frozen_ is latched at
  // swing start in TransitionLocked; per leg, so L and R can be in different
  // parts of their cycle. Falls back to the live value whenever the latch has
  // not fired yet (step_frozen_ok_ false), which is also the OFF path.
  const bool kFreeze = GetNumberOrDefault(0.0, model, "trot_freeze_place") > 0.5;
  double step_x_l = step_x, step_y_l = step_y;   // per-leg placement (L)
  double step_x_r = step_x, step_y_r = step_y;   //                   (R)
  if (kFreeze) {
    if (residual_.step_frozen_ok_[0]) {
      step_x_l = residual_.step_frozen_[0];
      step_y_l = residual_.step_frozen_[1];
    }
    if (residual_.step_frozen_ok_[1]) {
      step_x_r = residual_.step_frozen_[2];
      step_y_r = residual_.step_frozen_[3];
    }
  }
  // swing-height scale (live numeric, default 1.0): LOWER lift = smaller per-step
  // disturbance = stabler trot (trades visible clearance for hold-rate).
  int sh_id = mj_name2id(model, mjOBJ_NUMERIC, "trot_swing_h");
  double sh = (sh_id >= 0) ? model->numeric_data[model->numeric_adr[sh_id]] : 1.0;
  // R4 TOUCHDOWN RELEASE (trot_release, default 0 = off = byte-identical): fade
  // the swing script's authority over the LAST `rel` fraction of the swing so
  // the SAMPLER loads the landing leg instead of the script holding placement
  // targets through touchdown (the real-robot "pushes off fine, lands weak"
  // asymmetry, 2026-06-30: a landing foot being POSITION-driven cannot be
  // LOAD-driven). pl saturates by s~0.85 anyway, so releasing costs no accuracy;
  // mirrors the 15% blend-IN at liftoff. Cost side needs no twin change -- the
  // g_bump sin bell already fades to 0 at touchdown.
  int rl_id = mj_name2id(model, mjOBJ_NUMERIC, "trot_release");
  double rel = (rl_id >= 0) ? model->numeric_data[model->numeric_adr[rl_id]] : 0.0;
  rel = clip(rel, 0.0, 0.5);
  // R5 STEP WIDTH (trot_step_width, default 0 = off = byte-identical): per-leg
  // lateral placement DELTA from the home stance width -- left +w/2 (outward),
  // right -w/2 (outward) -- so the lateral axis has a limit-cycle reference and
  // the feet cannot scissor under a lateral catch (step_y alone is a SHARED
  // catch term, same sign both legs; the widening is what needs the per-leg
  // sign). Kept small: widening trades lateral catch reach for anti-scissor.
  int swd_id = mj_name2id(model, mjOBJ_NUMERIC, "trot_step_width");
  double w2 = 0.5 * ((swd_id >= 0)
      ? model->numeric_data[model->numeric_adr[swd_id]] : 0.0);
  // ★ R7: per-leg, because the frozen placement latches independently for L and
  // R (they are half a cycle apart, so they freeze at different instants). With
  // trot_freeze_place off, step_x_l == step_x_r == step_x and this is the old
  // single dHipP exactly.
  double dHipP_l = clip(-step_x_l / 0.80, -0.45, 0.45);  // foot FWD -> less hip_pitch
  double dHipP_r = clip(-step_x_r / 0.80, -0.45, 0.45);

  // ---- ★ R7 GROUND-SEEK (trot_ground_seek [rad/s], default 0 = OFF) -------
  // The reference walker's sin_adapt() keeps driving the swing foot DOWN while
  // tau > 1 (z_offset = -(tau-1)); SwingBell returns to exactly 0 at s=1 and
  // holds, so a foot over ground even a centimetre lower than expected simply
  // hangs and the planner has to catch the stall -- one of the step-boundary
  // hitches. gait_overrun_ is the time a leg has been airborne past its nominal
  // swing end WITHOUT an accepted touchdown (latched in TransitionLocked, so it
  // is 0 unless contact tracking is on). Extends the knee and ankle toward the
  // floor, saturating so a genuinely missing floor cannot command a lunge.
  const double kSeekRate = GetNumberOrDefault(0.0, model, "trot_ground_seek");
  const double seek_l = GroundSeek(residual_.gait_overrun_[0], kSeekRate, 0.30);
  const double seek_r = GroundSeek(residual_.gait_overrun_[1], kSeekRate, 0.30);

  // home stand pose = the "stumble_trot" keyframe (fallback home).
  int pk = mj_name2id(model, mjOBJ_KEY, kfname.c_str());
  if (pk < 0) pk = 0;
  const mjtNum *q0 = model->key_qpos + pk * model->nq;
  constexpr double kSwingHip = 0.35, kSwingKnee = 0.70, kSwingAnk = 0.12;

  // ONE foot swings at a time (antiphase/duty): lift (clearance bell, peaks
  // mid-swing) + place (capture ramp -> foot at xi by touchdown). Blend in over
  // the first 15% so liftoff is smooth; held to s=1 (or released over the last
  // trot_release fraction, R4) so the foot lands placed, then the next phase
  // makes it stance and the planner takes the planted leg. ysign carries the
  // per-leg step-width direction (R5): +1 left (outward=+y), -1 right.
  // ctrl[i] == joint target for qpos[7+i]. L: hipP1 hipR2 knee3 ankP4; R:7 8 9 10
  auto do_leg = [&](double ph, int iHipP, int iHipR, int iKnee, int iAnkP,
                    double ysign, double dHipP, double sxy_y, double seek) {
    if (ph < kDuty) return;                        // stance -> planner owns
    double s = (ph - kDuty) / (1.0 - kDuty);       // swing progress 0..1
    double cl = SwingBell(s);                           // clearance bell 0..1..0 (Bezier, soft touchdown)
    double pl = s * s * (3.0 - 2.0 * s);                // placement smoothstep
    double w = mju_min(s / 0.15, 1.0) * g_amp;          // blend-in weight
    if (rel > 1e-6) {                              // R4: hand back before landing
      double r = mju_min((1.0 - s) / rel, 1.0);    // 1 until s=1-rel, 0 at s=1
      w *= r * r * (3.0 - 2.0 * r);                // smooth release
    }
    // ---- ★ SEPARATION REGULATOR (stumble_sep_hold [m], default 0 = OFF) ----
    // MEASURED 2026-09-09 across three lateral drift runs: the sidestep does not
    // widen the stance, it SCISSORS it shut --
    //     0.382 -> 0.216 m,   0.407 -> 0.163 m,   0.330 -> 0.106 m
    // i.e. the ankles end up 10-22 cm apart having started near 38, and in all
    // three runs every single step was the SAME foot.
    //
    // CAUSE. `step_y` above is ONE lateral capture term applied with the SAME
    // SIGN to both legs. On a leftward fall the weight shifts onto the LEFT
    // foot, so the left foot is the one that cannot lift and the RIGHT foot is
    // the one that actually swings -- and it is then driven leftward, into the
    // stance foot. The cost side picks the correct foot (stepL = ey > 0) but
    // physics picks the other one, and nothing reconciles them. The
    // trot_step_width comment already names this failure ("the feet cannot
    // scissor under a lateral catch ... the widening is what needs the per-leg
    // sign") but that knob is only a FIXED outward bias, and it is 0.
    //
    // FIX. Regulate the achieved separation instead of biasing it: drive the
    // SWING foot so ankle-to-ankle returns to stumble_sep_hold. The swing foot
    // takes up the WHOLE error because it is the only foot free to move. This
    // runs every tick, so it opposes the scissor DURING the swing, not merely
    // between steps. ysign is +1 left / -1 right with +1 meaning outward, so
    // ysign * (target - measured) pushes both legs outward when the feet are
    // too close and inward when too far.
    // 0 = OFF = byte-identical to the shared-sign behaviour above.
    double dSep = 0.0;
    {
      const double kSepHold = GetNumberOrDefault(0.0, model, "stumble_sep_hold");
      const double kSepGain = GetNumberOrDefault(1.0, model, "stumble_sep_gain");
      if (kSepHold > 1e-6 && residual_.stance_sep_y_ > 0.0) {
        dSep = clip(ysign * kSepGain * (kSepHold - residual_.stance_sep_y_),
                    -0.12, 0.12);
      }
    }
    double dHipR = clip((sxy_y + ysign * w2 + dSep) / 0.79, -0.25, 0.25);  // R5 widen
    double tHipP = q0[7 + iHipP] - kSwingHip * sh * cl * g_amp + dHipP * pl * g_amp;
    double tHipR = q0[7 + iHipR] + dHipR * pl * g_amp;
    // ★ R7 ground-seek: EXTEND the knee (reduce the fold) and plantarflex the
    // ankle by `seek` once the leg is overrunning its swing window, so the foot
    // keeps reaching for the floor instead of holding a fixed airborne pose.
    // seek == 0 (the default / OFF path) leaves both terms exactly as before.
    double tKnee = q0[7 + iKnee] + kSwingKnee * sh * cl * g_amp - seek;
    double tAnkP = q0[7 + iAnkP] - kSwingAnk * sh * cl * g_amp - 0.4 * seek;
    ctrl[iHipP] += w * (tHipP - ctrl[iHipP]);
    ctrl[iHipR] += w * (tHipR - ctrl[iHipR]);
    ctrl[iKnee] += w * (tKnee - ctrl[iKnee]);
    ctrl[iAnkP] += w * (tAnkP - ctrl[iAnkP]);
  };
  // LEFT swing window ph_l in [duty,1]; RIGHT ph_r. Each leg carries its own
  // frozen placement (R7) and its own ground-seek overrun.
  do_leg(ph_l, 1, 2, 3, 4, +1.0, dHipP_l, step_y_l, seek_l);
  do_leg(ph_r, 7, 8, 9, 10, -1.0, dHipP_r, step_y_r, seek_r);

  // ---- STANCE-SIDE ROCK (trot_rock_ff, default 0 = OFF = byte-identical) ----
  // The missing half of a step: unload the foot BEFORE asking it to swing. This
  // is ADDITIVE (not a blend-to-target like do_leg) on purpose -- the sampler
  // still OWNS stance balance; we only bias the hip roll it was already being
  // asked (weight 30, target shifted identically in Residual) to find. The
  // swing leg's own hip_roll is set by do_leg above and is airborne anyway, so
  // only the STANCE side gets the delta: g_bump_l>0 means LEFT is swinging, so
  // the delta lands on the RIGHT (stance) hip_roll, and vice versa. Both bumps
  // are 0 in double support => no delta => quiet stand untouched.
  {
    const double rock_m = GetNumberOrDefault(0.0, model, "trot_rock_ff");
    if (rock_m > 1e-6) {
      double droll_l = 0.0, droll_r = 0.0;
      TrotRockHipRoll(rock_m,
                      GetNumberOrDefault(0.0, model, "trot_rock_lead"), g_amp,
                      ph_l, ph_r, kDuty, &droll_l, &droll_r);
      ctrl[2] += droll_l;    // L hip_roll (nu idx 2) -- stance while RIGHT swings
      ctrl[8] += droll_r;    // R hip_roll (nu idx 8) -- stance while LEFT swings
    }
  }
}

std::map<std::string, double> loco::PlannerNumericOverrides(int strategy) const {
  const auto names = GetStrategyNames();
  if (strategy < 0 || strategy >= static_cast<int>(names.size())) return {};
  const std::string &name = names[strategy];
  // "Stumble" is the only stepping strategy: its gait clock oscillates the legs,
  // which the stand-tuned sampling_spline_points=3 cannot represent (the swing
  // foot never lifts). spline=5 is the twin-validated sweet spot -- spline=8
  // over-actuates and destabilises even a plain stand (2026-06-18). exploration
  // 0.05 == the XML default, pinned here so stumble is unaffected if that
  // default ever changes. Adding a future strategy that needs a different planner
  // bandwidth is a one-line edit HERE (fork side) -- the deploy node stays
  // strategy-agnostic and never changes.
  // ===== STEPPING STRATEGIES (trot 23, drive 24): the DEPLOY-REAL budget ==========
  // spline 5: the stand-tuned 3 knots cannot represent the leg oscillation.
  //
  // trajectories 17 (was 36): 36 was a LOCKSTEP-TWIN conclusion (16 = 1/5 hold, 36 = 5/5)
  // and the lockstep twin is STRUCTURALLY BLIND to plan rate -- it waits for physics, so
  // rollouts are free. On the REAL robot they are not: CEM schedules trajectories + 1 jobs
  // (the nominal rides along) and blocks on ALL of them, so a plan iteration costs
  // ceil((N+1)/threads) thread-WAVES. 36 traj on the old 12-thread pool = 4 waves =
  // 27-30 plans/s measured -- BELOW the 50-100 Hz band that every real legged sampling-MPC
  // deployment needs (CMU Go1 whole-body MPPI, the reference Tassa points hardware users
  // to: 30 samples @ 100 Hz on CPU, and explicitly "limited computing is much better spent
  // on achieving a ~100 Hz policy than additional sample evaluations during agile
  // locomotion"; their cost plateaus at ~40 samples. Unitree's own H1-2 RL deploys decide
  // at 50 Hz). Real 2026-07-11 at 18 traj / 18 threads measured 45-52 plans/s -- in band.
  // 17 (not 18) because the NOMINAL rollout is scheduled alongside: 17 + 1 = 18 jobs = ONE
  // wave on the auto-sized 18-thread pool (hw 24 - reserve 6), where 18 + 1 = 19 spilled a
  // second wave that the whole pool then waited on for a single straggler. The node now
  // prints a PLAN-RATE WARNING whenever (traj + 1) > threads, so this can't go silent again.
  // Rate beats samples here because our swing is FORCED open-loop by loco::ModifyControl
  // (rate-independent) -- the sampler only has to BALANCE, and the stance-leg weight shift
  // is exactly what starves first.
  //
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
  if (name == "h12_simple_trot" || name == "h12_simple_drive") {
    return {{"sampling_spline_points", 5.0},
            {"sampling_exploration", 0.05},
            {"sampling_trajectories", 17.0}};
  }
  if (name == "h12_simple_stumble") {
    // foot-lift fix (2026-06-24): spline 5 (the stand-tuned 3 cannot represent the
    // swing) + a mild rollout bump. The A (cost-release) + B (raised swing fold)
    // changes are the deploy-ready lift (own-sim ~2.3cm, held). Tier-C aggressive
    // exploration (std_initial 0.25 / explore_fraction 0.4) lifts HIGHER (~9-12cm)
    // but DESTABILISES this marginal biped (yaw runaway -> topple ~8s) -- the
    // documented sampling-only-humanoid frontier -- so it is left OFF here (the
    // std_initial/explore_fraction/knot_var_growth numerics + planner code remain
    // for twin experiments; default = legacy/off). 48 rollouts is also ~5x compute
    // = not real-time on CPU. KEEP it conservative for the real robot.
    return {{"sampling_spline_points", 5.0},
            {"sampling_exploration", 0.05},
            {"sampling_trajectories", 16.0}};
  }
  return {};
}

void loco::ComputeMetrics(const mjModel *model, const mjData *data,
                          std::map<std::string, double> *metrics,
                          std::string *phase_name) const {
  if (metrics) metrics->clear();
  if (phase_name) phase_name->clear();
  if (!metrics) return;

  // ----- Phase identity -------------------------------------------------- //
  const auto &kf = residual_.residual_keyframe_;
  if (phase_name) *phase_name = kf.name;

  // Phase ramp progress (matches Residual()'s smoothstep)
  double time_in_phase =
      mju_max(0.0, data->time - residual_.keyframe_start_time_);
  double alpha_lin =
      mju_min(time_in_phase / ResidualFn::kPhaseRampSeconds, 1.0);
  double alpha_smooth = alpha_lin * alpha_lin * (3.0 - 2.0 * alpha_lin);
  (*metrics)["phase_time"] = time_in_phase;
  (*metrics)["phase_alpha_linear"] = alpha_lin;
  (*metrics)["phase_alpha"] = alpha_smooth;

  // ----- Sensor reads (bail out if any missing) -------------------------- //
  double *foot_left = SensorByName(model, data, "foot_left_pos");
  double *foot_right = SensorByName(model, data, "foot_right_pos");
  double *subcom = SensorByName(model, data, "torso_subcom");
  double *com_vel = SensorByName(model, data, "torso_subtreelinvel");
  if (!foot_left || !foot_right || !subcom) return;

  // Front-most foot edge: +0.10 m toe offset because foot_*_pos is at the
  // ankle site, not the toe.
  double front_edge_x = mju_max(foot_left[0], foot_right[0]) + 0.10;

  // ----- CoM excursion from midfoot -------------------------------------- //
  double midfoot_x = 0.5 * (foot_left[0] + foot_right[0]);
  double midfoot_y = 0.5 * (foot_left[1] + foot_right[1]);
  double com_dx = subcom[0] - midfoot_x;
  double com_dy = subcom[1] - midfoot_y;
  (*metrics)["com_excursion_sagittal"] = com_dx;
  (*metrics)["com_excursion_lateral"] = com_dy;
  (*metrics)["com_excursion"] = mju_sqrt(com_dx * com_dx + com_dy * com_dy);
  (*metrics)["com_x"] = subcom[0];
  (*metrics)["com_y"] = subcom[1];
  (*metrics)["com_z"] = subcom[2];

  // ----- CoP / ZMP from floor contacts ----------------------------------- //
  // Floor geom is conventionally named "floor" in MJCF; lookup once.
  // Approximation: world vertical component of each contact = f_normal * n_z,
  // where n_z = c.frame[2] (z-component of contact normal).
  int floor_geom_id = mj_name2id(model, mjOBJ_GEOM, "floor");
  double cop_x_num = 0.0, cop_y_num = 0.0, cop_denom = 0.0;
  double foot_force_total = 0.0;
  for (int i = 0; i < data->ncon; ++i) {
    const mjContact &c = data->contact[i];
    bool involves_floor = (floor_geom_id >= 0) &&
                          (c.geom[0] == floor_geom_id ||
                           c.geom[1] == floor_geom_id);
    if (!involves_floor) continue;
    double f6[6];
    mj_contactForce(model, data, i, f6);
    double n_z = c.frame[2];
    double fz = f6[0] * n_z;
    if (fz > 0.0) {
      cop_x_num += c.pos[0] * fz;
      cop_y_num += c.pos[1] * fz;
      cop_denom += fz;
      foot_force_total += fz;
    }
  }
  if (cop_denom > 0.0) {
    (*metrics)["cop_x"] = cop_x_num / cop_denom;
    (*metrics)["cop_y"] = cop_y_num / cop_denom;
  } else {
    (*metrics)["cop_x"] = midfoot_x;
    (*metrics)["cop_y"] = midfoot_y;
  }
  (*metrics)["foot_force_total"] = foot_force_total;

  // ----- torque saturation (max fraction across actuators) --------------- //
  // |ctrl[i] - midpoint| / half-range, max over actuators. Uses MuJoCo's
  // actuator_forcerange (set by ctrlrange / forcerange in MJCF / URDF).
  double max_torque_frac = 0.0;
  for (int i = 0; i < model->nu; ++i) {
    double lo = model->actuator_forcerange[2 * i];
    double hi = model->actuator_forcerange[2 * i + 1];
    if (hi > lo) {
      double range = 0.5 * (hi - lo);
      double mid = 0.5 * (hi + lo);
      double frac = mju_abs(data->ctrl[i] - mid) / range;
      if (frac > max_torque_frac) max_torque_frac = frac;
    }
  }
  (*metrics)["torque_saturation_max"] = max_torque_frac;

  // ----- max joint velocity (raw -- % vs URDF limits in analyzer) -------- //
  double max_vel = 0.0;
  for (int i = 6; i < model->nv; ++i) {  // skip free-joint base
    double v = mju_abs(data->qvel[i]);
    if (v > max_vel) max_vel = v;
  }
  (*metrics)["joint_velocity_max"] = max_vel;

  // ----- Instantaneous Capture Point ------------------------------------ //
  // xi = r_CoM + v_CoM * sqrt(z_CoM / g). Cheap predictive stability margin.
  if (com_vel) {
    double z = mju_max(subcom[2], 0.1);
    double omega = mju_sqrt(9.81 / z);
    (*metrics)["icp_x"] = subcom[0] + com_vel[0] / omega;
    (*metrics)["icp_y"] = subcom[1] + com_vel[1] / omega;
  }

  // ----- Support-polygon excursion --------------------------------------- //
  // Signed forward distance from the CoM ground projection (and the measured
  // CoP) to the front foot edge. Positive = ahead of the foot polygon.
  (*metrics)["com_beyond_foot_edge"] = subcom[0] - front_edge_x;
  (*metrics)["cop_beyond_foot_edge"] = (*metrics)["cop_x"] - front_edge_x;

  // ----- Foot positions exposed for the top-down panel ------------------- //
  (*metrics)["foot_left_x"] = foot_left[0];
  (*metrics)["foot_left_y"] = foot_left[1];
  (*metrics)["foot_left_z"] = foot_left[2];
  (*metrics)["foot_right_x"] = foot_right[0];
  (*metrics)["foot_right_y"] = foot_right[1];
  (*metrics)["foot_right_z"] = foot_right[2];

  // ----- Drive / gait state (strat 24 teleop + R7) ----------------------- //
  (*metrics)["drive_gait_amp"] = residual_.drive_gait_amp_;
  (*metrics)["drive_yaw_des"] = residual_.drive_yaw_des_;
  (*metrics)["cmd_active"] = residual_.cmd_active_ ? 1.0 : 0.0;
  (*metrics)["cmd_vdes_world_x"] = residual_.cmd_vdes_world_[0];
  (*metrics)["cmd_vdes_world_y"] = residual_.cmd_vdes_world_[1];
  (*metrics)["gait_phase_offset"] = residual_.gait_phase_offset_;
}

}  // namespace mjpc