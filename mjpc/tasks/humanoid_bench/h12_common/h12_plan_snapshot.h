// H1-2 task-twin shared per-plan snapshot (stage 4a of the 2026-07-18 reorg).
//
// Every field here is ROLLOUT-VISIBLE state: written by the canonical residual
// under the transition lock (TransitionLocked, once per plan, before rollout
// workers fan out) and read by the per-worker residual copies. The task's
// ResidualFn derives from (task-specific subclass of) this struct, and
// ResidualLocked copies it WHOLESALE into each rollout copy -- one struct
// assignment instead of the old 14-arg ctor + post-hoc field list, whose
// forgot-to-copy failure mode produced the 2026-07-12 walk-ceiling bug
// (rollouts silently costing an in-place trot while ModifyControl drove the
// swing forward; history: see mjpc/tasks/humanoid_bench/HISTORY.md).
//
// RULES:
//  - A field belongs here IFF a rollout residual copy must read it. Canonical-
//    only bookkeeping (governor filters, FSM latches, log-once flags) stays a
//    plain ResidualFn member and is deliberately NOT propagated.
//  - Adding a field here propagates it automatically -- no ResidualLocked edit.
//  - Cost/forcer coherence (I10): several of these are the shared state that
//    keeps Residual (cost), ModifyControl (open-loop swing forcer) and
//    TransitionLocked (latches/governors) in agreement. They must keep
//    consuming THIS state, never re-derived copies.

#ifndef MJPC_TASKS_HUMANOID_BENCH_H12_COMMON_H12_PLAN_SNAPSHOT_H_
#define MJPC_TASKS_HUMANOID_BENCH_H12_COMMON_H12_PLAN_SNAPSHOT_H_

#include <mujoco/mujoco.h>

#include "mjpc/tasks/humanoid/interact/contact_keyframe.h"

namespace mjpc {
namespace h12 {

struct PlanSnapshotBase {
  // ----- Active strategy keyframe + phase-transition ramp state -----
  // keyframe_start_time_: time the current keyframe became active (set in
  // TransitionLocked); Residual() uses data->time - keyframe_start_time_ for
  // ramp progress. prev_phase_*: the scales in effect just before the last
  // transition, so Residual() lerps smoothly into the new phase's scales
  // (the WBC-style handoff that avoids lurching when a contact cost flips on).
  mjpc::humanoid::ContactKeyframe residual_keyframe_;
  mjtNum keyframe_start_time_ = 0.0;
  mjtNum prev_phase_reach_scale_ = 0.0;
  mjtNum prev_phase_brace_pos_scale_ = 0.0;
  // Posture scale starts at 1.0 (no boost) and ramps to 3.0 during stand_up.
  mjtNum prev_phase_posture_scale_ = 1.0;
  // Previous phase's brace_force_target, smoothstepped across phase
  // boundaries so MPC never sees a step change in the force demand.
  mjtNum prev_phase_brace_force_target_ = 0.0;
  // Previous phase's posture keyframe id (model <key> index), captured at
  // every transition so Residual() can ramp the TARGET pose from it over
  // kPhaseRampSeconds. 0 = home on cold start.
  int prev_posture_key_id_ = 0;
  // Number of phases (keyframes) in the active strategy; the target-pose ramp
  // is gated on num_phases_ > 1 so single-phase strategies never enter it.
  int num_phases_ = 1;

  // Per-contact-pair "is new this phase" flags: a pair that just appeared gets
  // its residual smoothstepped in over kPhaseRampSeconds (same window as the
  // weights) so the planner never slams toward a brand-new contact target.
  bool contact_pair_is_new_[5] = {false, false, false, false, false};

  // ----- STRAIGHTEN (strat 25) live-seed funnel -----
  // Full qpos + pelvis tilt captured at straighten entry (the posture/upright
  // ramps start FROM these); straighten_seeded_ = captured (else static target).
  mjtNum straighten_start_qpos_[64] = {0};
  double straighten_start_tilt_ = 0.0;
  bool straighten_seeded_ = false;

  // ----- LIVE cmd_vel teleop: the governed command -----
  // Written ONLY by the TransitionLocked governor; read by Residual() and by
  // the task's ModifyControl. cmd_active_=false => both readers take the
  // legacy static-numeric path (byte-identical to the validated configs).
  // MUST be in the snapshot: rollouts that miss it cost an in-place trot
  // while ModifyControl drives the swing forward (the walk-ceiling bug).
  bool   cmd_active_ = false;
  double cmd_vdes_world_[2] = {0.0, 0.0};  // governed v_des, WORLD frame

  // ----- WSS drive FSM outputs (strat 24) -----
  // drive_gait_amp_ (0..1): gait-enable multiplier -- the COST gates g_amp
  // with exactly the value ModifyControl uses. drive_yaw_des_: integrated
  // desired WORLD heading [rad] (Body Yaw reference).
  double drive_gait_amp_ = 0.0;
  double drive_yaw_des_ = 0.0;
};

}  // namespace h12
}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_BENCH_H12_COMMON_H12_PLAN_SNAPSHOT_H_
