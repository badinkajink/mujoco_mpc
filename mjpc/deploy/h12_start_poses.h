// H1-2 deploy align start poses -- the ONE canonical copy (stage 2 of the
// 2026-07-18 reorg; previously duplicated byte-for-byte in the lower and split
// mains). R^12 leg stances in DDS motor order (lowcmd rows 0..11).
//
// LEG-OWNING MAINS ONLY (lower, split). Never wire these through a shared
// skeleton: on the upper main motor_offset=12 would re-interpret these leg
// angles as shoulder/elbow targets. The full-body main aligns to the model's
// 'stand' keyframe instead (align_pose_set=false fallback).

#ifndef MJPC_DEPLOY_H12_START_POSES_H_
#define MJPC_DEPLOY_H12_START_POSES_H_

namespace h12deploy {

// START POSE (--align_start): stance the node drags the legs into BEFORE
// handover to MJPC. Tuning knob -- edit freely, but keep it in sync with the
// stand keyframe legs strategy-6's Posture pulls toward, or the handover
// jumps. (history: see mjpc/deploy/HISTORY.md)
inline constexpr double kLowerStartPose[12] = {
     0.00,   // [ 0] left_hip_yaw_joint        0 deg
    -0.15,   // [ 1] left_hip_pitch_joint    -8.6 deg -- == 'stand_up' keyframe
     0.12,   // [ 2] left_hip_roll_joint     +6.9 deg -- opens the stance
     0.35,   // [ 3] left_knee_joint        +20.1 deg -- bent (== keyframe, was 0.37)
    -0.28,   // [ 4] left_ankle_pitch_joint -16.0 deg -- sole flat (== keyframe, was -0.21)
    -0.12,   // [ 5] left_ankle_roll_joint   -6.9 deg -- sole flat laterally
     0.00,   // [ 6] right_hip_yaw_joint      0 deg
    -0.15,   // [ 7] right_hip_pitch_joint   -8.6 deg
    -0.12,   // [ 8] right_hip_roll_joint    -6.9 deg  (mirror of left)
     0.35,   // [ 9] right_knee_joint       +20.1 deg
    -0.28,   // [10] right_ankle_pitch_joint-16.0 deg
     0.12,   // [11] right_ankle_roll_joint  +6.9 deg  (mirror of left)
};

// LOCKSTAND (--strategy 26 on the STABILIZE task) align target: LOCKED knees +
// WIDE stance (~0.635 m). Tuning knob -- edit freely, but keep it == the
// 'lockstand' keyframe legs; feet must START wide (a balance hold cannot widen
// planted feet). (history: see mjpc/deploy/HISTORY.md)
inline constexpr double kLockstandStartPose[12] = {
     0.00,   // [ 0] left_hip_yaw_joint       0 deg
    -0.03,   // [ 1] left_hip_pitch_joint    -1.7 deg -- thigh near-vertical (straight leg)
     0.19,   // [ 2] left_hip_roll_joint    +10.9 deg -- WIDE splay
     0.08,   // [ 3] left_knee_joint         +4.6 deg -- LOCKED strut
    -0.14,   // [ 4] left_ankle_pitch_joint  -8.0 deg -- sole flat under straight shin
    -0.19,   // [ 5] left_ankle_roll_joint  -10.9 deg -- sole flat laterally at the wide stance
     0.00,   // [ 6] right_hip_yaw_joint      0 deg
    -0.03,   // [ 7] right_hip_pitch_joint   -1.7 deg
    -0.19,   // [ 8] right_hip_roll_joint   -10.9 deg  (mirror of left)
     0.08,   // [ 9] right_knee_joint        +4.6 deg
    -0.14,   // [10] right_ankle_pitch_joint -8.0 deg
     0.19,   // [11] right_ankle_roll_joint +10.9 deg  (mirror of left)
};

// Strategy 26 selects the lockstand stance; every other strategy keeps the
// tested bent-knee stand bring-up. Feet must START wide (a hold can't slide
// planted feet outward), so the align pose owns the stance width. NOTE the
// slot-26 collision: on the LEAN tasks slot 26 is h12_simple_jump, yet an
// --align_start boot at 26 still aligns to the lockstand stance (documented
// in the split main's --strategy help).
inline const double* AlignPoseForStrategy(int strategy) {
  return strategy == 26 ? kLockstandStartPose : kLowerStartPose;
}

}  // namespace h12deploy

#endif  // MJPC_DEPLOY_H12_START_POSES_H_
