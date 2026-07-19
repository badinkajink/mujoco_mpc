// H1-2 deploy gain/limit tables -- the ONE canonical copy (stage 2 of the
// 2026-07-18 reorg; previously duplicated per main). DDS motor order:
// rows 0..11 legs, 12 torso, 13..26 arms (kMaxNU = 27).
//
// Slicing convention (structural, replaces the old "must equal rows N..M of the
// full table" comments): the legs-only main passes the row-0 pointers with
// nu=12; the upper main passes (pointer + kUpperOffset) with nu=15 for the
// TAU tables and its OWN kp/kv gate tables below.
//
// Cross-component contracts (do not change one side alone):
//   - kKp27/kKv27 == the h1_2_modified actuator classes == the twin PD ==
//     what PatchActuators writes into the planner+latency models.
//   - kTauEstop27 == the safety layer's estop table (estop torque_ratio x URDF
//     torque limit). Feeds the over-budget telemetry and the --frc_parity
//     forcerange patch; emitted torque is NOT clamped -- the estop is the backstop.
//   - kTauLimit27 == the operational URDF actuatorfrcrange (B0 report basis).
//   - kFrcLimit27 is patched ARMS-ONLY (frc_limit_begin = kArmsBegin): leg/torso
//     forceranges stay at the model default (tightening them regressed the hold).

#ifndef MJPC_DEPLOY_H12_GAIN_TABLES_H_
#define MJPC_DEPLOY_H12_GAIN_TABLES_H_

namespace h12deploy {

inline constexpr int kNU27 = 27;      // whole-body actuated joint count
inline constexpr int kLegsCount = 12; // rows 0..11
inline constexpr int kUpperOffset = 12;  // torso+arms start row
inline constexpr int kUpperCount = 15;   // torso + two 7-DoF arms
inline constexpr int kArmsBegin = 13;    // first ARM row (frc_limit patch start)

// Per-joint gains. ARMS kp 30/20/15 (shoulder_p/r, shoulder_yaw+elbow, wrist)
// so the onboard arm PD torque stays under the arm estop bound; legs/torso kp
// left at original (lowering them regressed the hold). Ankles keep kp 80/kv 4
// (softening A/B-tested and REJECTED -- stiffness is load-bearing; see
// mjpc/deploy/HISTORY.md).
inline constexpr double kKp27[kNU27] = {
    150, 200, 200, 200, 80, 80,  150, 200, 200, 200, 80, 80,  200,
    30, 30, 20, 20, 15, 15, 15,   30, 30, 20, 20, 15, 15, 15};
inline constexpr double kKv27[kNU27] = {
    5, 5, 5, 5, 4, 4,  5, 5, 5, 5, 4, 4,  5,
    10, 10, 10, 10, 2, 2, 2,  10, 10, 10, 10, 2, 2, 2};

// SAFETY-LAYER TAU-ESTOP thresholds (== the safety layer's estop table).
inline constexpr double kTauEstop27[kNU27] = {
    60, 130, 200, 300, 54, 36,  60, 130, 200, 300, 54, 36,  40,
    32, 32, 14.4, 14.4, 9.5, 9.5, 9.5,
    32, 32, 14.4, 14.4, 9.5, 9.5, 9.5};

// OPERATIONAL H1-2 joint torque limits (Nm) = Unitree URDF actuatorfrcrange.
inline constexpr double kTauLimit27[kNU27] = {
    200, 200, 200, 300, 60, 40,  200, 200, 200, 300, 60, 40, 200,
    40, 40, 18, 18, 19, 19, 19,  40, 40, 18, 18, 19, 19, 19};

// ARM actuator force limit = operational URDF, patched into the planner+latency
// model for the ARMS ONLY (rows kArmsBegin..26).
inline constexpr double kFrcLimit27[kNU27] = {
    200, 200, 200, 300, 60, 40,  200, 200, 200, 300, 60, 40,  200,
    40, 40, 18, 18, 19, 19, 19,   40, 40, 18, 18, 19, 19, 19};

inline constexpr const char* kJointNames27[kNU27] = {
    "LhipY", "LhipP", "LhipR", "Lknee", "LankP", "LankR",
    "RhipY", "RhipP", "RhipR", "Rknee", "RankP", "RankR", "torso",
    "LshP", "LshR", "LshY", "Lelb", "LwrR", "LwrP", "LwrY",
    "RshP", "RshR", "RshY", "Relb", "RwrR", "RwrP", "RwrY"};

// UPPER-BODY node gate tables (nu=15, rows == kUpperOffset..26 of the wire).
// DELIBERATELY different from kKp27/kKv27's arm rows: ALL arm joints kp 40
// (== the h1_2_modified actuator classes == the generated upper planner model
// == the twin PD), keeping the node byte-identical to the P6.2 precision-gate
// model it was validated against. Do NOT unify with the whole-body tables --
// the divergence is the contract.
inline constexpr double kUpperKpGate15[kUpperCount] = {
    200,  40, 40, 40, 40, 40, 40, 40,  40, 40, 40, 40, 40, 40, 40};
inline constexpr double kUpperKvGate15[kUpperCount] = {
    5,  10, 10, 10, 10, 2, 2, 2,  10, 10, 10, 10, 2, 2, 2};

}  // namespace h12deploy

#endif  // MJPC_DEPLOY_H12_GAIN_TABLES_H_
