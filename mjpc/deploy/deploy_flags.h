// Declarations for the shared deploy flag manifest (deploy_flags.inc).
// deploy_common.cc includes this so FillCommonConfig() can read the flags each
// thin main DEFINES by including the .inc -- every binary links exactly one
// main, so each flag has exactly one definition per binary.

#ifndef MJPC_DEPLOY_DEPLOY_FLAGS_H_
#define MJPC_DEPLOY_DEPLOY_FLAGS_H_

#include <string>

#include <absl/flags/declare.h>

ABSL_DECLARE_FLAG(double, gravity_ff);
ABSL_DECLARE_FLAG(double, twin_dt);
ABSL_DECLARE_FLAG(std::string, sportstate_topic);
ABSL_DECLARE_FLAG(double, imu_pitch_offset_deg);
ABSL_DECLARE_FLAG(double, bad_orient_rad);
ABSL_DECLARE_FLAG(double, imu_roll_offset_deg);
ABSL_DECLARE_FLAG(double, ankle_roll_offset_l_deg);
ABSL_DECLARE_FLAG(double, ankle_roll_offset_r_deg);
ABSL_DECLARE_FLAG(double, ankle_pitch_offset_l_deg);
ABSL_DECLARE_FLAG(double, ankle_pitch_offset_r_deg);
ABSL_DECLARE_FLAG(std::string, network_interface);
ABSL_DECLARE_FLAG(int, domain_id);
ABSL_DECLARE_FLAG(int, grpc_port);
ABSL_DECLARE_FLAG(int, plan_trajectories);
ABSL_DECLARE_FLAG(int, plan_threads);
ABSL_DECLARE_FLAG(bool, straighten_start);
ABSL_DECLARE_FLAG(bool, cost);
ABSL_DECLARE_FLAG(int, frc_parity);
ABSL_DECLARE_FLAG(double, stale_sec);
ABSL_DECLARE_FLAG(double, latency_rtf);
ABSL_DECLARE_FLAG(bool, arm_aware);
ABSL_DECLARE_FLAG(bool, align_start);
ABSL_DECLARE_FLAG(double, align_sec);
ABSL_DECLARE_FLAG(double, align_tol);
ABSL_DECLARE_FLAG(double, align_ki);
ABSL_DECLARE_FLAG(double, align_i_max);
ABSL_DECLARE_FLAG(bool, align_wait);
ABSL_DECLARE_FLAG(double, align_timeout);
ABSL_DECLARE_FLAG(std::string, plan_topic);
ABSL_DECLARE_FLAG(double, plan_hz);

#endif  // MJPC_DEPLOY_DEPLOY_FLAGS_H_
