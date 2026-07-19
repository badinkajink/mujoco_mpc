// Deploy robot-state types + quaternion helper (stage 3a of the 2026-07-18
// reorg; formerly internal to deploy_common.cc). The pelvis-from-IMU-site
// reconstruction itself (fill_state) still lives in RunDeployNode -- it is
// loop-state-coupled and moves in stage 3b with NodeRuntime.

#ifndef MJPC_DEPLOY_DEPLOY_STATE_H_
#define MJPC_DEPLOY_DEPLOY_STATE_H_

#include <chrono>
#include <cstdint>
#include <mutex>

#include "mjpc/deploy/deploy_common.h"

namespace h12deploy {

// rotate vec v by quaternion q (wxyz) -- matches mjpc_dds_bridge.py:_quat_rot.
void QuatRot(const double q[4], const double v[3], double out[3]);

// Plain, copyable snapshot of the latest robot state.
struct StateData {
  bool have_ls = false, have_ss = false;
  double q[kMaxNU] = {0}, dq[kMaxNU] = {0};       // the cfg.nu ACTUATED joints (motor_offset + i)
  double qu[15] = {0}, dqu[15] = {0};  // complement joints (X-aware): arm-aware = the 15 upper
                                       // (motor 12..26, legs-only node); leg-aware = the 12 legs
                                       // (motor 0..11, upper-body node)
  double quat[4] = {1, 0, 0, 0}, gyro[3] = {0};  // rt/lowstate IMU (wxyz, body gyro)
  double site_p[3] = {0}, site_v[3] = {0};       // rt/sportmodestate (IMU-site world pose)
  uint8_t mode_machine = 0;
  uint32_t tick = 0;          // rt/lowstate tick = twin sim-step count (twin sim_time = tick * twin_dt)
  // H1 watchdog: wall-clock receive stamps of the two streams (steady_clock).
  std::chrono::steady_clock::time_point ls_stamp{}, ss_stamp{};
};
// Mutex-guarded holder (std::mutex isn't copyable, so it stays out of the snapshot).
struct RobotState {
  std::mutex mu;
  StateData d;
};

}  // namespace h12deploy

#endif  // MJPC_DEPLOY_DEPLOY_STATE_H_
