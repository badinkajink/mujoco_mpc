// Shared core of the H1-2 MJPC deploy nodes -- see deploy_common.h for the
// architecture note and the flag-diet rationale. This file is the line-for-line
// merge of the formerly-duplicated h12_control_node.cc / h12_lower_body_controller.cc
// (HAMS_integration 2026-07-02), parameterized by NodeConfig:
//   - kNU / gain tables       -> cfg.nu + cfg.kp/kv/tau_estop/tau_limit/frc_limit
//   - arm-aware machinery     -> gated on cfg.upper_count > 0 (legs-only node)
//   - status-line variant     -> cfg.telemetry
// plus the four defensive fixes ported from the HAMS rclcpp rewrite (audit
// H1/H2/M4/M5, see mjpc_deploy_lowerbody_controller.cpp):
//   H1  input-freshness watchdog: state older than kStaleSec -> damping safe-hold
//   H2  torque clamp on the FULL budget tau_ff + KP*e + KV*dq (was KP*e only)
//   M4  tau%%estop telemetry graded against TAU_ESTOP (was mislabeled TAU_LIMIT)
//   M5  mju_error -> emit safe-hold BEFORE terminating (was a bare exit)
//
// WHY THIS EXISTS / ARCHITECTURE / STATE reconstruction: see the header block
// of h12_control_node.cc (kept there -- it is still the entry point doc).

#include "mjpc/deploy/deploy_common.h"

#include <algorithm>
#include <limits>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// POSIX networking -- auto-pin the wired robot-subnet NIC (see AutoDetectRobotInterface).
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <mujoco/mujoco.h>

#include "mjpc/agent.h"
#include "mjpc/planners/cross_entropy/planner.h"
#include "mjpc/task.h"
#include "mjpc/tasks/tasks.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

// Unitree SDK2 (C++): DDS channel API + unitree_hg / unitree_go IDL.
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>
// std_msgs::msg::dds_::String_ -- byte-identical on the wire to ROS 2's
// std_msgs/msg/String (same DDS typename "std_msgs.msg.dds_.String_"), so the
// debug plan topic is a real ROS topic without a bridge. See cfg.plan_pub_topic.
#include <unitree/idl/ros2/String_.hpp>

#ifdef H12_NODE_GRPC
// Optional in-process MJPC gRPC server so the existing monitor can attach
// (display + live Strategy switch). Enabled by the CMake when the grpc service
// is built. Reuses the exact helpers agent_server uses.
#include <absl/strings/str_cat.h>
#include <grpc/grpc_security_constants.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include "mjpc/grpc/agent.grpc.pb.h"
#include "mjpc/grpc/agent.pb.h"
#include "mjpc/grpc/grpc_agent_util.h"
#endif

namespace h12deploy {

using unitree::robot::ChannelFactory;
using unitree::robot::ChannelPublisher;
using unitree::robot::ChannelPublisherPtr;
using unitree::robot::ChannelSubscriber;
using unitree::robot::ChannelSubscriberPtr;
using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;
using SportState = unitree_go::msg::dds_::SportModeState_;

namespace {

// Auto-detect the interface holding a 192.168.123.x address (the wired H1-2
// robot subnet), so an EMPTY --network_interface binds the robot link instead
// of CycloneDDS autodetermine grabbing WiFi/Tailscale -- the trap that silently
// makes the node hear the same-host twin but never the real robot. Mirrors
// dds_tools/dds_topic_check.py. Returns "" when no robot-subnet NIC is present
// (-> caller keeps autodetermine/loopback, the right default for the twin).
std::string AutoDetectRobotInterface() {
  const char kRobotPrefix[] = "192.168.123.";
  std::string result;
  struct ifaddrs* ifaddr = nullptr;
  if (getifaddrs(&ifaddr) == -1) return result;
  for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) continue;
    if (ifa->ifa_addr->sa_family != AF_INET) continue;
    char host[INET_ADDRSTRLEN] = {0};
    const void* sin_addr =
        &reinterpret_cast<const struct sockaddr_in*>(ifa->ifa_addr)->sin_addr;
    if (inet_ntop(AF_INET, sin_addr, host, sizeof(host)) == nullptr) continue;
    if (std::strncmp(host, kRobotPrefix, sizeof(kRobotPrefix) - 1) == 0) {
      result = ifa->ifa_name;
      break;
    }
  }
  freeifaddrs(ifaddr);
  return result;
}

// rotate vec v by quaternion q (wxyz) -- matches mjpc_dds_bridge.py:_quat_rot.
void QuatRot(const double q[4], const double v[3], double out[3]) {
  double w = q[0], x = q[1], y = q[2], z = q[3];
  double tx = 2 * (y * v[2] - z * v[1]);
  double ty = 2 * (z * v[0] - x * v[2]);
  double tz = 2 * (x * v[1] - y * v[0]);
  out[0] = v[0] + w * tx + (y * tz - z * ty);
  out[1] = v[1] + w * ty + (z * tx - x * tz);
  out[2] = v[2] + w * tz + (x * ty - y * tx);
}

// Unitree CRC (matches example/h1/low_level + unitree_sdk2py.utils.crc).
uint32_t Crc32Core(uint32_t* ptr, uint32_t len) {
  uint32_t CRC32 = 0xFFFFFFFF;
  const uint32_t dwPolynomial = 0x04c11db7;
  for (uint32_t i = 0; i < len; i++) {
    uint32_t xbit = 1u << 31;
    uint32_t data = ptr[i];
    for (uint32_t bits = 0; bits < 32; bits++) {
      if (CRC32 & 0x80000000) {
        CRC32 <<= 1;
        CRC32 ^= dwPolynomial;
      } else {
        CRC32 <<= 1;
      }
      if (data & xbit) CRC32 ^= dwPolynomial;
      xbit >>= 1;
    }
  }
  return CRC32;
}

// Patch a freshly-loaded model's <position> actuators to the node's authoritative
// gains (+ estop-bound forceranges where the config asks -- full-body node: ARMS
// ONLY, idx >= 13, so the planner can't plan a motor-peak (120 Nm) arm torque.
// LEG/TORSO forceranges are LEFT at the model default: tightening them to the
// estop bound clamped the planner's hip/ankle balance authority and regressed
// the hold). <position>: gainprm[0]=kp, biasprm[1]=-kp, biasprm[2]=-kv. Call on
// the loaded model BEFORE Agent::Initialize (it is const after GetModel()).
//
// ACTUATOR-AUTHORITY PARITY (frc_parity, 2026-07-11): see the NodeConfig::frc_parity
// comment for the full diagnosis. When ON, EVERY actuator's forcerange is tightened to
// the torque the node can actually emit -- kClampRatio * tau_estop, the exact H2 clamp
// budget -- so the sampler stops planning single-support balance on phantom ankle/torso
// authority. It only ever TIGHTENS (never loosens) an existing limit, so the arm patch
// above still binds where it is stricter. This is a MODEL-PARITY fix, not a control law:
// the planner keeps sampling, it just samples inside the envelope it will really get.
void PatchActuators(mjModel* m, const NodeConfig& cfg) {
  // Task-side default lives in the model as the `deploy_frc_parity` numeric (set by
  // Task::PlannerNumericOverrides, applied by the caller BEFORE this runs). CLI wins.
  bool parity = false;
  if (cfg.frc_parity >= 0) {
    parity = cfg.frc_parity > 0;
  } else {
    int pid = mj_name2id(m, mjOBJ_NUMERIC, "deploy_frc_parity");
    parity = (pid >= 0) && (m->numeric_data[m->numeric_adr[pid]] > 0.5);
  }

  for (int i = 0; i < cfg.nu && i < m->nu; i++) {
    m->actuator_gainprm[i * mjNGAIN + 0] = cfg.kp[i];
    m->actuator_biasprm[i * mjNBIAS + 1] = -cfg.kp[i];
    m->actuator_biasprm[i * mjNBIAS + 2] = -cfg.kv[i];
    if (cfg.frc_limit && i >= cfg.frc_limit_begin) {
      m->actuator_forcelimited[i] = 1;
      m->actuator_forcerange[i * 2 + 0] = -cfg.frc_limit[i];
      m->actuator_forcerange[i * 2 + 1] = cfg.frc_limit[i];
    }
  }
  // PatchActuators runs on BOTH the planner model and the latency-comp rollout model (they
  // must agree, else predict-forward simulates torque the planner/node cannot produce).
  // Only narrate the first pass so the log stays readable.
  static bool narrated = false;
  const bool say = !narrated;
  narrated = true;
  if (!parity) {
    if (say)
      std::fprintf(stderr,
                   "[node] actuator-authority parity OFF: the planner may plan torque the "
                   "H2 clamp will not emit (legacy model; --frc_parity=1 to enable)\n");
    return;
  }
  if (say)
    std::fprintf(stderr,
                 "[node] actuator-authority parity ON (planner + latency models): forcerange "
                 "-> the REAL emitted budget (%.2f x tau_estop = the H2 clamp). Tightened:\n",
                 kClampRatio);
  for (int i = 0; i < cfg.nu && i < m->nu; i++) {
    const double budget = kClampRatio * cfg.tau_estop[i];
    const double had = m->actuator_forcelimited[i]
                           ? m->actuator_forcerange[i * 2 + 1]
                           : std::numeric_limits<double>::infinity();
    if (!(budget < had)) continue;             // never LOOSEN an existing limit
    m->actuator_forcelimited[i] = 1;
    m->actuator_forcerange[i * 2 + 0] = -budget;
    m->actuator_forcerange[i * 2 + 1] = budget;
    if (say)
      std::fprintf(stderr, "[node]   %-7s %7.1f -> %6.1f Nm  (planner was %.2fx over)\n",
                   cfg.joint_names ? cfg.joint_names[i] : "?", had, budget,
                   budget > 0.0 ? had / budget : 0.0);
  }
}

// Plain, copyable snapshot of the latest robot state.
struct StateData {
  bool have_ls = false, have_ss = false;
  double q[kMaxNU] = {0}, dq[kMaxNU] = {0};       // the cfg.nu ACTUATED joints (motor_offset + i)
  double tau[kMaxNU] = {0};                       // tau_est: LOADED gate for the ankle auto-calib
  double qu[15] = {0}, dqu[15] = {0};  // complement joints (X-aware): arm-aware = the 15 upper
                                       // (motor 12..26, legs-only node); leg-aware = the 12 legs
                                       // (motor 0..11, upper-body node)
  double quat[4] = {1, 0, 0, 0}, gyro[3] = {0};  // rt/lowstate IMU (wxyz, body gyro)
  double acc[3] = {0};   // rt/lowstate IMU accelerometer (specific force, sensor frame):
                         // at rest = R^T*(0,0,+9.81) = an ABSOLUTE inclinometer. Consumed
                         // by the autocalib gravity anchor (fused-quat pitch/roll carries
                         // filter lag + support-motion error; time-averaged raw gravity
                         // in a verified-still window does not -- Guedelha 2016 / INS
                         // "levelling update").
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

// ===========================================================================
// ANKLE ZERO AUTO-CALIB (--ankle_autocalib, OFF by default)
//
// The H1-2 stores NO ankle calibration: each ankle's zero = wherever the A/B
// linkage sat at encoder power-on (Unitree's own FAQ: "ensure the position of
// ... ankles are correct before powering on"). A few degrees of zero error
// physically rotates everything above the ankle (the motor PD servos the
// ENCODER to the target, so q_true = tgt - off) -> steady lean, CoP parked at
// the toe, the per-power-on stand lottery.
//
// METHOD (C++ port of ankle_zero_snap.py, validated 2026-07-17: recovers a
// planted +-6 deg to 0.000 deg through physics, incl. the IMU offset in the
// loop): the FLOOR is the reference. Loaded, the sole MUST be flat, so with
// the (corrected) IMU quat + measured leg encoders, secant-solve the ankle
// pitch/roll that would make each sole flat; offset = measured - solved.
//
// WHEN: sampled during the scripted bring-up ramp HOLD -- the robot is still,
// loaded, and the planner has zero authority. Purely OBSERVATIONAL: no
// choreography change; the solve lands before policy handover so the planner
// starts with a corrected belief. Gates (LOADED + SETTLED + STABLE + halves
// agree + |off| cap) mirror the snap tool; ANY failure -> apply NOTHING, keep
// the manual --ankle_*_offset flags, log why. It can never bake a bad number.
// ===========================================================================
namespace anklecalib {

// world pitch/roll of a body from its row-major xmat (== ankle_zero_snap.foot_pr)
inline void FootPR(const double* x, double* pitch, double* roll) {
  *pitch = std::atan2(-x[6], std::hypot(x[0], x[3]));
  *roll = std::atan2(x[7], x[8]);
}

// Secant-solve ankle (pitch, roll) qpos so the foot body matches the flat
// reference orientation. Mirrors ankle_zero_snap.solve_flat exactly.
inline void SolveFlat(const mjModel* m, mjData* d, double* qpos, int bid,
                      int adr_p, int adr_r, double ref_p, double ref_r,
                      double* sp, double* sr) {
  auto pr = [&](double* q, double* p, double* r) {
    mju_copy(d->qpos, q, m->nq);
    mj_forward(m, d);
    FootPR(d->xmat + 9 * bid, p, r);
  };
  for (int it = 0; it < 30; it++) {
    double p, r;
    pr(qpos, &p, &r);
    double ep = p - ref_p, er = r - ref_r;
    if (std::fabs(ep) < 1e-5 && std::fabs(er) < 1e-5) break;
    const int adrs[2] = {adr_p, adr_r};
    const double errs[2] = {ep, er};
    for (int k = 0; k < 2; k++) {
      double q2[64];  // nq <= 41 on every deploy model; hard-capped below
      int nq = m->nq < 64 ? m->nq : 64;
      mju_copy(q2, qpos, nq);
      q2[adrs[k]] += 1e-4;
      double p2, r2;
      pr(q2, &p2, &r2);
      double g = ((k == 0 ? p2 : r2) - (k == 0 ? p : r)) / 1e-4;
      if (std::fabs(g) > 1e-6) qpos[adrs[k]] -= errs[k] / g;
    }
  }
  *sp = qpos[adr_p];
  *sr = qpos[adr_r];
}

// GRAVITY ANCHOR (2026-07-18). The solve's base-orientation input was the FUSED
// IMU quaternion -- an estimate that carries gyro-bias leakage, filter lag after
// motion, and accel-fusion corruption from the operator handling the robot during
// bring-up. Measured cost on real: the same power-on zeros solved 0.5-0.7 deg
// apart across two runs (pitch L +4.19 vs +3.50) -- run-to-run calibration
// variance that IS the "autocalib lottery on top of the encoder lottery".
// The fix is the standard one (Guedelha Humanoids'16; INS levelling update):
// during a VERIFIED-STILL window the time-averaged raw accelerometer reads the
// gravity plumb-line exactly -- an absolute inclinometer (~0.05-0.1 deg MEMS
// floor) that no support force can bias (a static hold changes contact forces,
// never the direction of gravity in the sensor frame).
//   At rest the specific force is f = R^T * (0,0,+g)  (third ROW of the
//   world-from-body matrix), so with ZYX Euler angles:
//     pitch = atan2(-fx, hypot(fy, fz)),  roll = atan2(fy, fz).
// Yaw is copied from the fused quat purely for log readability: FootPR reads the
// third row of the foot xmat, and a world-yaw premultiplication Rz*R leaves that
// row untouched, so the flat-sole solve is yaw-INVARIANT by construction.
inline void GravityQuat(const double f[3], const double* quat_fused,
                        double out[4]) {
  double pitch = std::atan2(-f[0], std::hypot(f[1], f[2]));
  double roll = std::atan2(f[1], f[2]);
  double w = quat_fused[0], x = quat_fused[1], y = quat_fused[2],
         z = quat_fused[3];
  double yaw = std::atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  double qz[4], qy[4], qx[4], t[4];
  const double az[3] = {0, 0, 1}, ay[3] = {0, 1, 0}, ax[3] = {1, 0, 0};
  mju_axisAngle2Quat(qz, az, yaw);
  mju_axisAngle2Quat(qy, ay, pitch);
  mju_axisAngle2Quat(qx, ax, roll);
  mju_mulQuat(t, qz, qy);
  mju_mulQuat(out, t, qx);
  mju_normalize4(out);
}

struct Result {
  bool ok = false;
  double poff_l = 0, roff_l = 0, poff_r = 0, roff_r = 0;  // DEG, off = encoder - true
  char why[160] = "";
};

// (mean leg encoders q12[0..11] in MOTOR order, node-frame base quat wxyz) -> offsets.
// Mirrors ankle_zero_snap.estimate: flat reference = foot orientation at the
// 'stand' keyframe (flat by construction); legs are qpos[7..18] on every deploy
// model (printed joint order, both nq=41 task + nq=34 twin).
inline Result Solve(const mjModel* m, const double* q12, const double* quat_node) {
  Result out;
  int kid = mj_name2id(m, mjOBJ_KEY, "stand");
  if (kid < 0) { std::snprintf(out.why, sizeof(out.why), "no 'stand' keyframe"); return out; }
  int bl = mj_name2id(m, mjOBJ_BODY, "left_ankle_roll_link");
  int br = mj_name2id(m, mjOBJ_BODY, "right_ankle_roll_link");
  if (bl < 0 || br < 0) { std::snprintf(out.why, sizeof(out.why), "ankle_roll_link bodies missing"); return out; }
  mjData* d = mj_makeData(m);
  if (!d) { std::snprintf(out.why, sizeof(out.why), "mj_makeData failed"); return out; }
  double key[64];
  int nq = m->nq < 64 ? m->nq : 64;
  mju_copy(key, m->key_qpos + kid * m->nq, nq);
  // flat reference at the keyframe
  mju_copy(d->qpos, key, nq);
  mj_forward(m, d);
  double ref_pl, ref_rl, ref_pr, ref_rr;
  FootPR(d->xmat + 9 * bl, &ref_pl, &ref_rl);
  FootPR(d->xmat + 9 * br, &ref_pr, &ref_rr);
  // measured pose: node-frame quat + the 12 measured leg encoders
  double qpos[64];
  mju_copy(qpos, key, nq);
  mju_copy4(qpos + 3, quat_node);
  mju_normalize4(qpos + 3);
  for (int i = 0; i < 12; i++) qpos[7 + i] = q12[i];
  const int aLP = 11, aLR = 12, aRP = 17, aRR = 18;  // qpos adr (printed joint order)
  double q2[64], sp, sr;
  mju_copy(q2, qpos, nq);
  SolveFlat(m, d, q2, bl, aLP, aLR, ref_pl, ref_rl, &sp, &sr);
  out.poff_l = (qpos[aLP] - sp) * 180.0 / M_PI;
  out.roff_l = (qpos[aLR] - sr) * 180.0 / M_PI;
  mju_copy(q2, qpos, nq);
  SolveFlat(m, d, q2, br, aRP, aRR, ref_pr, ref_rr, &sp, &sr);
  out.poff_r = (qpos[aRP] - sp) * 180.0 / M_PI;
  out.roff_r = (qpos[aRR] - sr) * 180.0 / M_PI;
  mj_deleteData(d);
  out.ok = true;
  return out;
}

// Closed-loop selftest (--ankle_autocalib_selftest): plant known offsets on a
// physically-consistent flat-foot truth (solved at NON-keyframe base poses, the
// same recipe that validated the python solver to 0.000 deg) and demand recovery.
inline int Selftest(const mjModel* m) {
  int kid = mj_name2id(m, mjOBJ_KEY, "stand");
  int bl = mj_name2id(m, mjOBJ_BODY, "left_ankle_roll_link");
  int br = mj_name2id(m, mjOBJ_BODY, "right_ankle_roll_link");
  if (kid < 0 || bl < 0 || br < 0) { std::fprintf(stderr, "[autocalib-selftest] model missing key/bodies\n"); return 1; }
  mjData* d = mj_makeData(m);
  double key[64];
  int nq = m->nq < 64 ? m->nq : 64;
  mju_copy(key, m->key_qpos + kid * m->nq, nq);
  mju_copy(d->qpos, key, nq);
  mj_forward(m, d);
  double ref[4];
  FootPR(d->xmat + 9 * bl, &ref[0], &ref[1]);
  FootPR(d->xmat + 9 * br, &ref[2], &ref[3]);
  const int aLP = 11, aLR = 12, aRP = 17, aRR = 18;
  struct Case { double off_p, off_r, base_pitch_deg, base_roll_deg; };
  const Case cases[] = {{+6, 0, 0, 0},    {+6, 0, +2, 0},      {-6, 0, -1, 0},
                        {+6, +3, +2, 0},  {+4.01, +5.95, +1, 0},
                        // gravity-anchor era: base ROLL tilt too (the plumb-line
                        // now supplies roll as well as pitch)
                        {+3.5, -2, +1, +2}, {-4, +4, -2, -1.5}};
  double worst = 0.0;
  for (const Case& c : cases) {
    // physically-consistent truth: tilt the base, SOLVE the flat ankles there
    double truth[64];
    mju_copy(truth, key, nq);
    double tilt_q[4], ax[3] = {0, 1, 0}, t4[4];
    mju_axisAngle2Quat(tilt_q, ax, c.base_pitch_deg * M_PI / 180.0);
    mju_mulQuat(t4, truth + 3, tilt_q);
    mju_copy4(truth + 3, t4);
    double axr[3] = {1, 0, 0};
    mju_axisAngle2Quat(tilt_q, axr, c.base_roll_deg * M_PI / 180.0);
    mju_mulQuat(t4, truth + 3, tilt_q);
    mju_copy4(truth + 3, t4);
    mju_normalize4(truth + 3);
    double sp, sr, q2[64];
    mju_copy(q2, truth, nq);
    SolveFlat(m, d, q2, bl, aLP, aLR, ref[0], ref[1], &sp, &sr);
    truth[aLP] = sp; truth[aLR] = sr;
    mju_copy(q2, truth, nq);
    SolveFlat(m, d, q2, br, aRP, aRR, ref[2], ref[3], &sp, &sr);
    truth[aRP] = sp; truth[aRR] = sr;
    // encoders read truth + off (MOTOR order legs = qpos[7..18])
    double q12[12];
    for (int i = 0; i < 12; i++) q12[i] = truth[7 + i];
    q12[4] += c.off_p * M_PI / 180.0;  q12[5] += c.off_r * M_PI / 180.0;
    q12[10] += c.off_p * M_PI / 180.0; q12[11] += c.off_r * M_PI / 180.0;
    Result r = Solve(m, q12, truth + 3);
    double e = std::fmax(std::fmax(std::fabs(r.poff_l - c.off_p), std::fabs(r.poff_r - c.off_p)),
                         std::fmax(std::fabs(r.roff_l - c.off_r), std::fabs(r.roff_r - c.off_r)));
    worst = std::fmax(worst, e);
    std::fprintf(stderr,
                 "[autocalib-selftest] plant P%+.2f R%+.2f @baseP%+.1f R%+.1f -> "
                 "P L%+.2f R%+.2f  R L%+.2f R%+.2f  err %.4f deg  %s\n",
                 c.off_p, c.off_r, c.base_pitch_deg, c.base_roll_deg,
                 r.poff_l, r.poff_r, r.roff_l,
                 r.roff_r, e, (r.ok && e < 0.05) ? "OK" : "FAIL <<<");
    // GRAVITY PATH: synthesize the still-window accel this truth pose would
    // produce, f = R^T*(0,0,g) (f_i = g*R[6+i], row-major world-from-body),
    // rebuild the base quat via GravityQuat, and demand the SAME recovery.
    // This closes the loop on the accel->tilt math and its sign conventions.
    {
      double R9[9];
      mju_quat2Mat(R9, truth + 3);
      double f[3] = {9.81 * R9[6], 9.81 * R9[7], 9.81 * R9[8]};
      double bq_g[4];
      GravityQuat(f, truth + 3, bq_g);
      Result rg = Solve(m, q12, bq_g);
      double eg = std::fmax(
          std::fmax(std::fabs(rg.poff_l - c.off_p), std::fabs(rg.poff_r - c.off_p)),
          std::fmax(std::fabs(rg.roff_l - c.off_r), std::fabs(rg.roff_r - c.off_r)));
      worst = std::fmax(worst, eg);
      std::fprintf(stderr,
                   "[autocalib-selftest]   gravity path                  -> "
                   "P L%+.2f R%+.2f  R L%+.2f R%+.2f  err %.4f deg  %s\n",
                   rg.poff_l, rg.poff_r, rg.roff_l, rg.roff_r, eg,
                   (rg.ok && eg < 0.05) ? "OK" : "FAIL <<<");
    }
  }
  mj_deleteData(d);
  std::fprintf(stderr, "[autocalib-selftest] worst %.4f deg -> %s\n", worst,
               worst < 0.05 ? "PASS" : "FAIL");
  return worst < 0.05 ? 0 : 1;
}

}  // namespace anklecalib

// Globals for the MuJoCo sensor callback (mirror grpc/agent_service.cc).
mjpc::Agent g_agent;
const mjModel* g_agent_model = nullptr;
mjModel* g_model = nullptr;
mjpc::Task* g_task = nullptr;
std::atomic<bool> g_exit{false};
// --- LIVE STRATEGY-SWITCH BLEND: ease the pre-switch pose -> the new policy target over
//     g_switch_blend sec (same idea as the cold-start policy-blend, but re-armed on EVERY
//     live switch, so pressing 8 mid-run descends smoothly instead of snapping). g_switch_from
//     is written only by the main loop; the stdin thread just raises g_switch_pending and sets
//     the duration.
std::atomic<bool> g_switch_pending{false};
std::atomic<double> g_switch_wall{-1e9};   // PLANT time (phase_t) the active switch-blend started
std::atomic<double> g_switch_blend{0.0};   // duration of the active switch-blend (sec)
double g_switch_from[kMaxNU] = {0};        // captured pre-switch commanded pose

// ---- PHASE-A start-pose align: operator GO gate (--align_wait) ----
// The node parks on the aligned stance and holds it (still publishing at 200 Hz, so the safety
// layer stays fed and the legs stay stiff) until the operator presses Enter. That is the moment
// the operator lets go / settles the robot -- the planner must not inherit the robot mid-handling.
// g_align_waiting is written only by the main loop; the stdin thread only raises g_align_go.
std::atomic<bool> g_align_waiting{false};  // main loop -> stdin thread: "a bare Enter means GO"
std::atomic<bool> g_align_go{false};       // stdin thread -> main loop: "hand over now"

void residual_sensor_callback(const mjModel* m, mjData* d, int stage) {
  if (m == g_agent_model || m == g_model) {
    if (stage == mjSTAGE_ACC) {
      g_task->Residual(m, d, d->sensordata);
    }
  }
}

void on_signal(int) { g_exit.store(true); }

// M5: MuJoCo fatal-error handler. Rather than a bare exit that abandons the
// robot at its last latched command, publish a damping safe-hold first, then
// terminate. (Threads cannot be joined from inside mju_error, so we still exit
// -- but the robot is left in a damped stop, not driving a stale target.)
std::function<void()> g_emit_safe_hold;   // set once the cmd publisher exists
void FatalMjuError(const char* msg) {
  std::fprintf(stderr, "[mju_error] %s -- emitting safe-hold then aborting\n", msg);
  if (g_emit_safe_hold) g_emit_safe_hold();
  std::fflush(stderr);
  std::_Exit(1);
}
void LogMjuWarning(const char* msg) { std::fprintf(stderr, "[mju_warning] %s\n", msg); }

#ifdef H12_NODE_GRPC
// gRPC service over the node's LIVE agent so the existing MJPC monitor can
// attach. READ-ONLY reflection (state/action/residuals/costs/metrics/params)
// delegates to the same grpc_agent_util helpers agent_server uses; the only
// mutator exposed is SetTaskParameters (the live Strategy switch). All
// state/planner-driving RPCs are inert no-ops so the monitor (or a stray
// button) can never disturb the node's own control loop. A mutex serialises
// concurrent gRPC calls on the service's scratch mjData.
class NodeAgentService final : public agent::Agent::Service {
 public:
  NodeAgentService(mjpc::Agent* ag, const mjModel* m)
      : agent_(ag), model_(m), data_(mj_makeData(m)), rollout_data_(mj_makeData(m)) {}
  ~NodeAgentService() override {
    if (data_) mj_deleteData(data_);
    if (rollout_data_) mj_deleteData(rollout_data_);
  }
  grpc::Status GetState(grpc::ServerContext*, const agent::GetStateRequest*,
                        agent::GetStateResponse* response) override {
    std::lock_guard<std::mutex> lk(mu_);
    agent_->state.CopyTo(model_, data_);
    return grpc_agent_util::GetState(model_, data_, response);
  }
  grpc::Status GetAction(grpc::ServerContext*, const agent::GetActionRequest* request,
                         agent::GetActionResponse* response) override {
    std::lock_guard<std::mutex> lk(mu_);
    return grpc_agent_util::GetAction(request, agent_, model_, rollout_data_,
                                      &rollout_state_, response);
  }
  grpc::Status GetResiduals(grpc::ServerContext*, const agent::GetResidualsRequest* request,
                            agent::GetResidualsResponse* response) override {
    std::lock_guard<std::mutex> lk(mu_);
    agent_->state.CopyTo(model_, data_);
    mj_forward(model_, data_);
    return grpc_agent_util::GetResiduals(request, agent_, model_, data_, response);
  }
  grpc::Status GetCostValuesAndWeights(
      grpc::ServerContext*, const agent::GetCostValuesAndWeightsRequest* request,
      agent::GetCostValuesAndWeightsResponse* response) override {
    std::lock_guard<std::mutex> lk(mu_);
    agent_->state.CopyTo(model_, data_);
    mj_forward(model_, data_);
    return grpc_agent_util::GetCostValuesAndWeights(request, agent_, model_, data_, response);
  }
  grpc::Status GetMetrics(grpc::ServerContext*, const agent::GetMetricsRequest* request,
                          agent::GetMetricsResponse* response) override {
    std::lock_guard<std::mutex> lk(mu_);
    agent_->state.CopyTo(model_, data_);
    mj_forward(model_, data_);
    grpc::Status status =
        grpc_agent_util::GetMetrics(request, agent_, model_, data_, response);
    // Identity beacon: this gRPC service is the REAL-HARDWARE deploy node
    // (reading rt/lowstate), NOT an MJPC sim agent_server. The monitor reads
    // this sentinel to auto-label the connection as "REAL HW" and stream live
    // -- no special port or flag needed. It's just a map<string,double> key,
    // so adding it requires no agent.proto change. The monitor strips it
    // before plotting, so it never shows up as a metric.
    (*response->mutable_values())["__real_hardware__"] = 1.0;
    return status;
  }
  grpc::Status GetTaskParameters(grpc::ServerContext*, const agent::GetTaskParametersRequest* request,
                                 agent::GetTaskParametersResponse* response) override {
    return grpc_agent_util::GetTaskParameters(request, agent_, response);
  }
  grpc::Status SetTaskParameters(grpc::ServerContext*, const agent::SetTaskParametersRequest* request,
                                 agent::SetTaskParametersResponse*) override {
    return grpc_agent_util::SetTaskParameters(request, agent_);  // <-- live Strategy switch
  }
  grpc::Status GetMode(grpc::ServerContext*, const agent::GetModeRequest* request,
                       agent::GetModeResponse* response) override {
    return grpc_agent_util::GetMode(request, agent_, response);
  }
  grpc::Status GetAllModes(grpc::ServerContext*, const agent::GetAllModesRequest* request,
                           agent::GetAllModesResponse* response) override {
    return grpc_agent_util::GetAllModes(request, agent_, response);
  }
  // inert: the node owns state + planning; never let a client drive/reset them.
  grpc::Status Init(grpc::ServerContext*, const agent::InitRequest*,
                    agent::InitResponse*) override { return grpc::Status::OK; }
  grpc::Status SetState(grpc::ServerContext*, const agent::SetStateRequest*,
                        agent::SetStateResponse*) override { return grpc::Status::OK; }
  grpc::Status PlannerStep(grpc::ServerContext*, const agent::PlannerStepRequest*,
                           agent::PlannerStepResponse*) override { return grpc::Status::OK; }
  grpc::Status Step(grpc::ServerContext*, const agent::StepRequest*,
                    agent::StepResponse*) override { return grpc::Status::OK; }
  grpc::Status Reset(grpc::ServerContext*, const agent::ResetRequest*,
                     agent::ResetResponse*) override { return grpc::Status::OK; }

 private:
  mjpc::Agent* agent_;
  const mjModel* model_;
  mjData* data_;
  mjData* rollout_data_;
  mjpc::State rollout_state_;
  std::mutex mu_;
};
#endif  // H12_NODE_GRPC

// ---- DEBUG PLAN PUBLISH: serialize a Trajectory's qpos rows to JSON ----
// Hand-rolled rather than nlohmann: libmjpc links nlohmann_json but the colcon
// package that actually builds this binary (core_ws/src/h12_deploy_mjpc/
// CMakeLists.txt) does not put _deps/json-src/include on the include path, so
// <nlohmann/json.hpp> would not compile here. It is also far faster to append
// ~4k doubles into one reserved buffer than to build a DOM and dump it.
//
// Emits qpos ONLY (not the full nq+nv+na state): the visualizer's ghost needs
// nothing else and it halves the payload (~180 KB -> ~90 KB).
//
// LANDMINE: traj->states is Allocate()d to kMaxTrajectoryHorizon (512) but only
// traj->horizon rows are valid -- Rollout overwrites `horizon` with the actual
// step count. Iterate horizon (101 for this task), never states.size().
void AppendPlanJson(const mjpc::Trajectory* traj, int nq, std::int64_t plan_iter,
                    std::string* out) {
  char buf[40];
  auto num = [&](double v) {
    // %.6g keeps rad-scale qpos well inside double round-trip needs for a
    // debug view while holding the payload to ~90 KB.
    int n = std::snprintf(buf, sizeof(buf), "%.6g", v);
    out->append(buf, n > 0 ? n : 0);
  };
  const int H = traj->horizon;
  const int ds = traj->dim_state;

  out->clear();
  out->reserve(static_cast<std::size_t>(H) * nq * 9 + 256);
  out->append("{\"stamp\":");
  num(H > 0 ? traj->times[0] : 0.0);
  out->append(",\"plan_iter\":");
  {
    int n = std::snprintf(buf, sizeof(buf), "%lld",
                          static_cast<long long>(plan_iter));
    out->append(buf, n > 0 ? n : 0);
  }
  out->append(",\"nq\":");
  { int n = std::snprintf(buf, sizeof(buf), "%d", nq); out->append(buf, n > 0 ? n : 0); }
  out->append(",\"horizon\":");
  { int n = std::snprintf(buf, sizeof(buf), "%d", H); out->append(buf, n > 0 ? n : 0); }

  out->append(",\"times\":[");
  for (int t = 0; t < H; t++) {
    if (t) out->push_back(',');
    num(traj->times[t]);
  }
  out->append("],\"qpos\":[");
  for (int t = 0; t < H; t++) {                   // row-major: H rows of nq
    const double* q = traj->states.data() + static_cast<std::size_t>(t) * ds;
    for (int i = 0; i < nq; i++) {
      if (t || i) out->push_back(',');
      num(q[i]);
    }
  }
  out->append("],\"total_return\":");
  num(traj->total_return);
  out->append(",\"failure\":");
  out->append(traj->failure ? "true" : "false");
  out->append("}");
}

}  // namespace

int RunDeployNode(const NodeConfig& cfg) {
  const std::string& task_id = cfg.task_id;
  const double gff = cfg.gravity_ff;
  const double ctrl_hz = kCtrlHz;
  const double ctrl_dt = 1.0 / ctrl_hz;
  const double warmup_sec = kWarmupSec;
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  mju_user_error = FatalMjuError;      // headless: safe-hold + exit instead of blocking on getchar
  mju_user_warning = LogMjuWarning;

  // ---- MJPC Agent init (mirrors AgentService::Init) ----
  g_agent.SetTaskList(mjpc::GetTasks());
  int task_index = g_agent.GetTaskIdByName(task_id);
  if (task_index < 0) {
    std::fprintf(stderr, "[node] unknown task '%s'\n", task_id.c_str());
    return 1;
  }
  g_agent.gui_task_id = task_index;   // LoadModel() + Initialize() key off gui_task_id
  g_agent.SetTaskByIndex(task_index);
  auto lm = g_agent.LoadModel();
  if (!lm.model) {
    std::fprintf(stderr, "[node] LoadModel failed: %s\n", lm.error.c_str());
    return 1;
  }
  // ---- Per-strategy planner overrides (GENERIC -- no strategy hardcoded here) --
  // A strategy may need different planner model numerics than the stock stand --
  // e.g. "Stumble" needs a higher control bandwidth (sampling_spline_points 3->5)
  // so its gait clock can lift the swing foot; the stand-tuned 3 knots cannot
  // represent the leg oscillation (8 over-actuates and destabilises even a plain
  // stand -- 5 is the twin-validated sweet spot, 2026-06-18). The node stays
  // strategy-AGNOSTIC: it asks the TASK what numerics this strategy wants
  // (Task::PlannerNumericOverrides, default empty; the lean task overrides it per
  // strategy name) and applies them blindly. Adding a strategy with custom
  // bandwidth is a fork-side edit to the task only -- this deploy node never
  // changes. Applied HERE -- before Agent::Initialize, when the sampling policy
  // reads these numerics (planners/sampling/policy.cc). Keyed off the STARTUP
  // strategy (the planner policy size is fixed at Initialize; live-switching
  // INTO another strategy keeps the startup bandwidth).
  {
    mjModel* sm = lm.model.get();
    const int strat = cfg.strategy;
    for (const auto& kv : g_agent.ActiveTask()->PlannerNumericOverrides(strat)) {
      int id = mj_name2id(sm, mjOBJ_NUMERIC, kv.first.c_str());
      if (id >= 0) {
        sm->numeric_data[sm->numeric_adr[id]] = kv.second;
        std::fprintf(stderr, "[node] planner override: %s = %.4g (strategy %d)\n",
                     kv.first.c_str(), kv.second, strat);
      } else {
        std::fprintf(stderr,
                     "[node] planner override NUMERIC MISSING: %s (strategy %d)\n",
                     kv.first.c_str(), strat);
      }
    }
    // CLI override of sampling_trajectories, applied LAST so it beats the
    // per-strategy default (e.g. trot's 36). The R2 sweep lever: rollouts-vs-
    // replan-rate on real hardware without a rebuild. 0 = leave the task default.
    if (cfg.plan_trajectories > 0) {
      int id = mj_name2id(sm, mjOBJ_NUMERIC, "sampling_trajectories");
      if (id >= 0) {
        sm->numeric_data[sm->numeric_adr[id]] =
            static_cast<double>(cfg.plan_trajectories);
        std::fprintf(stderr,
                     "[node] --plan_trajectories CLI override: sampling_trajectories = %d "
                     "(beats strategy %d default)\n", cfg.plan_trajectories, strat);
      } else {
        std::fprintf(stderr,
                     "[node] --plan_trajectories set but 'sampling_trajectories' numeric "
                     "MISSING in model -> ignored\n");
      }
    }
  }
  PatchActuators(lm.model.get(), cfg);   // safety-operational gains + estop forceranges BEFORE the agent uses it
  g_agent.Initialize(lm.model.get());
  g_agent.Allocate();
  g_agent.Reset();
  if (cfg.cem_best_action) {
    // --cem_best_action: hand the plant the single lowest-cost elite instead of
    // the elite MEAN. Anti-hedging for the commit surge: under state noise the
    // elites disagree and their mean attenuates the commanded step.
    auto* cem = dynamic_cast<mjpc::CrossEntropyPlanner*>(&g_agent.ActivePlanner());
    if (cem) {
      cem->SetBestAction(true);
      std::fprintf(stderr, "[node] --cem_best_action: executing lowest-cost ELITE "
                           "(distribution update unchanged)\n");
    } else {
      std::fprintf(stderr, "[node] --cem_best_action IGNORED: active planner is "
                           "not CrossEntropy\n");
    }
  }
  g_task = g_agent.ActiveTask();
  g_agent_model = g_agent.GetModel();
  g_model = mj_copyModel(nullptr, g_agent_model);
  PatchActuators(g_model, cfg);   // latency-comp rollout model (the planner model was patched at LoadModel above)
  // --ankle_autocalib_selftest: validate the C++ solver against planted offsets
  // (no robot, no DDS) and exit. Run this after ANY edit to anklecalib::.
  if (cfg.ankle_autocalib_selftest) std::exit(anklecalib::Selftest(g_model));
  mjData* data = mj_makeData(g_model);
  int home = mj_name2id(g_model, mjOBJ_KEY, "home");
  if (home < 0) home = mj_name2id(g_model, mjOBJ_KEY, "stand");  // upper task has only "stand"
  if (home >= 0) mj_resetDataKeyframe(g_model, data, home);
  mjcb_sensor = residual_sensor_callback;
  g_agent.SetState(data);
  g_agent.plan_enabled = true;
  g_agent.action_enabled = true;
  // Strategy 18 ("Squatter") is a native multi-phase MJPC strategy -- it loads like any other
  // index and the task's phase machine (ActiveTask()->Transition() in the loop) cycles stand_up<->
  // crouch on its own. The cold-start ramp + policy_blend eases into phase 0 (stand) just like
  // strat 6, then the first crouch fires once stand is achieved + sustained (success_sustain_time).
  g_agent.SetParamByName("residual_Strategy", static_cast<double>(cfg.strategy));
  // --straighten_start: DISARM the strategy's live-seed funnel until the ENTER
  // hand-over reaches FULL planner authority -- else the target glide
  // (target_ramp_sec) burns down during the operator hold + settle/blend and
  // the planner faces the full posture gap at handover (real-run forensics
  // 2026-07-15: Posture residual plateaued at 4.79 before GO -> CoM launch ->
  // toe-park at the ankle budget). The task re-pins its seed + phase clocks
  // each tick while disarmed; the hand-over-complete branch below arms it.
  // SetParamByName returns -1 harmlessly on tasks without the param (Lean
  // keeps its boot-seeded funnel until the same fix is ported there).
  if (cfg.straighten_start)
    g_agent.SetParamByName("residual_Funnel Arm", 0.0);

  const int nq = g_model->nq, nv = g_model->nv, nu = g_model->nu;
  const int nact = nu < cfg.nu ? nu : cfg.nu;

  // ---- GRASP arm-hold latch: resolve the LIVE q*-hold params by NAME so the
  // control loop can freeze the RIGHT ARM at the orchestrator's IK q* during the
  // gripper close -- the precise terminal pose the sampling-MPC can't hit. These
  // params exist ONLY on the Grasp task (appended after Cmd Wz so lean.h indices
  // never shift), so this resolves to -1/disabled on Lean/Stabilize/etc, and stays
  // OFF until the orchestrator sets "Arm Hold"=1 over gRPC -> every non-grasp run,
  // and every grasp run before the close, is byte-identical. Right arm = actuator
  // /motor rows 20..26 == qadr 27..33 (verified 2026-07-21). ----
  int armhold_idx = -1;
  int armhold_q_idx[7] = {-1, -1, -1, -1, -1, -1, -1};
  double armhold_lo[7] = {0}, armhold_hi[7] = {0};
  {
    auto pidx = [&](const char* nm) -> int {              // task->parameters index by name
      int id = mj_name2id(g_model, mjOBJ_NUMERIC,
                          (std::string("residual_") + nm).c_str());
      if (id < 0) return -1;
      int first = 0;
      for (; first < g_model->nnumeric; first++) {
        const char* n = mj_id2name(g_model, mjOBJ_NUMERIC, first);
        if (n && std::string(n).rfind("residual_", 0) == 0) break;
      }
      return id - first;
    };
    const char* qn[7] = {"Arm Hold qR0", "Arm Hold qR1", "Arm Hold qR2", "Arm Hold qR3",
                         "Arm Hold qR4", "Arm Hold qR5", "Arm Hold qR6"};
    int a = pidx("Arm Hold");
    bool ok = (a >= 0) && (cfg.nu >= 27) && (g_model->nu >= 27);
    for (int k = 0; k < 7 && ok; k++) { armhold_q_idx[k] = pidx(qn[k]); ok = armhold_q_idx[k] >= 0; }
    if (ok) {
      armhold_idx = a;
      for (int k = 0; k < 7; k++) {                        // clamp bounds = right-arm joint limits
        int jid = g_model->actuator_trnid[(20 + k) * 2];
        armhold_lo[k] = g_model->jnt_range[jid * 2];
        armhold_hi[k] = g_model->jnt_range[jid * 2 + 1];
      }
    }
    std::fprintf(stderr, "[node] grasp arm-hold latch: %s\n",
                 armhold_idx >= 0
                     ? "present (right arm 20..26 <- IK q* when 'Arm Hold'=1; OFF until signalled)"
                     : "absent (non-grasp task)");
  }

  // ---- ACTIVE-CONFIG ECHO: print the LIVE strat-21 reach numerics this binary
  // actually loaded, so a stale binary / wrong value is obvious in the log header
  // (the 2026-06-23 "node 2h stale, reach_com_back ignored" class). g_model is the
  // fully-loaded planner model -> these are exactly what the residual reads. ----
  {
    auto NUM = [&](const char* nm, double dflt) {
      int id = mj_name2id(g_model, mjOBJ_NUMERIC, nm);
      return id >= 0 ? g_model->numeric_data[g_model->numeric_adr[id]] : dflt;
    };
    int rh = static_cast<int>(std::lround(NUM("reach_hand", -1)));
    std::fprintf(stderr,
      "[node] reach config (LIVE numerics): reach_hand=%d(%s) radius=%.2f drop=%.2f "
      "com_back=%.3f brace_hold=%.1f  [if these are wrong -> stale build / bad XML]\n",
      rh, rh == 2 ? "RIGHT" : rh == 1 ? "LEFT" : "auto",
      NUM("reach_radius", 0.46), NUM("reach_drop", 0.36),
      NUM("reach_com_back", 0.0), NUM("reach_brace_hold", 1.0));

    // ---- LEAN GATE ECHO (2026-07-28). The line above covers strat 21 only, and
    // that cost us: the 07-28 overnight batch produced ONE full-pipeline completion
    // whose counterbalance_radius / brace_target_face values could not be recovered
    // afterwards, because the driver lived in a scratchpad that was cleared and NO
    // log recorded them. Every gate that changes lean behaviour is echoed here so a
    // trace is always self-documenting. Printed only when the lean gates exist, so
    // non-lean tasks keep their current header verbatim.
    if (mj_name2id(g_model, mjOBJ_NUMERIC, "counterbalance_radius") >= 0) {
      std::fprintf(stderr,
        "[node] lean gates (LIVE numerics): counterbalance_radius=%.3f brace_target_face=%.0f "
        "brace_press_depth=%.3f phase_ramp_sec=%.2f table_contact_exclusive=%.0f "
        "com_x_offset_support=%.3f posture_leg_level=%.0f brace_gate_fix=%.0f\n",
        NUM("counterbalance_radius", 0.0), NUM("brace_target_face", 0.0),
        NUM("brace_press_depth", 0.06), NUM("phase_ramp_sec", 0.0),
        NUM("table_contact_exclusive", 0.0), NUM("com_x_offset_support", 0.0),
        NUM("posture_leg_level", 0.0), NUM("brace_gate_fix", 0.0));
    }
  }
  std::fprintf(stderr,
               "[node] task='%s' nq=%d nv=%d nu=%d strategy=%d gravity_ff=%.2f ctrl_hz=%.0f\n",
               task_id.c_str(), nq, nv, nu, cfg.strategy, gff, ctrl_hz);

#ifdef H12_NODE_GRPC
  // ---- MJPC gRPC server (started early so the monitor can attach anytime) ----
  std::unique_ptr<NodeAgentService> grpc_service;
  std::unique_ptr<grpc::Server> grpc_server;
  const int grpc_port = cfg.grpc_port;
  if (grpc_port > 0) {
    grpc_service = std::make_unique<NodeAgentService>(&g_agent, g_model);
    grpc::ServerBuilder builder;
    builder.AddListeningPort(absl::StrCat("[::]:", grpc_port),
                             grpc::experimental::LocalServerCredentials(LOCAL_TCP));
    builder.SetMaxReceiveMessageSize(50 * 1024 * 1024);
    builder.RegisterService(grpc_service.get());
    grpc_server = builder.BuildAndStart();
    if (grpc_server) {
      std::fprintf(stderr,
                   "[node] MJPC gRPC server on :%d -- attach the monitor to this "
                   "port to view state + switch Strategy live\n", grpc_port);
    } else {
      std::fprintf(stderr,
                   "[node] WARNING: gRPC server failed on :%d (port in use?) -- "
                   "continuing without the monitor\n", grpc_port);
    }
  }
#endif

  // ---- DDS: subscribe rt/lowstate + rt/sportmodestate, publish lowcmd ----
  // Resolve the interface: explicit --network_interface wins; otherwise auto-pin
  // the 192.168.123.x robot-subnet NIC (so NO flag is needed on the real robot);
  // fall back to autodetermine/loopback for the same-host twin.
  std::string net_if = cfg.network_interface;
  if (net_if.empty()) {
    std::string auto_if = AutoDetectRobotInterface();
    if (!auto_if.empty()) {
      net_if = auto_if;
      std::fprintf(stderr,
                   "[node] auto-pinned DDS interface '%s' (192.168.123.x robot "
                   "subnet); pass --network_interface to override\n",
                   net_if.c_str());
    } else {
      std::fprintf(stderr,
                   "[node] no robot-subnet NIC found -> DDS autodetermine "
                   "(twin/loopback); pass --network_interface for the robot\n");
    }
  } else {
    std::fprintf(stderr,
                 "[node] DDS interface '%s' (explicit --network_interface)\n",
                 net_if.c_str());
  }
  ChannelFactory::Instance()->Init(cfg.domain_id, net_if);
  RobotState rs;

  const int n_upper = cfg.upper_count;
  const int moff = cfg.motor_offset;       // actuated joint i <-> motor (moff + i)
  const int coff = cfg.comp_motor_offset;  // complement joint i <-> motor (coff + i)
  ChannelSubscriberPtr<LowState> ls_sub(
      new ChannelSubscriber<LowState>(cfg.lowstate_topic));
  ls_sub->InitChannel(
      [&rs, &cfg, n_upper, moff, coff](const void* msg) {
        const LowState* s = static_cast<const LowState*>(msg);
        std::lock_guard<std::mutex> lk(rs.mu);
        for (int i = 0; i < cfg.nu; i++) {
          rs.d.q[i] = s->motor_state().at(moff + i).q();
          rs.d.dq[i] = s->motor_state().at(moff + i).dq();
          rs.d.tau[i] = s->motor_state().at(moff + i).tau_est();
        }
        // X-aware: also latch the COMPLEMENT joints (arm-aware: motor 12..26 on the
        // legs-only node; leg-aware: motor 0..11 on the upper-body node) so the
        // planner CoM/eq-locks can track them. motor_state() is a fixed-size array
        // on the H1-2 (>=27); guard defensively.
        for (int i = 0; i < n_upper && static_cast<size_t>(coff + i) < s->motor_state().size(); i++) {
          rs.d.qu[i] = s->motor_state().at(coff + i).q();
          rs.d.dqu[i] = s->motor_state().at(coff + i).dq();
        }
        for (int k = 0; k < 4; k++) rs.d.quat[k] = s->imu_state().quaternion().at(k);
        for (int k = 0; k < 3; k++) rs.d.gyro[k] = s->imu_state().gyroscope().at(k);
        for (int k = 0; k < 3; k++) rs.d.acc[k] = s->imu_state().accelerometer().at(k);
        rs.d.mode_machine = s->mode_machine();
        rs.d.tick = s->tick();
        rs.d.ls_stamp = std::chrono::steady_clock::now();   // H1 watchdog stamp
        rs.d.have_ls = true;
      },
      10);

  ChannelSubscriberPtr<SportState> ss_sub(
      new ChannelSubscriber<SportState>(cfg.sportstate_topic));
  ss_sub->InitChannel(
      [&rs](const void* msg) {
        const SportState* s = static_cast<const SportState*>(msg);
        std::lock_guard<std::mutex> lk(rs.mu);
        for (int k = 0; k < 3; k++) {
          rs.d.site_p[k] = s->position().at(k);
          rs.d.site_v[k] = s->velocity().at(k);
        }
        rs.d.ss_stamp = std::chrono::steady_clock::now();   // H1 watchdog stamp
        rs.d.have_ss = true;
      },
      10);

  ChannelPublisherPtr<LowCmd> cmd_pub(
      new ChannelPublisher<LowCmd>(cfg.lowcmd_topic));
  cmd_pub->InitChannel();

  // ---- H1/M5 safe-hold: damping stop (kp=0, small kd, tau=0) at the last measured
  // pose (harmless with kp=0). Callable from the control loop (stale state) or from
  // FatalMjuError; snapshots the state under the mutex.
  auto emit_safe_hold = [&rs, &cmd_pub, &cfg, moff]() {
    StateData snap;
    {
      std::lock_guard<std::mutex> lk(rs.mu);
      snap = rs.d;
    }
    LowCmd cmd{};
    cmd.mode_pr() = 0;
    cmd.mode_machine() = snap.mode_machine;
    for (int i = 0; i < cfg.nu; i++) {
      auto& mc = cmd.motor_cmd().at(moff + i);
      mc.mode() = 1;
      mc.q() = snap.have_ls ? static_cast<float>(snap.q[i]) : 0.0f;
      mc.dq() = 0.0f;
      mc.tau() = 0.0f;
      mc.kp() = 0.0f;
      mc.kd() = kSafeHoldKd;
    }
    cmd.crc() = Crc32Core(reinterpret_cast<uint32_t*>(&cmd), (sizeof(LowCmd) >> 2) - 1);
    cmd_pub->Write(cmd);
  };
  g_emit_safe_hold = emit_safe_hold;   // M5: mju_error path publishes this before exiting

  // ---- X-AWARE balance: map each equality lock -> the COMPLEMENT-joint index it
  //      holds (qposadr-7), so fill_state can retarget the lock to the MEASURED
  //      pose every tick. Legs-only node = ARM-aware (locks hold torso+arms);
  //      upper-body node = LEG-aware (locks hold the 12 legs, plus the pelvis
  //      world-weld follows the measured base). The planner never actuates the
  //      complement joints. ----
  const bool arm_aware = (n_upper > 0) && cfg.arm_aware;
  std::vector<int> eq_motor(g_model->neq, -1);   // eq e -> complement motor idx, -1 otherwise
  std::vector<char> eq_weld(g_model->neq, 0);    // eq e is a world-weld (upper node: pelvis)
  if (arm_aware) {
    for (int e = 0; e < g_model->neq; e++) {
      if (g_model->eq_type[e] == mjEQ_WELD) { eq_weld[e] = 1; continue; }
      if (g_model->eq_type[e] != mjEQ_JOINT) continue;
      int midx = g_model->jnt_qposadr[g_model->eq_obj1id[e]] - 7;  // joint/motor index (0..26)
      if (midx >= coff && midx < coff + n_upper) eq_motor[e] = midx;
    }
    int n = 0; for (int e = 0; e < g_model->neq; e++) if (eq_motor[e] >= 0) n++;
    std::fprintf(stderr, "[node] %s-AWARE ON: reading %d complement joints (motor %d..%d) from "
                         "rt/lowstate; %d equality locks retargeted to the measured pose each tick "
                         "(node actuates only its own nu=%d rows)\n",
                 coff == 12 ? "ARM" : "LEG", n_upper, coff, coff + n_upper - 1, n, cfg.nu);
  } else if (n_upper > 0) {
    std::fprintf(stderr, "[node] X-aware OFF: complement joints held at home in the planner "
                         "(validated home-locked stand). Pass --arm_aware to track the measured pose.\n");
  }

  // ---- wait for the first full state ----
  std::fprintf(stderr, "[node] waiting for %s + %s ...\n",
               cfg.lowstate_topic.c_str(), cfg.sportstate_topic.c_str());
  while (!g_exit.load()) {
    {
      std::lock_guard<std::mutex> lk(rs.mu);
      if (rs.d.have_ls && rs.d.have_ss) break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  std::fprintf(stderr, "[node] state stream up -> starting continuous planner.\n");

  // ---- planner: ASYNC background PlanIteration thread (the deployed mode; the
  // old --sync_plan / --plan_rate_hz synchronous paths were never used in any
  // documented run and were removed in the 2026-07-02 flag diet). ----
  std::atomic<bool> plan_exit{false};
  std::atomic<long> plan_count{0};  // REAL planner-iteration counter (PlanSteps() is the HORIZON, not iters)
  const int hw_threads = mjpc::NumAvailableHardwareThreads();
  const int n_plan_threads =
      cfg.plan_threads > 0
          ? cfg.plan_threads
          : (kPlanThreads > 0
                 ? kPlanThreads
                 : std::max(kPlanThreadsMin, hw_threads - kPlanThreadsReserve));
  std::fprintf(stderr,
               "[node] planner ThreadPool: %d threads (%d hw available; %s; reserve %d for "
               "control/DDS/safety)\n",
               n_plan_threads, hw_threads,
               cfg.plan_threads > 0 ? "--plan_threads CLI"
                                    : (kPlanThreads > 0 ? "compiled kPlanThreads"
                                                        : "AUTO = hw - reserve"),
               kPlanThreadsReserve);

  // ---- ONE-WAVE CHECK (2026-07-11) -----------------------------------------------
  // CEM Rollouts schedules num_trajectory + 1 jobs (the NOMINAL rollout rides along --
  // cross_entropy/planner.cc:469 waits for count + num_trajectory + 1) and blocks on ALL
  // of them, so one plan iteration costs ceil((N+1)/threads) thread-WAVES and the replan
  // rate divides by that. This was silent, and it starved every stepping strategy on real
  // for two weeks (trot 36 traj / 12 threads = 4 waves = 27-30 plans/s, vs the 50-100 Hz
  // band real legged sampling-MPC needs). Never let it be silent again.
  {
    int ntraj = 0;
    int tid = mj_name2id(g_agent.GetModel(), mjOBJ_NUMERIC, "sampling_trajectories");
    if (tid >= 0) {
      const mjModel* am = g_agent.GetModel();
      ntraj = static_cast<int>(am->numeric_data[am->numeric_adr[tid]]);
    }
    if (ntraj > 0) {
      const int jobs = ntraj + 1;   // + the nominal rollout
      const int waves = (jobs + n_plan_threads - 1) / n_plan_threads;
      if (waves > 1) {
        std::fprintf(stderr,
                     "[node] ** PLAN-RATE WARNING: %d trajectories + 1 nominal = %d jobs on %d "
                     "threads = %d WAVES -> replan rate is ~1/%d of one-wave. Real legged "
                     "sampling-MPC needs 50-100 plans/s; the open-loop swing is rate-independent "
                     "but the stance-leg weight shift is SAMPLER-owned and starves first "
                     "(feet cycle, never unload). Fix: --plan_trajectories <= %d, or "
                     "--plan_threads >= %d.\n",
                     ntraj, jobs, n_plan_threads, waves, waves, n_plan_threads - 1, jobs);
      } else {
        std::fprintf(stderr,
                     "[node] plan schedule: %d traj + 1 nominal = %d jobs on %d threads = 1 wave "
                     "(optimal: replan rate is not divided)\n",
                     ntraj, jobs, n_plan_threads);
      }
    }
  }
  std::fprintf(stderr,
               "[node] torque-budget clamp ON (audit H2): |tau_ff + kp*(tgt-q) + kv*dq| <= %.1f x "
               "tau-estop per joint (command-side estops impossible for ANY planner output)\n",
               kClampRatio);

  // ---- DEBUG PLAN PUBLISH (off unless --plan_topic is set) ----
  // Constructed BEFORE the planner thread so the lambda can capture it; when the
  // topic is empty nothing is constructed and the planner loop below is the exact
  // pre-2026-07-15 two-liner (real deployments byte-unchanged).
  const bool plan_pub_on = !cfg.plan_pub_topic.empty();
  ChannelPublisherPtr<std_msgs::msg::dds_::String_> plan_pub;
  if (plan_pub_on) {
    plan_pub.reset(
        new ChannelPublisher<std_msgs::msg::dds_::String_>(cfg.plan_pub_topic));
    plan_pub->InitChannel();
    std::fprintf(stderr,
                 "[node] DEBUG plan publish ON -> '%s' @ %.1f Hz (qpos rows of the "
                 "active planner's best trajectory, JSON; ROS sees it as '/%s')\n",
                 cfg.plan_pub_topic.c_str(), cfg.plan_pub_hz,
                 cfg.plan_pub_topic.compare(0, 3, "rt/") == 0
                     ? cfg.plan_pub_topic.c_str() + 3
                     : cfg.plan_pub_topic.c_str());
  }

  mjpc::ThreadPool plan_pool(n_plan_threads);
  std::thread planner([&] {
    // Serialization scratch + rate gate live in the planner thread. The plan is
    // ~90 KB of JSON and the planner free-runs at 45-52/s; publishing every
    // iteration would burn ~4.5 MB/s for a 30 fps viewer that cannot use it. We
    // SKIP iterations against a steady_clock deadline rather than sleeping --
    // sleeping here would cost real plan rate, which is control-critical.
    std::string plan_json;
    const auto plan_pub_period = std::chrono::duration<double>(
        cfg.plan_pub_hz > 0.0 ? 1.0 / cfg.plan_pub_hz : 0.0);
    auto next_plan_pub = std::chrono::steady_clock::now();
    std_msgs::msg::dds_::String_ plan_msg;

    while (!plan_exit.load()) {
      g_agent.PlanIteration(&plan_pool);
      const std::int64_t iter = plan_count.fetch_add(1) + 1;

      // PlanIteration has returned => every rollout job (incl. the nominal one
      // BestTrajectory() hands back) has completed and no worker thread is
      // touching the buffer; this thread is the only one that rewrites it. So
      // reading it here is race-free WITHOUT taking the planner's mtx_.
      if (!plan_pub_on) continue;
      const auto now = std::chrono::steady_clock::now();
      if (now < next_plan_pub) continue;
      next_plan_pub = now + std::chrono::duration_cast<
          std::chrono::steady_clock::duration>(plan_pub_period);

      const mjpc::Trajectory* best = g_agent.ActivePlanner().BestTrajectory();
      // Null until the first iteration completes (SamplingPlanner returns null
      // while winner < 0); horizon 0 would serialize an empty plan.
      if (best == nullptr || best->horizon <= 0) continue;
      AppendPlanJson(best, nq, iter, &plan_json);
      plan_msg.data(plan_json);
      plan_pub->Write(plan_msg);
    }
  });

  // ---- live strategy switch via stdin: type a number 0-20 (+Enter), q=quit ----
  std::fprintf(stderr, "[node] live switch ready: type a strategy number 0-20 + Enter (q=quit)\n");
  std::thread stdin_thread([&] {
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line == "q" || line == "quit") { g_exit.store(true); break; }
      // A bare Enter (or "go") releases the start-pose hold. Only meaningful while the main loop
      // is actually parked on the aligned stance; otherwise it stays the harmless no-op it was.
      if (line.empty() || line == "g" || line == "go") {
        if (g_align_waiting.load() && !g_align_go.exchange(true))
          std::fprintf(stderr, "[node] >>> GO: releasing the start-pose hold -> MJPC takes over\n");
        continue;
      }
      try {
        int s = std::stoi(line);
        // Any strategy (incl. 18 = native Squatter, 19 = native Jab) loads the same way: set the
        // residual strategy and arm the live-switch settle+blend so the target eases from the current
        // pose into the new policy target instead of snapping. The phase machine handles the internal
        // cycling of multi-phase strategies (18 squat cycle, 19 jab guard<->extend).
        g_agent.SetParamByName("residual_Strategy", static_cast<double>(s));
        g_switch_blend.store(kPolicyBlendSec);
        g_switch_pending.store(true);   // main loop captures the from-pose + arms the blend
        std::fprintf(stderr, "[node] >>> Strategy -> %d (eases into the new pose over %.1fs)\n",
                     s, kPolicyBlendSec);
      } catch (...) {
        std::fprintf(stderr, "[node] (enter a strategy number 0-20, or q to quit)\n");
      }
    }
  });
  stdin_thread.detach();

  // scratch mjData: sd = real state for SetState; gd = qvel-zeroed for gravity FF.
  // Object/task slots beyond the robot (qpos[34:], qvel[33:]) keep home defaults.
  mjData* sd = mj_makeData(g_model);
  mjData* gd = mj_makeData(g_model);
  mjData* pdat = mj_makeData(g_model);   // latency-comp: forward-prediction rollout scratch
  if (home >= 0) {
    mj_resetDataKeyframe(g_model, sd, home);
    mj_resetDataKeyframe(g_model, gd, home);
    mj_resetDataKeyframe(g_model, pdat, home);
  }

  // IMU zero-offset calibration (deg->rad): a measured constant mounting/zero bias in the
  // base orientation feeds straight into the planner (qpos[3:7] below) and makes it balance
  // around a false vertical -> a steady lean (sim-confirmed). Cancel it here.
  const double imu_pitch_off = cfg.imu_pitch_offset_deg * M_PI / 180.0;
  const double imu_roll_off = cfg.imu_roll_offset_deg * M_PI / 180.0;
  // SESSION IMU ALIGNMENT (2026-07-19, --ac_imu_align): the autocalib's gravity
  // anchor measures the fused quat's pitch/roll error against the plumb-line in
  // every still window -- and on real it came out CONSTANT (+2.1..+2.6 deg pitch
  // across 7 windows / 3 runs). The PLANNER, however, keeps balancing in the
  // fused frame: its "upright" is that same ~2.3 deg from true, which parks the
  // believed CoM ~z*tan(2.3) ~ 3 cm off-centre -- 40% of the 8 cm heel margin
  // spent before any disturbance. Fix: when the autocalib APPLIES, also add the
  // measured (mean witness+confirm) gravity-vs-fused delta to the planning-path
  // IMU correction, blended with the same jolt-free ramp as the ankle offsets.
  // Written only from the main control loop (same thread as fill_state) => no
  // race. Cap +-4 deg; 0 windows with a gravity fix -> stays 0 (old behavior).
  double imu_align_p = 0.0, imu_align_r = 0.0;   // rad, additive to the corr above
  // EXTENDED CALIB HOLD (see Cfg::ac_hold_extra_sec): autocalib needs TWO quiet
  // windows (witness + confirm) BEFORE policy handover; the stock 3.0 s hold
  // fits one. Only --ankle_autocalib runs are affected; 0 extra = stock timing.
  const double ramp_hold_eff =
      kRampHoldSec + ((cfg.ankle_autocalib && cfg.motor_offset == 0)
                          ? std::fmax(0.0, cfg.ac_hold_extra_sec) : 0.0);
  if (ramp_hold_eff != kRampHoldSec)
    std::fprintf(stderr,
                 "[node] autocalib EXTENDED HOLD: scripted post-ramp hold %.1fs "
                 "(stock %.1fs + %.1fs) so witness+confirm+apply land BEFORE "
                 "policy handover (--ac_hold_extra 0 for stock timing)\n",
                 ramp_hold_eff, kRampHoldSec, ramp_hold_eff - kRampHoldSec);
  if (imu_pitch_off != 0.0 || imu_roll_off != 0.0)
    std::printf("[node] IMU zero-offset calibration ON: perceived base orientation rotated "
                "pitch%+.2f roll%+.2f deg before planning\n",
                cfg.imu_pitch_offset_deg, cfg.imu_roll_offset_deg);
  // NOT const: --ankle_autocalib REPLACES these with its own solve at the end of
  // the bring-up ramp hold (absolute offsets, encoder - true -- not deltas on the
  // manual flags). Written only from the main control loop, the same thread that
  // reads them in fill_state and the wire shift => no race.
  double ankle_off_l = cfg.ankle_roll_offset_l_deg * M_PI / 180.0;
  double ankle_off_r = cfg.ankle_roll_offset_r_deg * M_PI / 180.0;
  double ankle_poff_l = cfg.ankle_pitch_offset_l_deg * M_PI / 180.0;
  double ankle_poff_r = cfg.ankle_pitch_offset_r_deg * M_PI / 180.0;
  bool ankle_calib_on = (ankle_off_l != 0.0 || ankle_off_r != 0.0 ||
                         ankle_poff_l != 0.0 || ankle_poff_r != 0.0);
  if (ankle_calib_on)
    std::printf("[node] ankle zero-offset calibration ON (deg, off = encoder - true): "
                "pitch L%+.2f R%+.2f  roll L%+.2f R%+.2f\n",
                cfg.ankle_pitch_offset_l_deg, cfg.ankle_pitch_offset_r_deg,
                cfg.ankle_roll_offset_l_deg, cfg.ankle_roll_offset_r_deg);

  auto fill_state = [&](const StateData& cur) {
    double roff[3];
    QuatRot(cur.quat, kImuOffset, roff);
    double ww[3];
    QuatRot(cur.quat, cur.gyro, ww);
    double cr[3] = {ww[1] * roff[2] - ww[2] * roff[1], ww[2] * roff[0] - ww[0] * roff[2],
                    ww[0] * roff[1] - ww[1] * roff[0]};
    for (int k = 0; k < 3; k++) sd->qpos[k] = cur.site_p[k] - roff[k];  // pelvis = site - R*offset
    for (int k = 0; k < 3; k++) sd->qvel[k] = cur.site_v[k] - cr[k];    // pelvis world linvel
    // base orientation = IMU quat, with the measured zero-offset cancelled (body-frame
    // post-multiply by the small pitch/roll correction; wxyz, MuJoCo convention).
    double bq[4];
    mju_copy4(bq, cur.quat);
    // static mount corr + session gravity alignment (same axis => additive angle)
    const double pcorr = imu_pitch_off + imu_align_p;
    const double rcorr = imu_roll_off + imu_align_r;
    if (pcorr != 0.0) {
      double d[4], ax[3] = {0, 1, 0}, t[4];
      mju_axisAngle2Quat(d, ax, pcorr);
      mju_mulQuat(t, bq, d); mju_copy4(bq, t);
    }
    if (rcorr != 0.0) {
      double d[4], ax[3] = {1, 0, 0}, t[4];
      mju_axisAngle2Quat(d, ax, rcorr);
      mju_mulQuat(t, bq, d); mju_copy4(bq, t);
    }
    mju_normalize4(bq);
    for (int k = 0; k < 4; k++) sd->qpos[3 + k] = bq[k];
    for (int i = 0; i < cfg.nu; i++) sd->qpos[7 + moff + i] = cur.q[i];
    // ankle zero-offset calibration: the planner sees the CORRECTED angles (idx 4/10 =
    // ankle_pitch L/R, 5/11 = ankle_roll L/R) so it doesn't chase a foot only the encoder
    // thinks is tilted. off = encoder - true. Pairs with the wire-command shift before
    // publish below. 0 = no-op. LEG indices -> nodes that actuate the legs only
    // (motor_offset 0); on the upper node these rows are ARM joints.
    if (moff == 0) {
      sd->qpos[7 + 4]  -= ankle_poff_l;
      sd->qpos[7 + 5]  -= ankle_off_l;
      sd->qpos[7 + 10] -= ankle_poff_r;
      sd->qpos[7 + 11] -= ankle_off_r;
    }
    for (int k = 0; k < 3; k++) sd->qvel[3 + k] = cur.gyro[k];        // free-joint angvel == body gyro
    for (int i = 0; i < cfg.nu; i++) sd->qvel[6 + moff + i] = cur.dq[i];
    // X-aware: inject the MEASURED complement pose so the planner's CoM/dynamics track it,
    // and retarget the equality locks to HOLD it there during each rollout (instead of
    // snapping to home). The planner has no actuator on these joints -> it cannot command
    // them; it only plans its own rows against them. eq_data is retargeted on BOTH models
    // (g_model = Transition + latency rollout; g_agent_model = the planner's rollout model).
    // No-op when the complement is at home.
    if (arm_aware) {
      for (int i = 0; i < n_upper; i++) {
        sd->qpos[7 + coff + i] = cur.qu[i];
        sd->qvel[6 + coff + i] = cur.dqu[i];
      }
      for (int e = 0; e < g_model->neq; e++) {
        if (eq_weld[e]) {
          // pelvis world-weld (upper node): relpose (eq_data[3..9]) = measured base
          // pos+quat, so every rollout pins the base where it ACTUALLY is (the lower
          // MPC owns balance; this model only believes the result). Mirrors the
          // task-side retarget in upper.cc TransitionLocked onto the PLANNER model.
          for (int k = 0; k < 3; k++) {
            g_model->eq_data[e * mjNEQDATA + 3 + k] = sd->qpos[k];
            const_cast<mjtNum*>(g_agent_model->eq_data)[e * mjNEQDATA + 3 + k] = sd->qpos[k];
          }
          for (int k = 0; k < 4; k++) {
            g_model->eq_data[e * mjNEQDATA + 6 + k] = sd->qpos[3 + k];
            const_cast<mjtNum*>(g_agent_model->eq_data)[e * mjNEQDATA + 6 + k] = sd->qpos[3 + k];
          }
          continue;
        }
        if (eq_motor[e] < 0) continue;
        mjtNum v = cur.qu[eq_motor[e] - coff];
        g_model->eq_data[e * mjNEQDATA + 0] = v;
        const_cast<mjtNum*>(g_agent_model->eq_data)[e * mjNEQDATA + 0] = v;
      }
    }
  };

  // ---- latency compensation: roll the planner model forward by Δ under the in-flight command ----
  // The planner model's <position> actuators have kp/kv == the node's KP[]/KV[] == the twin's PD,
  // so this rollout reproduces the twin's joint law; we only add the gravity tau_ff the twin also
  // applies. Object/task slots are left at home (we copy only the robot dofs back). NaN-guarded.
  const double lat_fixed = kLatencyFixedMs * 1e-3;
  const double lat_extra = cfg.latency_extra_ms * 1e-3;   // default 4.0 == old kLatencyExtraMs
  const double lat_max   = kLatencyMaxMs * 1e-3;
  const double pred_dt   = g_model->opt.timestep;     // native model step (0.002) == twin granularity
  double ewma_comp = 1.0 / ctrl_hz;                   // measured per-tick compute time (EWMA), seeded
  double last_cmd_q[kMaxNU];
  for (int i = 0; i < cfg.nu; i++) last_cmd_q[i] = sd->qpos[7 + moff + i];   // home target until first publish
  // ---- bring-up ramp state: blend ALL joints from the measured power-on pose to the
  //      home/policy target over kStartRampSec so the warmup->policy switch never snaps.
  const double start_ramp_sec = kStartRampSec;
  std::fprintf(stderr, "[node] bring-up ramp ON: ALL joints measured->home/policy over %.1fs "
                       "(no-op if already at home)\n", start_ramp_sec);
  double home_q[kMaxNU];
  { // bring-up destination = the OPERATING stance ("stand", bent-knee), NOT the singular
    // straight-knee "home": ramping to knee=0 hands the policy the hyperextension basin
    // (knees snapped to the -0.12 stop within 1 s of ramp end -- live-verified).
    // cfg.ramp_dest_key lets an ASYMMETRIC strategy (stagger) name its own destination: ramping a
    // staggered stance to the symmetric `stand` shears the planted soles for the whole scripted
    // window. Unset -> `stand`, exactly as before.
    int hk = cfg.ramp_dest_key ? mj_name2id(g_model, mjOBJ_KEY, cfg.ramp_dest_key) : -1;
    if (hk < 0 && cfg.ramp_dest_key)
      std::fprintf(stderr, "[node] WARNING: ramp_dest_key '%s' is not a keyframe -> falling back "
                           "to 'stand' (the SQUARE stance)\n", cfg.ramp_dest_key);
    if (hk < 0) hk = mj_name2id(g_model, mjOBJ_KEY, "stand");
    if (hk < 0) hk = mj_name2id(g_model, mjOBJ_KEY, "home");
    if (hk < 0) hk = 0;
    std::fprintf(stderr, "[node] bring-up ramp destination keyframe: '%s'\n",
                 mj_id2name(g_model, mjOBJ_KEY, hk));
    for (int i = 0; i < cfg.nu; i++) home_q[i] = g_model->key_qpos[hk * nq + 7 + moff + i]; }
  double arm_q_init[kMaxNU];
  for (int i = 0; i < cfg.nu; i++) arm_q_init[i] = sd->qpos[7 + i];   // refined to measured pose on first live state
  bool arm_init_set = false;
  double ramp_eff = start_ramp_sec;   // rescaled at latch by distance-from-home (see loop)
  // ---- BRING-UP WAYPOINTS (2026-08-12, HAMS frame-task pattern): keyframes named
  //      bringup_wp1..bringup_wp8 in the task XML define intermediate stations for the
  //      scripted rise (arms swing BACK-tucked, then settle to the stance) so the ramp
  //      never sweeps the hands forward through an obstacle (real run 6: gripper snagged
  //      the table edge during the straight measured->stand lerp). Absent keys = the
  //      original single-lerp path, byte-identical.
  double bringup_wp[8][kMaxNU];
  int n_bringup_wp = 0;
  bool bringup_wps_live = false;   // latched with ramp_eff on the first live state
  for (int w = 0; w < 8; w++) {
    char nm[16];
    snprintf(nm, sizeof(nm), "bringup_wp%d", w + 1);
    int kid = mj_name2id(g_model, mjOBJ_KEY, nm);
    if (kid < 0) break;
    for (int i = 0; i < cfg.nu; i++)
      bringup_wp[n_bringup_wp][i] = g_model->key_qpos[kid * nq + 7 + moff + i];
    n_bringup_wp++;
  }
  if (n_bringup_wp > 0)
    std::fprintf(stderr, "[node] bring-up WAYPOINT ramp: measured -> %d station(s) -> "
                         "destination (%.1fs/segment; falls back to direct lerp when "
                         "starting near home)\n", n_bringup_wp, kStartRampSec * 0.5);
  // piecewise-linear path through {arm_q_init, wp1..wpN, dest} at parameter s in [0,1]
  auto bringup_path_at = [&](double s, const double* dest, int joint) -> double {
    const int nseg = n_bringup_wp + 1;
    double fs = std::fmin(std::fmax(s, 0.0), 1.0) * nseg;
    int seg = std::fmin((int)fs, nseg - 1);
    double u = fs - seg;
    const double* a = (seg == 0) ? arm_q_init : bringup_wp[seg - 1];
    const double* b = (seg == n_bringup_wp) ? dest : bringup_wp[seg];
    return (1.0 - u) * a[joint] + u * b[joint];
  };
  bool handover_active = false;       // align GO -> settle+policy-blend into MJPC (not a cold re-run)
  double handover_t0 = 0.0;           // WALL time at which the align hand-over began

  // ---- PHASE A: start-pose align (Cfg::align_start) ----
  double align_q[kMaxNU];
  for (int i = 0; i < cfg.nu; i++)
    align_q[i] = cfg.align_pose_set ? cfg.align_pose[i] : home_q[i];
  bool   align_active = cfg.align_start;
  // PHASE A': --straighten_start reuses align_start's ENTER->handover but with NO drag: hold the
  // measured (slumped) pose, wait for ENTER, then hand to the planner from the slump via the same
  // handover_active SETTLE->BLEND. Mutually exclusive with align_start (straighten wins). The align
  // drag and the shared cold-start lerp are untouched when the flag is off.
  bool   straighten_active = cfg.straighten_start;
  bool   straighten_armed  = false;   // one-way: set at the GO that arms the handover
  bool   straighten_funnel_wait = cfg.straighten_start;  // funnel disarmed at boot; arm at blend-complete
  double straighten_hold_q[kMaxNU] = {0};   // FIXED hold target latched at the first hold tick, so
  bool   straighten_hold_set = false;       // the PD RESISTS the gravity sag instead of tracking it
  if (straighten_active) align_active = false;
  bool   align_reported = false;      // one-shot drag-done banner (the GO gate can hold for long)
  double align_t0 = -1.0;             // WALL stamp of the first live lowstate
  double align_i[kMaxNU] = {0};       // anti-stiction integral, added to the commanded target
  double align_cmd[kMaxNU] = {0};     // LAST target emitted during the align (= align_q + integral).
                                      // The hand-over must re-latch the ramp to THIS, not to the
                                      // measured pose: the emitted command is deliberately
                                      // OVER-DRIVEN (it carries the gravity droop error plus the
                                      // whole anti-stiction integral), so re-latching to cur.q
                                      // would drop the target by err+I in a single 5 ms tick --
                                      // on the knee that is 0.43 rad = an 86 Nm torque step, i.e.
                                      // the legs going soft the instant the operator presses Enter.
  bool   align_cmd_set = false;
  double align_next_print = 0.0;      // progress ticker (names the joint that is refusing)
  if (align_active) {
    std::fprintf(stderr,
                 "[node] START-POSE ALIGN ON (%s): min-jerk drag over %.1fs (WALL), then hold "
                 "until the legs SETTLE (|dq|<0.05 rad/s) or %.1fs. %s The planner is NOT "
                 "consulted until hand-over. Residual >%.1f deg prints a NOTE.\n",
                 cfg.align_pose_set ? "explicit vector" : "model 'stand' keyframe",
                 cfg.align_sec, cfg.align_timeout,
                 cfg.align_wait ? "Then it PARKS on the stance and waits for you to press ENTER."
                                : "Then it hands over immediately (--noalign_wait).",
                 cfg.align_tol * 180.0 / M_PI);
    std::fprintf(stderr, "[node] align target (rad):");
    for (int i = 0; i < cfg.nu; i++) std::fprintf(stderr, " %.3f", align_q[i]);
    std::fprintf(stderr, "\n");
  }

  auto predict_forward = [&](mjData* s, double dt_ahead, const double* cmd_q) {
    int nsub = static_cast<int>(std::lround(dt_ahead / pred_dt));
    if (nsub < 1) return;
    mju_copy(pdat->qpos, s->qpos, nq);
    mju_copy(pdat->qvel, s->qvel, nv);
    if (g_model->na > 0) mju_zero(pdat->act, g_model->na);
    mju_zero(pdat->qfrc_applied, nv);
    if (gff != 0.0) {                                   // same constant tau_ff the twin's LowCmd carries
      mju_copy(gd->qpos, s->qpos, nq);
      mju_zero(gd->qvel, nv);
      mj_forward(g_model, gd);
      for (int i = 0; i < cfg.nu; i++)
        pdat->qfrc_applied[6 + moff + i] = gff * gd->qfrc_bias[6 + moff + i];
    }
    for (int k = 0; k < nsub; k++) {
      for (int i = 0; i < nact; i++) pdat->ctrl[i] = cmd_q[i];   // position-actuator target = in-flight cmd
      mj_step(g_model, pdat);
    }
    bool ok = true;                                     // discard a blown-up rollout, keep measured state
    for (int k = 0; ok && k < 7 + moff + cfg.nu; k++) if (!std::isfinite(pdat->qpos[k])) ok = false;
    for (int k = 0; ok && k < 6 + moff + cfg.nu; k++) if (!std::isfinite(pdat->qvel[k])) ok = false;
    if (!ok) return;
    for (int k = 0; k < 7 + moff + cfg.nu; k++) s->qpos[k] = pdat->qpos[k];
    for (int k = 0; k < 6 + moff + cfg.nu; k++) s->qvel[k] = pdat->qvel[k];
  };
  std::fprintf(stderr, "[node] latency-comp ON: predict-forward %s%s (extra %.1fms, cap %.0fms); "
               "plan + policy read at the landing time\n",
               lat_fixed > 0.0 ? "FIXED " : "AUTO measured", lat_fixed > 0.0 ? "" : "+extra",
               lat_extra * 1e3, lat_max * 1e3);

  std::vector<double> action(nu, 0.0);

  // ---- Title-5 baseline (B0) accumulators: tracking error, applied torque, balance ----
  // (policy phase only; printed as a summary on exit. The SAME node runs on the real
  //  robot -> per-row sim2real delta.)
  long m_ticks = 0;
  double m_err_sum[kMaxNU] = {0}, m_err_sq[kMaxNU] = {0}, m_err_max[kMaxNU] = {0};
  double m_tau_sum[kMaxNU] = {0}, m_tau_max[kMaxNU] = {0};
  double m_z_sum = 0, m_z_sq = 0, m_z_min = 1e9, m_tilt_sum = 0, m_tilt_max = 0;
  long m_sat_count[kMaxNU] = {0};   // ticks where |tau| exceeds the real H1-2 limit
  // ticks where the H2 clamp actually BIT (the emitted target was cut back because the
  // full torque budget kClampRatio*tau_estop was exhausted). A joint pinned here is OUT OF
  // AUTHORITY: the planner is asking for torque the node will not emit, and whatever the
  // plan wanted from that joint did not happen. This was the invisible failure in the real
  // trot runs -- the stance ankle sat at 100% and nothing reported it. 2026-07-11.
  long m_clamp_count[kMaxNU] = {0};
  bool m_sat_warned = false;

  // ---- control loop @ ctrl_hz (mirrors app.cc physics thread) ----
  const double twin_dt = cfg.twin_dt;
  uint32_t tick0 = 0; bool tick0_set = false;   // zero the twin sim-clock at the first state
  // single-stream base-velocity finite-diff + LPF state (see kVelLpfMs).
  const double vel_lpf_tau = kVelLpfMs * 1e-3;
  double fd_prev_p[3] = {0}, fd_vel[3] = {0}, fd_prev_t = 0.0; bool fd_have = false;
  std::fprintf(stderr,
               "[node] base linvel: SINGLE-STREAM finite-diff + %.0fms LPF (drops the two-stream phantom)\n",
               kVelLpfMs);
  bool stale_warned = false;   // H1 watchdog: warn once per stale episode
  auto t0 = std::chrono::steady_clock::now();
  auto next = t0;
  long ticks = 0;
  while (!g_exit.load()) {
    StateData cur;
    {
      std::lock_guard<std::mutex> lk(rs.mu);
      cur = rs.d;
    }
    auto t_tick = std::chrono::steady_clock::now();
    double wall = std::chrono::duration<double>(t_tick - t0).count();

    // H1: input-freshness watchdog. If either stream is stale, damp and bail --
    // never drive a target computed from a dead state stream. Threshold =
    // cfg.stale_sec (default kStaleSec 50ms, the REAL-robot value); heavyweight
    // sims whose lowstate publisher stalls on a shared sim lock (RoboCasa sensor
    // renders hold it 50-60ms) pass a looser --stale_sec.
    const double ls_age = std::chrono::duration<double>(t_tick - cur.ls_stamp).count();
    const double ss_age = std::chrono::duration<double>(t_tick - cur.ss_stamp).count();
    if (ls_age > cfg.stale_sec || ss_age > cfg.stale_sec) {
      if (!stale_warned) {
        std::fprintf(stderr, "[node] WARNING: state stale (lowstate %.0fms / sportstate %.0fms > %.0fms)"
                             " -> safe-hold (damping stop)\n",
                     ls_age * 1e3, ss_age * 1e3, cfg.stale_sec * 1e3);
        stale_warned = true;
      }
      emit_safe_hold();
      next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(ctrl_dt));
      std::this_thread::sleep_until(next);
      continue;
    }
    stale_warned = false;

    if (!tick0_set && cur.have_ls) { tick0 = cur.tick; tick0_set = true; }
    double twin_time = static_cast<double>(static_cast<int64_t>(cur.tick) -
                                           static_cast<int64_t>(tick0)) * twin_dt;

    // BRING-UP PHASE CLOCK (2026-07-03): the scripted phases (warmup -> ramp ->
    // hold -> policy blend, and the live-switch settle/blend below) progress on
    // PLANT time (lowstate tick * twin_dt), NOT wall. On any realtime plant
    // (real robot's 1 kHz tick, the twin) plant time advances at 1.0x wall, so
    // this is behavior-identical to the original wall clock; on a slower-than-
    // realtime plant (RoboCasa kitchen, ~0.2-0.35x) the choreography unfolds at
    // the speed the ROBOT experiences instead of compressing 3-5x into a jolt,
    // and a lowstate stall PAUSES the script instead of blindly advancing it.
    // Wall stays the clock for everything physical (publish cadence, latency
    // comp, watchdog, telemetry).
    const double phase_t = tick0_set ? twin_time : 0.0;
    bool warming = phase_t < warmup_sec;

    fill_state(cur);

    // ---- ANKLE ZERO AUTO-CALIB (--ankle_autocalib; observational, fail-safe) ----
    // Sample RAW encoders + RAW IMU during the scripted ramp HOLD (still, loaded,
    // planner muzzled), solve just BEFORE policy handover, and only then swap the
    // offsets. Skipped entirely under --straighten_start (different choreography;
    // its hold is operator-timed, not scripted). Statics: one call per process.
    if (cfg.ankle_autocalib && moff == 0 && !straighten_active && arm_init_set &&
        cur.have_ls) {
      static bool ac_done = false;
      static int ac_n = 0, ac_n1 = 0;
      static double ac_q[12] = {0}, ac_q1[12] = {0};   // full-window + first-half sums
      static double ac_quat[4] = {0}, ac_load = 0, ac_dq = 0;
      static double ac_ps = 0, ac_pss = 0, ac_rs = 0, ac_rss = 0;  // base pitch/roll stats (deg)
      // GRAVITY ANCHOR accumulators (2026-07-18): raw accel mean (the plumb-line),
      // |a| mean/var (ZUPT still-detector: |a| != g or noisy |a| = the robot is
      // being handled/accelerating -> window rejected), mean |gyro| (rotating).
      static double ac_acc[3] = {0}, ac_an = 0, ac_ann = 0, ac_gy = 0;
      static double ac_load_l = 0, ac_load_r = 0;  // per-leg |tau| sums (both feet must carry)
      // RETRY + BLEND state (2026-07-17): a squat start eats the whole ramp, so the
      // first window lands on a still-recovering robot -> gate REJECT. The old
      // one-shot then ran the WHOLE SESSION UNCALIBRATED (observed on real: 74s of
      // the old lean, REJECT at t=7.7). Now a gate-reject re-arms the window ~1s
      // later, up to kAcMaxAttempts tries; a solve that lands after policy handover
      // BLENDs the offsets in over kAcBlendSec (an instant 3.8-deg ankle-target step
      // mid-policy is a kp*0.066 ~ 5 Nm jolt; the ramp-hold application keeps its
      // instant path implicitly since blending during the scripted hold is harmless).
      constexpr int kAcMaxAttempts = 10;   // witness + confirm need >=2 good windows
      constexpr double kAcBlendSec = 1.0;
      static int ac_attempt = 0;
      static double ac_t0 = -1.0, ac_t1 = -1.0;
      static bool ac_blending = false;
      // blend slots 0-3 = ankle offsets, 4-5 = session IMU alignment (pitch, roll)
      static double ac_from[6], ac_to[6], ac_blend_t0 = 0.0;
      // gravity-vs-fused delta (deg) measured by the anchor, per window + witness
      static double ac_dg[2] = {0, 0};      static bool ac_have_dg = false;
      static double ac_wit_dg[2] = {0, 0};  static bool ac_wit_have_dg = false;
      // DRIFT WITNESS (2026-07-17 night, 3rd gate hole learned on real): a true
      // encoder zero is boot-CONSTANT, so every valid window must solve to the
      // same offsets. On a badly-yawed placement the feet CREEP out of flat --
      // solves drifted R pitch +1.2 -> +2.8 -> +5.2 across one run -- and the
      // applied window passed every stillness gate while being 1.6 deg wrong.
      // The feet are flattest right after the human places them: remember a
      // solve from the FIRST loaded, roughly-quiet window (looser gates) and
      // refuse any later apply that disagrees with it.
      static bool ac_wit = false;
      static double ac_wit_off[4] = {0};        // deg: poff_l, poff_r, roff_l, roff_r
      static double ac_wit_t = 0.0;
      if (ac_blending) {
        double f = std::fmin(1.0, (phase_t - ac_blend_t0) / kAcBlendSec);
        ankle_poff_l = ac_from[0] + f * (ac_to[0] - ac_from[0]);
        ankle_poff_r = ac_from[1] + f * (ac_to[1] - ac_from[1]);
        ankle_off_l = ac_from[2] + f * (ac_to[2] - ac_from[2]);
        ankle_off_r = ac_from[3] + f * (ac_to[3] - ac_from[3]);
        imu_align_p = ac_from[4] + f * (ac_to[4] - ac_from[4]);
        imu_align_r = ac_from[5] + f * (ac_to[5] - ac_from[5]);
        ankle_calib_on = true;
        if (f >= 1.0) ac_blending = false;
      }
      if (!ac_done) {
        if (ac_t0 < 0.0) {                                         // arm the first window
          ac_t0 = std::fmax(ramp_eff, warmup_sec) + 0.5;           // settle after the ramp motion
          // first window is ~2.2s (matching the retry windows), NOT the whole
          // hold: with the extended hold the confirm window must also fit
          // before handover (hold end = ramp_eff + ramp_hold_eff).
          ac_t1 = std::fmin(ac_t0 + 2.2, ramp_eff + ramp_hold_eff - 0.3);
          if (ac_t1 < ac_t0 + 1.0) ac_t1 = ac_t0 + 2.2;            // degenerate ramp: keep 2.2s
        }
        const double t0s = ac_t0;
        const double t1s = ac_t1;
        if (phase_t >= t0s && phase_t < t1s) {
          double load = 0, dqm = 0;
          for (int i = 0; i < 12 && i < cfg.nu; i++) {
            ac_q[i] += cur.q[i];
            load += std::fabs(cur.tau[i]);
            // PER-LEG load (2026-07-18): the summed gate passes with ONE foot
            // carrying everything -- the unloaded sole need not be flat and its
            // solve is garbage. Legs are motor 0-5 (L) / 6-11 (R).
            if (i < 6) ac_load_l += std::fabs(cur.tau[i]);
            else       ac_load_r += std::fabs(cur.tau[i]);
            dqm += std::fabs(cur.dq[i]);
          }
          if (phase_t < 0.5 * (t0s + t1s)) {           // temporal first half (agreement gate)
            for (int i = 0; i < 12 && i < cfg.nu; i++) ac_q1[i] += cur.q[i];
            ac_n1++;
          }
          ac_load += load;
          ac_dq += dqm / 12.0;
          for (int k = 0; k < 4; k++) ac_quat[k] += cur.quat[k];
          {  // gravity anchor: accumulate raw specific force + stillness stats
            double an = std::sqrt(cur.acc[0] * cur.acc[0] +
                                  cur.acc[1] * cur.acc[1] +
                                  cur.acc[2] * cur.acc[2]);
            for (int k = 0; k < 3; k++) ac_acc[k] += cur.acc[k];
            ac_an += an;
            ac_ann += an * an;
            ac_gy += std::sqrt(cur.gyro[0] * cur.gyro[0] +
                               cur.gyro[1] * cur.gyro[1] +
                               cur.gyro[2] * cur.gyro[2]);
          }
          double w = cur.quat[0], x = cur.quat[1], y = cur.quat[2], z = cur.quat[3];
          double bp = std::asin(std::fmax(-1.0, std::fmin(1.0, 2 * (w * y - z * x)))) * 180.0 / M_PI;
          double brl = std::atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)) * 180.0 / M_PI;
          ac_ps += bp; ac_pss += bp * bp; ac_rs += brl; ac_rss += brl * brl;
          ac_n++;
        } else if (phase_t >= t1s) {
          bool fatal = false;   // solver/model failures never retry; gate failures do
          // ---- WITNESS + CONFIRM (2026-07-17 redesign, 4 real runs of lessons) --
          // Stillness gates (SETTLED 0.03 / STABLE 0.5 / UPRIGHT / halves 0.5) were
          // the wrong instrument twice over: a still window can be the WRONG POSE
          // (mid-rise, harness-lean, crept feet -- solves drifted R +1.2 -> +2.8 ->
          // +5.2 across one run), and once the policy is live the robot is NEVER
          // still (run 4: the first window solved the CORRECT zeros at t=5, then
          // all 5 strict windows rejected and the session ran uncalibrated). The
          // invariant that identifies an encoder zero is CONSTANCY: it solves to
          // the same value in ANY two loaded windows. So: LOOSE per-window gates,
          // STRICT cross-window agreement -- the first plausible window is the
          // WITNESS (anchor; the feet are flattest right after the human places
          // them), a later window CONFIRMS if it agrees within kAcAgreeDeg, and
          // the mean of the two is applied. Creeping feet never re-agree with the
          // anchor -> give up -> manual flags.
          constexpr double kAcAgreeDeg = 0.8;
          const char* rej = nullptr;
          double q12[12] = {0}, quatm[4] = {1, 0, 0, 0};
          double dpitch = 0;                     // worst half-vs-full ankle delta (deg)
          double bq[4] = {1, 0, 0, 0};           // node-frame quat (IMU corr applied)
          if (ac_n < 100) rej = "window too short (<0.5s of samples)";
          if (!rej) {
            for (int i = 0; i < 12; i++) q12[i] = ac_q[i] / ac_n;
            for (int k = 0; k < 4; k++) quatm[k] = ac_quat[k] / ac_n;
            mju_normalize4(quatm);
            // node-frame quat: raw IMU x corr(imu_pitch) x corr(imu_roll), the SAME
            // correction fill_state applies (and the same one the snap tool applies).
            mju_copy4(bq, quatm);
            if (imu_pitch_off != 0.0) {
              double dq4[4], ax[3] = {0, 1, 0}, t4[4];
              mju_axisAngle2Quat(dq4, ax, imu_pitch_off);
              mju_mulQuat(t4, bq, dq4); mju_copy4(bq, t4);
            }
            if (imu_roll_off != 0.0) {
              double dq4[4], ax[3] = {1, 0, 0}, t4[4];
              mju_axisAngle2Quat(dq4, ax, imu_roll_off);
              mju_mulQuat(t4, bq, dq4); mju_copy4(bq, t4);
            }
            mju_normalize4(bq);
            if (ac_load / ac_n < 15.0) rej = "NOT LOADED (sum|tau| legs < 15 Nm -- feet not carrying weight)";
            // per-leg: each foot must carry real load or its sole isn't
            // necessarily flat (one-footed pass was a gate hole)
            if (!rej && (ac_load_l / ac_n < 5.0 || ac_load_r / ac_n < 5.0))
              rej = "ONE FOOT UNLOADED (per-leg sum|tau| < 5 Nm -- that sole may "
                    "not be flat; re-place the stance)";
          }
          // loose plausibility gates: slow motion keeps the q/quat means valid; the
          // agreement gate below is what separates a good pose from a wrong one
          if (!rej && ac_dq / ac_n > 0.06) rej = "MOVING (mean |dq| > 0.06 rad/s)";
          if (!rej) {
            double vp = ac_pss / ac_n - (ac_ps / ac_n) * (ac_ps / ac_n);
            double vr = ac_rss / ac_n - (ac_rs / ac_n) * (ac_rs / ac_n);
            if (std::sqrt(std::fmax(0.0, vp)) > 2.0 || std::sqrt(std::fmax(0.0, vr)) > 2.0)
              rej = "BASE MOVING (pitch/roll std > 2 deg)";
          }
          // ---- GRAVITY ANCHOR (2026-07-18, see anklecalib::GravityQuat) -------
          // ZUPT gates on the raw accel, then swap the solve's base-orientation
          // reference from the fused quat to the gravity plumb-line. If the accel
          // stream is EMPTY (twin bridges that don't populate it), fall back to
          // the fused quat -- twin/bench behavior unchanged.
          if (!rej && ac_n > 0 && cfg.ankle_autocalib_gravity) {
            double am[3] = {ac_acc[0] / ac_n, ac_acc[1] / ac_n, ac_acc[2] / ac_n};
            double an_mean = ac_an / ac_n;
            double an_std = std::sqrt(
                std::fmax(0.0, ac_ann / ac_n - an_mean * an_mean));
            double gy_mean = ac_gy / ac_n;
            if (an_mean < 3.0) {
              // < 3 covers BOTH an unpopulated stream (~0, twin bridges) and a
              // bridge reporting in g-units (~1.0) -- either way the m/s^2
              // plumb-line math doesn't apply; fall back rather than flip-flop
              // around a tight threshold. The logged value says which case.
              static bool warned = false;
              if (!warned) {
                warned = true;
                std::fprintf(stderr,
                             "[autocalib] gravity anchor: accel stream unusable "
                             "(|a| ~ %.2f, expected ~9.81 m/s^2) -> fused-quat "
                             "reference (empty stream? g-units bridge?)\n",
                             an_mean);
              }
            } else if (std::fabs(an_mean - 9.81) > 0.5) {
              rej = "EXTERNAL ACCEL (| |a| - g | > 0.5 m/s^2 -- robot moving or "
                    "being handled; gravity reference invalid)";
            } else if (an_std > 0.35) {
              rej = "ACCEL NOISY (std |a| > 0.35 m/s^2 -- vibrating/handled; "
                    "not a still window)";
            } else if (gy_mean > 0.15) {
              rej = "GYRO ACTIVE (mean |w| > 0.15 rad/s -- rotating; not a "
                    "still window)";
            } else {
              // still + gravity-clean: rebuild bq from the plumb-line (keeps the
              // fused yaw), then re-apply the SAME imu mount corrections.
              double bq_grav[4];
              anklecalib::GravityQuat(am, quatm, bq_grav);
              if (imu_pitch_off != 0.0) {
                double dq4[4], axm[3] = {0, 1, 0}, t4[4];
                mju_axisAngle2Quat(dq4, axm, imu_pitch_off);
                mju_mulQuat(t4, bq_grav, dq4); mju_copy4(bq_grav, t4);
              }
              if (imu_roll_off != 0.0) {
                double dq4[4], axm[3] = {1, 0, 0}, t4[4];
                mju_axisAngle2Quat(dq4, axm, imu_roll_off);
                mju_mulQuat(t4, bq_grav, dq4); mju_copy4(bq_grav, t4);
              }
              mju_normalize4(bq_grav);
              // log the anchor swap: how far the fused tilt was from gravity
              // (this delta IS the run-to-run variance source being removed)
              double gp = std::atan2(-am[0], std::hypot(am[1], am[2])) * 180.0 / M_PI;
              double gr = std::atan2(am[1], am[2]) * 180.0 / M_PI;
              std::fprintf(stderr,
                           "[autocalib] gravity anchor: base P%+.2f R%+.2f deg "
                           "(accel, |a|=%.2f+-%.2f) vs fused P%+.2f R%+.2f -> "
                           "d P%+.2f R%+.2f deg (using GRAVITY)\n",
                           gp, gr, an_mean, an_std,
                           ac_ps / ac_n, ac_rs / ac_n,
                           gp - ac_ps / ac_n, gr - ac_rs / ac_n);
              ac_dg[0] = gp - ac_ps / ac_n;   // deg: what the PLANNER's frame is
              ac_dg[1] = gr - ac_rs / ac_n;   // missing vs the plumb-line
              ac_have_dg = true;
              mju_copy4(bq, bq_grav);
            }
          }
          if (!rej && ac_n1 > 20) {              // halves agreement on the measured ankles
            const int aidx[4] = {4, 5, 10, 11};
            for (int k = 0; k < 4; k++) {
              double d1 = (ac_q1[aidx[k]] / ac_n1 - q12[aidx[k]]) * 180.0 / M_PI;
              dpitch = std::fmax(dpitch, std::fabs(d1));
            }
            if (dpitch > 0.8) rej = "HALVES DISAGREE (>0.8 deg -- ankles moved inside the window)";
          }
          anklecalib::Result r;
          if (!rej) {
            // solver-side failures below are FATAL (missing keyframe/bodies -- no
            // amount of retrying fixes a model), gate failures above are transient
            r = anklecalib::Solve(g_model, q12, bq);   // ~10-20ms per window (held
                                                       // targets; a 2-4 tick stall is
                                                       // absorbed by the PD on last cmd)
            if (!r.ok) { rej = r.why; fatal = true; }
          }
          if (!rej) {
            double mx = std::fmax(std::fmax(std::fabs(r.poff_l), std::fabs(r.poff_r)),
                                  std::fmax(std::fabs(r.roff_l), std::fabs(r.roff_r)));
            if (mx > 8.0) rej = "|offset| > 8 deg cap -- check feet were FLAT + LOADED";
          }
          bool applied = false;
          if (!rej) {
            if (!ac_wit) {
              ac_wit = true;
              ac_wit_off[0] = r.poff_l; ac_wit_off[1] = r.poff_r;
              ac_wit_off[2] = r.roff_l; ac_wit_off[3] = r.roff_r;
              ac_wit_dg[0] = ac_dg[0]; ac_wit_dg[1] = ac_dg[1];
              ac_wit_have_dg = ac_have_dg;
              ac_wit_t = t0s;
              std::fprintf(stderr,
                           "[autocalib] WITNESS @%.1f-%.1fs: pitch L%+.2f R%+.2f roll "
                           "L%+.2f R%+.2f (load %.0f Nm, |dq| %.4f, halves d%.2f) -- "
                           "awaiting a confirming window within %.1f deg\n",
                           t0s, t1s, r.poff_l, r.poff_r, r.roff_l, r.roff_r,
                           ac_load / ac_n, ac_dq / ac_n, dpitch, kAcAgreeDeg);
            } else {
              double dd = std::fmax(
                  std::fmax(std::fabs(r.poff_l - ac_wit_off[0]),
                            std::fabs(r.poff_r - ac_wit_off[1])),
                  std::fmax(std::fabs(r.roff_l - ac_wit_off[2]),
                            std::fabs(r.roff_r - ac_wit_off[3])));
              if (dd <= kAcAgreeDeg) {
                applied = true;
                ac_done = true;
                // BLEND the mean of witness+confirm in (jolt-free post-handover)
                ac_from[0] = ankle_poff_l; ac_from[1] = ankle_poff_r;
                ac_from[2] = ankle_off_l; ac_from[3] = ankle_off_r;
                ac_to[0] = 0.5 * (r.poff_l + ac_wit_off[0]) * M_PI / 180.0;
                ac_to[1] = 0.5 * (r.poff_r + ac_wit_off[1]) * M_PI / 180.0;
                ac_to[2] = 0.5 * (r.roff_l + ac_wit_off[2]) * M_PI / 180.0;
                ac_to[3] = 0.5 * (r.roff_r + ac_wit_off[3]) * M_PI / 180.0;
                // SESSION IMU ALIGNMENT: also hand the planner the measured
                // gravity-vs-fused delta (mean over the windows that have one).
                ac_from[4] = imu_align_p; ac_from[5] = imu_align_r;
                ac_to[4] = imu_align_p; ac_to[5] = imu_align_r;   // default: unchanged
                if (cfg.ankle_autocalib_gravity && cfg.ankle_autocalib_imu_align) {
                  double dp = 0, dr = 0; int nd = 0;
                  if (ac_wit_have_dg) { dp += ac_wit_dg[0]; dr += ac_wit_dg[1]; nd++; }
                  if (ac_have_dg)     { dp += ac_dg[0];     dr += ac_dg[1];     nd++; }
                  if (nd > 0) {
                    dp /= nd; dr /= nd;
                    if (std::fabs(dp) <= 4.0 && std::fabs(dr) <= 4.0) {
                      ac_to[4] = dp * M_PI / 180.0;
                      ac_to[5] = dr * M_PI / 180.0;
                      std::fprintf(stderr,
                                   "[autocalib] SESSION IMU ALIGN: planner frame "
                                   "corrected by the measured gravity-vs-fused delta "
                                   "P%+.2f R%+.2f deg (on top of the static "
                                   "%+.2f/%+.2f flags; blended over %.1fs; "
                                   "--ac_imu_align=0 disables)\n",
                                   dp, dr, cfg.imu_pitch_offset_deg,
                                   cfg.imu_roll_offset_deg, kAcBlendSec);
                    } else {
                      std::fprintf(stderr,
                                   "[autocalib] SESSION IMU ALIGN skipped: measured "
                                   "delta P%+.2f R%+.2f deg exceeds the 4 deg cap "
                                   "(check IMU/accel health)\n", dp, dr);
                    }
                  }
                }
                ac_blend_t0 = phase_t;
                ac_blending = true;
                std::fprintf(stderr,
                             "[autocalib] CONFIRMED @%.1f-%.1fs agrees with witness @%.1fs "
                             "(max d%.2f deg) -> APPLIED mean (off = encoder - true, deg): "
                             "pitch L%+.2f R%+.2f  roll L%+.2f R%+.2f  (blended over %.1fs) "
                             "-- REPLACES the manual flags\n",
                             t0s, t1s, ac_wit_t, dd,
                             0.5 * (r.poff_l + ac_wit_off[0]), 0.5 * (r.poff_r + ac_wit_off[1]),
                             0.5 * (r.roff_l + ac_wit_off[2]), 0.5 * (r.roff_r + ac_wit_off[3]),
                             kAcBlendSec);
              } else {
                rej = "DISAGREES with the witness (feet CREEPING or witness was a bad "
                      "pose; a true encoder zero is constant)";
                std::fprintf(stderr,
                             "[autocalib]   candidate @%.1f-%.1fs pitch L%+.2f R%+.2f roll "
                             "L%+.2f R%+.2f vs witness max d%.2f > %.1f deg\n",
                             t0s, t1s, r.poff_l, r.poff_r, r.roff_l, r.roff_r, dd, kAcAgreeDeg);
              }
            }
          }
          if (!applied && !ac_done) {
            ac_attempt++;
            bool more = !fatal && ac_attempt < kAcMaxAttempts;
            if (more) {
              // re-arm a fresh window ~1s out with clean accumulators (also the
              // path a freshly-recorded witness takes to get its confirm window)
              ac_n = 0; ac_n1 = 0; ac_load = 0; ac_dq = 0;
              ac_ps = 0; ac_pss = 0; ac_rs = 0; ac_rss = 0;
              ac_an = 0; ac_ann = 0; ac_gy = 0;
              ac_load_l = 0; ac_load_r = 0;
              ac_have_dg = false;
              for (int k = 0; k < 3; k++) ac_acc[k] = 0;
              for (int k = 0; k < 12; k++) { ac_q[k] = 0; ac_q1[k] = 0; }
              for (int k = 0; k < 4; k++) ac_quat[k] = 0;
              ac_t0 = phase_t + 1.0;
              ac_t1 = ac_t0 + 2.2;
              if (rej)
                std::fprintf(stderr,
                             "[autocalib] window %d/%d REJECT: %s -> next window at t=%.1fs "
                             "(offsets unchanged meanwhile)\n",
                             ac_attempt, kAcMaxAttempts, rej, ac_t0);
            } else {
              ac_done = true;
              std::fprintf(stderr,
                           "[autocalib] GIVING UP after window %d (%s): applying NOTHING, "
                           "keeping the manual flags (pitch L%+.2f R%+.2f roll L%+.2f R%+.2f)"
                           "%s\n",
                           ac_attempt, rej ? rej : "no confirming window",
                           ankle_poff_l * 180.0 / M_PI, ankle_poff_r * 180.0 / M_PI,
                           ankle_off_l * 180.0 / M_PI, ankle_off_r * 180.0 / M_PI,
                           ac_wit ? " -- a witness existed but was never confirmed: "
                                    "feet likely creeping, re-place the stance" : "");
            }
          }
        }
      }
    }

    if (!arm_init_set && cur.have_ls) {                 // latch the measured power-on pose for the ramp
      for (int i = 0; i < cfg.nu; i++) arm_q_init[i] = cur.q[i];
      arm_init_set = true;
      // scale the ramp (and its policy-holdout) by how far off-home we latched: the
      // holdout is only survivable while a harness supports the robot -- from a HOME
      // start a full 5 s without active balance topples it (bench: fell at 2.7 s).
      // >=0.5 rad off-home -> full ramp; at home -> ~0 -> policy right after warmup
      // (the proven home-start behavior).
      double d0 = 0.0;
      for (int i = 0; i < cfg.nu; i++) d0 = std::fmax(d0, std::fabs(arm_q_init[i] - home_q[i]));
      ramp_eff = start_ramp_sec * std::fmin(1.0, d0 / 0.5);
      // WAYPOINT ramp engages only on genuinely off-home starts (d0 >= 0.5 rad, the
      // harness-supported regime): the path is longer than the direct lerp, so each
      // segment gets half the stock ramp time. Near-home starts keep the fast direct
      // path (a home-started robot has nothing to swing clear of).
      bringup_wps_live = (n_bringup_wp > 0 && d0 >= 0.5 && !straighten_active);
      if (bringup_wps_live) {
        ramp_eff = kStartRampSec * 0.5 * (n_bringup_wp + 1);
        std::fprintf(stderr, "[node] bring-up WAYPOINT path ACTIVE: %d station(s), "
                             "%.1fs total\n", n_bringup_wp, ramp_eff);
      }
      if (straighten_active) ramp_eff = 0.0;   // straighten-start: the planner drives the rise;
                                               // the scripted bring-up ramp/muzzle is bypassed
                                               // (hold slump -> ENTER -> handover -> full authority),
                                               // so the post-handover path lands on full authority.
      std::fprintf(stderr, "[node] bring-up ramp: latched pose is %.2f rad from home -> "
                           "effective ramp %.1fs\n", d0, ramp_eff);
      // PHASE A: stamp the align clock on the first live state (WALL, not plant time).
      if (align_active) { align_t0 = wall; }
    }
    // SINGLE-STREAM base linvel: replace the cross-stream (site_v - gyro x r) value with a finite
    // diff of the reconstructed pelvis position over the sim-clock + LPF. Updates only when the twin
    // tick advances (a genuinely new state); holds between. Set before the latency rollout so the
    // prediction starts from the de-noised velocity.
    if (cur.have_ls && cur.have_ss) {
      if (fd_have && twin_time > fd_prev_t + 1e-9) {       // new state -> differentiate over real dt
        double dt = twin_time - fd_prev_t;
        double a = dt / (vel_lpf_tau + dt);                // 1st-order LPF, step-size independent
        for (int k = 0; k < 3; k++) {
          double raw = (sd->qpos[k] - fd_prev_p[k]) / dt;
          fd_vel[k] += a * (raw - fd_vel[k]);
          fd_prev_p[k] = sd->qpos[k];
        }
        fd_prev_t = twin_time;
      } else if (!fd_have) {                                // seed at rest (v=0) on the first state
        for (int k = 0; k < 3; k++) { fd_prev_p[k] = sd->qpos[k]; fd_vel[k] = 0.0; }
        fd_prev_t = twin_time; fd_have = true;
      }
      for (int k = 0; k < 3; k++) sd->qvel[k] = fd_vel[k];
    }
    // snapshot the MEASURED base pose NOW -- latency prediction overwrites sd->qpos below, but B0
    // and the status line must report the real (measured) height/tilt/knees, not the predicted ones.
    double meas_base_z = sd->qpos[2];
    double meas_R8, meas_lean_fwd = 0.0, meas_lean_lat = 0.0;
    { double Rm[9]; mju_quat2Mat(Rm, sd->qpos + 3); meas_R8 = Rm[8];
      // SIGNED torso lean from the measured base quat (the |tilt| above hides the
      // direction -> can't tell a forward CREEP from a backward one). The torso
      // up-axis in world = column 2 of R = (Rm[2], Rm[5], Rm[8]); its fore-aft tip
      // is +x (robot-forward), its lateral tip is +y (robot-left).
      meas_lean_fwd = std::atan2(Rm[2], Rm[8]) * 57.29577951308232;  // + = FORWARD
      meas_lean_lat = std::atan2(Rm[5], Rm[8]) * 57.29577951308232;  // + = LEFT
    }
    double meas_kneeL = sd->qpos[10], meas_kneeR = sd->qpos[16];
    double tau_now[kMaxNU] = {0};   // per-tick applied torque, for the status-line headroom readout

    // LATENCY COMPENSATION: predict the state forward by Δ and plan from THERE, so the action we
    // emit is in-phase when it lands on the plant. dlt=0 during warmup -> identical to before.
    double dlt = 0.0;
    if (!warming) {
      // Scale the WALL-clock horizon to SIM time by the measured real-time factor
      // (cfg.latency_rtf): predict_forward rolls dlt forward as SIM steps, so on a
      // below-realtime plant an unscaled wall dlt over-leads the plant by 1/RTF.
      // 1.0 = identity (real/twin run RTF~1); a slow sim passes its measured RTF.
      dlt = ((lat_fixed > 0.0) ? lat_fixed : (ewma_comp + lat_extra)) * cfg.latency_rtf;
      if (dlt > lat_max) dlt = lat_max;
      predict_forward(sd, dlt, last_cmd_q);
    }
    // The planner is timed by the TWIN's sim-clock (rt/lowstate tick * twin_dt), the settled
    // deploy mode: wall-clock races ahead of a slower-than-realtime plant and mis-times the
    // policy. + dlt reads the policy at the LANDING time, consistent with the predicted state.
    sd->time = twin_time + dlt;
                       // FIX 2026-06-02: advance the planner clock. It was NEVER set (stuck at 0),
                       // so ActionFromPolicy(state.time) read the START of the plan every tick ->
                       // the node executed only the first instant of each trajectory. Static holds
                       // (stand) looked fine; DYNAMIC motions (crouch, arms_overhead) never played
                       // out. app.cc advances d->time the same way so the policy is read forward.
    mj_forward(g_model, sd);
    // THE MISSING CALL (found 2026-06-12 via residual forensics): the GUI's physics loop
    // runs Task::Transition every step -- it is what LOADS the strategy JSON (posture
    // keyframe + per-phase weights) and advances multi-phase strategies. This node NEVER
    // called it, so every deployment planned toward keyframe 0 (straight-knee 'home',
    // z=1.028): the hand-off knee snap, the 1.028 base parking, and the inert keyframe
    // edits were all this one absence.
    g_agent.ActiveTask()->Transition(g_model, sd);
    g_agent.SetState(sd);

    // policy: target joint positions (rad). Held at measured q during warmup.
    if (!warming) {
      g_agent.ActivePlanner().ActionFromPolicy(action.data(), g_agent.state.state().data(),
                                             g_agent.state.time());
    }

    // gravity feedforward: tau = gff * qfrc_bias evaluated at qvel = 0.
    double tau[kMaxNU] = {0};
    if (gff != 0.0) {
      mju_copy(gd->qpos, sd->qpos, nq);
      mju_zero(gd->qvel, nv);
      mj_forward(g_model, gd);
      for (int i = 0; i < cfg.nu; i++) tau[i] = gff * gd->qfrc_bias[6 + moff + i];
    }

    // ---- live strategy-switch blend: on a pending switch, snapshot the CURRENT measured pose
    //      as the blend start + stamp the switch time, so the target eases from where the robot
    //      IS into the new policy target (no snap). Serves stdin switches (strategy 18's internal
    //      stand<->crouch cycling is handled by the task phase machine, not here).
    if (g_switch_pending.exchange(false)) {
      for (int i = 0; i < cfg.nu; i++) g_switch_from[i] = cur.q[i];
      g_switch_wall.store(phase_t);   // switch settle/blend run on the plant clock too
    }

    double tgt_q[kMaxNU];

    // ================= PHASE A: START-POSE ALIGN (2026-07-13) =================
    // Runs BEFORE the planner is ever consulted (action[] is not read here). Drags the legs
    // from whatever geometry the robot powered on with -- twisted, fore-aft, one knee folded --
    // into a known stance, so MJPC always inherits the SAME basin instead of a lottery.
    // Everything here is WALL-clocked: the bring-up choreography below runs on plant time
    // (tick*twin_dt), which with --twin_dt 0.001 on this 200 Hz plant dilates 5x. The align
    // must not inherit that dilation.
    bool in_align = false;
    if (align_active && align_t0 >= 0.0) {
      constexpr double kAlignSettleDq = 0.05;   // rad/s: "the legs have stopped moving"
      const double at = wall - align_t0;
      double err = 0.0, vmax = 0.0;
      for (int i = 0; i < cfg.nu; i++) {
        err  = std::fmax(err,  std::fabs(cur.q[i] - align_q[i]));
        vmax = std::fmax(vmax, std::fabs(cur.dq[i]));
      }
      // The exit gate is SETTLED, not "reached". The robot is LOAD-BEARING during the align, so
      // a knee commanded to 0.37 comes to rest nearer 0.55 under the deploy PD's gravity droop
      // (measured in the good run: cmd 0.35 vs q 0.568 -- a 13 deg standing sag, and that pose
      // stands perfectly well). Gating on |q - target| would therefore NEVER fire and the align
      // would always burn its full timeout. The legs having stopped moving is the physically
      // reachable signal that the drag is done. align_tol is a QUALITY WARNING, not a blocker.
      // Exit when the legs have either ARRIVED (every joint inside align_tol) or STALLED (they
      // have stopped moving and the anti-stiction integral can push no further). Distinguishing
      // the two matters: "reached" is a clean start; "stalled" means a joint is jammed and the
      // operator should look before handing the robot to the planner.
      const bool arrived = (at >= cfg.align_sec) && (err < cfg.align_tol);
      const bool stalled = (at >= cfg.align_sec) && (vmax < kAlignSettleDq);
      const bool drag_done = arrived || stalled || at >= cfg.align_timeout;

      // one-shot report the instant the drag finishes (the GO gate below may hold us here for
      // minutes afterwards, and the operator must not have to scroll for this line).
      if (drag_done && !align_reported) {
        align_reported = true;
        int worst = 0;
        for (int i = 0; i < cfg.nu; i++)
          if (std::fabs(cur.q[i] - align_q[i]) > std::fabs(cur.q[worst] - align_q[worst]))
            worst = i;
        std::fprintf(stderr,
                     "[node] START-POSE ALIGN %s after %.1fs -- worst joint %s err %.1f deg "
                     "(push %+.3f rad), |dq|max %.3f rad/s\n",
                     arrived ? "REACHED" :
                     stalled ? "STALLED (legs stopped short -- check for a jam)" : "TIMED OUT",
                     at,
                     cfg.joint_names ? cfg.joint_names[worst] : "?",
                     err * 180.0 / M_PI, align_i[worst], vmax);
        if (err > cfg.align_tol)
          std::fprintf(stderr,
                       "[node]   NOTE residual %.1f deg > --align_tol %.1f deg. Expected if the "
                       "robot is bearing its own weight (gravity droop); a problem only if a "
                       "SINGLE joint is far off (a leg jammed on the floor).\n",
                       err * 180.0 / M_PI, cfg.align_tol * 180.0 / M_PI);
        if (cfg.align_wait) {
          g_align_waiting.store(true);   // from here a bare Enter on stdin means GO
          std::fprintf(stderr,
              "[node] ==============================================================\n"
              "[node]  HOLDING the start pose. The planner is NOT driving the robot.\n"
              "[node]  Settle it / let go, then press ENTER to hand over to MJPC.\n"
              "[node]  (q + Enter quits without ever engaging the planner.)\n"
              "[node] ==============================================================\n");
        }
      }

      // GO gate: park on the aligned stance until the operator says the robot is settled. The
      // legs stay stiff and we keep publishing at 200 Hz throughout, so the safety layer never
      // sees a stale command and the pose does not drift while we wait.
      const bool go = !cfg.align_wait || g_align_go.load();
      if (drag_done && go) {
        g_align_waiting.store(false);
        align_active = false;
        // HAND-OVER = a live strategy switch FROM the aligned pose, NOT a re-run of the cold-start
        // choreography. Latch the align's LAST command (= align_q + anti-stiction integral, an
        // OVER-DRIVEN stiff hold) as the from-pose: re-latching to the measured q would drop the
        // command by (droop + integral) in one tick and the legs go SOFT (observed real 2026-07-13).
        // The OLD path then reset the phase clock and re-ran warmup -> ramp -> 3 s HOLD at home_q ->
        // blend, which walks the legs OFF the aligned stance toward home_q and holds them there
        // statically (no active balance) -- and under the dilated plant clock that hold is ~5x
        // longer in wall time, so the robot droops and "loses the point of align_start". INSTEAD:
        // hold the aligned command briefly (SETTLE -- the planner has been converging on this exact
        // pose throughout the align hold, so it is already warm) then ease into the LIVE policy over
        // kPolicyBlendSec -- the same settle->blend a mid-run strategy switch uses. Runs on WALL time
        // (see the target loop) so MJPC engages promptly on the real robot regardless of the
        // twin_dt/tick dilation. tick0 is deliberately NOT reset (keeps the base-vel finite-diff and
        // latency comp continuous through the hand-over).
        for (int i = 0; i < cfg.nu; i++)
          arm_q_init[i] = align_cmd_set ? align_cmd[i] : cur.q[i];
        handover_active = true;
        handover_t0 = wall;   // WALL clock: dilation-immune, snappy on the real robot
        double d0 = 0.0;
        for (int i = 0; i < cfg.nu; i++)
          d0 = std::fmax(d0, std::fabs(arm_q_init[i] - home_q[i]));
        std::fprintf(stderr,
                     "[node] HAND-OVER from the aligned pose (%.3f rad from home, C0-continuous): "
                     "SETTLE %.1fs -> POLICY BLEND %.1fs -> full MJPC authority ~%.1fs from now "
                     "(WALL seconds). No cold-start re-run -> the legs stay on the aligned stance.\n",
                     d0, kSwitchSettleSec, kPolicyBlendSec,
                     kSwitchSettleSec + kPolicyBlendSec);
      } else {
        in_align = true;
      }
    }

    // ============= PHASE A': STRAIGHTEN-START hold + ENTER gate (2026-07-15) =============
    // --straighten_start reuses align_start's ENTER->handover with NO drag: hold the measured
    // (slumped) pose until the operator presses ENTER, then arm the SAME handover_active SETTLE->
    // BLEND (below) FROM the slump so the planner (straighten strat) drives the recovery. ramp_eff
    // was zeroed at latch for this mode, so once the handover finishes the normal branch lands on
    // full planner authority (no scripted muzzle). Guarded by straighten_active -> flag off is
    // byte-identical. straighten_armed is one-way: once armed, this block never re-engages.
    bool straighten_hold = false;
    if (straighten_active && !straighten_armed && arm_init_set) {
      if (cfg.align_wait && !g_align_waiting.load() && !g_align_go.load()) {
        g_align_waiting.store(true);   // from here a bare Enter on stdin means GO
        std::fprintf(stderr,
            "[node] =============================================================\n"
            "[node]  STRAIGHTEN-START: holding the measured pose. Planner NOT driving.\n"
            "[node]  Set/confirm the harness, then press ENTER to hand to the planner.\n"
            "[node]  (q + Enter quits without ever engaging the planner.)\n"
            "[node] =============================================================\n");
      }
      if (!cfg.align_wait || g_align_go.load()) {
        g_align_waiting.store(false);
        straighten_armed = true;
        handover_active = true;
        handover_t0 = wall;   // WALL clock (dilation-immune), like the align hand-over
        for (int i = 0; i < cfg.nu; i++) arm_q_init[i] = cur.q[i];   // hand over FROM the slump
        std::fprintf(stderr,
            "[node] STRAIGHTEN-START HAND-OVER from the measured slump: SETTLE %.1fs -> POLICY "
            "BLEND %.1fs -> full planner authority (WALL seconds).\n",
            kSwitchSettleSec, kPolicyBlendSec);
      } else {
        straighten_hold = true;   // still waiting for ENTER: hold the slump this tick
      }
    }

    if (straighten_hold) {
      // Latch the pose ONCE and hold that FIXED target, so the motor PD resists the gravity sag.
      // (Tracking cur.q every tick = zero position error = no restoring torque -> the robot slowly
      //  collapses into a deeper toes-up crouch during the wait -- observed on real 2026-07-15.)
      if (!straighten_hold_set) {
        for (int i = 0; i < cfg.nu; i++) straighten_hold_q[i] = cur.q[i];
        straighten_hold_set = true;
      }
      for (int i = 0; i < cfg.nu; i++) tgt_q[i] = straighten_hold_q[i];   // FIXED -> resists sag
    } else if (in_align) {
      // min-jerk 10s^3 - 15s^4 + 6s^5: zero velocity AND zero acceleration at BOTH ends, so a
      // twisted leg is dragged into place, never yanked. At s=0 the target is exactly the
      // measured pose, so there is no step at t=0 either. Gravity feedforward (tau) and the H2
      // torque clamp below still apply, so a leg that jams on the floor cannot be forced.
      const double at = wall - align_t0;
      const double s = (cfg.align_sec > 1e-6)
                           ? std::fmin(1.0, std::fmax(0.0, at / cfg.align_sec))
                           : 1.0;
      const double mjk = s * s * s * (10.0 + s * (-15.0 + 6.0 * s));
      const double dt_i = 1.0 / kCtrlHz;
      for (int i = 0; i < cfg.nu; i++) {
        const double ref = (1.0 - mjk) * arm_q_init[i] + mjk * align_q[i];
        // ANTI-STICTION PUSH (see Cfg::align_ki). Once the reference has ARRIVED (mjk==1, so we
        // never fight the drag profile), integrate the residual into the commanded target: the
        // motor PD then applies kp*(tgt - q), which keeps GROWING until the joint breaks its
        // static friction. Deadbanded by align_tol so an arrived joint stops being pushed, and
        // hard-limited by align_i_max so a genuinely jammed joint parks a bounded target rather
        // than winding up forever. The H2 torque clamp below still bounds what actually reaches
        // the motor to 0.9 x the safety estop -- this cannot exceed the safety envelope.
        if (mjk >= 1.0 && cfg.align_ki > 0.0) {
          const double e = align_q[i] - cur.q[i];
          if (std::fabs(e) > cfg.align_tol) {
            align_i[i] += cfg.align_ki * e * dt_i;
            align_i[i] = std::fmin(cfg.align_i_max,
                                   std::fmax(-cfg.align_i_max, align_i[i]));
          }
        }
        tgt_q[i] = ref + align_i[i];
      }
      // remember what we actually emitted: the GO hand-over re-latches the bring-up ramp to THIS
      // (see align_cmd), so the command never steps when the operator presses Enter.
      for (int i = 0; i < cfg.nu; i++) align_cmd[i] = tgt_q[i];
      align_cmd_set = true;
      // progress ticker: name the joint that is refusing to arrive, and how hard we are pushing
      // it. If the same joint sits at the windup limit for seconds, it is mechanically stuck.
      if (wall >= align_next_print) {
        align_next_print = wall + 2.0;
        int worst = 0;
        for (int i = 0; i < cfg.nu; i++)
          if (std::fabs(cur.q[i] - align_q[i]) > std::fabs(cur.q[worst] - align_q[worst]))
            worst = i;
        std::fprintf(stderr,
                     "[node]   align t=%4.1fs  worst %s err %5.1f deg  push %+.3f rad"
                     " (%.0f%% of windup limit)\n",
                     at, cfg.joint_names ? cfg.joint_names[worst] : "?",
                     std::fabs(cur.q[worst] - align_q[worst]) * 180.0 / M_PI, align_i[worst],
                     cfg.align_i_max > 0.0
                         ? 100.0 * std::fabs(align_i[worst]) / cfg.align_i_max : 0.0);
      }
    } else {
    // per-joint commanded position target with the BRING-UP ramp: blend from the measured
    // power-on pose to the home/policy target on ALL joints (kills the warmup->policy SNAP
    // from off-home starts).
    for (int i = 0; i < cfg.nu; i++) {
      double ramp_dur = ramp_eff;
      double base;
      if (handover_active) {
        // ALIGN HAND-OVER (WALL clock): hold the aligned command (SETTLE) while the already-
        // converged planner stays warm on this pose, then ease into the live policy over
        // kPolicyBlendSec -- the same ramp a live strategy switch uses, but FROM the aligned pose
        // so the legs never walk back toward home_q and droop (the old warmup->ramp->hold sag).
        // Wall time (not the dilated plant clock) keeps this a snappy few seconds on the real robot.
        const double hd = wall - handover_t0;
        if (i >= nact) {
          base = cur.q[i];                                         // non-actuated (arm) rows: hold
        } else if (hd < kSwitchSettleSec) {
          base = arm_q_init[i];                                   // SETTLE: stiff hold of aligned cmd
        } else if (hd < kSwitchSettleSec + kPolicyBlendSec) {
          const double bb = (hd - kSwitchSettleSec) / kPolicyBlendSec;
          base = (1.0 - bb) * arm_q_init[i] + bb * action[i];     // POLICY BLEND aligned pose -> MJPC
        } else {
          base = action[i];                                       // full MJPC authority
        }
      } else if (ramp_dur > 0.0 && arm_init_set) {
        double aa = std::fmin(1.0, std::fmax(0.0, phase_t / ramp_dur));
        // SCRIPTED rise: target the stance for the whole ramp (policy steering a half-
        // risen, moving robot topples it -- bench-proven), then HOLD the stance scripted
        // for kRampHoldSec more so CEM converges around the STATIC operating pose
        // before getting authority (kills the hand-off "first tug" toward the old basin).
        // hold end = ramp + the (possibly autocalib-extended) scripted hold.
        const double t_ho = ramp_dur + ramp_hold_eff;
        const double pblend = kPolicyBlendSec;
        double tgt;
        if (aa < 1.0 || warming || i >= nact || phase_t < t_ho) {
          tgt = home_q[i];                       // rising / warmup / non-policy joint / scripted hold
        } else if (pblend > 0.0 && phase_t < t_ho + pblend) {
          // POLICY-BLEND: ease the scripted stance -> the LIVE policy target over pblend so a
          // cold-started non-stand task descends smoothly instead of snapping at the handoff.
          double bb = (phase_t - t_ho) / pblend;  // 0..1
          tgt = (1.0 - bb) * home_q[i] + bb * action[i];
        } else {
          // full policy authority -- UNLESS a LIVE switch just armed a blend. Mirror the proven
          // cold-start path (rise -> HOLD while CEM converges -> blend): first hold the pre-switch
          // pose for kSwitchSettleSec so the planner re-converges on the NEW strategy, THEN ease
          // into its target over g_switch_blend. Without the settle the robot descends mid-replan
          // and collapses past the target (the Squatter "snap": cmd blended to 0.63 but q hit 0.99).
          double sw_dt = phase_t - g_switch_wall.load();
          double settle = kSwitchSettleSec;
          double sbl = g_switch_blend.load();
          if (sw_dt >= 0.0 && sw_dt < settle) {
            tgt = g_switch_from[i];              // SETTLE: hold pose, let the planner converge
          } else if (sbl > 0.0 && sw_dt >= settle && sw_dt < settle + sbl) {
            double sb = (sw_dt - settle) / sbl;  // 0..1
            tgt = (1.0 - sb) * g_switch_from[i] + sb * action[i];
          } else {
            tgt = action[i];                     // full policy authority
          }
        }
        // WAYPOINT rise: while still climbing (aa<1) route through the bring-up
      // stations instead of the straight blend. tgt == home_q for the whole
      // rise (see above), so the path ends exactly where the old lerp did and
      // every post-rise branch (hold/blend/switch) is untouched.
      if (bringup_wps_live && aa < 1.0)
        base = bringup_path_at(aa, home_q, i);
      else
        base = (1.0 - aa) * arm_q_init[i] + aa * tgt;
      } else {
        base = (warming || i >= nact) ? cur.q[i] : action[i];
      }
      tgt_q[i] = base;
    }
    // hand-over done: drop back to the normal full-authority path (which also handles future
    // live strategy switches). C0-continuous -- both branches were already emitting `action`.
    if (handover_active && (wall - handover_t0) >= kSwitchSettleSec + kPolicyBlendSec) {
      handover_active = false;
      // --straighten_start: FULL planner authority just landed -- arm the
      // strategy's live-seed funnel NOW. Arming at GO would burn the glide
      // during SETTLE+BLEND (the same bug at smaller scale); arming here means
      // the target glides from the pose the robot actually holds at this
      // moment, exactly while the planner can act on it.
      if (straighten_armed && straighten_funnel_wait) {
        straighten_funnel_wait = false;
        g_agent.SetParamByName("residual_Funnel Arm", 1.0);
        std::fprintf(stderr,
                     "[node] straighten funnel ARMED (full planner authority) -> the "
                     "strategy's target glide starts from the pose held right now\n");
      }
    }
    }   // end !in_align


    // ankle zero-offset: shift the WIRE command by the calibration so the physical joint
    // reaches the planner's intended (corrected) angle. Pairs with the belief correction in
    // fill_state. Applied BEFORE the torque clamp so the clamp bounds the delta the motor PD
    // actually sees (encoder frame) -- the old shift-after-clamp order left a kp*off hole in
    // the estop guarantee. tgt_q is ENCODER/wire frame from here on; last_cmd_q un-shifts it
    // back to the true frame below (2026-07-10: storing the shifted value double-counted the
    // offset in the latency prediction -> feet-wide/backward-lean on the real A/B). 0 = no-op.
    // LEG indices 4/5/10/11 -> gated to leg-actuating nodes (motor_offset 0): on the upper
    // node those rows are ARM joints and the ankle range clamps would destroy their commands.
    if (moff == 0 && ankle_calib_on) {
      tgt_q[4]  += ankle_poff_l;
      tgt_q[5]  += ankle_off_l;
      tgt_q[10] += ankle_poff_r;
      tgt_q[11] += ankle_off_r;
    }

    // H2 torque-budget clamp: bound the emitted position target so the FULL commanded torque
    // the onboard/safety PD applies -- tau_ff + KP*(tgt-q) + KV*(0-dq) -- stays within
    // kClampRatio x the safety estop. The old clamp only bounded the KP*(tgt-q) term and
    // ignored the gravity feedforward tau_ff and the KV*dq transient, so the "estop
    // impossible" guarantee was false. The PD headroom is reduced by |tau_ff| and KV*|dq|
    // before converting to a position delta. Applied to the FINAL target (ramp + policy
    // uniformly).
    for (int i = 0; i < cfg.nu; i++) {
      const double budget = kClampRatio * cfg.tau_estop[i];
      const double pd_headroom = budget - std::fabs(tau[i]) - cfg.kv[i] * std::fabs(cur.dq[i]);
      const double dmax = (pd_headroom > 0.0) ? pd_headroom / cfg.kp[i] : 0.0;
      const double lo = cur.q[i] - dmax, hi = cur.q[i] + dmax;
      // clamp-BIT telemetry: the plan asked for more than this joint's real budget, so the
      // emitted command is NOT what the planner planned (see m_clamp_count).
      if (tgt_q[i] < lo) { tgt_q[i] = lo; if (!warming) m_clamp_count[i]++; }
      else if (tgt_q[i] > hi) { tgt_q[i] = hi; if (!warming) m_clamp_count[i]++; }
    }

    // SAFETY: hard-clamp the ankle commands to the joint ctrl range so the zero-offset can
    // NEVER drive past a limit (pitch [-0.897, 0.524] rad, roll +-0.262 rad).
    if (moff == 0 && ankle_calib_on) {
      tgt_q[4]  = std::fmin(0.524, std::fmax(-0.897, tgt_q[4]));
      tgt_q[10] = std::fmin(0.524, std::fmax(-0.897, tgt_q[10]));
      tgt_q[5]  = std::fmin(0.26, std::fmax(-0.26, tgt_q[5]));
      tgt_q[11] = std::fmin(0.26, std::fmax(-0.26, tgt_q[11]));
    }

    // ---- R6 (2026-07-04): bad-orientation damp fallback. Unitree's own
    // deploy FSM does bad_orientation(1.0 rad) -> Passive; without it the
    // planner thrashes an unrecoverable fall (the post-35deg flail in every
    // FALL trace). Latch PERMANENTLY (restart to clear): kp=0, kd=damp,
    // tau=0 -- same shape as their Passive state. cfg.bad_orient_rad 0 = off.
    static bool g_bad_orient_latched = false;
    if (cfg.bad_orient_rad > 0.0 && !g_bad_orient_latched) {
      double bx = cur.quat[1], by = cur.quat[2];       // Unitree quat = w,x,y,z
      double up_z = 1.0 - 2.0 * (bx * bx + by * by);   // torso up-axis z
      up_z = std::fmax(-1.0, std::fmin(1.0, up_z));
      if (std::acos(up_z) > cfg.bad_orient_rad) {
        g_bad_orient_latched = true;
        std::fprintf(stderr,
                     "[node] !!! BAD ORIENTATION (tilt %.1f deg > %.1f deg): "
                     "latching damp mode (kp=0, kd=3) until restart\n",
                     std::acos(up_z) * 180.0 / M_PI,
                     cfg.bad_orient_rad * 180.0 / M_PI);
      }
    }

    // ---- GRASP arm-hold latch: overwrite the right-arm command rows with the held
    // IK q* so the onboard PD holds the precise grasp pose stiff while the Magpie
    // closes. Legs/torso/left-arm keep their MPC command; kp/kd + gravity-FF tau are
    // unchanged (gravity FF prevents droop at the arm's kp=40). OFF unless the
    // orchestrator set "Arm Hold"=1 -> a normal stand/reach run never enters here.
    // tgt_q is final by this point (last write/clamp @ ~2262); clamped to the joint
    // limits, and the safety layer clamps again downstream. ----
    if (armhold_idx >= 0 && g_task->parameters[armhold_idx] > 0.5) {
      for (int k = 0; k < 7; k++) {
        double q = g_task->parameters[armhold_q_idx[k]];
        tgt_q[20 + k] = std::fmin(armhold_hi[k], std::fmax(armhold_lo[k], q));
      }
    }

    // build unitree_hg LowCmd_. The node writes ONLY its own motor rows
    // (moff..moff+nu-1); the safety layer's split merge reads exactly those
    // rows from this channel (upper node: rows 12..26 of lowcmd_upper_in),
    // so the untouched default-zero rows are never applied.
    LowCmd cmd{};
    cmd.mode_pr() = 0;                       // PR mode
    cmd.mode_machine() = cur.mode_machine;   // echo (required by the real robot)
    for (int i = 0; i < cfg.nu; i++) {
      auto& mc = cmd.motor_cmd().at(moff + i);
      mc.mode() = 1;  // 1 = enable
      if (g_bad_orient_latched) {
        mc.q() = static_cast<float>(cur.q[i]);   // irrelevant at kp=0
        mc.dq() = 0.0f;
        mc.tau() = 0.0f;
        mc.kp() = 0.0f;
        mc.kd() = 3.0f;                          // pure damping (Passive-like)
        continue;
      }
      mc.q() = static_cast<float>(tgt_q[i]);
      mc.dq() = 0.0f;
      mc.tau() = static_cast<float>(tau[i]);
      mc.kp() = static_cast<float>(cfg.kp[i]);
      mc.kd() = static_cast<float>(cfg.kv[i]);
    }
    cmd.crc() = Crc32Core(reinterpret_cast<uint32_t*>(&cmd), (sizeof(LowCmd) >> 2) - 1);
    cmd_pub->Write(cmd);

    // latency-comp bookkeeping: this command is the in-flight target during the NEXT prediction
    // window; and (AUTO mode) fold this tick's compute time tick-start->post-write into the EWMA Δ.
    // The ankle rows are un-shifted back to the TRUE frame: fill_state hands predict_forward a
    // belief-corrected (true-frame) model, so feeding it the encoder-frame wire command would
    // double-count the zero offset in every prediction (exactly off = up to 5 deg of phantom
    // ankle target -- the 2026-07-10 feet-wide/backward-lean failure). wire - off is exact even
    // after the clamps above.
    for (int i = 0; i < cfg.nu; i++)
      last_cmd_q[i] = tgt_q[i];
    if (moff == 0 && ankle_calib_on) {
      last_cmd_q[4]  -= ankle_poff_l;
      last_cmd_q[5]  -= ankle_off_l;
      last_cmd_q[10] -= ankle_poff_r;
      last_cmd_q[11] -= ankle_off_r;
    }
    if (lat_fixed <= 0.0) {
      double comp = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_tick).count();
      ewma_comp = 0.9 * ewma_comp + 0.1 * comp;
    }

    // ---- per-tick metrics (B0 baseline; balance every tick, accumulate in policy) ----
    // use the MEASURED snapshot taken before prediction (sd may now hold the predicted state).
    double base_z = meas_base_z;
    double tilt = std::acos(std::fmax(-1.0, std::fmin(1.0, meas_R8))) * 57.29577951308232;
    if (!warming) {
      m_ticks++;
      m_z_sum += base_z;
      m_z_sq += base_z * base_z;
      if (base_z < m_z_min) m_z_min = base_z;
      m_tilt_sum += tilt;
      if (tilt > m_tilt_max) m_tilt_max = tilt;
      for (int i = 0; i < cfg.nu; i++) {
        double cmd_q = tgt_q[i];    // the actual ramped command the twin/estop sees
        double err = cmd_q - cur.q[i];                              // commanded - measured
        double ae = std::fabs(err);
        m_err_sum[i] += ae;
        m_err_sq[i] += err * err;
        if (ae > m_err_max[i]) m_err_max[i] = ae;
        // torque the onboard/twin PD actually applies (matches SimInterface law)
        double total_tau = tau[i] + cfg.kp[i] * err + cfg.kv[i] * (0.0 - cur.dq[i]);
        double at = std::fabs(total_tau);
        tau_now[i] = at;
        m_tau_sum[i] += at;
        if (at > m_tau_max[i]) m_tau_max[i] = at;
        if (at > cfg.tau_limit[i]) {
          m_sat_count[i]++;
          if (!m_sat_warned) {
            std::fprintf(stderr,
                         "[node] WARNING: torque on %s = %.0f Nm > operational H1-2 limit %.0f Nm "
                         "(the safety-layer estop trips BELOW this -> address before deploy)\n",
                         cfg.joint_names[i], at, cfg.tau_limit[i]);
            m_sat_warned = true;
          }
        }
      }
    }
    if (++ticks % static_cast<long>(ctrl_hz) == 0) {
      static long last_pc = 0; static double last_w = 0.0;
      long pc = plan_count.load();
      double prate = (wall > last_w) ? (pc - last_pc) / (wall - last_w) : 0.0;
      last_pc = pc; last_w = wall;
      if (cfg.telemetry == Telemetry::kFullBody) {
        std::fprintf(stderr,
                     "[node] t=%5.1fs(twin=%5.1f) %s z=%.3f tilt=%4.1f lean(fwd/lat)=%+.1f/%+.1f "
                     "knee=%+.2f/%+.2f  Rsh(cmd/ms)=%+.0f/%+.0f Lsh=%+.0f/%+.0f[neg=FWD]  "
                     "plan=%.0f/s lat=%.0fms\n",
                     wall, twin_time, warming ? "WARMUP" : "policy", base_z, tilt,
                     meas_lean_fwd, meas_lean_lat,
                     meas_kneeL, meas_kneeR,
                     tgt_q[20] * 57.29578, cur.q[20] * 57.29578,
                     tgt_q[13] * 57.29578, cur.q[13] * 57.29578,
                     prate, dlt * 1e3);
        // torque headroom on the joints that have repeatedly limited us (elbow estop
        // 18 Nm; ankle; shoulder), as %estop so a near-trip / saturation is obvious at
        // a glance. M4: graded against TAU_ESTOP (the threshold that actually trips),
        // not the higher operational TAU_LIMIT. Indices: Rsh=20 Relb=23 Rank=10 |
        // Lsh=13 Lelb=16 Lank=4.
        std::fprintf(stderr,
                     "[node]        tau%%estop  Rsh=%.0f Relb=%.0f Rank=%.0f | "
                     "Lsh=%.0f Lelb=%.0f Lank=%.0f  (>90 = near-trip)\n",
                     100 * tau_now[20] / cfg.tau_estop[20], 100 * tau_now[23] / cfg.tau_estop[23],
                     100 * tau_now[10] / cfg.tau_estop[10], 100 * tau_now[13] / cfg.tau_estop[13],
                     100 * tau_now[16] / cfg.tau_estop[16], 100 * tau_now[4] / cfg.tau_estop[4]);
      } else if (cfg.telemetry == Telemetry::kUpperBody) {
        // UPPER-BODY node (nu=15, actuator order: torso, L arm 1..7, R arm 8..14).
        // Torso + shoulder-pitch cmd/measured (the reach DoFs), elbows, base tilt
        // (believed via IMU/weld -- the LOWER node owns balancing it).
        std::fprintf(stderr,
                     "[node] t=%5.1fs(twin=%5.1f) %s tilt=%4.1f "
                     "torso(cmd/ms)=%+.0f/%+.0f LshP=%+.0f/%+.0f RshP=%+.0f/%+.0f "
                     "Lelb=%+.0f/%+.0f Relb=%+.0f/%+.0f deg  plan=%.0f/s lat=%.0fms\n",
                     wall, twin_time, warming ? "WARMUP" : "policy", tilt,
                     tgt_q[0] * 57.29578, cur.q[0] * 57.29578,
                     tgt_q[1] * 57.29578, cur.q[1] * 57.29578,
                     tgt_q[8] * 57.29578, cur.q[8] * 57.29578,
                     tgt_q[4] * 57.29578, cur.q[4] * 57.29578,
                     tgt_q[11] * 57.29578, cur.q[11] * 57.29578,
                     prate, dlt * 1e3);
        // torque headroom (%estop) on the arm joints that limit us (shoulder-pitch
        // estop 32 Nm, elbow 14.4, wrist-pitch 9.5). nu-idx: LshP=1 Lelb=4 LwrP=6 |
        // RshP=8 Relb=11 RwrP=13.
        std::fprintf(stderr,
                     "[node]        tau%%estop  LshP=%.0f Lelb=%.0f LwrP=%.0f | "
                     "RshP=%.0f Relb=%.0f RwrP=%.0f  (>90 = near-trip)\n",
                     100 * tau_now[1] / cfg.tau_estop[1], 100 * tau_now[4] / cfg.tau_estop[4],
                     100 * tau_now[6] / cfg.tau_estop[6], 100 * tau_now[8] / cfg.tau_estop[8],
                     100 * tau_now[11] / cfg.tau_estop[11], 100 * tau_now[13] / cfg.tau_estop[13]);
      } else {
        std::fprintf(stderr,
                     "[node] t=%5.1fs(twin=%5.1f) %s z=%.3f tilt=%4.1f lean(fwd/lat)=%+.1f/%+.1f "
                     "knee=%+.2f/%+.2f  ankP(L/R)=%+.0f/%+.0f deg  "
                     "plan=%.0f/s lat=%.0fms\n",
                     wall, twin_time, warming ? "WARMUP" : "policy", base_z, tilt,
                     meas_lean_fwd, meas_lean_lat,
                     meas_kneeL, meas_kneeR,
                     cur.q[4] * 57.29578, cur.q[10] * 57.29578,
                     prate, dlt * 1e3);
        // torque headroom (%estop) on the LEG joints that limit the stand: knee
        // (300 Nm), ankle pitch (54), ankle roll (36). M4: graded against TAU_ESTOP
        // (the threshold that actually trips). Leg nu-idx: Lknee=3 Rknee=9 LankP=4
        // RankP=10 LankR=5 RankR=11.
        std::fprintf(stderr,
                     "[node]        tau%%estop  Lknee=%.0f Rknee=%.0f LankP=%.0f RankP=%.0f "
                     "LankR=%.0f RankR=%.0f  (>90 = near-trip)\n",
                     100 * tau_now[3] / cfg.tau_estop[3], 100 * tau_now[9] / cfg.tau_estop[9],
                     100 * tau_now[4] / cfg.tau_estop[4], 100 * tau_now[10] / cfg.tau_estop[10],
                     100 * tau_now[5] / cfg.tau_estop[5], 100 * tau_now[11] / cfg.tau_estop[11]);
      }
      // REACH extent + CoM-vs-feet margin from the measured-state sensors (mj_forward
      // on the real qpos ran above). Rhand_fwd = how far the right (reaching) hand is
      // ahead of the torso. CoM_margin = CoM_x - midfoot_x: + = CoM AHEAD of the feet
      // (leaning out over the toes -> the forward-creep tip risk), - = behind midfoot.
      static int adr_rh = -2, adr_tp = -2, adr_flp = -2, adr_frp = -2, adr_com = -2;
      static int adr_flf = -1, adr_frf = -1;
      if (adr_rh == -2) {
        auto SA = [&](const char* nm) {
          int id = mj_name2id(g_model, mjOBJ_SENSOR, nm);
          return id >= 0 ? g_model->sensor_adr[id] : -1;
        };
        adr_rh = SA("right_hand_pos"); adr_tp = SA("torso_position");
        adr_flp = SA("foot_left_pos"); adr_frp = SA("foot_right_pos");
        adr_com = SA("torso_subcom");
        adr_flf = SA("foot_left_forward"); adr_frf = SA("foot_right_forward");
      }
      if (adr_rh >= 0 && adr_tp >= 0 && adr_com >= 0 && adr_flp >= 0 && adr_frp >= 0) {
        // SUPPORT FRAME (2026-07-16): these were world-x differences, and the base
        // yaw fed to the planner is the RAW IMU yaw (fill_state cancels pitch/roll
        // only) -- an arbitrary power-on heading plus gyro drift. So CoM_margin
        // INVERTED SIGN with heading: yaw_probe.py measured the identical 17 cm-
        // forward pose reading +0.171 at yaw 0 and -0.171 at yaw 180 ("safely
        // behind the feet" while faceplanting). This telemetry is what the real-run
        // diagnoses are read from, so it must measure the robot's OWN forward:
        // fwd_ax = the foot-line perpendicular, exactly as stabilize.cc's residuals
        // now do. At yaw 0 fwd_ax == (1,0) -> identical to the old numbers, so the
        // reference table (stand +0.03-0.07, launch +0.22) still holds.
        double lx = sd->sensordata[adr_flp + 0], ly = sd->sensordata[adr_flp + 1];
        double rx = sd->sensordata[adr_frp + 0], ry = sd->sensordata[adr_frp + 1];
        // 2026-07-16 rev2: the frame is the mean of the FEET'S OWN headings, not the
        // perpendicular to the line between them -- see the long note at the lat_ax /
        // fwd_ax block in stabilize.cc. The two agree only for a SQUARE stance; for a
        // stagger the position form is rotated by atan2(S,W) and this readout must
        // match the residual's frame or the telemetry lies about the cost being paid.
        double ax = 0.0, ay = 0.0;
        if (adr_flf >= 0 && adr_frf >= 0) {
          ax = sd->sensordata[adr_flf + 0] + sd->sensordata[adr_frf + 0];
          ay = sd->sensordata[adr_flf + 1] + sd->sensordata[adr_frf + 1];
        }
        double alen = std::sqrt(ax * ax + ay * ay);
        double fwd_x = 1.0, fwd_y = 0.0, yaw_deg = 0.0;
        if (alen > 1.0e-6) {
          fwd_x = ax / alen; fwd_y = ay / alen;
          yaw_deg = std::atan2(fwd_y, fwd_x) * 180.0 / M_PI;
        }
        // stagger depth along the frame's own forward: + = LEFT foot leads. Zero for
        // every square strategy; this is the number the operator's hand-placement sets.
        double stag_S = (lx - rx) * fwd_x + (ly - ry) * fwd_y;
        double mid_x = 0.5 * (lx + rx), mid_y = 0.5 * (ly + ry);
        double com_margin = (sd->sensordata[adr_com + 0] - mid_x) * fwd_x +
                            (sd->sensordata[adr_com + 1] - mid_y) * fwd_y;
        double rhand_fwd = (sd->sensordata[adr_rh + 0] - sd->sensordata[adr_tp + 0]) * fwd_x +
                           (sd->sensordata[adr_rh + 1] - sd->sensordata[adr_tp + 1]) * fwd_y;
        std::fprintf(stderr,
                     "[node]        Rhand_fwd=%+.2fm  CoM_margin=%+.3fm "
                     "(+=CoM ahead of feet=fwd-tip risk)  heading=%+.0fdeg  "
                     "stagger_S=%+.3fm\n",
                     rhand_fwd, com_margin, yaw_deg, stag_S);
      }
      // Cost-term dump (debug, 2026-07-11): once/sec, every residual term on the
      // live planning state (sd->sensordata already filled by residual_sensor_callback
      // during mj_forward). Sorted by weighted contribution so the terms driving the
      // planner jump out right before a tip/fall. Does NOT change any weights.
      // Gated behind --cost (cfg.cost_log); OFF by default so the terminal stays quiet.
      if (cfg.cost_log) {
        const mjpc::Task* task = g_agent.ActiveTask();
        double terms_u[mjpc::kMaxCostTerms];
        double terms_w[mjpc::kMaxCostTerms];
        task->UnweightedCostTerms(terms_u, sd->sensordata);
        task->CostTerms(terms_w, sd->sensordata);
        struct CostRow {
          const char* name;
          double w, r, c;
        };
        std::vector<CostRow> rows;
        rows.reserve(static_cast<size_t>(task->num_term));
        double total = 0.0;
        for (int i = 0; i < task->num_term; ++i) {
          const char* nm =
              (i < static_cast<int>(task->weight_names.size()) &&
               !task->weight_names[i].empty())
                  ? task->weight_names[i].c_str()
                  : (g_model->names + g_model->name_sensoradr[i]);
          rows.push_back(CostRow{nm, task->weight[i], terms_u[i], terms_w[i]});
          total += terms_w[i];
        }
        std::sort(rows.begin(), rows.end(),
                  [](const CostRow& a, const CostRow& b) { return a.c > b.c; });
        std::fprintf(stderr,
                     "[cost] t=%.1fs total=%.4f  "
                     "(sorted by weighted contrib; w=weight r=norm c=weighted)\n",
                     wall, total);
        for (const CostRow& row : rows) {
          std::fprintf(stderr,
                       "[cost]   %-22s  w=%8.3f  r=%10.5f  c=%10.5f\n",
                       row.name, row.w, row.r, row.c);
        }
      }
    }

    next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(ctrl_dt));
    std::this_thread::sleep_until(next);
  }

  std::fprintf(stderr, "[node] shutting down ...\n");
  // ---- Title-5 baseline (B0) summary ----
  if (m_ticks > 0) {
    double inv = 1.0 / m_ticks;
    double zmean = m_z_sum * inv;
    double zsd = std::sqrt(std::fmax(0.0, m_z_sq * inv - zmean * zmean));
    std::fprintf(stderr,
                 "\n[B0] ===== baseline strategy=%d over %ld policy ticks (%.1fs @ %.0fHz) =====\n",
                 cfg.strategy, m_ticks, m_ticks * ctrl_dt, ctrl_hz);
    std::fprintf(stderr,
                 "[B0] base_z mean=%.3f sd=%.3f min=%.3f m | tilt mean=%.1f max=%.1f deg\n",
                 zmean, zsd, m_z_min, m_tilt_sum * inv, m_tilt_max);
    std::fprintf(stderr,
                 "[B0] joint   trackRMSE trackMax |tau|mean |tau|peak  limit  peak%%  sat%%  "
                 "budget clamp%%\n");
    std::fprintf(stderr,
                 "[B0]          (deg)    (deg)     (Nm)     (Nm)      (Nm)              (Nm)\n");
    int n_over = 0, n_starved = 0;
    for (int i = 0; i < cfg.nu; i++) {
      double pk = m_tau_max[i], lim = cfg.tau_limit[i];
      double pkpct = 100.0 * pk / lim, satpct = 100.0 * m_sat_count[i] * inv;
      double budget = kClampRatio * cfg.tau_estop[i];
      double clpct = 100.0 * m_clamp_count[i] * inv;
      if (pk > lim) n_over++;
      const bool starved = clpct >= 5.0;   // out of authority for >=5% of the run
      if (starved) n_starved++;
      std::fprintf(stderr,
                   "[B0] %-7s %8.2f %8.2f %8.1f %9.1f %6.0f %6.0f %5.1f %7.1f %5.1f%s%s\n",
                   cfg.joint_names[i], std::sqrt(m_err_sq[i] * inv) * 57.29577951308232,
                   m_err_max[i] * 57.29577951308232, m_tau_sum[i] * inv, pk, lim, pkpct, satpct,
                   budget, clpct, starved ? "  <<OUT-OF-AUTHORITY" : "",
                   pk > lim ? "  <<OVER-LIMIT" : "");
    }
    std::fprintf(stderr,
                 "[B0] torque headroom: %d/%d joints exceed the real H1-2 limit%s\n",
                 n_over, cfg.nu,
                 n_over == 0 ? " -> torque-safe for hardware" : " <-- ADDRESS before deploy");
    // clamp% is the authority read: a joint pinned at its budget did NOT do what the plan
    // asked. On the 2026-07-11 real trot the stance ankle + torso sat at 100% of budget for
    // 6 s and the stance knee locked to a passive prop -- that is what this row now exposes.
    std::fprintf(stderr,
                 "[B0] authority: %d/%d joints hit the H2 torque budget >=5%% of ticks%s\n",
                 n_starved, cfg.nu,
                 n_starved == 0
                     ? " -> the planner got the torque it planned"
                     : " <-- OUT OF AUTHORITY: the plan asked for torque the node cannot emit."
                       " Enable --frc_parity=1 so the PLANNER knows (else it keeps planning"
                       " motions that physically cannot execute), and/or reduce the demand.");
    std::fprintf(stderr, "[B0] (run the SAME node on the real robot -> per-row sim2real delta)\n");
  }
#ifdef H12_NODE_GRPC
  if (grpc_server) grpc_server->Shutdown();
#endif
  g_emit_safe_hold = nullptr;   // publisher is about to die; M5 handler must not use it
  plan_exit.store(true);
  if (planner.joinable()) planner.join();
  mj_deleteData(gd);
  mj_deleteData(sd);
  mj_deleteData(pdat);
  mj_deleteData(data);
  mj_deleteModel(g_model);
  return 0;
}

}  // namespace h12deploy
