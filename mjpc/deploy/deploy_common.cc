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
void PatchActuators(mjModel* m, const NodeConfig& cfg) {
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
}

// Plain, copyable snapshot of the latest robot state.
struct StateData {
  bool have_ls = false, have_ss = false;
  double q[kMaxNU] = {0}, dq[kMaxNU] = {0};
  double qu[15] = {0}, dqu[15] = {0};  // arm-aware: the 15 UPPER joints (torso+arms, motor 12..26)
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

}  // namespace

int RunDeployNode(const NodeConfig& cfg) {
  const std::string& task_id = cfg.task_id;
  const double gff = cfg.gravity_ff;
  // FABEL (2026-07-07, env H12_CTRL_HZ): override the compiled 200 Hz control/
  // publish rate. 500 shrinks the command-side ZOH age 5 -> 2 ms -- part of
  // the dev-identified "close the ~10 ms gap structurally" ladder (the
  // post-ABI-fix chain stands at rtf 0.5 and loses the release/hunt race at
  // rtf 1 by a hair). Unset = compiled kCtrlHz (unchanged).
  double ctrl_hz = kCtrlHz;
  if (const char* e = std::getenv("H12_CTRL_HZ")) {
    double v = std::atof(e);
    if (v >= 50.0 && v <= 1000.0) ctrl_hz = v;
    std::fprintf(stderr, "[node] FABEL H12_CTRL_HZ=%.0f\n", ctrl_hz);
  }
  const double ctrl_dt = 1.0 / ctrl_hz;
  // FABEL bench knob (2026-07-07): env H12_WARMUP_SEC overrides the compiled
  // 1.0 s warmup hold. On a sim bench whose plant FREEZES the pose until the
  // first stiff command (the twin's crouch rehearsal), the warmup statue is
  // dead time during which an unsupported robot tips ~1 deg/s; a short warmup
  // hands the policy a still-clean state. Unset = compiled default.
  double warmup_sec = kWarmupSec;
  if (const char* e = std::getenv("H12_WARMUP_SEC")) {
    double v = std::atof(e);
    if (v >= 0.0 && v <= 10.0) warmup_sec = v;
  }
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
  g_task = g_agent.ActiveTask();
  g_agent_model = g_agent.GetModel();
  g_model = mj_copyModel(nullptr, g_agent_model);
  PatchActuators(g_model, cfg);   // latency-comp rollout model (the planner model was patched at LoadModel above)
  mjData* data = mj_makeData(g_model);
  int home = mj_name2id(g_model, mjOBJ_KEY, "home");
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

  const int nq = g_model->nq, nv = g_model->nv, nu = g_model->nu;
  const int nact = nu < cfg.nu ? nu : cfg.nu;

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
  ChannelSubscriberPtr<LowState> ls_sub(
      new ChannelSubscriber<LowState>(cfg.lowstate_topic));
  ls_sub->InitChannel(
      [&rs, &cfg, n_upper](const void* msg) {
        const LowState* s = static_cast<const LowState*>(msg);
        std::lock_guard<std::mutex> lk(rs.mu);
        for (int i = 0; i < cfg.nu; i++) {
          rs.d.q[i] = s->motor_state().at(i).q();
          rs.d.dq[i] = s->motor_state().at(i).dq();
        }
        // arm-aware (legs-only node): also latch the upper joints (motor 12..26) so the
        // planner CoM can track them. motor_state() is a fixed-size array on the H1-2
        // (>=27); guard defensively.
        for (int i = 0; i < n_upper && static_cast<size_t>(12 + i) < s->motor_state().size(); i++) {
          rs.d.qu[i] = s->motor_state().at(12 + i).q();
          rs.d.dqu[i] = s->motor_state().at(12 + i).dq();
        }
        for (int k = 0; k < 4; k++) rs.d.quat[k] = s->imu_state().quaternion().at(k);
        for (int k = 0; k < 3; k++) rs.d.gyro[k] = s->imu_state().gyroscope().at(k);
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
  auto emit_safe_hold = [&rs, &cmd_pub, &cfg]() {
    StateData snap;
    {
      std::lock_guard<std::mutex> lk(rs.mu);
      snap = rs.d;
    }
    LowCmd cmd{};
    cmd.mode_pr() = 0;
    cmd.mode_machine() = snap.mode_machine;
    for (int i = 0; i < cfg.nu; i++) {
      auto& mc = cmd.motor_cmd().at(i);
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

  // ---- ARM-AWARE balance (legs-only node): map each equality lock -> the upper-joint
  //      index it holds (qposadr-7), so fill_state can retarget the lock to the MEASURED
  //      arm pose every tick. The planner then balances against the real arm CoM with the
  //      legs; it still has NO upper-body actuator. ----
  const bool arm_aware = (n_upper > 0) && cfg.arm_aware;
  std::vector<int> eq_motor(g_model->neq, -1);   // eq e -> upper motor idx (12..26), -1 otherwise
  if (arm_aware) {
    for (int e = 0; e < g_model->neq; e++) {
      if (g_model->eq_type[e] != mjEQ_JOINT) continue;
      int midx = g_model->jnt_qposadr[g_model->eq_obj1id[e]] - 7;  // joint/motor index (0..26)
      if (midx >= 12 && midx <= 26) eq_motor[e] = midx;
    }
    int n = 0; for (int e = 0; e < g_model->neq; e++) if (eq_motor[e] >= 0) n++;
    std::fprintf(stderr, "[node] ARM-AWARE ON: reading %d upper joints from rt/lowstate; %d equality "
                         "locks retargeted to the measured arm pose each tick (legs still nu=%d; the "
                         "node NEVER commands the arms -- FrameTask IK owns lowcmd_upper_in)\n",
                 n_upper, n, cfg.nu);
  } else if (n_upper > 0) {
    std::fprintf(stderr, "[node] arm-aware OFF: upper joints held at home in the planner (validated "
                         "home-locked stand). Pass --arm_aware for loco-manip with the IK.\n");
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
  const int n_plan_threads =
      cfg.plan_threads > 0 ? cfg.plan_threads
      : (kPlanThreads > 0 ? kPlanThreads : mjpc::NumAvailableHardwareThreads());
  std::fprintf(stderr, "[node] planner ThreadPool: %d threads (%d hw available; %s "
                       "leaves CPU for the twin/safety layer)\n",
               n_plan_threads, mjpc::NumAvailableHardwareThreads(),
               cfg.plan_threads > 0 ? "--plan_threads CLI" : "compiled kPlanThreads");
  std::fprintf(stderr,
               "[node] torque-budget clamp ON (audit H2): |tau_ff + kp*(tgt-q) + kv*dq| <= %.1f x "
               "tau-estop per joint (command-side estops impossible for ANY planner output)\n",
               kClampRatio);

  mjpc::ThreadPool plan_pool(n_plan_threads);
  // FABEL (2026-07-07, env H12_PLAN_GATE=1): hold the planner until the ctrl
  // loop's FIRST Transition() has configured the task (strategy JSON posture
  // keyframe + weights). Un-gated, the free-running planner iterates on the
  // unconfigured task and its CEM warm-start settles in the straight-knee
  // home basin -- measured: on a bit-identical frozen stand state the node
  // emits hipP +0.09 (home) where the once-configured probe emits -0.04
  // (stand). The agent server never plans unconfigured; this makes the node
  // match. Unset = unchanged.
  std::atomic<bool> plan_gate{false};
  const bool plan_gate_on = [] {
    const char* e = std::getenv("H12_PLAN_GATE");
    return e && e[0] == '1';
  }();
  if (plan_gate_on)
    std::fprintf(stderr, "[node] FABEL H12_PLAN_GATE=1: planner held until "
                         "first task Transition\n");
  std::thread planner([&] {
    if (plan_gate_on)
      while (!plan_gate.load() && !plan_exit.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    while (!plan_exit.load()) { g_agent.PlanIteration(&plan_pool); plan_count.fetch_add(1); }
  });

  // ---- live strategy switch via stdin: type a number 0-20 (+Enter), q=quit ----
  std::fprintf(stderr, "[node] live switch ready: type a strategy number 0-20 + Enter (q=quit)\n");
  std::thread stdin_thread([&] {
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line == "q" || line == "quit") { g_exit.store(true); break; }
      if (line.empty()) continue;
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
  if (imu_pitch_off != 0.0 || imu_roll_off != 0.0)
    std::printf("[node] IMU zero-offset calibration ON: perceived base orientation rotated "
                "pitch%+.2f roll%+.2f deg before planning\n",
                cfg.imu_pitch_offset_deg, cfg.imu_roll_offset_deg);
  const double ankle_off_l = cfg.ankle_roll_offset_l_deg * M_PI / 180.0;
  const double ankle_off_r = cfg.ankle_roll_offset_r_deg * M_PI / 180.0;

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
    if (imu_pitch_off != 0.0) {
      double d[4], ax[3] = {0, 1, 0}, t[4];
      mju_axisAngle2Quat(d, ax, imu_pitch_off);
      mju_mulQuat(t, bq, d); mju_copy4(bq, t);
    }
    if (imu_roll_off != 0.0) {
      double d[4], ax[3] = {1, 0, 0}, t[4];
      mju_axisAngle2Quat(d, ax, imu_roll_off);
      mju_mulQuat(t, bq, d); mju_copy4(bq, t);
    }
    mju_normalize4(bq);
    for (int k = 0; k < 4; k++) sd->qpos[3 + k] = bq[k];
    for (int i = 0; i < cfg.nu; i++) sd->qpos[7 + i] = cur.q[i];
    // ankle-roll zero-offset calibration: the planner sees the CORRECTED roll (idx 5=L, 11=R
    // ankle_roll) so it doesn't chase a foot only the encoder thinks is rolled. Pairs with the
    // command shift before publish below. 0 = no-op.
    sd->qpos[7 + 5]  -= ankle_off_l;
    sd->qpos[7 + 11] -= ankle_off_r;
    for (int k = 0; k < 3; k++) sd->qvel[3 + k] = cur.gyro[k];        // free-joint angvel == body gyro
    for (int i = 0; i < cfg.nu; i++) sd->qvel[6 + i] = cur.dq[i];
    // arm-aware (legs-only node): inject the MEASURED upper-body pose so the planner's
    // CoM/dynamics track the arms, and retarget the equality locks to HOLD them there during
    // each rollout (instead of snapping to home). The planner has no upper-body actuator ->
    // it cannot command the arms; it only balances against them with the legs. eq_data is
    // retargeted on BOTH models (g_model = Transition + latency rollout; g_agent_model = the
    // planner's rollout model). No-op when the arms are at home.
    if (arm_aware) {
      for (int i = 0; i < n_upper; i++) {
        sd->qpos[7 + 12 + i] = cur.qu[i];     // qpos[19..33] = torso + 14 arm joints
        sd->qvel[6 + 12 + i] = cur.dqu[i];    // qvel[18..32]
      }
      for (int e = 0; e < g_model->neq; e++) {
        if (eq_motor[e] < 0) continue;
        mjtNum v = cur.qu[eq_motor[e] - 12];
        g_model->eq_data[e * mjNEQDATA + 0] = v;
        const_cast<mjtNum*>(g_agent_model->eq_data)[e * mjNEQDATA + 0] = v;
      }
    }
  };

  // ---- latency compensation: roll the planner model forward by Δ under the in-flight command ----
  // The planner model's <position> actuators have kp/kv == the node's KP[]/KV[] == the twin's PD,
  // so this rollout reproduces the twin's joint law; we only add the gravity tau_ff the twin also
  // applies. Object/task slots are left at home (we copy only the robot dofs back). NaN-guarded.
  double lat_fixed = kLatencyFixedMs * 1e-3;
  // FABEL (2026-07-07): env H12_LAT_FIXED_MS overrides the compensation
  // horizon. AUTO (EWMA compute + 4 ms) covers ~9 ms; the full
  // state->plan->action->ZOH age is ~20+ ms, and the measured stability
  // boundary of the stand policy on the twin plant is 10-25 ms of state
  // age (probe lag sweep). Unset = compiled default.
  if (const char* e = std::getenv("H12_LAT_FIXED_MS")) {
    double v = std::atof(e);
    if (v >= 0.0 && v <= 40.0) lat_fixed = v * 1e-3;
    std::fprintf(stderr, "[node] FABEL H12_LAT_FIXED_MS=%.1f\n", v);
  }
  const double lat_extra = kLatencyExtraMs * 1e-3;
  const double lat_max   = kLatencyMaxMs * 1e-3;
  const double pred_dt   = g_model->opt.timestep;     // native model step (0.002) == twin granularity
  double ewma_comp = 1.0 / ctrl_hz;                   // measured per-tick compute time (EWMA), seeded
  double last_cmd_q[kMaxNU];
  for (int i = 0; i < cfg.nu; i++) last_cmd_q[i] = sd->qpos[7 + i];   // home target until first publish
  // ---- bring-up ramp state: blend ALL joints from the measured power-on pose to the
  //      home/policy target over kStartRampSec so the warmup->policy switch never snaps.
  const double start_ramp_sec = kStartRampSec;
  std::fprintf(stderr, "[node] bring-up ramp ON: ALL joints measured->home/policy over %.1fs "
                       "(no-op if already at home)\n", start_ramp_sec);
  double home_q[kMaxNU];
  { // bring-up destination = the OPERATING stance ("stand", bent-knee), NOT the singular
    // straight-knee "home": ramping to knee=0 hands the policy the hyperextension basin
    // (knees snapped to the -0.12 stop within 1 s of ramp end -- live-verified).
    int hk = mj_name2id(g_model, mjOBJ_KEY, "stand");
    if (hk < 0) hk = mj_name2id(g_model, mjOBJ_KEY, "home");
    if (hk < 0) hk = 0;
    std::fprintf(stderr, "[node] bring-up ramp destination keyframe: '%s'\n",
                 mj_id2name(g_model, mjOBJ_KEY, hk));
    for (int i = 0; i < cfg.nu; i++) home_q[i] = g_model->key_qpos[hk * nq + 7 + i]; }
  double arm_q_init[kMaxNU];
  for (int i = 0; i < cfg.nu; i++) arm_q_init[i] = sd->qpos[7 + i];   // refined to measured pose on first live state
  bool arm_init_set = false;
  double ramp_eff = start_ramp_sec;   // rescaled at latch by distance-from-home (see loop)

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
      for (int i = 0; i < cfg.nu; i++) pdat->qfrc_applied[6 + i] = gff * gd->qfrc_bias[6 + i];
    }
    for (int k = 0; k < nsub; k++) {
      for (int i = 0; i < nact; i++) pdat->ctrl[i] = cmd_q[i];   // position-actuator target = in-flight cmd
      mj_step(g_model, pdat);
    }
    bool ok = true;                                     // discard a blown-up rollout, keep measured state
    for (int k = 0; ok && k < 7 + cfg.nu; k++) if (!std::isfinite(pdat->qpos[k])) ok = false;
    for (int k = 0; ok && k < 6 + cfg.nu; k++) if (!std::isfinite(pdat->qvel[k])) ok = false;
    if (!ok) return;
    for (int k = 0; k < 7 + cfg.nu; k++) s->qpos[k] = pdat->qpos[k];
    for (int k = 0; k < 6 + cfg.nu; k++) s->qvel[k] = pdat->qvel[k];
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
  bool m_sat_warned = false;

  // ---- control loop @ ctrl_hz (mirrors app.cc physics thread) ----
  const double twin_dt = cfg.twin_dt;
  uint32_t tick0 = 0; bool tick0_set = false;   // zero the twin sim-clock at the first state
  // single-stream base-velocity finite-diff + LPF state (see kVelLpfMs).
  // FABEL bench knob (2026-07-07): env H12_VEL_LPF_MS overrides the compiled
  // 30 ms LPF. With a TRUTH sportstate (sim benches) the position stream is
  // noise-free, so the LPF's only effect is ~30 ms of velocity lag -- ~10% of
  // the inverted-pendulum divergence constant -- which erodes the planner's
  // balance damping. Unset = compiled default (real robot unchanged).
  double vel_lpf_ms = kVelLpfMs;
  if (const char* e = std::getenv("H12_VEL_LPF_MS")) {
    double v = std::atof(e);
    if (v >= 0.0 && v <= 200.0) vel_lpf_ms = v;
  }
  const double vel_lpf_tau = vel_lpf_ms * 1e-3;
  double fd_prev_p[3] = {0}, fd_vel[3] = {0}, fd_prev_t = 0.0; bool fd_have = false;
  std::fprintf(stderr,
               "[node] base linvel: SINGLE-STREAM finite-diff + %.0fms LPF (drops the two-stream phantom)\n",
               vel_lpf_ms);
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
      // FABEL bench knob (2026-07-07): env H12_SKIP_BRINGUP=1 forces
      // ramp_eff = 0.0 exactly -> the target ladder's else-branch gives
      // tgt = action from the first post-warmup tick (no ramp/hold/blend).
      // The measured blend-end "first tug" walk-off and the support-
      // exploitation wind-up both live inside that ladder; a plant that
      // freezes the pose until the first stiff command (twin crouch
      // rehearsal) makes the ladder unnecessary: the policy inherits a
      // clean statically-posed robot -- the condition under which the
      // planner-only probe stood 300 s. Bench-only; unset = unchanged.
      if (const char* e = std::getenv("H12_SKIP_BRINGUP")) {
        if (e[0] == '1') {
          ramp_eff = 0.0;
          std::fprintf(stderr, "[node] FABEL H12_SKIP_BRINGUP=1: ramp/hold/"
                               "blend ladder BYPASSED (tgt=action post-warmup)\n");
        }
      }
      std::fprintf(stderr, "[node] bring-up ramp: latched pose is %.2f rad from home -> "
                           "effective ramp %.1fs\n", d0, ramp_eff);
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
      dlt = (lat_fixed > 0.0) ? lat_fixed : (ewma_comp + lat_extra);
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
    // FABEL (2026-07-07, env H12_KEY_MOCAP=1): the Stabilize residual reads
    // data->mocap_pos as its LIVE reach/lean target; the node's sd carries
    // raw mj_makeData mocap (XML body pos), while the validated agent-server
    // path plans with KEYFRAME mocap (mj_resetDataKeyframe -> key_mpos).
    // Seed sd's mocap from the bring-up keyframe once. Unset = unchanged.
    {
      static int km_on = -1;
      if (km_on < 0) {
        const char* e = std::getenv("H12_KEY_MOCAP");
        km_on = (e && e[0] == '1') ? 1 : 0;
        if (km_on && g_model->nmocap > 0) {
          int hk = mj_name2id(g_model, mjOBJ_KEY, "stand");
          if (hk < 0) hk = 0;
          mju_copy(sd->mocap_pos, g_model->key_mpos + hk * 3 * g_model->nmocap,
                   3 * g_model->nmocap);
          mju_copy(sd->mocap_quat, g_model->key_mquat + hk * 4 * g_model->nmocap,
                   4 * g_model->nmocap);
          std::fprintf(stderr, "[node] FABEL H12_KEY_MOCAP=1: mocap seeded "
                               "from key (%.2f %.2f %.2f)\n",
                       sd->mocap_pos[0], sd->mocap_pos[1], sd->mocap_pos[2]);
        }
      }
    }
    g_agent.ActiveTask()->Transition(g_model, sd);
    g_agent.SetState(sd);
    plan_gate.store(true);   // FABEL: task is configured -> release the planner
    // FABEL (2026-07-07, env H12_DUMP=1): one-shot dump of the task config
    // ACTUALLY in effect once the policy is active -- weights + residual
    // parameters + mode -- for the diff against the healthy agent server
    // (get_cost_weights / get_task_parameters) on the same frozen state.
    {
      static int dump_on = -1;
      if (dump_on < 0) {
        const char* e = std::getenv("H12_DUMP");
        dump_on = (e && e[0] == '1') ? 1 : 0;
      }
      if (dump_on == 1 && !warming && phase_t > 1.0) {
        dump_on = 2;
        const mjpc::Task* tk = g_agent.ActiveTask();
        {
          int nu_g = 0, nu_a = 0;
          const mjModel* am = g_agent.GetModel();
          for (int si = 0; si < g_model->nsensor; si++)
            if (g_model->sensor_type[si] == mjSENS_USER) nu_g++;
          for (int si = 0; si < am->nsensor; si++)
            if (am->sensor_type[si] == mjSENS_USER) nu_a++;
          std::fprintf(stderr,
                       "DUMP sensors: g_model nsensor=%d user=%d | agent "
                       "nsensor=%d user=%d | weights=%zu\n",
                       g_model->nsensor, nu_g, am->nsensor, nu_a,
                       tk->weight.size());
        }
        std::fprintf(stderr, "DUMP mode=%d\nDUMP weights:", tk->mode);
        {
          // pair each weight with its USER-sensor name (the residual term
          // list) -- kills the name-index ambiguity of the first dump
          size_t k = 0;
          for (int si = 0; si < g_model->nsensor && k < tk->weight.size(); si++) {
            if (g_model->sensor_type[si] != mjSENS_USER) continue;
            const char* nm = mj_id2name(g_model, mjOBJ_SENSOR, si);
            std::fprintf(stderr, " [%s]=%.4g", nm ? nm : "?", tk->weight[k]);
            k++;
          }
          for (; k < tk->weight.size(); k++)
            std::fprintf(stderr, " [pad]=%.4g", tk->weight[k]);
        }
        std::fprintf(stderr, "\nDUMP params:");
        for (size_t i = 0; i < tk->parameters.size(); i++)
          std::fprintf(stderr, " %.4g", tk->parameters[i]);
        std::fprintf(stderr, "\nDUMP state:");
        const auto& st = g_agent.state.state();
        for (size_t i = 0; i < st.size(); i++)
          std::fprintf(stderr, " %.5g", st[i]);
        std::fprintf(stderr, "\n");
      }
    }

    // policy: target joint positions (rad). Held at measured q during warmup.
    if (!warming) {
      // FABEL (2026-07-07): env H12_ACTION_LEAD_MS reads the policy spline
      // this far AHEAD of now -- feedforward compensation for the
      // action-path delay (cmd publish -> safety ZOH -> plant apply) that
      // the state-side predict_forward cannot cover. 0/unset = unchanged.
      static double act_lead = -1.0;
      if (act_lead < 0.0) {
        act_lead = 0.0;
        if (const char* e = std::getenv("H12_ACTION_LEAD_MS")) {
          double v = std::atof(e);
          if (v >= 0.0 && v <= 60.0) act_lead = v * 1e-3;
          std::fprintf(stderr, "[node] FABEL H12_ACTION_LEAD_MS=%.1f\n", v);
        }
      }
      g_agent.ActivePlanner().ActionFromPolicy(action.data(), g_agent.state.state().data(),
                                             g_agent.state.time() + act_lead);
      // FABEL (2026-07-08, env H12_ACT_LPF_MS): 1st-order low-pass on the
      // emitted action. The CEM elite-mean policy dithers at ~2-3 Hz
      // (unseeded per-iteration sampling noise, +-0.05-0.1 rad on the leg
      // targets); through the chain's ~15 ms age that dither excites the
      // very tilt excursions that escape the recoverable envelope
      // (post-ABI-fix hunts die at a ~1/150-300 s escape rate). An ~80 ms
      // pole attenuates the dither hard while costing little phase at the
      // ~0.5-1 Hz balance band. Unset/0 = unchanged.
      {
        static double lpf_tau = -1.0;
        static double lpf_a[kMaxNU];
        static bool lpf_init = false;
        if (lpf_tau < 0.0) {
          lpf_tau = 0.0;
          if (const char* e = std::getenv("H12_ACT_LPF_MS")) {
            double v = std::atof(e);
            if (v > 0.0 && v <= 500.0) lpf_tau = v * 1e-3;
            std::fprintf(stderr, "[node] FABEL H12_ACT_LPF_MS=%.0f\n", v);
          }
        }
        if (lpf_tau > 0.0) {
          if (!lpf_init) {
            lpf_init = true;
            for (int i = 0; i < nu && i < kMaxNU; i++) lpf_a[i] = action[i];
          }
          const double a = ctrl_dt / (lpf_tau + ctrl_dt);
          for (int i = 0; i < nu && i < kMaxNU; i++) {
            lpf_a[i] += a * (action[i] - lpf_a[i]);
            action[i] = lpf_a[i];
          }
        }
      }
    }

    // gravity feedforward: tau = gff * qfrc_bias evaluated at qvel = 0.
    double tau[kMaxNU] = {0};
    if (gff != 0.0) {
      mju_copy(gd->qpos, sd->qpos, nq);
      mju_zero(gd->qvel, nv);
      mj_forward(g_model, gd);
      for (int i = 0; i < cfg.nu; i++) tau[i] = gff * gd->qfrc_bias[6 + i];
    }

    // ---- live strategy-switch blend: on a pending switch, snapshot the CURRENT measured pose
    //      as the blend start + stamp the switch time, so the target eases from where the robot
    //      IS into the new policy target (no snap). Serves stdin switches (strategy 18's internal
    //      stand<->crouch cycling is handled by the task phase machine, not here).
    if (g_switch_pending.exchange(false)) {
      for (int i = 0; i < cfg.nu; i++) g_switch_from[i] = cur.q[i];
      g_switch_wall.store(phase_t);   // switch settle/blend run on the plant clock too
    }

    // per-joint commanded position target with the BRING-UP ramp: blend from the measured
    // power-on pose to the home/policy target on ALL joints (kills the warmup->policy SNAP
    // from off-home starts).
    double tgt_q[kMaxNU];
    for (int i = 0; i < cfg.nu; i++) {
      double ramp_dur = ramp_eff;
      double base;
      if (ramp_dur > 0.0 && arm_init_set) {
        double aa = std::fmin(1.0, std::fmax(0.0, phase_t / ramp_dur));
        // SCRIPTED rise: target the stance for the whole ramp (policy steering a half-
        // risen, moving robot topples it -- bench-proven), then HOLD the stance scripted
        // for kRampHoldSec more so CEM converges around the STATIC operating pose
        // before getting authority (kills the hand-off "first tug" toward the old basin).
        const double t_ho = ramp_dur + kRampHoldSec;
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
        base = (1.0 - aa) * arm_q_init[i] + aa * tgt;
      } else {
        base = (warming || i >= nact) ? cur.q[i] : action[i];
      }
      tgt_q[i] = base;
    }

    // H2 torque-budget clamp: bound the emitted position target so the FULL commanded torque
    // the onboard/safety PD applies -- tau_ff + KP*(tgt-q) + KV*(0-dq) -- stays within
    // kClampRatio x the safety estop. The old clamp only bounded the KP*(tgt-q) term and
    // ignored the gravity feedforward tau_ff and the KV*dq transient, so the "estop
    // impossible" guarantee was false. The PD headroom is reduced by |tau_ff| and KV*|dq|
    // before converting to a position delta. Applied to the FINAL target (ramp + policy
    // uniformly); last_cmd_q below stores the clamped value so the latency predictor models
    // what was actually sent.
    // FABEL (2026-07-07, env H12_CLAMP_URDF=1): clamp budget = the OPERATIONAL
    // URDF limit (cfg.tau_limit) instead of 0.9 x TAU_ESTOP. Pairs with
    // H12_FRC_PARITY=urdf (planner forceranges = URDF) so planned catches ==
    // deliverable catches == the motor's real capability. Bench-scoped: the
    // sim safety estop is disabled; on real HW the estop thresholds trip
    // below URDF, so keep this OFF there. Unset = unchanged (0.9 x estop).
    static int clamp_urdf = -1;
    if (clamp_urdf < 0) {
      const char* e = std::getenv("H12_CLAMP_URDF");
      clamp_urdf = (e && e[0] == '1') ? 1 : 0;
      if (clamp_urdf)
        std::fprintf(stderr, "[node] FABEL H12_CLAMP_URDF=1: torque budget = "
                             "URDF operational limits\n");
    }
    for (int i = 0; i < cfg.nu; i++) {
      const double budget = clamp_urdf ? cfg.tau_limit[i]
                                       : kClampRatio * cfg.tau_estop[i];
      const double pd_headroom = budget - std::fabs(tau[i]) - cfg.kv[i] * std::fabs(cur.dq[i]);
      const double dmax = (pd_headroom > 0.0) ? pd_headroom / cfg.kp[i] : 0.0;
      const double lo = cur.q[i] - dmax, hi = cur.q[i] + dmax;
      if (tgt_q[i] < lo) tgt_q[i] = lo;
      else if (tgt_q[i] > hi) tgt_q[i] = hi;
    }

    // ankle-roll zero-offset: shift the COMMAND by the calibration so the physical joint reaches
    // the planner's intended (corrected) angle. Pairs with the belief correction in fill_state;
    // applied to tgt_q so mc.q() AND last_cmd_q (latency model) stay consistent. 0 = no-op.
    tgt_q[5]  += ankle_off_l;
    tgt_q[11] += ankle_off_r;
    // SAFETY: the offset is applied after the torque clamp, so hard-clamp the ankle_roll command
    // to the joint's ctrl range (+-0.26 rad == +-14.9 deg) -> the offset can NEVER drive past limit.
    tgt_q[5]  = std::fmin(0.26, std::fmax(-0.26, tgt_q[5]));
    tgt_q[11] = std::fmin(0.26, std::fmax(-0.26, tgt_q[11]));

    // ---- FABEL AUTHORITY SLEW (2026-07-07, env-gated) ------------------
    // H12_SLEW=<rad/s> rate-limits the emitted target for the first
    // H12_SLEW_SEC (default 2) seconds of post-warmup authority, seeded at
    // the measured pose. The policy's first actions on a static stand pull
    // the knees straight / hips forward ~0.3 rad in one tick (the "first
    // tug"); the tight in-process probe survives its own tug, the DDS
    // chain's latency does not. Slewing the target lets the plant move with
    // the plan instead of being yanked. Unset = byte-identical behavior.
    {
      static double slew_rate = -1.0, slew_sec = 2.0, slew_t0 = -1.0;
      static double slew_q[kMaxNU];
      static bool slew_init = false;
      static int slew_on_motion = 0;
      if (slew_rate < 0.0) {
        slew_rate = 0.0;
        if (const char* e = std::getenv("H12_SLEW")) {
          double v = std::atof(e);
          if (v > 0.0 && v <= 20.0) slew_rate = v;
        }
        if (const char* e = std::getenv("H12_SLEW_SEC")) {
          double v = std::atof(e);
          if (v > 0.0 && v <= 30.0) slew_sec = v;
        }
        // FABEL (2026-07-07, env H12_SLEW_ON_MOTION=1): seed the envelope at
        // MOTION ONSET (release) instead of warmup end. On the freeze-hold
        // bench the plant pins qpos with dq == 0 until release; the moment of
        // authority is when the robot first MOVES, which is when the
        // release-tug transient (plan equilibrium != frozen pose, CEM wobble
        // phase) needs the clamp. Warmup-end seeding expires long before the
        // release and covers nothing.
        if (const char* e = std::getenv("H12_SLEW_ON_MOTION"))
          slew_on_motion = (e[0] == '1') ? 1 : 0;
        if (slew_rate > 0.0)
          std::fprintf(stderr, "[node] FABEL H12_SLEW=%.2f rad/s for first "
                               "%.1fs of authority%s\n", slew_rate, slew_sec,
                       slew_on_motion ? " (seeded at MOTION ONSET)" : "");
      }
      if (slew_rate > 0.0 && !warming) {
        bool may_seed = true;
        if (slew_on_motion && !slew_init) {
          double dqmax = 0.0;
          for (int i = 0; i < cfg.nu; i++)
            dqmax = std::fmax(dqmax, std::fabs(cur.dq[i]));
          may_seed = dqmax > 0.05;   // plant froze dq==0; motion = release
        }
        if (!slew_init && may_seed) {
          slew_init = true;
          slew_t0 = phase_t;
          for (int i = 0; i < cfg.nu; i++) slew_q[i] = cur.q[i];
          std::fprintf(stderr, "[node] FABEL slew ACTIVE at phase_t=%.2fs "
                               "(seeded at measured pose)\n", phase_t);
        }
        if (!slew_init) {
          // armed, waiting for motion: leave targets untouched
        } else
        if (phase_t - slew_t0 < slew_sec) {
          // ENVELOPE clamp (not a rate limiter): targets confined to a band
          // around the SEED pose that grows at slew_rate rad/s. Fast small
          // balance corrections are free inside the band; the 0.3-rad
          // first-tug step is blocked; fully open once the band exceeds the
          // plan's excursion (a pure rate limiter also throttled the
          // CORRECTIONS -> open-loop statue -> passive topple in 2 s).
          const double env = slew_rate * (phase_t - slew_t0) + 0.05;
          for (int i = 0; i < cfg.nu; i++) {
            double lo = slew_q[i] - env, hi = slew_q[i] + env;
            tgt_q[i] = std::fmin(hi, std::fmax(lo, tgt_q[i]));
          }
        }
      }
    }

    // ---- FABEL STREAM (2026-07-07, env H12_STREAM=1): 20 Hz machine-parse
    // line of exactly what the planner consumed and emitted this tick --
    // for the tick-by-tick diff against the in-process probe (the two
    // converge to different actions on the same believed state; the first
    // diverging field is the bug's address). Unset = silent.
    {
      static int fs_on = -1;
      static int fs_i = 0;
      if (fs_on < 0) {
        const char* e = std::getenv("H12_STREAM");
        fs_on = (e && e[0] == '1') ? 1 : 0;
      }
      if (fs_on && !warming && (++fs_i % 10 == 0)) {
        const mjpc::Trajectory* bt = g_agent.ActivePlanner().BestTrajectory();
        std::fprintf(stderr, "FSCOST %.5g\n",
                     bt ? bt->total_return : -1.0);
        std::fprintf(stderr,
                     "FS t=%.3f z=%.4f R8=%.5f vx=%.4f vy=%.4f gy=%.4f "
                     "q1=%.4f dq1=%.4f q3=%.4f q4=%.4f "
                     "a1=%.4f a3=%.4f a4=%.4f tg1=%.4f\n",
                     twin_time, meas_base_z, meas_R8, fd_vel[0], fd_vel[1],
                     cur.gyro[1], cur.q[1], cur.dq[1], cur.q[3], cur.q[4],
                     action[1], action[3], action[4], tgt_q[1]);
      }
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

    // build unitree_hg LowCmd_
    LowCmd cmd{};
    cmd.mode_pr() = 0;                       // PR mode
    cmd.mode_machine() = cur.mode_machine;   // echo (required by the real robot)
    for (int i = 0; i < cfg.nu; i++) {
      auto& mc = cmd.motor_cmd().at(i);
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
    for (int i = 0; i < cfg.nu; i++)
      last_cmd_q[i] = tgt_q[i];
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
        // FABEL discriminator (2026-07-07): raw planner action vs post-ramp
        // target vs measured, hipP (1) + knee (3) + ankle pitch (4).
        // Separates "the planner COMMANDS the fold" (action tracks the fold)
        // from "the deploy transforms corrupt a good action" (action at stand,
        // tgt/q folding).
        std::fprintf(stderr,
                     "[node]        FABEL act/tgt/q  hipP %+.2f/%+.2f/%+.2f  "
                     "knee %+.2f/%+.2f/%+.2f  ankP %+.2f/%+.2f/%+.2f\n",
                     action[1], tgt_q[1], cur.q[1],
                     action[3], tgt_q[3], cur.q[3],
                     action[4], tgt_q[4], cur.q[4]);
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
      if (adr_rh == -2) {
        auto SA = [&](const char* nm) {
          int id = mj_name2id(g_model, mjOBJ_SENSOR, nm);
          return id >= 0 ? g_model->sensor_adr[id] : -1;
        };
        adr_rh = SA("right_hand_pos"); adr_tp = SA("torso_position");
        adr_flp = SA("foot_left_pos"); adr_frp = SA("foot_right_pos");
        adr_com = SA("torso_subcom");
      }
      if (adr_rh >= 0 && adr_tp >= 0 && adr_com >= 0 && adr_flp >= 0 && adr_frp >= 0) {
        double rhand_fwd = sd->sensordata[adr_rh] - sd->sensordata[adr_tp];
        double midfoot_x = 0.5 * (sd->sensordata[adr_flp] + sd->sensordata[adr_frp]);
        double com_margin = sd->sensordata[adr_com] - midfoot_x;
        std::fprintf(stderr,
                     "[node]        Rhand_fwd=%+.2fm  CoM_margin=%+.3fm "
                     "(+=CoM ahead of feet=fwd-tip risk)\n",
                     rhand_fwd, com_margin);
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
                 "[B0] joint   trackRMSE trackMax |tau|mean |tau|peak  limit  peak%%  sat%%\n");
    std::fprintf(stderr,
                 "[B0]          (deg)    (deg)     (Nm)     (Nm)      (Nm)\n");
    int n_over = 0;
    for (int i = 0; i < cfg.nu; i++) {
      double pk = m_tau_max[i], lim = cfg.tau_limit[i];
      double pkpct = 100.0 * pk / lim, satpct = 100.0 * m_sat_count[i] * inv;
      if (pk > lim) n_over++;
      std::fprintf(stderr, "[B0] %-7s %8.2f %8.2f %8.1f %9.1f %6.0f %6.0f %5.1f%s\n",
                   cfg.joint_names[i], std::sqrt(m_err_sq[i] * inv) * 57.29577951308232,
                   m_err_max[i] * 57.29577951308232, m_tau_sum[i] * inv, pk, lim, pkpct, satpct,
                   pk > lim ? "  <<OVER-LIMIT" : "");
    }
    std::fprintf(stderr,
                 "[B0] torque headroom: %d/%d joints exceed the real H1-2 limit%s\n",
                 n_over, cfg.nu,
                 n_over == 0 ? " -> torque-safe for hardware" : " <-- ADDRESS before deploy");
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
