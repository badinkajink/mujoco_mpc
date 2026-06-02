// MJPC embedded DDS control node for the Unitree H1-2 (digital twin + real robot).
// See README_EMBED.md.  Build: cmake .. -DMJPC_BUILD_DEPLOY=ON  (needs unitree_sdk2).
//
// WHY THIS EXISTS
//   The Python gRPC bridge proved the control logic (every free-standing simple
//   task balances the full-weight torque twin with CEM once the planning model is
//   gravcomp=0 and gravity is fed back as joint tau).  But each gRPC GetAction
//   round-trips and starves the planner (~65x slower than realtime).  Embedding
//   mjpc::Agent in-process and running its planner continuously in a background
//   thread removes that wall -- exactly how mjpc/app.cc's GUI runs full-rate.
//
// ARCHITECTURE (mirrors app.cc: a continuous planner thread + a control thread)
//   planner thread : g_agent.Plan(exit, uiload)   -- replans forever on the latest state
//   control thread : @ctrl_hz  read DDS state -> SetState -> ActionFromPolicy
//                    -> q* + gravity-FF tau -> unitree_hg LowCmd_ -> rt/safety/lowcmd_in
//   (feed + control share one thread, like app.cc's physics thread, so we never
//    race a half-written g_agent.state between SetState and ActionFromPolicy.)
//
// STATE  pelvis (free-joint, qpos[0:7]) is backed out of the reported IMU-site
//   pose:  base_p = site_p - R(quat)*IMU_OFFSET ;  base_v = site_v - (R*gyro) x roff.
//   Identical math to mjpc_dds_bridge.py:pelvis_from_site (unit-tested vs ground truth).

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <mujoco/mujoco.h>

#include "mjpc/agent.h"
#include "mjpc/task.h"
#include "mjpc/tasks/tasks.h"

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

using unitree::robot::ChannelFactory;
using unitree::robot::ChannelPublisher;
using unitree::robot::ChannelPublisherPtr;
using unitree::robot::ChannelSubscriber;
using unitree::robot::ChannelSubscriberPtr;
using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;
using SportState = unitree_go::msg::dds_::SportModeState_;

ABSL_FLAG(std::string, task, "Lean H12", "MJPC task id");
ABSL_FLAG(int, strategy, 6,
          "Lean Strategy parameter (6=stand 8=crouch 11=arms_overhead 13=lean_left ...)");
ABSL_FLAG(double, gravity_ff, 0.85,
          "joint gravity feedforward scale (tau = scale * qfrc_bias); 0 disables");
ABSL_FLAG(double, ctrl_hz, 200.0, "control / publish rate (Hz)");
ABSL_FLAG(double, warmup_sec, 1.0,
          "seconds to converge the planner while HOLDING the measured pose before releasing the policy");
ABSL_FLAG(std::string, lowcmd_topic, "rt/safety/lowcmd_in",
          "DDS LowCmd output topic (through the safety layer)");
ABSL_FLAG(std::string, lowstate_topic, "rt/lowstate", "DDS LowState input topic");
ABSL_FLAG(std::string, sportstate_topic, "rt/sportmodestate",
          "DDS SportModeState input topic (IMU-site world pose)");
ABSL_FLAG(int, domain_id, 0, "DDS domain id");
ABSL_FLAG(std::string, network_interface, "",
          "DDS network interface (empty = local/loopback for the twin; e.g. 'eth0' for the robot)");
#ifdef H12_NODE_GRPC
ABSL_FLAG(int, grpc_port, 10000,
          "if >0, host an MJPC gRPC server on this port so the monitor can attach "
          "(view state + switch Strategy live); 0 disables");
#endif

namespace {
constexpr int kNU = 27;  // actuated joints on the handless H1-2
// Per-joint gains == h1_2_modified actuator classes == real LowCmd kp/kd.
// (Must match mjpc_dds_bridge.py / _lockstep_capability.py.)
const double KP[kNU] = {150, 200, 200, 200, 80, 80,  150, 200, 200, 200, 80, 80,  200,
                        40, 40, 40, 40, 40, 40, 40,   40, 40, 40, 40, 40, 40, 40};
const double KV[kNU] = {5, 5, 5, 5, 4, 4,  5, 5, 5, 5, 4, 4,  5,
                        10, 10, 10, 10, 2, 2, 2,  10, 10, 10, 10, 2, 2, 2};
// short joint names (qpos[7..33] order) for the Title-5 baseline (B0) report.
const char* const JOINT_NAMES[kNU] = {
    "LhipY", "LhipP", "LhipR", "Lknee", "LankP", "LankR",
    "RhipY", "RhipP", "RhipR", "Rknee", "RankP", "RankR", "torso",
    "LshP", "LshR", "LshY", "Lelb", "LwrR", "LwrP", "LwrY",
    "RshP", "RshR", "RshY", "Relb", "RwrR", "RwrP", "RwrY"};
// Real H1-2 joint torque limits (Nm) from h12-lab-docs/docs/specs.md motor table,
// in the qpos[7..33] order. The TWIN motors are UNLIMITED, so this is the real-robot
// ceiling the sim must respect — checked against B0 |tau| per joint.
//   legs: hipY 200, hipP/hipR/knee 300, ankle 75; torso 200;
//   arms: shoulderP/R 120, shoulderY 75, elbow 120, wrist 25.
const double TAU_LIMIT[kNU] = {200, 300, 300, 300, 75, 75,
                               200, 300, 300, 300, 75, 75, 200,
                               120, 120, 75, 120, 25, 25, 25,
                               120, 120, 75, 120, 25, 25, 25};
// IMU site position in the pelvis (free-joint) frame, from h1_2_handless.xml.
const double IMU_OFFSET[3] = {-0.04452, -0.01891, 0.27756};

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

// Plain, copyable snapshot of the latest robot state.
struct StateData {
  bool have_ls = false, have_ss = false;
  double q[kNU] = {0}, dq[kNU] = {0};
  double quat[4] = {1, 0, 0, 0}, gyro[3] = {0};  // rt/lowstate IMU (wxyz, body gyro)
  double site_p[3] = {0}, site_v[3] = {0};       // rt/sportmodestate (IMU-site world pose)
  uint8_t mode_machine = 0;
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

void residual_sensor_callback(const mjModel* m, mjData* d, int stage) {
  if (m == g_agent_model || m == g_model) {
    if (stage == mjSTAGE_ACC) {
      g_task->Residual(m, d, d->sensordata);
    }
  }
}

void on_signal(int) { g_exit.store(true); }

// Headless MuJoCo error/warning handlers (the defaults block on a "Press Enter" prompt).
void FatalMjuError(const char* msg) {
  std::fprintf(stderr, "[mju_error] %s\n", msg);
  std::exit(1);
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
    return grpc_agent_util::GetMetrics(request, agent_, model_, data_, response);
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

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const std::string task_id = absl::GetFlag(FLAGS_task);
  const double gff = absl::GetFlag(FLAGS_gravity_ff);
  const double ctrl_hz = absl::GetFlag(FLAGS_ctrl_hz);
  const double ctrl_dt = 1.0 / ctrl_hz;
  const double warmup_sec = absl::GetFlag(FLAGS_warmup_sec);
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  mju_user_error = FatalMjuError;      // headless: log + exit instead of blocking on getchar
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
  g_agent.Initialize(lm.model.get());
  g_agent.Allocate();
  g_agent.Reset();
  g_task = g_agent.ActiveTask();
  g_agent_model = g_agent.GetModel();
  g_model = mj_copyModel(nullptr, g_agent_model);
  mjData* data = mj_makeData(g_model);
  int home = mj_name2id(g_model, mjOBJ_KEY, "home");
  if (home >= 0) mj_resetDataKeyframe(g_model, data, home);
  mjcb_sensor = residual_sensor_callback;
  g_agent.SetState(data);
  g_agent.plan_enabled = true;
  g_agent.action_enabled = true;
  g_agent.SetParamByName("residual_Strategy", absl::GetFlag(FLAGS_strategy));

  const int nq = g_model->nq, nv = g_model->nv, nu = g_model->nu;
  const int nact = nu < kNU ? nu : kNU;
  std::fprintf(stderr,
               "[node] task='%s' nq=%d nv=%d nu=%d strategy=%d gravity_ff=%.2f ctrl_hz=%.0f\n",
               task_id.c_str(), nq, nv, nu, absl::GetFlag(FLAGS_strategy), gff, ctrl_hz);

#ifdef H12_NODE_GRPC
  // ---- MJPC gRPC server (started early so the monitor can attach anytime) ----
  std::unique_ptr<NodeAgentService> grpc_service;
  std::unique_ptr<grpc::Server> grpc_server;
  const int grpc_port = absl::GetFlag(FLAGS_grpc_port);
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
  ChannelFactory::Instance()->Init(absl::GetFlag(FLAGS_domain_id),
                                   absl::GetFlag(FLAGS_network_interface));
  RobotState rs;

  ChannelSubscriberPtr<LowState> ls_sub(
      new ChannelSubscriber<LowState>(absl::GetFlag(FLAGS_lowstate_topic)));
  ls_sub->InitChannel(
      [&rs](const void* msg) {
        const LowState* s = static_cast<const LowState*>(msg);
        std::lock_guard<std::mutex> lk(rs.mu);
        for (int i = 0; i < kNU; i++) {
          rs.d.q[i] = s->motor_state().at(i).q();
          rs.d.dq[i] = s->motor_state().at(i).dq();
        }
        for (int k = 0; k < 4; k++) rs.d.quat[k] = s->imu_state().quaternion().at(k);
        for (int k = 0; k < 3; k++) rs.d.gyro[k] = s->imu_state().gyroscope().at(k);
        rs.d.mode_machine = s->mode_machine();
        rs.d.have_ls = true;
      },
      10);

  ChannelSubscriberPtr<SportState> ss_sub(
      new ChannelSubscriber<SportState>(absl::GetFlag(FLAGS_sportstate_topic)));
  ss_sub->InitChannel(
      [&rs](const void* msg) {
        const SportState* s = static_cast<const SportState*>(msg);
        std::lock_guard<std::mutex> lk(rs.mu);
        for (int k = 0; k < 3; k++) {
          rs.d.site_p[k] = s->position().at(k);
          rs.d.site_v[k] = s->velocity().at(k);
        }
        rs.d.have_ss = true;
      },
      10);

  ChannelPublisherPtr<LowCmd> cmd_pub(
      new ChannelPublisher<LowCmd>(absl::GetFlag(FLAGS_lowcmd_topic)));
  cmd_pub->InitChannel();

  // ---- wait for the first full state ----
  std::fprintf(stderr, "[node] waiting for %s + %s ...\n",
               absl::GetFlag(FLAGS_lowstate_topic).c_str(),
               absl::GetFlag(FLAGS_sportstate_topic).c_str());
  while (!g_exit.load()) {
    {
      std::lock_guard<std::mutex> lk(rs.mu);
      if (rs.d.have_ls && rs.d.have_ss) break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  std::fprintf(stderr, "[node] state stream up -> starting continuous planner.\n");

  // ---- continuous planner thread (the throughput fix) ----
  std::atomic<bool> plan_exit{false};
  std::atomic<int> uiload{0};
  std::thread planner([&] { g_agent.Plan(plan_exit, uiload); });

  // ---- live strategy switch via stdin: type a number 0-16 (+Enter), q=quit ----
  std::fprintf(stderr, "[node] live switch ready: type a strategy number 0-16 + Enter (q=quit)\n");
  std::thread stdin_thread([&] {
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line == "q" || line == "quit") { g_exit.store(true); break; }
      if (line.empty()) continue;
      try {
        int s = std::stoi(line);
        g_agent.SetParamByName("residual_Strategy", static_cast<double>(s));
        std::fprintf(stderr, "[node] >>> Strategy -> %d (task eases into the new pose)\n", s);
      } catch (...) {
        std::fprintf(stderr, "[node] (enter a strategy number 0-16, or q to quit)\n");
      }
    }
  });
  stdin_thread.detach();

  // scratch mjData: sd = real state for SetState; gd = qvel-zeroed for gravity FF.
  // Object/task slots beyond the robot (qpos[34:], qvel[33:]) keep home defaults.
  mjData* sd = mj_makeData(g_model);
  mjData* gd = mj_makeData(g_model);
  if (home >= 0) {
    mj_resetDataKeyframe(g_model, sd, home);
    mj_resetDataKeyframe(g_model, gd, home);
  }

  auto fill_state = [&](const StateData& cur) {
    double roff[3];
    QuatRot(cur.quat, IMU_OFFSET, roff);
    double ww[3];
    QuatRot(cur.quat, cur.gyro, ww);
    double cr[3] = {ww[1] * roff[2] - ww[2] * roff[1], ww[2] * roff[0] - ww[0] * roff[2],
                    ww[0] * roff[1] - ww[1] * roff[0]};
    for (int k = 0; k < 3; k++) sd->qpos[k] = cur.site_p[k] - roff[k];  // pelvis = site - R*offset
    for (int k = 0; k < 4; k++) sd->qpos[3 + k] = cur.quat[k];
    for (int i = 0; i < kNU; i++) sd->qpos[7 + i] = cur.q[i];
    for (int k = 0; k < 3; k++) sd->qvel[k] = cur.site_v[k] - cr[k];  // pelvis world linvel
    for (int k = 0; k < 3; k++) sd->qvel[3 + k] = cur.gyro[k];        // free-joint angvel == body gyro
    for (int i = 0; i < kNU; i++) sd->qvel[6 + i] = cur.dq[i];
  };

  std::vector<double> action(nu, 0.0);

  // ---- Title-5 baseline (B0) accumulators: tracking error, applied torque, balance ----
  // (policy phase only; printed as a summary on exit. The SAME node runs on the real
  //  robot -> per-row sim2real delta.)
  long m_ticks = 0;
  double m_err_sum[kNU] = {0}, m_err_sq[kNU] = {0}, m_err_max[kNU] = {0};
  double m_tau_sum[kNU] = {0}, m_tau_max[kNU] = {0};
  double m_z_sum = 0, m_z_sq = 0, m_z_min = 1e9, m_tilt_sum = 0, m_tilt_max = 0;
  long m_sat_count[kNU] = {0};   // ticks where |tau| exceeds the real H1-2 limit
  bool m_sat_warned = false;

  // ---- control loop @ ctrl_hz (mirrors app.cc physics thread) ----
  auto t0 = std::chrono::steady_clock::now();
  auto next = t0;
  long ticks = 0;
  while (!g_exit.load()) {
    StateData cur;
    {
      std::lock_guard<std::mutex> lk(rs.mu);
      cur = rs.d;
    }
    fill_state(cur);
    mj_forward(g_model, sd);
    g_agent.SetState(sd);

    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    bool warming = wall < warmup_sec;

    // policy: 27 target joint positions (rad). Held at measured q during warmup.
    if (!warming) {
      g_agent.ActivePlanner().ActionFromPolicy(action.data(), g_agent.state.state().data(),
                                             g_agent.state.time());
    }

    // gravity feedforward: tau = gff * qfrc_bias evaluated at qvel = 0.
    double tau[kNU] = {0};
    if (gff != 0.0) {
      mju_copy(gd->qpos, sd->qpos, nq);
      mju_zero(gd->qvel, nv);
      mj_forward(g_model, gd);
      for (int i = 0; i < kNU; i++) tau[i] = gff * gd->qfrc_bias[6 + i];
    }

    // build unitree_hg LowCmd_
    LowCmd cmd{};
    cmd.mode_pr() = 0;                       // PR mode
    cmd.mode_machine() = cur.mode_machine;   // echo (required by the real robot)
    for (int i = 0; i < kNU; i++) {
      auto& mc = cmd.motor_cmd().at(i);
      mc.mode() = 1;  // 1 = enable
      mc.q() = (warming || i >= nact) ? static_cast<float>(cur.q[i]) : static_cast<float>(action[i]);
      mc.dq() = 0.0f;
      mc.tau() = static_cast<float>(tau[i]);
      mc.kp() = static_cast<float>(KP[i]);
      mc.kd() = static_cast<float>(KV[i]);
    }
    cmd.crc() = Crc32Core(reinterpret_cast<uint32_t*>(&cmd), (sizeof(LowCmd) >> 2) - 1);
    cmd_pub->Write(cmd);

    // ---- per-tick metrics (B0 baseline; balance every tick, accumulate in policy) ----
    double base_z = sd->qpos[2];
    double R[9];
    mju_quat2Mat(R, sd->qpos + 3);
    double tilt = std::acos(std::fmax(-1.0, std::fmin(1.0, R[8]))) * 57.29577951308232;
    if (!warming) {
      m_ticks++;
      m_z_sum += base_z;
      m_z_sq += base_z * base_z;
      if (base_z < m_z_min) m_z_min = base_z;
      m_tilt_sum += tilt;
      if (tilt > m_tilt_max) m_tilt_max = tilt;
      for (int i = 0; i < kNU; i++) {
        double cmd_q = (i < nact) ? action[i] : cur.q[i];
        double err = cmd_q - cur.q[i];                              // commanded - measured
        double ae = std::fabs(err);
        m_err_sum[i] += ae;
        m_err_sq[i] += err * err;
        if (ae > m_err_max[i]) m_err_max[i] = ae;
        // torque the onboard/twin PD actually applies (matches SimInterface law)
        double total_tau = tau[i] + KP[i] * err + KV[i] * (0.0 - cur.dq[i]);
        double at = std::fabs(total_tau);
        m_tau_sum[i] += at;
        if (at > m_tau_max[i]) m_tau_max[i] = at;
        if (at > TAU_LIMIT[i]) {
          m_sat_count[i]++;
          if (!m_sat_warned) {
            std::fprintf(stderr,
                         "[node] WARNING: torque on %s = %.0f Nm > real H1-2 limit %.0f Nm "
                         "(twin is unlimited; real robot would saturate)\n",
                         JOINT_NAMES[i], at, TAU_LIMIT[i]);
            m_sat_warned = true;
          }
        }
      }
    }
    if (++ticks % static_cast<long>(ctrl_hz) == 0) {
      std::fprintf(stderr, "[node] t=%5.1fs %s base_z=%.3f tilt=%4.1fdeg knee(L/R)=%+.2f/%+.2f\n",
                   wall, warming ? "WARMUP-hold" : "policy", base_z, tilt,
                   sd->qpos[10], sd->qpos[16]);
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
                 absl::GetFlag(FLAGS_strategy), m_ticks, m_ticks * ctrl_dt, ctrl_hz);
    std::fprintf(stderr,
                 "[B0] base_z mean=%.3f sd=%.3f min=%.3f m | tilt mean=%.1f max=%.1f deg\n",
                 zmean, zsd, m_z_min, m_tilt_sum * inv, m_tilt_max);
    std::fprintf(stderr,
                 "[B0] joint   trackRMSE trackMax |tau|mean |tau|peak  limit  peak%%  sat%%\n");
    std::fprintf(stderr,
                 "[B0]          (deg)    (deg)     (Nm)     (Nm)      (Nm)\n");
    int n_over = 0;
    for (int i = 0; i < kNU; i++) {
      double pk = m_tau_max[i], lim = TAU_LIMIT[i];
      double pkpct = 100.0 * pk / lim, satpct = 100.0 * m_sat_count[i] * inv;
      if (pk > lim) n_over++;
      std::fprintf(stderr, "[B0] %-7s %8.2f %8.2f %8.1f %9.1f %6.0f %6.0f %5.1f%s\n",
                   JOINT_NAMES[i], std::sqrt(m_err_sq[i] * inv) * 57.29577951308232,
                   m_err_max[i] * 57.29577951308232, m_tau_sum[i] * inv, pk, lim, pkpct, satpct,
                   pk > lim ? "  <<OVER-LIMIT" : "");
    }
    std::fprintf(stderr,
                 "[B0] torque headroom: %d/%d joints exceed the real H1-2 limit%s\n",
                 n_over, kNU,
                 n_over == 0 ? " -> torque-safe for hardware" : " <-- ADDRESS before deploy");
    std::fprintf(stderr, "[B0] (run the SAME node on the real robot -> per-row sim2real delta)\n");
  }
#ifdef H12_NODE_GRPC
  if (grpc_server) grpc_server->Shutdown();
#endif
  plan_exit.store(true);
  if (planner.joinable()) planner.join();
  mj_deleteData(gd);
  mj_deleteData(sd);
  mj_deleteData(data);
  mj_deleteModel(g_model);
  return 0;
}
