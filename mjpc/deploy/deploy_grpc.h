// Optional in-process MJPC gRPC monitor for the deploy nodes (stage 3a of the
// 2026-07-18 reorg; formerly internal to deploy_common.cc). Compiled only when
// H12_NODE_GRPC is defined (fork builds with MJPC_BUILD_GRPC_SERVICE; the
// production colcon cores never define it). Header-only.

#ifndef MJPC_DEPLOY_DEPLOY_GRPC_H_
#define MJPC_DEPLOY_DEPLOY_GRPC_H_

#ifdef H12_NODE_GRPC

#include <mutex>

#include <mujoco/mujoco.h>

#include "mjpc/agent.h"
#include "mjpc/grpc/agent.grpc.pb.h"
#include "mjpc/grpc/agent.pb.h"
#include "mjpc/grpc/grpc_agent_util.h"
#include "mjpc/states/state.h"

namespace h12deploy {

// gRPC service over the node's LIVE agent so the existing MJPC monitor can
// attach. READ-ONLY reflection (state/action/residuals/costs/metrics/params)
// delegates to the same grpc_agent_util helpers agent_server uses; the only
// mutator exposed is SetTaskParameters (the live Strategy switch / the upper
// node's goal ingest). All state/planner-driving RPCs are inert no-ops so the
// monitor (or a stray button) can never disturb the node's own control loop.
// A mutex serialises concurrent gRPC calls on the service's scratch mjData.
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

}  // namespace h12deploy

#endif  // H12_NODE_GRPC

#endif  // MJPC_DEPLOY_DEPLOY_GRPC_H_
