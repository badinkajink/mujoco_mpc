// Deploy telemetry helpers (stage 3a of the 2026-07-18 reorg; formerly
// internal to deploy_common.cc). The status lines / B0 report themselves still
// live in RunDeployNode -- loop-state-coupled, they move in stage 3b.

#ifndef MJPC_DEPLOY_DEPLOY_TELEMETRY_H_
#define MJPC_DEPLOY_DEPLOY_TELEMETRY_H_

#include <cstdint>
#include <string>

#include "mjpc/trajectory.h"

namespace h12deploy {

// Serialize a Trajectory's qpos rows to JSON for the debug plan topic.
// Hand-rolled rather than nlohmann: the colcon package that builds the cores
// does not put _deps/json-src/include on the include path, and appending ~4k
// doubles into one reserved buffer beats building a DOM. Emits qpos ONLY (the
// visualizer's ghost needs nothing else; halves the payload to ~90 KB).
//
// LANDMINE: traj->states is Allocate()d to kMaxTrajectoryHorizon (512) but
// only traj->horizon rows are valid -- Rollout overwrites `horizon` with the
// actual step count. Iterate horizon, never states.size().
void AppendPlanJson(const mjpc::Trajectory* traj, int nq, std::int64_t plan_iter,
                    std::string* out);

}  // namespace h12deploy

#endif  // MJPC_DEPLOY_DEPLOY_TELEMETRY_H_
