// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/planners/random_shooting/planner.h"

#include <mutex>
#include <shared_mutex>
#include <vector>

#include "mjpc/planners/sampling/planner.h"
#include "mjpc/threadpool.h"

namespace mjpc {

int RandomShootingPlanner::OptimizePolicyCandidates(int ncandidates,
                                                    int horizon,
                                                    ThreadPool& pool) {
  // Drop the incumbent. SamplingPlanner::OptimizePolicyCandidates starts by
  // calling ResamplePolicy, which reads `policy` into parameters_scratch, and
  // AddNoiseToPolicy then perturbs around that. Clearing the plan first makes
  // the spline evaluate to zero everywhere, so this iteration's samples are
  // drawn around zero control instead of around the previous winner.
  //
  // Candidate 0 is left un-noised by the base class, so zero control is always
  // in the sample set and the planner cannot do worse than doing nothing.
  {
    const std::unique_lock<std::shared_mutex> lock(mtx_);
    std::vector<double> zero(model->nu, 0.0);
    policy.Reset(horizon, zero.data());
  }

  return SamplingPlanner::OptimizePolicyCandidates(ncandidates, horizon, pool);
}

}  // namespace mjpc
