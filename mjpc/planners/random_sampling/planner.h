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

#ifndef MJPC_PLANNERS_RANDOM_SAMPLING_PLANNER_H_
#define MJPC_PLANNERS_RANDOM_SAMPLING_PLANNER_H_

#include "mjpc/planners/sampling/planner.h"
#include "mjpc/threadpool.h"

namespace mjpc {

// Memoryless random sampling: the control-baseline for every sampling planner
// in this repo.
//
// MJPC's SamplingPlanner is Predictive Sampling, not MPPI -- it takes the
// single best candidate (planner.cc CopyCandidateToPolicy) rather than an
// exponentially-weighted average -- and it warm-starts each iteration from the
// previous winner, so its samples concentrate wherever search has already been.
// Cross Entropy, PSO and Annealed Sampling all add machinery on top of that
// same warm start. Comparing them only to each other cannot say whether the
// machinery earns its cost or whether the warm start is doing the work.
//
// This planner removes exactly one thing: it discards the incumbent before
// each iteration, so every rollout is drawn afresh around zero control. Any
// planner that fails to beat it is not searching, it is coasting on its own
// previous answer.
class RandomSamplingPlanner : public SamplingPlanner {
 public:
  RandomSamplingPlanner() = default;
  ~RandomSamplingPlanner() override = default;

  // discard the incumbent, then run one ordinary sampling iteration
  int OptimizePolicyCandidates(int ncandidates, int horizon,
                               ThreadPool& pool) override;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_RANDOM_SAMPLING_PLANNER_H_
