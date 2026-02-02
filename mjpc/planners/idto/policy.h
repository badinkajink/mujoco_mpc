// Copyright 2024 DeepMind Technologies Limited
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

#ifndef MJPC_PLANNERS_IDTO_POLICY_H_
#define MJPC_PLANNERS_IDTO_POLICY_H_

#include <vector>

#include "mjpc/trajectory.h"

namespace mjpc {

// IDTO policy representation
// Stores the optimized trajectory as a time-indexed sequence of states and
// controls
class IDTOPolicy {
 public:
  // Constructor
  IDTOPolicy() = default;

  // Destructor
  ~IDTOPolicy() = default;

  // Copy from trajectory
  void CopyFrom(const Trajectory& trajectory);

  // Representation
  Trajectory trajectory;

  // Nominal configuration trajectory q[0:T]
  std::vector<double> configuration;

  // Nominal velocity trajectory v[0:T]
  std::vector<double> velocity;

  // Nominal force trajectory tau[0:T-1]
  std::vector<double> force;

  // Times corresponding to each timestep
  std::vector<double> times;

  // Timestep
  double timestep;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_IDTO_POLICY_H_
