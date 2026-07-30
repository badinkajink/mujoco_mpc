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

#ifndef MJPC_TASKS_TRIPLE_PENDULUM_CARTPOLE_TRIPLE_PENDULUM_CARTPOLE_H_
#define MJPC_TASKS_TRIPLE_PENDULUM_CARTPOLE_TRIPLE_PENDULUM_CARTPOLE_H_

#include <memory>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {

// Triple pendulum on a cart threading a corridor of two disk obstacles.
//
// Benchmark system from Caldwell & Correll, "Fast Sample-Based Planning for
// Dynamic Systems by Zero-Control Linearization-Based Steering", ISRR 2015,
// Section 5. See task.xml for the physical parameters and the geometry of the
// obstacle corridor.
//
// The system is underactuated (nu = 1 force on the cart, nv = 4) and the
// 3-link pendulum is chaotic. The obstacle gap (0.5 m) is narrower than the
// pendulum (1.0 m), so the pendulum must be swung down, laid out through the
// corridor, and re-erected on the far side.
class TriplePendulumCartpole : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;

  class ResidualFn : public BaseResidualFn {
   public:
    explicit ResidualFn(const TriplePendulumCartpole* task)
        : BaseResidualFn(task) {}

    // ------- Residuals for triple pendulum cartpole ------
    //   Number of residuals: 15
    //     Residual (0):     cart position error to the goal
    //     Residual (1-3):   per-link deviation from upright, cos(theta_i) - 1
    //     Residual (4-7):   joint velocities
    //     Residual (8):     control
    //     Residual (9-14):  obstacle avoidance, 3 link heads x 2 obstacles
    // -----------------------------------------------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;

   private:
    friend class TriplePendulumCartpole;

    // number of pendulum links, and number of disk obstacles
    static constexpr int kNumLinks = 3;
    static constexpr int kNumObstacles = 2;

    // site ids of the three link heads (end of link 1, 2, 3)
    int head_site_id_[kNumLinks] = {-1, -1, -1};
    // geom ids and radii of the disk obstacles
    int obstacle_geom_id_[kNumObstacles] = {-1, -1};
    double obstacle_radius_[kNumObstacles] = {0.0, 0.0};
  };

  TriplePendulumCartpole() : residual_(this) {}

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(residual_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }
  void ResetLocked(const mjModel* model) override;

 private:
  ResidualFn residual_;
};

}  // namespace mjpc

#endif  // MJPC_TASKS_TRIPLE_PENDULUM_CARTPOLE_TRIPLE_PENDULUM_CARTPOLE_H_
