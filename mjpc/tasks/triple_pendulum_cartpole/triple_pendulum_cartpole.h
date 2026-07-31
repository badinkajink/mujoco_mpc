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

  // Geometry of the obstacle corridor: which sites are the link heads, which
  // geoms are the disks, and how the clearance between them is measured.
  //
  // Both the Avoidance residual and the benchmark's collision metric need
  // this, and they have to agree. If one measured 3-D distance and the other
  // in-plane distance, a run could be scored as clear while being charged for
  // a collision, or the reverse. Keeping the names and the distance convention
  // in one place is what makes them agree by construction.
  struct Corridor {
    static constexpr int kNumLinks = 3;
    static constexpr int kNumObstacles = 2;

    // resolve site and geom ids against the model; errors if any are missing
    void Initialize(const mjModel* model);

    // Signed surface-to-surface clearance of link head `link` from disk
    // `obstacle`, in the x-z plane -- the disks are cylinders extruded along y,
    // so the out-of-plane distance is not part of the constraint.
    //
    // Both radii are subtracted, so zero means the sphere is exactly touching
    // the disk and negative is real overlap. That matters: measuring from the
    // head's centre instead would put the cost's idea of "touching" a full head
    // radius inside MuJoCo's, so the avoidance term would still read as
    // clearance while the contact solver was already pushing back.
    double Clearance(const mjData* data, int link, int obstacle) const;

    // smallest clearance over every (head, disk) pair
    double MinClearance(const mjData* data) const;

    // site ids and sphere radii of the three link heads (end of link 1, 2, 3)
    int head_site_id[kNumLinks] = {-1, -1, -1};
    double head_radius[kNumLinks] = {0.0, 0.0, 0.0};
    // geom ids and radii of the disk obstacles
    int obstacle_geom_id[kNumObstacles] = {-1, -1};
    double obstacle_radius[kNumObstacles] = {0.0, 0.0};
  };

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

    Corridor corridor_;
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
