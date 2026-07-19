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

#ifndef MJPC_TASKS_UR5_UR5_H_
#define MJPC_TASKS_UR5_UR5_H_

#include <memory>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
class UR5 : public Task {
 public:
  std::string Name() const override = 0;
  std::string XmlPath() const override = 0;

  class ResidualFn : public BaseResidualFn {
   public:
    explicit ResidualFn(const UR5 *task) : BaseResidualFn(task) {}
    void Residual(const mjModel *model, const mjData *data,
                  double *residual) const override;
  };
  UR5() : residual_(this) {}

  // On bring success (mean box-site-to-target distance < 1.5 cm):
  // re-randomize the box position and the mocap target pose, open the gripper.
  void TransitionLocked(mjModel *model, mjData *data) override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn *InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};

class UR5_Magpie : public UR5 {
 public:
  std::string XmlPath() const override{
    return GetModelPath("ur5/task_magpie.xml");
  }
  std::string Name() const override{ return "UR5 Magpie"; }
};

class UR5_Inspire : public UR5 {
 public:
  std::string XmlPath() const override{
    return GetModelPath("ur5/task_inspire.xml");
  }
  std::string Name() const override{ return "UR5 Inspire"; }
};

}  // namespace mjpc
#endif  // MJPC_TASKS_UR5_UR5_H_
