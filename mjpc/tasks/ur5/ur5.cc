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

#include "mjpc/tasks/ur5/ur5.h"

#include <string>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
// ------- Residuals for cube manipulation task ------
//     Cube position: (3)
//     Cube orientation: (3)
//     Cube linear velocity: (3)
//     Control: (16), there are 16 servos
//     Nominal pose: (16)
//     Joint velocity: (16)
// ------------------------------------------
void UR5::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                   double *residual) const {
  int counter = 0;

  // -- null residual test only
  // double *null = SensorByName(model, data, "null");
  // mju_copy(residual+counter, null, 0);
  // counter += 1;

  // reach
  double* hand = SensorByName(model, data, "hand");
  double* box = SensorByName(model, data, "box");
  mju_sub3(residual + counter, hand, box);
  counter += 3;

  // bring
  double* box1 = SensorByName(model, data, "box1");
  double* target1 = SensorByName(model, data, "target1");
  mju_sub3(residual + counter, box1, target1);
  counter += 3;
  double* box2 = SensorByName(model, data, "box2");
  double* target2 = SensorByName(model, data, "target2");
  mju_sub3(residual + counter, box2, target2);
  counter += 3;

  // Sanity check
  CheckSensorDim(model, counter);
}

void UR5::TransitionLocked(mjModel *model, mjData *data) {
  double residuals[100];
  residual_.Residual(model, data, residuals);
  double bring_dist = (mju_norm3(residuals+3) + mju_norm3(residuals+6)) / 2;

  // reset:
  if (data->time > 0 && bring_dist < .015) {
    // box:
    absl::BitGen gen_;
    data->qpos[12] = absl::Uniform<double>(gen_, -.5, .5);
    data->qpos[13] = absl::Uniform<double>(gen_, -.5, .5);
    data->qpos[14] = .05;

    // target:
    data->mocap_pos[0] = absl::Uniform<double>(gen_, -.5, .5);
    data->mocap_pos[1] = absl::Uniform<double>(gen_, -.5, .5);
    data->mocap_pos[2] = absl::Uniform<double>(gen_, .03, 1);
    data->mocap_quat[0] = absl::Uniform<double>(gen_, -1, 1);
    data->mocap_quat[1] = absl::Uniform<double>(gen_, -1, 1);
    data->mocap_quat[2] = absl::Uniform<double>(gen_, -1, 1);
    data->mocap_quat[3] = absl::Uniform<double>(gen_, -1, 1);
    mju_normalize4(data->mocap_quat);

    // open gripper
    data->qpos[6] = 0.0;
    data->qpos[9] = 0.0;
  }
    // DEBUG
  // printf("\nJoint order for qpos:\n");
  // for (int i = 0; i < model->njnt; i++) {
  //   const char* jnt_name = mj_id2name(model, mjOBJ_JOINT, i);
  //   int qpos_adr = model->jnt_qposadr[i];
  //   printf("  Joint %d: %s (qpos index %d)\n", i, jnt_name ? jnt_name : "unnamed", qpos_adr);
  // }

}

}  // namespace mjpc
