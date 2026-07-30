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

#include "mjpc/tasks/triple_pendulum_cartpole/triple_pendulum_cartpole.h"

#include <cmath>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {

std::string TriplePendulumCartpole::XmlPath() const {
  return GetModelPath("triple_pendulum_cartpole/task.xml");
}
std::string TriplePendulumCartpole::Name() const {
  return "Triple Pendulum Cartpole";
}

void TriplePendulumCartpole::ResetLocked(const mjModel* model) {
  // link head sites, ordered base to tip
  const char* head_names[ResidualFn::kNumLinks] = {"head1", "head2", "tip"};
  for (int i = 0; i < ResidualFn::kNumLinks; i++) {
    residual_.head_site_id_[i] = mj_name2id(model, mjOBJ_SITE, head_names[i]);
    if (residual_.head_site_id_[i] < 0) {
      mju_error_s("site '%s' not found", head_names[i]);
    }
  }

  // disk obstacles: cylinders whose axis is along y, so geom_size[0] is the
  // disk radius in the x-z plane
  const char* obstacle_names[ResidualFn::kNumObstacles] = {"obstacle_upper",
                                                           "obstacle_lower"};
  for (int i = 0; i < ResidualFn::kNumObstacles; i++) {
    int id = mj_name2id(model, mjOBJ_GEOM, obstacle_names[i]);
    if (id < 0) mju_error_s("geom '%s' not found", obstacle_names[i]);
    residual_.obstacle_geom_id_[i] = id;
    residual_.obstacle_radius_[i] = model->geom_size[3 * id];
  }
}

// ------- Residuals for triple pendulum cartpole ------
//   Cart:      cart position should reach the goal (parameter "Goal")
//   Upright:   every link should end at theta_i = 0 (pointing up)
//   Velocity:  joint velocities should be small
//   Control:   control should be small
//   Avoidance: link heads should stay clear of the disk obstacles
// -----------------------------------------------------
void TriplePendulumCartpole::ResidualFn::Residual(const mjModel* model,
                                                 const mjData* data,
                                                 double* residual) const {
  int counter = 0;

  // ---------- Cart ----------
  // parameters_[0] is the goal cart position ("residual_Goal")
  residual[counter++] = data->qpos[0] - parameters_[0];

  // ---------- Upright ----------
  // theta_i = 0 points link i along +z. cos(theta) - 1 is zero exactly at the
  // upright equilibrium and is smooth across the +/-pi wrap, so the planner
  // never sees an artificial discontinuity from angle unwrapping.
  for (int i = 0; i < kNumLinks; i++) {
    residual[counter++] = std::cos(data->qpos[1 + i]) - 1.0;
  }

  // ---------- Velocity ----------
  for (int i = 0; i < model->nv; i++) {
    residual[counter++] = data->qvel[i];
  }

  // ---------- Control ----------
  for (int i = 0; i < model->nu; i++) {
    residual[counter++] = data->ctrl[i];
  }

  // ---------- Avoidance ----------
  // Hinge-loss clearance in the x-z plane: charge only once a head is within
  // (radius + clearance) of an obstacle centre, so the term is exactly zero
  // away from the corridor and does not bias the swing-up.
  //
  // The obstacles are also real collision geoms, so a plan that drives a head
  // into a disk is additionally penalised by the dynamics. This cost exists to
  // give the sampler a gradient *before* contact, which contact alone (a
  // discontinuity) does not provide.
  double clearance = parameters_[1];
  for (int i = 0; i < kNumLinks; i++) {
    const double* head = data->site_xpos + 3 * head_site_id_[i];
    for (int j = 0; j < kNumObstacles; j++) {
      const double* obstacle = data->geom_xpos + 3 * obstacle_geom_id_[j];
      // distance in the x-z plane (obstacle cylinders are extruded along y)
      double dx = head[0] - obstacle[0];
      double dz = head[2] - obstacle[2];
      double distance = std::sqrt(dx * dx + dz * dz);
      double threshold = obstacle_radius_[j] + clearance;
      residual[counter++] = std::max(0.0, threshold - distance);
    }
  }

  // sanity check: residual dimension must match the `user` sensors in task.xml
  CheckSensorDim(model, counter);
}

}  // namespace mjpc
