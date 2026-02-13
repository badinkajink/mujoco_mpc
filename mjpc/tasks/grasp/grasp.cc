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

#include "mjpc/tasks/grasp/grasp.h"

#include <string>
#include <cmath>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {

// ------- Residuals for grasp synthesis task ------
//   Cost components:
//     1. Distance from end-effector to object CoM (reach): (3)
//     2. Contact count penalty (encourage contact): (1)
//     3. Contact quality (sum of normal forces): (1)
//     4. Object linear velocity (discourage motion): (3)
//     5. Object angular velocity: (3)
//     6. Control magnitude: (num_actuators)
//     7. Action smoothness: (num_actuators)
// --------------------------------------------------

int GraspSynthesis::ResidualFn::CountFingerContacts(
    const mjModel *model, const mjData *data, int box_geom_id) const {
  int contact_count = 0;

  // Define finger body name patterns
  const char* finger_patterns[] = {
    "thumb_", "index_", "middle_", "ring_", "pinky_"
  };
  const int num_patterns = 5;

  // Scan all active contacts
  for (int i = 0; i < data->ncon; i++) {
    mjContact* contact = &data->contact[i];
    int geom1 = contact->geom[0];
    int geom2 = contact->geom[1];

    // Check if one geom is the box
    int finger_geom = -1;
    if (geom1 == box_geom_id) {
      finger_geom = geom2;
    } else if (geom2 == box_geom_id) {
      finger_geom = geom1;
    } else {
      continue;  // Neither geom is the box
    }

    // Check if the other geom belongs to a finger
    int finger_body = model->geom_bodyid[finger_geom];
    const char* body_name = mj_id2name(model, mjOBJ_BODY, finger_body);

    if (body_name != nullptr) {
      for (int p = 0; p < num_patterns; p++) {
        if (std::strstr(body_name, finger_patterns[p]) != nullptr) {
          contact_count++;
          break;  // Count each contact once
        }
      }
    }
  }

  return contact_count;
}

double GraspSynthesis::ResidualFn::GetContactQuality(
    const mjModel *model, const mjData *data, int box_geom_id) const {
  double total_normal_force = 0.0;

  // Define finger body name patterns
  const char* finger_patterns[] = {
    "thumb_", "index_", "middle_", "ring_", "pinky_"
  };
  const int num_patterns = 5;

  // Temporary array for contact forces
  mjtNum force[6];

  // Scan all active contacts
  for (int i = 0; i < data->ncon; i++) {
    mjContact* contact = &data->contact[i];
    int geom1 = contact->geom[0];
    int geom2 = contact->geom[1];

    // Check if one geom is the box
    int finger_geom = -1;
    if (geom1 == box_geom_id) {
      finger_geom = geom2;
    } else if (geom2 == box_geom_id) {
      finger_geom = geom1;
    } else {
      continue;
    }

    // Check if the other geom belongs to a finger
    int finger_body = model->geom_bodyid[finger_geom];
    const char* body_name = mj_id2name(model, mjOBJ_BODY, finger_body);

    if (body_name != nullptr) {
      bool is_finger = false;
      for (int p = 0; p < num_patterns; p++) {
        if (std::strstr(body_name, finger_patterns[p]) != nullptr) {
          is_finger = true;
          break;
        }
      }

      if (is_finger) {
        // Get contact force (normal component is in force[0])
        mj_contactForce(model, data, i, force);
        total_normal_force += std::abs(force[0]);  // Normal force magnitude
      }
    }
  }

  return total_normal_force;
}

void GraspSynthesis::ResidualFn::Residual(
    const mjModel *model, const mjData *data, double *residual) const {
  int counter = 0;

  // Get object (box) geom ID
  int box_geom_id = mj_name2id(model, mjOBJ_GEOM, "box");
  if (box_geom_id < 0) {
    // Fallback: box not found, return zero residuals
    for (int i = 0; i < model->nsensordata; i++) {
      residual[i] = 0.0;
    }
    return;
  }

  // ----- Cost 1: Distance from end-effector to object CoM -----
  // This encourages the hand to reach towards the object
  double* eeff_pos = SensorByName(model, data, "hand");
  double* box_pos = SensorByName(model, data, "box");
  mju_sub3(residual + counter, eeff_pos, box_pos);
  counter += 3;

  // ----- Cost 2: Contact count penalty -----
  // We want to maximize contacts, so we penalize the negative of contact count
  // Target is 5 contacts (one per finger), so residual = (5 - num_contacts)
  int num_contacts = CountFingerContacts(model, data, box_geom_id);
  residual[counter] = 5.0 - static_cast<double>(num_contacts);
  counter += 1;

  // ----- Cost 3: Contact quality (force closure proxy) -----
  // Sum of normal forces - higher is better, so we negate it
  // We want high normal forces, so residual = -total_force (with some scaling)
  double contact_quality = GetContactQuality(model, data, box_geom_id);
  residual[counter] = -contact_quality;  // Negative because we want to maximize
  counter += 1;

  // ----- Cost 4: Object linear velocity -----
  // Discourage object motion during grasping
  // Get box body ID and extract velocity from qvel
  int box_body_id = mj_name2id(model, mjOBJ_BODY, "box");
  if (box_body_id >= 0) {
    // Get body velocity (first 3 components of body's qvel are linear velocity)
    int box_jnt_id = model->body_jntadr[box_body_id];
    if (box_jnt_id >= 0) {
      int qvel_adr = model->jnt_qposadr[box_jnt_id];
      // For freejoint: qvel has 6 components (3 linear, 3 angular)
      mju_copy3(residual + counter, data->qvel + qvel_adr);
    } else {
      mju_zero3(residual + counter);
    }
  } else {
    mju_zero3(residual + counter);
  }
  counter += 3;

  // ----- Cost 5: Object angular velocity -----
  if (box_body_id >= 0) {
    int box_jnt_id = model->body_jntadr[box_body_id];
    if (box_jnt_id >= 0) {
      int qvel_adr = model->jnt_qposadr[box_jnt_id];
      // Angular velocity is components 3-5 of freejoint qvel
      mju_copy3(residual + counter, data->qvel + qvel_adr + 3);
    } else {
      mju_zero3(residual + counter);
    }
  } else {
    mju_zero3(residual + counter);
  }
  counter += 3;

  // ----- Cost 6: Control magnitude -----
  // Penalize large control inputs
  for (int i = 0; i < model->nu; i++) {
    residual[counter] = data->ctrl[i];
    counter++;
  }

  // ----- Cost 7: Action smoothness (control rate) -----
  // Penalize rapid changes in control
  // For position actuators, compare ctrl to current joint position
  for (int i = 0; i < model->nu; i++) {
    // Get the joint associated with this actuator
    int jnt_id = model->actuator_trnid[2 * i];  // Actuator transmission joint ID
    if (jnt_id >= 0 && jnt_id < model->njnt) {
      int qpos_adr = model->jnt_qposadr[jnt_id];
      residual[counter] = data->ctrl[i] - data->qpos[qpos_adr];
    } else {
      residual[counter] = 0.0;
    }
    counter++;
  }

  // Sanity check
  CheckSensorDim(model, counter);
}

void GraspSynthesis::TransitionLocked(mjModel *model, mjData *data) {
  // Optional: Implement curriculum learning or randomization
  // For now, this is a placeholder

  // Example: Reset box position if it falls or grasp is achieved
  // You can add logic here to randomize box position/orientation
  // or to reset when force closure is achieved

  // Get box body
  int box_body_id = mj_name2id(model, mjOBJ_BODY, "box");
  if (box_body_id < 0) return;

  // Check if box has fallen (z < threshold)
  double box_z = data->xpos[box_body_id * 3 + 2];

  if (box_z < 0.05 && data->time > 1.0) {
    // Reset box to initial position
    int box_jnt_id = model->body_jntadr[box_body_id];
    if (box_jnt_id >= 0) {
      int qpos_adr = model->jnt_qposadr[box_jnt_id];
      // Reset position
      data->qpos[qpos_adr + 0] = 0.0;
      data->qpos[qpos_adr + 1] = 0.0;
      data->qpos[qpos_adr + 2] = 0.02;
      // Reset orientation to identity quaternion
      data->qpos[qpos_adr + 3] = 1.0;
      data->qpos[qpos_adr + 4] = 0.0;
      data->qpos[qpos_adr + 5] = 0.0;
      data->qpos[qpos_adr + 6] = 0.0;

      // Zero velocities
      int qvel_adr = qpos_adr;  // Same for freejoint
      for (int i = 0; i < 6; i++) {
        data->qvel[qvel_adr + i] = 0.0;
      }
    }
  }

}

}  // namespace mjpc
