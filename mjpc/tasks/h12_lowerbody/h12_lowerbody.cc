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

#include "mjpc/tasks/h12_lowerbody/h12_lowerbody.h"

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

// Cosine similarity of the below->above segment with world +Z, minus 1: 0
// when `above` is directly above `below`, -1 when the segment is horizontal,
// -2 when inverted.
double UprightCosFromPoints(const double* below, const double* above) {
  double segment[3];
  mju_sub3(segment, above, below);
  double length = mju_norm3(segment);
  // Endpoints are rigid-link separated in this model, so a degenerate
  // segment can't occur physically; guard anyway rather than divide by ~0.
  if (length < 1.0e-6) return 0.0;
  return segment[2] / length - 1.0;
}

// Center of a foot's geometry: the ankle_roll body origin offset along the
// foot's forward (x) axis — the ankle joint sits 3.3 cm behind the foot mesh
// center (measured).
constexpr double kFootCenterOffset = 0.0334;  // foot frame, meters

void FootCenter(const mjModel* model, const mjData* data,
                const std::string& pos_sensor, const std::string& xaxis_sensor,
                double* center) {
  mju_addScl3(center, SensorByName(model, data, pos_sensor),
              SensorByName(model, data, xaxis_sensor), kFootCenterOffset);
}

// Height of a foot-geometry center above the sole at flat stance (measured
// at qpos0). Added to heights-above-stance-foot so they read as heights
// above ground.
constexpr double kFootCenterHeight = 0.0468;  // meters

// Height of the lower foot-geometry center: the kinematic stance reference.
// Heights measured against this (instead of world z) are observable on the
// real robot from encoders + IMU alone — absolute world z is not. The
// mju_min kink is benign: it only switches when the feet are level, where
// both branches agree.
double StanceFootZ(const mjModel* model, const mjData* data) {
  double left_center[3], right_center[3];
  FootCenter(model, data, "left_foot_pos", "left_foot_xaxis", left_center);
  FootCenter(model, data, "right_foot_pos", "right_foot_xaxis", right_center);
  return mju_min(left_center[2], right_center[2]);
}

}  // namespace

std::string H12Lowerbody::Name() const { return "H1-2 Lowerbody"; }

std::string H12Lowerbody::XmlPath() const {
  return GetModelPath("h12_lowerbody/task.xml");
}

double PelvisUpResidual(const mjModel* model, const mjData* data) {
  // World-frame pelvis z-axis from the "pelvis_up" framezaxis sensor
  // (position-stage, so already computed when the residual runs at the
  // acceleration stage).
  double* pelvis_up = SensorByName(model, data, "pelvis_up");
  return pelvis_up[2] - 1.0;
}

double TorsoAboveHipsResidual(const mjModel* model, const mjData* data) {
  // Trunk axis: hips center -> shoulders center. (torso_link's own origin is
  // at the waist, only 16 cm above the hip line — too short a segment for a
  // robust direction; the shoulder joints are 59 cm up.)
  double* left_hip = SensorByName(model, data, "left_hip_pos");
  double* right_hip = SensorByName(model, data, "right_hip_pos");
  double* left_shoulder = SensorByName(model, data, "left_shoulder_pos");
  double* right_shoulder = SensorByName(model, data, "right_shoulder_pos");
  double hips[3], shoulders[3];
  mju_add3(hips, left_hip, right_hip);
  mju_scl3(hips, hips, 0.5);
  mju_add3(shoulders, left_shoulder, right_shoulder);
  mju_scl3(shoulders, shoulders, 0.5);
  return UprightCosFromPoints(hips, shoulders);
}

double HeightResidual(const mjModel* model, const mjData* data) {
  // Signed height deficit, measured kinematically: pelvis height above the
  // lower foot-geometry center, plus that center's nominal ground clearance
  // — equals pelvis-above-ground on a flat stance, but stays observable on
  // the real robot (relative FK, no world z). Negative — free under the
  // one-sided rectify norm in task.xml — while above the minimum standing
  // height (default standing pelvis is 1.03 m); linear in the drop below.
  constexpr double kMinHeight = 1.0;  // meters
  double height = SensorByName(model, data, "pelvis_pos")[2] -
                  StanceFootZ(model, data) + kFootCenterHeight;
  return kMinHeight - height;
}

double CoMProjectionResidual(const mjModel* model, const mjData* data) {
  // LIPM capture point: CoM projected by sqrt(h / g), with h the CoM height
  // above the stance foot (+ nominal ground clearance) — kinematically
  // observable, unlike world z — so the projection horizon shortens as the
  // CoM lowers.
  double* com_position = SensorByName(model, data, "com_position");
  double* com_velocity = SensorByName(model, data, "com_velocity");
  double g = mju_max(-model->opt.gravity[2], mjMINVAL);
  double com_height =
      com_position[2] - StanceFootZ(model, data) + kFootCenterHeight;
  double k = mju_sqrt(mju_max(com_height, 0.0) / g);
  double capture_point[2] = {com_position[0] + k * com_velocity[0],
                             com_position[1] + k * com_velocity[1]};

  // Support center: mean of the two foot-geometry centers.
  double left_center[3], right_center[3];
  FootCenter(model, data, "left_foot_pos", "left_foot_xaxis", left_center);
  FootCenter(model, data, "right_foot_pos", "right_foot_xaxis", right_center);
  double error[2];
  for (int i = 0; i < 2; i++) {
    error[i] = 0.5 * (left_center[i] + right_center[i]) - capture_point[i];
  }
  return mju_norm(error, 2);
}

void JointTorquesResidual(const mjModel* model, const mjData* data,
                          double* residual) {
  // Per-actuator torque: these are gear=1 torque motors (ctrl IS the joint
  // torque; actuator_force = ctrl clamped by ctrlrange — the model sets
  // ctrlrange to mirror each joint's actuatorfrcrange, and forcerange is
  // unset). Normalized by the capacity max(|ctrlrange|) — robust to the
  // finger actuators' one-sided ranges — so each entry is the fraction of
  // available torque in use and one weight treats a 300 N·m knee like a
  // 2.7 N gripper. All nu actuators, not just the legs: the frozen
  // arm/gripper motors get a gradient to stop fighting the freeze
  // equalities.
  for (int i = 0; i < model->nu; i++) {
    double capacity = mju_max(mju_abs(model->actuator_ctrlrange[2 * i]),
                              mju_abs(model->actuator_ctrlrange[2 * i + 1]));
    residual[i] = data->actuator_force[i] / (capacity > 0 ? capacity : 1.0);
  }
}

// Writes one residual entry per <user> cost sensor dim in task.xml, in
// declaration order. The posture-shaping terms (leg/foot geometry, hip
// symmetry, joint velocity) were removed after a cost-weight sweep + N=25
// ablation found them redundant for standing (tuning/run_sweep.sh); the five
// terms below carry the stand: uprightness, height, CoM balance, and the
// torque regularizer that conditions iLQG.
void H12Lowerbody::ResidualFn::Residual(const mjModel* model,
                                        const mjData* data,
                                        double* residual) const {
  int counter = 0;

  // ----- Pelvis Up (dim 1) ----- //
  residual[counter++] = PelvisUpResidual(model, data);

  // ----- Torso Above Hips (dim 1) ----- //
  residual[counter++] = TorsoAboveHipsResidual(model, data);

  // ----- Height (dim 1: signed pelvis height deficit) ----- //
  residual[counter++] = HeightResidual(model, data);

  // ----- COM_Projection (dim 1: ||capture point - support center||) ----- //
  residual[counter++] = CoMProjectionResidual(model, data);

  // ----- JointTorques (dim nu = 31: capacity-normalized servo forces) --- //
  JointTorquesResidual(model, data, residual + counter);
  counter += model->nu;

  CheckSensorDim(model, counter);
}

// SKELETON transition: no mode/target logic yet.
void H12Lowerbody::TransitionLocked(mjModel* /*model*/, mjData* /*data*/) {
  // TODO(h12_lowerbody): mode/target updates.
}

}  // namespace mjpc
