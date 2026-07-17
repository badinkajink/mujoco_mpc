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

#ifndef MJPC_TASKS_H12_LOWERBODY_H12_LOWERBODY_H_
#define MJPC_TASKS_H12_LOWERBODY_H12_LOWERBODY_H_

#include <memory>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {

// Pelvis uprightness error: (pelvis frame z-axis · world z) - 1, read from the
// "pelvis_up" framezaxis sensor in task.xml. 0 when the pelvis z-axis points
// at world +Z, -1 when horizontal, -2 when inverted. Backs the "Pelvis Up"
// cost term.
double PelvisUpResidual(const mjModel* model, const mjData* data);

// Per-leg thigh uprightness: cosine similarity of the knee->hip segment with
// world +Z, minus 1, from the *_hip_pos / *_knee_pos framepos sensors (joint
// centers). Writes residual[0] = left, residual[1] = right; each is 0 when
// the hip is directly above the knee, -1 when the thigh is horizontal, -2
// when inverted. Backs the "Hips Above Knees" cost term.
void HipsAboveKneesResidual(const mjModel* model, const mjData* data,
                            double* residual);

// Same contract for the shin: cos(ankle->knee segment, world +Z) - 1 per
// leg. Backs the "Knees Above Ankles" cost term.
void KneesAboveAnklesResidual(const mjModel* model, const mjData* data,
                              double* residual);

// Trunk uprightness: cos(hips-center -> shoulders-center segment, world +Z)
// - 1, from the hip and shoulder joint-center framepos sensors. 0 when the
// shoulders are directly above the hips. Backs the "Torso Above Hips" cost
// term.
double TorsoAboveHipsResidual(const mjModel* model, const mjData* data);

// Signed stance margin: ||footL_center - footR_center|| in the horizontal
// (xy) plane minus the allowed maximum stance (0.4572 m = 1.5 ft); vertical
// separation is not charged. Negative while the feet are close enough,
// positive by the excess when they spread too far. Backs the "Foot Width"
// cost term (one-sided rectify norm — under-threshold is free).
double FootWidthResidual(const mjModel* model, const mjData* data);

// Per-foot side margin vs the pelvis sagittal plane (pelvis y-axis flattened
// to horizontal): residual[0] = clearance - left_offset, residual[1] =
// clearance + right_offset, where offsets are the foot-geometry centers'
// signed lateral distance from the pelvis (positive = robot's left) and
// clearance = 0.05 m. Each entry is negative while its foot is on its own
// side with margin, positive when it enters the band or crosses. Backs the
// "Foot Crossing" cost term (one-sided rectify norm). Zeroed when the pelvis
// y-axis is vertical (robot on its side).
void FootCrossingResidual(const mjModel* model, const mjData* data,
                          double* residual);

// Mirror-symmetry error between the two hips' joint angles, from the
// *_hip_{pitch,roll,yaw}_q jointpos sensors. Both sides share identical
// joint axes, so a sagittally mirrored pose has equal pitches and opposite
// rolls/yaws: residual[0] = pitchL - pitchR, residual[1] = rollL + rollR,
// residual[2] = yawL + yawR. All zero when the legs move in mirrored unison,
// regardless of what that motion is. Backs the "Hip Parity" cost term.
void HipParityResidual(const mjModel* model, const mjData* data,
                       double* residual);

// Hip yaw angles directly (residual[0] = left, residual[1] = right), from
// the *_hip_yaw_q jointpos sensors: zero = toes straight forward. Covers the
// mirrored toe-out/toe-in direction that Hip Parity's yawL + yawR entry is
// blind to. Backs the "Hip Yaw Zero" cost term.
void HipYawZeroResidual(const mjModel* model, const mjData* data,
                        double* residual);

// Hip roll angles directly (residual[0] = left, residual[1] = right), from
// the *_hip_roll_q jointpos sensors: zero = legs hanging straight down.
// Covers the mirrored splay/adduction direction that Hip Parity's
// rollL + rollR entry is blind to. Backs the "Hip Roll Zero" cost term.
void HipRollZeroResidual(const mjModel* model, const mjData* data,
                         double* residual);

// Signed pelvis height deficit: 1.0 m minus the pelvis height measured
// above the lower foot-geometry center (+ its 0.0468 m nominal ground
// clearance, so the number reads as height-above-ground on flat stance;
// default standing pelvis is 1.03 m). Kinematic, not world z — observable
// on the real robot from encoders + IMU. Negative while standing tall,
// positive by the shortfall below 1 m. Backs the "Height" cost term
// (one-sided rectify norm — above-threshold is free).
double HeightResidual(const mjModel* model, const mjData* data);

// Projected-CoM balance error: ||capture_point_xy - support_center_xy||.
// Capture point = whole-robot CoM projected by the live LIPM constant
// sqrt(h / g), with h the CoM height above the lower foot-geometry center
// (kinematically observable — no world z); support center = mean of the two
// foot-geometry centers
// (ankle_roll origins offset +3.3 cm along each foot's forward axis). 0 when
// the projected CoM is centered in the support polygon; the velocity
// projection also serves as the damping term. Backs the "COM_Projection"
// cost term (SmoothAbs norm).
double CoMProjectionResidual(const mjModel* model, const mjData* data);

// Joint velocities: qvel[6:], all model->nv - 6 non-root DOFs (including the
// passive gripper linkage). Backs the "Joint Vel." cost term.
void JointVelResidual(const mjModel* model, const mjData* data,
                      double* residual);

// Per-actuator effort: actuator_force[i] / max(|ctrlrange[i]|) for all nu
// actuators — the fraction of each motor's torque capacity in use (these are
// gear=1 torque motors, so actuator_force is the ctrlrange-clamped ctrl;
// capacity-normalized so one weight treats every joint equally, guarded to
// /1.0 if an actuator had no ctrl limits). Includes the frozen arm/gripper
// actuators deliberately: penalizing their output stops the planner
// commanding torques that fight the freeze equalities. Backs the
// "JointTorques" cost term.
void JointTorquesResidual(const mjModel* model, const mjData* data,
                          double* residual);

// MJPC task for the Unitree H1-2 (+ Magpie) lower body. The model is loaded
// from h12_lowerbody/task.xml (which bind-mounts h1_2_magpie.xml from
// CL_Assets at container runtime). Cost terms so far: Pelvis Up. The
// transition is still an empty stub.
class H12Lowerbody : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;

  class ResidualFn : public BaseResidualFn {
   public:
    explicit ResidualFn(const H12Lowerbody* task) : BaseResidualFn(task) {}

    // Writes one entry per <user> cost sensor dim in task.xml, in declaration
    // order. Terms: Pelvis Up (dim 1).
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;

   private:
    friend class H12Lowerbody;
  };

  H12Lowerbody() : residual_(this) {}

  // TODO(h12_lowerbody): mode/target bookkeeping as the controller is built.
  void TransitionLocked(mjModel* model, mjData* data) override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};

}  // namespace mjpc

#endif  // MJPC_TASKS_H12_LOWERBODY_H12_LOWERBODY_H_
