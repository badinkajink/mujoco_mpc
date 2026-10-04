// Learned dynamics residual for the planner's imagined rollouts (branch mbrl, 2026-10-04).
// A small MLP (SiLU) trained offline on real H1-2 recordings predicts a joint-velocity correction from
// planner-known quantities; it is applied inside Task::ModifyRolloutState (planner threads only, never on the
// real-robot path). Loaded from the flat binary written by mbrl/export_mlp.py when the env var
// H12_RESIDUAL_MLP names a file; absent or unreadable -> no-op.
#pragma once
#include <string>
#include <vector>
#include <mujoco/mujoco.h>

namespace mjpc {

class ResidualMLP {
 public:
  // returns false (and stays disabled) if the file is missing or malformed
  bool Load(const std::string& path);
  bool enabled() const { return enabled_; }
  int nin() const { return nin_; }
  int nout() const { return nout_; }
  // y[nout] = clip(net((x - mu)/sd) * ysd + ymu). Thread-safe (read-only weights, caller-provided scratch).
  void Forward(const float* x, float* y, std::vector<float>& scratch) const;

 private:
  struct Layer { int rows, cols; std::vector<float> W, b; };
  bool enabled_ = false; int nin_ = 0, nout_ = 0;
  std::vector<Layer> layers_;
  std::vector<float> mu_, sd_, ymu_, ysd_, clip_;
};

// Feature vector (168) for the H1-2 lean plant, matching mbrl/residual_fit.py features():
// [q(27), dq(27), tau_cmd(27), ctrl-q(27), kp/100(27), kd/10(27), gravity dir in IMU frame(3), pelvis angvel(3)].
// kp/kd come from the position actuators' gain/bias params; tau_cmd is the clipped actuator force.
// Returns false if the model does not look like the 27-joint H1-2 plant.
bool H12ResidualFeatures(const mjModel* m, const mjData* d, int imu_site, float* x168);

// Apply the residual as a pre-step joint-velocity correction. No-op if the MLP is not enabled.
void H12ApplyResidual(const ResidualMLP& mlp, const mjModel* m, mjData* d, int imu_site, double scale);

}  // namespace mjpc
