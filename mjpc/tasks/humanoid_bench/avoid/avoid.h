#ifndef MJPC_TASKS_HUMANOID_BENCH_AVOID_AVOID_H_
#define MJPC_TASKS_HUMANOID_BENCH_AVOID_AVOID_H_

#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <string>
#include <cmath>

#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include "mujoco/mujoco.h"

namespace mjpc {
class Avoid : public Task {
 public:
  std::string Name() const override = 0;

  std::string XmlPath() const override = 0;

  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Avoid *task)
        : mjpc::BaseResidualFn(task), task_(const_cast<Avoid *>(task)) {}

    void Residual(const mjModel *model, const mjData *data,
                  double *residual) const override;

    private:
      Avoid *task_;
      friend class avoid;

  };

  Avoid() : residual_(this) {}

  void TransitionLocked(mjModel *model, mjData *data) override;
  void ResetLocked(const mjModel *model) override;

  void MoveObstacle(double dx, double dy, double dz) {
    obstacle_move_x_ += dx;
    obstacle_move_y_ += dy;
    obstacle_move_z_ += dz;
  }

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const

      override {
    return std::make_unique<ResidualFn>(this);
  }

  ResidualFn *InternalResidual()

      override {
    return &residual_;
  }

private:
  ResidualFn residual_;
  double obstacle_move_x_ = 0.0;
  double obstacle_move_y_ = 0.0;
  double obstacle_move_z_ = 0.0;
};

class Avoid_H12 : public Avoid {
 public:
  std::string Name() const override { return "Avoid H12"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/avoid/Avoid_H12.xml");
  }
};


class CapacitiveSkin {
 public:
  // CapacitiveSkin(const mjModel *model, const mjData *data,
  //                double eps = 1.0, double sensing_radius = 0.15)
  //     : model_(model), data_(data), eps_(eps), sensing_radius_(sensing_radius) {}

  CapacitiveSkin(const mjModel *model, double eps = 1.0, double sensing_radius = 0.15)
      : model_(model), eps_(eps), sensing_radius_(sensing_radius) {}


  void RegisterAllSkinSites() {
    sensor_site_ids_.clear();
    for (int i = 0; i < model_->nsite; ++i) {
      const char *name = mj_id2name(model_, mjOBJ_SITE, i);
      if (name && std::string(name).find("sensor") != std::string::npos) {
        sensor_site_ids_.push_back(i);
      }
    }
  }

  // Distance-based capacitance to a single obstacle
  double ComputeCapacitancePair(const mjtNum *sensor_pos,
                                const mjtNum *obstacle_pos,
                                double obstacle_radius) const {
    double dx = sensor_pos[0] - obstacle_pos[0];
    double dy = sensor_pos[1] - obstacle_pos[1];
    double dz = sensor_pos[2] - obstacle_pos[2];
    double d = std::sqrt(dx*dx + dy*dy + dz*dz);

    if (d > sensing_radius_ + obstacle_radius) return -1;

    double effective_d = std::max(0.01, d - obstacle_radius);
    return eps_ / effective_d;
  }

  // Simple distance check (for Python-style compute_distance)
  double ComputeDistance(const mjtNum *sensor_pos, const mjtNum *obstacle_pos) const {
    double dx = sensor_pos[0] - obstacle_pos[0];
    double dy = sensor_pos[1] - obstacle_pos[1];
    double dz = sensor_pos[2] - obstacle_pos[2];
    double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    return (dist > sensing_radius_) ? -1 : dist;
  }

  // Compute all sensor readings (distance or capacitance)
  std::unordered_map<int,double> ComputeAllCapacitances(const mjModel *model,
                                                        const mjData *data) const {
    std::unordered_map<int,double> readings;

    // assume one dynamic obstacle named "obstacle"
    int obstacle_id = mj_name2id(model, mjOBJ_BODY, "obstacle");
    if (obstacle_id < 0) return readings; // no obstacle

    const mjtNum *opos = &data->xpos[3 * obstacle_id];
    int geom_id = model->body_geomadr[obstacle_id];
    double radius = model->geom_size[geom_id];  // first size component

    for (int sid : sensor_site_ids_) {
      const mjtNum *spos = &data->site_xpos[3 * sid];
      readings[sid] = ComputeCapacitancePair(spos, opos, radius);
      // readings[sid] = ComputeDistance(spos, opos);  // Python test version
    }

    return readings;
  }

  const std::vector<int>& SensorIds() const { return sensor_site_ids_; }

 private:
  const mjModel *model_;
  // const mjData *data_;
  double eps_;
  double sensing_radius_;
  std::vector<int> sensor_site_ids_;
};

}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_BENCH_AVOID_AVOID_H_