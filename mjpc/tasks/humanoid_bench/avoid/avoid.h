#ifndef MJPC_TASKS_HUMANOID_BENCH_AVOID_AVOID_H_
#define MJPC_TASKS_HUMANOID_BENCH_AVOID_AVOID_H_

#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include "mujoco/mujoco.h"
#include "mjpc/tasks/humanoid_bench/avoid/offline_obstacles.h"

struct EpisodeLogger {
  std::string out_dir = ".";         // output directory
  int episode_idx = 0;
  std::ofstream csv;                 // open CSV
  std::string csv_path;
  bool open_episode() {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s/episode_%04d.csv", out_dir.c_str(), episode_idx);
    csv_path = buf;
    csv.open(csv_path, std::ios::out);
    if (!csv.is_open()) return false;
    // header (customize as needed)
    csv << "step, time";
    // we'll append dynamic headers later from code using model sizes
    csv << "\n";
    return true;
  }
  void close_episode() {
    if (csv.is_open()) csv.close();
    episode_idx++;
  }
};

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
      friend class Avoid;

      // task state variables, managed by Transition
      // bool use_cap = false;
      // bool use_tof = false;
      // double range_cap = 0;
      // double range_tof = 0;

      // constants, computed in Reset()
      int use_cap_id_ = -1;
      int use_tof_id_ = -1;
      int range_cap_id_ = -1;
      int range_tof_id_ = -1;

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
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn *InternalResidual() override { return &residual_; }

  EpisodeLogger logger_;
  double sim_time_per_step_ = 0.0;

private:
  friend class ResidualFn;
  ResidualFn residual_;
  bool logging_enabled_ = false;
  bool use_offline_obstacles_ = false;
  bool use_tof_sensors_ = true;
  bool use_cap_sensors_ = true;
  double tof_sensor_range_ = 4.0;     // default range
  double cap_sensor_range_ = 0.15;    // default range

  double obstacle_move_x_ = 0.0;
  double obstacle_move_y_ = 0.0;
  double obstacle_move_z_ = 0.0;

  // Obstacle trajectory state
  bool obstacle_launched_ = false;
  double obstacle_start_pos_[3] = {0};
  double obstacle_target_pos_[3] = {0};
  double obstacle_velocity_[3] = {0};
  double min_obstacle_dist_ = 1.0e6;

  // offline replay (forgive me)
  int current_obstacle_index_ = 0;

  // Configurations from GOOD trajectories (successful avoidance)
  static constexpr auto& good_obstacles_ = offline_obstacles::good_obstacles_;
  static constexpr auto& bad_obstacles_ = offline_obstacles::bad_obstacles_;
  static constexpr auto& all_obstacles_ = offline_obstacles::all_obstacles_;
  static constexpr int num_good_obstacles_ = offline_obstacles::num_good_obstacles_;
  static constexpr int num_bad_obstacles_ = offline_obstacles::num_bad_obstacles_;
  static constexpr int num_all_obstacles_ = offline_obstacles::num_all_obstacles_;
};

class Avoid_H12 : public Avoid {
 public:
  std::string Name() const override { return "Avoid H12 Cap"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/avoid/Avoid_H12.xml");
  }
};

class Avoid_H12_ToF : public Avoid {
 public:
  std::string Name() const override { return "Avoid H12 ToF"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/avoid/Avoid_H12_ToF.xml");
  }
};

class Avoid_H12_ToF_Cap : public Avoid {
 public:
  std::string Name() const override { return "Avoid H12 Cap ToF"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/avoid/Avoid_H12_ToF_Cap.xml");
  }
};

class CapacitiveSkin {
 public:
  CapacitiveSkin(const mjModel *model, double eps = 1.0, double cap_sensing_radius = 0.25, double tof_range = 1.0)
      : model_(model), eps_(eps), sensing_radius_(cap_sensing_radius), tof_range_(tof_range) {}


  void RegisterAllSkinSites() {
    sensor_site_ids_.clear();
    for (int i = 0; i < model_->nsite; ++i) {
      const char *name = mj_id2name(model_, mjOBJ_SITE, i);
      if (name && std::string(name).find("sensor") != std::string::npos) {
        sensor_site_ids_.push_back(i);
      }
    }
  }

  void RegisterAllToFSensors() {
    tof_sensor_ids_.clear();
    for (int i = 0; i < model_->nsensor; ++i) {
      if (model_->sensor_type[i] == mjSENS_RANGEFINDER) {
        const char *name = mj_id2name(model_, mjOBJ_SENSOR, i);
        if (name && std::string(name).find("sensor") != std::string::npos) {
          tof_sensor_ids_.push_back(i);
        }
      }
    }
  }

  std::unordered_map<int, double> ReadToFSensors(const mjData *data) const {
    std::unordered_map<int, double> readings;
    for (int sensor_id : tof_sensor_ids_) {
      int adr = model_->sensor_adr[sensor_id];
      double range = data->sensordata[adr];
      readings[sensor_id] = range;
    }
    return readings;
  }

  // Distance-based capacitance to a single obstacle
  double ComputeCapacitancePair(const double d,
                                const mjtNum *sensor_pos,
                                const mjtNum *obstacle_pos,
                                double obstacle_radius) const {
    if (d > sensing_radius_ + obstacle_radius) return -1;

    double effective_d = std::max(0.01, d - obstacle_radius);
    // add progressive dampening term for obstacles farther than 0.25
    // if (d > 0.25) {
    //   double dampening = 1.0 - (d - 0.25) / (sensing_radius_ - 0.25);
    //   dampening = std::max(0.0, dampening);
    //   return eps_ / effective_d * dampening;
    // }
    return eps_ / effective_d;
  }

  // Simple distance check (for Python-style compute_distance)
  double ComputeDistance(const mjtNum *sensor_pos, const mjtNum *obstacle_pos) const {
    double dx = sensor_pos[0] - obstacle_pos[0];
    double dy = sensor_pos[1] - obstacle_pos[1];
    double dz = sensor_pos[2] - obstacle_pos[2];
    double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    return dist;
  }

  std::unordered_map<int,double> ComputeAllDistances(const mjModel *model,
                                                     const mjData *data,
                                                     bool is_tof = true
                                                  ) const {
    std::unordered_map<int,double> distances;

    // assume one dynamic obstacle named "obstacle"
    int obstacle_id = mj_name2id(model_, mjOBJ_BODY, "obstacle");
    if (obstacle_id < 0) return distances; // no obstacle

    const mjtNum *opos = &data->xpos[3 * obstacle_id];
    auto ids = is_tof ? tof_sensor_ids_ : sensor_site_ids_;

    for (int sid : ids) {
      const mjtNum *spos = &data->site_xpos[3 * sid];
      double dist = ComputeDistance(spos, opos);
      distances[sid] = dist;
    }

    return distances;
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
      double dist = ComputeDistance(spos, opos);
      readings[sid] = ComputeCapacitancePair(dist, spos, opos, radius);
    }

    return readings;
  }

  const std::vector<int>& SensorIds() const { return sensor_site_ids_; }
  const std::vector<int>& ToFSensorIds() const { return tof_sensor_ids_; }
  double CapSensingRadius() const { return sensing_radius_; }
  double ToFSensorRange() const { return tof_range_; }
  double NumToFSensors() const { return static_cast<double>(tof_sensor_ids_.size()); }
  double NumCapSensors() const { return static_cast<double>(sensor_site_ids_.size()); }

 private:
  const mjModel *model_;
  // const mjData *data_;
  double eps_;
  double sensing_radius_;
  double tof_range_;
  std::vector<int> sensor_site_ids_;
  std::vector<int> tof_sensor_ids_;
};

}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_BENCH_AVOID_AVOID_H_