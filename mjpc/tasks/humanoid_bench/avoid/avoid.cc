#include "mjpc/tasks/humanoid_bench/avoid/avoid.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>

#include "mujoco/mujoco.h"

namespace mjpc {

namespace {
// thread-safe random number generator
thread_local std::mt19937 generator(std::random_device{}());
}  // namespace


// ------------------ Residuals for humanoid stand task ------------
//   Number of residuals:
//      Residual(0): 0
//      Residual(1): Height: head feet vertical error
//      Residual(2): Balance: CoM Velocity
//      Residual(3): joint velocity
//      Residual(4): balance
//      Residual(5): upright
//      Residual(6): posture
//      Residual(7): velocity
//      Residual(8): control
//      Residual(9): obstacle proximity
//      Residual(10): COM distance to obstacle
//   Number of parameters:
//      Parameter(0): head height goal
// ----------------------------------------------------------------
void Avoid::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                double *residual) const {
  // Capacitive skin readings
//   static CapacitiveSkin cap(model, data);
  static CapacitiveSkin skin(model, 1.0, task_->cap_sensor_range_, task_->tof_sensor_range_);

  static bool initialized = false;
  if (!initialized) {
    skin.RegisterAllSkinSites();
    skin.RegisterAllToFSensors();
    initialized = true;
  }
  auto readings = skin.ComputeAllCapacitances(model, data);

  //--------------- Beginning of reward calculation -----------------//

  double const height_goal = parameters_[0];
  int counter = 0;

  double const success = 1000;

  // ============ OBSTACLE AVOIDANCE REWARD ============

  // Compute weighted centroid of obstacles (where danger is)
  double obstacle_centroid[3] = {0, 0, 0};
  double total_weight = 0.0;
  int num_detections = 0;

  for (auto &p : readings) {
    int sid = p.first;
    double capacitance = p.second;

    if (capacitance > 0) {  // Obstacle detected
      num_detections++;
      const mjtNum *sensor_pos = &data->site_xpos[3 * sid];

      // Weight by inverse distance (closer = higher weight)
      double weight = capacitance;  // Already 1/distance
      obstacle_centroid[0] += weight * sensor_pos[0];
      obstacle_centroid[1] += weight * sensor_pos[1];
      obstacle_centroid[2] += weight * sensor_pos[2];
      total_weight += weight;
    }
  }

  if (total_weight > 0) {
    obstacle_centroid[0] /= total_weight;
    obstacle_centroid[1] /= total_weight;
    obstacle_centroid[2] /= total_weight;
  }

  // Reward: negative of average proximity danger
  double avg_danger = (num_detections > 0) ? total_weight / num_detections : 0.0;
  // double reward = 0.0 * avg_danger ;  // Penalty for proximity
  double reward = 0.0 * avg_danger * success ;  // Penalty for proximity
  //--------------- End of reward calculation -----------------//

  // residual[counter++] = success - reward;
  residual[counter++] = reward;

  // -------------- Below are additional residuals -------------- //

  // ----- Height: head feet vertical error ----- //
  // Note: Reduced importance vs push task since avoiding lowers head

  // feet sensor positions
  double *foot_right_pos = SensorByName(model, data, "foot_right_pos");
  double *foot_left_pos = SensorByName(model, data, "foot_left_pos");

  double *head_position = SensorByName(model, data, "head_position");
  double head_feet_error =
      head_position[2] - 0.5 * (foot_right_pos[2] + foot_left_pos[2]);
  residual[counter++] = head_feet_error - height_goal;

  // ----- Balance: CoM-feet xy error ----- //

  // capture point
  double *com_velocity = SensorByName(model, data, "torso_subtreelinvel");

  // ----- COM xy velocity should be 0 ----- //
  mju_copy(&residual[counter], com_velocity, 2);
  counter += 2;

  // ----- joint velocity ----- //
  mju_copy(residual + counter, data->qvel + 6, model->nu);
  counter += model->nu;

  // ----- torso height ----- //
  double torso_height = SensorByName(model, data, "torso_position")[2];

  // ----- balance ----- //
  // capture point
  double *subcom = SensorByName(model, data, "torso_subcom");
  double *subcomvel = SensorByName(model, data, "torso_subcomvel");

  double capture_point[3];
  mju_addScl(capture_point, subcom, subcomvel, 0.3, 3);
  capture_point[2] = 1.0e-3;

  // project onto line segment

  double axis[3];
  double center[3];
  double vec[3];
  double pcp[3];
  mju_sub3(axis, foot_right_pos, foot_left_pos);
  axis[2] = 1.0e-3;
  double length = 0.5 * mju_normalize3(axis) - 0.05;
  mju_add3(center, foot_right_pos, foot_left_pos);
  mju_scl3(center, center, 0.5);
  mju_sub3(vec, capture_point, center);

  // project onto axis
  double t = mju_dot3(vec, axis);

  // clamp
  t = mju_max(-length, mju_min(length, t));
  mju_scl3(vec, axis, t);
  mju_add3(pcp, vec, center);
  pcp[2] = 1.0e-3;

  // is standing
  double standing =
      torso_height / mju_sqrt(torso_height * torso_height + 0.45 * 0.45) - 0.4;

  mju_sub(&residual[counter], capture_point, pcp, 2);
  mju_scl(&residual[counter], &residual[counter], standing, 2);

  counter += 2;

  // ----- upright ----- //
  double *torso_up = SensorByName(model, data, "torso_up");
  double *pelvis_up = SensorByName(model, data, "pelvis_up");
  double *foot_right_up = SensorByName(model, data, "foot_right_up");
  double *foot_left_up = SensorByName(model, data, "foot_left_up");

  double z_ref[3] = {0.0, 0.0, 1.0};

  // torso
  residual[counter++] = torso_up[2] - 1.0;

  // pelvis
  residual[counter++] = 0.3 * (pelvis_up[2] - 1.0);

  // right foot
  mju_sub3(&residual[counter], foot_right_up, z_ref);
  mju_scl3(&residual[counter], &residual[counter], 0.1 * standing);
  counter += 3;

  mju_sub3(&residual[counter], foot_left_up, z_ref);
  mju_scl3(&residual[counter], &residual[counter], 0.1 * standing);
  counter += 3;

  // ----- posture ----- //
  // Reduced weight vs push task to allow more deviation for avoiding
  mju_sub(&residual[counter], data->qpos + 7, model->key_qpos + 7, model->nu);
  counter += model->nu;

  // com vel
  double *waist_lower_subcomvel =
      SensorByName(model, data, "waist_lower_subcomvel");
  double *torso_velocity = SensorByName(model, data, "torso_velocity");
  double com_vel[2];
  mju_add(com_vel, waist_lower_subcomvel, torso_velocity, 2);
  mju_scl(com_vel, com_vel, 0.5, 2);

  // ----- move feet ----- //
  double *foot_right_vel = SensorByName(model, data, "foot_right_vel");
  double *foot_left_vel = SensorByName(model, data, "foot_left_vel");
  double move_feet[2];
  mju_copy(move_feet, com_vel, 2);
  mju_addToScl(move_feet, foot_right_vel, -0.5, 2);
  mju_addToScl(move_feet, foot_left_vel, -0.5, 2);

  mju_copy(&residual[counter], move_feet, 2);
  mju_scl(&residual[counter], &residual[counter], standing, 2);
  counter += 2;

  // ----- control ----- //
  mju_sub(&residual[counter], data->ctrl, model->key_qpos + 7,
          model->nu);  // because of pos control
  counter += model->nu;

  // ============ NEW OBSTACLE AVOIDANCE RESIDUALS ============

  // ----- Obstacle Proximity (per-sensor penalty) ----- //
  if (task_->use_cap_sensors_) {
    const auto& sensor_ids = skin.SensorIds();
    for (size_t i = 0; i < sensor_ids.size(); ++i) {
      int sid = sensor_ids[i];
      auto it = readings.find(sid);

      if (it != readings.end() && it->second > 0) {
        // Penalty: inverse distance squared (stronger when close)
        double capacitance = it->second; // (1/d)
        double normalized_range = capacitance / (1.0 / skin.CapSensingRadius()); 
        residual[counter] = 1 - normalized_range;  // 0 when far, 1 when very very close
      } else {
        residual[counter] = 0.0;  // No obstacle detected
      }
      counter++;
  }
  } else {
    // Zero out capacitive residuals if disabled
    for (int i = 0; i < static_cast<int>(skin.NumCapSensors()); i++) {  // 63 capacitive sensors
      residual[counter++] = 0.0;
    }
  }

  // ----- CoM Away From Obstacle ----- //
  // Encourage CoM to shift away from obstacle centroid (xy plane only)
  if (total_weight > 0 && task_->use_cap_sensors_) {
    double *com_pos = SensorByName(model, data, "torso_subcom");

    // Direction from obstacle to CoM (desired direction)
    double away_dir[2];
    away_dir[0] = com_pos[0] - obstacle_centroid[0];
    away_dir[1] = com_pos[1] - obstacle_centroid[1];
    double away_dist = mju_sqrt(away_dir[0]*away_dir[0] + away_dir[1]*away_dir[1]);

    if (away_dist > 1e-6) {
      away_dir[0] /= away_dist;
      away_dir[1] /= away_dist;

      // Target: CoM should be 0.2m away from obstacle centroid in xy
      double desired_offset = 0.2;
      double current_offset = away_dist;

      residual[counter++] = (current_offset - desired_offset) * away_dir[0];
      residual[counter++] = (current_offset - desired_offset) * away_dir[1];
    } else {
      residual[counter++] = 0.0;
      residual[counter++] = 0.0;
    }
  } else {
    // No obstacle detected, no preference
    residual[counter++] = 0.0;
    residual[counter++] = 0.0;
  }

  // --------------------- ToF Obstacle Proximity ---------------------
  if (task_->use_tof_sensors_) {
    auto tof_readings = skin.ReadToFSensors(data);
    const auto& tof_ids = skin.ToFSensorIds();
    
    int tof_idx = 0;
    for (int sensor_id : tof_ids) {
      double range = tof_readings[sensor_id];
      
      if (range > 0 && range < task_->tof_sensor_range_) {
        // Penalize proximity: closer = higher cost
        // Use inverse distance, capped at some max
        double normalized_range = range / task_->tof_sensor_range_;
        residual[counter++] = 1.0 - normalized_range;  // 0 when far, 1 when very close
      } else {
        residual[counter++] = 0.0;  // No detection = no cost
      }
      tof_idx++;
    }
  } else {
    // Zero out ToF residuals if disabled
    for (int i = 0; i < skin.NumToFSensors(); i++) {
      residual[counter++] = 0.0;
    }
  }
  
  // --------------------- CoM Away From ToF Obstacle ---------------------
  if (task_->use_tof_sensors_) {
    auto tof_readings = skin.ReadToFSensors(data);
    const auto& tof_ids = skin.ToFSensorIds();
    
    // Compute weighted obstacle direction from all sensor detections
    double weighted_dir[3] = {0, 0, 0};
    double total_weight = 0.0;
    
    for (int sensor_id : tof_ids) {
      double range = tof_readings[sensor_id];
      
      if (range > 0 && range < skin.ToFSensorRange()) {
        int site_id = model->sensor_objid[sensor_id];
        const mjtNum* sensor_dir = &data->site_xmat[9 * site_id + 6]; // z-axis
        
        // Weight by inverse distance (closer sensors matter more)
        double weight = 1.0 / std::max(0.1, range);
        
        weighted_dir[0] += weight * sensor_dir[0];
        weighted_dir[1] += weight * sensor_dir[1];
        weighted_dir[2] += weight * sensor_dir[2];
        total_weight += weight;
      }
    }
    if (total_weight > 0) {
      // Normalize weighted direction
      weighted_dir[0] /= total_weight;
      weighted_dir[1] /= total_weight;
      weighted_dir[2] /= total_weight;
      mju_normalize3(weighted_dir);
      
      double *com_pos = SensorByName(model, data, "torso_subcom");
      // Project CoM position onto weighted direction
      double com_offset = com_pos[0]*weighted_dir[0] + com_pos[1]*weighted_dir[1] + com_pos[2]*weighted_dir[2];
      // Desired offset away from obstacle direction
      double desired_offset = 0.2;
      
      residual[counter++] = (com_offset - desired_offset) * weighted_dir[0];
      residual[counter++] = (com_offset - desired_offset) * weighted_dir[1];
    }
    else {
      residual[counter++] = 0.0;
      residual[counter++] = 0.0;
    }
  } else {
    residual[counter++] = 0.0;
    residual[counter++] = 0.0;
  }

  // sensor dim sanity check
  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    mju_error(
        "mismatch between total user-sensor dimension %d "
        "and actual length of residual %d",
        user_sensor_dim, counter);
  }
}

// -------- Transition for humanoid_bench avoid task -------- //
// ------------------------------------------------------------ //
void Avoid::TransitionLocked(mjModel *model, mjData *data) {
  int obstacle_id = mj_name2id(model, mjOBJ_BODY, "obstacle");
  if (obstacle_id < 0) return;

  int jnt_id = model->body_jntadr[obstacle_id];
  if (jnt_id < 0) return;
  int qpos_adr = model->jnt_qposadr[jnt_id];
  int qvel_adr = model->jnt_dofadr[jnt_id];

  // Capacitance
  static CapacitiveSkin skin(model);
  static bool initialized = false;
  if (!initialized) {
    skin.RegisterAllSkinSites();
    initialized = true;
  }

  if (!obstacle_launched_) {     // --- Set up a new trajectory ---
    if (use_offline_obstacles_) {
      // Get next configuration (wrap around if we've exhausted all configs)
      int idx = current_obstacle_index_ % num_all_obstacles_;
      const auto& config = all_obstacles_[idx].config;

      // Set obstacle position, vel directly from recorded config
      mju_copy3(obstacle_start_pos_, config.position);
      mju_copy3(obstacle_velocity_, config.velocity);

      // Set obstacle radius from recorded config
      int obstacle_geom_id = mj_name2id(model, mjOBJ_GEOM, "obstacle_geom");
      if (obstacle_geom_id >= 0) {
        mjModel* mutable_model = const_cast<mjModel*>(model);
        mutable_model->geom_size[3 * obstacle_geom_id] = config.radius;
      }

      // Set target position (assume it hits torso, we don't record this)
      double *torso_pos = SensorByName(model, data, "torso_position");
      mju_copy3(obstacle_target_pos_, torso_pos);

      // Apply initial position and velocity to obstacle
      mju_copy3(data->qpos + qpos_adr, obstacle_start_pos_);
      data->qpos[qpos_adr + 3] = 1.0;  // qw
      data->qpos[qpos_adr + 4] = 0.0;  // qx
      data->qpos[qpos_adr + 5] = 0.0;  // qy
      data->qpos[qpos_adr + 6] = 0.0;  // qz
      mju_copy3(data->qvel + qvel_adr, obstacle_velocity_);

      current_obstacle_index_++;

      // Debug output
      std::printf("Offline obstacle %d/%d: pos=[%.2f,%.2f,%.2f], vel=[%.2f,%.2f,%.2f], r=%.3f, source=%s\n",
                  idx + 1, num_all_obstacles_,
                  config.position[0], config.position[1], config.position[2],
                  config.velocity[0], config.velocity[1], config.velocity[2],
                  config.radius,
                  all_obstacles_[idx].from_good ? "GOOD" : "BAD");

    } else {
      // 1. Select a random sensor site as the target
      double *torso_pos = SensorByName(model, data, "torso_position");
      const auto& sensor_ids = skin.SensorIds();

      // Randomly select a sensor site as target
      std::uniform_int_distribution<int> site_dist(0, sensor_ids.size() - 1);
      int target_site_id = sensor_ids[site_dist(generator)];

      // Get target sensor position
      const mjtNum *target_site_pos = &data->site_xpos[3 * target_site_id];
      obstacle_target_pos_[0] = target_site_pos[0];
      obstacle_target_pos_[1] = target_site_pos[1];
      obstacle_target_pos_[2] = target_site_pos[2];

      // 2. Initialize obstacle position in a bounded spherical shell around the robot (outside)
      std::uniform_real_distribution<double> sphere_u(0.0, 1.0);
      std::normal_distribution<double> normal(0.0, 1.0);

      double dir[3] = { normal(generator), normal(generator), normal(generator) };
      mju_normalize3(dir); // unit vector

      double min_dist = 1.25; // minimum distance from torso (tweakable)
      double max_dist = 2.00; // maximum distance from torso (tweakable)
      std::uniform_real_distribution<double> dist_dist(min_dist, max_dist);
      double dist = dist_dist(generator);

      obstacle_start_pos_[0] = torso_pos[0] + dir[0] * dist;
      obstacle_start_pos_[1] = torso_pos[1] + dir[1] * dist;
      obstacle_start_pos_[2] = torso_pos[2] + dir[2] * dist;
      // ensure start above ground
      obstacle_start_pos_[2] = mju_max(1.0, obstacle_start_pos_[2]);
      // ensure not dropped from too high
      obstacle_start_pos_[2] = mju_min(obstacle_start_pos_[2], 3.0);

      // Place obstacle qpos (assumes free joint/body qpos maps directly)
      mju_copy3(data->qpos + qpos_adr, obstacle_start_pos_);

      // 3. Sample random speed
      std::uniform_real_distribution<double> speed_dist(2.0, 10.0);
      double speed = speed_dist(generator);

      // 4. Compute travel time and initial ballistic velocity (simple constant-accel model)
      // horizontal distance used to estimate travel time; avoid zero division
      double dx = obstacle_target_pos_[0] - obstacle_start_pos_[0];
      double dy = obstacle_target_pos_[1] - obstacle_start_pos_[1];
      double dz = obstacle_target_pos_[2] - obstacle_start_pos_[2];
      double straight_dist = std::sqrt(dx*dx + dy*dy + dz*dz);
      double travel_time = (straight_dist > 1e-6) ? (straight_dist / speed) : 0.5;

      // Ensure travel_time is not vanishingly small
      if (travel_time < 0.05) travel_time = 0.05;

      // Solve for initial velocity v so that:
      // target = start + v * t + 0.5 * g * t^2  =>  v = (target - start - 0.5*g*t^2) / t
      double g[3] = { model->opt.gravity[0], model->opt.gravity[1], model->opt.gravity[2] };
      obstacle_velocity_[0] = (dx - 0.5 * g[0] * travel_time * travel_time) / travel_time;
      obstacle_velocity_[1] = (dy - 0.5 * g[1] * travel_time * travel_time) / travel_time;
      obstacle_velocity_[2] = (dz - 0.5 * g[2] * travel_time * travel_time) / travel_time;

      // Cap velocity magnitude to a safe maximum (in case of tiny travel_time)
      double vmax = 10.0;
      double vmagsq = obstacle_velocity_[0]*obstacle_velocity_[0] +
                      obstacle_velocity_[1]*obstacle_velocity_[1] +
                      obstacle_velocity_[2]*obstacle_velocity_[2];
      if (vmagsq > vmax*vmax) {
        double scale = vmax / std::sqrt(vmagsq);
        obstacle_velocity_[0] *= scale;
        obstacle_velocity_[1] *= scale;
        obstacle_velocity_[2] *= scale;
      }
      // Apply qvel
      mju_copy3(data->qvel + qvel_adr, obstacle_velocity_);
    }

    obstacle_launched_ = true;
    min_obstacle_dist_ = 1.0e6;
  } else {
    // --- Monitor existing trajectory ---
    double* obstacle_pos = data->qpos + qpos_adr;
    double* torso_pos = SensorByName(model, data, "torso_position");
    double dist_to_torso = mju_dist3(obstacle_pos, torso_pos);

    // Check if we've passed the closest point -- tweakable
    if (dist_to_torso > min_obstacle_dist_ + 0.7 ||
        mju_dist3(obstacle_pos, torso_pos) > 5.0 ||
        obstacle_pos[2] < -0.5) { // Passed and moving away, or obstacle goes too far away, or falls through the floor
      obstacle_launched_ = false; // Trigger reset on next step
      if (logger_.csv.is_open()) {
        logger_.csv.close();
        logger_.episode_idx++;
        ResetLocked(model);
        // set home keyframe
        int home_id = mj_name2id(model, mjOBJ_KEY, "home");
        if (home_id >= 0) mj_resetDataKeyframe(model, data, home_id);
      }
    } else {
      min_obstacle_dist_ = mju_min(min_obstacle_dist_, dist_to_torso);
    }
  }

  // ** Apply keyboard movement commands **
  double* pos = data->qpos + qpos_adr;
  pos[0] += obstacle_move_x_;
  pos[1] += obstacle_move_y_;
  pos[2] += obstacle_move_z_;
  // Clear movement commands
  obstacle_move_x_ = 0.0;
  obstacle_move_y_ = 0.0;
  obstacle_move_z_ = 0.0;

  // Logging
  // after obstacle launch / monitor logic and just before returning from TransitionLocked:
  if (logger_.csv.is_open()) {
    int step = data->time / sim_time_per_step_; // approximate step index
    double time = data->time;
    // obstacle qpos/qvel addresses retrieved earlier as qpos_adr/qvel_adr
    double* obst_pos = data->qpos + qpos_adr;
    double* obst_vel = data->qvel + qvel_adr;

    // obstacle radius: read geom size if available
    int obstacle_geom_id = mj_name2id(model, mjOBJ_GEOM, "obstacle_geom");
    double obst_radius = 0.0;
    if (obstacle_geom_id >= 0) obst_radius = model->geom_size[3*obstacle_geom_id];

    // write base columns
    logger_.csv << step << "," << time << ","
                << obst_pos[0] << "," << obst_pos[1] << "," << obst_pos[2] << ","
                << obst_vel[0] << "," << obst_vel[1] << "," << obst_vel[2] << ","
                << obst_radius;

    int obstacle_body = mj_name2id(model, mjOBJ_BODY, "obstacle");
    int obstacle_jnt = model->body_jntadr[obstacle_body];
    int obstacle_qpos0 = model->jnt_qposadr[obstacle_jnt];
    int obstacle_qvel0 = model->jnt_dofadr[obstacle_jnt];

    static const int OBST_QPOS_DOF = 7;
    static const int OBST_QVEL_DOF = 6;

    // qpos
    for (int i = 0; i < model->nq; i++) {
        if (i >= obstacle_qpos0 && i < obstacle_qpos0 + OBST_QPOS_DOF)
            continue;
        logger_.csv << "," << data->qpos[i];
    }
    for (int i = 0; i < model->nv; i++) {
        if (i >= obstacle_qvel0 && i < obstacle_qvel0 + OBST_QVEL_DOF)
            continue;
        logger_.csv << "," << data->qvel[i];
    }
    auto readings = skin.ComputeAllCapacitances(model, data);
    auto sensor_ids = skin.SensorIds();   // deterministic ordering captured at init
    for (int sid : sensor_ids) {
        double reading = 0.0;
        auto it = readings.find(sid);
        if (it != readings.end()) reading = it->second;
        logger_.csv << "," << reading;
    }

    // commanded velocity (we stored the command in obstacle_velocity_ at launch)
    logger_.csv << "," << obstacle_velocity_[0] << "," << obstacle_velocity_[1] << "," << obstacle_velocity_[2] << "\n";
  }

}

// Replace ResetLocked with this version (keeps your radius randomization)
void Avoid::ResetLocked(const mjModel *model) {
  // Read configuration numeric parameters
  for (int i = 0; i < model->nnumeric; i++) {
    const char* name = mj_id2name(model, mjOBJ_NUMERIC, i);
    if (!name) continue;
    
    int adr = model->numeric_adr[i];  // GET THE ADDRESS!
    
    if (strcmp(name, "logging_enabled") == 0) {
      logging_enabled_ = (bool)model->numeric_data[adr];
    } else if (strcmp(name, "use_offline_obstacles") == 0) {
      use_offline_obstacles_ = (bool)model->numeric_data[adr];
    } else if (strcmp(name, "use_tof_sensors") == 0) {
      use_tof_sensors_ = (bool)model->numeric_data[adr];
    } else if (strcmp(name, "use_cap_sensors") == 0) {
      use_cap_sensors_ = (bool)model->numeric_data[adr];
    } else if (strcmp(name, "tof_sensor_range") == 0) {
      tof_sensor_range_ = model->numeric_data[adr];
    } else if (strcmp(name, "cap_sensor_range") == 0) {
      cap_sensor_range_ = model->numeric_data[adr];
    }
  }

  std::printf("Avoid config: logging=%d, offline=%d, tof=%d, cap=%d, tof_range=%.2f, cap_range=%.2f\n",
              logging_enabled_, use_offline_obstacles_, use_tof_sensors_,
              use_cap_sensors_, tof_sensor_range_, cap_sensor_range_);

  // Reset obstacle state
  obstacle_launched_ = false;
  min_obstacle_dist_ = 1.0e6;

  // Randomize obstacle size
  int obstacle_geom_id = mj_name2id(model, mjOBJ_GEOM, "obstacle_geom");
  if (obstacle_geom_id >= 0) {
    // Accessing mutable model is risky but necessary for this effect.
    // This is safe during reset.
    mjModel* mutable_model = const_cast<mjModel*>(model);
    double min_radius = 0.03;
    double max_radius = 0.15;
    std::uniform_real_distribution<double> radius_dist(min_radius, max_radius);
    mutable_model->geom_size[3 * obstacle_geom_id] =
        radius_dist(generator);
  }

  // DEBUG
  // printf("\nJoint order for qpos:\n");
  // for (int i = 0; i < model->njnt; i++) {
  //   const char* jnt_name = mj_id2name(model, mjOBJ_JOINT, i);
  //   int qpos_adr = model->jnt_qposadr[i];
  //   printf("  Joint %d: %s (qpos index %d)\n", i, jnt_name ? jnt_name : "unnamed", qpos_adr);
  // }

  // ** Logging **
  int obstacle_body = mj_name2id(model, mjOBJ_BODY, "obstacle");
  int obstacle_jnt = model->body_jntadr[obstacle_body];
  int obstacle_qpos0 = model->jnt_qposadr[obstacle_jnt];
  int obstacle_qvel0 = model->jnt_dofadr[obstacle_jnt];

  static const int OBST_QPOS_DOF = 7;
  static const int OBST_QVEL_DOF = 6;

  sim_time_per_step_ = model->opt.timestep;
  logger_.out_dir = "./traj_logs";
  // ensure directory exists (posix)
  system("mkdir -p ./traj_logs");

  // open and write header
  char buf[256];
  // append epoch time to name
  time_t now = time(nullptr);
  std::snprintf(buf, sizeof(buf), "%s/episode_%04d_%ld.csv", logger_.out_dir.c_str(), logger_.episode_idx, now);
  logger_.csv_path = buf;
  logger_.csv.open(logger_.csv_path, std::ios::out);
  if (!logger_.csv.is_open()) {
    printf("[Avoid] Failed to open logger file %s\n", logger_.csv_path.c_str());
  } else {
    // Build header dynamically based on model sizes (nq,nv,nsensor)
    logger_.csv << "step,time,obst_px,obst_py,obst_pz,obst_vx,obst_vy,obst_vz,obst_radius";
    // robot qpos
    for (int i = 0; i < model->nq; i++) {
        if (i >= obstacle_qpos0 && i < obstacle_qpos0 + OBST_QPOS_DOF)
            continue;   // skip freejoint qpos
        logger_.csv << ",qpos_" << i;
    }
    for (int i = 0; i < model->nv; i++) {
        if (i >= obstacle_qvel0 && i < obstacle_qvel0 + OBST_QVEL_DOF)
            continue;   // skip freejoint qvel
        logger_.csv << ",qvel_" << i;
    }
    // sensors
    static CapacitiveSkin cap(model);
    static bool initialized = false;
    if (!initialized) {
      cap.RegisterAllSkinSites();
      initialized = true;
    }
    auto site_ids = cap.SensorIds();
    // Write header columns for capacitance sites
    for (int sid : site_ids) {
        const char* name = mj_id2name(model, mjOBJ_SITE, sid);
        if (name) {
            logger_.csv << ",cap_site_" << sid << "_" << name;
        } else {
            logger_.csv << ",cap_site_" << sid;
        }
    }

    // commanded velocity (log what you commanded at launch)
    logger_.csv << ",cmd_vx,cmd_vy,cmd_vz\n";
    logger_.csv << std::fixed << std::setprecision(4);
  }
}
}  // namespace mjpc