#include "mjpc/tasks/humanoid_bench/avoid/avoid.h"

#include <algorithm>
#include <cmath>
#include <random>

#include "mujoco/mujoco.h"

namespace mjpc {
// ------------------ Residuals for humanoid stand task ------------
//   Number of residuals:
//      Residual(0): 1 - humanoid_bench reward
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
  static CapacitiveSkin cap(model);
  static bool initialized = false;
  if (!initialized) {
    cap.RegisterAllSkinSites();
    initialized = true;
  }

  auto readings = cap.ComputeAllCapacitances(model, data);

  // for (auto &p : readings) {
  //   int sid = p.first;
  //   double val = p.second;
  //   std::cout << "Sensor " << sid << ": " << val << std::endl;
  //   // example residual: inverse distance
  //   // residual[sid] = (val == std::numeric_limits<double>::infinity()) ? 0.0 : 1.0 / val;
  // }

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
  double reward = -50.0 * avg_danger;  // Penalty for proximity
  //--------------- End of reward calculation -----------------//

  residual[counter++] = success - reward;

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
  const auto& sensor_ids = cap.SensorIds();
  for (size_t i = 0; i < sensor_ids.size(); ++i) {
    int sid = sensor_ids[i];
    auto it = readings.find(sid);

    if (it != readings.end() && it->second > 0) {
      // Penalty: inverse distance squared (stronger when close)
      double capacitance = it->second;
      residual[counter] = capacitance * capacitance;  // (1/d)^2
    } else {
      residual[counter] = 0.0;  // No obstacle detected
    }
    counter++;
  }

  // ----- CoM Away From Obstacle ----- //
  // Encourage CoM to shift away from obstacle centroid (xy plane only)
  if (total_weight > 0) {
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

  double* pos = data->qpos + qpos_adr;

  // Apply accumulated movement
  pos[0] += obstacle_move_x_;
  pos[1] += obstacle_move_y_;
  pos[2] += obstacle_move_z_;

  // Clear movement commands
  obstacle_move_x_ = 0.0;
  obstacle_move_y_ = 0.0;
  obstacle_move_z_ = 0.0;
}

void Avoid::ResetLocked(const mjModel *model) {
  // DEBUG
  printf("\nJoint order for qpos:\n");
  for (int i = 0; i < model->njnt; i++) {
    const char* jnt_name = mj_id2name(model, mjOBJ_JOINT, i);
    int qpos_adr = model->jnt_qposadr[i];
    printf("  Joint %d: %s (qpos index %d)\n", i, jnt_name ? jnt_name : "unnamed", qpos_adr);
  }
}
}  // namespace mjpc