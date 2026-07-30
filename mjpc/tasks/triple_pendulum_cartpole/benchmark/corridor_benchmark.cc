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
//
// Task-level benchmark for the triple-pendulum-cartpole corridor task.
//
// `testspeed` reports only an average cost per step, which cannot distinguish
// "swung up but never left the start" from "threaded the corridor and reached
// the goal". This harness reports what the Caldwell & Correll setup actually
// asks for: did the cart get through the obstacle corridor to the goal with the
// pendulum re-erected, and was any obstacle penetrated on the way.
//
// The combined task confounds two capabilities. A planner that never reaches
// the goal upright might be failing to thread the corridor, or failing to
// capture the pendulum once past it, and one aggregate number cannot say
// which. --stage splits them:
//
//   corridor  obstacles, goal x=6, Upright and Velocity weights zeroed.
//             Only asks: can the cart be driven to the goal without putting a
//             head inside a disk? Pure obstacle avoidance under the same
//             underactuated dynamics.
//   balance   no obstacles, goal x=0, started fully hung. Only asks: can this
//             planner erect a 3-link underactuated pendulum and hold it?
//   combined  the paper's task, unchanged: thread the corridor, then capture.
//
// A planner that passes both isolated stages and fails combined is failing at
// the composition, which is a different diagnosis from failing a component.
//
// Usage:
//   corridor_benchmark --planner=0 --stage=combined --total_time=20 --repeats=3

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <mujoco/mujoco.h>

#include "mjpc/agent.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/tasks/tasks.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

ABSL_FLAG(int, planner, -1,
          "Planner index to force (overrides agent_planner in the XML). "
          "0=PredictiveSampling 1=Gradient 2=iLQG 3=iLQS 4=RobustSampling "
          "5=CrossEntropy 6=SampleGradient 7=PSO 8=AnnealedSampling "
          "9=RandomShooting. -1 keeps the XML value.");
ABSL_FLAG(std::string, stage, "combined",
          "Which capability to isolate: 'corridor' (obstacle avoidance only), "
          "'balance' (swing-up and capture only), or 'combined' (the full "
          "task). See the file comment.");
ABSL_FLAG(double, total_time, 20.0, "Simulated seconds per repeat.");
ABSL_FLAG(int, repeats, 1, "Number of repeats.");
ABSL_FLAG(int, planner_thread, mjpc::NumAvailableHardwareThreads() - 5,
          "Number of planner threads.");
ABSL_FLAG(int, steps_per_planning_iteration, 2,
          "Simulation steps between planning iterations.");
ABSL_FLAG(double, goal_tolerance, 0.3,
          "Cart position tolerance (m) counted as reaching the goal.");
ABSL_FLAG(double, upright_tolerance, 0.95,
          "Minimum cos(theta_i) at the end counted as upright.");
ABSL_FLAG(double, penetration_tolerance, 0.02,
          "Obstacle penetration depth (m) beyond which a run counts as a "
          "collision. MuJoCo's soft contacts always allow a few mm on touch, "
          "so a hard zero threshold would flag harmless grazing.");
ABSL_FLAG(double, contact_fraction_tolerance, 0.02,
          "Fraction of steps in contact beyond which a run counts as a "
          "collision, independent of depth. Distinguishes a brief graze from "
          "dragging along an obstacle.");
ABSL_FLAG(double, speed_tolerance, 1.0,
          "Max ||qvel|| counted as 'at rest' in the goal set. Without this, a "
          "pendulum swinging THROUGH vertical at 5 rad/s scores as solved: "
          "measured median speed in the goal configuration is 4.7 rad/s, so "
          "the configuration test alone is not a solve.");
ABSL_FLAG(int, pso_publish_evaluated, -1,
          "PSO only. 1: publish the parameters that were rolled out. 0: "
          "publish the post-swarm-update iterate, whose cost was never "
          "evaluated (the original behaviour). -1 keeps the XML value.");

namespace {

enum class Stage { kCorridor, kBalance, kCombined };

Stage ParseStage(const std::string& s) {
  if (s == "corridor") return Stage::kCorridor;
  if (s == "balance") return Stage::kBalance;
  if (s == "combined") return Stage::kCombined;
  std::cerr << "unknown --stage '" << s
            << "'; expected corridor, balance, or combined\n";
  std::exit(1);
}

// Set a cost term's weight by sensor name. Task::Initialize reads weights out
// of model->sensor_user (see mjpc/task.cc, "user data: [norm, weight, ...]"),
// so editing the model before Agent::Initialize is enough -- and it reaches
// the planner's internal rollouts too, which writing to task->weight after
// initialization would not do reliably.
void SetTermWeight(mjModel* model, const char* sensor_name, double weight) {
  int id = mj_name2id(model, mjOBJ_SENSOR, sensor_name);
  if (id < 0) mju_error_s("sensor '%s' not found", sensor_name);
  model->sensor_user[id * model->nuser_sensor + 1] = weight;
}

// Which keyframe a stage starts from, and how the model is altered for it.
// Returns the keyframe name.
const char* ConfigureStage(mjModel* model, Stage stage) {
  const char* obstacle_names[2] = {"obstacle_upper", "obstacle_lower"};

  switch (stage) {
    case Stage::kCombined:
      return "home";

    case Stage::kCorridor:
      // Zero the two terms that ask for capture, keeping Cart, Control and
      // Avoidance. What remains scores exactly one thing: reaching x=6 without
      // driving a head into a disk. The pendulum is free to flail, so the
      // dynamics stay as underactuated and chaotic as in the full task -- this
      // is an easier objective, not an easier system.
      SetTermWeight(model, "Upright", 0.0);
      SetTermWeight(model, "Velocity", 0.0);
      return "home";

    case Stage::kBalance:
      // Remove the corridor entirely: push the disks far off in y (so the
      // Avoidance residual, which measures distance in the x-z plane, still
      // sees them -- hence also clear their contact bits) and detach them from
      // collision. Goal moves to x=0 so the cart is not asked to travel.
      for (int i = 0; i < 2; i++) {
        int id = mj_name2id(model, mjOBJ_GEOM, obstacle_names[i]);
        if (id < 0) mju_error_s("geom '%s' not found", obstacle_names[i]);
        model->geom_pos[3 * id + 0] = 1e3;  // out of the x-z plane of motion
        model->geom_pos[3 * id + 2] = 1e3;
        model->geom_contype[id] = 0;
        model->geom_conaffinity[id] = 0;
      }
      if (double* g = mjpc::GetCustomNumericData(model, "residual_Goal")) {
        *g = 0.0;
      }
      return "hanging";
  }
  return "home";
}

mjpc::Task* g_task;
void ResidualCallback(const mjModel* model, mjData* data, int stage) {
  if (stage == mjSTAGE_ACC) {
    g_task->Residual(model, data, data->sensordata);
  }
}

struct RunResult {
  double final_cart = 0;
  double min_cos_upright = 0;   // worst link at the end
  double max_cart = 0;          // furthest right the cart ever got
  double min_clearance = 1e9;   // min over run of (dist to disk centre - radius)
  int contact_steps = 0;
  double contact_fraction = 0;
  double mean_cost = 0;
  double final_speed = 0;
  double wall_time = 0;
  bool reached_goal = false;
  bool upright = false;
  bool collided = false;
  bool passed_corridor = false;

  // This is a planning problem: the question is whether the goal state was ever
  // reached, not only whether the run happened to end there. Judging the final
  // state alone scores a run that reaches the goal upright at t=2.5s and then
  // loses balance as a total failure, which is the wrong reading -- the plan
  // succeeded and the terminal controller did not.
  bool ever_solved = false;     // goal + upright simultaneously, at any step
  double first_solve_time = -1;
  double solved_fraction = 0;   // fraction of steps in the solved set
  int solved_steps = 0;
};

RunResult RunOnce(const std::string& task_name, int planner_override,
                  Stage stage, int pso_publish_evaluated,
                  double total_time, int planner_thread_count,
                  int steps_per_planning_iteration, double goal_tolerance,
                  double upright_tolerance, double penetration_tolerance,
                  double contact_fraction_tolerance,
                  double speed_tolerance) {
  RunResult r;

  mjpc::Agent agent;
  agent.SetTaskList(mjpc::GetTasks());
  agent.gui_task_id = agent.GetTaskIdByName(task_name);
  if (agent.gui_task_id == -1) {
    std::cerr << "Invalid task '" << task_name << "'. Valid values:\n"
              << agent.GetTaskNames();
    std::exit(1);
  }
  mjpc::Agent::LoadModelResult load_model = agent.LoadModel();
  mjModel* model = load_model.model.get();
  if (!model) {
    std::cerr << load_model.error << "\n";
    std::exit(1);
  }

  // Override the planner selected by the XML. Agent::Initialize reads
  // "agent_planner" from the model's custom numerics, so writing it here is
  // enough and avoids touching Agent's private state.
  if (planner_override >= 0) {
    double* p = mjpc::GetCustomNumericData(model, "agent_planner");
    if (p) *p = static_cast<double>(planner_override);
  }

  if (pso_publish_evaluated >= 0) {
    double* p = mjpc::GetCustomNumericData(model, "pso_publish_evaluated");
    if (p) *p = static_cast<double>(pso_publish_evaluated);
  }

  // Alter the model for the stage before Agent::Initialize reads it, so the
  // planner optimizes the same objective the run is scored against.
  const char* key_name = ConfigureStage(model, stage);

  mjData* data = mj_makeData(model);
  int key_id = mj_name2id(model, mjOBJ_KEY, key_name);
  if (key_id >= 0) mj_resetDataKeyframe(model, data, key_id);
  mj_forward(model, data);

  agent.estimator_enabled = false;
  agent.Initialize(model);
  agent.Allocate();
  agent.Reset(data->ctrl);
  agent.plan_enabled = true;

  g_task = agent.ActiveTask();
  mjcb_sensor = &ResidualCallback;

  // ---- model ids for the outcome metrics ----
  const char* head_names[3] = {"head1", "head2", "tip"};
  int head_site[3];
  for (int i = 0; i < 3; i++) {
    head_site[i] = mj_name2id(model, mjOBJ_SITE, head_names[i]);
  }
  const char* obstacle_names[2] = {"obstacle_upper", "obstacle_lower"};
  int obstacle_geom[2];
  double obstacle_radius[2];
  for (int i = 0; i < 2; i++) {
    obstacle_geom[i] = mj_name2id(model, mjOBJ_GEOM, obstacle_names[i]);
    obstacle_radius[i] = model->geom_size[3 * obstacle_geom[i]];
  }
  double goal = mjpc::GetNumberOrDefault(6.0, model, "residual_Goal");

  mjpc::ThreadPool pool(planner_thread_count);
  int total_steps = std::ceil(total_time / model->opt.timestep);
  double total_cost = 0;

  auto loop_start = std::chrono::steady_clock::now();
  for (int i = 0; i < total_steps; i++) {
    agent.ActiveTask()->Transition(model, data);
    agent.state.Set(model, data);
    agent.ActivePlanner().ActionFromPolicy(data->ctrl,
                                           agent.state.state().data(),
                                           agent.state.time(),
                                           /*use_previous=*/false);
    mj_step(model, data);
    total_cost += agent.ActiveTask()->CostValue(data->sensordata);

    if (i % steps_per_planning_iteration == 0) agent.PlanIteration(&pool);

    // ---- outcome metrics ----
    r.max_cart = std::max(r.max_cart, data->qpos[0]);
    if (data->ncon > 0) r.contact_steps++;
    double step_min_clearance = 1e9;
    for (int h = 0; h < 3; h++) {
      const double* head = data->site_xpos + 3 * head_site[h];
      for (int o = 0; o < 2; o++) {
        const double* obs = data->geom_xpos + 3 * obstacle_geom[o];
        double dx = head[0] - obs[0], dz = head[2] - obs[2];
        double clearance = std::sqrt(dx * dx + dz * dz) - obstacle_radius[o];
        step_min_clearance = std::min(step_min_clearance, clearance);
      }
    }
    r.min_clearance = std::min(r.min_clearance, step_min_clearance);

    // ---- was the goal state reached at this step? ----
    double step_min_cos = 1.0;
    for (int j = 0; j < 3; j++) {
      step_min_cos = std::min(step_min_cos, std::cos(data->qpos[1 + j]));
    }
    double speed = mju_norm(data->qvel, model->nv);
    // The corridor stage is not asked to capture anything, so requiring the
    // pendulum to be upright and at rest there would score it against an
    // objective it was not optimizing -- its Upright and Velocity weights are
    // zero. Reaching the goal box is the whole test.
    bool in_goal_config = stage == Stage::kCorridor
                              ? true
                              : (step_min_cos > upright_tolerance &&
                                 speed < speed_tolerance);
    if (std::abs(data->qpos[0] - goal) < goal_tolerance && in_goal_config) {
      r.solved_steps++;
      if (!r.ever_solved) {
        r.ever_solved = true;
        r.first_solve_time = data->time;
      }
    }
  }
  r.wall_time = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - loop_start)
                    .count() / 1e6;

  r.final_cart = data->qpos[0];
  r.min_cos_upright = 1.0;
  for (int i = 0; i < 3; i++) {
    r.min_cos_upright = std::min(r.min_cos_upright, std::cos(data->qpos[1 + i]));
  }
  r.mean_cost = total_cost / total_steps;
  r.contact_fraction = static_cast<double>(r.contact_steps) / total_steps;
  r.solved_fraction = static_cast<double>(r.solved_steps) / total_steps;
  r.final_speed = mju_norm(data->qvel, model->nv);
  r.reached_goal = std::abs(r.final_cart - goal) < goal_tolerance;
  r.upright = stage == Stage::kCorridor ||
              (r.min_cos_upright > upright_tolerance &&
               r.final_speed < speed_tolerance);
  // Two independent ways to fail the obstacle constraint: go deep into a disk
  // once, or ride along one for a sustained fraction of the run.
  r.collided = r.min_clearance < -penetration_tolerance ||
               r.contact_fraction > contact_fraction_tolerance;
  r.passed_corridor = r.max_cart > 3.5;

  mj_deleteData(data);
  mjcb_sensor = nullptr;
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const std::string task_name = "Triple Pendulum Cartpole";
  int planner = absl::GetFlag(FLAGS_planner);
  int repeats = absl::GetFlag(FLAGS_repeats);
  const std::string stage_name = absl::GetFlag(FLAGS_stage);
  Stage stage = ParseStage(stage_name);

  // Index 0 is Predictive Sampling, not MPPI: it publishes the single best
  // candidate (mjpc/planners/sampling/planner.cc CopyCandidateToPolicy), not
  // an exponentially-weighted average of them.
  static const char* kPlannerLabel[] = {
      "PredictiveSampling", "Gradient", "iLQG", "iLQS",
      "RobustSampling", "CrossEntropy", "SampleGradient",
      "PSO", "AnnealedSampling", "RandomShooting"};
  constexpr int kNumPlannerLabels =
      sizeof(kPlannerLabel) / sizeof(kPlannerLabel[0]);

  std::printf("task: %s   stage: %s   planner: %s   %.0fs x %d repeat(s)\n",
              task_name.c_str(), stage_name.c_str(),
              (planner >= 0 && planner < kNumPlannerLabels)
                  ? kPlannerLabel[planner]
                  : "(xml)",
              absl::GetFlag(FLAGS_total_time), repeats);
  std::printf("%-4s %7s %6s %8s %8s %8s %7s %7s  %s\n", "run", "t_solve",
              "held%", "cart_end", "cos_end", "spd_end", "cont%", "cost",
              "outcome");

  int solved = 0, corridor = 0, collided = 0;
  int held_to_end = 0;
  for (int k = 0; k < repeats; k++) {
    RunResult r = RunOnce(task_name, planner, stage,
                          absl::GetFlag(FLAGS_pso_publish_evaluated),
                          absl::GetFlag(FLAGS_total_time),
                          absl::GetFlag(FLAGS_planner_thread),
                          absl::GetFlag(FLAGS_steps_per_planning_iteration),
                          absl::GetFlag(FLAGS_goal_tolerance),
                          absl::GetFlag(FLAGS_upright_tolerance),
                          absl::GetFlag(FLAGS_penetration_tolerance),
                          absl::GetFlag(FLAGS_contact_fraction_tolerance),
                          absl::GetFlag(FLAGS_speed_tolerance));
    // "reached" = the goal state was attained at some point without a
    // disqualifying collision. "held" = it was still there at the end.
    bool reached = r.ever_solved && !r.collided;
    bool held = r.reached_goal && r.upright && !r.collided;
    solved += reached;
    held_to_end += held;
    corridor += r.passed_corridor;
    collided += r.collided;

    // describe the dominant outcome, most informative first
    const char* outcome;
    if (stage == Stage::kBalance) {
      if (held) {
        outcome = "SOLVED, held to end";
      } else if (reached) {
        outcome = "erected, then fell";
      } else if (r.min_cos_upright > 0.0) {
        outcome = "partly erected, never captured";
      } else {
        outcome = "never left hanging";
      }
    } else if (held) {
      outcome = "SOLVED, held to end";
    } else if (reached) {
      outcome = stage == Stage::kCorridor ? "SOLVED, drifted off goal"
                                          : "SOLVED, then lost balance";
    } else if (r.collided && r.ever_solved) {
      outcome = "reached goal but collided";
    } else if (r.collided) {
      outcome = "collided";
    } else if (!r.passed_corridor) {
      outcome = "stuck before corridor";
    } else if (stage == Stage::kCorridor) {
      outcome = "past corridor, never reached goal";
    } else {
      outcome = "past corridor, never at goal upright";
    }
    if (r.first_solve_time >= 0) {
      std::printf("%-4d %7.2f %5.1f%% %8.3f %8.3f %8.2f %6.1f%% %7.2f  %s\n", k,
                  r.first_solve_time, 100.0 * r.solved_fraction, r.final_cart,
                  r.min_cos_upright, r.final_speed,
                  100.0 * r.contact_fraction, r.mean_cost, outcome);
    } else {
      std::printf("%-4d %7s %5.1f%% %8.3f %8.3f %8.2f %6.1f%% %7.2f  %s\n", k,
                  "never", 100.0 * r.solved_fraction, r.final_cart,
                  r.min_cos_upright, r.final_speed,
                  100.0 * r.contact_fraction, r.mean_cost, outcome);
    }
  }
  if (stage == Stage::kBalance) {
    std::printf("summary[balance]: erected %d/%d | held to end %d/%d\n",
                solved, repeats, held_to_end, repeats);
    std::printf("  (goal set = cart within %.2f m of x=0, all cos(theta) > "
                "%.2f, ||qvel|| < %.2f)\n",
                absl::GetFlag(FLAGS_goal_tolerance),
                absl::GetFlag(FLAGS_upright_tolerance),
                absl::GetFlag(FLAGS_speed_tolerance));
  } else {
    std::printf("summary[%s]: reached goal state %d/%d | held to end %d/%d | "
                "past corridor %d/%d | collided %d/%d\n",
                stage_name.c_str(), solved, repeats, held_to_end, repeats,
                corridor, repeats, collided, repeats);
    if (stage == Stage::kCorridor) {
      std::printf("  (goal set = cart within %.2f m of x=6; Upright and "
                  "Velocity weights are zero for this stage, so pendulum "
                  "state is not scored)\n",
                  absl::GetFlag(FLAGS_goal_tolerance));
    } else {
      std::printf("  (goal set = cart within %.2f m, all cos(theta) > %.2f, "
                  "||qvel|| < %.2f;\n   t_solve = first entry, held%% = "
                  "fraction of the run inside it)\n",
                  absl::GetFlag(FLAGS_goal_tolerance),
                  absl::GetFlag(FLAGS_upright_tolerance),
                  absl::GetFlag(FLAGS_speed_tolerance));
    }
  }
  return 0;
}
