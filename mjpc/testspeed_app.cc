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

#include <string>

#include <absl/flags/parse.h>
#include <absl/flags/flag.h>

#include "mjpc/testspeed.h"
#include "mjpc/utilities.h"

ABSL_FLAG(std::string, task, "Cube Solving", "Which model to load on startup.");
ABSL_FLAG(int, planner_thread, mjpc::NumAvailableHardwareThreads() - 5,
          "Number of planner threads to use.");
ABSL_FLAG(int, steps_per_planning_iteration, 4,
          "How many physics steps to take between planning iterations.");
ABSL_FLAG(double, total_time, 10, "Total time to simulate (seconds).");
ABSL_FLAG(int, dump_residual, -1,
          "If >= 0, take this many planning steps from the model's default "
          "keyframe, print every mjSENS_USER residual scalar as "
          "idx<TAB>name[off]<TAB>value at %.17g, and exit. 0 dumps at the "
          "untouched keyframe, before any (stochastic) planning has run, which "
          "makes the dump a fully deterministic oracle. Negative disables.");
ABSL_FLAG(double, dump_perturb, 0.0,
          "With --dump_residual, first kick qpos/qvel off the keyframe by this "
          "magnitude using a fixed-seed deterministic generator. At the pristine "
          "keyframe most of the residual is identically zero and a zero term "
          "cannot disagree, so this is what gives the dump discriminating power.");
ABSL_FLAG(std::string, dump_traj, "",
          "If non-empty, write the rollout to this path as CSV: one row per "
          "physics step with time, cost, qpos, qvel, ctrl and actuator_force. "
          "Lets a run be scored with the same static analysis as an offline "
          "solve instead of only by its average cost.");
ABSL_FLAG(int, strategy, -1,
          "Task strategy parameter (parameter index 1 -- the GUI's Strategy "
          "slider). -1 leaves the model's own default. For the lean task: 6 = "
          "stand, 21 = reach, 22 = forearm brace, 34 = the 4-phase mission.");
ABSL_FLAG(std::string, start_qpos, "",
          "Path to a file of nq numbers used as the starting configuration, "
          "applied after --start_key. Lets a run begin at a pose computed "
          "outside the model (e.g. the offline QP solution).");
ABSL_FLAG(std::string, start_key, "",
          "Keyframe to start from. Empty = the model's 'home' key (the existing "
          "behaviour). A named key that does not exist is an error, not a "
          "silent fallback.");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  std::string task_name = absl::GetFlag(FLAGS_task);
  int planner_thread_count = absl::GetFlag(FLAGS_planner_thread);
  int steps_per_planning_iteration =
      absl::GetFlag(FLAGS_steps_per_planning_iteration);
  double total_time = absl::GetFlag(FLAGS_total_time);
  int dump_residual_steps = absl::GetFlag(FLAGS_dump_residual);
  double dump_perturb = absl::GetFlag(FLAGS_dump_perturb);
  std::string dump_traj = absl::GetFlag(FLAGS_dump_traj);
  std::string start_key = absl::GetFlag(FLAGS_start_key);
  double cost = mjpc::SynchronousPlanningCost(
      task_name, planner_thread_count, steps_per_planning_iteration, total_time,
      dump_residual_steps, dump_perturb, dump_traj, start_key,
      absl::GetFlag(FLAGS_strategy), absl::GetFlag(FLAGS_start_qpos));
  if (cost < 0) {
    return -1;
  }
  return 0;
}
