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

#include "mjpc/testspeed.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "mjpc/agent.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"
#include "mjpc/tasks/tasks.h"

namespace mjpc {

namespace {
Task* task;
void residual_callback(const mjModel* model, mjData* data, int stage) {
  if (stage == mjSTAGE_ACC) {
    task->Residual(model, data, data->sensordata);
  }
}

// Print every mjSENS_USER residual SCALAR of `data->sensordata`, in sensor
// declaration order, as `idx<TAB>name[off]<TAB>value` at full %.17g precision.
// `idx` is the scalar's position in the task residual vector (a running counter
// over user sensors only), `off` its offset inside the owning sensor, so a span
// that has SHIFTED shows up immediately instead of masquerading as 20 value
// mismatches. The trailing fflush is not optional: buffered stdout plus a killed
// process has already produced checks in this project that could not fail.
void DumpUserResidual(const mjModel* model, const mjData* data,
                      const std::string& task_name, int steps, double perturb) {
  int n_user = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) n_user += model->sensor_dim[i];
  }
  // Leading newline: task loaders print with std::printf and do not always
  // terminate their last line, which would otherwise glue the marker onto it.
  // nq/nv are in the header so a comparison across two models can VERIFY that
  // the perturbation below drew the same numbers into the same slots.
  std::printf("\nRESIDUAL_DUMP task=%s steps=%d perturb=%.17g nq=%d nv=%d "
              "nscalar=%d time=%.17g\n",
              task_name.c_str(), steps, perturb, model->nq, model->nv, n_user,
              data->time);
  int idx = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] != mjSENS_USER) continue;
    const char* name = mj_id2name(model, mjOBJ_SENSOR, i);
    int adr = model->sensor_adr[i];
    for (int off = 0; off < model->sensor_dim[i]; off++) {
      std::printf("%d\t%s[%d]\t%.17g\n", idx, name ? name : "(unnamed)", off,
                  data->sensordata[adr + off]);
      idx++;
    }
  }
  std::printf("RESIDUAL_DUMP_END\n");
  std::fflush(stdout);
}

// Deterministically shove the state off the keyframe.
//
// At the pristine keyframe most of the residual is identically zero (the pose IS
// the target and the robot is at rest), so a bit-identical dump there proves very
// little: a term can only disagree where it is nonzero. This walks a fixed-seed
// xorshift64* over qpos and qvel so that velocity-, asymmetry- and tilt-driven
// terms all become nonzero, while staying a pure function of the seed -- run it
// twice, or on two engines whose nq/nv agree, and you get the same state to the
// last bit. Quaternions are renormalised after the kick.
void PerturbState(const mjModel* model, mjData* data, double scale) {
  uint64_t s = 0x2545F4914F6CDD1DULL;
  auto next = [&s]() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    // 53-bit mantissa -> [-1, 1)
    return static_cast<double>(s >> 11) * (1.0 / 4503599627370496.0) - 1.0;
  };
  for (int i = 0; i < model->nq; i++) data->qpos[i] += scale * next();
  for (int i = 0; i < model->nv; i++) data->qvel[i] += scale * next();
  mj_normalizeQuat(model, data->qpos);
}
}  // namespace

// Run synchronous planning, print timing info,return 0 if nothing failed.
double SynchronousPlanningCost(std::string task_name, int planner_thread_count,
                               int steps_per_planning_iteration,
                               double total_time, int dump_residual_steps,
                               double dump_perturb) {
  std::cout << "Test MJPC Speed: " << task_name << "\n";
  std::cout << " MuJoCo version " << mj_versionString() << "\n";
  if (mjVERSION_HEADER != mj_version()) {
    mju_error("Headers and library have different versions");
  }
  std::cout << " Hardware threads:  " << NumAvailableHardwareThreads() << "\n";

  Agent agent;
  agent.SetTaskList(GetTasks());
  agent.gui_task_id = agent.GetTaskIdByName(task_name);
  if (agent.gui_task_id == -1) {
    std::cerr << "Invalid --task flag: '" << task_name
              << "'. Valid values:\n";
    std::cerr << agent.GetTaskNames();
    return -1;
  }
  Agent::LoadModelResult load_model = agent.LoadModel();
  mjModel* model = load_model.model.get();
  if (!model) {
    std::cerr << load_model.error << "\n";
    return -1;
  }
  mjData* data = mj_makeData(model);

  int home_id = mj_name2id(model, mjOBJ_KEY, "home");
  if (home_id >= 0) {
    std::cout << "home_id: " << home_id << "\n";
    mj_resetDataKeyframe(model, data, home_id);
  }
  mj_forward(model, data);

  // the planner and its initial configuration is set in the XML
  agent.estimator_enabled = false;
  agent.Initialize(model);
  agent.Allocate();
  agent.Reset(data->ctrl);
  agent.plan_enabled = true;

  // make task available for global callback:
  task = agent.ActiveTask();
  mjcb_sensor = &residual_callback;

  std::cout << " Planning threads:  " << planner_thread_count << "\n";
  ThreadPool pool(planner_thread_count);

  // Residual dump mode -- the deterministic oracle. The residual is a pure
  // function of (model, data), so with dump_residual_steps == 0 nothing
  // stochastic has run yet and two engines at the same strategy must agree
  // bit-for-bit on every surviving span. dump_residual_steps > 0 lets the
  // sampling planner move the state first, which is only comparable across
  // engines when they happen to sample the same trajectory prefix.
  if (dump_residual_steps >= 0) {
    if (dump_perturb > 0.0) PerturbState(model, data, dump_perturb);
    for (int i = 0; i < dump_residual_steps; i++) {
      agent.ActiveTask()->Transition(model, data);
      agent.state.Set(model, data);
      agent.ActivePlanner().ActionFromPolicy(
          data->ctrl, agent.state.state().data(),
          agent.state.time(), /*use_previous=*/false);
      mj_step(model, data);
      if (i % steps_per_planning_iteration == 0) { agent.PlanIteration(&pool); }
    }
    // Recompute the sensors at THIS state: mj_step evaluates sensors before it
    // integrates, so without this the dump would describe the previous state.
    agent.ActiveTask()->Transition(model, data);
    mj_forward(model, data);
    DumpUserResidual(model, data, task_name, dump_residual_steps, dump_perturb);
    mj_deleteData(data);
    mjcb_sensor = nullptr;
    return 0;
  }

  int total_steps = ceil(total_time / model->opt.timestep);
  int current_time = 0;
  double total_cost = 0;
  auto loop_start = std::chrono::steady_clock::now();
  for (int i = 0; i < total_steps; i++) {
    agent.ActiveTask()->Transition(model, data);
    agent.state.Set(model, data);

    agent.ActivePlanner().ActionFromPolicy(
        data->ctrl, agent.state.state().data(),
        agent.state.time(), /*use_previous=*/false);
    mj_step(model, data);
    double cost = agent.ActiveTask()->CostValue(data->sensordata);
    total_cost += cost;

    if (i % steps_per_planning_iteration == 0) { agent.PlanIteration(&pool); }

    if (floor(data->time) > current_time) {
      current_time++;
      std::cout << "sim time: " << current_time << ", cost: " << cost << "\n";
    }
  }
  auto wall_run_time = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - loop_start)
                            .count() /
                        1e6;
  std::cout << "Total wall time ("
            << (int)ceil(total_steps / steps_per_planning_iteration)
            << " planning steps): " << wall_run_time << " s ("
            << total_time / wall_run_time << "x realtime)\n";
  std::cout << "Average cost per step (lower is better): "
            << total_cost / total_steps << "\n";

  mj_deleteData(data);
  mjcb_sensor = nullptr;
  return total_cost;
}
}  // namespace mjpc
