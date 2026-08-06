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
#include <stdexcept>
#include <string>
#include <utility>
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

// Parse "t0:p0,t1:p1,..." into ascending (sim time, phase index) pairs. Returns
// false on any malformed or out-of-order entry: a schedule the harness silently
// reinterpreted would make a transition experiment describe a sequence nobody
// asked for.
bool ParsePhaseSchedule(const std::string& spec,
                        std::vector<std::pair<double, int>>* out) {
  size_t i = 0;
  while (i < spec.size()) {
    size_t comma = spec.find(',', i);
    if (comma == std::string::npos) comma = spec.size();
    const std::string item = spec.substr(i, comma - i);
    size_t colon = item.find(':');
    if (colon == std::string::npos) return false;
    try {
      const double t = std::stod(item.substr(0, colon));
      const int p = std::stoi(item.substr(colon + 1));
      if (!out->empty() && t < out->back().first) return false;
      out->push_back({t, p});
    } catch (const std::exception&) {
      return false;
    }
    i = comma + 1;
  }
  return !out->empty();
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
                               double dump_perturb, std::string dump_traj,
                               std::string start_key, int strategy,
                               std::string start_qpos,
                               std::string phase_schedule) {
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

  // --start_key names the keyframe to start from; "home" stays the default so
  // every existing invocation is byte-identical. A named key that does not
  // exist is an error rather than a silent fallback to home: quietly starting
  // somewhere else would make the run look like a result about the planner.
  const char* key_name = start_key.empty() ? "home" : start_key.c_str();
  int start_id = mj_name2id(model, mjOBJ_KEY, key_name);
  if (start_id < 0 && !start_key.empty()) {
    std::cerr << "Invalid --start_key '" << start_key << "': no such keyframe\n";
    mj_deleteData(data);
    return -1;
  }
  if (start_id >= 0) {
    std::cout << "start keyframe: " << key_name << " (id " << start_id << ")\n";
    mj_resetDataKeyframe(model, data, start_id);
  }
  // --start_qpos overrides the keyframe with an externally computed pose. Read
  // strictly: a short or long file is an error, because silently zero-padding a
  // configuration would start the robot somewhere nobody chose.
  if (!start_qpos.empty()) {
    std::FILE* f = std::fopen(start_qpos.c_str(), "r");
    if (!f) {
      std::cerr << "cannot open --start_qpos '" << start_qpos << "'\n";
      mj_deleteData(data);
      return -1;
    }
    std::vector<double> q;
    double v;
    while (std::fscanf(f, " %lf%*[ ,\t\n]", &v) == 1) q.push_back(v);
    std::fclose(f);
    if ((int)q.size() != model->nq) {
      std::cerr << "--start_qpos has " << q.size() << " values, model nq is "
                << model->nq << "\n";
      mj_deleteData(data);
      return -1;
    }
    for (int i = 0; i < model->nq; i++) data->qpos[i] = q[i];
    mju_zero(data->qvel, model->nv);
    std::cout << "start qpos from " << start_qpos << " (" << q.size()
              << " values)\n";
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

  // --strategy selects the task's strategy parameter (the GUI slider) so a
  // headless run can be pointed at something other than the model default.
  // Index 1 is kLeanStrategyParameterIndex; tasks that do not use parameter 1
  // as a strategy selector should not be passed this flag. -1 = leave the XML
  // default alone, which is what every existing invocation does.
  if (strategy >= 0) {
    if (task->parameters.size() > 1) {
      task->parameters[1] = strategy;
      std::cout << " strategy parameter: " << strategy << "\n";
    } else {
      std::cerr << "--strategy given but task has no parameter 1\n";
      mj_deleteData(data);
      mjcb_sensor = nullptr;
      return -1;
    }
  }

  // --phase_schedule drives parameter index 2 (the lean task's Phase slider) as
  // a function of sim time. Parameter 2 is the MANUAL phase scrubber: any value
  // >= 0 pins the keyframe and disables auto-advance, so the schedule is the
  // only thing sequencing the strategy. Left empty, the parameter keeps its XML
  // default and nothing here runs.
  std::vector<std::pair<double, int>> phases;
  size_t next_phase = 0;
  if (!phase_schedule.empty()) {
    if (task->parameters.size() <= 2) {
      std::cerr << "--phase_schedule given but task has no parameter 2\n";
      mj_deleteData(data);
      mjcb_sensor = nullptr;
      return -1;
    }
    if (!ParsePhaseSchedule(phase_schedule, &phases)) {
      std::cerr << "malformed --phase_schedule '" << phase_schedule
                << "': expected ascending \"t0:p0,t1:p1,...\"\n";
      mj_deleteData(data);
      mjcb_sensor = nullptr;
      return -1;
    }
    std::cout << " phase schedule:   ";
    for (const auto& tp : phases)
      std::cout << tp.first << "s->p" << tp.second << " ";
    std::cout << "\n";
  }

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

  // Trajectory dump. Written incrementally and flushed per row so a run that is
  // killed (or topples and is aborted) still leaves everything up to that point
  // on disk -- a truncated rollout is evidence, an empty file is not.
  std::FILE* traj = nullptr;
  if (!dump_traj.empty()) {
    traj = std::fopen(dump_traj.c_str(), "w");
    if (!traj) {
      std::cerr << "cannot open --dump_traj '" << dump_traj << "'\n";
      mj_deleteData(data);
      return -1;
    }
    std::fprintf(traj, "# task=%s nq=%d nv=%d nu=%d start_key=%s dt=%.17g\n",
                 task_name.c_str(), model->nq, model->nv, model->nu,
                 key_name, model->opt.timestep);
    std::fprintf(traj, "time,cost");
    for (int j = 0; j < model->nq; j++) std::fprintf(traj, ",qpos%d", j);
    for (int j = 0; j < model->nv; j++) std::fprintf(traj, ",qvel%d", j);
    for (int j = 0; j < model->nu; j++) std::fprintf(traj, ",ctrl%d", j);
    for (int j = 0; j < model->nu; j++) std::fprintf(traj, ",afrc%d", j);
    // `phase` is the commanded phase parameter, not an observation of it. Under
    // --phase_schedule they are the same thing (manual mode pins the keyframe);
    // without it the column reads the XML default and says nothing.
    std::fprintf(traj, ",phase\n");
  }

  int total_steps = ceil(total_time / model->opt.timestep);
  int current_time = 0;
  double total_cost = 0;
  auto loop_start = std::chrono::steady_clock::now();
  for (int i = 0; i < total_steps; i++) {
    // Advance the phase BEFORE Transition: Transition is what reads parameter 2
    // and swaps the keyframe, so setting it afterwards would apply one step late
    // and, at t == 0, leave the first Transition running the XML default.
    while (next_phase < phases.size() &&
           phases[next_phase].first <= data->time) {
      task->parameters[2] = phases[next_phase].second;
      std::cout << "phase -> " << phases[next_phase].second << " at t "
                << data->time << "\n";
      next_phase++;
    }
    agent.ActiveTask()->Transition(model, data);
    agent.state.Set(model, data);

    agent.ActivePlanner().ActionFromPolicy(
        data->ctrl, agent.state.state().data(),
        agent.state.time(), /*use_previous=*/false);
    mj_step(model, data);
    double cost = agent.ActiveTask()->CostValue(data->sensordata);
    total_cost += cost;

    if (i % steps_per_planning_iteration == 0) { agent.PlanIteration(&pool); }

    if (traj) {
      std::fprintf(traj, "%.17g,%.17g", data->time, cost);
      for (int j = 0; j < model->nq; j++)
        std::fprintf(traj, ",%.17g", data->qpos[j]);
      for (int j = 0; j < model->nv; j++)
        std::fprintf(traj, ",%.17g", data->qvel[j]);
      for (int j = 0; j < model->nu; j++)
        std::fprintf(traj, ",%.17g", data->ctrl[j]);
      for (int j = 0; j < model->nu; j++)
        std::fprintf(traj, ",%.17g", data->actuator_force[j]);
      std::fprintf(traj, ",%.17g\n",
                   task->parameters.size() > 2 ? task->parameters[2] : -1.0);
      std::fflush(traj);
    }

    if (floor(data->time) > current_time) {
      current_time++;
      std::cout << "sim time: " << current_time << ", cost: " << cost << "\n";
    }
  }
  if (traj) {
    std::fclose(traj);
    std::cout << "wrote trajectory: " << dump_traj << "\n";
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
