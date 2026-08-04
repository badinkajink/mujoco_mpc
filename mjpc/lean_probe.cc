// Lean-task torque probe (analysis tool, not part of the shipped app).
//
// Runs the lean task headless under MPC exactly the way testspeed.cc does, but
// logs the per-joint actuator torque, the table contact forces, and the CoM /
// support-polygon geometry every step to a CSV. Written to answer: "what joint
// torques does the H12 hold at the sustained braced lean?"
//
//   ./lean_probe --task "Lean H12" --strategy 4 --time 40 --out lean.csv

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
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

namespace {
mjpc::Task* g_task;
void residual_callback(const mjModel* model, mjData* data, int stage) {
  if (stage == mjSTAGE_ACC) {
    g_task->Residual(model, data, data->sensordata);
  }
}

std::string Arg(int argc, char** argv, const std::string& key,
                const std::string& fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) return argv[i + 1];
  }
  return fallback;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string task_name = Arg(argc, argv, "--task", "Lean H12");
  const double strategy = std::stod(Arg(argc, argv, "--strategy", "4"));
  const double total_time = std::stod(Arg(argc, argv, "--time", "40"));
  const std::string out_path = Arg(argc, argv, "--out", "lean_probe.csv");
  const int threads = std::stoi(Arg(argc, argv, "--threads", "8"));
  const int steps_per_plan = std::stoi(Arg(argc, argv, "--steps_per_plan", "1"));

  mjpc::Agent agent;
  agent.SetTaskList(mjpc::GetTasks());
  agent.gui_task_id = agent.GetTaskIdByName(task_name);
  if (agent.gui_task_id == -1) {
    std::cerr << "Invalid --task '" << task_name << "'. Valid:\n"
              << agent.GetTaskNames();
    return 1;
  }
  mjpc::Agent::LoadModelResult load = agent.LoadModel();
  mjModel* model = load.model.get();
  if (!model) {
    std::cerr << load.error << "\n";
    return 1;
  }
  mjData* data = mj_makeData(model);

  int home_id = mj_name2id(model, mjOBJ_KEY, "home");
  if (home_id >= 0) mj_resetDataKeyframe(model, data, home_id);
  mj_forward(model, data);

  agent.estimator_enabled = false;
  agent.Initialize(model);
  agent.Allocate();
  agent.Reset(data->ctrl);
  agent.plan_enabled = true;

  // Select the strategy (task parameter index 1, see lean.h
  // kLeanStrategyParameterIndex) before the first Transition.
  {
    std::vector<double>& params = agent.ActiveTask()->parameters;
    if (params.size() > 1) params[1] = strategy;
  }

  g_task = agent.ActiveTask();
  mjcb_sensor = &residual_callback;

  // Table geom id, so we can sum only robot<->table contact forces.
  int table_geom = mj_name2id(model, mjOBJ_GEOM, "table_top_collision");

  std::ofstream csv(out_path);
  csv << "time,phase,cost";
  for (int i = 0; i < model->nu; ++i) {
    const char* nm = mj_id2name(model, mjOBJ_ACTUATOR, i);
    csv << ",tau_" << (nm ? nm : std::to_string(i));
  }
  for (int i = 0; i < model->nu; ++i) {
    const char* nm = mj_id2name(model, mjOBJ_ACTUATOR, i);
    csv << ",ctrl_" << (nm ? nm : std::to_string(i));
  }
  csv << ",com_x,com_y,com_z,table_fx,table_fy,table_fz,table_ncon\n";

  int total_steps = std::ceil(total_time / model->opt.timestep);
  mjpc::ThreadPool pool(threads);
  int whole_sec = 0;

  for (int i = 0; i < total_steps; ++i) {
    agent.ActiveTask()->Transition(model, data);
    agent.state.Set(model, data);
    agent.ActivePlanner().ActionFromPolicy(data->ctrl,
                                           agent.state.state().data(),
                                           agent.state.time(),
                                           /*use_previous=*/false);
    mj_step(model, data);
    double cost = agent.ActiveTask()->CostValue(data->sensordata);
    if (i % steps_per_plan == 0) agent.PlanIteration(&pool);

    // --- log ---------------------------------------------------------------
    std::map<std::string, double> metrics;
    std::string phase;
    agent.ActiveTask()->ComputeMetrics(model, data, &metrics, &phase);

    // Whole-body CoM.
    double mass = 0, com[3] = {0, 0, 0};
    for (int b = 1; b < model->nbody; ++b) {
      mass += model->body_mass[b];
      for (int k = 0; k < 3; ++k)
        com[k] += model->body_mass[b] * data->xipos[3 * b + k];
    }
    if (mass > 0) for (int k = 0; k < 3; ++k) com[k] /= mass;

    // Net robot->table contact force (world frame).
    double tf[3] = {0, 0, 0};
    int tncon = 0;
    for (int c = 0; c < data->ncon; ++c) {
      const mjContact& con = data->contact[c];
      bool hits_table = (table_geom >= 0) &&
                        (con.geom[0] == table_geom || con.geom[1] == table_geom);
      if (!hits_table) continue;
      mjtNum f6[6] = {0};
      mj_contactForce(model, data, c, f6);
      // contact frame -> world; frame is row-major 3x3 in con.frame
      for (int r = 0; r < 3; ++r)
        for (int k = 0; k < 3; ++k) tf[k] += con.frame[3 * r + k] * f6[r];
      tncon++;
    }

    csv << data->time << "," << phase << "," << cost;
    for (int a = 0; a < model->nu; ++a) csv << "," << data->actuator_force[a];
    for (int a = 0; a < model->nu; ++a) csv << "," << data->ctrl[a];
    csv << "," << com[0] << "," << com[1] << "," << com[2] << "," << tf[0]
        << "," << tf[1] << "," << tf[2] << "," << tncon << "\n";

    if (std::floor(data->time) > whole_sec) {
      whole_sec++;
      std::cout << "t=" << whole_sec << "s phase=" << phase
                << " cost=" << cost << " table_fz=" << tf[2] << "\n";
    }
  }

  csv.close();
  std::cout << "wrote " << out_path << "\n";
  mj_deleteData(data);
  mjcb_sensor = nullptr;
  return 0;
}
