#include "mjpc/deploy/deploy_model.h"

#include <cstdio>
#include <limits>

#include <mujoco/mujoco.h>

namespace h12deploy {

void PatchActuators(mjModel* m, const NodeConfig& cfg) {
  // Task-side default lives in the model as the `deploy_frc_parity` numeric (set by
  // Task::PlannerNumericOverrides, applied by the caller BEFORE this runs). CLI wins.
  bool parity = false;
  if (cfg.frc_parity >= 0) {
    parity = cfg.frc_parity > 0;
  } else {
    int pid = mj_name2id(m, mjOBJ_NUMERIC, "deploy_frc_parity");
    parity = (pid >= 0) && (m->numeric_data[m->numeric_adr[pid]] > 0.5);
  }

  for (int i = 0; i < cfg.nu && i < m->nu; i++) {
    m->actuator_gainprm[i * mjNGAIN + 0] = cfg.kp[i];
    m->actuator_biasprm[i * mjNBIAS + 1] = -cfg.kp[i];
    m->actuator_biasprm[i * mjNBIAS + 2] = -cfg.kv[i];
    if (cfg.frc_limit && i >= cfg.frc_limit_begin) {
      m->actuator_forcelimited[i] = 1;
      m->actuator_forcerange[i * 2 + 0] = -cfg.frc_limit[i];
      m->actuator_forcerange[i * 2 + 1] = cfg.frc_limit[i];
    }
  }
  // Runs on BOTH the planner model and the latency-comp rollout model; only
  // narrate the first pass so the log stays readable.
  static bool narrated = false;
  const bool say = !narrated;
  narrated = true;
  if (!parity) {
    if (say)
      std::fprintf(stderr,
                   "[node] actuator-authority parity OFF: the planner may plan torque over the "
                   "safety-estop budget -- with no emit clamp it is published as-is and can TRIP "
                   "the safety estop (legacy model; --frc_parity=1 to enable)\n");
    return;
  }
  if (say)
    std::fprintf(stderr,
                 "[node] actuator-authority parity ON (planner + latency models): forcerange "
                 "-> the deployment budget (%.2f x tau_estop, under the safety estop). Tightened:\n",
                 kBudgetRatio);
  for (int i = 0; i < cfg.nu && i < m->nu; i++) {
    const double budget = kBudgetRatio * cfg.tau_estop[i];
    const double had = m->actuator_forcelimited[i]
                           ? m->actuator_forcerange[i * 2 + 1]
                           : std::numeric_limits<double>::infinity();
    if (!(budget < had)) continue;             // never LOOSEN an existing limit
    m->actuator_forcelimited[i] = 1;
    m->actuator_forcerange[i * 2 + 0] = -budget;
    m->actuator_forcerange[i * 2 + 1] = budget;
    if (say)
      std::fprintf(stderr, "[node]   %-7s %7.1f -> %6.1f Nm  (planner was %.2fx over)\n",
                   cfg.joint_names ? cfg.joint_names[i] : "?", had, budget,
                   budget > 0.0 ? had / budget : 0.0);
  }
}

}  // namespace h12deploy
