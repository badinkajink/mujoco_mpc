// Deploy model preparation (stage 3a of the 2026-07-18 reorg; formerly
// internal to deploy_common.cc).

#ifndef MJPC_DEPLOY_DEPLOY_MODEL_H_
#define MJPC_DEPLOY_DEPLOY_MODEL_H_

#include <mujoco/mujoco.h>

#include "mjpc/deploy/deploy_common.h"

namespace h12deploy {

// Patch a freshly-loaded model's <position> actuators to the node's
// authoritative gains (+ estop-bound forceranges where the config asks --
// full-body node: ARMS ONLY, idx >= frc_limit_begin, so the planner can't plan
// a motor-peak (120 Nm) arm torque. LEG/TORSO forceranges are LEFT at the
// model default: tightening them to the estop bound clamped the planner's
// hip/ankle balance authority and regressed the hold). <position>:
// gainprm[0]=kp, biasprm[1]=-kp, biasprm[2]=-kv. Call on the loaded model
// BEFORE Agent::Initialize (it is const after GetModel()), and on BOTH the
// planner model and the latency-comp rollout model (they must agree, else
// predict-forward simulates torque the planner/node cannot produce).
//
// ACTUATOR-AUTHORITY PARITY (frc_parity): see the NodeConfig::frc_parity
// comment. When ON, EVERY actuator's forcerange is tightened to the deployment
// torque budget -- kBudgetRatio * tau_estop, just under the safety-layer estop
// (there is no emit clamp: a plan over this budget is published as-is and can
// trip the estop) -- so the sampler stops planning single-support balance on
// phantom ankle/torso authority. It only ever TIGHTENS (never loosens) an
// existing limit, so the arm patch above still binds where it is stricter.
void PatchActuators(mjModel* m, const NodeConfig& cfg);

}  // namespace h12deploy

#endif  // MJPC_DEPLOY_DEPLOY_MODEL_H_
