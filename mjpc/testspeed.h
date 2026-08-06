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

#ifndef MJPC_MJPC_TESTSPEED_H_
#define MJPC_MJPC_TESTSPEED_H_

#include <string>

namespace mjpc {
// Runs synchronous planning for `total_time` seconds and returns the total cost.
//
// `dump_residual_steps` >= 0 switches the run into RESIDUAL DUMP mode: instead of
// timing a full run, take exactly that many planning steps from the model's default
// keyframe, then print every mjSENS_USER residual SCALAR as `idx<TAB>name[off]<TAB>value`
// at %.17g and return 0. The residual is a pure function of (model, data) -- no
// sampling -- so the dump is deterministic and can be diffed bit-for-bit between two
// engines. dump_residual_steps == 0 dumps at the untouched keyframe, before any
// planning has run, which removes the stochastic planner from the comparison entirely.
// A negative value (the default) disables dump mode.
//
// `dump_perturb` > 0 first kicks qpos/qvel off the keyframe with a fixed-seed
// deterministic pseudo-random offset of that magnitude. At the pristine keyframe
// most of the residual is identically zero, and a term that is zero cannot
// disagree, so the unperturbed dump has little discriminating power; the kick
// lights up the velocity-, asymmetry- and tilt-driven terms while keeping the
// state an exact function of the seed.
// `dump_traj`, if non-empty, writes the whole rollout to that path as CSV:
// one row per physics step with time, qpos, qvel, ctrl and actuator_force. That
// is everything needed to score MJPC's own result with the SAME static-analysis
// machinery the offline QP study uses (contact set, torque ratios, equilibrium
// margin), instead of comparing an average cost against a pose.
//
// `start_qpos`, if non-empty, is a path to a whitespace/comma separated list of
// nq values used as the starting configuration, applied AFTER start_key. It
// exists to start a run at a pose produced OUTSIDE the model -- e.g. the offline
// QP's braced solution -- which separates "the planner cannot find this pose"
// from "this pose does not survive contact dynamics". Those are different
// problems with different fixes and the same symptom.
//
// `start_key`, if non-empty, selects the starting keyframe by name instead of
// "home". The handoff question is whether sampling can FIND a braced pose, so
// where it starts from is the experiment, not a detail: starting on top of the
// answer measures nothing.
//
// `phase_schedule`, if non-empty, drives the task's PHASE parameter (index 2)
// over sim time as "t0:p0,t1:p1,...". A multi-phase strategy whose keyframes all
// carry success_sustain_time 9999 can never auto-advance, so a headless run sits
// in keyframe 0 for its whole duration and reports that phase's cost as if it
// were the strategy's. Scheduling the phase externally is what makes the
// stand -> brace -> hold -> return sequence testable without a GUI, and it keeps
// the sequencing policy in the experiment harness rather than in the task.
double SynchronousPlanningCost(std::string task_name, int planner_thread_count,
                               int steps_per_planning_iteration,
                               double total_time,
                               int dump_residual_steps = -1,
                               double dump_perturb = 0.0,
                               std::string dump_traj = "",
                               std::string start_key = "",
                               int strategy = -1,
                               std::string start_qpos = "",
                               std::string phase_schedule = "");
}  // namespace mjpc

#endif  // MJPC_MJPC_TESTSPEED_H_
