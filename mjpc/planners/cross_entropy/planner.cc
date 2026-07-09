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

#include "mjpc/planners/cross_entropy/planner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <shared_mutex>
#include <vector>

#include <absl/random/random.h>
#include <absl/types/span.h>
#include <mujoco/mujoco.h>
#include "mjpc/array_safety.h"
#include "mjpc/planners/planner.h"
#include "mjpc/planners/sampling/planner.h"
#include "mjpc/spline/spline.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

namespace mjpc {

namespace mju = ::mujoco::util_mjpc;
using mjpc::spline::TimeSpline;

// initialize data and settings
void CrossEntropyPlanner::Initialize(mjModel* model, const Task& task) {
  // delete mjData instances since model might have changed.
  data_.clear();

  // allocate one mjData for nominal.
  ResizeMjData(model, 1);

  // model
  this->model = model;

  // task
  this->task = &task;

  // sampling noise
  std_initial_ =
      GetNumberOrDefault(0.1, model,
                         "sampling_exploration");         // initial variance
  std_min_ = GetNumberOrDefault(0.01, model, "std_min");  // minimum variance
  // fraction of the trajectories that will use full exploration noise
  explore_fraction_ =
      GetNumberOrDefault(0.0, model, "explore_fraction");

  // set number of trajectories to rollout
  num_trajectory_ = GetNumberOrDefault(10, model, "sampling_trajectories");

  // set number of elite samples max(best 10%, 2)
  n_elite_ =
      GetNumberOrDefault(std::max(num_trajectory_ / 10, 2), model, "n_elite");

  if (num_trajectory_ > kMaxTrajectory) {
    mju_error_i("Too many trajectories, %d is the maximum allowed.",
                kMaxTrajectory);
  }
}

// allocate memory
void CrossEntropyPlanner::Allocate() {
  // initial state
  int num_state = model->nq + model->nv + model->na;

  // state
  state.resize(num_state);
  mocap.resize(7 * model->nmocap);
  userdata.resize(model->nuserdata);

  // policy
  int num_max_parameter = model->nu * kMaxTrajectoryHorizon;
  policy.Allocate(model, *task, kMaxTrajectoryHorizon);
  resampled_policy.Allocate(model, *task, kMaxTrajectoryHorizon);
  previous_policy.Allocate(model, *task, kMaxTrajectoryHorizon);
  best_policy_.Allocate(model, *task, kMaxTrajectoryHorizon);

  // scratch
  parameters_scratch.resize(num_max_parameter);
  times_scratch.resize(kMaxTrajectoryHorizon);

  // noise
  noise.resize(kMaxTrajectory * (model->nu * kMaxTrajectoryHorizon));

  // variance
  variance.resize(model->nu * kMaxTrajectoryHorizon);  // (nu * horizon)

  // need to initialize an arbitrary order of the trajectories
  trajectory_order.resize(kMaxTrajectory);
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory_order[i] = i;
  }

  // trajectories and parameters
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i].Initialize(num_state, model->nu, task->num_residual,
                             task->num_trace, kMaxTrajectoryHorizon);
    trajectory[i].Allocate(kMaxTrajectoryHorizon);
    candidate_policy[i].Allocate(model, *task, kMaxTrajectoryHorizon);
  }
  nominal_trajectory.Initialize(num_state, model->nu, task->num_residual,
                                task->num_trace, kMaxTrajectoryHorizon);
  nominal_trajectory.Allocate(kMaxTrajectoryHorizon);
}

// reset memory to zeros
void CrossEntropyPlanner::Reset(int horizon,
                                const double* initial_repeated_action) {
  // state
  std::fill(state.begin(), state.end(), 0.0);
  std::fill(mocap.begin(), mocap.end(), 0.0);
  std::fill(userdata.begin(), userdata.end(), 0.0);
  time = 0.0;

  // policy parameters
  policy.Reset(horizon, initial_repeated_action);
  resampled_policy.Reset(horizon, initial_repeated_action);
  previous_policy.Reset(horizon, initial_repeated_action);
  best_policy_.Reset(horizon, initial_repeated_action);

  // scratch
  std::fill(parameters_scratch.begin(), parameters_scratch.end(), 0.0);
  std::fill(times_scratch.begin(), times_scratch.end(), 0.0);

  // noise
  std::fill(noise.begin(), noise.end(), 0.0);

  // variance
  double var = std_initial_ * std_initial_;
  std::fill(variance.begin(), variance.end(), var);

  // trajectory samples
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i].Reset(kMaxTrajectoryHorizon);
    candidate_policy[i].Reset(horizon);
  }
  nominal_trajectory.Reset(kMaxTrajectoryHorizon);

  for (const auto& d : data_) {
    mju_zero(d->ctrl, model->nu);
  }

  // improvement
  improvement = 0.0;
}

// set state
void CrossEntropyPlanner::SetState(const State& state) {
  state.CopyTo(this->state.data(), this->mocap.data(), this->userdata.data(),
               &this->time);
}

// optimize nominal policy using random sampling
void CrossEntropyPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
  resampled_policy.plan.SetInterpolation(interpolation_);

  // if num_trajectory_ has changed, use it in this new iteration.
  // num_trajectory_ might change while this function runs. Keep it constant
  // for the duration of this function.
  int num_trajectory = num_trajectory_;

  // n_elite_ might change in the GUI - keep constant for in this function
  n_elite_ = std::min(n_elite_, num_trajectory);
  int n_elite = std::min(n_elite_, num_trajectory);

  // resize number of mjData
  ResizeMjData(model, pool.NumThreads());

  // copy nominal policy
  {
    const std::shared_lock<std::shared_mutex> lock(mtx_);
    resampled_policy.CopyFrom(policy, policy.num_spline_points);
  }

  // resample nominal policy to current time
  this->ResamplePolicy(horizon);

  // ----- rollout noisy policies ----- //
  // start timer
  auto rollouts_start = std::chrono::steady_clock::now();

  // simulate noisy policies
  this->Rollouts(num_trajectory, horizon, pool);

  // sort candidate policies and trajectories by score
  for (int i = 0; i < num_trajectory; i++) {
    trajectory_order[i] = i;
  }

  // sort so that the first ncandidates elements are the best candidates, and
  // the rest are in an unspecified order
  std::partial_sort(
      trajectory_order.begin(), trajectory_order.begin() + num_trajectory,
      trajectory_order.begin() + num_trajectory,
      [&trajectory = trajectory](int a, int b) {
        return trajectory[a].total_return < trajectory[b].total_return;
      });

  // stop timer
  rollouts_compute_time = GetDuration(rollouts_start);

  // ----- update policy ----- //
  // start timer
  auto policy_update_start = std::chrono::steady_clock::now();

  // dimensions
  int num_spline_points = resampled_policy.num_spline_points;
  int num_parameters = num_spline_points * model->nu;

  // averaged return over elites
  double avg_return = 0.0;

  // reset parameters scratch
  std::fill(parameters_scratch.begin(), parameters_scratch.end(), 0.0);

  // loop over elites to compute average
  for (int i = 0; i < n_elite; i++) {
    // ordered trajectory index
    int idx = trajectory_order[i];
    const TimeSpline& elite_plan = candidate_policy[idx].plan;

    // add parameters
    for (int t = 0; t < num_spline_points; t++) {
      TimeSpline::ConstNode n = elite_plan.NodeAt(t);
      for (int j = 0; j < model->nu; j++) {
        parameters_scratch[t * model->nu + j] += n.values()[j];
      }
    }

    // add total return
    avg_return += trajectory[idx].total_return;
  }

  // normalize
  mju_scl(parameters_scratch.data(), parameters_scratch.data(), 1.0 / n_elite,
          num_parameters);
  avg_return /= n_elite;

  // loop over elites to compute variance
  std::fill(variance.begin(), variance.end(), 0.0);  // reset variance to zero
  for (int i = 0; i < n_elite; i++) {
    int idx = trajectory_order[i];
    const TimeSpline& elite_plan = candidate_policy[idx].plan;
    for (int t = 0; t < num_spline_points; t++) {
      TimeSpline::ConstNode n = elite_plan.NodeAt(t);
      for (int j = 0; j < model->nu; j++) {
        // average
        double p_avg = parameters_scratch[t * model->nu + j];

        // candidate parameter
        double pi = n.values()[j];
        double diff = pi - p_avg;
        variance[t * model->nu + j] += diff * diff / (n_elite - 1);
      }
    }
  }

  // update
  {
    const std::unique_lock<std::shared_mutex> lock(mtx_);
    policy.plan.Clear();
    policy.plan.SetInterpolation(interpolation_);
    for (int t = 0; t < num_spline_points; t++) {
      absl::Span<const double> values =
          absl::MakeConstSpan(parameters_scratch.data() + t * model->nu,
                              parameters_scratch.data() + (t + 1) * model->nu);
      policy.plan.AddNode(times_scratch[t], values);
    }

    // execute-best: stash the single lowest-cost elite's spline so
    // ActionFromPolicy can emit it (arXiv:2511.19204) instead of the elite
    // MEAN computed above. The CEM distribution update (policy = mean) is left
    // intact; only what is HANDED TO THE PLANT changes when best_action_ is on.
    best_policy_.CopyFrom(candidate_policy[trajectory_order[0]],
                          num_spline_points);
    best_policy_.plan.SetInterpolation(interpolation_);
  }

  // improvement: compare nominal to elite average
  improvement =
      mju_max(avg_return - trajectory[trajectory_order[0]].total_return, 0.0);

  // stop timer
  policy_update_compute_time = GetDuration(policy_update_start);

  // FABEL (2026-07-07, env H12_PDUMP=1): dump the CEM internals ACTUALLY in
  // effect at iteration time -- the deploy node and the agent server converge
  // to different optima on a bit-identical frozen state after every
  // config/input channel was proven equal from the OUTSIDE; this prints the
  // inside. Both processes link this same object (libmjpc.a), so the lines
  // are directly diffable. One 'cfg' line on the first gated call (static
  // planner config + model fingerprint + weights/params), then an 'it' line
  // every H12_PDUMP_EVERY-th call (default 25) with the state consumed, the
  // winner, the LIVE elite-variance state (the CEM's only persistent
  // history), and the elite-mean action at the plan time. Unset = silent,
  // byte-identical.
  {
    static int pd_on = -1;
    static long pd_n = 0;
    static long pd_every = 25;
    if (pd_on < 0) {
      const char* e = std::getenv("H12_PDUMP");
      pd_on = (e && e[0] == '1') ? 1 : 0;
      if (const char* ev = std::getenv("H12_PDUMP_EVERY")) {
        long v = std::atol(ev);
        if (v > 0) pd_every = v;
      }
      if (pd_on) {
        int ks = mj_name2id(model, mjOBJ_KEY, "stand");
        double ksz = ks >= 0 ? model->key_qpos[ks * model->nq + 2] : -1.0;
        double ksh = ks >= 0 ? model->key_qpos[ks * model->nq + 8] : 0.0;
        double ksk = ks >= 0 ? model->key_qpos[ks * model->nq + 10] : 0.0;
        std::fprintf(stderr,
            "PDUMP cem-cfg ntraj=%d nelite=%d nsp=%d std0=%.6g stdmin=%.6g "
            "explore=%.6g interp=%d bestact=%d ts=%.6g integ=%d nq=%d nu=%d "
            "nkey=%d nsensor=%d mass=%.8g key_stand id=%d z=%.6g hipP=%.6g "
            "knee=%.6g knotgrow=%.6g\n",
            num_trajectory_, n_elite, num_spline_points, std_initial_,
            std_min_, explore_fraction_, (int)interpolation_,
            (int)best_action_, model->opt.timestep, model->opt.integrator,
            model->nq, model->nu, model->nkey, model->nsensor,
            mj_getTotalmass(model),
            ks, ksz, ksh, ksk,
            GetNumberOrDefault(0.0, model, "sampling_knot_var_growth"));
        // dynamics fingerprint: checksums of every model array that shapes
        // the rollout physics -- rollout costs explode node-side on an
        // identical state + step-0 residual, so SOME dynamics array differs.
        std::fprintf(stderr,
            "PDUMP cem-fp gain=%.8g bias=%.8g frcrange=%.8g ctrlrange=%.8g "
            "frclim=%d damp=%.8g arma=%.8g fric=%.8g ipos=%.8g inertia=%.8g "
            "jntrange=%.8g qpos0=%.8g keyq=%.8g neq=%d eqdata=%.8g eqact=%d "
            "grav=%.6g imprat=%.6g cone=%d solver=%d iter=%d lsiter=%d "
            "enable=%d disable=%d dens=%.6g visc=%.6g\n",
            mju_sum(model->actuator_gainprm, model->nu * mjNGAIN),
            mju_sum(model->actuator_biasprm, model->nu * mjNBIAS),
            mju_sum(model->actuator_forcerange, model->nu * 2),
            mju_sum(model->actuator_ctrlrange, model->nu * 2),
            [&]{ int s=0; for (int i=0;i<model->nu;i++) s+=model->actuator_forcelimited[i]; return s; }(),
            mju_sum(model->dof_damping, model->nv),
            mju_sum(model->dof_armature, model->nv),
            mju_sum(model->dof_frictionloss, model->nv),
            mju_sum(model->body_ipos, model->nbody * 3),
            mju_sum(model->body_inertia, model->nbody * 3),
            mju_sum(model->jnt_range, model->njnt * 2),
            mju_sum(model->qpos0, model->nq),
            mju_sum(model->key_qpos, model->nkey * model->nq),
            model->neq,
            mju_sum(model->eq_data, model->neq * mjNEQDATA),
            [&]{ int s=0; for (int i=0;i<model->neq;i++) s+=model->eq_active0[i]; return s; }(),
            model->opt.gravity[2], model->opt.impratio, model->opt.cone,
            model->opt.solver, model->opt.iterations, model->opt.ls_iterations,
            model->opt.enableflags, model->opt.disableflags,
            model->opt.density, model->opt.viscosity);
        std::fprintf(stderr, "PDUMP weights(40):");
        for (size_t k = 0; k < task->weight.size() && k < 40; k++)
          std::fprintf(stderr, " %.6g", task->weight[k]);
        std::fprintf(stderr, "\nPDUMP params:");
        for (size_t k = 0; k < task->parameters.size(); k++)
          std::fprintf(stderr, " %.6g", task->parameters[k]);
        std::fprintf(stderr, "\n");
      }
    }
    if (pd_on && (pd_n++ % pd_every == 0)) {
      // live elite-variance state (persistent across iterations)
      double vsum = 0.0, vmax = 0.0;
      for (int k = 0; k < num_parameters; k++) {
        vsum += variance[k];
        if (variance[k] > vmax) vmax = variance[k];
      }
      double std_avg = num_parameters > 0 ? std::sqrt(vsum / num_parameters)
                                          : 0.0;
      double act[64] = {0};
      {
        const std::shared_lock<std::shared_mutex> lock(mtx_);
        policy.Action(act, state.data(), time);
      }
      std::fprintf(stderr,
          "PDUMP cem-it n=%ld t=%.5f z=%.5f hipP=%.5f knee=%.5f ankP=%.5f "
          "vz=%.5f horizon=%d best=%.6g nom=%.6g avg_el=%.6g impr=%.4g "
          "stdavg=%.6g stdmax=%.6g mode=%d act_hipP=%.5f act_knee=%.5f "
          "act_ankP=%.5f\n",
          pd_n - 1, time, state[2], state[8], state[10], state[11],
          state[model->nq + 2], horizon,
          trajectory[trajectory_order[0]].total_return,
          nominal_trajectory.total_return, avg_return, improvement,
          std_avg, std::sqrt(vmax), task->mode, act[1], act[3], act[4]);
      // per-term WEIGHTED cost of the best trajectory's FIRST step (= the
      // residual of the state the planner was handed) + step-0 cost along
      // the horizon at steps 0/25/50/100 -- names the exact term that scores
      // differently between the node and the agent server on an identical
      // state. Full consumed state vector too (object/task slots included).
      const Trajectory& bt = trajectory[trajectory_order[0]];
      if (bt.dim_residual > 0 && (int)bt.residual.size() >= bt.dim_residual) {
        std::vector<double> terms(task->num_term, 0.0);
        task->CostTerms(terms.data(), bt.residual.data());
        std::fprintf(stderr, "PDUMP terms0:");
        for (int k = 0; k < task->num_term; k++)
          std::fprintf(stderr, " %.5g", terms[k]);
        std::fprintf(stderr, "\nPDUMP costs h0/h25/h50/h100: %.5g %.5g %.5g "
                     "%.5g\n",
                     bt.costs[0],
                     bt.horizon > 25 ? bt.costs[25] : -1.0,
                     bt.horizon > 50 ? bt.costs[50] : -1.0,
                     bt.horizon > 100 ? bt.costs[100] : -1.0);
      }
      std::fprintf(stderr, "PDUMP state:");
      for (size_t k = 0; k < state.size(); k++)
        std::fprintf(stderr, " %.5g", state[k]);
      std::fprintf(stderr, "\n");
      // contact-softness fingerprint (MakeDifferentiable mutates solimp at
      // plan time; the static cfg fingerprint missed solref/solimp/friction)
      std::fprintf(stderr,
          "PDUMP soft jsolimp=%.8g gsolref=%.8g gsolimp=%.8g gfric=%.8g "
          "psolref=%.8g npair=%d\n",
          mju_sum(model->jnt_solimp, model->njnt * mjNIMP),
          mju_sum(model->geom_solref, model->ngeom * mjNREF),
          mju_sum(model->geom_solimp, model->ngeom * mjNIMP),
          mju_sum(model->geom_friction, model->ngeom * 3),
          model->npair > 0
              ? mju_sum(model->pair_solref, model->npair * mjNREF) : 0.0,
          model->npair);
      // live policy spline knots (the CEM's warm-start content)
      {
        const std::shared_lock<std::shared_mutex> lock(mtx_);
        std::fprintf(stderr, "PDUMP knots");
        for (int k = 0; k < (int)policy.plan.Size(); k++) {
          auto nd = policy.plan.NodeAt(k);
          std::fprintf(stderr, " |%d t=%.3f hipP=%.4f knee=%.4f ankP=%.4f",
                       k, nd.time(), nd.values()[1], nd.values()[3],
                       nd.values()[4]);
        }
        std::fprintf(stderr, "\n");
      }
      // CANONICAL DETERMINISTIC ROLLOUT: same code, same inputs, run inside
      // BOTH processes -- constant ctrl = the 'stand' key leg pose, from the
      // CURRENT planner state, 100 steps on the planner model with a private
      // mjData. If the two processes disagree here, the execution
      // environment (model internals / callbacks) differs; if they agree,
      // the divergence is in the search history/policy content.
      {
        static mjData* cd = nullptr;
        if (!cd) cd = mj_makeData(model);
        mj_resetData(model, cd);
        mju_copy(cd->qpos, state.data(), model->nq);
        mju_copy(cd->qvel, state.data() + model->nq, model->nv);
        cd->time = time;
        int ks = mj_name2id(model, mjOBJ_KEY, "stand");
        double cctrl[64] = {0};
        for (int j = 0; j < model->nu && j < 64; j++)
          cctrl[j] = ks >= 0 ? model->key_qpos[ks * model->nq + 7 + j] : 0.0;
        // PRE-STEP probe: forces at the pristine state, before any stepping
        {
          mju_copy(cd->ctrl, cctrl, model->nu);
          mj_forward(model, cd);
          std::fprintf(stderr,
              "PDUMP prestep qact=%.6g f1=%.6g f2=%.6g f7=%.6g q8=%.6f "
              "q9=%.6f ctrl1=%.4f qfrc_bias7=%.4g qfrc_pass=%.6g "
              "eqlen=%.6g\n",
              mju_norm(cd->qfrc_actuator, model->nv),
              cd->actuator_force[1], cd->actuator_force[2],
              cd->actuator_force[7], cd->qpos[8], cd->qpos[9], cd->ctrl[1],
              cd->qfrc_bias[7], mju_norm(cd->qfrc_passive, model->nv),
              mju_norm(cd->actuator_length, model->nu));
          std::fprintf(stderr, "PDUMP trn");
          for (int j = 0; j < model->nu && j < 12; j++) {
            int jid = model->actuator_trnid[j * 2];
            std::fprintf(stderr, " |%d ty=%d gear=%.4g qadr=%d len=%.5g",
                         j, model->actuator_trntype[j],
                         model->actuator_gear[j * 6],
                         jid >= 0 && jid < model->njnt
                             ? model->jnt_qposadr[jid] : -1,
                         cd->actuator_length[j]);
          }
          std::fprintf(stderr, "\n");
        }
        double ctot = 0.0, c0 = 0.0, c50 = 0.0, c99 = 0.0;
        double ztrace[4] = {0}, vtrace[4] = {0};
        for (int s = 0; s < 100; s++) {
          mju_copy(cd->ctrl, cctrl, model->nu);
          mj_step(model, cd);
          double cs = task->CostValue(cd->sensordata);
          ctot += cs;
          if (s == 0) {
            c0 = cs;
            // step-1 forensics: where does the kick come from?
            int amax = 0;
            for (int v = 1; v < model->nv; v++)
              if (std::fabs(cd->qvel[v]) > std::fabs(cd->qvel[amax])) amax = v;
            std::fprintf(stderr,
                "PDUMP step1 ncon=%d qfrc_con=%.6g amax_dof=%d amax_v=%.6g "
                "vbase=%.4g %.4g %.4g %.4g %.4g %.4g varm=%.6g vobj=%.6g\n",
                cd->ncon, mju_norm(cd->qfrc_constraint, model->nv),
                amax, cd->qvel[amax],
                cd->qvel[0], cd->qvel[1], cd->qvel[2], cd->qvel[3],
                cd->qvel[4], cd->qvel[5],
                mju_norm(cd->qvel + 18, 15),
                model->nv >= 39 ? mju_norm(cd->qvel + 33, 6) : 0.0);
            std::fprintf(stderr, "PDUMP eqs");
            for (int e = 0; e < model->neq; e++)
              std::fprintf(stderr, " %d:%d:%.4g", e, cd->eq_active[e],
                           model->eq_data[e * mjNEQDATA]);
            std::fprintf(stderr, "\nPDUMP cons");
            for (int c = 0; c < cd->ncon && c < 10; c++) {
              const char* g1 = mj_id2name(model, mjOBJ_GEOM,
                                          cd->contact[c].geom[0]);
              const char* g2 = mj_id2name(model, mjOBJ_GEOM,
                                          cd->contact[c].geom[1]);
              std::fprintf(stderr, " |%s~%s d=%.5f", g1 ? g1 : "?",
                           g2 ? g2 : "?", cd->contact[c].dist);
            }
            std::fprintf(stderr, "\nPDUMP qfrc_con_legs");
            for (int v = 6; v < 18 && v < model->nv; v++)
              std::fprintf(stderr, " %.3g", cd->qfrc_constraint[v]);
            std::fprintf(stderr, "\nPDUMP qfrc_con_base");
            for (int v = 0; v < 6; v++)
              std::fprintf(stderr, " %.3g", cd->qfrc_constraint[v]);
            std::fprintf(stderr, " arms:");
            for (int v = 18; v < 33 && v < model->nv; v++)
              std::fprintf(stderr, " %.3g", cd->qfrc_constraint[v]);
            std::fprintf(stderr, "\nPDUMP act nefc=%d nplugin=%d "
                         "qfrc_act=%.6g qfrc_smooth_legs:",
                         cd->nefc, model->nplugin,
                         mju_norm(cd->qfrc_actuator, model->nv));
            for (int v = 6; v < 18 && v < model->nv; v++)
              std::fprintf(stderr, " %.3g", cd->qfrc_smooth[v]);
            std::fprintf(stderr, " ctrl:");
            for (int j = 0; j < model->nu && j < 12; j++)
              std::fprintf(stderr, " %.4g", cd->ctrl[j]);
            std::fprintf(stderr, "\nPDUMP actrow");
            for (int j = 0; j < model->nu && j < 12; j++)
              std::fprintf(stderr, " |%d g=%.4g b1=%.4g b2=%.4g trn=%d f=%.4g",
                           j, model->actuator_gainprm[j * mjNGAIN],
                           model->actuator_biasprm[j * mjNBIAS + 1],
                           model->actuator_biasprm[j * mjNBIAS + 2],
                           model->actuator_trnid[j * 2],
                           cd->actuator_force[j]);
            std::fprintf(stderr, "\nPDUMP passive stiff=%.8g spring=%.8g "
                         "qfrc_pass_legs:",
                         mju_sum(model->jnt_stiffness, model->njnt),
                         mju_sum(model->qpos_spring, model->nq));
            for (int v = 6; v < 18 && v < model->nv; v++)
              std::fprintf(stderr, " %.3g", cd->qfrc_passive[v]);
            std::fprintf(stderr, "\n");
          }
          if (s == 50) c50 = cs;
          if (s == 99) c99 = cs;
          int ti = s == 0 ? 0 : s == 4 ? 1 : s == 9 ? 2 : s == 24 ? 3 : -1;
          if (ti >= 0) {
            ztrace[ti] = cd->qpos[2];
            vtrace[ti] = mju_norm(cd->qvel, model->nv);
          }
        }
        std::fprintf(stderr,
            "PDUMP canon tot=%.6g c0=%.5g c50=%.5g c99=%.5g z=%.5f "
            "hipP=%.5f knee=%.5f tilt_qx=%.5f\n",
            ctot, c0, c50, c99, cd->qpos[2], cd->qpos[8], cd->qpos[10],
            cd->qpos[4]);
        std::fprintf(stderr,
            "PDUMP canontrace z1=%.8f z5=%.8f z10=%.8f z25=%.8f "
            "v1=%.8f v5=%.8f v10=%.8f v25=%.8f\n",
            ztrace[0], ztrace[1], ztrace[2], ztrace[3],
            vtrace[0], vtrace[1], vtrace[2], vtrace[3]);
        // definitive model comparison: byte-sum of the fully serialized model
        {
          int msz = mj_sizeModel(model);
          static std::vector<uint8_t> mbuf;
          mbuf.resize(msz);
          mj_saveModel(model, nullptr, mbuf.data(), msz);
          unsigned long long bsum = 0;
          for (int b = 0; b < msz; b++) bsum += mbuf[b] * (1ull + b % 251);
          std::fprintf(stderr, "PDUMP mhash bytes=%d sum=%llu jac=%d tol=%.3g "
                       "lstol=%.3g noslip=%d ccd=%d actgrpdis=%d\n",
                       msz, bsum, model->opt.jacobian, model->opt.tolerance,
                       model->opt.ls_tolerance, model->opt.noslip_iterations,
                       model->opt.ccd_iterations,
                       model->opt.disableactuator);
          std::fprintf(stderr,
              "PDUMP geom ngeom=%d nmesh=%d nnames=%d npaths=%d nmeshvert=%d "
              "gsize=%.8g gpos=%.8g bpos=%.8g bquat=%.8g mvert=%.8g "
              "gcontype=%d gmargin=%.6g gap=%.6g\n",
              model->ngeom, model->nmesh, model->nnames, model->npaths,
              model->nmeshvert,
              mju_sum(model->geom_size, model->ngeom * 3),
              mju_sum(model->geom_pos, model->ngeom * 3),
              mju_sum(model->body_pos, model->nbody * 3),
              mju_sum(model->body_quat, model->nbody * 4),
              [&]{ double s=0; int n = model->nmeshvert > 1000000 ? 3000000 : model->nmeshvert * 3; for (int v=0;v<n;v++) s+=model->mesh_vert[v]; return s; }(),
              [&]{ int s=0; for (int g=0;g<model->ngeom;g++) s+=model->geom_contype[g]+model->geom_conaffinity[g]; return s; }(),
              mju_sum(model->geom_margin, model->ngeom),
              mju_sum(model->geom_gap, model->ngeom));
        }
      }
    }
  }
}

// compute trajectory using nominal policy
void CrossEntropyPlanner::NominalTrajectory(int horizon) {
  // set policy
  auto nominal_policy = [&cp = resampled_policy](
                            double* action, const double* state, double time) {
    cp.Action(action, state, time);
  };

  // rollout nominal policy
  nominal_trajectory.Rollout(nominal_policy, task, model,
                             data_[ThreadPool::WorkerId()].get(), state.data(),
                             time, mocap.data(), userdata.data(), horizon);
}
void CrossEntropyPlanner::NominalTrajectory(int horizon, ThreadPool& pool) {
  NominalTrajectory(horizon);
}

// set action from policy
void CrossEntropyPlanner::ActionFromPolicy(double* action, const double* state,
                                           double time, bool use_previous) {
  const std::shared_lock<std::shared_mutex> lock(mtx_);
  if (use_previous) {
    previous_policy.Action(action, state, time);
  } else if (best_action_) {
    best_policy_.Action(action, state, time);
  } else {
    policy.Action(action, state, time);
  }
  // capture-point footstep override (default no-op): mirror the rollout swing
  // onto the EXECUTED action so what runs == what the planner rolled out. state
  // layout = [qpos(nq), qvel(nv), act(na)] -> qvel = state + nq.
  if (task) task->ModifyControl(model, state, state + model->nq, time, action);
}

// update policy via resampling
void CrossEntropyPlanner::ResamplePolicy(int horizon) {
  // dimensions
  int num_spline_points = resampled_policy.num_spline_points;

  // time
  double nominal_time = time;
  double time_shift = mju_max(
      (horizon - 1) * model->opt.timestep / (num_spline_points - 1), 1.0e-5);

  // get spline points
  for (int t = 0; t < num_spline_points; t++) {
    times_scratch[t] = nominal_time;
    resampled_policy.Action(DataAt(parameters_scratch, t * model->nu), nullptr,
                            nominal_time);
    nominal_time += time_shift;
  }

  // copy resampled policy parameters
  resampled_policy.plan.Clear();
  for (int t = 0; t < num_spline_points; t++) {
    absl::Span<const double> values =
        absl::MakeConstSpan(parameters_scratch.data() + t * model->nu,
                            parameters_scratch.data() + (t + 1) * model->nu);
    resampled_policy.plan.AddNode(times_scratch[t], values);
  }
  resampled_policy.plan.SetInterpolation(policy.plan.Interpolation());
}

// add random noise to nominal policy
void CrossEntropyPlanner::AddNoiseToPolicy(int i, double std_min) {
  // start timer
  auto noise_start = std::chrono::steady_clock::now();

  // dimensions
  int num_spline_points = candidate_policy[i].num_spline_points;

  // sampling token
  absl::BitGen gen_;

  // shift index
  int shift = i * (model->nu * kMaxTrajectoryHorizon);

  // sample noise
  // variance[k] is the standard deviation for the k^th control parameter over
  // the elite samples we draw a bunch of control actions from this distribution
  // (which i indexes) - the noise is stored in `noise`.
  // ACTION-LEVEL annealing (foot-lift Tier C, 2026-06-24): far-horizon spline
  // knots sample at HIGHER variance than near-term knots, so an aggressive future
  // swing/flight phase can be proposed without destabilising the immediate
  // committed action (DIAL-MPC's horizon_diffuse_factor). knot_growth=0 -> uniform
  // (legacy default) -> byte-identical for every other strategy/task.
  double knot_growth = GetNumberOrDefault(0.0, model, "sampling_knot_var_growth");
  for (int t = 0; t < num_spline_points; t++) {
    double knot_frac = (num_spline_points > 1)
                           ? static_cast<double>(t) / (num_spline_points - 1)
                           : 0.0;
    double knot_scale = 1.0 + knot_growth * knot_frac;
    for (int j = 0; j < model->nu; j++) {
      int k = t * model->nu + j;
      noise[k + shift] = absl::Gaussian<double>(
          gen_, 0.0, std::max(std::sqrt(variance[k]), std_min) * knot_scale);
    }
  }

  for (int k = 0; k < candidate_policy[i].plan.Size(); k++) {
    TimeSpline::Node n = candidate_policy[i].plan.NodeAt(k);
    // add noise
    mju_addTo(n.values().data(), DataAt(noise, shift + k * model->nu),
              model->nu);
    // clamp parameters
    Clamp(n.values().data(), model->actuator_ctrlrange, model->nu);
  }

  // end timer
  IncrementAtomic(noise_compute_time, GetDuration(noise_start));
}

// compute candidate trajectories
void CrossEntropyPlanner::Rollouts(int num_trajectory, int horizon,
                                   ThreadPool& pool) {
  // reset noise compute time
  noise_compute_time = 0.0;

  // lock std_min
  double std_min = std_min_;
  double std_initial = std_initial_;

  // random search
  int count_before = pool.GetCount();
  for (int i = 0; i < num_trajectory; i++) {
    double std;
    if (i < num_trajectory * explore_fraction_) {
      std = std_initial;
    } else {
      std = std_min;
    }
    pool.Schedule([&s = *this, &model = this->model, &task = this->task,
                   &state = this->state, &time = this->time,
                   &mocap = this->mocap, &userdata = this->userdata, horizon,
                   std, i]() {
      // copy nominal policy and sample noise
      {
        const std::shared_lock<std::shared_mutex> lock(s.mtx_);
        s.candidate_policy[i].CopyFrom(s.resampled_policy,
                                       s.resampled_policy.num_spline_points);
        s.candidate_policy[i].plan.SetInterpolation(
            s.resampled_policy.plan.Interpolation());

        // sample noise
        s.AddNoiseToPolicy(i, std);
      }

      // ----- rollout sample policy ----- //

      // policy
      auto sample_policy_i = [&candidate_policy = s.candidate_policy, &i](
                                 double* action, const double* state,
                                 double time) {
        candidate_policy[i].Action(action, state, time);
      };

      // policy rollout
      s.trajectory[i].Rollout(
          sample_policy_i, task, model, s.data_[ThreadPool::WorkerId()].get(),
          state.data(), time, mocap.data(), userdata.data(), horizon);
    });
  }
  // nominal
  pool.Schedule([&s = *this, horizon]() { s.NominalTrajectory(horizon); });

  // wait
  pool.WaitCount(count_before + num_trajectory + 1);
  pool.ResetCount();
}

// returns the **nominal** trajectory (this is the purple trace)
const Trajectory* CrossEntropyPlanner::BestTrajectory() {
  return &nominal_trajectory;
}

// visualize planner-specific traces
void CrossEntropyPlanner::Traces(mjvScene* scn) {
  // sample color
  float color[4];
  color[0] = 1.0;
  color[1] = 1.0;
  color[2] = 1.0;
  color[3] = 1.0;

  // width of a sample trace, in pixels
  double width = GetNumberOrDefault(3, model, "agent_sample_width");

  // scratch
  double zero3[3] = {0};
  double zero9[9] = {0};

  // best
  auto best = this->BestTrajectory();

  // sample traces
  int n_elite = n_elite_;
  for (int k = 0; k < n_elite; k++) {
    // plot sample
    for (int i = 0; i < best->horizon - 1; i++) {
      if (scn->ngeom + task->num_trace > scn->maxgeom) break;
      for (int j = 0; j < task->num_trace; j++) {
        // initialize geometry
        mjv_initGeom(&scn->geoms[scn->ngeom], mjGEOM_LINE, zero3, zero3, zero9,
                     color);

        // elite index
        int idx = trajectory_order[k];
        // make geometry
        mjv_connector(
            &scn->geoms[scn->ngeom], mjGEOM_LINE, width,
            trajectory[idx].trace.data() + 3*task->num_trace * i + 3 * j,
            trajectory[idx].trace.data() + 3*task->num_trace * (i + 1) + 3 * j);

        // increment number of geometries
        scn->ngeom += 1;
      }
    }
  }
}

// planner-specific GUI elements
void CrossEntropyPlanner::GUI(mjUI& ui) {
  mjuiDef defCrossEntropy[] = {
      {mjITEM_SLIDERINT, "Rollouts", 2, &num_trajectory_, "0 1"},
      {mjITEM_SELECT, "Spline", 2, &interpolation_,
       "Zero\nLinear\nCubic"},
      {mjITEM_SLIDERINT, "Spline Pts", 2, &policy.num_spline_points, "0 1"},
      {mjITEM_SLIDERNUM, "Init. Std", 2, &std_initial_, "0 1"},
      {mjITEM_SLIDERNUM, "Min. Std", 2, &std_min_, "0.01 0.5"},
      {mjITEM_SLIDERNUM, "Explore", 2, &explore_fraction_, "0.0 1.0"},
      {mjITEM_SLIDERINT, "Elite", 2, &n_elite_, "2 128"},
      {mjITEM_END}};

  // set number of trajectory slider limits
  mju::sprintf_arr(defCrossEntropy[0].other, "%i %i", 1, kMaxTrajectory);

  // set spline point limits
  mju::sprintf_arr(defCrossEntropy[2].other, "%i %i", MinSamplingSplinePoints,
                   MaxSamplingSplinePoints);

  // set noise standard deviation limits
  mju::sprintf_arr(defCrossEntropy[3].other, "%f %f", MinNoiseStdDev,
                   MaxNoiseStdDev);

  // add cross entropy planner
  mjui_add(&ui, defCrossEntropy);
}

// planner-specific plots
void CrossEntropyPlanner::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
                                int planner_shift, int timer_shift,
                                int planning, int* shift) {
  // ----- planner ----- //
  double planner_bounds[2] = {-6.0, 6.0};

  // improvement
  mjpc::PlotUpdateData(fig_planner, planner_bounds,
                       fig_planner->linedata[0 + planner_shift][0] + 1,
                       mju_log10(mju_max(improvement, 1.0e-6)), 100,
                       0 + planner_shift, 0, 1, -100);

  // legend
  mju::strcpy_arr(fig_planner->linename[0 + planner_shift], "Avg - Best");

  fig_planner->range[1][0] = planner_bounds[0];
  fig_planner->range[1][1] = planner_bounds[1];

  // bounds
  double timer_bounds[2] = {0.0, 1.0};

  // ----- timer ----- //

  PlotUpdateData(
      fig_timer, timer_bounds, fig_timer->linedata[0 + timer_shift][0] + 1,
      1.0e-3 * noise_compute_time * planning, 100, 0 + timer_shift, 0, 1, -100);

  PlotUpdateData(fig_timer, timer_bounds,
                 fig_timer->linedata[1 + timer_shift][0] + 1,
                 1.0e-3 * rollouts_compute_time * planning, 100,
                 1 + timer_shift, 0, 1, -100);

  PlotUpdateData(fig_timer, timer_bounds,
                 fig_timer->linedata[2 + timer_shift][0] + 1,
                 1.0e-3 * policy_update_compute_time * planning, 100,
                 2 + timer_shift, 0, 1, -100);

  // legend
  mju::strcpy_arr(fig_timer->linename[0 + timer_shift], "Noise");
  mju::strcpy_arr(fig_timer->linename[1 + timer_shift], "Rollout");
  mju::strcpy_arr(fig_timer->linename[2 + timer_shift], "Policy Update");

  // planner shift
  shift[0] += 1;

  // timer shift
  shift[1] += 3;
}

}  // namespace mjpc
