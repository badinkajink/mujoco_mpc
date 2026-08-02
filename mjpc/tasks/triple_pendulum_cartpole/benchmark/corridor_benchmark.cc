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
//   corridor  obstacles, goal x=6, Upright zeroed, Velocity small and
//             Avoidance raised. Only asks: can the cart be driven to the goal
//             without putting a head inside a disk? Obstacle avoidance under
//             the same underactuated dynamics.
//   balance   no obstacles, goal x=0, started fully hung, Upright and Velocity
//             raised. Only asks: can this planner erect a 3-link underactuated
//             pendulum and hold it at rest?
//   combined  the paper's task, unchanged: thread the corridor, then capture.
//
// Each stage reweights toward what it scores, so a planner is never judged
// against an objective it was not given. That also means costs are comparable
// across planners within a stage but not across stages.
//
// A planner that passes both isolated stages and fails combined is failing at
// the composition, which is a different diagnosis from failing a component.
//
// Usage:
//   corridor_benchmark --planner=0 --stage=combined --total_time=20 --repeats=3
//   corridor_benchmark --planner=5 --stage=corridor --repeats=100 --speed=0.25 \
//                      --weights=1,0,0.1,0.01,500

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <mujoco/mujoco.h>

#include "mjpc/agent.h"
#include "mjpc/planners/annealed_sampling/planner.h"
#include "mjpc/planners/cross_entropy/planner.h"
#include "mjpc/planners/ilqg/planner.h"
#include "mjpc/planners/planner.h"
#include "mjpc/planners/pso/planner.h"
#include "mjpc/planners/sampling/planner.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/tasks/tasks.h"
#include "mjpc/tasks/triple_pendulum_cartpole/triple_pendulum_cartpole.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

ABSL_FLAG(int, planner, -1,
          "Planner index to force (overrides agent_planner in the XML). "
          "0=PredictiveSampling 1=Gradient 2=iLQG 3=iLQS 4=RobustSampling "
          "5=CrossEntropy 6=SampleGradient 7=PSO 8=AnnealedSampling "
          "9=RandomSampling. -1 keeps the XML value.");
ABSL_FLAG(std::string, task, "corridor",
          "Which world: 'corridor' (the paper's single bottleneck at x=3, "
          "goal x=6) or 'slalom' (three bottlenecks at x=3/6/9, goal x=11). "
          "The slalom is the same system and the same residual with more "
          "disks, and its first gap is the corridor's unmoved, so the drop "
          "between them is attributable to the extra bottlenecks.");
ABSL_FLAG(std::string, stage, "combined",
          "Which capability to isolate: 'corridor' (obstacle avoidance only), "
          "'balance' (swing-up and capture only), or 'combined' (the full "
          "task). See the file comment.");
ABSL_FLAG(double, total_time, 20.0, "Simulated seconds per repeat.");
ABSL_FLAG(int, repeats, 1, "Number of repeats.");
ABSL_FLAG(int, planner_thread, mjpc::NumAvailableHardwareThreads() - 5,
          "Number of planner threads.");
ABSL_FLAG(double, speed, 1.0,
          "Simulated seconds per real second, the GUI's speed slider. Control "
          "is applied every timestep of SIMULATED time at any speed (200 Hz "
          "at the model's 0.005 s), but in WALL time the loop runs at "
          "200*speed Hz -- speed=0.25 is 50 Hz control -- and the planner gets "
          "1/speed iterations per control step, so four instead of one. "
          "Slowing down buys iterations per decision and pays in control "
          "rate; the planner's budget per iteration is the timestep either "
          "way. Every planner gets the same iteration count regardless of what "
          "an iteration costs it, which is what makes the comparison about the "
          "algorithm rather than about throughput; the timing block is where "
          "iteration cost shows up.");
ABSL_FLAG(double, goal_tolerance, 0.3,
          "Cart position tolerance (m) counted as reaching the goal.");
ABSL_FLAG(double, upright_tolerance, 0.95,
          "Minimum cos(theta_i) at the end counted as upright.");
ABSL_FLAG(double, penetration_tolerance, 0.0,
          "Obstacle penetration depth (m) beyond which a run counts as a "
          "collision. Zero means any overlap at all, which is the physically "
          "correct reading of an avoidance constraint and the default.\n"
          "This used to be 0.02 m, on the theory that MuJoCo's soft contacts "
          "allow a few harmless mm on touch. That theory is wrong, and the "
          "dumps disprove it: across 30 rollouts, `ncon > 0` and "
          "`min_clearance < 0` agree on every single step. Soft contacts "
          "govern how DEEP the overlap gets once the surfaces meet; they "
          "never report a contact between separated geoms. So there is no "
          "grazing band to protect -- and at 0.02 m on a 0.028 m head, a "
          "sphere could sit 71% inside a disk and still score as clean. "
          "Measured penetrations run continuously from 2 mm to 39 mm with no "
          "gap between 'numerical' and 'real', which is the other reason a "
          "depth threshold cannot be principled here.");
ABSL_FLAG(double, contact_fraction_tolerance, 0.0,
          "Fraction of steps in contact beyond which a run counts as a "
          "collision, independent of depth. Zero means any contact step. "
          "Subsumed by a zero penetration_tolerance (the two fire on exactly "
          "the same steps); kept so the old, laxer criterion can be "
          "reproduced by setting both.");
ABSL_FLAG(double, speed_tolerance, 1.0,
          "Max ||qvel|| counted as 'at rest' in the goal set. Without this, a "
          "pendulum swinging THROUGH vertical at 5 rad/s scores as solved: "
          "measured median speed in the goal configuration is 4.7 rad/s, so "
          "the configuration test alone is not a solve.");
ABSL_FLAG(int, pso_publish_evaluated, -1,
          "PSO only. 1: publish the parameters that were rolled out. 0: "
          "publish the post-swarm-update iterate, whose cost was never "
          "evaluated (the original behaviour). -1 keeps the XML value.");
ABSL_FLAG(std::string, dump, "",
          "If set, write a per-step trajectory CSV to this path (run index is "
          "appended for repeats > 1). Render it with "
          "mjpc/tasks/triple_pendulum_cartpole/benchmark/filmstrip.py");
ABSL_FLAG(int, dump_runs, 1,
          "How many repeats to dump when --dump is set. Dumping all 100 runs "
          "of a sweep is rarely wanted; the first few are enough to look at.");
ABSL_FLAG(std::string, weights, "",
          "Override every cost weight: "
          "'cart,upright,velocity,control,avoidance'. Takes precedence over "
          "the stage weights, so the run is scored and planned against "
          "exactly these numbers. Empty keeps the stage's own weights.");
ABSL_FLAG(double, clearance, -1.0,
          "Override the Avoidance hinge margin (m): the residual is zero "
          "until a head is this close to a disk surface. Negative keeps the "
          "stage value (task.xml's 0.08, or 0.25 for --stage=corridor). "
          "Weight sets the size of the penalty; margin sets how early it "
          "arrives, and only the second buys the planner time to steer.");
ABSL_FLAG(double, exploration, -1.0,
          "Override sampling_exploration, the sampling planners' noise "
          "standard deviation as a fraction of half the control range. "
          "Negative keeps task.xml's value. Exists to make the samplers "
          "comparable at equal noise: PSO hard-codes its own scale at "
          "0.1*ctrlrange (planners/pso/planner.cc, InitializeParticles), "
          "which is half what sampling_exploration=0.4 gives the others, and "
          "that difference is otherwise invisible in a planner comparison.");
ABSL_FLAG(int, spline_points, -1,
          "Override sampling_spline_points, the number of knots the policy "
          "spline carries. Negative keeps task.xml's value. This is the "
          "companion to --horizon and the two are not independent: the knots "
          "are spread over the horizon, so raising the horizon at a fixed "
          "knot count coarsens the control near t=0, which is the part that "
          "actually gets executed. Hold knot spacing constant by scaling this "
          "with the horizon if the question is 'does lookahead help' rather "
          "than 'does coarser control hurt'.");
ABSL_FLAG(double, horizon, -1.0,
          "Override the planner horizon (s). Negative keeps task.xml's "
          "agent_horizon. Longer horizons are what a multi-bottleneck "
          "corridor needs to see the second gap while still in the first.");
ABSL_FLAG(double, beta_action, -1.0,
          "Override annealing_beta_action, DIAL-MPC's action-level "
          "temperature (Annealed Sampling only). Negative keeps the model's "
          "value. This is the third knob tied to --horizon: the schedule is "
          "exp(-(H-1-h)/(beta*H)) over knot INDEX, so it rescales itself to "
          "whatever H is and the noise on the executed knots is a function of "
          "their fraction of the horizon, not of how far ahead they are in "
          "seconds. Tripling the horizon therefore cuts exploration of the "
          "first second from a 0.16-1.00 multiplier to 0.14-0.26 -- the "
          "planner explores the part of the plan it is about to execute less, "
          "the further ahead it looks. Raise this with the horizon (~0.95 at "
          "2 s, ~1.08 at 3 s) to hold near-horizon exploration fixed.");
ABSL_FLAG(bool, early_exit, true,
          "End a run as soon as its outcome is decided -- the goal state was "
          "reached, or an obstacle was hit hard enough to disqualify it -- "
          "instead of simulating out the full --total_time. On the corridor "
          "stage this is exactly the success test, so it costs no "
          "information and makes a 100-trial sweep several times cheaper. It "
          "does suppress 'held to end', which needs the whole run: with "
          "--early_exit the held column is only meaningful for runs that "
          "never solved.");
ABSL_FLAG(double, init_noise, 0.02,
          "Gaussian std of the per-trial perturbation applied to the "
          "starting qpos and qvel (rad, m, and their rates). Without it "
          "repeats differ only through the planner's own sampling, so a "
          "deterministic planner like iLQG reports the same run 100 times "
          "and its success rate is 0 or 1 by construction. Perturbing the "
          "start is what makes 'succeeded 87/100' mean something for every "
          "planner. Set 0 for the paper's exact initial condition.");
ABSL_FLAG(int, seed, 1,
          "Seed for the initial-state perturbation. Trial k of every planner "
          "sees the same start, so planners are compared on identical "
          "problems. This does not seed the planner's own sampling; see "
          "--planner_seed.");
ABSL_FLAG(int, planner_seed, 0,
          "Seed for the sampling planner's noise. 0 draws from system entropy, "
          "which means two runs of the same command give different numbers -- "
          "on the slalom, one configuration measured anywhere from 6/50 to "
          "16/50 that way. Any non-zero value pins the noise, so a row of a "
          "results table can be reproduced exactly. Trial k uses "
          "planner_seed + k, so trials within a run still differ. Currently "
          "honoured by the sampling planner (--planner=0) and its derivatives; "
          "the other planners still draw from entropy.");
ABSL_FLAG(std::string, label, "",
          "Name to report in the RESULT line instead of the planner's. Two "
          "runs of the same planner under different settings are otherwise "
          "indistinguishable in a sweep table.");
ABSL_FLAG(bool, per_run, true,
          "Print one line per repeat. Off leaves only the summary, which is "
          "what a 100-trial sweep usually wants.");

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

double GetTermWeight(const mjModel* model, const char* sensor_name) {
  int id = mj_name2id(model, mjOBJ_SENSOR, sensor_name);
  if (id < 0) mju_error_s("sensor '%s' not found", sensor_name);
  return model->sensor_user[id * model->nuser_sensor + 1];
}

// Stage cost weights. An isolation stage exists to make one capability the
// dominant term in the objective, so each stage reweights toward the thing it
// is scoring rather than inheriting the combined task's balance.
//
// corridor: Avoidance is what is being measured, so it outweighs Cart by 30x
// instead of 5x -- reaching the goal a little later is cheap, touching a disk
// is not. The margin is widened with it: the residual is a hinge loss that is
// exactly zero until a head is within `margin` of a disk, so at 0.08 m the
// penalty arrives about one control interval before contact, which is too late
// to steer around. At 0.25 m the sampler feels the disk while it can still act
// on it. Widening the margin, not just the weight, is what turns the term from
// a fine into a barrier.
//
// corridor keeps a small Velocity weight rather than zeroing it, which is a
// change from treating this stage as "Upright and Velocity off". Zeroing both
// leaves the pendulum ballistic -- measured 14.7 rad/s at the end of a run --
// and at that speed a single cart actuator has no authority over where the
// heads go inside a 1 s horizon. The avoidance term then has nothing to act
// through, and no weight rescues it: collisions were insensitive to raising
// Avoidance while Velocity was zero. At 0.5 the pendulum settles to 1-4 rad/s
// and collisions roughly halve, so the stage measures obstacle avoidance
// instead of measuring luck. It is still far below the balance stage's 3.0,
// and Upright stays at zero, so the stage does not start scoring capture.
//
// balance: the two terms that define capture. Upright is the reorientation
// (get every link to theta = 0) and Velocity is the hold (arrive at rest, not
// swinging through). Velocity moves the most, 0.1 -> 3.0, because that is the
// term the combined task under-weights: the measured median speed in the goal
// configuration is 4.7 rad/s, i.e. planners were passing through the goal
// rather than stopping in it.
constexpr double kCorridorAvoidanceWeight = 300.0;
constexpr double kCorridorClearanceMargin = 0.25;
constexpr double kCorridorVelocityWeight = 0.5;
constexpr double kBalanceUprightWeight = 80.0;
constexpr double kBalanceVelocityWeight = 3.0;

// Geom ids of the disk obstacles, found the same way Corridor::Initialize
// finds them: by the "obstacle" name prefix. Kept in step with that on
// purpose -- if the two disagreed, a stage could disable a disk the residual
// still charges for, or the reverse.
std::vector<int> ObstacleGeoms(const mjModel* model) {
  std::vector<int> ids;
  for (int id = 0; id < model->ngeom; id++) {
    const char* name = mj_id2name(model, mjOBJ_GEOM, id);
    if (name && std::strncmp(name, "obstacle", 8) == 0) ids.push_back(id);
  }
  return ids;
}

// The distinct x positions of the bottlenecks, ascending. Disks that share an
// x are the two halves of one gap, so this is the list of gaps the cart has to
// get through -- what makes "passed 2 of 3" reportable instead of just
// "failed".
std::vector<double> BottleneckPositions(const mjModel* model) {
  std::vector<double> xs;
  for (int id : ObstacleGeoms(model)) {
    double x = model->geom_pos[3 * id];
    bool seen = false;
    for (double v : xs) seen = seen || std::abs(v - x) < 1e-6;
    if (!seen) xs.push_back(x);
  }
  std::sort(xs.begin(), xs.end());
  return xs;
}

// The five cost terms, in the order --weights takes them.
constexpr int kNumTerms = 5;
const char* kTermName[kNumTerms] = {"Cart", "Upright", "Velocity", "Control",
                                    "Avoidance"};

// Parse 'cart,upright,velocity,control,avoidance'. Exits on anything that is
// not five numbers: a silently mis-parsed weight would produce a plausible
// looking table scored against an objective nobody chose.
std::vector<double> ParseWeights(const std::string& s) {
  std::vector<double> w;
  size_t start = 0;
  while (start <= s.size()) {
    size_t comma = s.find(',', start);
    std::string field =
        s.substr(start, comma == std::string::npos ? std::string::npos
                                                   : comma - start);
    try {
      w.push_back(std::stod(field));
    } catch (const std::exception&) {
      std::cerr << "--weights: '" << field << "' is not a number\n";
      std::exit(1);
    }
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  if (w.size() != kNumTerms) {
    std::cerr << "--weights needs " << kNumTerms
              << " comma-separated values (cart,upright,velocity,control,"
                 "avoidance), got " << w.size() << "\n";
    std::exit(1);
  }
  return w;
}

// Which keyframe a stage starts from, and how the world is altered for it --
// geometry and goal only. The objective is set separately by ConfigureStage-
// Weights, so that --weights can replace the whole objective without also
// having to reproduce the stage's world.
const char* ConfigureStageWorld(mjModel* model, Stage stage) {
  switch (stage) {
    case Stage::kCombined:
    case Stage::kCorridor:
      // The corridor stage keeps the world exactly as the paper has it. What
      // makes it an isolation stage is the objective, not the geometry: the
      // pendulum is free to flail, so the dynamics stay as underactuated and
      // chaotic as in the full task -- an easier objective, not an easier
      // system.
      return "home";

    case Stage::kBalance:
      // Remove the corridor entirely: push the disks far off in y (so the
      // Avoidance residual, which measures distance in the x-z plane, still
      // sees them -- hence also clear their contact bits) and detach them from
      // collision. Goal moves to x=0 so the cart is not asked to travel.
      for (int id : ObstacleGeoms(model)) {
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

// The stage's objective. Skipped entirely when --weights is given: a caller
// who states all five weights has replaced the objective, and silently keeping
// the stage's clearance margin on top of that would mean the run is not
// planning against the numbers it printed.
void ConfigureStageWeights(mjModel* model, Stage stage) {
  switch (stage) {
    case Stage::kCombined:
      return;

    case Stage::kCorridor:
      // Zero the term that asks for capture, keeping Cart, Control and
      // Avoidance. What remains scores exactly one thing: reaching x=6 without
      // driving a head into a disk.
      SetTermWeight(model, "Upright", 0.0);
      SetTermWeight(model, "Velocity", kCorridorVelocityWeight);
      SetTermWeight(model, "Avoidance", kCorridorAvoidanceWeight);
      if (double* c = mjpc::GetCustomNumericData(model, "residual_Clearance")) {
        *c = kCorridorClearanceMargin;
      }
      return;

    case Stage::kBalance:
      SetTermWeight(model, "Upright", kBalanceUprightWeight);
      SetTermWeight(model, "Velocity", kBalanceVelocityWeight);
      return;
  }
}

double GetSeconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now() - start)
             .count() / 1e9;
}

mjpc::Task* g_task;
void ResidualCallback(const mjModel* model, mjData* data, int stage) {
  if (stage == mjSTAGE_ACC) {
    g_task->Residual(model, data, data->sensordata);
  }
}

// Where a planning iteration's wall time went, in seconds, summed over a run.
//
// Held separately from the harness's own stopwatch because the two answer
// different questions: the harness measures what one iteration costs (which is
// what decides whether the planner can keep up), and this breakdown says which
// part of the iteration to attack if it cannot.
//
// The planners publish these as public members rather than through the Planner
// interface, so reading them means naming the concrete types. That is contained
// here: a planner not listed simply reports no breakdown, and nothing in the
// planners changes to be measured.
struct PlannerTimers {
  bool available = false;
  double noise = 0;        // sampling family: perturbing the nominal
  double rollouts = 0;     // every planner: the simulated trajectories
  double update = 0;       // publishing / averaging the winner
  double derivatives = 0;  // iLQG: model + cost derivatives
  double backward = 0;     // iLQG: the Riccati sweep
  double nominal = 0;      // iLQG: the nominal rollout

  double Total() const {
    return noise + rollouts + update + derivatives + backward + nominal;
  }
};

// Add one iteration's internal timers. The sampling-family planners set these
// once per OptimizePolicy call, so this must be called after each iteration and
// before the next one overwrites them.
//
// `noise_compute_time` is a sum over worker threads, not a wall-clock span, so
// it can exceed the iteration's wall time on a 15-thread pool. Reported as
// thread-seconds and labelled as such.
void AccumulatePlannerTimers(const mjpc::Planner& planner, PlannerTimers* t) {
  if (auto* p = dynamic_cast<const mjpc::iLQGPlanner*>(&planner)) {
    t->available = true;
    t->nominal += p->nominal_compute_time;
    t->derivatives +=
        p->model_derivative_compute_time + p->cost_derivative_compute_time;
    t->backward += p->backward_pass_compute_time;
    t->rollouts += p->rollouts_compute_time;
    t->update += p->policy_update_compute_time;
  } else if (auto* p = dynamic_cast<const mjpc::PSOPlanner*>(&planner)) {
    t->available = true;
    t->rollouts += p->rollouts_compute_time;
    // the swarm velocity/position update, PSO's analogue of "noise"
    t->noise += p->velocity_update_time;
    t->update += p->policy_update_compute_time;
  } else if (auto* p =
                 dynamic_cast<const mjpc::AnnealedSamplingPlanner*>(&planner)) {
    t->available = true;
    t->noise += p->noise_compute_time;
    t->rollouts += p->rollouts_compute_time;
    t->update += p->policy_update_compute_time;
  } else if (auto* p =
                 dynamic_cast<const mjpc::CrossEntropyPlanner*>(&planner)) {
    t->available = true;
    t->noise += p->noise_compute_time;
    t->rollouts += p->rollouts_compute_time;
    t->update += p->policy_update_compute_time;
  } else if (auto* p = dynamic_cast<const mjpc::SamplingPlanner*>(&planner)) {
    // RandomSamplingPlanner derives from SamplingPlanner, so this arm covers
    // both. It has to come last for that reason.
    t->available = true;
    t->noise += p->noise_compute_time;
    t->rollouts += p->rollouts_compute_time;
    t->update += p->policy_update_compute_time;
  }
}

// Everything a run needs that is not the run index. Collected into one struct
// because the flag list outgrew a readable argument list, and because every
// field here has to reach both the planner's model and the scoring code --
// passing them individually is how the two quietly drift apart.
struct RunConfig {
  std::string task_name;  // as registered with the Agent
  std::string task_key;   // "corridor" / "slalom", recorded in dumps
  int planner_override = -1;
  Stage stage = Stage::kCombined;
  std::string stage_name = "combined";
  int pso_publish_evaluated = -1;
  std::vector<double> weights;   // empty: keep the stage weights
  double clearance = -1.0;       // negative: keep the stage margin
  double horizon = -1.0;         // negative: keep task.xml's agent_horizon
  double exploration = -1.0;     // negative: keep task.xml's value
  int spline_points = -1;        // negative: keep task.xml's value
  double beta_action = -1.0;     // negative: keep the model's value
  double total_time = 20.0;
  double speed = 1.0;
  int planner_thread_count = 1;
  double goal_tolerance = 0.3;
  double upright_tolerance = 0.95;
  double penetration_tolerance = 0.02;
  double contact_fraction_tolerance = 0.02;
  double speed_tolerance = 1.0;
  bool early_exit = true;
  double init_noise = 0.02;
  int seed = 1;
  int planner_seed = 0;          // 0: planner noise drawn from entropy
};

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
  bool ended_early = false;     // --early_exit stopped it before total_time
  double sim_time = 0;          // simulated seconds actually run
  double timestep = 0;          // model->opt.timestep, the per-iteration budget

  // How far through the obstacle field the cart got, in bottlenecks. On a
  // single corridor this is 0 or 1 and adds nothing to passed_corridor; on the
  // slalom it is the difference between "failed" and "cleared two of three",
  // which is the whole question that task asks.
  int gaps_passed = 0;
  int num_gaps = 1;

  // ---- timing ----
  // The harness holds planning *iterations* constant across planners, so the
  // per-iteration wall time is the planner's price and the only figure that
  // says whether it could run outside this benchmark.
  int plan_iterations = 0;
  double plan_wall = 0;      // wall seconds inside PlanIteration
  double step_wall = 0;      // wall seconds inside mj_step + cost evaluation
  double iter_ms_mean = 0;
  double iter_ms_p50 = 0;
  double iter_ms_p95 = 0;
  double iter_ms_max = 0;
  PlannerTimers timers;
};

RunResult RunOnce(const RunConfig& cfg, int run_index,
                  const std::string& dump_path) {
  RunResult r;

  mjpc::Agent agent;
  agent.SetTaskList(mjpc::GetTasks());
  agent.gui_task_id = agent.GetTaskIdByName(cfg.task_name);
  if (agent.gui_task_id == -1) {
    std::cerr << "Invalid task '" << cfg.task_name << "'. Valid values:\n"
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
  if (cfg.planner_override >= 0) {
    double* p = mjpc::GetCustomNumericData(model, "agent_planner");
    if (p) *p = static_cast<double>(cfg.planner_override);
  }

  if (cfg.pso_publish_evaluated >= 0) {
    double* p = mjpc::GetCustomNumericData(model, "pso_publish_evaluated");
    if (p) *p = static_cast<double>(cfg.pso_publish_evaluated);
  }

  if (cfg.horizon > 0) {
    double* h = mjpc::GetCustomNumericData(model, "agent_horizon");
    if (h) *h = cfg.horizon;
  }

  if (cfg.exploration >= 0) {
    double* e = mjpc::GetCustomNumericData(model, "sampling_exploration");
    if (e) *e = cfg.exploration;
  }

  if (cfg.spline_points > 0) {
    double* n = mjpc::GetCustomNumericData(model, "sampling_spline_points");
    if (n) *n = static_cast<double>(cfg.spline_points);
  }

  if (cfg.beta_action > 0) {
    double* b = mjpc::GetCustomNumericData(model, "annealing_beta_action");
    if (b) *b = cfg.beta_action;
  }

  // Pin the planner's noise so the run is reproducible. Offset by run_index so
  // trial k still differs from trial k+1 -- otherwise every trial of a run
  // would draw the identical noise sequence and the repeats would measure only
  // the initial-state perturbation.
  if (cfg.planner_seed != 0) {
    double* s = mjpc::GetCustomNumericData(model, "sampling_seed");
    if (s) *s = static_cast<double>(cfg.planner_seed + run_index);
  }

  // Alter the model for the stage before Agent::Initialize reads it, so the
  // planner optimizes the same objective the run is scored against.
  const char* key_name = ConfigureStageWorld(model, cfg.stage);

  // Explicit weights replace the stage's objective outright, margin included;
  // the stage's world (obstacles removed for balance, goal moved) still holds.
  if (cfg.weights.empty()) {
    ConfigureStageWeights(model, cfg.stage);
  } else {
    for (int i = 0; i < kNumTerms; i++) {
      SetTermWeight(model, kTermName[i], cfg.weights[i]);
    }
  }
  if (cfg.clearance >= 0) {
    if (double* c = mjpc::GetCustomNumericData(model, "residual_Clearance")) {
      *c = cfg.clearance;
    }
  }

  // Does the goal set include the pendulum's state? Only if the objective
  // actually asks for it. Derived from the final weights rather than from the
  // stage so that --weights cannot put the scoring and the objective at odds.
  const bool score_pendulum = GetTermWeight(model, "Upright") > 0;

  mjData* data = mj_makeData(model);
  int key_id = mj_name2id(model, mjOBJ_KEY, key_name);
  if (key_id >= 0) mj_resetDataKeyframe(model, data, key_id);

  // Per-trial perturbation of the start. Seeded from run_index so trial k is
  // the same problem for every planner: the comparison is then between
  // planners on one set of starts, not between planners on different ones.
  if (cfg.init_noise > 0) {
    std::mt19937 rng(cfg.seed * 7919 + run_index);
    std::normal_distribution<double> normal(0.0, cfg.init_noise);
    for (int i = 0; i < model->nq; i++) data->qpos[i] += normal(rng);
    for (int i = 0; i < model->nv; i++) data->qvel[i] += normal(rng);
  }
  mj_forward(model, data);

  agent.estimator_enabled = false;
  agent.Initialize(model);
  agent.Allocate();
  agent.Reset(data->ctrl);
  agent.plan_enabled = true;

  g_task = agent.ActiveTask();
  mjcb_sensor = &ResidualCallback;

  // Same corridor geometry the Avoidance residual uses, so "collided" here and
  // "charged for clearance" there cannot disagree about what counts as close.
  mjpc::TriplePendulumCartpole::Corridor corridor;
  corridor.Initialize(model);
  double goal = mjpc::GetNumberOrDefault(6.0, model, "residual_Goal");

  mjpc::ThreadPool pool(cfg.planner_thread_count);
  int total_steps = std::ceil(cfg.total_time / model->opt.timestep);
  // The per-iteration deadline reported at the end is one timestep, so it has
  // to come from the model rather than from a constant: editing <option
  // timestep> is exactly the experiment someone runs here, and a hard-coded
  // 5 ms would keep scoring against a budget the model no longer has.
  r.timestep = model->opt.timestep;
  double total_cost = 0;
  int steps_run = 0;
  // Planning iterations owed to the planner, accumulated at 1/speed per
  // control step so fractional rates (speed 0.25 -> four per step) work
  // without special-casing.
  double planning_credit = 0;
  // Per-iteration wall times. Kept rather than summarized on the fly because
  // the tail matters: a planner whose mean iteration fits the control period
  // but whose p95 does not will miss deadlines exactly when the task is hard.
  std::vector<double> iter_seconds;
  iter_seconds.reserve(static_cast<size_t>(total_steps / cfg.speed) + 1);

  // Optional per-step trajectory dump. Aggregate cost cannot distinguish
  // "swung up and reached the goal" from "drove the cart there with the
  // pendulum hanging"; the dump exists so the rollout can be rendered and
  // looked at. See benchmark/filmstrip.py.
  std::FILE* dump = nullptr;
  if (!dump_path.empty()) {
    dump = std::fopen(dump_path.c_str(), "w");
    if (!dump) {
      std::cerr << "could not open --dump path '" << dump_path << "'\n";
      std::exit(1);
    }
    // Record the stage in the dump. The balance stage removes the obstacles
    // from the model, so a renderer that reloads task.xml without knowing that
    // draws a corridor the run never had. Carrying it in the file means the
    // render cannot be wrong about which world it is showing.
    // Record the world as well as the stage: the renderer has to load the
    // right XML, and "corridor" and "slalom" are different files.
    std::fprintf(dump, "# task=%s\n", cfg.task_key.c_str());
    std::fprintf(dump, "# stage=%s\n", cfg.stage_name.c_str());
    std::fprintf(dump, "step,time,cart,th1,th2,th3,dcart,dth1,dth2,dth3,"
                       "ctrl,cost,ncon,min_clearance\n");
  }

  auto loop_start = std::chrono::steady_clock::now();
  for (int i = 0; i < total_steps; i++) {
    agent.ActiveTask()->Transition(model, data);
    agent.state.Set(model, data);
    agent.ActivePlanner().ActionFromPolicy(data->ctrl,
                                           agent.state.state().data(),
                                           agent.state.time(),
                                           /*use_previous=*/false);
    auto step_start = std::chrono::steady_clock::now();
    mj_step(model, data);
    double cost = agent.ActiveTask()->CostValue(data->sensordata);
    r.step_wall += GetSeconds(step_start);
    total_cost += cost;
    steps_run++;

    for (planning_credit += 1.0 / cfg.speed; planning_credit >= 1.0;
         planning_credit -= 1.0) {
      auto plan_start = std::chrono::steady_clock::now();
      agent.PlanIteration(&pool);
      double elapsed = GetSeconds(plan_start);
      r.plan_wall += elapsed;
      iter_seconds.push_back(elapsed);
      AccumulatePlannerTimers(agent.ActivePlanner(), &r.timers);
    }

    // ---- outcome metrics ----
    r.max_cart = std::max(r.max_cart, data->qpos[0]);
    if (data->ncon > 0) r.contact_steps++;
    double step_min_clearance = corridor.MinClearance(data);
    r.min_clearance = std::min(r.min_clearance, step_min_clearance);

    // ---- was the goal state reached at this step? ----
    double step_min_cos = 1.0;
    for (int j = 0; j < 3; j++) {
      step_min_cos = std::min(step_min_cos, std::cos(data->qpos[1 + j]));
    }
    double speed = mju_norm(data->qvel, model->nv);
    // Whether the pendulum's state is part of the goal set follows the
    // objective, not the stage label: a run with Upright weighted at zero was
    // never asked to capture anything, so requiring it to end upright would
    // score it against a term it was not given. Reading this off the weights
    // rather than off `stage` keeps --weights and the scoring in agreement.
    bool in_goal_config = !score_pendulum ||
                          (step_min_cos > cfg.upright_tolerance &&
                           speed < cfg.speed_tolerance);
    bool in_goal_box = std::abs(data->qpos[0] - goal) < cfg.goal_tolerance;
    if (in_goal_box && in_goal_config) {
      r.solved_steps++;
      if (!r.ever_solved) {
        r.ever_solved = true;
        r.first_solve_time = data->time;
      }
    }

    // Disqualifying contact, evaluated per step so --early_exit can act on it.
    // The contact fraction is taken over the *nominal* run length rather than
    // the steps run so far, so that the criterion is monotone and a run cut
    // short by --early_exit gets the same verdict a full run would have given
    // it up to that point. (With the default zero tolerance the fraction test
    // fires on the first contact step regardless of denominator; the
    // denominator still matters if a depth threshold is dialled back in.)
    r.contact_fraction =
        static_cast<double>(r.contact_steps) / total_steps;
    bool collided_now = r.min_clearance < -cfg.penetration_tolerance ||
                        r.contact_fraction > cfg.contact_fraction_tolerance;

    if (dump) {
      std::fprintf(dump, "%d,%.4f", i, data->time);
      for (int j = 0; j < model->nq; j++) {
        std::fprintf(dump, ",%.6f", data->qpos[j]);
      }
      for (int j = 0; j < model->nv; j++) {
        std::fprintf(dump, ",%.6f", data->qvel[j]);
      }
      std::fprintf(dump, ",%.6f,%.6f,%d,%.6f\n", data->ctrl[0], cost,
                   data->ncon, step_min_clearance);
    }

    // ---- stop once the outcome is decided ----
    // Both terminating conditions are monotone: a solve stays a solve and a
    // disqualifying contact stays disqualifying, so cutting the run here
    // cannot change the verdict, only the seconds spent reaching it. What it
    // does cost is "held to end", which needs the whole run -- see the
    // --early_exit help.
    if (cfg.early_exit && (collided_now || (r.ever_solved && !score_pendulum))) {
      r.ended_early = true;
      break;
    }
  }
  if (dump) std::fclose(dump);
  r.wall_time = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - loop_start)
                    .count() / 1e6;

  r.final_cart = data->qpos[0];
  r.min_cos_upright = 1.0;
  for (int i = 0; i < 3; i++) {
    r.min_cos_upright = std::min(r.min_cos_upright, std::cos(data->qpos[1 + i]));
  }
  r.sim_time = data->time;
  r.mean_cost = total_cost / steps_run;
  r.solved_fraction = static_cast<double>(r.solved_steps) / steps_run;
  r.final_speed = mju_norm(data->qvel, model->nv);
  r.reached_goal = std::abs(r.final_cart - goal) < cfg.goal_tolerance;
  r.upright = !score_pendulum ||
              (r.min_cos_upright > cfg.upright_tolerance &&
               r.final_speed < cfg.speed_tolerance);
  // Fail the obstacle constraint by touching a disk at all. With the default
  // zero tolerances these two tests fire on the same steps -- MuJoCo reports a
  // contact exactly when the surfaces overlap, which is exactly when clearance
  // goes negative -- so this is one criterion expressed twice, kept separate
  // only so a laxer depth threshold can still be dialled in for comparison.
  r.collided = r.min_clearance < -cfg.penetration_tolerance ||
               r.contact_fraction > cfg.contact_fraction_tolerance;
  // A bottleneck counts as passed once the cart is half a metre beyond its
  // disks, which is far enough that the pendulum is through too rather than
  // still lying across the gap.
  std::vector<double> gaps = BottleneckPositions(model);
  r.num_gaps = gaps.empty() ? 1 : static_cast<int>(gaps.size());
  for (double x : gaps) {
    if (r.max_cart > x + 0.5) r.gaps_passed++;
  }
  r.passed_corridor = !gaps.empty() && r.max_cart > gaps.front() + 0.5;

  r.plan_iterations = static_cast<int>(iter_seconds.size());
  if (!iter_seconds.empty()) {
    std::sort(iter_seconds.begin(), iter_seconds.end());
    size_t n = iter_seconds.size();
    r.iter_ms_mean = 1e3 * r.plan_wall / n;
    r.iter_ms_p50 = 1e3 * iter_seconds[n / 2];
    r.iter_ms_p95 = 1e3 * iter_seconds[std::min(n - 1, (95 * n) / 100)];
    r.iter_ms_max = 1e3 * iter_seconds.back();
  }

  mj_deleteData(data);
  mjcb_sensor = nullptr;
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const std::string task_key = absl::GetFlag(FLAGS_task);
  std::string task_name;
  if (task_key == "corridor") {
    task_name = "Triple Pendulum Cartpole";
  } else if (task_key == "slalom") {
    task_name = "Triple Pendulum Cartpole Slalom";
  } else {
    std::cerr << "unknown --task '" << task_key
              << "'; expected corridor or slalom\n";
    return 1;
  }
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
      "PSO", "AnnealedSampling", "RandomSampling"};
  constexpr int kNumPlannerLabels =
      sizeof(kPlannerLabel) / sizeof(kPlannerLabel[0]);

  RunConfig cfg;
  cfg.task_name = task_name;
  cfg.task_key = task_key;
  cfg.planner_override = planner;
  cfg.stage = stage;
  cfg.stage_name = stage_name;
  cfg.pso_publish_evaluated = absl::GetFlag(FLAGS_pso_publish_evaluated);
  cfg.clearance = absl::GetFlag(FLAGS_clearance);
  cfg.horizon = absl::GetFlag(FLAGS_horizon);
  cfg.exploration = absl::GetFlag(FLAGS_exploration);
  cfg.spline_points = absl::GetFlag(FLAGS_spline_points);
  cfg.beta_action = absl::GetFlag(FLAGS_beta_action);
  cfg.total_time = absl::GetFlag(FLAGS_total_time);
  cfg.speed = absl::GetFlag(FLAGS_speed);
  cfg.planner_thread_count = absl::GetFlag(FLAGS_planner_thread);
  cfg.goal_tolerance = absl::GetFlag(FLAGS_goal_tolerance);
  cfg.upright_tolerance = absl::GetFlag(FLAGS_upright_tolerance);
  cfg.penetration_tolerance = absl::GetFlag(FLAGS_penetration_tolerance);
  cfg.contact_fraction_tolerance =
      absl::GetFlag(FLAGS_contact_fraction_tolerance);
  cfg.speed_tolerance = absl::GetFlag(FLAGS_speed_tolerance);
  cfg.early_exit = absl::GetFlag(FLAGS_early_exit);
  cfg.init_noise = absl::GetFlag(FLAGS_init_noise);
  cfg.seed = absl::GetFlag(FLAGS_seed);
  cfg.planner_seed = absl::GetFlag(FLAGS_planner_seed);
  const std::string weights_flag = absl::GetFlag(FLAGS_weights);
  if (!weights_flag.empty()) cfg.weights = ParseWeights(weights_flag);
  if (cfg.speed <= 0) {
    std::cerr << "--speed must be positive\n";
    return 1;
  }

  std::printf("task: %s   stage: %s   planner: %s   %.0fs x %d repeat(s)\n",
              task_name.c_str(), stage_name.c_str(),
              (planner >= 0 && planner < kNumPlannerLabels)
                  ? kPlannerLabel[planner]
                  : "(xml)",
              cfg.total_time, repeats);
  if (!cfg.weights.empty()) {
    std::printf("  weights:");
    for (int i = 0; i < kNumTerms; i++) {
      std::printf(" %s %g", kTermName[i], cfg.weights[i]);
    }
    std::printf("\n");
  } else {
    switch (stage) {
      case Stage::kCorridor:
        std::printf("  stage weights: Avoidance %.0f (margin %.2f m), "
                    "Velocity %.1f, Upright 0\n",
                    kCorridorAvoidanceWeight, kCorridorClearanceMargin,
                    kCorridorVelocityWeight);
        break;
      case Stage::kBalance:
        std::printf("  stage weights: Upright %.0f, Velocity %.1f\n",
                    kBalanceUprightWeight, kBalanceVelocityWeight);
        break;
      case Stage::kCombined:
        std::printf("  stage weights: task.xml defaults\n");
        break;
    }
  }
  std::printf("  speed %g (%.1f planner iterations per control step)", cfg.speed,
              1.0 / cfg.speed);
  if (cfg.horizon > 0) std::printf("   horizon %.2fs", cfg.horizon);
  if (cfg.exploration >= 0) std::printf("   exploration %g", cfg.exploration);
  if (cfg.spline_points > 0) std::printf("   knots %d", cfg.spline_points);
  if (cfg.clearance >= 0) std::printf("   margin %.2fm", cfg.clearance);
  std::printf("   init_noise %g (seed %d)", cfg.init_noise, cfg.seed);
  if (cfg.planner_seed != 0) {
    std::printf("   planner_seed %d", cfg.planner_seed);
  } else {
    std::printf("   planner_seed 0 (NOT REPRODUCIBLE)");
  }
  std::printf("%s\n", cfg.early_exit ? "   early_exit" : "");
  if (absl::GetFlag(FLAGS_per_run)) {
    std::printf("%-4s %7s %6s %8s %8s %8s %7s %7s %7s  %s\n", "run", "t_solve",
                "held%", "cart_end", "cos_end", "spd_end", "cont%", "cost",
                "wall_s", "outcome");
  }

  // For repeats > 1, suffix each dump so runs do not overwrite each other.
  const std::string dump_flag = absl::GetFlag(FLAGS_dump);
  const int dump_runs = absl::GetFlag(FLAGS_dump_runs);
  auto dump_path_for_run = [&dump_flag, repeats, dump_runs](int k)
      -> std::string {
    if (dump_flag.empty() || k >= dump_runs) return "";
    if (repeats == 1) return dump_flag;
    size_t dot = dump_flag.rfind('.');
    if (dot == std::string::npos) return dump_flag + "_" + std::to_string(k);
    return dump_flag.substr(0, dot) + "_" + std::to_string(k) +
           dump_flag.substr(dot);
  };

  int solved = 0, corridor = 0, collided = 0;
  int held_to_end = 0;
  std::vector<double> solve_times;
  double total_wall = 0;
  int total_gaps_passed = 0, num_gaps = 1;
  int ended_early = 0;
  double total_sim_time = 0;
  long total_plan_iterations = 0;
  double total_plan_wall = 0, total_step_wall = 0;
  std::vector<double> iter_ms_p95s, iter_ms_maxes;
  PlannerTimers agg;
  double model_timestep = 0.005;
  for (int k = 0; k < repeats; k++) {
    RunResult r = RunOnce(cfg, k, dump_path_for_run(k));
    // "reached" = the goal state was attained at some point without a
    // disqualifying collision. "held" = it was still there at the end.
    bool reached = r.ever_solved && !r.collided;
    bool held = r.reached_goal && r.upright && !r.collided;
    solved += reached;
    held_to_end += held;
    corridor += r.passed_corridor;
    collided += r.collided;
    if (reached) solve_times.push_back(r.first_solve_time);
    total_wall += r.wall_time;
    total_gaps_passed += r.gaps_passed;
    ended_early += r.ended_early;
    total_sim_time += r.sim_time;
    num_gaps = r.num_gaps;
    total_plan_iterations += r.plan_iterations;
    total_plan_wall += r.plan_wall;
    model_timestep = r.timestep;
    total_step_wall += r.step_wall;
    iter_ms_p95s.push_back(r.iter_ms_p95);
    iter_ms_maxes.push_back(r.iter_ms_max);
    agg.noise += r.timers.noise;
    agg.rollouts += r.timers.rollouts;
    agg.update += r.timers.update;
    agg.derivatives += r.timers.derivatives;
    agg.backward += r.timers.backward;
    agg.nominal += r.timers.nominal;
    agg.available = agg.available || r.timers.available;

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
    // wall time is the planner's cost: the same planning iterations per
    // simulated second for every planner, so the difference is entirely how
    // expensive each iteration was. With --early_exit it is also shortened by
    // solving sooner, so compare it against sim_time, not across runs.
    if (!absl::GetFlag(FLAGS_per_run)) {
      // nothing per-run; the summary is the output
    } else if (r.first_solve_time >= 0) {
      std::printf("%-4d %7.2f %5.1f%% %8.3f %8.3f %8.2f %6.1f%% %7.2f %7.1f"
                  "  %s\n", k, r.first_solve_time, 100.0 * r.solved_fraction,
                  r.final_cart, r.min_cos_upright, r.final_speed,
                  100.0 * r.contact_fraction, r.mean_cost, r.wall_time,
                  outcome);
    } else {
      std::printf("%-4d %7s %5.1f%% %8.3f %8.3f %8.2f %6.1f%% %7.2f %7.1f"
                  "  %s\n", k, "never", 100.0 * r.solved_fraction,
                  r.final_cart, r.min_cos_upright, r.final_speed,
                  100.0 * r.contact_fraction, r.mean_cost, r.wall_time,
                  outcome);
    }
  }
  // Median rather than mean solve time: the distribution is long-tailed (a run
  // that nearly stalls before the corridor and recovers takes several times
  // the typical solve), and one such run moves a mean enough to reverse a
  // ranking.
  auto median = [](std::vector<double> v) -> double {
    if (v.empty()) return -1;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
  };
  // Wald interval on the success rate. With 100 trials a 87/100 and a 82/100
  // are about one standard error apart, so a ranking that ignores this is
  // reading noise.
  auto stderr_pct = [repeats](int k) {
    double p = static_cast<double>(k) / repeats;
    return 100.0 * std::sqrt(p * (1 - p) / repeats);
  };

  if (stage == Stage::kBalance) {
    std::printf("summary[balance]: erected %d/%d | held to end %d/%d\n",
                solved, repeats, held_to_end, repeats);
    std::printf("  (goal set = cart within %.2f m of x=0, all cos(theta) > "
                "%.2f, ||qvel|| < %.2f)\n",
                cfg.goal_tolerance, cfg.upright_tolerance,
                cfg.speed_tolerance);
  } else {
    std::printf("summary[%s]: reached goal state %d/%d | held to end %d/%d | "
                "past corridor %d/%d | collided %d/%d\n",
                stage_name.c_str(), solved, repeats, held_to_end, repeats,
                corridor, repeats, collided, repeats);
    if (!cfg.weights.empty() && cfg.weights[1] == 0.0) {
      std::printf("  (goal set = cart within %.2f m of the goal; Upright "
                  "weight is zero, so pendulum state is not scored)\n",
                  cfg.goal_tolerance);
    } else if (stage == Stage::kCorridor) {
      std::printf("  (goal set = cart within %.2f m of x=6; Upright and "
                  "Velocity weights are zero for this stage, so pendulum "
                  "state is not scored)\n",
                  cfg.goal_tolerance);
    } else {
      std::printf("  (goal set = cart within %.2f m, all cos(theta) > %.2f, "
                  "||qvel|| < %.2f;\n   t_solve = first entry, held%% = "
                  "fraction of the run inside it)\n",
                  cfg.goal_tolerance, cfg.upright_tolerance,
                  cfg.speed_tolerance);
    }
  }
  // One machine-greppable line per invocation, so a 7-planner x 100-trial
  // sweep can be tabulated without re-parsing the prose above.
  if (num_gaps > 1) {
    std::printf("  bottlenecks cleared: %.2f of %d on average\n",
                static_cast<double>(total_gaps_passed) / repeats, num_gaps);
  }

  if (cfg.early_exit && ended_early) {
    std::printf("  early exit ended %d/%d runs; %.1f simulated seconds per run "
                "on average against a %.1f s cap\n",
                ended_early, repeats, total_sim_time / repeats,
                cfg.total_time);
  }

  // ---- timing ----
  // Two clocks, and conflating them is the easy mistake here.
  //
  // Simulated time: control is applied once per mj_step, i.e. every 5 ms of
  // task time (200 Hz), at every --speed. That never changes.
  //
  // Wall time: at --speed s one simulated second takes 1/s real seconds, so a
  // control step may take timestep/s of wall time and the loop is running at
  // 200*s Hz in the real world. At s = 0.25 that is 50 Hz control, not 200.
  //
  // The planner is asked for 1/s iterations inside each of those steps, so its
  // budget per iteration is (timestep/s)/(1/s) = timestep -- the speed cancels.
  // That is why the per-iteration deadline is the same 5 ms at every speed,
  // while the control rate the machine actually delivers is not. Slowing down
  // buys iterations per decision and pays for them in control rate.
  const double timestep_ms = 1e3 * model_timestep;
  double iter_ms = total_plan_iterations
                       ? 1e3 * total_plan_wall / total_plan_iterations
                       : 0.0;
  std::printf("timing: %.3f ms/iteration over %ld iterations "
              "(p95 %.3f, worst %.3f)\n",
              iter_ms, total_plan_iterations, median(iter_ms_p95s),
              iter_ms_maxes.empty()
                  ? 0.0
                  : *std::max_element(iter_ms_maxes.begin(),
                                      iter_ms_maxes.end()));
  std::printf("  planning %.1f s of %.1f s wall (%.0f%%), physics %.1f s\n",
              total_plan_wall, total_wall,
              total_wall > 0 ? 100.0 * total_plan_wall / total_wall : 0.0,
              total_step_wall);
  std::printf("  at speed %g: %.0f Hz control in wall time, %.0f "
              "iteration(s) per control step, %.1f ms of wall per step\n",
              cfg.speed, cfg.speed / model_timestep, 1.0 / cfg.speed,
              timestep_ms / cfg.speed);
  std::printf("  per-iteration budget is %.1f ms at every speed; this planner "
              "uses %.1f%% of it -> %s\n",
              timestep_ms, 100.0 * iter_ms / timestep_ms,
              iter_ms <= timestep_ms ? "fits" : "DOES NOT FIT");
  if (agg.available && agg.Total() > 0) {
    double tot = agg.Total();
    std::printf("  where an iteration goes:");
    auto share = [tot](const char* name, double v) {
      if (v > 0) std::printf("  %s %.1f%%", name, 100.0 * v / tot);
    };
    share("rollouts", agg.rollouts);
    share("noise/swarm", agg.noise);
    share("derivatives", agg.derivatives);
    share("backward", agg.backward);
    share("nominal", agg.nominal);
    share("publish", agg.update);
    std::printf("\n  (noise is summed over %d worker threads, so it is "
                "thread-seconds, not wall)\n",
                cfg.planner_thread_count);
  }
  const std::string label = absl::GetFlag(FLAGS_label);
  // The RESULT line is what sweep tables are built from, so it carries every
  // knob needed to re-run the row: without clearance and planner_seed on it,
  // two rows that differ only in margin, or a row that cannot be reproduced at
  // all, look identical to whatever parses this.
  std::printf("RESULT planner=%s task=%s stage=%s speed=%g horizon=%g "
              "clearance=%g seed=%d planner_seed=%d "
              "trials=%d solved=%d solved_pct=%.1f+-%.1f collided=%d "
              "collided_pct=%.1f t_solve_median=%.2f gaps_mean=%.2f "
              "num_gaps=%d wall_total_s=%.1f ms_per_iter=%.3f "
              "ms_per_iter_p95=%.3f plan_iters=%ld rollout_frac=%.2f\n",
              !label.empty() ? label.c_str()
              : (planner >= 0 && planner < kNumPlannerLabels)
                  ? kPlannerLabel[planner]
                  : "xml",
              task_key.c_str(), stage_name.c_str(), cfg.speed, cfg.horizon,
              cfg.clearance, cfg.seed, cfg.planner_seed,
              repeats, solved, 100.0 * solved / repeats, stderr_pct(solved),
              collided, 100.0 * collided / repeats, median(solve_times),
              static_cast<double>(total_gaps_passed) / repeats, num_gaps,
              total_wall, iter_ms, median(iter_ms_p95s), total_plan_iterations,
              agg.Total() > 0 ? agg.rollouts / agg.Total() : 0.0);
  return 0;
}
