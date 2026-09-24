// lean_bench.cc -- headless bench for the phase-scheduled lean pipeline
// ("Lean H12 Magpie" + a Strategy slot), with a swept TABLE HEIGHT.
//
// WHY THIS EXISTS (2026-08-26, extended 2026-09-04). The deploy pipeline's
// duration and its brace quality are not readable from `testspeed`: that driver
// takes no --strategy, dumps no trajectory, and cannot set a task parameter. This
// one runs the same agent loop, sets Strategy and `Table H`, and logs one row per
// decimated step from lean::ComputeMetrics -- the same metric stack the Research
// GUI reads -- plus the per-body table contact forces the load-path analysis needs.
//
// It changes NO task behaviour. Strategy variants come from the JSONs (lean.h
// loads them from SOURCE_DIR at runtime); the table height comes from parameter
// index 7, which defaults to 0 = OFF = the compiled slab.
//
// usage:
//   lean_bench --task "Lean H12 Magpie" --strategy 25 --table_h 0.86 --seed 0
//              --total_time 120 --out run.csv [--qpos_out qpos.csv] [--threads 6]
//              [--pose_track 1]   re-solve the brace keyframes for the slab
//              [--numeric name=value ...]  override any model <numeric>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include <mujoco/mujoco.h>

#include "mjpc/agent.h"
#include "mjpc/planners/cross_entropy/planner.h"
#include "mjpc/planners/icem/planner.h"
#include "mjpc/planners/icem_dr/planner.h"
#include "mjpc/planners/mppi/planner.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"
#include "mjpc/tasks/tasks.h"
#include "mjpc/tasks/humanoid_bench/lean/lean.h"

namespace {
mjpc::Task* g_task = nullptr;
void residual_callback(const mjModel* model, mjData* data, int stage) {
  if (stage == mjSTAGE_ACC) g_task->Residual(model, data, data->sensordata);
}

// mj_setConst needs an mjData and leaves it at qpos0; never hand it the plant's.
// Normal load the LEFT (bracing) arm puts on the table, from the contacts of
// `data` under `model` -- the same rule as the brace_normal_N log column, so the
// planner-model and plant-model numbers are comparable.
double BraceNormal(const mjModel* model, const mjData* data, int table_body) {
  double f = 0.0;
  mjtNum ft[6];
  for (int c = 0; c < data->ncon; c++) {
    const mjContact& con = data->contact[c];
    int b1 = model->geom_bodyid[con.geom[0]], b2 = model->geom_bodyid[con.geom[1]];
    bool t1 = (b1 == table_body), t2 = (b2 == table_body);
    if (t1 == t2) continue;
    int other = t1 ? b2 : b1;
    const char* bn = mj_id2name(model, mjOBJ_BODY, other);
    if (!bn || std::strncmp(bn, "left_", 5) != 0) continue;
    mj_contactForce(model, data, c, ft);
    f += std::max(0.0, (double)ft[0]);
  }
  return f;
}

void SetConstScratch(mjModel* m) {
  mjData* tmp = mj_makeData(m);
  mj_setConst(m, tmp);
  mj_deleteData(tmp);
}

// ★ 2026-09-24 FRICTION BY SURFACE (studies/brace_friction). The brace pads meet
// the slab through explicit <pair>s, and MuJoCo takes a pair's friction from
// pair_friction, never from geom_friction, so --plant_friction_scale left the
// pad<->table contact at 1.0 in every arm it ran (it did reach the feet). These
// set each surface directly. Table: every <pair> that names a table geom, plus
// the table geoms themselves, which carry priority 1 and so set the friction of
// every other contact they make. Feet: the floor plane and the collision geoms
// of the two ankle-roll bodies; equal priority, so the contact takes the max
// and setting both sides to mu gives mu.
void SetTableMu(mjModel* m, double mu) {
  const int table = mj_name2id(m, mjOBJ_BODY, "table");
  for (int g = 0; g < m->ngeom; g++)
    if (m->geom_bodyid[g] == table) m->geom_friction[3 * g] = mu;
  for (int p = 0; p < m->npair; p++)
    if (m->geom_bodyid[m->pair_geom1[p]] == table || m->geom_bodyid[m->pair_geom2[p]] == table)
      m->pair_friction[5 * p] = m->pair_friction[5 * p + 1] = mu;
}

void SetFootMu(mjModel* m, double mu) {
  const int floor = mj_name2id(m, mjOBJ_GEOM, "floor");
  const int lf = mj_name2id(m, mjOBJ_BODY, "left_ankle_roll_link");
  const int rf = mj_name2id(m, mjOBJ_BODY, "right_ankle_roll_link");
  for (int g = 0; g < m->ngeom; g++) {
    const bool foot = (m->geom_bodyid[g] == lf || m->geom_bodyid[g] == rf) &&
                      (m->geom_contype[g] || m->geom_conaffinity[g]);
    if (g == floor || foot) m->geom_friction[3 * g] = mu;
  }
  for (int p = 0; p < m->npair; p++) {
    const int g1 = m->pair_geom1[p], g2 = m->pair_geom2[p];
    if (g1 == floor || g2 == floor)
      m->pair_friction[5 * p] = m->pair_friction[5 * p + 1] = mu;
  }
}

// `--plant_table_stiff 1`: a rigid slab. The pad pairs ship solref 0.02 with a
// soft onset (solimp 0.015 -> 1 over 22 mm) and the slab geom solref 0.08, and
// under that softness MuJoCo's friction acts like a damper: the loaded pads
// slide 3-80 mm/s at shear ratios 0.05-0.9 of a mu = 1 contact (measured on the
// planner-ablation state tracks), where a forearm on grip tape sticks until it
// breaks away. This sets every contact the slab makes to solref 0.01 and solimp
// (0.95, 0.99, 0.001), the stiff end of MuJoCo's range at a 2 ms step.
void SetTableStiff(mjModel* m) {
  const int table = mj_name2id(m, mjOBJ_BODY, "table");
  const mjtNum ref[2] = {0.01, 1.0}, imp[5] = {0.95, 0.99, 0.001, 0.5, 2.0};
  for (int g = 0; g < m->ngeom; g++)
    if (m->geom_bodyid[g] == table) {
      mju_copy(m->geom_solref + mjNREF * g, ref, 2);
      mju_copy(m->geom_solimp + mjNIMP * g, imp, 5);
    }
  for (int p = 0; p < m->npair; p++)
    if (m->geom_bodyid[m->pair_geom1[p]] == table || m->geom_bodyid[m->pair_geom2[p]] == table) {
      mju_copy(m->pair_solref + mjNREF * p, ref, 2);
      mju_copy(m->pair_solreffriction + mjNREF * p, ref, 2);
      mju_copy(m->pair_solimp + mjNIMP * p, imp, 5);
    }
}

// One surface's share of the contact set at one step: total force ON THE ROBOT
// in world axes, summed normal force, the worst per-contact use of the friction
// pyramid (|f_t1|/mu1 + |f_t2|/mu2 [+ |tau|/mu3]; 1 = on the pyramid = sliding),
// and the normal-force-weighted tangential slip speed at the contact points.
struct ContactGroup {
  double fn = 0.0, F[3] = {0.0, 0.0, 0.0}, util = 0.0, slip_fw = 0.0;
  int n = 0;
  double slip() const { return fn > 1e-9 ? slip_fw / fn : 0.0; }
};

// Velocity of the material point of body b that sits at world point p.
void PointVelocity(const mjModel* m, const mjData* d, int b, const mjtNum* p, mjtNum v[3]) {
  mjtNum res[6];
  mj_objectVelocity(m, d, mjOBJ_BODY, b, res, 0);   // (ang:lin) at xipos, world axes
  mjtNum r[3], wxr[3];
  mju_sub3(r, p, d->xipos + 3 * b);
  mju_cross(wxr, res, r);
  mju_add3(v, res + 3, wxr);
}

std::string Arg(int argc, char** argv, const char* key, const char* dflt) {
  for (int i = 1; i + 1 < argc; i++)
    if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
  return dflt;
}

// Same xorshift perturbation testspeed uses, but seeded, so "robustness over N
// seeds" means N genuinely different starts rather than N identical runs.
void PerturbState(const mjModel* m, mjData* d, double scale, uint64_t seed) {
  if (scale <= 0.0) return;
  uint64_t s = 0x2545F4914F6CDD1DULL + seed * 0x9E3779B97F4A7C15ULL;
  auto next = [&s]() {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return static_cast<double>(s >> 11) * (1.0 / 4503599627370496.0) - 1.0;
  };
  for (int i = 0; i < 20; i++) next();               // warm the state up
  for (int i = 0; i < m->nq; i++) d->qpos[i] += scale * next();
  for (int i = 0; i < m->nv; i++) d->qvel[i] += scale * next();
  mj_normalizeQuat(m, d->qpos);
}

// The ComputeMetrics keys logged, in column order. Anything the task does not
// report in a given phase comes out as nan rather than silently as 0.
const char* const kMetricKeys[] = {
    "phase_time",        "brace_force",     "brace_force_target",
    "brace_hand_x",      "brace_hand_z",    "reach_err",
    "reach_hand_x",      "reach_hand_z",    "reach_tgt_x",
    "reach_tgt_z",       "com_x",           "com_z",
    "com_excursion_sagittal", "com_beyond_foot_edge",
    "cop_beyond_foot_edge",   "icp_x",      "foot_force_total",
    "torque_saturation_max",  "joint_velocity_max",
    "palm_contact_force", "reach_hand_contact_force"};
constexpr int kNMetric = sizeof(kMetricKeys) / sizeof(kMetricKeys[0]);
}  // namespace

int main(int argc, char** argv) {
  const std::string task_name = Arg(argc, argv, "--task", "Lean H12 Magpie");
  const int strategy   = std::atoi(Arg(argc, argv, "--strategy", "25").c_str());
  const int seed       = std::atoi(Arg(argc, argv, "--seed", "0").c_str());
  const double table_h = std::atof(Arg(argc, argv, "--table_h", "0").c_str());
  const double perturb = std::atof(Arg(argc, argv, "--perturb", "0.003").c_str());
  const double total_time = std::atof(Arg(argc, argv, "--total_time", "160").c_str());
  const int threads    = std::atoi(Arg(argc, argv, "--threads", "6").c_str());
  const int spp        = std::atoi(Arg(argc, argv, "--spp", "4").c_str());
  const double log_hz  = std::atof(Arg(argc, argv, "--log_hz", "50").c_str());
  const double vid_hz  = std::atof(Arg(argc, argv, "--video_hz", "50").c_str());
  const double hold_after_final =
      std::atof(Arg(argc, argv, "--hold_after_final", "3.0").c_str());
  const std::string out      = Arg(argc, argv, "--out", "");
  const std::string qpos_out = Arg(argc, argv, "--qpos_out", "");
  // Passive provenance for independent, time-aligned evaluation. These do not
  // change the state, controller, sampling distribution or stopping condition.
  const std::string state_out = Arg(argc, argv, "--state_out", "");
  const std::string model_out = Arg(argc, argv, "--model_out", "");
  // ★ 2026-09-06 STANCE SHIFT (`--stance_shift_x`, m forward; 0 = OFF =
  // byte-identical). BENCH-ONLY on purpose: it moves the reset pose, not the
  // task, so `lean.cc` and the XMLs stay untouched and the shipped controller is
  // unchanged.
  //   WHY. The `home` keyframe every entry point resets to (here, app.cc:423,
  //   deploy_common.cc:899) puts the feet 260 mm behind the slab's near edge.
  //   All three `forearm_brace_*` keyframes were authored for 197 mm -- a 63 mm
  //   step the ladder cannot take, because `Foot Left/Right Up` carry weight 2000
  //   on all nine rungs of strategy 25 and there is no stepping rung. Measured
  //   consequence (studies/table_height/probe_stance.py): the forearm pad's best
  //   reach past the near edge is +0.140 m at a 0.985 m face, +0.054 at 1.035 and
  //   -0.066 at 1.085, crossing zero exactly where completions stop.
  //   SAFE TO SHIFT IN THIS STRATEGY. Every live forward-x term is measured from
  //   midfoot and travels with the feet (`Pelvis Forward` band -> midfoot+0.05
  //   and midfoot+`pelvis_cap_fwd`; `com_cap_fwd`; `brace_com_hold`). The one
  //   absolute-world-x constant, `brace_lead_x0` 0.24 against `data->qpos[0]`,
  //   sits behind JSON weight "Brace Reach Lead" = 0.0 on all nine rungs, so it
  //   cannot bite. Re-check both facts before using this flag on another strategy.
  const double stance_shift_x =
      std::atof(Arg(argc, argv, "--stance_shift_x", "0").c_str());
  // Model <numeric> overrides, applied to the loaded model before the first
  // Transition. `brace_pose_track` ships 0 (off, byte-identical) so the A and B
  // arms of a comparison differ by one value rather than by two model files.
  const double pose_track =
      std::atof(Arg(argc, argv, "--pose_track", "-1").c_str());
  // `--numeric name=value`, repeatable: any model <numeric> can be overridden
  // from the command line, so an A/B differs by one value on one line rather
  // than by two model files. `--pose_track` is the same mechanism, kept because
  // it is the one every sweep in this study uses.
  std::vector<std::pair<std::string, double>> numeric_over;
  for (int i = 1; i + 1 < argc; i++) {
    if (std::strcmp(argv[i], "--numeric") != 0) continue;
    std::string kv = argv[i + 1];
    size_t eq = kv.find('=');
    if (eq == std::string::npos) {
      std::fprintf(stderr, "[bench] --numeric wants name=value, got '%s'\n",
                   kv.c_str());
      return 2;
    }
    numeric_over.emplace_back(kv.substr(0, eq), std::atof(kv.c_str() + eq + 1));
  }

  // ★ 2026-09-11 `--gains deploy`: put the DEPLOY NODE's joint PD on the plant
  // and on the planner model. The lean XML ships arm kp 40 on every arm joint
  // and ankle-roll kp 200; the node (h12_control_node.cc KP[]/KV[], patched into
  // its planner model by PatchActuators so "node KP == planner kp == twin PD")
  // runs shoulder 90/60, yaw 40, elbow 90, wrists 15 and ankle roll 80. A bench
  // run at the node's plan rate with the XML gains is therefore not the robot's
  // joint law. Default off = the XML = every earlier bench run. Values copied
  // from h12_control_node.cc (2026-08-22 table); keep them in sync by hand.
  const std::string gains = Arg(argc, argv, "--gains", "xml");
  // ★ 2026-09-13 PLAN LATENCY (`--latency_steps L`, plant steps; 0 = OFF =
  // byte-identical). Each plan is computed from the plant state L steps in the
  // past and executed now, i.e. an uncompensated compute/transport delay of
  // L x 2 ms. The deploy node predicts the state forward before planning, so
  // the robot sits between L = 0 and the raw delay; the bench with L = 0 is
  // the compensated ideal, L = spp is one whole plan interval of staleness.
  const int latency_steps = std::atoi(Arg(argc, argv, "--latency_steps", "0").c_str());
  // `--latency_compensate 1`: what the deploy node does -- before planning,
  // roll the delayed snapshot forward by latency_steps under the current
  // policy on a scratch mjData and plan from that predicted state (at the
  // current time). 0 = raw delay (the option above).
  const bool latency_compensate =
      std::atoi(Arg(argc, argv, "--latency_compensate", "0").c_str()) != 0;
  // ★ 2026-09-13 MODEL MISMATCH on the PLANT only (the planner keeps the model
  // it was initialised with, i.e. the --gains table): `--plant_kp_scale s`
  // multiplies every actuator kp on the plant by s after the planner copy is
  // synced; `--plant_mass_scale s` multiplies every body mass (and inertia) on
  // the plant by s. 1 = OFF = byte-identical. The planner then plans on a
  // model that is wrong by that factor, which is the sim-to-real question.
  const double plant_kp_scale = std::atof(Arg(argc, argv, "--plant_kp_scale", "1").c_str());
  const double plant_mass_scale = std::atof(Arg(argc, argv, "--plant_mass_scale", "1").c_str());
  // `--plant_friction_scale s`: every geom's sliding friction on the plant x s
  // (MuJoCo takes the pair friction as the max of the two geoms', so the
  // pad-table and foot-floor contacts scale by s). The planner keeps the
  // model's friction. 1 = OFF.
  const double plant_friction_scale = std::atof(Arg(argc, argv, "--plant_friction_scale", "1").c_str());
  // ★ 2026-09-24 (studies/brace_friction): sliding friction of the robot<->table
  // and foot<->floor contacts set to an absolute value, on the plant
  // (--plant_*_mu) and on the planner's model (--planner_*_mu), independently.
  // See SetTableMu / SetFootMu. <= 0 = OFF = byte-identical.
  const double plant_table_mu = std::atof(Arg(argc, argv, "--plant_table_mu", "0").c_str());
  const double plant_foot_mu = std::atof(Arg(argc, argv, "--plant_foot_mu", "0").c_str());
  const double planner_table_mu = std::atof(Arg(argc, argv, "--planner_table_mu", "0").c_str());
  const double planner_foot_mu = std::atof(Arg(argc, argv, "--planner_foot_mu", "0").c_str());
  const bool plant_table_stiff = std::atoi(Arg(argc, argv, "--plant_table_stiff", "0").c_str()) != 0;
  // `--plant_table_dz m`: the plant's slab sits m higher than the planner's (a
  // table-height or arm-droop error the robot does not know about). The shipped
  // CEM parks the forearm pad 10-30 mm above the slab it believes in, so a
  // slab a little higher than believed is met while the planner is still
  // bringing the torso down. 0 = OFF.
  const double plant_table_dz = std::atof(Arg(argc, argv, "--plant_table_dz", "0").c_str());
  // ★ 2026-09-24 PULL-BACK ASSIST (`--assist_fx N`, 0 = OFF = byte-identical).
  // On hardware the braced lean is made to settle by an operator pulling the
  // robot backwards as it comes down, so it falls INTO the brace instead of
  // shearing forward along the slab. This is that force: a constant world-x
  // force on `--assist_body` (negative = backwards, away from the table),
  // eased in over --assist_ramp s while the ladder is in rungs
  // [--assist_from_phase, --assist_until_phase], and zero otherwise.
  //   It goes in `data->xfrc_applied`, which the planner's state copy does not
  // carry, so the planner is blind to it exactly as it is blind to the
  // operator. Everything the assist does to the plan is through the state it
  // produces, which is the point.
  const double assist_fx = std::atof(Arg(argc, argv, "--assist_fx", "0").c_str());
  const double assist_ramp = std::atof(Arg(argc, argv, "--assist_ramp", "0.5").c_str());
  const int assist_from_phase = std::atoi(Arg(argc, argv, "--assist_from_phase", "1").c_str());
  const int assist_until_phase = std::atoi(Arg(argc, argv, "--assist_until_phase", "2").c_str());
  const std::string assist_body = Arg(argc, argv, "--assist_body", "torso_link");
  // `--assist_gap m`: only pull while the forearm pad is within m of the slab
  // face, i.e. while the arm is coming down onto it, which is when a person
  // actually takes hold of the robot. <= 0 = the whole phase window. Pulling
  // from the start of the 12 s lean rung instead makes the lean-onset backward
  // fall more likely, which is the opposite of the intervention.
  const double assist_gap = std::atof(Arg(argc, argv, "--assist_gap", "0.15").c_str());
  // `--contact_out f.csv`: per-step contact record for the brace (left arm vs
  // table), each foot vs the floor and the rest of the robot vs the table, plus
  // the forearm/wrist pad and sole positions and the robot CoM, every
  // 1/--contact_hz s (default every 2 ms plant step: a brace impact lasts tens
  // of ms and the 50 Hz state track cannot see it). Written after mj_step from
  // the forward pass that step used, so forces, kinematics and time agree.
  const std::string contact_out = Arg(argc, argv, "--contact_out", "");
  const double contact_hz = std::atof(Arg(argc, argv, "--contact_hz", "500").c_str());

  // ★ 2026-09-14 PAYLOAD (studies/brace_payload). A point mass attached to the
  // REACHING gripper body when the ladder enters --payload_phase (strat 11 =
  // strat 27 + timeout_advance: rung 5 is the first rung after the grasp
  // close). `--payload_true M` goes on the plant, ramped in over --payload_ramp s
  // (the object leaving the table); `--payload_belief M` goes on the planner's
  // model at once (the planner "knows" what it just grasped). `--bft_add X`
  // writes X N into the brace_force_target_add numeric of BOTH models at the
  // same rung and clears it once the ladder passes --bft_add_until, so a mass
  // belief can preload the brace for the carry. All 0 = OFF = byte-identical.
  const double payload_true = std::atof(Arg(argc, argv, "--payload_true", "0").c_str());
  const double payload_belief = std::atof(Arg(argc, argv, "--payload_belief", "0").c_str());
  const int payload_phase = std::atoi(Arg(argc, argv, "--payload_phase", "5").c_str());
  const double payload_ramp = std::atof(Arg(argc, argv, "--payload_ramp", "0.3").c_str());
  const double bft_add = std::atof(Arg(argc, argv, "--bft_add", "0").c_str());
  const int bft_add_until = std::atoi(Arg(argc, argv, "--bft_add_until", "8").c_str());
  const std::string payload_body = Arg(argc, argv, "--payload_body", "right_magpie_gripper");
  // ★ 2026-09-15 SCENARIO BASELINE (studies/brace_payload). `--scenario_masses
  // 0,2,4` switches the planner to iCEM-DR (agent_planner 8) in scenario mode:
  // from the attach on, every candidate is rolled out on one model copy per
  // listed mass (kg added to --payload_body) and scored by --scenario_agg
  // (min | mean | max). Before the attach the set is {0}, i.e. plain iCEM.
  // --payload_belief still sets the nominal model the set is built from.
  const std::string scenario_masses_s = Arg(argc, argv, "--scenario_masses", "");
  const std::string scenario_agg_s = Arg(argc, argv, "--scenario_agg", "mean");
  std::vector<double> scenario_masses;
  if (!scenario_masses_s.empty()) {
    std::stringstream ss(scenario_masses_s);
    std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) scenario_masses.push_back(std::atof(tok.c_str()));
  }
  const int scenario_agg = scenario_agg_s == "min" ? 0 : scenario_agg_s == "mean" ? 1 :
                           scenario_agg_s == "max" ? 2 : -1;
  if (scenario_agg < 0) {
    std::fprintf(stderr, "[bench] --scenario_agg wants min|mean|max\n");
    return 2;
  }
  // ★ 2026-09-15 PLAN DUMP. `--plan_out f.csv` writes, after every plan
  // iteration in phases >= --plan_out_from_phase, the nominal trajectory the
  // planner intends to follow, every --plan_out_stride-th step of the horizon,
  // evaluated twice from the same predicted state: on the PLANNER model
  // (what the planner believes: *_belief) and on the PLANT model (*_true).
  const std::string plan_out = Arg(argc, argv, "--plan_out", "");
  const int plan_out_from_phase = std::atoi(Arg(argc, argv, "--plan_out_from_phase", "8").c_str());
  const int plan_out_stride = std::max(1, std::atoi(Arg(argc, argv, "--plan_out_stride", "5").c_str()));
  // ★ 2026-09-21 SAMPLE DUMP (talk figures for Alg. 1 of the lean paper).
  // `--samples_out DIR --samples_at t1,t2,...` writes, at the first plan tick
  // at or after each listed plant time, what one CEM iteration holds: the
  // centre (the resampled nominal spline) and its rollout, the per-parameter
  // sampling std and the drawn noise, every candidate spline with its rollout
  // and return, the elite order, and the folded spline (elite mean) rolled out
  // from the same state. One directory per tick. CEM (agent_planner 5) only;
  // the flag is ignored for every other planner. Empty = OFF = byte-identical.
  const std::string samples_out = Arg(argc, argv, "--samples_out", "");
  std::vector<double> samples_at;
  {
    std::stringstream ss(Arg(argc, argv, "--samples_at", ""));
    std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) samples_at.push_back(std::atof(tok.c_str()));
  }
  size_t samples_next = 0;

  // Diagnostic compatibility switch: 0 reproduces the historical bench.
  // Agent::Initialize copies mjModel before Table H / pose retargeting runs.
  // Synchronize once after the first Transition so rollout physics and posture
  // references see the same environment as the plant (fixed-height bench).
  const bool sync_planning_model =
      std::atoi(Arg(argc, argv, "--sync_planning_model", "1").c_str()) != 0;

  mjpc::Agent agent;
  agent.SetTaskList(mjpc::GetTasks());
  agent.gui_task_id = agent.GetTaskIdByName(task_name);
  if (agent.gui_task_id == -1) {
    std::fprintf(stderr, "bad --task '%s'\n", task_name.c_str());
    return 2;
  }
  auto load = agent.LoadModel();
  mjModel* model = load.model.get();
  if (!model) { std::fprintf(stderr, "%s\n", load.error.c_str()); return 2; }
  if (pose_track >= 0.0) {
    int n = mj_name2id(model, mjOBJ_NUMERIC, "brace_pose_track");
    if (n < 0) {
      std::fprintf(stderr, "[bench] --pose_track: the model has no "
                           "brace_pose_track numeric\n");
      return 2;
    }
    model->numeric_data[model->numeric_adr[n]] = pose_track;
    std::fprintf(stderr, "[bench] brace_pose_track = %.1f\n", pose_track);
  }
  for (const auto& kv : numeric_over) {
    int n = mj_name2id(model, mjOBJ_NUMERIC, kv.first.c_str());
    if (n < 0) {
      std::fprintf(stderr, "[bench] --numeric: no such numeric '%s'\n",
                   kv.first.c_str());
      return 2;
    }
    model->numeric_data[model->numeric_adr[n]] = kv.second;
    std::fprintf(stderr, "[bench] %s = %g\n", kv.first.c_str(), kv.second);
  }
  if (!scenario_masses.empty()) {
    int n = mj_name2id(model, mjOBJ_NUMERIC, "agent_planner");
    if (n < 0) { std::fprintf(stderr, "[bench] model has no agent_planner numeric\n"); return 2; }
    model->numeric_data[model->numeric_adr[n]] = 8;   // iCEM-DR
    std::fprintf(stderr, "[bench] agent_planner = 8 (iCEM-DR, scenario mode %s over %s)\n",
                 scenario_agg_s.c_str(), scenario_masses_s.c_str());
  }
  if (gains == "deploy") {
    static const double kKP[27] = {150, 200, 200, 200, 200, 80, 150, 200, 200, 200, 200, 80, 200,
                                   90, 60, 40, 90, 15, 15, 15, 90, 60, 40, 90, 15, 15, 15};
    static const double kKV[27] = {5, 5, 5, 5, 4, 4, 5, 5, 5, 5, 4, 4, 5,
                                   10, 10, 10, 10, 2, 2, 2, 10, 10, 10, 10, 2, 2, 2};
    if (model->nu != 27) {
      std::fprintf(stderr, "[bench] --gains deploy expects nu=27, model has %d\n", model->nu);
      return 2;
    }
    for (int i = 0; i < model->nu; i++) {
      model->actuator_gainprm[i * mjNGAIN + 0] = kKP[i];
      model->actuator_biasprm[i * mjNBIAS + 1] = -kKP[i];
      model->actuator_biasprm[i * mjNBIAS + 2] = -kKV[i];
    }
    std::fprintf(stderr, "[bench] gains = deploy (h12_control_node KP/KV on plant + planner)\n");
  } else if (gains != "xml") {
    std::fprintf(stderr, "[bench] --gains wants xml|deploy\n");
    return 2;
  }
  mjData* data = mj_makeData(model);

  const std::string start_key = Arg(argc, argv, "--start_key", "home");
  int home_id = mj_name2id(model, mjOBJ_KEY, start_key.c_str());
  if (home_id < 0) {
    std::fprintf(stderr, "[bench] unknown --start_key '%s'\n", start_key.c_str());
    return 2;
  }
  std::fprintf(stderr, "[bench] start_key = %s\n", start_key.c_str());
  if (home_id >= 0) mj_resetDataKeyframe(model, data, home_id);
  // qpos[0..6] is the pelvis free joint; the legs follow by kinematics, so this
  // translates the whole robot. The free `object` lives further down qpos and is
  // deliberately NOT moved -- it belongs to the table frame, like the slab.
  if (stance_shift_x != 0.0) {
    data->qpos[0] += stance_shift_x;
    std::fprintf(stderr, "[bench] stance_shift_x = %+.3f m (base_x %.3f)\n",
                 stance_shift_x, data->qpos[0]);
  }
  PerturbState(model, data, perturb, static_cast<uint64_t>(seed));
  mj_forward(model, data);

  agent.estimator_enabled = false;
  agent.Initialize(model);
  agent.Allocate();
  agent.Reset(data->ctrl);
  agent.plan_enabled = true;
  // Strategy must land BEFORE the first Transition: lean.cc reloads the strategy
  // JSON when it sees the parameter change, and reloading mid-run would reset the
  // phase clock we are here to measure. Table H likewise -- it moves the slab on
  // the first Transition, before any cost has been evaluated against the old one.
  if (agent.SetParamByName("residual_Strategy", strategy) < 0)
    std::fprintf(stderr, "[bench] WARNING: task has no residual_Strategy param\n");
  if (table_h > 0.0 && agent.SetParamByName("residual_Table H", table_h) < 0)
    std::fprintf(stderr, "[bench] WARNING: task has no residual_Table H param\n");

  g_task = agent.ActiveTask();
  mjcb_sensor = &residual_callback;
  auto* mppi = dynamic_cast<mjpc::MPPIPlanner*>(&agent.ActivePlanner());
  auto* cem = dynamic_cast<mjpc::CrossEntropyPlanner*>(&agent.ActivePlanner());
  auto* icem = dynamic_cast<mjpc::iCEMPlanner*>(&agent.ActivePlanner());
  auto* icemdr = dynamic_cast<mjpc::iCEMDRPlanner*>(&agent.ActivePlanner());
  if (!scenario_masses.empty() && !icemdr) {
    std::fprintf(stderr, "[bench] --scenario_masses needs the iCEM-DR planner\n");
    return 2;
  }
  auto* lean_task = dynamic_cast<mjpc::lean*>(g_task);
  if (!lean_task) {
    std::fprintf(stderr, "[bench] --task is not a lean task; phase log unavailable\n");
  }

  const int table_body = mj_name2id(model, mjOBJ_BODY, "table");
  // The load path, by body. `left_elbow_link` IS the forearm: the
  // `left_forearm_pad` capsule is a geom on it (checked against the compiled
  // model). `left_wrist_yaw_link` carries `left_wrist_pad` and is here because
  // an unnamed body quietly taking the brace is the recurring failure in this
  // task -- if it is not a column, it does not get looked at.
  // ★ 2026-09-04 the RIGHT arm is here too. The nominal run showed 15-20 N of
  // robot-on-table contact all through stand_up with every LEFT body reading
  // zero, i.e. something undeclared was already resting on the slab before the
  // brace began. Naming the reach arm is how that gets attributed instead of
  // disappearing into a residual.
  const char* kBraceBodies[] = {"left_shoulder_yaw_link", "left_elbow_link",
                                "left_wrist_yaw_link", "left_magpie_gripper",
                                "right_elbow_link", "right_wrist_yaw_link",
                                "right_magpie_gripper",
                                "torso_link", "pelvis"};
  constexpr int kNBrace = sizeof(kBraceBodies) / sizeof(kBraceBodies[0]);
  int brace_id[kNBrace];
  for (int i = 0; i < kNBrace; i++)
    brace_id[i] = mj_name2id(model, mjOBJ_BODY, kBraceBodies[i]);
  const int torso_id = mj_name2id(model, mjOBJ_BODY, "torso_link");
  const int assist_bid = mj_name2id(model, mjOBJ_BODY, assist_body.c_str());
  if (assist_fx != 0.0 && assist_bid < 0) {
    std::fprintf(stderr, "[bench] --assist_body '%s' not found\n", assist_body.c_str());
    return 2;
  }
  double assist_t0 = -1.0;      // when the assist window opened
  double assist_now = 0.0;      // the force applied this step, for the log

  // Slab face + forearm-pad geometry, read from the COMPILED model after the
  // parameter has been applied (first Transition below), so the pad clearance
  // column means the same thing at every height.
  const int object_body = mj_name2id(model, mjOBJ_BODY, "object");
  const int rgrip_body = mj_name2id(model, mjOBJ_BODY, "right_magpie_gripper");
  const int target_body = mj_name2id(model, mjOBJ_BODY, "target");
  const int target_mocap =
      (target_body >= 0) ? model->body_mocapid[target_body] : -1;
  const int tt_gid  = mj_name2id(model, mjOBJ_GEOM, "table_top");
  const int pad_gid = mj_name2id(model, mjOBJ_GEOM, "left_forearm_pad");

  FILE* fo = out.empty() ? stdout : std::fopen(out.c_str(), "w");
  FILE* fq = qpos_out.empty() ? nullptr : std::fopen(qpos_out.c_str(), "w");
  FILE* fs = state_out.empty() ? nullptr : std::fopen(state_out.c_str(), "w");
  FILE* fp = plan_out.empty() ? nullptr : std::fopen(plan_out.c_str(), "w");
  FILE* fc = contact_out.empty() ? nullptr : std::fopen(contact_out.c_str(), "w");
  if ((!state_out.empty() && !fs) || !fo || (!qpos_out.empty() && !fq) ||
      (!plan_out.empty() && !fp) || (!contact_out.empty() && !fc)) {
    std::fprintf(stderr, "[bench] failed to open requested output file\n");
    return 2;
  }
  // contact record ids (see --contact_out)
  const int floor_gid = mj_name2id(model, mjOBJ_GEOM, "floor");
  const int lfoot_body = mj_name2id(model, mjOBJ_BODY, "left_ankle_roll_link");
  const int rfoot_body = mj_name2id(model, mjOBJ_BODY, "right_ankle_roll_link");
  const int wpad_gid = mj_name2id(model, mjOBJ_GEOM, "left_wrist_pad");
  const int pelvis_body = mj_name2id(model, mjOBJ_BODY, "pelvis");
  // Sole material point in the ankle-roll frame: mid-sole, on the sole plane
  // (the collision hull spans x -0.086..0.174, y +-0.043, sole at z -0.045).
  // The ankle-roll ORIGIN is 45 mm above the sole and moves whenever the ankle
  // rotates, so it is not a slip measure.
  const mjtNum kSoleLocal[3] = {0.044, 0.0, -0.045};
  const int contact_every =
      std::max(1, static_cast<int>(std::lround(1.0 / (contact_hz * model->opt.timestep))));
  if (fc) {
    std::fprintf(fc, "t,phase");
    for (const char* g : {"br", "fl", "fr"})
      std::fprintf(fc, ",%s_fn,%s_fx,%s_fy,%s_fz,%s_util,%s_slip,%s_n", g, g, g, g, g, g, g);
    std::fprintf(fc, ",ot_fn,ot_fx,ot_fy,ot_fz,pd_fn,pd_fx,pd_fy,pd_fz,pd_util,pd_slip,pd_n,pad_x,pad_y,pad_z,padv_x,padv_y,padv_z,"
                     "wpad_x,wpad_y,wpad_z,soleL_x,soleL_y,soleR_x,soleR_y,com_x,com_y,com_z,pelvis_z,assist_fx\n");
  }
  if (fp) {
    std::fprintf(fp, "t_plan,phase,k,t_pred,cost_k,pelvis_x,pelvis_z,rhand_x,"
                     "com_edge_belief,cop_edge_belief,icp_x_belief,brace_belief,"
                     "com_edge_true,cop_edge_true,icp_x_true,brace_true\n");
  }
  // Scratch data for evaluating one state on the planner model (belief) and on
  // the plant model (truth). Same layout; `pd` is made from the planner copy.
  mjData* pd = mj_makeData(agent.GetModel());
  mjData* td = mj_makeData(model);
  // Evaluate the metrics of a (qpos, qvel, ctrl) triple on a model: forward
  // dynamics from that state, then the task's metrics + the brace load.
  auto eval_state = [&](const mjModel* m, mjData* d, const double* qpos, const double* qvel,
                        const double* ctrl, double t, double out[4]) {
    mju_copy(d->qpos, qpos, m->nq);
    mju_copy(d->qvel, qvel, m->nv);
    if (ctrl) mju_copy(d->ctrl, ctrl, m->nu); else mju_zero(d->ctrl, m->nu);
    if (m->nmocap) {
      mju_copy(d->mocap_pos, data->mocap_pos, 3 * m->nmocap);
      mju_copy(d->mocap_quat, data->mocap_quat, 4 * m->nmocap);
    }
    if (m->nuserdata) mju_copy(d->userdata, data->userdata, m->nuserdata);
    d->time = t;
    mj_forward(m, d);
    std::map<std::string, double> mm;
    std::string pn;
    g_task->ComputeMetrics(m, d, &mm, &pn);
    auto get = [&](const char* k) { auto it = mm.find(k); return it == mm.end() ? std::nan("") : it->second; };
    out[0] = get("com_beyond_foot_edge");
    out[1] = get("cop_beyond_foot_edge");
    out[2] = get("icp_x");
    out[3] = BraceNormal(m, d, table_body);
  };
  if (fs) {
    std::fprintf(fs, "t,phase");
    for (int k = 0; k < model->nq; ++k) std::fprintf(fs, ",q%d", k);
    for (int k = 0; k < model->nv; ++k) std::fprintf(fs, ",v%d", k);
    for (int k = 0; k < model->nu; ++k) std::fprintf(fs, ",u%d", k);
    for (int k = 0; k < model->nv; ++k) std::fprintf(fs, ",warm%d", k);
    std::fprintf(fs, "\n");
  }
  std::fprintf(fo, "t,phase,phase_name,pelvis_z,torso_tilt_deg,face_z,pad_clear,"
                   "f_shoulder,f_forearm,f_wrist,f_gripper,"
                   "f_r_elbow,f_r_wrist,f_r_gripper,f_torso,f_pelvis,"
                   "f_other,f_robot_total,cost,"
                   "rhand_x,rhand_y,rhand_z,tgt_x,tgt_y,tgt_z");
  std::fprintf(fo, ",jaw_x,jaw_y,jaw_z,brace_normal_N,trunk_normal_N");
  // Planner-side channels. plan_return = BestTrajectory()->total_return: the
  // NOMINAL rollout's return for CEM/iCEM, the WINNING sample's for PS/MPPI
  // (the two are not the same quantity; compare within a family only).
  // mppi_ess / mppi_spread are MPPI's softmax effective sample size and the
  // batch return spread; nan for every other planner.
  // cem_std_mean: CEM/iCEM's refit elite std, mean over the (knots x nu)
  // parameters actually in use, BEFORE the std_min floor is applied -- so it
  // can be read against std_min to see whether the adaptive variance is doing
  // anything. nan for PS/MPPI.
  std::fprintf(fo, ",plan_return,mppi_ess,mppi_spread,cem_std_mean");
  // The plant's current state evaluated on the PLANNER model: where the planner
  // believes the CoM is, and what load it believes the brace carries.
  std::fprintf(fo, ",com_edge_belief,cop_edge_belief,brace_belief");
  for (int k = 0; k < kNMetric; k++) std::fprintf(fo, ",%s", kMetricKeys[k]);
  std::fprintf(fo, "\n");
  if (fq) {
    std::fprintf(fq, "t");
    for (int i = 0; i < model->nq; i++) std::fprintf(fq, ",q%d", i);
    std::fprintf(fq, "\n");
  }

  mjpc::ThreadPool pool(threads);
  const int total_steps = static_cast<int>(std::ceil(total_time / model->opt.timestep));
  const int log_every = std::max(1, static_cast<int>(1.0 / (log_hz * model->opt.timestep)));
  const int vid_every = std::max(1, static_cast<int>(1.0 / (vid_hz * model->opt.timestep)));

  int last_phase = -1, n_phases = 0;
  double final_phase_since = -1.0, t_complete = -1.0;
  // payload bookkeeping
  const int payload_bid = mj_name2id(model, mjOBJ_BODY, payload_body.c_str());
  const int bft_add_nid = mj_name2id(model, mjOBJ_NUMERIC, "brace_force_target_add");
  double payload_t0 = -1.0, payload_base_mass = 0.0, payload_applied = 0.0;
  bool payload_belief_done = false, bft_add_on = false, bft_add_cleared = false;
  int max_phase_seen = -1;
  double peak_brace_carry = 0.0, min_pelvis_carry = 9.0, max_tilt_carry = 0.0;
  if ((payload_true != 0.0 || payload_belief != 0.0) && payload_bid < 0) {
    std::fprintf(stderr, "[bench] --payload_body '%s' not found\n", payload_body.c_str());
    return 2;
  }
  if (bft_add != 0.0 && bft_add_nid < 0) {
    std::fprintf(stderr, "[bench] model has no brace_force_target_add numeric\n");
    return 2;
  }
  if (payload_bid >= 0) payload_base_mass = model->body_mass[payload_bid];
  if (icemdr) {
    // Scenario mode from step 0 with the single member {0}: the planner is
    // then the plain iCEM until the attach swaps in the listed set.
    icemdr->scenario_body_ = payload_bid;
    icemdr->scenario_mass_ = {0.0};
    icemdr->scenario_agg_ = scenario_agg;
  }
  bool fell = false;
  double face_z = 0.0;
  std::vector<double> phase_enter(64, -1.0);
  std::map<std::string, double> metrics;
  std::string phase_name_m;
  auto wall0 = std::chrono::steady_clock::now();

  // Ring buffer of plant snapshots for --latency_steps (see the option above).
  struct Snap { std::vector<double> qpos, qvel, act, mocap_pos, mocap_quat, userdata; double time; };
  std::vector<Snap> snaps(std::max(1, latency_steps + 1));
  for (auto& sn : snaps) {
    sn.qpos.resize(model->nq); sn.qvel.resize(model->nv); sn.act.resize(std::max(1, model->na));
    sn.mocap_pos.resize(std::max(1, 3 * model->nmocap)); sn.mocap_quat.resize(std::max(1, 4 * model->nmocap));
    sn.userdata.resize(std::max(1, model->nuserdata)); sn.time = 0.0;
  }

  for (int i = 0; i < total_steps; i++) {
    g_task->Transition(model, data);
    if (i == 0) {
      mjModel* pm = agent.GetModel();
      const int key = mj_name2id(model, mjOBJ_KEY, "forearm_brace_lean");
      auto face = [table_body, tt_gid](const mjModel* m) {
        return m->body_pos[3 * table_body + 2] + m->geom_pos[3 * tt_gid + 2] +
               m->geom_size[3 * tt_gid + 2];
      };
      if (key >= 0 && table_body >= 0 && tt_gid >= 0) std::fprintf(stderr,
          "[bench-model-before] plant_face=%.4f planner_face=%.4f "
          "plant_brace_z=%.4f planner_brace_z=%.4f sync=%d\n",
          face(model), face(pm), model->key_qpos[key * model->nq + 2],
          pm->key_qpos[key * pm->nq + 2], int(sync_planning_model));
      if (sync_planning_model) {
        // Same model layout; keep the pointer held by the planners and their
        // allocated mjData. PlanIteration sets the planning timestep itself.
        mj_copyModel(pm, model);
        if (key >= 0 && table_body >= 0 && tt_gid >= 0) std::fprintf(stderr,
            "[bench-model-after] plant_face=%.4f planner_face=%.4f "
            "plant_brace_z=%.4f planner_brace_z=%.4f\n",
            face(model), face(pm), model->key_qpos[key * model->nq + 2],
            pm->key_qpos[key * pm->nq + 2]);
      }
      if (!model_out.empty()) {
        mj_saveModel(model, model_out.c_str(), nullptr, 0);
        mj_saveModel(pm, (model_out + ".planner").c_str(), nullptr, 0);
      }
      // plant-only mismatch, applied AFTER the planner copy so only the plant changes
      if (plant_kp_scale != 1.0) {
        for (int k = 0; k < model->nu; k++) {
          model->actuator_gainprm[k * mjNGAIN + 0] *= plant_kp_scale;
          model->actuator_biasprm[k * mjNBIAS + 1] *= plant_kp_scale;
        }
        std::fprintf(stderr, "[bench] plant kp x %g (planner unchanged)\n", plant_kp_scale);
      }
      if (plant_friction_scale != 1.0) {
        for (int g = 0; g < model->ngeom; g++) model->geom_friction[3 * g + 0] *= plant_friction_scale;
        std::fprintf(stderr, "[bench] plant sliding friction x %g (planner unchanged)\n", plant_friction_scale);
      }
      if (plant_table_mu > 0.0) SetTableMu(model, plant_table_mu);
      if (plant_table_dz != 0.0 && table_body >= 0) {
        model->body_pos[3 * table_body + 2] += plant_table_dz;
        std::fprintf(stderr, "[bench] plant slab %+.3f m relative to the planner's\n", plant_table_dz);
      }
      if (plant_table_stiff) {
        SetTableStiff(model);
        std::fprintf(stderr, "[bench] plant slab contact stiff: solref 0.01, solimp 0.95 0.99 0.001 "
                             "(planner unchanged)\n");
      }
      if (plant_foot_mu > 0.0) SetFootMu(model, plant_foot_mu);
      if (planner_table_mu > 0.0) SetTableMu(pm, planner_table_mu);
      if (planner_foot_mu > 0.0) SetFootMu(pm, planner_foot_mu);
      if (plant_table_mu > 0.0 || plant_foot_mu > 0.0 || planner_table_mu > 0.0 ||
          planner_foot_mu > 0.0) {
        // echo what the contact set will actually use, read back from the models
        auto pair_mu = [](const mjModel* m, const char* name) {
          int p = -1;
          for (int k = 0; k < m->npair; k++) {
            const char* pn = mj_id2name(m, mjOBJ_PAIR, k);
            if (pn && std::strcmp(pn, name) == 0) p = k;
          }
          return p >= 0 ? m->pair_friction[5 * p] : std::nan("");
        };
        auto geom_mu = [](const mjModel* m, const char* name) {
          int g = mj_name2id(m, mjOBJ_GEOM, name);
          return g >= 0 ? m->geom_friction[3 * g] : std::nan("");
        };
        auto foot_mu = [](const mjModel* m) {
          int b = mj_name2id(m, mjOBJ_BODY, "left_ankle_roll_link");
          for (int g = 0; g < m->ngeom; g++)
            if (m->geom_bodyid[g] == b && (m->geom_contype[g] || m->geom_conaffinity[g]))
              return m->geom_friction[3 * g];
          return std::nan("");
        };
        std::fprintf(stderr,
                     "[bench-friction] plant: forearm_pair %.3f wrist_pair %.3f table_geom %.3f "
                     "floor %.3f foot %.3f | planner: forearm_pair %.3f wrist_pair %.3f "
                     "table_geom %.3f floor %.3f foot %.3f\n",
                     pair_mu(model, "left_forearm_brace_pair"), pair_mu(model, "left_wrist_brace_pair"),
                     geom_mu(model, "table_top_collision"), geom_mu(model, "floor"), foot_mu(model),
                     pair_mu(pm, "left_forearm_brace_pair"), pair_mu(pm, "left_wrist_brace_pair"),
                     geom_mu(pm, "table_top_collision"), geom_mu(pm, "floor"), foot_mu(pm));
      }
      if (plant_mass_scale != 1.0) {
        for (int b = 1; b < model->nbody; b++) {
          model->body_mass[b] *= plant_mass_scale;
          for (int k = 0; k < 3; k++) model->body_inertia[3 * b + k] *= plant_mass_scale;
        }
        // ★ 2026-09-14 mj_setConst evaluates at qpos0 and LEAVES data->qpos there
        // (measured: base x 0.189 -> 0.000, the seed perturbation and the object
        // pose wiped). Every --plant_mass_scale run before this fix started the
        // plant 19 cm further from the table, unperturbed. Use a scratch data.
        SetConstScratch(model);
        std::fprintf(stderr, "[bench] plant mass x %g (planner unchanged)\n", plant_mass_scale);
      }
    }
    agent.state.Set(model, data);
    if (latency_steps > 0) {
      Snap& sn = snaps[i % snaps.size()];
      mju_copy(sn.qpos.data(), data->qpos, model->nq);
      mju_copy(sn.qvel.data(), data->qvel, model->nv);
      if (model->na) mju_copy(sn.act.data(), data->act, model->na);
      if (model->nmocap) {
        mju_copy(sn.mocap_pos.data(), data->mocap_pos, 3 * model->nmocap);
        mju_copy(sn.mocap_quat.data(), data->mocap_quat, 4 * model->nmocap);
      }
      if (model->nuserdata) mju_copy(sn.userdata.data(), data->userdata, model->nuserdata);
      sn.time = data->time;
    }
    agent.ActivePlanner().ActionFromPolicy(data->ctrl, agent.state.state().data(),
                                           agent.state.time(), /*use_previous=*/false);
    if (assist_fx != 0.0 && assist_bid >= 0) {
      const int ph_now = lean_task ? lean_task->BenchPhaseIndex() : -1;
      bool on = ph_now >= assist_from_phase && ph_now <= assist_until_phase;
      if (on && assist_gap > 0.0 && tt_gid >= 0 && pad_gid >= 0) {
        const double fz = data->geom_xpos[3 * tt_gid + 2] + model->geom_size[3 * tt_gid + 2];
        const double gap = data->geom_xpos[3 * pad_gid + 2] - model->geom_size[3 * pad_gid] - fz;
        on = gap < assist_gap;
      }
      if (on && assist_t0 < 0.0) assist_t0 = data->time;
      if (!on) assist_t0 = -1.0;
      const double a = (on && assist_ramp > 1e-9)
                           ? mju_min(1.0, (data->time - assist_t0) / assist_ramp)
                           : (on ? 1.0 : 0.0);
      assist_now = a * assist_fx;
      mju_zero(data->xfrc_applied + 6 * assist_bid, 6);
      data->xfrc_applied[6 * assist_bid + 0] = assist_now;
    }
    if (fs && i % log_every == 0) {
      // Capture BEFORE integration: qpos, qvel, control and warm-start all refer
      // to precisely this time. Offline mj_forward reconstructs its contacts.
      std::fprintf(fs, "%.9g,%d", data->time,
                   lean_task ? lean_task->BenchPhaseIndex() : 0);
      for (int k = 0; k < model->nq; ++k) std::fprintf(fs, ",%.17g", data->qpos[k]);
      for (int k = 0; k < model->nv; ++k) std::fprintf(fs, ",%.17g", data->qvel[k]);
      for (int k = 0; k < model->nu; ++k) std::fprintf(fs, ",%.17g", data->ctrl[k]);
      for (int k = 0; k < model->nv; ++k) std::fprintf(fs, ",%.17g", data->qacc_warmstart[k]);
      std::fprintf(fs, "\n");
    }
    mj_step(model, data);
    if (fc && i % contact_every == 0) {
      // mj_step leaves contacts, efc forces and kinematics from the forward pass
      // at the pre-step state; only qpos/qvel/time have advanced.
      // 0 brace (left arm vs table), 1 left foot, 2 right foot, 3 rest of the
      // robot vs table, 4 the brace pads alone (forearm + wrist pad pairs; the
      // jaw of the left gripper rests on the slab through stand-up and is in 0)
      ContactGroup grp[5];
      mjtNum f6[6];
      for (int c = 0; c < data->ncon; c++) {
        const mjContact& con = data->contact[c];
        if (con.efc_address < 0) continue;              // excluded / margin-only
        const int b0 = model->geom_bodyid[con.geom[0]], b1 = model->geom_bodyid[con.geom[1]];
        int k = -1, robot_side = -1;
        if ((b0 == table_body) != (b1 == table_body)) {
          const int other = (b0 == table_body) ? b1 : b0;
          if (other == 0 || other == object_body) continue;
          const char* bn = mj_id2name(model, mjOBJ_BODY, other);
          k = (bn && std::strncmp(bn, "left_", 5) == 0) ? 0 : 3;
          robot_side = (b0 == table_body) ? 1 : 0;
        } else if (con.geom[0] == floor_gid || con.geom[1] == floor_gid) {
          const int other = (con.geom[0] == floor_gid) ? b1 : b0;
          k = other == lfoot_body ? 1 : other == rfoot_body ? 2 : -1;
          robot_side = (con.geom[0] == floor_gid) ? 1 : 0;
        }
        if (k < 0) continue;
        mj_contactForce(model, data, c, f6);
        const mjtNum* fr = con.frame;                   // rows: normal, tangent1, tangent2
        mjtNum fw[3];                                   // world force on geom[1]
        for (int j = 0; j < 3; j++) fw[j] = fr[j] * f6[0] + fr[3 + j] * f6[1] + fr[6 + j] * f6[2];
        const double sgn = robot_side == 1 ? 1.0 : -1.0;
        const int rg = con.geom[robot_side];
        const bool is_pad = k == 0 && (rg == pad_gid || rg == wpad_gid);
        for (int kk : {k, is_pad ? 4 : -1}) {
          if (kk < 0) continue;
          for (int j = 0; j < 3; j++) grp[kk].F[j] += sgn * fw[j];
          grp[kk].fn += f6[0];
          grp[kk].n++;
        }
        if (k == 3) continue;
        ContactGroup& gr = grp[k];
        ContactGroup* gp = is_pad ? &grp[4] : nullptr;
        if (f6[0] > 2.0) {
          double u = std::fabs(f6[1]) / con.friction[0] + std::fabs(f6[2]) / con.friction[1];
          if (con.dim >= 4 && con.friction[2] > 0.0) u += std::fabs(f6[3]) / con.friction[2];
          gr.util = std::max(gr.util, u / f6[0]);
          if (gp) gp->util = std::max(gp->util, u / f6[0]);
        }
        // tangential slip of the robot's material point against the other body
        mjtNum vr[3], vo[3] = {0, 0, 0}, vrel[3];
        const int rb = robot_side == 1 ? b1 : b0, ob = robot_side == 1 ? b0 : b1;
        PointVelocity(model, data, rb, con.pos, vr);
        if (ob != 0 && model->body_weldid[ob] != 0) PointVelocity(model, data, ob, con.pos, vo);
        mju_sub3(vrel, vr, vo);
        const double vn = mju_dot3(vrel, fr);
        mjtNum vt[3] = {vrel[0] - vn * fr[0], vrel[1] - vn * fr[1], vrel[2] - vn * fr[2]};
        gr.slip_fw += std::max(0.0, (double)f6[0]) * mju_norm3(vt);
        if (gp) gp->slip_fw += std::max(0.0, (double)f6[0]) * mju_norm3(vt);
      }
      std::fprintf(fc, "%.4f,%d", data->time - model->opt.timestep,
                   lean_task ? lean_task->BenchPhaseIndex() : 0);
      for (int k = 0; k < 3; k++)
        std::fprintf(fc, ",%.2f,%.2f,%.2f,%.2f,%.3f,%.4f,%d", grp[k].fn, grp[k].F[0], grp[k].F[1],
                     grp[k].F[2], grp[k].util, grp[k].slip(), grp[k].n);
      std::fprintf(fc, ",%.2f,%.2f,%.2f,%.2f", grp[3].fn, grp[3].F[0], grp[3].F[1], grp[3].F[2]);
      std::fprintf(fc, ",%.2f,%.2f,%.2f,%.2f,%.3f,%.4f,%d", grp[4].fn, grp[4].F[0], grp[4].F[1],
                   grp[4].F[2], grp[4].util, grp[4].slip(), grp[4].n);
      mjtNum padv[6] = {0, 0, 0, 0, 0, 0};
      const mjtNum* pp = pad_gid >= 0 ? data->geom_xpos + 3 * pad_gid : padv;
      if (pad_gid >= 0) mj_objectVelocity(model, data, mjOBJ_GEOM, pad_gid, padv, 0);
      const mjtNum* wp = wpad_gid >= 0 ? data->geom_xpos + 3 * wpad_gid : padv;
      mjtNum sL[3] = {0, 0, 0}, sR[3] = {0, 0, 0};
      if (lfoot_body >= 0) {
        mju_mulMatVec3(sL, data->xmat + 9 * lfoot_body, kSoleLocal);
        mju_addTo3(sL, data->xpos + 3 * lfoot_body);
      }
      if (rfoot_body >= 0) {
        mju_mulMatVec3(sR, data->xmat + 9 * rfoot_body, kSoleLocal);
        mju_addTo3(sR, data->xpos + 3 * rfoot_body);
      }
      const mjtNum* com = pelvis_body >= 0 ? data->subtree_com + 3 * pelvis_body : sL;
      std::fprintf(fc, ",%.5f,%.5f,%.5f,%.4f,%.4f,%.4f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,"
                       "%.5f,%.5f,%.5f,%.5f,%.1f\n",
                   pp[0], pp[1], pp[2], padv[3], padv[4], padv[5], wp[0], wp[1], wp[2],
                   sL[0], sL[1], sR[0], sR[1], com[0], com[1], com[2], data->qpos[2], assist_now);
    }
    if (i % spp == 0) {
      if (latency_steps > 0 && i >= latency_steps) {
        // plan from the snapshot taken latency_steps plant steps ago
        const Snap& sn = snaps[(i - latency_steps) % snaps.size()];
        if (!latency_compensate) {
          agent.state.Set(model, sn.qpos.data(), sn.qvel.data(), sn.act.data(),
                          sn.mocap_pos.data(), sn.mocap_quat.data(), sn.userdata.data(), sn.time);
        } else {
          // predict the delayed state forward to now under the current policy
          static mjData* pred = nullptr;
          if (!pred) pred = mj_makeData(model);
          mju_copy(pred->qpos, sn.qpos.data(), model->nq);
          mju_copy(pred->qvel, sn.qvel.data(), model->nv);
          if (model->na) mju_copy(pred->act, sn.act.data(), model->na);
          if (model->nmocap) {
            mju_copy(pred->mocap_pos, sn.mocap_pos.data(), 3 * model->nmocap);
            mju_copy(pred->mocap_quat, sn.mocap_quat.data(), 4 * model->nmocap);
          }
          if (model->nuserdata) mju_copy(pred->userdata, sn.userdata.data(), model->nuserdata);
          pred->time = sn.time;
          std::vector<double> ps(model->nq + model->nv + model->na);
          for (int k = 0; k < latency_steps; k++) {
            mju_copy(ps.data(), pred->qpos, model->nq);
            mju_copy(ps.data() + model->nq, pred->qvel, model->nv);
            if (model->na) mju_copy(ps.data() + model->nq + model->nv, pred->act, model->na);
            agent.ActivePlanner().ActionFromPolicy(pred->ctrl, ps.data(), pred->time, false);
            mj_step(model, pred);
          }
          agent.state.Set(model, pred);
        }
      }
      // sample dump, part 1: the std each parameter will be SAMPLED at this
      // tick is max(sqrt(variance), std_min) with the variance the LAST refit
      // left, so it has to be read before the iteration overwrites it.
      const bool dump_now = cem && !samples_out.empty() &&
                            samples_next < samples_at.size() &&
                            data->time >= samples_at[samples_next] - 1e-9;
      std::vector<double> sigma_used;
      double std_min_used = 0.0;
      if (dump_now) {
        const double live = g_task->LiveStdMinOverride();
        std_min_used = live > 0.0 ? live : cem->std_min_;
        sigma_used.resize(cem->variance.size());
        for (size_t k = 0; k < sigma_used.size(); k++)
          sigma_used[k] = cem->variance_fixed_ > 0.0
                              ? cem->variance_fixed_
                              : std::max(std::sqrt(cem->variance[k]), std_min_used);
      }
      agent.PlanIteration(&pool);
      if (dump_now) {
        const mjModel* pm = cem->model;
        const int nu = pm->nu, nq = pm->nq;
        const int ns = cem->nominal_trajectory.dim_state;
        const int H = cem->nominal_trajectory.horizon;
        const int N = cem->num_trajectory_;
        const int ne = std::min(cem->n_elite_, N);
        const int nk = cem->resampled_policy.num_spline_points;
        char dname[64];
        std::snprintf(dname, sizeof dname, "tick%02d_t%.3f", (int)samples_next, data->time);
        const std::filesystem::path dir = std::filesystem::path(samples_out) / dname;
        std::filesystem::create_directories(dir);
        auto openf = [&](const char* name) {
          FILE* f = std::fopen((dir / name).c_str(), "w");
          if (!f) { std::fprintf(stderr, "[bench] samples: cannot write %s\n", name); }
          return f;
        };
        auto knots = [&](FILE* f, const mjpc::SamplingPolicy& pol, int k) {
          for (int t = 0; t < pol.plan.Size(); t++) {
            mjpc::spline::TimeSpline::ConstNode n = pol.plan.NodeAt(t);
            if (k >= 0) std::fprintf(f, "%d,", k);
            std::fprintf(f, "%d,%.6f", t, n.time());
            for (int j = 0; j < nu; j++) std::fprintf(f, ",%.6f", n.values()[j]);
            std::fprintf(f, "\n");
          }
        };
        auto traj = [&](FILE* f, const mjpc::Trajectory& tr, int k) {
          for (int t = 0; t < tr.horizon; t++) {
            const double* st = tr.states.data() + t * ns;
            std::fprintf(f, "%d,%d,%.4f,%.6f", k, t, tr.times[t], tr.costs[t]);
            for (int j = 0; j < nq; j++) std::fprintf(f, ",%.6f", st[j]);
            std::fprintf(f, "\n");
          }
        };
        auto header_u = [&](FILE* f, const char* lead) {
          std::fprintf(f, "%s", lead);
          for (int j = 0; j < nu; j++) std::fprintf(f, ",u%d", j);
          std::fprintf(f, "\n");
        };
        // centre: the resampled nominal spline the noise was added to
        if (FILE* f = openf("center.csv")) { header_u(f, "knot,t"); knots(f, cem->resampled_policy, -1); std::fclose(f); }
        // spread: std used this tick, std the refit just produced, and the noise drawn
        if (FILE* f = openf("sigma.csv")) {
          std::fprintf(f, "knot,j,sigma_used,sigma_next\n");
          for (int t = 0; t < nk; t++)
            for (int j = 0; j < nu; j++)
              std::fprintf(f, "%d,%d,%.6f,%.6f\n", t, j, sigma_used[t * nu + j], std::sqrt(cem->variance[t * nu + j]));
          std::fclose(f);
        }
        if (FILE* f = openf("noise.csv")) {
          header_u(f, "k,knot");
          for (int k = 0; k < N; k++) {
            const int shift = k * (nu * mjpc::kMaxTrajectoryHorizon);
            for (int t = 0; t < nk; t++) {
              std::fprintf(f, "%d,%d", k, t);
              for (int j = 0; j < nu; j++) std::fprintf(f, ",%.6f", cem->noise[shift + t * nu + j]);
              std::fprintf(f, "\n");
            }
          }
          std::fclose(f);
        }
        if (FILE* f = openf("candidates.csv")) { header_u(f, "k,knot,t"); for (int k = 0; k < N; k++) knots(f, cem->candidate_policy[k], k); std::fclose(f); }
        // scores + elite order
        if (FILE* f = openf("scores.csv")) {
          std::fprintf(f, "k,rank,elite,total_return\n");
          std::vector<int> rank(N, -1);
          for (int r = 0; r < N; r++) rank[cem->trajectory_order[r]] = r;
          for (int k = 0; k < N; k++)
            std::fprintf(f, "%d,%d,%d,%.6f\n", k, rank[k], (int)(rank[k] < ne), cem->trajectory[k].total_return);
          std::fclose(f);
        }
        // rollouts: k = -1 nominal (centre), k = -2 the fold, 0..N-1 the candidates
        if (FILE* f = openf("rollouts.csv")) {
          std::fprintf(f, "k,step,t,cost");
          for (int j = 0; j < nq; j++) std::fprintf(f, ",q%d", j);
          std::fprintf(f, "\n");
          traj(f, cem->nominal_trajectory, -1);
          for (int k = 0; k < N; k++) traj(f, cem->trajectory[k], k);
          // fold: the elite mean, rolled out from the same state on a scratch
          // trajectory so nothing the planner holds changes
          mjpc::Trajectory ft;
          ft.Initialize(ns, nu, g_task->num_residual, g_task->num_trace, mjpc::kMaxTrajectoryHorizon);
          ft.Allocate(mjpc::kMaxTrajectoryHorizon);
          auto fold_policy = [cem](double* action, const double* state, double time) {
            cem->policy.Action(action, state, time);
          };
          ft.Rollout(fold_policy, cem->task, pm, cem->data_[0].get(), cem->state.data(),
                     cem->time, cem->mocap.data(), cem->userdata.data(), H);
          traj(f, ft, -2);
          std::fclose(f);
          if (FILE* g = openf("fold.csv")) {
            header_u(g, "knot,t"); knots(g, cem->policy, -1); std::fclose(g);
            std::fprintf(stderr, "[bench] samples %s: N=%d elites=%d H=%d J(nominal)=%.4f J(best)=%.4f J(fold)=%.4f std_min=%.4f\n",
                         dname, N, ne, H, cem->nominal_trajectory.total_return,
                         cem->trajectory[cem->trajectory_order[0]].total_return, ft.total_return, std_min_used);
          }
        }
        if (FILE* f = openf("meta.json")) {
          std::fprintf(f, "{\"t_plan\": %.4f, \"t_state\": %.4f, \"phase\": %d, \"phase_name\": \"%s\", "
                          "\"n\": %d, \"n_elite\": %d, \"horizon_steps\": %d, \"dt\": %.4f, "
                          "\"nu\": %d, \"nq\": %d, \"knots\": %d, \"std_min_used\": %.5f, "
                          "\"std_min\": %.5f, \"spp\": %d, \"seed\": %d, \"strategy\": %d, \"gains\": \"%s\"}\n",
                       data->time, cem->time, lean_task ? lean_task->BenchPhaseIndex() : -1,
                       lean_task ? lean_task->BenchPhaseName().c_str() : "", N, ne, H, pm->opt.timestep,
                       nu, nq, nk, std_min_used, cem->std_min_, spp, seed, strategy, gains.c_str());
          std::fclose(f);
        }
        samples_next++;
      }
      if (fp && lean_task && lean_task->BenchPhaseIndex() >= plan_out_from_phase) {
        const mjpc::Trajectory* best = agent.ActivePlanner().BestTrajectory();
        const mjModel* pm = agent.GetModel();
        const int ns = pm->nq + pm->nv + pm->na;
        if (best && best->horizon > 0 && (int)best->states.size() >= best->horizon * ns) {
          const int ph = lean_task->BenchPhaseIndex();
          for (int k = 0; k < best->horizon; k += plan_out_stride) {
            const double* st = best->states.data() + k * ns;
            const double* u = (k < best->horizon - 1 && (int)best->actions.size() >= (k + 1) * pm->nu)
                                  ? best->actions.data() + k * pm->nu : nullptr;
            const double tk = ((int)best->times.size() > k) ? best->times[k] : data->time;
            double ob[4], ot[4];
            eval_state(pm, pd, st, st + pm->nq, u, tk, ob);
            eval_state(model, td, st, st + model->nq, u, tk, ot);
            const double ck = ((int)best->costs.size() > k) ? best->costs[k] : std::nan("");
            const double rhx = (rgrip_body >= 0) ? pd->xpos[3 * rgrip_body] : std::nan("");
            std::fprintf(fp, "%.4f,%d,%d,%.4f,%.5f,%.5f,%.5f,%.5f,"
                             "%.5f,%.5f,%.5f,%.3f,%.5f,%.5f,%.5f,%.3f\n",
                         data->time, ph, k, tk, ck, st[0], st[2], rhx,
                         ob[0], ob[1], ob[2], ob[3], ot[0], ot[1], ot[2], ot[3]);
          }
        }
      }
    }

    // The strategy JSON loads on the FIRST Transition, so the phase count is 0
    // until then -- re-read it every step rather than latching a stale 0.
    n_phases = lean_task ? lean_task->BenchPhaseCount() : 1;
    const int phase = lean_task ? lean_task->BenchPhaseIndex() : 0;
    if (phase != last_phase) {
      if (phase >= 0 && phase < 64 && phase_enter[phase] < 0)
        phase_enter[phase] = data->time;
      std::fprintf(stderr, "[bench] t=%7.2f  phase %d -> %d (%s)\n", data->time,
                   last_phase, phase,
                   lean_task ? lean_task->BenchPhaseName().c_str() : "?");
      last_phase = phase;
      if (phase > max_phase_seen) max_phase_seen = phase;
      if (n_phases > 0 && phase == n_phases - 1) final_phase_since = data->time;
      // payload attach: first entry into payload_phase (a later reset to 0 does
      // not re-arm it -- the mass stays on the hand, as it would on the robot)
      if (phase == payload_phase && payload_t0 < 0.0) {
        payload_t0 = data->time;
        mjModel* pm = agent.GetModel();
        if (payload_belief != 0.0 && !payload_belief_done) {
          pm->body_mass[payload_bid] += payload_belief;
          SetConstScratch(pm);
          payload_belief_done = true;
        }
        if (bft_add != 0.0) {
          model->numeric_data[model->numeric_adr[bft_add_nid]] = bft_add;
          pm->numeric_data[pm->numeric_adr[bft_add_nid]] = bft_add;
          bft_add_on = true;
        }
        if (icemdr && !scenario_masses.empty()) {
          icemdr->scenario_mass_ = scenario_masses;
          std::fprintf(stderr, "[bench] t=%7.2f  SCENARIO set: %s kg on %s, agg %s\n",
                       data->time, scenario_masses_s.c_str(), payload_body.c_str(),
                       scenario_agg_s.c_str());
        }
        std::fprintf(stderr, "[bench] t=%7.2f  PAYLOAD attach: plant %.2f kg over %.2f s, "
                             "planner %.2f kg, bft_add %.1f N\n", data->time, payload_true,
                     payload_ramp, payload_belief, bft_add);
      }
      if (bft_add_on && !bft_add_cleared && phase > bft_add_until) {
        mjModel* pm = agent.GetModel();
        model->numeric_data[model->numeric_adr[bft_add_nid]] = 0.0;
        pm->numeric_data[pm->numeric_adr[bft_add_nid]] = 0.0;
        bft_add_cleared = true;
        std::fprintf(stderr, "[bench] t=%7.2f  bft_add cleared (phase %d)\n", data->time, phase);
      }
    }
    // plant payload ramp (the object leaving the table), re-evaluated every
    // 10 ms until it is fully on; mj_setConst through a scratch data each time
    if (payload_t0 >= 0.0 && payload_true != 0.0 && payload_applied != payload_true &&
        i % 5 == 0) {
      double a = payload_ramp > 0.0 ? mju_min(1.0, (data->time - payload_t0) / payload_ramp) : 1.0;
      double target = a * payload_true;
      if (target != payload_applied) {
        model->body_mass[payload_bid] = payload_base_mass + target;
        SetConstScratch(model);
        payload_applied = target;
      }
    }
    if (payload_t0 >= 0.0) {
      // carry-phase extremes (from attach to the end), for the summary line
      double brace_now = 0.0;
      mjtNum ftc[6];
      for (int c = 0; c < data->ncon; c++) {
        const mjContact& con = data->contact[c];
        int b1 = model->geom_bodyid[con.geom[0]], b2 = model->geom_bodyid[con.geom[1]];
        bool t1 = (b1 == table_body), t2 = (b2 == table_body);
        if (t1 == t2) continue;
        int other = t1 ? b2 : b1;
        if (other == 0 || other == object_body) continue;
        mj_contactForce(model, data, c, ftc);
        brace_now += mju_max(0.0, ftc[0]);
      }
      peak_brace_carry = mju_max(peak_brace_carry, brace_now);
      min_pelvis_carry = mju_min(min_pelvis_carry, data->qpos[2]);
      if (torso_id >= 0) {
        const double* R = data->xmat + 9 * torso_id;
        double tilt = std::acos(mju_max(-1.0, mju_min(1.0, R[8]))) * 180.0 / mjPI;
        max_tilt_carry = mju_max(max_tilt_carry, tilt);
      }
    }

    // Fall: the pelvis dropping below half its standing height is unambiguous and
    // needs no task-specific threshold tuning.
    if (data->qpos[2] < 0.5) { fell = true; break; }
    // Completed: reached the terminal phase and held it. The terminal phase carries
    // sustain 9999 so it never self-advances; the bench decides when it is done.
    if (final_phase_since > 0 && data->time - final_phase_since >= hold_after_final) {
      t_complete = data->time;
      break;
    }

    if (i % log_every == 0) {
      double f[kNBrace] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
      double f_total = 0.0;
      double brace_normal = 0.0, trunk_normal = 0.0;                           // ROBOT-on-table only
      mjtNum ft[6];
      for (int c = 0; c < data->ncon; c++) {
        const mjContact& con = data->contact[c];
        int b1 = model->geom_bodyid[con.geom[0]], b2 = model->geom_bodyid[con.geom[1]];
        bool t1 = (b1 == table_body), t2 = (b2 == table_body);
        if (t1 == t2) continue;                       // not a body-vs-table contact
        int other = t1 ? b2 : b1;
        // ★ 2026-09-04 SKIP THE WORLD. The table's four legs stand ON THE FLOOR,
        // and floor geoms belong to body 0, so a naive "one side is the table"
        // test books the TABLE'S OWN WEIGHT as table contact load -- measured
        // 166 N at h=0.785 with the arm still 228 mm above the face and every
        // named body reading zero. Anything derived from that total (in
        // particular an "unattributed load" residual meant to catch the wrist
        // pad) would have been pure table weight.
        if (other == 0) continue;
        // The free object rests on the slab too; it is cargo, not a brace.
        if (other == object_body) continue;
        mj_contactForce(model, data, c, ft);
        double mag = std::sqrt(ft[0] * ft[0] + ft[1] * ft[1] + ft[2] * ft[2]);
        f_total += mag;
        const char* bn = mj_id2name(model, mjOBJ_BODY, other);
        const std::string body_name = bn ? bn : "";
        if (body_name.rfind("left_", 0) == 0 &&
            (body_name.find("shoulder") != std::string::npos ||
             body_name.find("elbow") != std::string::npos ||
             body_name.find("wrist") != std::string::npos ||
             body_name.find("gripper") != std::string::npos ||
             body_name.find("magpie") != std::string::npos))
          brace_normal += std::max(0.0, ft[0]);
        if (body_name == "torso_link" || body_name == "pelvis")
          trunk_normal += std::max(0.0, ft[0]);
        for (int k = 0; k < kNBrace; k++) if (other == brace_id[k]) f[k] += mag;
      }
      // Load the named bodies did not account for: any OTHER robot link that is
      // leaning on the slab (a thigh on the edge, a knee under it).
      double f_named = 0.0;
      for (int k = 0; k < kNBrace; k++) f_named += f[k];
      const double f_other = std::max(0.0, f_total - f_named);
      // Reach channel, computed here rather than read from ComputeMetrics:
      // that metric's reach block is gated on `kf.name == "reach_to_target"`,
      // which strategy 25's targeting rung is NOT called, so it is nan for the
      // whole braced ladder. Right gripper vs the `target` mocap, both straight
      // out of the model.
      double rhand[3] = {0, 0, 0}, tgtp[3] = {0, 0, 0};
      if (rgrip_body >= 0) mju_copy3(rhand, data->xpos + 3 * rgrip_body);
      if (target_mocap >= 0) mju_copy3(tgtp, data->mocap_pos + 3 * target_mocap);
      double tilt = 0.0;
      if (torso_id >= 0) {
        const mjtNum* R = data->xmat + 9 * torso_id;   // body z-axis = column 2
        double uz = std::max(-1.0, std::min(1.0, R[8]));
        tilt = std::acos(uz) * 180.0 / M_PI;
      }
      // Slab face and the pad's clearance above it. Both derived from the geoms,
      // so they stay right whatever `Table H` did to the body.
      double pad_clear = std::nan("");
      if (tt_gid >= 0) {
        face_z = data->geom_xpos[3 * tt_gid + 2] + model->geom_size[3 * tt_gid + 2];
        if (pad_gid >= 0) {
          double pad_r = model->geom_size[3 * pad_gid + 0];
          pad_clear = data->geom_xpos[3 * pad_gid + 2] - pad_r - face_z;
        }
      }
      std::fprintf(fo, "%.4f,%d,%s,%.4f,%.3f,%.4f,%.4f,"
                       "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,"
                       "%.2f,%.2f,%.6f,"
                       "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
                   data->time, last_phase,
                   lean_task ? lean_task->BenchPhaseName().c_str() : "?",
                   data->qpos[2], tilt, face_z, pad_clear,
                   f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8],
                   f_other, f_total,
                   g_task->CostValue(data->sensordata),
                   rhand[0], rhand[1], rhand[2], tgtp[0], tgtp[1], tgtp[2]);
      // Match lean.cc kGripperTipLocal, including its lateral/vertical offsets.
      mjtNum jaw[3] = {0, 0, 0}, jaw_local[3] = {0.2254, -0.0118, -0.1062};
      if (rgrip_body >= 0) {
        mju_mulMatVec3(jaw, data->xmat + 9 * rgrip_body, jaw_local);
        mju_addTo3(jaw, data->xpos + 3 * rgrip_body);
      }
      std::fprintf(fo, ",%.6f,%.6f,%.6f,%.3f,%.3f", jaw[0], jaw[1], jaw[2],
                   brace_normal, trunk_normal);
      {
        const mjpc::Trajectory* best = agent.ActivePlanner().BestTrajectory();
        double plan_return = best ? best->total_return : std::nan("");
        double ess = std::nan(""), spread = std::nan("");
        if (mppi) { ess = mppi->last_ess_; spread = mppi->last_spread_; }
        double cem_std = std::nan("");
        const std::vector<double>* var = nullptr;
        int npar = 0;
        if (cem) { var = &cem->variance; npar = cem->policy.num_spline_points * model->nu; }
        if (icem) { var = &icem->variance; npar = icem->policy.num_spline_points * model->nu; }
        if (var && npar > 0) {
          double acc = 0.0;
          for (int k = 0; k < npar; k++) acc += std::sqrt(std::max(0.0, (*var)[k]));
          cem_std = acc / npar;
        }
        std::fprintf(fo, ",%.6f,%.3f,%.6f,%.6f", plan_return, ess, spread, cem_std);
      }
      {
        double ob[4];
        eval_state(agent.GetModel(), pd, data->qpos, data->qvel, data->ctrl, data->time, ob);
        std::fprintf(fo, ",%.5f,%.5f,%.3f", ob[0], ob[1], ob[3]);
      }
      metrics.clear();
      g_task->ComputeMetrics(model, data, &metrics, &phase_name_m);
      for (int k = 0; k < kNMetric; k++) {
        auto it = metrics.find(kMetricKeys[k]);
        if (it == metrics.end()) std::fprintf(fo, ",nan");
        else std::fprintf(fo, ",%.5f", it->second);
      }
      std::fprintf(fo, "\n");
    }
    if (fq && i % vid_every == 0) {
      std::fprintf(fq, "%.4f", data->time);
      for (int k = 0; k < model->nq; k++) std::fprintf(fq, ",%.6f", data->qpos[k]);
      std::fprintf(fq, "\n");
    }
  }

  double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
  // One machine-readable summary line on stderr; the sweep driver parses this.
  std::fprintf(stderr,
               "[bench-summary] payload_true=%.3f payload_belief=%.3f bft_add=%.1f payload_t=%.2f "
               "max_phase=%d peak_brace_carry=%.1f min_pelvis_carry=%.3f max_tilt_carry=%.1f "
               "assist_fx=%.1f task=%s strategy=%d planner=%d table_h=%.4f face_z=%.4f seed=%d "
               "fell=%d complete=%d t_complete=%.3f t_end=%.3f phases=%d wall_s=%.1f enter=",
               payload_true, payload_belief, bft_add, payload_t0, max_phase_seen,
               peak_brace_carry, (payload_t0 >= 0.0 ? min_pelvis_carry : -1.0), max_tilt_carry,
               assist_fx, task_name.c_str(), strategy, agent.PlannerId(), table_h, face_z, seed, (int)fell,
               (int)(t_complete > 0), t_complete, data->time, n_phases, wall);
  for (int p = 0; p < n_phases && p < 64; p++)
    std::fprintf(stderr, "%s%.2f", p ? ":" : "", phase_enter[p]);
  std::fprintf(stderr, "\n");

  if (fo != stdout) std::fclose(fo);
  if (fq) std::fclose(fq);
  if (fs) std::fclose(fs);
  if (fp) std::fclose(fp);
  if (fc) std::fclose(fc);
  mj_deleteData(pd);
  mj_deleteData(td);
  mj_deleteData(data);
  mjcb_sensor = nullptr;
  return fell ? 1 : 0;
}
