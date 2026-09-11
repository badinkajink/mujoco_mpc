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
#include <map>
#include <string>
#include <vector>
#include <utility>

#include <mujoco/mujoco.h>

#include "mjpc/agent.h"
#include "mjpc/planners/cross_entropy/planner.h"
#include "mjpc/planners/icem/planner.h"
#include "mjpc/planners/mppi/planner.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"
#include "mjpc/tasks/tasks.h"
#include "mjpc/tasks/humanoid_bench/lean/lean.h"

namespace {
mjpc::Task* g_task = nullptr;
void residual_callback(const mjModel* model, mjData* data, int stage) {
  if (stage == mjSTAGE_ACC) g_task->Residual(model, data, data->sensordata);
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
  if ((!state_out.empty() && !fs) || !fo || (!qpos_out.empty() && !fq)) {
    std::fprintf(stderr, "[bench] failed to open requested output file\n");
    return 2;
  }
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
  bool fell = false;
  double face_z = 0.0;
  std::vector<double> phase_enter(64, -1.0);
  std::map<std::string, double> metrics;
  std::string phase_name_m;
  auto wall0 = std::chrono::steady_clock::now();

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
    }
    agent.state.Set(model, data);
    agent.ActivePlanner().ActionFromPolicy(data->ctrl, agent.state.state().data(),
                                           agent.state.time(), /*use_previous=*/false);
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
    if (i % spp == 0) agent.PlanIteration(&pool);

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
      if (n_phases > 0 && phase == n_phases - 1) final_phase_since = data->time;
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
               "[bench-summary] task=%s strategy=%d planner=%d table_h=%.4f face_z=%.4f seed=%d "
               "fell=%d complete=%d t_complete=%.3f t_end=%.3f phases=%d wall_s=%.1f enter=",
               task_name.c_str(), strategy, agent.PlannerId(), table_h, face_z, seed, (int)fell,
               (int)(t_complete > 0), t_complete, data->time, n_phases, wall);
  for (int p = 0; p < n_phases && p < 64; p++)
    std::fprintf(stderr, "%s%.2f", p ? ":" : "", phase_enter[p]);
  std::fprintf(stderr, "\n");

  if (fo != stdout) std::fclose(fo);
  if (fq) std::fclose(fq);
  if (fs) std::fclose(fs);
  mj_deleteData(data);
  mjcb_sensor = nullptr;
  return fell ? 1 : 0;
}
