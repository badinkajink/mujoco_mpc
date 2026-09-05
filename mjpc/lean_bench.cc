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
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "mjpc/agent.h"
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
  mjData* data = mj_makeData(model);

  int home_id = mj_name2id(model, mjOBJ_KEY, "home");
  if (home_id >= 0) mj_resetDataKeyframe(model, data, home_id);
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
  std::fprintf(fo, "t,phase,phase_name,pelvis_z,torso_tilt_deg,face_z,pad_clear,"
                   "f_shoulder,f_forearm,f_wrist,f_gripper,"
                   "f_r_elbow,f_r_wrist,f_r_gripper,f_torso,f_pelvis,"
                   "f_other,f_robot_total,cost,"
                   "rhand_x,rhand_y,rhand_z,tgt_x,tgt_y,tgt_z");
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
    agent.state.Set(model, data);
    agent.ActivePlanner().ActionFromPolicy(data->ctrl, agent.state.state().data(),
                                           agent.state.time(), /*use_previous=*/false);
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
      double f_total = 0.0;                           // ROBOT-on-table only
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
               "[bench-summary] task=%s strategy=%d table_h=%.4f face_z=%.4f seed=%d "
               "fell=%d complete=%d t_complete=%.3f t_end=%.3f phases=%d wall_s=%.1f enter=",
               task_name.c_str(), strategy, table_h, face_z, seed, (int)fell,
               (int)(t_complete > 0), t_complete, data->time, n_phases, wall);
  for (int p = 0; p < n_phases && p < 64; p++)
    std::fprintf(stderr, "%s%.2f", p ? ":" : "", phase_enter[p]);
  std::fprintf(stderr, "\n");

  if (fo != stdout) std::fclose(fo);
  if (fq) std::fclose(fq);
  mj_deleteData(data);
  mjcb_sensor = nullptr;
  return fell ? 1 : 0;
}
