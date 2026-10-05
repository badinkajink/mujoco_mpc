"""Closed-loop evaluation of a distilled policy on the TWIN plant (scene *_37_real_ns_grid, dt 5 ms, PD per step,
kp 150/200/90 + the node's kd), 100 Hz policy. Modes:
  replay  : open-loop replay of the recorded cmd/tau_ff of the init run (plant-fidelity control)
  hold    : PD hold of the initial pose
  policy  : the BC policy with a scripted phase schedule
usage: python bc_eval.py <mode> [policy.npz]   env: MBRL_EVAL_T (30 s) MBRL_EVAL_PUSH ("t,fx,fy,dur;...")
      MBRL_EVAL_SCHED ("t:phase:kf;..." default "0:0:1") MBRL_EVAL_INIT (run dir) MBRL_EVAL_T0 (twin time of the init state)"""
import os, sys, glob, re, numpy as np, mujoco
from bc_common import obs_vec
H12 = os.path.expanduser("~/Desktop/h12")
SCENE = f"{H12}/h1_mujoco/unitree_robots/h1_2/scene_handless_magpie_table_block_37_real_ns_grid.xml"
INIT = os.environ.get("MBRL_EVAL_INIT", f"{H12}/logs/grid37_twin/d37/s37_B3_noassist_base_r4"); T0 = float(os.environ.get("MBRL_EVAL_T0", 40.0))
mode = sys.argv[1]
m = mujoco.MjModel.from_xml_path(SCENE); m.opt.timestep = 0.005; d = mujoco.MjData(m)
sid = lambda n: m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)]
IQ, IG = sid("imu_quat"), sid("imu_gyro")
tr = np.load(f"{INIT}/run.npz"); i0 = np.searchsorted(tr["t"], T0)
d.qpos[:] = tr["qpos_full"][i0]; d.qvel[:] = tr["qvel_full"][i0]; mujoco.mj_forward(m, d)
q0 = d.sensordata[:27].copy()
# recorded commands of the init run, aligned: node t = rec t - t_eng_rec ; twin t = node t + t_eng_twin
rec = np.load(glob.glob(f"{INIT}/*/*.npz")[0], allow_pickle=True); on = rec["kp"][:, 0] > 1
t_eng_rec = rec["t"][np.argmax(on)]; t_eng_twin = float(re.search(r"engaged at sim t=([0-9.]+)s", open(f"{INIT}/twin.log").read()).group(1))
rec_t_twin = rec["t"] - t_eng_rec + t_eng_twin
j0 = np.argmax(on) + 100; KP = rec["kp"][j0].copy(); KD = rec["kd"][j0].copy()   # the node's per-joint gains (legs 150/200, waist 200, arms 90/50)
print("gains kp", KP.astype(int), "kd", KD)
pol = None
S0 = np.where((rec_t_twin > T0 - 2.5) & (rec_t_twin < T0 + 2.5))[0]; CC = rec["cmd"][S0].mean(0); CT = rec["tau_ff"][S0].mean(0)   # const mode: mean recorded command around T0
if mode == "policy":
    from bc_policy import Policy; pol = Policy(sys.argv[2])
sched = [(float(a), int(b), int(c)) for a, b, c in (s.split(":") for s in os.environ.get("MBRL_EVAL_SCHED", "0:0:1").split(";"))]
pushes = [[float(v) for v in p.split(",")] for p in os.environ.get("MBRL_EVAL_PUSH", "").split(";") if p]
T = float(os.environ.get("MBRL_EVAL_T", 30)); torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
fore = [b for b in range(m.nbody) if "elbow" in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "wrist" in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)]
cmd = q0.copy(); tff = np.zeros(27); prev = q0.copy(); t = 0.0; log = []; fell = None; pad_contact = 0
while t < T + 1e-9:
    if round(t / 0.005) % 2 == 0:            # 100 Hz policy tick
        q = d.sensordata[:27]; dq = d.sensordata[27:54]
        ph, kf, tph = 0, 1, t
        for a, b, c in sched:
            if t >= a: ph, kf, tph = b, c, t - a
        if mode == "hold": cmd, tff = q0, np.zeros(27)
        elif mode == "const": cmd, tff = CC, CT
        elif mode == "replay":
            j = min(np.searchsorted(rec_t_twin, T0 + t), len(rec_t_twin) - 1); cmd, tff = rec["cmd"][j], rec["tau_ff"][j]
        else:
            x = obs_vec(q, dq, d.sensordata[IQ:IQ + 4], d.sensordata[IG:IG + 3], ph, kf, tph, prev)
            y = pol(x); cmd = q + y[:27]; tff = y[27:]
        prev = cmd.copy()
    d.xfrc_applied[:] = 0
    for p in pushes:
        if p[0] <= t < p[0] + p[3]: d.xfrc_applied[torso, 0:2] = p[1:3]
    q = d.sensordata[:27]; dq = d.sensordata[27:54]
    d.ctrl[:27] = tff + KP * (cmd - q) + KD * (0 - dq)
    mujoco.mj_step(m, d); t += 0.005
    pc = any((c.geom1 == table or c.geom2 == table) and (m.geom_bodyid[c.geom1] in fore or m.geom_bodyid[c.geom2] in fore) for c in d.contact[:d.ncon])
    pad_contact += pc * 0.005
    R = d.xmat[torso].reshape(3, 3); tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
    if round(t / 0.005) % 100 == 0: log.append((t, d.qpos[2], tilt, d.qpos[0], d.qpos[1], pc))
    if (d.qpos[2] < 0.55 or tilt > 70) and fell is None: fell = t; break
print(f"[{mode}] init {os.path.basename(INIT)} @twin t={T0}  fell={fell}  pad-contact {pad_contact:.1f}s  sched={sched}  pushes={pushes}")
for r in log[::max(1, len(log) // 12)]: print("  t=%5.1f pelvis z %.3f tilt %5.1f  xy %.3f %.3f  pad %d" % r)
