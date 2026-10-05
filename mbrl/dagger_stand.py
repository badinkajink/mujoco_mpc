"""DAgger distillation (run with MBRL_BC_PREV=1: the planner's action is ~70 % its own previous action -- a persistent plan --
so the student needs the last executed command as input; state alone explains none of the label, state+prev explains it to 0.6 deg) of the iCEM stand (Stage 3b). Teacher = agent_server (own-sim planner, task 'Lean H12 Magpie',
strategy slot 37, planner model with the NODE's actuator gains injected via Init(model.mjb)). Student env = the twin plant
(scene *_37_real_ns_grid, dt 5 ms, PD per step, node gains, tau_ff = 0.85*qfrc_bias on the planner model, like the node).
Loop: episodes from recorded standing states (+ random torso pushes), action = beta*teacher + (1-beta)*student, record
(obs, teacher cmd - q), refit the student on demos(stand) + aggregated data, evaluate the student alone.
env: DAG_ITERS 6, DAG_EPIS 12, DAG_T 8 (s), DAG_K 2 (planner steps per label), DAG_HZ 50, DAG_PORT 20292, DAG_SMOKE 1, DAG_TAG"""
import os, sys, time, glob, re, subprocess, numpy as np, mujoco, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common; from bc_common import obs_vec, NOBS; from bc_policy import Policy
H12 = os.path.expanduser("~/Desktop/h12"); MJ = f"{H12}/mujoco_mpc/mujoco_mpc"; HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f"{HERE}/out_bc"
PROTO = f"{H12}/mujoco_mpc/monitor/proto"; sys.path.insert(0, PROTO); sys.path.insert(0, os.path.dirname(PROTO))
import grpc
try: from proto import agent_pb2 as pb, agent_pb2_grpc as pbg
except ImportError: import agent_pb2 as pb, agent_pb2_grpc as pbg
E = lambda k, v: type(v)(os.environ.get(k, v))
ITERS, EPIS, T_EP, K, HZ, PORT, SMOKE, TAG = E("DAG_ITERS", 6), E("DAG_EPIS", 12), E("DAG_T", 8.0), E("DAG_K", 2), E("DAG_HZ", 50), E("DAG_PORT", 20292), E("DAG_SMOKE", 0), E("DAG_TAG", "dag")
DT = 0.005; SUB = int(round(1.0 / HZ / DT)); GFF = 0.85
rng = np.random.default_rng(int(os.environ.get("DAG_SEED", 0))); DART = np.radians(float(os.environ.get("DAG_DART", 0)))   # DART noise std (deg) on executed teacher commands

# ---- plant (twin) + planner model (for tau_ff) ----
plant = mujoco.MjModel.from_xml_path(f"{H12}/h1_mujoco/unitree_robots/h1_2/scene_handless_magpie_table_block_37_real_ns_grid.xml"); plant.opt.timestep = DT; pd = mujoco.MjData(plant)
sid = lambda n: plant.sensor_adr[mujoco.mj_name2id(plant, mujoco.mjtObj.mjOBJ_SENSOR, n)]; IQ, IG = sid("imu_quat"), sid("imu_gyro")
torso = mujoco.mj_name2id(plant, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
pm = common.load_model(disable_actuators=False); pdat = mujoco.MjData(pm)
# node gains from a recording
rec = np.load(glob.glob(f"{H12}/logs/grid37_twin/d37/s37_B3_noassist_base_r4/*/*.npz")[0], allow_pickle=True); j0 = np.argmax(rec["kp"][:, 0] > 1) + 100
KP, KD = rec["kp"][j0].copy(), rec["kd"][j0].copy()
for i in range(27):   # inject the node's gains into the teacher's planner model (own-sim skips PatchActuators)
    pm.actuator_gainprm[i, 0] = KP[i]; pm.actuator_biasprm[i, 1] = -KP[i]; pm.actuator_biasprm[i, 2] = -KD[i]
def model_xml():
    """planner XML with the node's gains, as a self-contained string (absolute meshdir) for Init(model.xml)."""
    spec = mujoco.MjSpec.from_file(common.MODEL); base = os.path.dirname(os.path.abspath(common.MODEL))
    for attr in ("meshdir", "texturedir"):
        v = getattr(spec, attr, "")
        if not os.path.isabs(v or ""): setattr(spec, attr, os.path.normpath(os.path.join(base, v or "")))
    for i, a in enumerate(spec.actuators):
        a.gainprm[0] = KP[i]; a.biasprm[1] = -KP[i]; a.biasprm[2] = -KD[i]
    xml = spec.to_xml(); mt = mujoco.MjModel.from_xml_string(xml); print("[teacher] model kp", mt.actuator_gainprm[:, 0].astype(int), flush=True); return xml

def tau_ff(qpos):
    pdat.qpos[:] = pm.qpos0; pdat.qpos[:34] = qpos[:34]; pdat.qvel[:] = 0; mujoco.mj_forward(pm, pdat); return GFF * pdat.qfrc_bias[6:33].copy()

# ---- init states: standing, harness released, before the lean (twin t 37..50), non-fallen runs ----
inits = []
for d in sorted(glob.glob(f"{H12}/logs/grid37_twin/d37/s37_B3_noassist_base*")):
    if not os.path.exists(f"{d}/run.npz"): continue
    tw = open(f"{d}/twin.log", errors="ignore").read(); mf = re.search(r"FALLEN for 8s \(since t=([0-9.]+)s", tw)
    if mf and float(mf.group(1)) < 60: continue
    tr = np.load(f"{d}/run.npz")
    for t0 in (38, 41, 44, 47, 50):
        i = np.searchsorted(tr["t"], t0); inits.append((os.path.basename(d), t0, tr["qpos_full"][i].copy(), tr["qvel_full"][i].copy()))
print(f"[dagger] {len(inits)} init states from {len(set(i[0] for i in inits))} runs; gains kp {KP.astype(int)}", flush=True)
# absolute action parametrisation: the student predicts cmd - QREF (fixed reference = mean standing pose of the init states),
# NOT cmd - q: relative targets make the posture a random walk (q drifts -> cmd drifts with it) and the stand integrates away.
QREF = np.mean([i[2][7:34] for i in inits], axis=0)
hold = [i for i in range(len(inits)) if i % 5 == 4]; train_i = [i for i in range(len(inits)) if i % 5 != 4]   # 1/5 of states held out for eval

# ---- teacher ----
class Teacher:
    def __init__(self):
        subprocess.run(["pkill", "-9", "-x", "agent_server"], stderr=subprocess.DEVNULL); time.sleep(1)
        self.err = open(f"{OUT}/agent_server_{TAG}.log", "w")
        env = {k: v for k, v in os.environ.items() if not k.startswith("H12_RESIDUAL")}
        self.p = subprocess.Popen(["taskset", "-c", "0-11", f"{MJ}/build/bin/agent_server", f"--mjpc_port={PORT}", "--mjpc_workers=12"], stdout=self.err, stderr=subprocess.STDOUT, env=env, cwd=MJ)
        time.sleep(3); ch = grpc.secure_channel(f"localhost:{PORT}", grpc.local_channel_credentials(grpc.LocalConnectionType.LOCAL_TCP))
        grpc.channel_ready_future(ch).result(timeout=30); self.s = pbg.AgentStub(ch)
        self.s.Init(pb.InitRequest(task_id="Lean H12 Magpie", model=pb.MjModel(xml=model_xml())), timeout=120)
        self.new_episode(None); print(f"[teacher] up: nq {self.nq} nv {self.nv} t0 {self.t0:.3f}", flush=True)
    def new_episode(self, qpos):
        """Reset the task (its internal timers: stand trim/park, success_sustain -> keyframe advance) and warm the planner
        at the episode's start state, so labels are 'stand, kf 1' and the time fed to the task is continuous from ~0."""
        self.s.Reset(pb.ResetRequest(), timeout=30)
        req = pb.SetTaskParametersRequest(); req.parameters["Strategy"].numeric = 37.0; self.s.SetTaskParameters(req, timeout=10)
        self.s.Step(pb.StepRequest(), timeout=30)
        st = self.s.GetState(pb.GetStateRequest(), timeout=10).state; self.nq = len(st.qpos); self.nv = len(st.qvel); self.t0 = st.time
        if qpos is not None:
            q = np.array(pm.qpos0, float); q[:34] = qpos[:34]
            self.s.SetState(pb.SetStateRequest(state=pb.State(time=self.t0, qpos=q.tolist(), qvel=[0.0] * pm.nv)), timeout=10)
        for _ in range(12): self.s.PlannerStep(pb.PlannerStepRequest(), timeout=30)
    def label(self, qpos, qvel, t, k=K):
        q = np.array(pm.qpos0, float); q[:34] = qpos[:34]; v = np.zeros(pm.nv); v[:33] = qvel[:33]
        self.s.SetState(pb.SetStateRequest(state=pb.State(time=self.t0 + t, qpos=q.tolist(), qvel=v.tolist())), timeout=10)
        for _ in range(k): self.s.PlannerStep(pb.PlannerStepRequest(), timeout=30)
        return np.array(self.s.GetAction(pb.GetActionRequest(), timeout=10).action, float)
    def close(self):
        self.p.kill(); self.p.wait(); self.err.close()

# ---- episode ----
def obs_now(t, prev):
    return obs_vec(pd.sensordata[:27].copy(), pd.sensordata[27:54].copy(), pd.sensordata[IQ:IQ + 4].copy(), pd.sensordata[IG:IG + 3].copy(), 0, 1, 30.0, prev)
def fallen():
    R = pd.xmat[torso].reshape(3, 3); return pd.qpos[2] < 0.6 or R[2, 2] < 0.5
def episode(init, teacher, student, beta, push, T=T_EP, record=True, k=None):
    name, t0, qp, qv = init; pd.qpos[:] = qp; pd.qvel[:] = qv; mujoco.mj_forward(plant, pd)
    if teacher is not None: teacher.new_episode(qp)
    X, Y = [], []; t = 0.0; cmd = pd.sensordata[:27].copy(); dart_n = np.zeros(27); tf = tau_ff(pd.qpos); nlab = 0; tw0 = time.time()
    while t < T - 1e-9:
        if round(t / DT) % SUB == 0:
            q = pd.sensordata[:27].copy(); x = obs_now(t, cmd)      # prev = the command actually executed last tick
            a_t = teacher.label(pd.qpos, pd.qvel, t, k=k or K) if teacher is not None else None
            a_s = QREF + student(x)[:27] if student is not None else None
            if a_t is not None and record: X.append(x); Y.append((a_t - QREF).astype(np.float32)); nlab += 1
            if a_t is not None and a_s is not None: cmd = beta * a_t + (1 - beta) * a_s
            else: cmd = a_t if a_t is not None else a_s
            if a_t is not None and DART > 0:      # DART: execute a noised teacher command (label stays clean) -> robust BC
                dart_n = 0.8 * dart_n + 0.6 * rng.normal(0, DART, 27); cmd = cmd + dart_n
            tf = tau_ff(pd.qpos)
        pd.xfrc_applied[:] = 0
        if push is not None and push[0] <= t < push[0] + 0.3: pd.xfrc_applied[torso, 0:2] = push[1:3]
        q = pd.sensordata[:27]; dq = pd.sensordata[27:54]; pd.ctrl[:27] = tf + KP * (cmd - q) - KD * dq
        mujoco.mj_step(plant, pd); t += DT
        if fallen(): break
    if t < T - 1e-9 and len(X) > 50: X, Y = X[:-50], Y[:-50]      # fell: drop the last second of labels (teacher was already losing it)
    return np.array(X), np.array(Y), t, nlab, time.time() - tw0

def rand_push():
    if rng.random() < 0.4: return None
    ang = rng.uniform(0, 2 * np.pi); f = rng.uniform(float(os.environ.get('DAG_PUSH_LO', 20)), float(os.environ.get('DAG_PUSH_HI', 70))); return (rng.uniform(1.0, T_EP - 2.0), f * np.cos(ang), f * np.sin(ang))

# ---- student fit (torch, GPU) ----
def fit(X, Y, W, tag, epochs=30, H=256, NL=3):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = X.mean(0), X.std(0) + 1e-6; ymu, ysd = Y.mean(0), Y.std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, device=dev); Yt = torch.tensor((Y - ymu) / ysd, device=dev); Wt = torch.tensor(W, device=dev)
    layers = []; nin = X.shape[1]
    for _ in range(NL): layers += [nn.Linear(nin, H), nn.ELU()]; nin = H
    net = nn.Sequential(*layers, nn.Linear(nin, Y.shape[1])).to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs); n = len(Xt); B = 2048
    noise = torch.tensor(np.r_[np.radians(0.3) * np.ones(27), 0.05 * np.ones(27), 0.01 * np.ones(3), 0.03 * np.ones(3), np.zeros(NOBS - 60)] / sd, device=dev, dtype=torch.float32)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, B):
            b = perm[i:i + B]; loss = (((net(Xt[b] + torch.randn_like(Xt[b]) * noise) - Yt[b]) ** 2).mean(1) * Wt[b]).mean(); opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    lin = [m for m in net if isinstance(m, nn.Linear)]
    Wd = {f"W{i}": l.weight.detach().cpu().numpy() for i, l in enumerate(lin)}; Wd.update({f"b{i}": l.bias.detach().cpu().numpy() for i, l in enumerate(lin)})
    # pad to the 54-wide action the Policy class expects (tau_ff head unused: zeros)
    ymu54 = np.r_[ymu, np.zeros(27)].astype(np.float32); ysd54 = np.r_[ysd, np.ones(27)].astype(np.float32)
    Wl = Wd[f"W{len(lin) - 1}"]; bl = Wd[f"b{len(lin) - 1}"]; Wd[f"W{len(lin) - 1}"] = np.r_[Wl, np.zeros((27, Wl.shape[1]), np.float32)]; Wd[f"b{len(lin) - 1}"] = np.r_[bl, np.zeros(27, np.float32)]
    path = f"{OUT}/bc_{tag}.npz"; np.savez(path, mu=mu, sd=sd, ymu=ymu54, ysd=ysd54, nl=len(lin), mask=np.ones(X.shape[1], np.float32), **Wd); return path

def evaluate(student, tag, T=15.0):
    res = []
    for i in hold:
        for push in (None, (3.0, 80.0, 0.0), (3.0, 0.0, 70.0)):
            _, _, tsurv, _, _ = episode(inits[i], None, student, 0.0, push, T=T, record=False); res.append((push is not None, tsurv >= T - 1e-6, tsurv))
    r = np.array(res, float); print(f"[eval {tag}] no-push survive {r[r[:,0]==0,1].mean():.2f}  push survive {r[r[:,0]==1,1].mean():.2f}  mean t {r[:,2].mean():.1f}/{T}", flush=True); return r

if __name__ == "__main__":
    teacher = Teacher()
    try:
        if os.environ.get("DAG_CHECK"):
            for kk in [int(v) for v in os.environ["DAG_CHECK"].split(",")]:
                for n, i in enumerate(train_i[:6]):
                    push = None if n < 4 else (2.0, 60.0 * (1 if n == 4 else 0), 60.0 * (0 if n == 4 else 1))
                    X, Y, tsurv, nlab, wall = episode(inits[i], teacher, None, 1.0, push, k=kk)
                    np.savez_compressed(f"{OUT}/check_K{kk}_{n}.npz", X=X, Y=Y)
                    print(f"[check K={kk}] {inits[i][0]}@{inits[i][1]} push {push} survived {tsurv:.1f}/{T_EP}s  labels {nlab}  {wall/max(nlab,1)*1e3:.0f} ms/label  |cmd-q| {np.degrees(np.sqrt((Y**2).mean())) if len(Y) else 0:.2f} deg", flush=True)
            sys.exit(0)
        if SMOKE:
            X, Y, tsurv, nlab, wall = episode(inits[train_i[0]], teacher, None, 1.0, None, T=min(T_EP, 5.0))
            print(f"[smoke] teacher-only: survived {tsurv:.2f}s of 5, labels {nlab}, wall {wall:.1f}s ({wall / max(nlab,1) * 1e3:.0f} ms/label), |cmd-q| deg rms {np.degrees(np.sqrt((Y**2).mean())):.2f}", flush=True)
            X2, Y2, tsurv2, _, _ = episode(inits[train_i[1]], teacher, None, 1.0, (1.5, 90.0, 0.0), T=min(T_EP, 5.0))
            print(f"[smoke] teacher-only with 90 N push: survived {tsurv2:.2f}s of 5", flush=True); sys.exit(0)
        # seed set: planner demonstrations (stand phase) from the twin recordings
        Xa, Ya, Wa = [], [], []
        if not os.environ.get("DAG_NO_DEMOS"):
            z = np.load(f"{OUT}/bc_data.npz"); keep = z["phase"] == 0; X0 = z["X"][keep]; Y0 = z["Y"][keep][:, :27]
            Xa, Ya, Wa = [X0], [Y0], [np.full(len(X0), 0.3, np.float32)]
        student = None; NDEMO = len(Xa)
        for it in range(ITERS):
            beta = 1.0 if it == 0 else (max(0.0, 0.5 ** it) if os.environ.get('DAG_BETA_DECAY') else 0.0)   # classic DAgger: teacher once, then pure student rollouts
            Xi, Yi, surv = [], [], []
            for e in range(EPIS):
                init = inits[train_i[rng.integers(len(train_i))]]; push = rand_push()
                X, Y, tsurv, nlab, wall = episode(init, teacher, student, beta, push)
                if len(X): Xi.append(X); Yi.append(Y)
                surv.append(tsurv); print(f"  it {it} ep {e} beta {beta:.2f} init {init[0]}@{init[1]} survived {tsurv:.1f}s labels {nlab} wall {wall:.0f}s push {None if push is None else (round(push[0],1), int(np.hypot(push[1], push[2])))}", flush=True)
            Xi = np.concatenate(Xi); Yi = np.concatenate(Yi); Xa.append(Xi); Ya.append(Yi); Wa.append(np.ones(len(Xi), np.float32))
            print(f"[it {it}] mixed-policy mean survival {np.mean(surv):.1f}/{T_EP}s; aggregated DAgger samples {sum(len(x) for x in Xa[NDEMO:])}", flush=True)
            path = fit(np.concatenate(Xa), np.concatenate(Ya), np.concatenate(Wa), f"{TAG}{it}"); student = Policy(path)
            evaluate(student, f"{TAG}{it}")
            np.savez_compressed(f"{OUT}/dagger_{TAG}_data.npz", X=np.concatenate(Xa[NDEMO:]), Y=np.concatenate(Ya[NDEMO:]))
    finally:
        teacher.close()
