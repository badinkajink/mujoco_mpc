"""Smoke test of the planner residual hook: run the own-sim agent_server (ONE at a time) on 'Lean H12 Magpie', strategy 25,
with and without H12_RESIDUAL_MLP, drive it like render_ownsim.py (plan_hz iterations per sim step) for a few seconds,
and report: the ON log line, finite states, and how far the two imagined-control trajectories diverge."""
import os, sys, time, subprocess, numpy as np
H12 = os.path.expanduser("~/Desktop/h12/mujoco_mpc/mujoco_mpc")
PROTO = sys.argv[1]; WEIGHTS = sys.argv[2]; SECONDS = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
sys.path.insert(0, PROTO); sys.path.insert(0, os.path.dirname(PROTO)); sys.path.insert(0, os.path.dirname(os.path.dirname(PROTO)))
import grpc
pb = pbg = None
for imp in ("mujoco_mpc.proto", "proto", ""):
    try:
        pb = __import__(f"{imp}.agent_pb2" if imp else "agent_pb2", fromlist=["x"]); pbg = __import__(f"{imp}.agent_pb2_grpc" if imp else "agent_pb2_grpc", fromlist=["x"]); break
    except ImportError: continue
assert pb is not None, "proto import failed"

def run(env_extra, port=20291, plan_hz=40.0, sim_dt=0.01):
    subprocess.run(["pkill", "-9", "-x", "agent_server"], stderr=subprocess.DEVNULL); time.sleep(1)
    env = dict(os.environ, **env_extra); errf = open(f"/tmp/claude-1000/agent_server_{'on' if env_extra else 'off'}.log", "w")
    proc = subprocess.Popen(["taskset", "-c", "0-11", f"{H12}/build/bin/agent_server", f"--mjpc_port={port}"], stdout=subprocess.DEVNULL, stderr=errf, env=env, cwd=H12)
    time.sleep(3); traj = []
    try:
        ch = grpc.secure_channel(f"localhost:{port}", grpc.local_channel_credentials(grpc.LocalConnectionType.LOCAL_TCP))
        grpc.channel_ready_future(ch).result(timeout=20); s = pbg.AgentStub(ch)
        s.Init(pb.InitRequest(task_id="Lean H12 Magpie"), timeout=60); s.Reset(pb.ResetRequest(), timeout=30)
        req = pb.SetTaskParametersRequest(); req.parameters["Strategy"].numeric = 25.0; s.SetTaskParameters(req, timeout=10)
        s.Step(pb.StepRequest(), timeout=30)
        for _ in range(20): s.PlannerStep(pb.PlannerStepRequest(), timeout=30)
        acc = 0.0; t = 0.0; t_wall = time.time()
        while t < SECONDS:
            st = s.GetState(pb.GetStateRequest(), timeout=30).state; t = st.time; traj.append((t, np.array(st.qpos, float)))
            acc += plan_hz * sim_dt; n = int(acc); acc -= n
            for _ in range(n): s.PlannerStep(pb.PlannerStepRequest(), timeout=30)
            s.Step(pb.StepRequest(), timeout=30)
        wall = time.time() - t_wall
    finally:
        proc.kill(); proc.wait(); errf.close(); time.sleep(0.5)
    log = open(errf.name).read()
    return traj, log, wall

off, log_off, w_off = run({})
on, log_on, w_on = run({"H12_RESIDUAL_MLP": WEIGHTS})
print("OFF: steps", len(off), f"wall {w_off:.0f}s", "residual lines:", [l for l in log_off.splitlines() if "residual" in l.lower()][:2])
print("ON : steps", len(on), f"wall {w_on:.0f}s", "residual lines:", [l for l in log_on.splitlines() if "residual" in l.lower()][:3])
n = min(len(off), len(on)); qo = np.array([q for _, q in off[:n]]); qn = np.array([q for _, q in on[:n]])
print("finite:", np.isfinite(qo).all(), np.isfinite(qn).all(), " pelvis z end off/on:", qo[-1, 2].round(3), qn[-1, 2].round(3))
print("joint rms diff on-vs-off over time (deg): ", np.degrees(np.sqrt(np.mean((qo[:, 7:34] - qn[:, 7:34]) ** 2, 1)))[::max(1, n // 8)].round(2))
