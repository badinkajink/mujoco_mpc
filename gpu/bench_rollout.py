"""P0 benchmark: iCEM-shaped rollouts of the deployed planner model on the GPU.

Mirrors one plan of the deployed iCEM (agent_planner 7): N rollouts x H steps at
agent_timestep 0.010 (horizon 1.0 s = 100 steps), controls = keyframe ctrl + a 3-knot
spline of Gaussian noise (sampling_spline_points 3, sampling_exploration 0.12), from a
braced keyframe (contacts live). Timing INCLUDES the per-plan host->device state upload
and the device->host download of the per-world cost proxy, as the node would pay.
Usage: bench_rollout.py <nworld> [horizon_steps] [keyframe]
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, warp as wp, mujoco_warp as mjw
import planner_model

nworld = int(sys.argv[1]); H = int(sys.argv[2]) if len(sys.argv) > 2 else 100
key = sys.argv[3] if len(sys.argv) > 3 else "forearm_brace_lean"
REPS = 30
wp.init()
m = planner_model.load("planner_lean_s25_parity", warp=True)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key))
mujoco.mj_forward(m, d)
rng = np.random.default_rng(0)
# 3-knot spline noise per world/actuator, linearly interpolated over H steps (zero-order on knots)
knots = rng.normal(0, 0.12, size=(3, nworld, m.nu)); knots[:, 0, :] = 0.0   # world 0 = nominal
t = np.linspace(0, 2, H)
ctrl = np.empty((H, nworld, m.nu), dtype=np.float32)
for k in range(H):
    i0 = min(int(np.floor(t[k])), 1); a = t[k] - i0
    ctrl[k] = d.ctrl[None, :] + (1 - a) * knots[i0] + a * knots[i0 + 1]
ctrl = np.clip(ctrl, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]).astype(np.float32)

# ---- CPU reference (pip MuJoCo, single thread): H steps of world 1 ----
dc = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, dc, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key))
t0 = time.perf_counter()
for k in range(H):
    dc.ctrl[:] = ctrl[k, 1]; mujoco.mj_step(m, dc)
cpu_step_us = (time.perf_counter() - t0) / H * 1e6
cpu_ncon = dc.ncon

# ---- GPU ----
mw = mjw.put_model(m)
NC = int(os.environ.get("NCON", 32)); NJ = int(os.environ.get("NJ", 128))
NA = int(os.environ.get("NACCD", 0))
dw = mjw.put_data(m, d, nworld=nworld, nconmax=NC * nworld, njmax=NJ * nworld, naccdmax=(NA * nworld or None))
print("sizes: naconmax", dw.naconmax, "njmax", dw.njmax, "naccdmax", dw.naccdmax, "nccdmax", getattr(dw, "nccdmax", None), file=sys.stderr)
ctrl_dev = wp.array(ctrl, dtype=wp.float32, device="cuda")       # [H, nworld, nu]
qpos0 = wp.array(np.tile(d.qpos, (nworld, 1)), dtype=wp.float32, device="cuda")
qvel0 = wp.array(np.tile(d.qvel, (nworld, 1)), dtype=wp.float32, device="cuda")
host_qpos = wp.array(np.tile(d.qpos, (nworld, 1)), dtype=wp.float32, device="cpu", pinned=True)
host_qvel = wp.array(np.tile(d.qvel, (nworld, 1)), dtype=wp.float32, device="cpu", pinned=True)
host_out = wp.zeros((nworld, m.nq), dtype=wp.float32, device="cpu", pinned=True)

CHUNK = int(os.environ.get("CHUNK", 10)); assert H % CHUNK == 0
step_i = wp.zeros(1, dtype=wp.int32, device="cuda")   # unused; chunks index ctrl by python-side offset
graphs = []
t1 = time.perf_counter()
for c in range(H // CHUNK):                            # one graph per chunk (a 100-step graph OOMs at 256 worlds)
    with wp.ScopedCapture() as cap:
        if c == 0: wp.copy(dw.qpos, host_qpos); wp.copy(dw.qvel, host_qvel)   # state upload (H2D)
        for k in range(c * CHUNK, (c + 1) * CHUNK):
            wp.copy(dw.ctrl, ctrl_dev[k]); mjw.step(mw, dw)
        if c == H // CHUNK - 1: wp.copy(host_out, dw.qpos)                     # cost proxy download (D2H)
    graphs.append(cap.graph)
wp.synchronize(); compile_s = time.perf_counter() - t1
times = []
for r in range(REPS):
    wp.synchronize(); ta = time.perf_counter()
    for g in graphs: wp.capture_launch(g)
    wp.synchronize()
    times.append(time.perf_counter() - ta)
    wp.copy(dw.qpos, qpos0); wp.copy(dw.qvel, qvel0)
times = np.array(times[3:]) * 1e3
out = host_out.numpy(); finite = bool(np.isfinite(out).all())
ncon = dw.nacon.numpy()
import subprocess
vram = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
res = dict(nworld=nworld, H=H, key=key, plan_ms_median=float(np.median(times)), plan_ms_min=float(times.min()), plan_ms_max=float(times.max()),
           us_per_world_step=float(np.median(times) * 1e3 / (nworld * H)), compile_s=compile_s, cpu_step_us_pip315=cpu_step_us,
           cpu_ncon=int(cpu_ncon), gpu_ncon_total=int(ncon.sum()) if ncon.ndim else int(ncon), finite=finite, vram_mb=vram,
           mujoco=mujoco.__version__, warp=wp.__version__)
print(json.dumps(res))
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "bench_results.jsonl"), "a") as f:
    f.write(json.dumps(res) + "\n")
