"""P1 feasibility: the look-back bank's per-tick cost. K worlds of the planner model, each
with its own table-height offset (model batch over body_pos), advanced ONE control tick
(agent_timestep 0.010 = 1 step, or 5 x 0.002 plant-rate substeps) from the SAME uploaded
real state with the SAME command, then the predicted qpos/qvel downloaded for scoring.
Timing covers upload + step(s) + download, as the node would pay every tick at 100 Hz.
Usage: bench_bank.py <K> [substeps] [dt]"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, warp as wp, mujoco_warp as mjw
import planner_model
K = int(sys.argv[1]); SUB = int(sys.argv[2]) if len(sys.argv) > 2 else 1; DT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.010
wp.init()
m = planner_model.load("planner_lean_s25_parity", warp=True); m.opt.timestep = DT
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "forearm_brace_lean")); mujoco.mj_forward(m, d)
tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "table")
# per-world model variation: table height offsets in [-4, +4] cm (body_pos batched over worlds)
mw = mjw.put_model(m, batch_sizes={"body_pos": K})
bp = np.tile(m.body_pos[None], (K, 1, 1)).astype(np.float32)
bp[:, tb, 2] += np.linspace(-0.04, 0.04, K)
mw.body_pos = wp.array(bp, dtype=wp.vec3, device="cuda")
dw = mjw.put_data(m, d, nworld=K, nconmax=32 * K, njmax=128 * K, naccdmax=32 * K)
h_qpos = wp.array(np.tile(d.qpos, (K, 1)), dtype=wp.float32, device="cpu", pinned=True)
h_qvel = wp.array(np.tile(d.qvel, (K, 1)), dtype=wp.float32, device="cpu", pinned=True)
h_ctrl = wp.array(np.tile(d.ctrl, (K, 1)), dtype=wp.float32, device="cpu", pinned=True)
o_qpos = wp.zeros((K, m.nq), dtype=wp.float32, device="cpu", pinned=True)
o_qvel = wp.zeros((K, m.nv), dtype=wp.float32, device="cpu", pinned=True)
def tick():
    wp.copy(dw.qpos, h_qpos); wp.copy(dw.qvel, h_qvel); wp.copy(dw.ctrl, h_ctrl)
    for _ in range(SUB): mjw.step(mw, dw)
    wp.copy(o_qpos, dw.qpos); wp.copy(o_qvel, dw.qvel)
with wp.ScopedCapture() as cap: tick()
wp.synchronize()
ts = []
for r in range(200):
    wp.synchronize(); t0 = time.perf_counter(); wp.capture_launch(cap.graph); wp.synchronize(); ts.append(time.perf_counter() - t0)
ts = np.array(ts[20:]) * 1e3
q = o_qpos.numpy(); nacon = dw.nacon.numpy()
spread = float(q[:, 2].max() - q[:, 2].min())
res = dict(K=K, substeps=SUB, dt=DT, tick_ms_median=float(np.median(ts)), tick_ms_p95=float(np.percentile(ts, 95)), tick_ms_max=float(ts.max()),
           finite=bool(np.isfinite(q).all()), ncon_total=int(nacon.sum()), pelvis_z_spread_after_1tick=spread)
print(json.dumps(res))
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "bank_results.jsonl"), "a") as f: f.write(json.dumps(res) + "\n")
