import time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, warp as wp, mujoco_warp as mjw
import planner_model
t0 = time.time()
wp.init()
print("warp", wp.__version__, "device", wp.get_device().name, "cuda", wp.get_device().arch if hasattr(wp.get_device(),'arch') else '')
m = planner_model.load("planner_lean_s25_parity", warp=True)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "forearm_brace_lean"))
mujoco.mj_forward(m, d)
coll = int(((m.geom_contype | m.geom_conaffinity) != 0).sum())
print(f"collidable geoms {coll}/{m.ngeom}, CPU contacts at brace keyframe: {d.ncon}")
nworld = int(sys.argv[1]) if len(sys.argv) > 1 else 1
mw = mjw.put_model(m)
dw = mjw.put_data(m, d, nworld=nworld)
print(f"put_model/put_data ok nworld={nworld}, {time.time()-t0:.1f}s; naconmax={dw.naconmax} njmax={dw.njmax}")
t1 = time.time()
with wp.ScopedCapture() as cap:
    mjw.step(mw, dw)
print(f"capture (kernel compile) {time.time()-t1:.1f}s")
wp.synchronize()
t2 = time.time(); N = 100
for _ in range(N): wp.capture_launch(cap.graph)
wp.synchronize()
dt = (time.time()-t2)/N
print(f"step: {dt*1e6:.0f} us/step for {nworld} worlds = {dt*1e6/nworld:.1f} us/world-step; ncon {dw.nacon.numpy()}")
qpos = dw.qpos.numpy(); print("qpos[0,:3] after 100 steps", qpos[0,:3], "finite", np.isfinite(qpos).all())
