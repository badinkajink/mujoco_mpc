"""C++ vs Python feature parity: read the H12_RESIDUAL_DEBUG rows (time, x[168], y[27], qpos[34], qvel[33], ctrl[27]) written
by the planner hook, rebuild the feature vector in Python from the same qpos/qvel/ctrl with the model's actuator gains
(own-sim: XML gains), and compare feature by feature; also check the MLP output against the numpy forward."""
import sys, struct, numpy as np, mujoco, torch
from common import *
import rollout_engine as RE
path, tag = sys.argv[1], sys.argv[2]
rows = []; rec = struct.Struct("<d168f27f34d33d27d")
with open(path, "rb") as f:
    while True:
        b = f.read(rec.size)
        if len(b) < rec.size: break
        v = rec.unpack(b); rows.append((v[0], np.array(v[1:169]), np.array(v[169:196]), np.array(v[196:230]), np.array(v[230:263]), np.array(v[263:290])))
print("rows", len(rows))
m = load_model(disable_actuators=False); d = mujoco.MjData(m)
kp = m.actuator_gainprm[:, 0].copy(); kd = -m.actuator_biasprm[:, 2].copy(); lim = m.actuator_forcerange[:, 1].copy()
st = torch.load(f"{OUT}/residual_mlp_{tag}.pt", map_location="cpu"); nz = np.load(f"{OUT}/residual_norm_{tag}.npz")
mlp = RE.NumpyMLP({k: v.numpy() for k, v in st.items()}, nz["mu"], nz["sd"], nz["ymu"], nz["ysd"], nz["clip"])
names = ["q"] * 27 + ["dq"] * 27 + ["tau"] * 27 + ["cmd-q"] * 27 + ["kp"] * 27 + ["kd"] * 27 + ["g"] * 3 + ["w"] * 3
maxerr = np.zeros(168); yerr = 0
for t, x, y, qpos, qvel, ctrl in rows:
    d.qpos[:34] = qpos; d.qvel[:33] = qvel; mujoco.mj_kinematics(m, d)
    q = qpos[7:34]; dq = qvel[6:33]; tau = np.clip(kp * (ctrl - q) - kd * dq, -lim, lim)
    bq = np.zeros(4); mujoco.mju_mat2Quat(bq, d.site("imu").xmat)
    xp = np.concatenate([q, dq, tau, ctrl - q, kp / 100, kd / 10, RE.gvec(bq[None])[0], qvel[3:6]])
    maxerr = np.maximum(maxerr, np.abs(xp - x)); yerr = max(yerr, np.abs(mlp(x[None].astype(np.float32))[0] - y).max())
for nm in dict.fromkeys(names):
    ids = [i for i, n in enumerate(names) if n == nm]; print(f"  {nm:6s} max |py - c++| = {maxerr[ids].max():.5f}")
print("MLP output max |numpy - c++| =", round(float(yerr), 6))
print("sample row 0: t", rows[0][0], " |y| max", np.abs(rows[0][2]).max().round(3), " g c++", rows[0][1][162:165].round(3), " w", rows[0][1][165:168].round(3))
