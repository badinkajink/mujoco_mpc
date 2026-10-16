"""Warp port of logs/twin_parity_0925/creep_test.py: hold a SEATED twin frame with a stiff
PD (+ recorded actuator force feed-forward) on the plant scene and measure pad/foot drift.
CPU reference (09-25, dt 0.001, 15 s): pyramidal 4.4 cm pad drift, elliptic 0.43 cm,
elliptic+noslip10 0.00 cm. Warp has no noslip, so this asks: how close can Warp's
elliptic solver get, sweeping iterations / impratio / solver.
Usage: creep_warp.py [--dur 15] [--variants a,b,c]"""
import argparse, os, sys, time, numpy as np, mujoco, warp as wp, mujoco_warp as mjw
H12 = os.path.expanduser("~/Desktop/h12")
SCENE = f"{H12}/h1_mujoco/unitree_robots/h1_2/scene_handless_magpie_table_block_37_real.xml"
RUN = f"{H12}/logs/grid37_twin/d37/s36_B3_real_g3"; T = 89.5

@wp.kernel
def pd_hold(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), qa: wp.array(dtype=int), va: wp.array(dtype=int),
            q0: wp.array(dtype=float), af: wp.array(dtype=float), kp: float, kd: float, ctrl: wp.array2d(dtype=float)):
    w, i = wp.tid()
    ctrl[w, i] = af[i] + kp * (q0[i] - qpos[w, qa[i]]) - kd * qvel[w, va[i]]

def variant(m, name):
    for part in name.split("+"):
        if part == "pyramidal": m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        elif part == "elliptic": m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        elif part.startswith("iter"): m.opt.iterations = int(part[4:])
        elif part.startswith("imp"): m.opt.impratio = float(part[3:])
        elif part.startswith("ls"): m.opt.ls_iterations = int(part[2:])
        elif part == "cg": m.opt.solver = mujoco.mjtSolver.mjSOL_CG
        elif part == "newton": m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        elif part.startswith("noslip"): m.opt.noslip_iterations = int(part[6:])
        elif part.startswith("dt"): m.opt.timestep = float(part[2:])
        else: raise SystemExit(f"unknown variant part {part}")

def yaw_of(R): return np.degrees(np.arctan2(R[1, 0], R[0, 0]))

def run(a, name):
    m = mujoco.MjModel.from_xml_path(a.scene); m.opt.timestep = a.dt
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD) | int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    variant(m, name)
    d = mujoco.MjData(m)
    tr = np.load(os.path.join(a.run, "run.npz"), allow_pickle=True)
    k = int(np.searchsorted(tr["t"], T))
    d.qpos[:] = tr["qpos_full"][k]; d.qvel[:] = 0
    af = tr["actuator_force"][k].astype(np.float32)
    jid = [m.actuator_trnid[i, 0] for i in range(m.nu)]
    qa = np.array([m.jnt_qposadr[j] for j in jid], dtype=np.int32); va = np.array([m.jnt_dofadr[j] for j in jid], dtype=np.int32)
    q0 = d.qpos[qa].astype(np.float32).copy()
    mujoco.mj_forward(m, d)
    fL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"); fR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    pad = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "left_forearm_pad")
    mw = mjw.put_model(m); dw = mjw.put_data(m, d, nworld=1, nconmax=256, njmax=1024)
    dqa, dva, dq0, daf = (wp.array(x, device="cuda") for x in (qa, va, q0, af))
    def snap():
        xmat = dw.xmat.numpy()[0]; xpos = dw.xpos.numpy()[0]; gx = dw.geom_xpos.numpy()[0]; q = dw.qpos.numpy()[0]
        return np.array([yaw_of(xmat[1]), yaw_of(xmat[fL]), yaw_of(xmat[fR]), *gx[pad][:2], *xpos[fL][:2], *xpos[fR][:2], q[2]])
    def chunk():
        for _ in range(a.chunk):
            wp.launch(pd_hold, dim=(1, m.nu), inputs=[dw.qpos, dw.qvel, dqa, dva, dq0, daf, a.kp, a.kd, dw.ctrl])
            mjw.step(mw, dw)
    t0 = time.time()
    with wp.ScopedCapture() as cap: chunk()
    wp.synchronize(); tc = time.time() - t0
    n_settle = int(1.0 / a.dt); n = int(a.dur / a.dt); done = 0; s0 = None; fell = False
    t0 = time.time()
    while done < n_settle + n:
        if done >= n_settle and s0 is None: s0 = snap()
        wp.capture_launch(cap.graph); done += a.chunk
        z = float(dw.qpos.numpy()[0, 2])
        if z < 0.5 or not np.isfinite(z): fell = True; break
    wp.synchronize(); ts = time.time() - t0
    s1 = snap(); dd = s1 - (s0 if s0 is not None else s1)
    ncon = int(dw.nacon.numpy().sum())
    print(f"{name:34s} base yaw {dd[0]:+6.2f} deg  footL yaw {dd[1]:+6.2f}  footR yaw {dd[2]:+6.2f}  "
          f"pad dxy {100*np.hypot(dd[3],dd[4]):5.2f} cm  footL dxy {100*np.hypot(dd[5],dd[6]):5.2f}  footR dxy {100*np.hypot(dd[7],dd[8]):5.2f}  "
          f"dz {100*dd[9]:+5.2f} cm  ncon {ncon:3d}  {'FELL' if fell else ''}  [{done} steps {ts:.1f}s = {ts/done*1e6:.0f} us/step, compile {tc:.0f}s]", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dur", type=float, default=15.0); ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--kp", type=float, default=400.0); ap.add_argument("--kd", type=float, default=20.0); ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--scene", default=SCENE); ap.add_argument("--run", default=RUN)
    ap.add_argument("--variants", default="pyramidal,elliptic,elliptic+iter300,elliptic+imp300,elliptic+iter300+imp1000,elliptic+cg+iter300")
    a = ap.parse_args(); wp.init()
    print(f"WARP creep test {os.path.basename(a.run)} @ t={T}  hold kp {a.kp} kd {a.kd}  dur {a.dur}s  dt {a.dt}  mujoco {mujoco.__version__} warp {wp.__version__}", flush=True)
    for v in a.variants.split(","): run(a, v)
