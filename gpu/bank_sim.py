"""Step 3: does a look-back bank identify a structured sim-to-real gap from (state, command) ticks?

Plant = the honest twin scene (elliptic + noslip, sysid'd robot) with an INJECTED gap: table height
offset dz_true and/or an ankle-pitch zero offset (reported angle = true + off). Controller = the node's
PD (KP/KV from h12_control_node.cc) on REPORTED joints toward a target pose (planner keyframe, with an
optional stand -> brace interpolation). Bank = K copies of the deployed planner model (pip MuJoCo,
agent_timestep 0.010, one step per tick) each carrying a hypothesis (table dz, ankle offset). Every
10 ms tick each world is reset to the reported state (ankle hypothesis applied), stepped once with the
same command, and scored by a windowed residual against the next reported state. We report whether
and how fast the true hypothesis leads. CPU only (the GPU version only changes where the K steps run).
Usage: bank_sim.py --dz_true -0.03 --ank_true 0.0 [--scenario hold|lean] [--dur 8]"""
import argparse, os, sys, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, planner_model
H12 = os.path.expanduser("~/Desktop/h12")
PLANT = f"{H12}/h1_mujoco/unitree_robots/h1_2/scene_handless_magpie_table_block_37_real_ns_grid.xml"
KP = np.array([150, 200, 200, 200, 200, 80, 150, 200, 200, 200, 200, 80, 200, 90, 60, 40, 90, 15, 15, 15, 90, 60, 40, 90, 15, 15, 15], float)
KV = np.array([5, 5, 5, 5, 4, 4, 5, 5, 5, 5, 4, 4, 5, 10, 10, 10, 10, 2, 2, 2, 10, 10, 10, 10, 2, 2, 2], float)
TICK = 0.010

def joint_maps(m):
    jn = [m.actuator(i).name.replace("_motor", "").replace("_act", "") for i in range(m.nu)]
    jid = [m.actuator_trnid[i, 0] for i in range(m.nu)]
    return np.array([m.jnt_qposadr[j] for j in jid]), np.array([m.jnt_dofadr[j] for j in jid]), [m.joint(j).name for j in jid]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dz_true", type=float, default=-0.03); ap.add_argument("--ank_true", type=float, default=0.0, help="rad, reported = true + off")
    ap.add_argument("--scenario", default="hold"); ap.add_argument("--dur", type=float, default=8.0); ap.add_argument("--window", type=float, default=0.3)
    ap.add_argument("--dz_grid", default="-0.04,-0.03,-0.02,-0.01,0,0.01,0.02,0.03,0.04"); ap.add_argument("--ank_grid", default="-0.035,-0.0175,0,0.0175,0.035")
    ap.add_argument("--key", default="forearm_brace_lean"); ap.add_argument("--lean_sec", type=float, default=4.0); ap.add_argument("--settle", type=float, default=1.0); ap.add_argument("--resid", default="all")
    ap.add_argument("--wiggle_amp", type=float, default=0.035); ap.add_argument("--wiggle_hz", type=float, default=0.5)
    a = ap.parse_args()
    # ---- plant ----
    P = mujoco.MjModel.from_xml_path(PLANT); P.opt.timestep = 0.002
    tbP = mujoco.mj_name2id(P, mujoco.mjtObj.mjOBJ_BODY, "table"); P.body_pos[tbP, 2] += a.dz_true
    pd = mujoco.MjData(P)
    qaP, vaP, namesP = joint_maps(P)
    # ---- bank worlds (planner model) ----
    B = planner_model.load("planner_lean_s25_parity")
    qaB, vaB, namesB = joint_maps(B)
    assert namesP == namesB, (namesP, namesB)
    tbB = mujoco.mj_name2id(B, mujoco.mjtObj.mjOBJ_BODY, "table")
    key = mujoco.mj_name2id(B, mujoco.mjtObj.mjOBJ_KEY, a.key); kq = B.key_qpos[key].copy()
    stand = B.key_qpos[mujoco.mj_name2id(B, mujoco.mjtObj.mjOBJ_KEY, "stand")].copy()
    hyps = list(itertools.product([float(x) for x in a.dz_grid.split(",")], [float(x) for x in a.ank_grid.split(",")]))
    worlds = []
    for dz, ank in hyps:
        mm = mujoco.MjModel.from_xml_string(B._xml) if hasattr(B, "_xml") else None
    # cheaper: one model per dz (body_pos differs), ankle hypothesis = state transform
    dzs = sorted(set(h[0] for h in hyps)); models = {}
    for dz in dzs:
        mm = planner_model.load("planner_lean_s25_parity"); mm.body_pos[tbB, 2] += dz; models[dz] = (mm, mujoco.MjData(mm))
    ank_idx = [i for i, n in enumerate(namesB) if "ankle_pitch" in n]
    # ---- init: both at the keyframe pose (plant base follows the planner keyframe) ----
    q0 = stand if a.scenario == "lean" else kq
    pd.qpos[:] = q0; pd.qvel[:] = 0; mujoco.mj_forward(P, pd)
    # settle the plant 1 s under PD hold before the bank starts (contacts equilibrate)
    tgt = q0[qaP].copy()
    def reported(d):
        q = d.qpos.copy(); q[qaP[ank_idx]] += a.ank_true; return q
    def pd_step(d, tgt, n):
        for _ in range(n):
            qr = reported(d)[qaP]; d.ctrl[:] = KP * (tgt - qr) - KV * d.qvel[vaP]; mujoco.mj_step(P, d)
    pd_step(pd, tgt, int(a.settle / 0.002))
    wig = [i for i, n in enumerate(namesP) if n in ("torso_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_hip_pitch_joint", "right_hip_pitch_joint")]
    # ---- loop ----
    n_ticks = int(a.dur / TICK); alpha = TICK / a.window
    score = np.zeros(len(hyps)); lead_hist = []; true_h = (a.dz_true, a.ank_true)
    true_i = min(range(len(hyps)), key=lambda i: abs(hyps[i][0] - a.dz_true) * 100 + abs(hyps[i][1] - a.ank_true))
    W = np.ones(B.nq); W[:3] = 10.0; W[3:7] = 5.0   # pelvis pos / quat weighted; joints 1
    VW = 0.05
    if a.resid == "focused":   # contact-relevant only: pelvis pose + legs + bracing (left) arm, no velocity
        W[:] = 0.0; W[:3] = 10.0; W[3:7] = 5.0
        for i, n in enumerate(namesB):
            if any(k in n for k in ("hip_pitch", "knee", "ankle_pitch", "torso", "left_shoulder", "left_elbow")): W[qaB[i]] = 1.0
        VW = 0.0
    t0 = time.time()
    for k in range(n_ticks):
        t = k * TICK
        if a.scenario == "lean":
            s = min(1.0, t / a.lean_sec); tgt = (1 - s) * stand[qaP] + s * kq[qaP]
        elif a.scenario == "wiggle":
            tgt = kq[qaP].copy(); tgt[wig] += a.wiggle_amp * np.sin(2 * np.pi * a.wiggle_hz * t)
        q_rep = reported(pd); v_rep = pd.qvel.copy(); cmd = tgt.copy()
        # bank predictions from the reported state
        preds = []
        for (dz, ank) in hyps:
            mm, dd = models[dz]
            dd.qpos[:] = q_rep; dd.qpos[qaB[ank_idx]] -= ank; dd.qvel[:] = v_rep
            dd.ctrl[:] = cmd + 0.0; dd.ctrl[ank_idx] -= 0.0
            # the plant PD acts on REPORTED angles; the world's position servo acts on its (true-hypothesis)
            # angles, so shift the ankle targets by the hypothesis to command the same torque
            dd.ctrl[ank_idx] = cmd[ank_idx] - ank
            mujoco.mj_step(mm, dd)
            q_pred = dd.qpos.copy(); q_pred[qaB[ank_idx]] += ank   # back to reported frame
            preds.append((q_pred, dd.qvel.copy()))
        pd_step(pd, tgt, 5)
        q_next = reported(pd); v_next = pd.qvel
        for i, (qp, vp) in enumerate(preds):
            r = float(np.sum((W * (qp - q_next)) ** 2)) + VW ** 2 * float(np.sum((vp - v_next) ** 2))
            score[i] = (1 - alpha) * score[i] + alpha * r if k else r
        lead = int(np.argmin(score)); lead_hist.append(lead)
        if k % int(0.5 / TICK) == 0:
            srt = np.argsort(score)
            print(f"t={t:4.1f}s pelvis z {pd.qpos[2]:.3f}  leader dz {hyps[lead][0]*100:+.0f} cm ank {np.degrees(hyps[lead][1]):+.1f} deg  "
                  f"score {score[lead]:.2e}  2nd ({hyps[srt[1]][0]*100:+.0f},{np.degrees(hyps[srt[1]][1]):+.1f}) {score[srt[1]]:.2e}  true {score[true_i]:.2e}  nominal {score[hyps.index((0.0,0.0))]:.2e}", flush=True)
        if pd.qpos[2] < 0.5: print("plant FELL at", t); break
    lh = np.array(lead_hist); n_eval = len(lh) - int(0.5 / TICK)
    frac = float((lh[int(0.5 / TICK):] == true_i).mean()) if n_eval > 0 else float("nan")
    dz_ok = float(np.mean([abs(hyps[i][0] - a.dz_true) <= 0.005 for i in lh[int(0.5 / TICK):]])) if n_eval > 0 else float("nan")
    first = next((i * TICK for i in range(len(lh)) if lh[i] == true_i), None)
    print(f"RESULT scenario={a.scenario} dz_true={a.dz_true} ank_true={a.ank_true} K={len(hyps)}: true hypothesis leads {frac*100:.0f}% of ticks after 0.5 s "
          f"(dz within 0.5 cm: {dz_ok*100:.0f}%), first lead at {first}s, {time.time()-t0:.1f}s wall for {len(lh)} ticks")

if __name__ == "__main__":
    main()
