#!/usr/bin/env python3
"""What posture would seat the brace pad on a slab at height h?

For each table height, solves the pose NEAREST the shipped `forearm_brace_lean`
keyframe that (a) puts the `left_forearm_pad` capsule centre one pad radius above
the slab at the keyframe's own pad x/y, and (b) leaves both feet exactly where
and how the keyframe puts them (flat, same stance). Then asks whether the family
q(h) is one-dimensional and whether its direction is already spanned by the
keyframes the model ships.

Kinematics only: no planner, no contact, no dynamics. It separates "the
controller is badly tuned" from "the controller is being told to hold a pose that
cannot reach the slab".
"""
import os, sys
import numpy as np
import mujoco

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
MODEL = os.path.join(ROOT, "build_cmake/mjpc/tasks/humanoid_bench/lean/Lean_H12_Magpie.xml")

BASE_DOF   = list(range(0, 6))
LEG_DOF    = list(range(6, 18))
TORSO_DOF  = [18]
LARM_DOF   = list(range(19, 26))
SEL = BASE_DOF + LEG_DOF + TORSO_DOF + LARM_DOF          # 26 dofs

def name2id(m, t, n): return mujoco.mj_name2id(m, t, n)

def set_table_height(model, face_z):
    tb = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    tg = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    slab_off = model.geom_pos[tg][2] + model.geom_size[tg][2]
    want = face_z - slab_off
    model.body_pos[tb][2] = want

def quat_err(q_cur, q_tgt):
    """rotation vector taking q_cur to q_tgt, in the world frame."""
    neg = np.zeros(4); mujoco.mju_negQuat(neg, q_cur)
    dq  = np.zeros(4); mujoco.mju_mulQuat(dq, q_tgt, neg)
    v   = np.zeros(3); mujoco.mju_quat2Vel(v, dq, 1.0)
    return v

def solve(m, d, q0, target_pad, foot_ref, iters=400, damp=1e-3, null_k=0.02):
    """Gauss-Newton with a null-space pull back toward q0."""
    pad = name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "left_forearm_pad")
    feet = [name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
            for b in ("left_ankle_roll_link", "right_ankle_roll_link")]
    d.qpos[:] = q0
    jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    for it in range(iters):
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        res = [d.geom_xpos[pad] - target_pad]
        mujoco.mj_jacGeom(m, d, jp, jr, pad)
        J = [jp[:, SEL].copy()]
        for k, b in enumerate(feet):
            res.append(d.xpos[b] - foot_ref[k][0])
            res.append(-quat_err(d.xquat[b], foot_ref[k][1]))
            mujoco.mj_jacBody(m, d, jp, jr, b)
            J.append(jp[:, SEL].copy()); J.append(jr[:, SEL].copy())
        r = np.concatenate(res); Jm = np.vstack(J)
        if np.max(np.abs(r)) < 1e-6:
            break
        # damped least squares + null-space bias to q0
        JJt = Jm @ Jm.T + damp * np.eye(Jm.shape[0])
        dq_task = -Jm.T @ np.linalg.solve(JJt, r)
        dqerr = np.zeros(m.nv); mujoco.mj_differentiatePos(m, dqerr, 1.0, q0, d.qpos)
        bias = -null_k * dqerr[SEL]
        Jpinv_J = Jm.T @ np.linalg.solve(JJt, Jm)
        dq = dq_task + (np.eye(len(SEL)) - Jpinv_J) @ bias
        full = np.zeros(m.nv); full[SEL] = np.clip(dq, -0.15, 0.15)
        mujoco.mj_integratePos(m, d.qpos, full, 1.0)
        # joint limits
        for j in range(1, m.njnt):
            if m.jnt_limited[j]:
                a = m.jnt_qposadr[j]
                d.qpos[a] = np.clip(d.qpos[a], m.jnt_range[j][0], m.jnt_range[j][1])
    mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
    return d.qpos.copy(), float(np.max(np.abs(r)))

def main():
    m = mujoco.MjModel.from_xml_path(MODEL)
    d = mujoco.MjData(m)
    kb = name2id(m, mujoco.mjtObj.mjOBJ_KEY, "forearm_brace_lean")
    q_brace = m.key_qpos[kb].copy()
    pad = name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "left_forearm_pad")
    pad_r = float(m.geom_size[pad][0])
    tgc = name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")

    d.qpos[:] = q_brace; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
    pad0 = d.geom_xpos[pad].copy()
    foot_ref = [(d.xpos[name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)].copy(),
                 d.xquat[name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)].copy())
                for b in ("left_ankle_roll_link", "right_ankle_roll_link")]
    com0 = d.subtree_com[name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")].copy()
    mid_x = 0.5*(foot_ref[0][0][0] + foot_ref[1][0][0])
    print("pad radius %.4f m ; keyframe pad %s" % (pad_r, np.round(pad0, 4)))
    print("keyframe CoM x %.4f, midfoot x %.4f, CoM-midfoot %+.1f mm"
          % (com0[0], mid_x, 1000*(com0[0]-mid_x)))
    face0 = 0.985
    print("keyframe pad z - (face0 + pad_r) = %+.1f mm\n"
          % (1000*(pad0[2] - (face0 + pad_r))))

    hs = np.round(np.arange(0.785, 1.1401, 0.025), 3)
    Q = {}
    print("%-7s %7s %7s %8s %8s %8s %8s %8s %8s %8s" % (
        "face","err_mm","base_z","pitch_d","hipL_d","knee_d","ankL_d","shoP_d","elb_d","com_mm"))
    for h in hs:
        m2 = mujoco.MjModel.from_xml_path(MODEL); set_table_height(m2, h)
        d2 = mujoco.MjData(m2)
        tgt = np.array([pad0[0], pad0[1], h + pad_r])
        q, err = solve(m2, d2, q_brace, tgt, foot_ref)
        Q[float(h)] = q.copy()
        d2.qpos[:] = q; mujoco.mj_kinematics(m2, d2); mujoco.mj_comPos(m2, d2)
        com = d2.subtree_com[name2id(m2, mujoco.mjtObj.mjOBJ_BODY, "pelvis")]
        qq = q[3:7]; sinp = 2*(qq[0]*qq[2]-qq[3]*qq[1])
        pitch = np.degrees(np.arcsin(np.clip(sinp, -1, 1)))
        deg = np.degrees
        print("%-7.3f %7.2f %7.3f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.1f" % (
            h, 1000*err, q[2], pitch, deg(q[8]), deg(q[10]), deg(q[11]),
            deg(q[20]), deg(q[23]), 1000*(com[0]-mid_x)))
    # --- how many dimensions does the family actually use? ---
    A = np.array([Q[float(h)] for h in hs])
    dv = []
    for h in hs:
        e = np.zeros(m.nv); mujoco.mj_differentiatePos(m, e, 1.0, q_brace, Q[float(h)])
        dv.append(e[SEL])
    D = np.array(dv)
    U, S, Vt = np.linalg.svd(D - D.mean(0), full_matrices=False)
    print("\nSVD of q(h)-q_brace (mean removed): singular values")
    print(np.round(S[:5], 4), " -> first mode explains %.2f%% of variance"
          % (100*S[0]**2/np.sum(S**2)))
    print("mean offset norm %.4f" % np.linalg.norm(D.mean(0)))

    # first mode, named
    v = Vt[0] / np.linalg.norm(Vt[0])
    lbl = {}
    for j in range(1, m.njnt):
        lbl[int(m.jnt_dofadr[j])] = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    for i in range(6): lbl[i] = ["base_x","base_y","base_z","base_rx","base_ry","base_rz"][i]
    order = np.argsort(-np.abs(v))
    print("\nfirst mode, largest components:")
    for k in order[:12]:
        print("   %-28s %+7.3f" % (lbl[SEL[k]], v[k]))

    # --- is that direction already in the model's keyframe library? ---
    print("\ncosine of the first mode with (key - forearm_brace_lean):")
    for kn in ("stand_up","forearm_brace_mid","forearm_brace_release","standback_r1",
               "standback_r2","standback_r3","standback_squat","crouch","straighten",
               "home","stand","squat_mid"):
        ki = name2id(m, mujoco.mjtObj.mjOBJ_KEY, kn)
        if ki < 0: continue
        e = np.zeros(m.nv)
        mujoco.mj_differentiatePos(m, e, 1.0, q_brace, m.key_qpos[ki])
        w = e[SEL]
        if np.linalg.norm(w) < 1e-9: continue
        print("   %-24s  cos %+0.3f   |w| %.3f" % (kn, float(v @ w/np.linalg.norm(w)),
                                                   np.linalg.norm(w)))

if __name__ == "__main__":
    main()
