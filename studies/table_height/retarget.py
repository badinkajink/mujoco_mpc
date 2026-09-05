#!/usr/bin/env python3
"""Re-solve the brace posture keyframes for a slab at a different height.

The shipped `forearm_brace_lean` pose is a solution of ONE table height: it puts
the left forearm pad 2 mm above the compiled 0.985 m face, and the file's own
header records that it was hand-solved ("--surface 0.955 --min-x 0.34") the last
time the slab moved. Nothing re-solves it when the slab moves again, so at any
other height the Posture cost and the Brace Pos cost pull the pad to two
different places and the gap grows linearly with the height error.

This tool does that re-solve. For each brace rung it finds the pose nearest the
shipped keyframe that shifts the pad by exactly (face - nominal_face) in z while
holding both feet where and how the keyframe holds them. Delta = 0 reproduces the
keyframe exactly, so patching at the nominal height is a no-op.

usage:
  retarget.py --face 0.885 --report            # what would change
  retarget.py --face 0.885 --write             # patch the build-tree XML
  retarget.py --restore                        # put the shipped keys back
"""
import argparse, os, re, shutil, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "../.."))
DEFAULT_XML = os.path.join(
    ROOT, "build_cmake/mjpc/tasks/humanoid_bench/lean/Lean_H12_Magpie.xml")

# Base (6) + both legs (12) + torso yaw (1) + LEFT arm (7). The right arm is the
# reaching arm and is left exactly as authored; the object dofs are untouched.
SEL = list(range(0, 6)) + list(range(6, 18)) + [18] + list(range(19, 26))
KEYS = ("forearm_brace_lean", "forearm_brace_reach", "forearm_brace_release")
# Both brace pads are constrained, not just the elbow: seating the forearm pad
# alone leaves the wrist free to rotate, and the null-space pull then parks the
# hand 122 mm INSIDE a 0.785 m slab. Two points make the forearm rigid, which is
# what "flat on the table" means.
PAD = "left_forearm_pad"
PAD2 = "left_wrist_pad"
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def nid(m, t, n):
    return mujoco.mj_name2id(m, t, n)


def num(m, name, default=None):
    """Read a model <numeric> by name -- the same lookup lean.cc does."""
    i = nid(m, mujoco.mjtObj.mjOBJ_NUMERIC, name)
    if i < 0:
        return default
    a, n = int(m.numeric_adr[i]), int(m.numeric_size[i])
    v = m.numeric_data[a:a + n]
    return float(v[0]) if n == 1 else np.array(v)


def face_of(m):
    """Slab face z from the compiled geometry -- never a literal."""
    tb = nid(m, mujoco.mjtObj.mjOBJ_BODY, "table")
    tg = nid(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    return float(m.body_pos[tb][2] + m.geom_pos[tg][2] + m.geom_size[tg][2])


def quat_err(q_cur, q_tgt):
    neg = np.zeros(4); mujoco.mju_negQuat(neg, q_cur)
    dq = np.zeros(4);  mujoco.mju_mulQuat(dq, q_tgt, neg)
    v = np.zeros(3);   mujoco.mju_quat2Vel(v, dq, 1.0)
    return v


def solve(m, q0, target_pad, target_pad2, iters=600, damp=1e-3, null_k=0.02):
    """Gauss-Newton on [both pads xyz, both feet 6-dof], null-space pull to q0."""
    d = mujoco.MjData(m)
    pad = nid(m, mujoco.mjtObj.mjOBJ_GEOM, PAD)
    pad2 = nid(m, mujoco.mjtObj.mjOBJ_GEOM, PAD2)
    feet = [nid(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in FEET]
    d.qpos[:] = q0
    mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
    ref = [(d.xpos[b].copy(), d.xquat[b].copy()) for b in feet]
    jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    r = np.zeros(18)
    for _ in range(iters):
        # mj_jac* read d.cdof, which mj_comPos fills -- kinematics alone leaves the
        # Jacobian identically zero and the solver silently returns the input pose.
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        res = [d.geom_xpos[pad] - target_pad]
        mujoco.mj_jacGeom(m, d, jp, jr, pad)
        J = [jp[:, SEL].copy()]
        res.append(d.geom_xpos[pad2] - target_pad2)
        mujoco.mj_jacGeom(m, d, jp, jr, pad2)
        J.append(jp[:, SEL].copy())
        for k, b in enumerate(feet):
            res.append(d.xpos[b] - ref[k][0])
            res.append(-quat_err(d.xquat[b], ref[k][1]))
            mujoco.mj_jacBody(m, d, jp, jr, b)
            J.append(jp[:, SEL].copy()); J.append(jr[:, SEL].copy())
        r = np.concatenate(res); Jm = np.vstack(J)
        if np.max(np.abs(r)) < 1e-7:
            break
        JJt = Jm @ Jm.T + damp * np.eye(Jm.shape[0])
        dq_task = -Jm.T @ np.linalg.solve(JJt, r)
        e = np.zeros(m.nv); mujoco.mj_differentiatePos(m, e, 1.0, q0, d.qpos)
        proj = np.eye(len(SEL)) - Jm.T @ np.linalg.solve(JJt, Jm)
        dq = dq_task + proj @ (-null_k * e[SEL])
        full = np.zeros(m.nv); full[SEL] = np.clip(dq, -0.15, 0.15)
        mujoco.mj_integratePos(m, d.qpos, full, 1.0)
        for j in range(1, m.njnt):
            if m.jnt_limited[j]:
                a = m.jnt_qposadr[j]
                d.qpos[a] = np.clip(d.qpos[a], m.jnt_range[j][0], m.jnt_range[j][1])
    return d.qpos.copy(), float(np.max(np.abs(r)))


def diagnose(m, q, face):
    """Sanity report for one retargeted pose, with the slab at `face`."""
    m2 = mujoco.MjModel.from_xml_path(m_path_of(m))
    tb = nid(m2, mujoco.mjtObj.mjOBJ_BODY, "table")
    tg = nid(m2, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    m2.body_pos[tb][2] = face - (m2.geom_pos[tg][2] + m2.geom_size[tg][2])
    d = mujoco.MjData(m2)
    d.qpos[:] = q
    mujoco.mj_forward(m2, d)
    pad = nid(m2, mujoco.mjtObj.mjOBJ_GEOM, PAD)
    padr = float(m2.geom_size[pad][0])
    lf = d.xpos[nid(m2, mujoco.mjtObj.mjOBJ_BODY, FEET[0])]
    rf = d.xpos[nid(m2, mujoco.mjtObj.mjOBJ_BODY, FEET[1])]
    mid = 0.5 * (lf[0] + rf[0])
    com = d.subtree_com[nid(m2, mujoco.mjtObj.mjOBJ_BODY, "pelvis")][0]
    worst, worst_j = 1e9, ""
    for j in range(1, m2.njnt):
        if not m2.jnt_limited[j] or int(m2.jnt_dofadr[j]) not in SEL:
            continue
        a = m2.jnt_qposadr[j]
        lo, hi = m2.jnt_range[j]
        marg = min(q[a] - lo, hi - q[a])
        if marg < worst:
            worst, worst_j = marg, mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_JOINT, j)
    touching, deepest = {}, 0.0
    for c in range(d.ncon):
        b1 = m2.geom_bodyid[d.contact[c].geom1]
        b2 = m2.geom_bodyid[d.contact[c].geom2]
        for a, b in ((b1, b2), (b2, b1)):
            if a == tb and b != 0:
                nm = mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_BODY, b)
                if nm == "object":      # the keyframe's stale cargo pose, not the robot
                    continue
                pen = -float(d.contact[c].dist)
                touching[nm] = max(touching.get(nm, 0.0), pen)
                deepest = max(deepest, pen)
    sh = d.xpos[nid(m2, mujoco.mjtObj.mjOBJ_BODY, "left_shoulder_yaw_link")]
    return dict(sh_above=float(sh[2] - face),
                pad_clear=float(d.geom_xpos[pad][2] - padr - face),
                com_mid=float(com - mid), lim_margin=float(worst),
                lim_joint=worst_j, touching=touching, deepest=deepest)


_MPATH = {}
def m_path_of(m):
    return _MPATH[id(m)]


def load(path):
    m = mujoco.MjModel.from_xml_path(path)
    _MPATH[id(m)] = path
    return m


def patch(path, new_q, backup=True):
    """Rewrite the active <key ...> lines. Commented-out twins are left alone."""
    orig = path + ".orig"
    if backup and not os.path.exists(orig):
        shutil.copy2(path, orig)
    lines = open(path).read().split("\n")
    done = {}
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s.startswith("<key ") or s.endswith("-->"):
            continue
        mt = re.match(r'<key\s+name="([^"]+)"', s)
        if not mt or mt.group(1) not in new_q or mt.group(1) in done:
            continue
        name = mt.group(1)
        q = " ".join("%.6f" % v for v in new_q[name])
        lines[i] = re.sub(r'qpos="[^"]*"', 'qpos="%s"' % q, ln)
        done[name] = i + 1
    open(path, "w").write("\n".join(lines))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face", type=float)
    ap.add_argument("--xml", default=DEFAULT_XML)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--keys", default=",".join(KEYS))
    a = ap.parse_args()

    if a.restore:
        orig = a.xml + ".orig"
        if not os.path.exists(orig):
            sys.exit("no backup at " + orig)
        shutil.copy2(orig, a.xml)
        os.remove(orig)
        print("restored", a.xml)
        return
    if a.face is None:
        sys.exit("--face is required unless --restore")

    src = a.xml + ".orig" if os.path.exists(a.xml + ".orig") else a.xml
    m = load(src)
    nom = face_of(m)
    delta = a.face - nom
    pad = nid(m, mujoco.mjtObj.mjOBJ_GEOM, PAD)
    pad2 = nid(m, mujoco.mjtObj.mjOBJ_GEOM, PAD2)
    print("nominal face %.3f m ; requested %.3f m ; delta %+.0f mm"
          % (nom, a.face, 1000 * delta))

    new_q = {}
    for kn in a.keys.split(","):
        ki = nid(m, mujoco.mjtObj.mjOBJ_KEY, kn)
        if ki < 0:
            print("  %-24s MISSING -- skipped" % kn); continue
        q0 = m.key_qpos[ki].copy()
        d = mujoco.MjData(m); d.qpos[:] = q0; mujoco.mj_kinematics(m, d)
        tgt = d.geom_xpos[pad].copy(); tgt[2] += delta
        tgt2 = d.geom_xpos[pad2].copy(); tgt2[2] += delta
        q, err = solve(m, q0, tgt, tgt2)
        dg = diagnose(m, q, a.face)
        new_q[kn] = q
        e = np.zeros(m.nv); mujoco.mj_differentiatePos(m, e, 1.0, q0, q)
        qq = q[3:7]
        pitch = np.degrees(np.arcsin(np.clip(2*(qq[0]*qq[2]-qq[3]*qq[1]), -1, 1)))
        print("  %-22s ik %5.2f mm | pitch %6.2f deg | base_z %.3f | "
              "hipL %6.1f deg | pad_clear %+5.1f mm | CoM-mid %+5.1f mm | "
              "shoulder %+5.0f mm | limit %.3f rad (%s)"
              % (kn, 1000*err, pitch, q[2], np.degrees(q[8]),
                 1000*dg["pad_clear"], 1000*dg["com_mid"], 1000*dg["sh_above"],
                 dg["lim_margin"], dg["lim_joint"]))
        if dg["touching"]:
            print("     slab contacts: " + ", ".join(
                "%s %.0f mm" % (k, 1000 * v)
                for k, v in sorted(dg["touching"].items(), key=lambda kv: -kv[1])))
        base = diagnose(m, q0, nom)
        if dg["lim_margin"] < 0.02 and dg["lim_margin"] < base["lim_margin"] - 1e-6:
            print("     ⚠ the retarget tightened a joint limit: %.3f -> %.3f rad (%s)"
                  % (base["lim_margin"], dg["lim_margin"], dg["lim_joint"]))

    if not a.write:
        print("(report only -- pass --write to patch)")
        return
    new_q = {k: q for k, q in new_q.items()
             if np.max(np.abs(q - m.key_qpos[nid(m, mujoco.mjtObj.mjOBJ_KEY, k)]))
             > 1e-12}
    if not new_q:
        print("delta is zero -- every key already solves this height; file untouched")
        return
    done = patch(a.xml, new_q)
    chk = mujoco.MjModel.from_xml_path(a.xml)
    for kn, q in new_q.items():
        ki = nid(chk, mujoco.mjtObj.mjOBJ_KEY, kn)
        if ki < 0 or np.max(np.abs(chk.key_qpos[ki] - q)) > 1e-5:
            sys.exit("VERIFY FAILED for %s -- the reloaded model does not carry "
                     "the patched pose" % kn)
    print("patched and verified:", ", ".join("%s@L%d" % (k, v) for k, v in done.items()))


if __name__ == "__main__":
    main()
