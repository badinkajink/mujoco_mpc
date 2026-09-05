#!/usr/bin/env python3
"""Report the brace geometry lean.cc computes, as a function of table height.

No physics, no planner: this is the same arithmetic `lean::Residual` does, run
offline so the height dependence of every brace target is visible in one table.
Every number is read from the compiled model -- no literal is repeated here.
"""
import os, sys, math
import numpy as np
import mujoco

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
MODEL = os.path.join(ROOT, "build_cmake/mjpc/tasks/humanoid_bench/lean/Lean_H12_Magpie.xml")

def num(m, name, default=None):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, name)
    if i < 0:
        return default
    a = m.numeric_adr[i]
    n = m.numeric_size[i]
    v = m.numeric_data[a:a+n]
    return float(v[0]) if n == 1 else np.array(v)

def set_table_height(model, face_z):
    tb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    tg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    slab_off = model.geom_pos[tg][2] + model.geom_size[tg][2]
    want = face_z - slab_off
    dz = want - model.body_pos[tb][2]
    model.body_pos[tb][2] = want
    under = model.geom_pos[tg][2] - model.geom_size[tg][2]
    for n_ in ("table_leg_1","table_leg_2","table_leg_3","table_leg_4",
               "table_leg_1_collision","table_leg_2_collision",
               "table_leg_3_collision","table_leg_4_collision"):
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n_)
        if g < 0: continue
        half = max(0.01, 0.5*(under+want))
        model.geom_size[g][2] = half
        model.geom_pos[g][2] = under - half
    tm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    if tm >= 0 and model.body_mocapid[tm] >= 0:
        model.body_pos[tm][2] += dz
    return dz

m = mujoco.MjModel.from_xml_path(MODEL)
d = mujoco.MjData(m)

kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "forearm_brace_lean")
print("keyframe forearm_brace_lean id", kid)

def gid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
def bid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
def sid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)

# --- static geometry of the slab, from the compiled model ---
tg = gid("table_top"); tgc = gid("table_top_collision")
print("table_top    pos", m.geom_pos[tg], "size", m.geom_size[tg])
print("table_top_col pos", m.geom_pos[tgc], "size", m.geom_size[tgc])
tb = bid("table")
print("table body pos", m.body_pos[tb])
for g in ("left_forearm_pad","left_wrist_pad"):
    i = gid(g)
    print(g, "body", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]),
          "size", m.geom_size[i], "pos", m.geom_pos[i], "type", m.geom_type[i])

print("\nnumerics:", {k: num(m,k) for k in
      ("brace_press_depth","brace_target_face","brace_target_slab","brace_target_inset",
       "brace_target_slab_lat","brace_lat_inset","com_cap_fwd","pelvis_cap_fwd",
       "brace_lead_x0","brace_lead_gain","reach_radius","reach_drop","lean_nominal_x",
       "brace_erect_target","stance_off_x","table_clear_margin")})

hs = [0.735,0.785,0.835,0.885,0.935,0.985,1.035,1.085,1.135]
print("\n%-7s %8s %8s %8s %8s %8s %8s %8s %8s %8s" % (
    "face","tbl_ctr","near_ed","bx_tgt","by_tgt","bz_tgt","pad_x","pad_z","dx","dz"))
base = mujoco.MjModel.from_xml_path(MODEL)
for h in hs:
    m2 = mujoco.MjModel.from_xml_path(MODEL)
    d2 = mujoco.MjData(m2)
    set_table_height(m2, h)
    mujoco.mj_resetDataKeyframe(m2, d2, kid)
    mujoco.mj_forward(m2, d2)
    tgc2 = gid(  "table_top_collision")
    tctr = d2.geom_xpos[tgc2]
    half = m2.geom_size[tgc2]
    near_edge = tctr[0] - half[0]
    face = tctr[2] + half[2]
    torso = d2.xpos[bid("torso_link")]
    bx = torso[0] + 0.4*(tctr[0]-torso[0])       # brace_target_slab OFF
    by = tctr[1] + num(m2,"brace_lat_inset")     # reach_right -> +lat  (brace = LEFT arm)
    bz = face - num(m2,"brace_press_depth")      # brace_target_face ON
    pg = gid("left_forearm_pad")
    pad = d2.geom_xpos[pg]
    print("%-7.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f" % (
        h, tctr[0], near_edge, bx, by, bz, pad[0], pad[2], bx-pad[0], bz-pad[2]))
