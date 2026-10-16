"""Load the deployed planner model (as the node patches it) into pip MuJoCo.

The node (MuJoCo commit 088079e, header 3.2.3) dumps <name>.xml via mj_saveLastXML and
<name>.arrays.txt with the mjModel arrays it patched (PatchActuators, estop margin, URDF
arm ranges) plus the geometry/contact arrays. We load the XML in the pip MuJoCo and
overwrite those arrays, so the Warp copy plans with exactly the node's numbers.
"""
import os, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")

INT_FIELDS = {"actuator_trnid", "geom_condim", "geom_contype", "geom_conaffinity"}
BYTE_FIELDS = {"actuator_forcelimited", "actuator_ctrllimited", "jnt_limited"}


def read_arrays(path):
    arrays, opt = {}, {}
    with open(path) as f:
        for line in f:
            tok = line.split()
            if tok[0] == "opt":
                for k, v in zip(tok[1::2], tok[2::2]):
                    opt[k] = float(v) if "." in v or "e" in v else int(v)
                continue
            name, n = tok[0], int(tok[1])
            vals = tok[2:]
            assert len(vals) == n, (name, n, len(vals))
            if name in INT_FIELDS or name in BYTE_FIELDS:
                arrays[name] = np.array([int(v) for v in vals], dtype=np.int32)
            else:
                arrays[name] = np.array([float(v) for v in vals])
    return arrays, opt


def load(variant="planner_lean_s25_parity", agent_timestep=True, verbose=False, warp=False):
    """warp=True: disable NATIVECCD + MULTICCD. The node links MuJoCo 3.2.3 (legacy box-box
    and libccd colliders, no multi-contact CCD), and mujoco_warp refuses non-zero geom
    margins (gripper keepaway 0.025) under those newer colliders. Legacy = closest to CPU."""
    xml = os.path.join(MODELS, variant + ".xml")
    # mj_saveLastXML keeps texture paths relative to the task dir; meshes are absolute.
    assets_dir = os.path.abspath(os.path.join(HERE, "..", "build", "mjpc", "tasks", "humanoid_bench", "assets"))
    text = open(xml).read().replace('file="../assets/', f'file="{assets_dir}/')
    m = mujoco.MjModel.from_xml_string(text)
    arrays, opt = read_arrays(os.path.join(MODELS, variant + ".arrays.txt"))
    diffs = {}
    for name, v in arrays.items():
        tgt = getattr(m, name)
        cur = np.asarray(tgt).reshape(-1)
        if cur.shape != v.shape:
            raise RuntimeError(f"{name}: node {v.shape} vs pip {cur.shape}")
        nd = int((~np.isclose(cur, v, atol=1e-9)).sum())
        if nd:
            diffs[name] = nd
            np.asarray(tgt).reshape(-1)[:] = v.astype(np.asarray(tgt).dtype)
    for k in ("integrator", "cone", "solver", "iterations", "ls_iterations", "noslip_iterations",
              "impratio", "disableflags", "enableflags", "jacobian", "tolerance"):
        if getattr(m.opt, k) != opt[k]:
            diffs["opt." + k] = (getattr(m.opt, k), opt[k])
            setattr(m.opt, k, opt[k])
    m.opt.timestep = opt["timestep"]
    if warp:
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD) | int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    if agent_timestep:
        nid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, "agent_timestep")
        m.opt.timestep = float(m.numeric_data[m.numeric_adr[nid]])  # agent.cc PlanIteration
    if verbose:
        print(f"[planner_model] {variant}: pip mujoco {mujoco.__version__}, nq {m.nq} nv {m.nv} nu {m.nu} "
              f"ngeom {m.ngeom}, dt {m.opt.timestep}, arrays overwritten: {diffs if diffs else 'none (XML round-trip exact)'}")
    return m


if __name__ == "__main__":
    for v in sys.argv[1:] or ["planner_lean_s25", "planner_lean_s25_parity"]:
        m = load(v, verbose=True)
        o = m.opt
        print("  opt: integrator", o.integrator, "cone", o.cone, "solver", o.solver, "iter", o.iterations,
              "ls", o.ls_iterations, "noslip", o.noslip_iterations, "impratio", o.impratio, "jacobian", o.jacobian)
        gt = np.bincount(m.geom_type, minlength=8)
        print("  geoms by type [plane hfield sphere capsule ellipsoid cylinder box mesh]:", gt.tolist(),
              "| nmesh", m.nmesh, "faces", int(m.mesh_face.shape[0]), "| npair", m.npair, "nexclude", m.nexclude,
              "| nplugin", m.nplugin, "ntendon", m.ntendon, "neq", m.neq, "nsensor", m.nsensor, "nkey", m.nkey)
        print("  max condim", int(m.geom_condim.max()), "| keys", [m.key(i).name for i in range(m.nkey)][:12])
