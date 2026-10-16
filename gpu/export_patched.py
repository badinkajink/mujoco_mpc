# Load the node's MJB (MuJoCo 3.2.3) and the saved XML in the SAME version, report which
# mjModel arrays differ, and export the patched arrays to npz for the Warp (3.15) side.
import sys, mujoco, numpy as np
mjb, xml, out = sys.argv[1:4]
a = mujoco.MjModel.from_binary_path(mjb)
b = mujoco.MjModel.from_xml_path(xml)
fields = ["actuator_gainprm","actuator_biasprm","actuator_forcerange","actuator_forcelimited",
          "actuator_ctrlrange","actuator_ctrllimited","jnt_range","jnt_limited","jnt_stiffness",
          "dof_damping","dof_armature","dof_frictionloss","geom_friction","geom_solref","geom_solimp",
          "geom_margin","geom_gap","geom_size","geom_pos","geom_quat","geom_condim","geom_contype","geom_conaffinity",
          "body_pos","body_quat","body_mass","body_inertia","qpos0","key_qpos","pair_geom1","pair_geom2","pair_solref","pair_friction"]
diff = {}
for f in fields:
    x, y = np.asarray(getattr(a, f)), np.asarray(getattr(b, f))
    if x.shape != y.shape or not np.allclose(x, y, atol=1e-9):
        n = int((~np.isclose(x, y, atol=1e-9)).sum()) if x.shape == y.shape else -1
        diff[f] = n
        print(f"DIFF {f}: shape {x.shape} vs {y.shape}, {n} entries differ")
print("opt mjb:", a.opt.timestep, a.opt.integrator, a.opt.cone, a.opt.solver, a.opt.iterations, a.opt.noslip_iterations, a.opt.impratio)
print("opt xml:", b.opt.timestep, b.opt.integrator, b.opt.cone, b.opt.solver, b.opt.iterations, b.opt.noslip_iterations, b.opt.impratio)
np.savez(out, **{f: np.asarray(getattr(a, f)) for f in fields}, opt_timestep=a.opt.timestep, opt_integrator=a.opt.integrator, opt_cone=a.opt.cone, opt_solver=a.opt.solver, opt_iterations=a.opt.iterations, opt_ls_iterations=a.opt.ls_iterations, opt_noslip=a.opt.noslip_iterations, opt_impratio=a.opt.impratio, opt_disableflags=a.opt.disableflags, opt_enableflags=a.opt.enableflags)
print("no diffs" if not diff else f"{len(diff)} differing fields", "-> saved", out)
