import sys, mujoco, numpy as np
m = mujoco.MjModel.from_binary_path(sys.argv[1])
o = m.opt
print("nq nv nu ngeom nmesh nbody njnt:", m.nq, m.nv, m.nu, m.ngeom, m.nmesh, m.nbody, m.njnt)
print("timestep", o.timestep, "integrator", o.integrator, "cone", o.cone, "solver", o.solver,
      "iterations", o.iterations, "ls_iter", o.ls_iterations, "noslip", o.noslip_iterations, "impratio", o.impratio, "jacobian", o.jacobian)
print("disableflags", o.disableflags, "enableflags", o.enableflags)
print("nplugin", m.nplugin, "nflex", m.nflex, "nhfield", m.nhfield, "ntendon", m.ntendon, "neq", m.neq, "nsensor", m.nsensor, "nuser_sensor", m.nuser_sensor, "nexclude", m.nexclude, "npair", m.npair)
gt = np.bincount(m.geom_type, minlength=8); print("geom types (plane,hfield,sphere,capsule,ellipsoid,cylinder,box,mesh):", gt)
print("mesh faces total", int(m.mesh_face.shape[0]), "max geom condim", int(m.geom_condim.max()))
print("actuator gaintypes", np.unique(m.actuator_gaintype), "biastypes", np.unique(m.actuator_biastype), "dyntypes", np.unique(m.actuator_dyntype))
print("jnt types", np.bincount(m.jnt_type, minlength=4))
nid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, "agent_timestep"); print("agent_timestep", m.numeric_data[m.numeric_adr[nid]])
print("keys", [m.key(i).name for i in range(m.nkey)])
print("sensors", [m.sensor(i).name for i in range(m.nsensor)][:60])
