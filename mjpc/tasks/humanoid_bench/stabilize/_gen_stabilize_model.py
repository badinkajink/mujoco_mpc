#!/usr/bin/env python3
"""Generate the legs-only (nu=12) stabilize H1-2 robot from the full magpie robot.

Build step (mirrors the repo's gen_lean_pipeline pattern). Reads the
build-generated h1_2_modified_magpie.xml (27 <position> actuators) and writes
h1_2_stabilize_magpie.xml which:

  * keeps ONLY the 12 leg actuators (joint name contains hip / knee / ankle), and
  * equality-locks the 15 removed upper-body joints (torso + 14 arms) at home
    (qpos = 0, the default <joint> equality polycoef[0]) so the upper body stays
    rigid while the planner controls the lower body only.

The 27-DOF joint layout is preserved (only actuators are dropped + locked), so
the stabilize residual's qpos indexing stays valid; the model->nu-coupled cost
terms shrink to 12.

usage: _gen_stabilize_model.py <input_magpie.xml> <output_stabilize.xml>
"""
import re
import sys

LEG_KEYS = ("hip", "knee", "ankle")

# FOOT_SPHERES = False: the MESH sole is the collision sole (current default;
# the sphere-sole experiment was reverted). Set True to re-run the R2
# experiment: 4 corner spheres per foot (G1 Menagerie pattern, H1-2
# dimensions), mesh demoted to visual, spheres inheriting condim 4 + friction
# 1/0.06 with bottoms on the mesh sole plane so standing height is unchanged.
# (history + the sphere-vs-mesh rationale: see
# mjpc/tasks/humanoid_bench/HISTORY.md)
FOOT_SPHERES = False
_SPHERE_XY = [(-0.065, 0.038), (-0.065, -0.038), (0.15, 0.038), (0.15, -0.038)]


def inject_foot_spheres(xml):
    for side in ("left", "right"):
        pat = re.compile(
            r'^(\s*)<geom (?![^>]*contype="0")[^>]*mesh="%s_ankle_roll_link"'
            r'[^>]*/>[ \t]*$' % side, re.M)
        m = pat.search(xml)
        if not m:
            raise SystemExit(f"no colliding {side} ankle_roll sole geom found")
        ind = m.group(1)
        demoted = m.group(0).replace(
            "<geom ", '<geom contype="0" conaffinity="0" ', 1)
        spheres = "\n".join(
            f'{ind}<geom name="{side}_foot_c{i}" type="sphere" size="0.005" '
            f'pos="{x} {y} -0.040" contype="1" conaffinity="1" priority="1" '
            f'condim="4" friction="1 0.06 0.0001" group="3" '
            f'rgba="0.8 0.2 0.2 1"/>'
            for i, (x, y) in enumerate(_SPHERE_XY))
        rep = (demoted + "\n" + ind +
               "<!-- R2: sole = 4 corner spheres; mesh demoted to visual -->\n"
               + spheres)
        xml = xml[:m.start()] + rep + xml[m.end():]
    return xml


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: _gen_stabilize_model.py <in.xml> <out.xml>")
    src, dst = sys.argv[1], sys.argv[2]
    with open(src) as f:
        xml = f.read()

    m = re.search(r"<actuator>(.*?)</actuator>", xml, re.DOTALL)
    if not m:
        raise SystemExit("no <actuator> block in " + src)

    kept_lines, locked_joints = [], []
    for ln in m.group(1).splitlines():
        if "<position" not in ln:
            kept_lines.append(ln)  # whitespace / non-actuator content
            continue
        jm = re.search(r'joint="([^"]+)"', ln)
        joint = jm.group(1) if jm else ""
        if any(k in joint for k in LEG_KEYS):
            kept_lines.append(ln)          # keep the 12 leg actuators
        else:
            locked_joints.append(joint)    # torso + arms: drop actuator, lock joint

    n_kept = sum(1 for ln in kept_lines if "<position" in ln)
    if n_kept != 12:
        raise SystemExit(
            f"expected 12 leg actuators, kept {n_kept} -- check joint naming")

    new_actuator = "<actuator>" + "\n".join(kept_lines) + "</actuator>"
    xml = xml[: m.start()] + new_actuator + xml[m.end():]

    # Equality-lock every removed (upper-body) joint at home (qpos = 0).
    eq = ["  <!-- nu=12 stabilize: torso + arms locked at home (qpos=0). -->",
          "  <equality>"]
    eq += [f'    <joint joint1="{j}"/>' for j in locked_joints]
    eq += ["  </equality>", ""]
    eq_block = "\n".join(eq)

    idx = xml.rstrip().rfind("</mujoco>")
    if idx < 0:
        raise SystemExit("no </mujoco> in " + src)
    xml = xml[:idx] + eq_block + xml[idx:]

    if FOOT_SPHERES:
        xml = inject_foot_spheres(xml)

    with open(dst, "w") as f:
        f.write(xml)
    print(f"[gen_stabilize] kept {n_kept} leg actuators, "
          f"locked {len(locked_joints)} upper joints -> {dst}")


if __name__ == "__main__":
    main()
