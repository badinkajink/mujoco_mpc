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
lean's residual stays valid; the model->nu-coupled cost terms shrink to 12.

usage: _gen_stabilize_model.py <input_magpie.xml> <output_stabilize.xml>
"""
import re
import sys

LEG_KEYS = ("hip", "knee", "ankle")


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

    with open(dst, "w") as f:
        f.write(xml)
    print(f"[gen_stabilize] kept {n_kept} leg actuators, "
          f"locked {len(locked_joints)} upper joints -> {dst}")


if __name__ == "__main__":
    main()
