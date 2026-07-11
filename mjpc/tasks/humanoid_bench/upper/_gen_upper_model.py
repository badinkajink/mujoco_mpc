#!/usr/bin/env python3
"""Generate the upper-body-only (nu=15) H1-2 robot from the full magpie robot.

Mirror of stabilize/_gen_stabilize_model.py, inverted (plan doc
h12/upper_body_mpc_controller_plan_2026-07-09.md P1). Reads the build-generated
h1_2_modified_magpie.xml (27 <position> actuators, gripperless deploy base) and
writes h1_2_upper_magpie.xml which:

  * keeps ONLY the 15 upper actuators (torso + 14 arms; joint name does NOT
    contain hip/knee/ankle),
  * equality-locks the 12 removed leg joints (upper::TransitionLocked
    retargets these to the MEASURED leg pose every plan -- "leg_aware",
    the mirror of stabilize's arm_aware F1-A), and
  * welds the pelvis to the world (v1 simplification: the LEG MPC owns
    balance; coupling closes through the arm_plan preview seam, not through
    this model). TransitionLocked retargets the weld to the measured base
    pose each plan.

usage: _gen_upper_model.py <input_magpie.xml> <output_upper.xml>
"""
import re
import sys

LEG_KEYS = ("hip", "knee", "ankle")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: _gen_upper_model.py <in.xml> <out.xml>")
    src, dst = sys.argv[1], sys.argv[2]
    with open(src) as f:
        xml = f.read()

    m = re.search(r"<actuator>(.*?)</actuator>", xml, re.DOTALL)
    if not m:
        raise SystemExit("no <actuator> block in " + src)

    kept_lines, locked_joints = [], []
    for ln in m.group(1).splitlines():
        if "<position" not in ln:
            kept_lines.append(ln)
            continue
        jm = re.search(r'joint="([^"]+)"', ln)
        joint = jm.group(1) if jm else ""
        if any(k in joint for k in LEG_KEYS):
            locked_joints.append(joint)    # legs: drop actuator, lock joint
        else:
            kept_lines.append(ln)          # keep torso + 14 arm actuators

    n_kept = sum(1 for ln in kept_lines if "<position" in ln)
    if n_kept != 15:
        raise SystemExit(
            f"expected 15 upper actuators, kept {n_kept} -- check joint naming")

    new_actuator = "<actuator>" + "\n".join(kept_lines) + "</actuator>"
    xml = xml[: m.start()] + new_actuator + xml[m.end():]

    eq = ["  <!-- nu=15 upper: legs locked (leg_aware retargets to measured",
          "       pose each plan) + pelvis welded to world (v1; retargeted to",
          "       the measured base pose each plan). -->",
          "  <equality>",
          '    <weld name="pelvis_world_weld" body1="pelvis"/>']
    eq += [f'    <joint joint1="{j}"/>' for j in locked_joints]
    eq += ["  </equality>", ""]
    eq_block = "\n".join(eq)

    idx = xml.rstrip().rfind("</mujoco>")
    if idx < 0:
        raise SystemExit("no </mujoco> in " + src)
    xml = xml[:idx] + eq_block + xml[idx:]

    with open(dst, "w") as f:
        f.write(xml)
    print(f"[gen_upper] kept {n_kept} upper actuators, locked "
          f"{len(locked_joints)} leg joints + pelvis weld -> {dst}")


if __name__ == "__main__":
    main()
