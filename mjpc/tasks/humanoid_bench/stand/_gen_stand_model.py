"""Derive the Stand H12 robot from the shared H1-2 planner base.

STAND-SCOPED ON PURPOSE. The shared base (h1_2_base/h1_2_pos.xml -> h1_2_modified.xml)
is what lean / stabilize / upper / walk / push and the deploy twin all load, and their
weights are tuned against its (wrong) numbers. Rather than re-tune every one of them,
this writes a stand-only variant carrying the REAL robot constants, so the stand task
plans against the machine that actually exists while everything else stays byte-identical.

Every constant below is copied from the robot's own source of truth, NOT from MJPC:
  gains  : core_ws/src/h12_ros2_controller/config/safety_full.yaml      (gains.kp / gains.kd)
  ratios : core_ws/src/h12_safety_layer/config/default_safety_full.yaml (limits.estop.torque_ratio)
  torque : core_ws/src/h12_safety_layer/h12_safety_layer/core/joint_limits.py (URDF_TORQUE_LIMITS)

What it changes and why (all four were measured, see the header of Stand_H12.xml):

1. PD GAINS -> the real tuned values. The base models the arms 3-6x TOO SOFT
   (shoulder_pitch kp 40 vs the real 240). That single error is most of why the
   planner saw ~40x less arm control authority than it really has.

2. joint damping 10 -> 0. The pristine Unitree h1_2 has NO joint damping; the base's
   damping="10" is not upstream, is uncommented (entered in ace98d1 "HAMS integration"),
   and DOUBLE-COUNTS the PD's own D term, which is already modelled as the actuator kv.
   The real robot's damping is kd. Dry friction is separately modelled as frictionloss.

3. forcerange -> kClampRatio * estop_ratio * URDF_torque. The base's forceranges were
   hand-derived from ratios that no longer match the yaml (19/27 entries were stale).
   This is the torque the deploy node will actually emit, so the planner stops solving
   physics the node cannot execute.

4. ctrlrange -> home +- forcerange/kp, i.e. the position envelope implied by the torque
   clamp. NOTE this is an approximation: the real bound is |tgt - q| with q MOVING, and
   ctrlrange is absolute. It holds while q stays near home (true for standing) and is a
   guardrail, not a guarantee.

Gains are written INLINE per actuator (overriding the class defaults) because the real
gains differ within a class -- shoulder1 covers shoulder_pitch AND shoulder_roll, whose
real kp are 240 and 200.
"""
import re
import sys

# ---- robot source of truth (see module docstring for provenance) ----
JOINTS = [
    "left_hip_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
# joint_limits.py URDF_TORQUE_LIMITS
URDF_TORQUE = [200, 200, 200, 300, 60, 40, 200, 200, 200, 300, 60, 40, 200,
               40, 40, 18, 18, 19, 19, 19, 40, 40, 18, 18, 19, 19, 19]
# default_safety_full.yaml limits.estop.torque_ratio
ESTOP_RATIO = [0.50, 0.70, 1.00, 1.00, 0.90, 0.90, 0.50, 0.70, 1.00, 1.00, 0.90, 0.90, 0.30,
               0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90]
# deploy_common.h kClampRatio: the node emits at most this fraction of the estop threshold
CLAMP_RATIO = 0.9
# LEGS: not in the yaml -- that controller drives the upper body only ("lower-body gains
# default to 0") and MJPC's own node commands the legs. These are the node's KP/KV, which
# the base already matches, so they are unchanged; listed here to keep all 27 in one table.
# TORSO + ARMS: safety_full.yaml gains.kp / gains.kd.
KP = [150, 200, 200, 200, 80, 80, 150, 200, 200, 200, 80, 80, 160,
      240, 200, 150, 150, 120, 120, 120, 240, 200, 150, 150, 120, 120, 120]
KD = [5, 5, 5, 5, 4, 4, 5, 5, 5, 5, 4, 4, 10,
      12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
# Stand_H12.xml `home` keyframe, joint order (the envelope is centred on it).
HOME = dict.fromkeys(JOINTS, 0.0)
HOME.update({"left_hip_pitch_joint": -0.4, "right_hip_pitch_joint": -0.4,
             "left_knee_joint": 0.8, "right_knee_joint": 0.8,
             "left_ankle_pitch_joint": -0.4, "right_ankle_pitch_joint": -0.4})

IDX = {j: i for i, j in enumerate(JOINTS)}


def main(src, dst):
    xml = open(src).read()

    # (2) fictitious joint damping -> upstream 0
    before = xml
    xml = xml.replace('<joint damping="10" armature="0.1"/>',
                      '<joint damping="0" armature="0.1"/>')
    if xml == before:
        raise SystemExit("_gen_stand_model: joint damping='10' not found -- base changed?")

    # (1)(3)(4) per-actuator gains / forcerange / ctrlrange, inline
    seen = []

    def patch(m):
        line, name = m.group(0), m.group(1)
        i = IDX[name]
        frc = CLAMP_RATIO * ESTOP_RATIO[i] * URDF_TORQUE[i]
        env = frc / KP[i]
        lo_c, hi_c = (float(x) for x in
                      re.search(r'ctrlrange="([-\d.]+) ([-\d.]+)"', line).groups())
        lo = max(lo_c, HOME[name] - env)
        hi = min(hi_c, HOME[name] + env)
        line = re.sub(r'ctrlrange="[^"]*"', 'ctrlrange="%.4g %.4g"' % (lo, hi), line)
        line = line.replace('/>', ' kp="%g" kv="%g" forcerange="%.4g %.4g"/>'
                            % (KP[i], KD[i], -frc, frc))
        seen.append(name)
        return line

    xml = re.sub(r'<position name="([^"]+)"[^/]*/>', patch, xml)
    missing = [j for j in JOINTS if j not in seen]
    if missing:
        raise SystemExit("_gen_stand_model: actuators not patched: %s" % missing)

    open(dst, "w").write(xml)
    print("_gen_stand_model: wrote %s (%d actuators, real PD gains, damping=0)"
          % (dst, len(seen)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
