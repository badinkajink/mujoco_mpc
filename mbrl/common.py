"""mbrl/common.py -- shared helpers for the H1-2 sim-gap / residual-dynamics study.

Model: the deployed planner plant Lean_H12_Magpie.xml (build tree). State layout:
  qpos[0:3] pelvis pos, qpos[3:7] pelvis quat (w,x,y,z), qpos[7:34] the 27 joints,
  qpos[34:41] object anchor free joint. qvel: 0:3 lin, 3:6 ang (pelvis frame), 6:33 joints.
The recorder's base_quat/gyro/accel are the TORSO IMU (site 'imu' on torso_link), not the
pelvis -- compose through the waist joint before writing the free joint.
"""
import os
import numpy as np
import mujoco

H12 = os.path.expanduser("~/Desktop/h12")
MODEL = os.path.join(H12, "mujoco_mpc/mujoco_mpc/build/mjpc/tasks/humanoid_bench/lean/Lean_H12_Magpie.xml")
MBRL = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("MBRL_DATA_ROOT", os.path.join(MBRL, "data"))   # separate roots for the real corpus and the twin (sim2sim) corpus
RUNS = os.path.join(DATA, "runs")
OUT = os.environ.get("MBRL_OUT", os.path.join(MBRL, "out"))
NJ = 27
JQ = slice(7, 7 + NJ)       # joint qpos
JV = slice(6, 6 + NJ)       # joint qvel / dof
LEG = list(range(0, 12)); WAIST = [12]; ARM = list(range(13, 27))
GROUPS = {"legs": LEG, "waist": WAIST, "arms": ARM}
REC_DT = 0.01

JOINT_NAMES = ['left_hip_yaw_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_knee_joint',
 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_yaw_joint', 'right_hip_pitch_joint',
 'right_hip_roll_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
 'torso_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
SHORT = [n.replace("_joint", "").replace("left_", "L_").replace("right_", "R_") for n in JOINT_NAMES]


def load_model(path=MODEL, disable_actuators=True):
    cwd = os.getcwd(); os.chdir(os.path.dirname(path))
    try:
        m = mujoco.MjModel.from_xml_path(path)
    finally:
        os.chdir(cwd)
    assert m.nq == 41 and m.nv == 39 and m.nu == NJ, (m.nq, m.nv, m.nu)
    for i in range(NJ):
        assert m.joint(JOINT_NAMES[i]).qposadr[0] == 7 + i
    if disable_actuators:
        # torque is applied through qfrc_applied from the recorded PD command / tau_est
        m.actuator_gainprm[:, 0] = 0.0
        m.actuator_biasprm[:, :] = 0.0
    return m


def quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def quat_from_mat(R):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, R.reshape(9)); return q


def imu_to_pelvis(m, d, q_joints, imu_quat, gyro):
    """Return (pelvis_quat, pelvis_angvel_body) from the torso-IMU quat/gyro at joint config q.
    R_imu = R_pelvis @ R_site_rel(q)  ->  R_pelvis = R_imu @ R_site_rel^T.
    omega_imu(imu frame) = R_site_rel^T omega_pelvis(pelvis frame) + dq_waist*z  (waist axis z)."""
    d.qpos[:] = 0; d.qpos[3] = 1.0; d.qpos[JQ] = q_joints
    mujoco.mj_kinematics(m, d)
    R_rel = d.site("imu").xmat.reshape(3, 3).copy()
    R_imu = np.zeros(9); mujoco.mju_quat2Mat(R_imu, imu_quat / np.linalg.norm(imu_quat))
    R_pel = R_imu.reshape(3, 3) @ R_rel.T
    return quat_from_mat(R_pel), R_rel @ (gyro - np.array([0, 0, 0.0]))  # waist rate handled by caller


def pd_torque(kp, kd, cmd, q, dq, tau_ff, lim):
    t = kp * (cmd - q) - kd * dq + tau_ff
    return np.clip(t, -lim, lim)


def tilt_deg(quat):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, quat / np.linalg.norm(quat))
    return float(np.degrees(np.arccos(np.clip(R[8], -1, 1))))
