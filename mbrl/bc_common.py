"""Shared pieces for Stage 3 (planner distillation): observation builder and phase vocabulary.
Observation (all available on the real robot from rt/lowstate + the node's own phase counter):
  q(27) dq(27) g_imu(3) gyro(3) phase one-hot(4) kf one-hot(9) t_phase/30 (1) [+ prev cmd-q (27) if MBRL_BC_PREV=1]
Action: cmd-q (27) and tau_ff (27)."""
import os, numpy as np
PHASES = ["stand_up", "forearm_brace_lean", "forearm_brace_release", "standback"]
NKF = 9
PREV = os.environ.get("MBRL_BC_PREV", "0") == "1"
NOBS = 27 + 27 + 3 + 3 + len(PHASES) + NKF + 1 + (27 if PREV else 0)
NACT = 54

def phase_id(name):
    if name.startswith("standback"): return 3
    return PHASES.index(name) if name in PHASES else 0

def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])

def obs_vec(q, dq, quat, gyro, phase, kf, t_phase, prev_cmd=None):
    R = quat_to_mat(quat)                       # imu frame -> world
    g = R.T @ np.array([0, 0, -1.0])            # gravity in imu frame
    ph = np.zeros(len(PHASES)); ph[phase] = 1
    kfv = np.zeros(NKF); kfv[min(max(kf - 1, 0), NKF - 1)] = 1
    parts = [q, dq, g, gyro, ph, kfv, [min(t_phase, 30.0) / 30.0]]
    if PREV: parts.append((prev_cmd if prev_cmd is not None else q) - q)
    return np.concatenate([np.asarray(p, float).ravel() for p in parts]).astype(np.float32)
