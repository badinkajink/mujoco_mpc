#!/usr/bin/env python3
"""Look-back bank process v2 (2026-10-13 day). bank_node.py + (a) CHEAP: DDS callbacks only keep the newest
message, unpacking happens once per tick; process reniced (--nice 15) so the planner threads always win the
cores (v1 cost the planner 3 plan/s -> lean-commit collapses); (b) SAFE: switch step capped (--max_step, 3 cm),
no switching while the robot is down (base z < --z_min) or in a violent transient (all scores > --score_max).

Subscribes (read-only DDS participant, like dds_trace.py):
  rt/lowstate          motor q/dq, IMU quat + gyro
  <sport topic>        imu-site world pos + linvel = the node's own base belief (default rt/sportmodestate_est)
  rt/lowcmd            q targets the node commands (= planner ctrl for the position servos)
Every --tick s it (1) takes the newest reported state, (2) scores every bank world's prediction made
at the previous tick against it (windowed EMA, contact-focused residual), (3) rolls every world
forward from the reported state with the current command, (4) with hysteresis writes the leader's
table dz to --out ("<stamp> <dz>") which h12_control_node --bank_file polls between PlanIterations.
Bank worlds = copies of the deployed planner model (gpu/models/*.xml + .arrays.txt) with table body
z offset per hypothesis. CPU MuJoCo here (K <= ~60 at 60 us/step fits the tick); the GPU version
only moves step (3) to mujoco_warp.
Run (twin venv): cd ~/Desktop/h12/h1_mujoco && .venv/bin/python ../mujoco_mpc/mujoco_mpc/gpu/bank_node.py
"""
import argparse, os, sys, time, signal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, planner_model
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
NJ = 27
IMU_OFFSET = np.array([-0.04452, -0.01891, 0.27756])  # imu site in pelvis frame (dds_trace.py / fill_state)

def quat2mat(q):
    w, x, y, z = q
    return np.array([[1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)]], float)

class Bus:
    def __init__(self):
        self.q = np.zeros(NJ); self.dq = np.zeros(NJ); self.quat = np.array([1, 0, 0, 0.]); self.gyro = np.zeros(3); self.tick = -1; self.t_state = 0.0
        self.sp = np.zeros(3); self.sv = np.zeros(3); self.have_sport = False
        self.cmd = np.zeros(NJ); self.have_cmd = False; self.n_state = 0
        self.m_state = None; self.m_sport = None; self.m_cmd = None
    def on_state(self, m: LowState_):
        self.m_state = m; self.tick = int(m.tick); self.n_state += 1
    def on_sport(self, m: SportModeState_):
        self.m_sport = m; self.have_sport = True
    def on_cmd(self, m: LowCmd_):
        self.m_cmd = m; self.have_cmd = True
    def unpack(self):
        m = self.m_state; ms = self.m_state.motor_state
        self.q = np.fromiter((ms[i].q for i in range(NJ)), float, NJ); self.dq = np.fromiter((ms[i].dq for i in range(NJ)), float, NJ)
        self.quat = np.array(m.imu_state.quaternion[:4], float); self.gyro = np.array(m.imu_state.gyroscope[:3], float)
        sp = self.m_sport; self.sp = np.array(sp.position[:3], float); self.sv = np.array(sp.velocity[:3], float)
        mc = self.m_cmd.motor_cmd; self.cmd = np.fromiter((mc[i].q for i in range(NJ)), float, NJ)
    def state(self):
        self.unpack()
        R = quat2mat(self.quat / (np.linalg.norm(self.quat) + 1e-9)); roff = R @ IMU_OFFSET
        base_p = self.sp - roff; base_v = self.sv - np.cross(R @ self.gyro, roff)
        return base_p, self.quat / (np.linalg.norm(self.quat) + 1e-9), base_v, self.gyro.copy(), self.q.copy(), self.dq.copy(), self.cmd.copy(), self.tick

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport_topic", default="rt/sportmodestate_est"); ap.add_argument("--out", default="/dev/shm/h12_bank.txt")
    ap.add_argument("--tick", type=float, default=0.010); ap.add_argument("--window", type=float, default=0.3)
    ap.add_argument("--dz_grid", default="-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0,0.01,0.02,0.03,0.04")
    ap.add_argument("--switch_ratio", type=float, default=0.7, help="leader must beat the applied world's score by this ratio")
    ap.add_argument("--switch_ticks", type=int, default=20, help="...for this many consecutive ticks (0.2 s)")
    ap.add_argument("--dwell", type=float, default=1.0, help="min s between switches"); ap.add_argument("--variant", default="planner_lean_s25_parity")
    ap.add_argument("--max_step", type=float, default=0.03, help="max |dz change| per switch (m)")
    ap.add_argument("--z_min", type=float, default=0.6, help="no switching while base z below this (robot down)")
    ap.add_argument("--score_max", type=float, default=5e-3, help="no switching while the best score exceeds this (violent transient)")
    ap.add_argument("--nice", type=int, default=15)
    ap.add_argument("--twin_dt", type=float, default=0.005, help="plant s per lowstate tick (realchain bench: 0.005)"); ap.add_argument("--log", default=""); ap.add_argument("--apply", type=int, default=1, help="0 = score only, never write --out")
    a = ap.parse_args()
    if a.nice: os.nice(a.nice)
    dzs = [float(x) for x in a.dz_grid.split(",")]
    worlds = []
    for dz in dzs:
        m = planner_model.load(a.variant); tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "table"); m.body_pos[tb, 2] += dz
        m.opt.timestep = a.tick; worlds.append((m, mujoco.MjData(m)))
    m0 = worlds[0][0]
    jid = [m0.actuator_trnid[i, 0] for i in range(m0.nu)]; qa = np.array([m0.jnt_qposadr[j] for j in jid]); va = np.array([m0.jnt_dofadr[j] for j in jid])
    W = np.zeros(m0.nq); W[:3] = 10.0; W[3:7] = 5.0
    for i, j in enumerate(jid):
        n = m0.joint(j).name
        if any(k in n for k in ("hip_pitch", "knee", "ankle_pitch", "torso", "left_shoulder", "left_elbow")): W[qa[i]] = 1.0
    obj_q = m0.key_qpos[mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_KEY, "forearm_brace_lean")][7 + NJ:]   # object free joint: park it
    ChannelFactoryInitialize(0); bus = Bus()
    ss = ChannelSubscriber(a.sport_topic, SportModeState_); ss.Init(bus.on_sport, 10)
    cs = ChannelSubscriber("rt/lowcmd", LowCmd_); cs.Init(bus.on_cmd, 10)
    ls = ChannelSubscriber("rt/lowstate", LowState_); ls.Init(bus.on_state, 10)
    print(f"[bank] K={len(dzs)} worlds dz {dzs}; sport topic {a.sport_topic}; out {a.out}; apply={a.apply}", flush=True)
    score = np.zeros(len(dzs)); prev = None; prev_tick = -1; applied = dzs.index(0.0) if 0.0 in dzs else 0; stamp = 0; streak = 0
    last_switch = time.time(); alpha = a.tick / a.window; logf = open(a.log, "w") if a.log else None; stop = False
    signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("_stop", True))
    t_next = time.time(); n = 0; last_print = 0.0; wait_printed = False
    while not globals().get("_stop", False):
        t_next += a.tick; dt_sleep = t_next - time.time()
        if dt_sleep > 0: time.sleep(dt_sleep)
        if not (bus.have_sport and bus.have_cmd and bus.tick >= 0):
            if not wait_printed and time.time() - last_print > 2: print(f"[bank] waiting: lowstate {bus.n_state} sport {bus.have_sport} cmd {bus.have_cmd}", flush=True); last_print = time.time()
            continue
        if not wait_printed: print("[bank] bus live", flush=True); wait_printed = True
        base_p, quat, base_v, gyro, q, dq, cmd, tick = bus.state()
        if tick == prev_tick: continue          # no new plant sample yet
        qpos = np.concatenate([base_p, quat, q, obj_q]); qvel = np.concatenate([base_v, gyro, dq, np.zeros(6)])
        # LOOK-BACK: the twin runs slower than wall time, so the plant advance between two samples is
        # dt = dtick * twin_dt (2-25 ms here), not our 10 ms wall tick. Predict lazily: from the PREVIOUS
        # sample + its command, step every world by the measured dt, compare with THIS sample.
        if prev is not None:
            dt = (tick - prev_tick) * a.twin_dt
            if 0.0015 <= dt <= 0.03:
                q_prev, v_prev, c_prev = prev
                for i, (mm, dd) in enumerate(worlds):
                    mm.opt.timestep = dt; dd.qpos[:] = q_prev; dd.qvel[:] = v_prev; dd.ctrl[:] = c_prev; mujoco.mj_step(mm, dd)
                    r = float(np.sum((W * (dd.qpos - qpos)) ** 2)); score[i] = (1 - alpha) * score[i] + alpha * r if n else r
                n += 1
        prev = (qpos, qvel, cmd); prev_tick = tick
        # (4) hysteresis switch
        lead = int(np.argmin(score))
        ok = base_p[2] >= a.z_min and score[lead] <= a.score_max
        if ok and n > int(a.window / a.tick) and lead != applied and score[lead] < a.switch_ratio * score[applied]: streak += 1
        else: streak = 0
        if streak >= a.switch_ticks and time.time() - last_switch >= a.dwell:
            # capped step toward the leader: pick the grid entry nearest lead within max_step of applied
            cand = [i for i in range(len(dzs)) if abs(dzs[i] - dzs[applied]) <= a.max_step + 1e-9]
            tgt = min(cand, key=lambda i: abs(dzs[i] - dzs[lead]))
            if tgt == applied: streak = 0; continue
            applied = tgt; stamp += 1; streak = 0; last_switch = time.time()
            if a.apply:
                tmp = a.out + ".tmp"; open(tmp, "w").write(f"{stamp} {dzs[applied]:.4f}\n"); os.replace(tmp, a.out)
            print(f"[bank] t={time.time():.1f} SWITCH -> dz {dzs[applied]*100:+.0f} cm (score {score[applied]:.2e}, nominal {score[dzs.index(0.0)]:.2e}) stamp {stamp}", flush=True)
        if logf and n % 10 == 0: logf.write(f"{time.time():.3f} {tick} {applied} {lead} " + " ".join(f"{s:.3e}" for s in score) + "\n")
        if time.time() - last_print > 2.0:
            last_print = time.time(); srt = np.argsort(score)
            print(f"[bank] n={n} tick {tick} z {base_p[2]:.3f} applied dz {dzs[applied]*100:+.0f} leader {dzs[lead]*100:+.0f} ({score[lead]:.2e}) 2nd {dzs[srt[1]]*100:+.0f} ({score[srt[1]]:.2e}) nominal {score[dzs.index(0.0)]:.2e} streak {streak}", flush=True)
    if logf: logf.close()

if __name__ == "__main__":
    main()
