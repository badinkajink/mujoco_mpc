#!/usr/bin/env python3
"""Contact forces the planner-ablation runs actually had, from their state tracks.

The ablation (studies/planner_ablation, 2026-09-11..13) logged qpos, qvel, ctrl
and the solver warm start once per plan (lean_bench --state_out --log_per_plan,
33 Hz at --spp 15) but never the contact forces. Re-running mj_forward on each
logged state with the plant's model (deploy joint PD, the run's plant friction)
reconstructs them: the brace pads' force on the robot, both feet, and the slip
speed at each contact, classified as lean_bench --contact_out does.

What this can and cannot see: one sample per plan (30 ms), so a brace impact
(tens of ms) is between samples; the quasi-static load is not. MuJoCo here is
the Python build (3.4.0) while lean_bench links 3.2.3; replay_check.py compares
the two on a run that has both records.

    ./replay_ablation.py --dirs gains_spp15,mismatch_spp15_mu0.6,mismatch_spp15_mu0.4 \
        --arms cem,icem,ps_raw01_zero,mppi_raw01_zero_l1 --out runs/replay_ablation.jsonl
"""
import argparse, json, os
import numpy as np
import pandas as pd
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "../../build_cmake/mjpc/tasks/humanoid_bench/lean/Lean_H12_Magpie.xml")
ABL = os.path.join(HERE, "../planner_ablation/runs")
KP = [150, 200, 200, 200, 200, 80, 150, 200, 200, 200, 200, 80, 200,
      90, 60, 40, 90, 15, 15, 15, 90, 60, 40, 90, 15, 15, 15]
KV = [5, 5, 5, 5, 4, 4, 5, 5, 5, 5, 4, 4, 5,
      10, 10, 10, 10, 2, 2, 2, 10, 10, 10, 10, 2, 2, 2]


def load_model(friction_scale=1.0, table_mu=0.0, foot_mu=0.0):
    cwd = os.getcwd()
    os.chdir(os.path.dirname(XML))
    m = mujoco.MjModel.from_xml_path(os.path.basename(XML))
    os.chdir(cwd)
    for i in range(m.nu):                       # lean_bench --gains deploy
        m.actuator_gainprm[i, 0] = KP[i]
        m.actuator_biasprm[i, 1] = -KP[i]
        m.actuator_biasprm[i, 2] = -KV[i]
    if friction_scale != 1.0:                   # lean_bench --plant_friction_scale (geoms only)
        m.geom_friction[:, 0] *= friction_scale
    return m


class Classifier:
    def __init__(self, m):
        self.m = m
        oid = lambda t, n: mujoco.mj_name2id(m, t, n)
        B, G = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM
        self.table = oid(B, "table")
        self.obj = oid(B, "object")
        self.floor = oid(G, "floor")
        self.lf, self.rf = oid(B, "left_ankle_roll_link"), oid(B, "right_ankle_roll_link")
        self.pad_gid = oid(G, "left_forearm_pad")
        self.pads = {self.pad_gid, oid(G, "left_wrist_pad")}
        self.left = {b for b in range(m.nbody)
                     if (mujoco.mj_id2name(m, B, b) or "").startswith("left_")}

    def point_vel(self, d, b, p):
        res = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY, b, res, 0)
        return res[3:] + np.cross(res[:3], p - d.xipos[b])

    def groups(self, d):
        """Return {group: (F_on_robot[3], fn, slip_fw, n)} for pad / arm / fl / fr."""
        m = self.m
        out = {k: [np.zeros(3), 0.0, 0.0, 0, 0.0] for k in ("pad", "arm", "fl", "fr")}
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.efc_address < 0:
                continue
            g0, g1 = c.geom
            b0, b1 = m.geom_bodyid[g0], m.geom_bodyid[g1]
            keys, side = [], -1
            if (b0 == self.table) != (b1 == self.table):
                other = b1 if b0 == self.table else b0
                side = 1 if b0 == self.table else 0
                if other in self.left:
                    keys.append("arm")
                    if (g1 if side == 1 else g0) in self.pads:
                        keys.append("pad")
            elif g0 == self.floor or g1 == self.floor:
                other = b1 if g0 == self.floor else b0
                side = 1 if g0 == self.floor else 0
                if other == self.lf:
                    keys.append("fl")
                elif other == self.rf:
                    keys.append("fr")
            if not keys:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            fr = c.frame.reshape(3, 3)
            fw = fr.T @ f6[:3] * (1.0 if side == 1 else -1.0)
            rb = b1 if side == 1 else b0
            vr = self.point_vel(d, rb, c.pos)
            vt = vr - np.dot(vr, fr[0]) * fr[0]
            u = 0.0
            if f6[0] > 2.0:   # this contact's use of its friction pyramid (1 = sliding)
                u = abs(f6[1]) / c.friction[0] + abs(f6[2]) / c.friction[1]
                if c.dim >= 4 and c.friction[2] > 0:
                    u += abs(f6[3]) / c.friction[2]
                u /= f6[0]
            for k in keys:
                g = out[k]
                g[0] += fw
                g[1] += f6[0]
                g[2] += max(0.0, f6[0]) * np.linalg.norm(vt)
                g[3] += 1
                g[4] = max(g[4], u)
        return out


def replay(m, cls, state_csv):
    s = pd.read_csv(state_csv)
    nq, nv, nu = m.nq, m.nv, m.nu
    q = s[[f"q{k}" for k in range(nq)]].to_numpy()
    v = s[[f"v{k}" for k in range(nv)]].to_numpy()
    u = s[[f"u{k}" for k in range(nu)]].to_numpy()
    w = s[[f"warm{k}" for k in range(nv)]].to_numpy()
    d = mujoco.MjData(m)
    rows = []
    for i in range(len(s)):
        d.qpos[:], d.qvel[:], d.ctrl[:], d.qacc_warmstart[:] = q[i], v[i], u[i], w[i]
        d.time = s.t.iloc[i]
        mujoco.mj_forward(m, d)
        g = cls.groups(d)
        r = {"t": s.t.iloc[i], "phase": int(s.phase.iloc[i]), "pelvis_z": q[i, 2]}
        for k, (F, fn, sfw, n, umax) in g.items():
            r[k + "_fx"], r[k + "_fy"], r[k + "_fz"] = F
            r[k + "_slip"] = sfw / fn if fn > 1e-9 else 0.0
            r[k + "_n"] = n
            r[k + "_util"] = umax
        r["pad_x"], r["pad_y"] = d.geom_xpos[cls.pad_gid, :2]
        for side, b in (("L", cls.lf), ("R", cls.rf)):
            p = d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.array([0.044, 0.0, -0.045])
            r["sole" + side + "_x"], r["sole" + side + "_y"] = p[:2]
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(df, dt):
    out = {}
    lean = df.phase.between(1, 7)
    ld = (df.pad_fz > 20) & lean
    ratio = np.hypot(df.pad_fx, df.pad_fy) / df.pad_fz.clip(lower=1e-9)
    out["pad_loaded_s"] = float(ld.sum() * dt)
    out["pad_peak_N"] = float(df.pad_fz[lean].max()) if lean.any() else 0.0
    out["arm_peak_N"] = float(df.arm_fz[lean].max()) if lean.any() else 0.0
    out["pad_ratio_p95"] = float(ratio[ld].quantile(0.95)) if ld.any() else None
    out["pad_ratio_max"] = float(ratio[ld].max()) if ld.any() else None
    for thr in (0.5, 0.6, 0.8):
        out[f"pad_frac_over_{thr:g}"] = float((ratio[ld] > thr).mean()) if ld.any() else None
    out["pad_slide_mm"] = float((df.pad_slip * ld).sum() * dt * 1000)
    fr_ = []
    for f in ("fl", "fr"):
        fl = (df[f + "_fz"] > 50) & lean
        r = np.hypot(df[f + "_fx"], df[f + "_fy"]) / df[f + "_fz"].clip(lower=1e-9)
        fr_.append((float(r[fl].quantile(0.99)) if fl.any() else 0.0, float(r[fl].max()) if fl.any() else 0.0,
                    float((r[fl] > 0.4).mean()) if fl.any() else 0.0,
                    float((df[f + "_slip"] * fl).sum() * dt * 1000)))
    out["foot_ratio_p99"] = max(x[0] for x in fr_)
    out["foot_ratio_max"] = max(x[1] for x in fr_)
    out["foot_frac_over_0.4"] = max(x[2] for x in fr_)
    out["foot_slide_mm"] = max(x[3] for x in fr_)
    # horizontal force the brace pads supply vs the feet, while the pads are loaded
    if ld.any():
        out["pad_share_of_shear"] = float(np.median(
            np.hypot(df.pad_fx[ld], df.pad_fy[ld]) /
            (np.hypot(df.pad_fx[ld], df.pad_fy[ld]) + np.hypot(df.fl_fx[ld] + df.fr_fx[ld], df.fl_fy[ld] + df.fr_fy[ld])
             ).clip(lower=1e-9)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="gains_spp15,mismatch_spp15_mu0.6,mismatch_spp15_mu0.4")
    ap.add_argument("--arms", default="cem,icem,ps_raw01_zero,mppi_raw01_zero_l1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tracks", default="", help="dir to keep each run's reconstructed track (csv)")
    a = ap.parse_args()
    arms = set(a.arms.split(","))
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["dir"], r["arm"], r["seed"]))
    models = {}
    with open(a.out, "a") as fo:
        for dname in a.dirs.split(","):
            ddir = os.path.join(ABL, dname)
            for line in open(os.path.join(ddir, "results.jsonl")):
                r = json.loads(line)
                if ("all" not in arms and r["arm"] not in arms) or (dname, r["arm"], int(r["seed"])) in done:
                    continue
                # early rows of gains_spp15 predate the spp field; the dir name fixes it
                spp = r.get("spp") or (15 if "spp15" in dname else 0)
                if r.get("gains") != "deploy" or int(spp) != 15:
                    continue
                st = r.get("state") or os.path.join(ddir, "%s_s%d.state.csv" % (r["arm"], int(r["seed"])))
                st = st if os.path.isabs(st) else os.path.join(ABL, "..", st)
                if not os.path.exists(st):
                    st = os.path.join(ddir, "%s_s%d.state.csv" % (r["arm"], int(r["seed"])))
                if not os.path.exists(st):
                    continue
                fs = float(r.get("plant_friction_scale", 1.0))
                if abs(float(r.get("plant_mass_scale", 1.0)) - 1) > 1e-9 or abs(float(r.get("plant_kp_scale", 1.0)) - 1) > 1e-9:
                    continue
                if fs not in models:
                    m = load_model(fs)
                    models[fs] = (m, Classifier(m))
                m, cls = models[fs]
                df = replay(m, cls, st)
                dt = float(np.median(np.diff(df.t))) if len(df) > 1 else 0.03
                rec = {"dir": dname, "arm": r["arm"], "seed": int(r["seed"]), "friction_scale": fs,
                       "fell": int(r.get("fell", 0)), "complete": int(r.get("complete", 0)),
                       "t_end": float(r.get("t_end", "nan")), "dt": dt}
                rec.update(summarize(df, dt))
                if a.tracks:
                    os.makedirs(a.tracks, exist_ok=True)
                    df.to_csv(os.path.join(a.tracks, "%s__%s_s%d.csv" % (dname, r["arm"], int(r["seed"]))), index=False)
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
                os.fsync(fo.fileno())
                print("%-24s %-20s s%-2d fell=%d pad_loaded=%5.1fs peak=%5.0fN ratio_p95=%s max=%s foot_max=%.2f slide pad %.0f foot %.0f mm"
                      % (dname, r["arm"], int(r["seed"]), rec["fell"], rec["pad_loaded_s"], rec["pad_peak_N"],
                         "%.2f" % rec["pad_ratio_p95"] if rec["pad_ratio_p95"] is not None else "-",
                         "%.2f" % rec["pad_ratio_max"] if rec["pad_ratio_max"] is not None else "-",
                         rec["foot_ratio_max"], rec["pad_slide_mm"], rec["foot_slide_mm"]), flush=True)


if __name__ == "__main__":
    main()
