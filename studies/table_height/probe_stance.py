#!/usr/bin/env python3
"""Where the robot stands, and how far the brace pad gets over the slab.

The brace keyframes were authored at the compiled 0.985 m face and they carry a
stance: `forearm_brace_lean` puts the feet 63 mm further forward than `home`
does.  Strategy 25's ladder pins both feet (`Foot Left/Right Up`, weight 2000)
and has no stepping rung, so the robot can never take that step.  This measures
the consequence -- foot midpoint and forearm-pad reach, both in the slab frame --
from the logged qpos of any set of runs.  No sim time.

usage: probe_stance.py --runs LABEL=dir [LABEL=dir ...] [--json out.json]
"""
import argparse, csv, json, os, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import retarget as R
from analyze_pose import pristine_model
from render_video import set_table_height

BRACE_PHASES = (1, 2)          # forearm_brace_lean x2 in strategy 25
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def bid(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)


def gid(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)


def slab(m, d):
    """near-edge x and face z of the collision slab, in world."""
    g = gid(m, "table_top_collision")
    c, h = d.geom_xpos[g].copy(), m.geom_size[g].copy()
    return float(c[0] - h[0]), float(c[2] + h[2])


def scan(m, d, qcsv, scsv):
    """foot midpoint and best pad reach over the brace rungs, slab-relative."""
    ph = {round(float(r["t"]), 3): int(float(r["phase"]))
          for r in csv.DictReader(open(scsv))}
    pad, wrist = gid(m, R.PAD), gid(m, R.PAD2)
    feet = [bid(m, n) for n in FEET]
    near, face = None, None
    foot_x, best = [], None
    for r in csv.DictReader(open(qcsv)):
        try:
            t = round(float(r["t"]), 3)
            q = [float(r["q%d" % i]) for i in range(m.nq)]
        except ValueError:
            continue                      # truncated final row of a killed run
        if ph.get(t) not in BRACE_PHASES:
            continue
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        if near is None:
            near, face = slab(m, d)
        foot_x.append(float(np.mean([d.xpos[b][0] for b in feet])))
        reach = float(d.geom_xpos[pad][0]) - near
        if best is None or reach > best["pad_reach"]:
            best = {"t": t, "pad_reach": reach,
                    "pad_clear": float(d.geom_xpos[pad][2] - m.geom_size[pad][0] - face),
                    "wrist_clear": float(d.geom_xpos[wrist][2] - m.geom_size[wrist][0] - face),
                    "base_x": float(q[0]),
                    "foot_x": float(np.mean([d.xpos[b][0] for b in feet])) - near}
    if best is None:
        return None
    best["foot_med"] = float(np.median(foot_x)) - near
    return best


def keyframes(m, face):
    """stance each authored keyframe assumes, slab-relative."""
    d = mujoco.MjData(m)
    out = []
    for name in ("home", "forearm_brace_lean", "forearm_brace_reach",
                 "forearm_brace_release"):
        k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, name)
        if k < 0:
            continue
        d.qpos[:] = np.array(m.key_qpos).reshape(-1, m.nq)[k]
        mujoco.mj_kinematics(m, d)
        near, _ = slab(m, d)
        fx = float(np.mean([d.xpos[bid(m, n)][0] for n in FEET]))
        out.append({"key": name, "base_x": float(d.qpos[0]), "foot_x": fx,
                    "foot_from_edge": fx - near,
                    "pad_reach": float(d.geom_xpos[gid(m, R.PAD)][0]) - near})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="LABEL=dir")
    ap.add_argument("--json")
    a = ap.parse_args()

    m0 = pristine_model()
    nominal = R.face_of(m0)
    print("compiled face %.3f m\n" % nominal)

    print("stance the authored keyframes assume, at the compiled face "
          "(m, + = past the slab's near edge)")
    print("  %-24s %8s %8s %10s %10s" % ("keyframe", "base_x", "feet_x",
                                         "feet-edge", "pad-edge"))
    kf = keyframes(m0, nominal)
    for r in kf:
        print("  %-24s %+8.3f %+8.3f %+10.3f %+10.3f"
              % (r["key"], r["base_x"], r["foot_x"], r["foot_from_edge"],
                 r["pad_reach"]))
    step = None
    if len(kf) > 1:
        step = kf[1]["foot_from_edge"] - kf[0]["foot_from_edge"]
        print("  -> the brace pose stands %.0f mm further forward than home" % (1000 * step))

    rows = []
    print("\nmeasured over the brace rungs (phases %s), median foot x and the "
          "pad's best reach" % (BRACE_PHASES,))
    print("  %-10s %6s %4s %10s %11s %11s %10s"
          % ("arm", "face", "seed", "feet-edge", "pad-edge", "pad clear", "base_x"))
    for spec in a.runs:
        label, d_ = spec.split("=", 1)
        for f in sorted(os.listdir(d_)):
            if not f.endswith(".qpos.csv"):
                continue
            tag = f[:-len(".qpos.csv")]
            h = int(tag[1:5]) / 1000.0
            seed = int(tag.split("_s")[1])
            m = pristine_model()
            set_table_height(m, h)
            d = mujoco.MjData(m)
            s = scan(m, d, os.path.join(d_, f), os.path.join(d_, tag + ".csv"))
            if s is None:
                print("  %-10s %6.3f %4d   (never entered a brace rung)"
                      % (label, h, seed))
                continue
            s.update(arm=label, h=h, seed=seed)
            rows.append(s)
            print("  %-10s %6.3f %4d %+10.3f %+11.3f %+11.3f %+10.3f"
                  % (label, h, seed, s["foot_med"], s["pad_reach"],
                     s["pad_clear"], s["base_x"]))

    if a.json:
        json.dump({"keyframes": kf, "step_m": step, "runs": rows},
                  open(a.json, "w"), indent=1)
        print("\n-> %s" % a.json)


if __name__ == "__main__":
    main()
