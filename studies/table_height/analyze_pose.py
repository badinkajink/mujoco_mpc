#!/usr/bin/env python3
"""Figures for the kinematic half of the study: where the brace posture keyframe
sits relative to the slab, what posture the slab actually requires, and how many
degrees of freedom that requirement uses.

No planner, no rollout. Everything here is forward kinematics on the compiled
model plus the same arithmetic `lean::Residual` does, so the figures are checkable
against the XML rather than against a run.

usage: analyze_pose.py --baseline runs/seeded --out figs
"""
import argparse, csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze import style, STATUS, SEQ, CAT      # one palette for the whole study
import retarget as R

ROOT = os.path.normpath(os.path.join(HERE, "../.."))
SRC_XML = os.path.join(ROOT, "mjpc/tasks/humanoid_bench/lean/Lean_H12_Magpie.xml")


def pristine_model():
    """Load the SHIPPED keyframes, even while a sweep has the build copy patched.

    The source XML cannot be loaded where it lives -- its <include>s are assembled
    into the build tree -- so the shipped file is copied next to the assembled one,
    where the include and mesh paths resolve, and removed again.
    """
    import shutil, atexit
    tmp = os.path.join(os.path.dirname(R.DEFAULT_XML), ".th_pristine.xml")
    shutil.copy2(SRC_XML, tmp)
    atexit.register(lambda: os.path.exists(tmp) and os.remove(tmp))
    return R.load(tmp)


def outcomes(rundir):
    out = {}
    p = os.path.join(rundir, "summary.csv")
    if not os.path.exists(p):
        return out
    for r in csv.DictReader(open(p)):
        h = round(float(r["table_h"]), 3)
        o = "fell" if r["fell"] == "1" else ("complete" if r["complete"] == "1"
                                             else "stalled")
        out.setdefault(h, []).append(o)
    return out


def geometry(m):
    """Everything lean.cc derives from the model, as a function of face height."""
    ki = R.nid(m, mujoco.mjtObj.mjOBJ_KEY, "forearm_brace_lean")
    q0 = m.key_qpos[ki].copy()
    pad = R.nid(m, mujoco.mjtObj.mjOBJ_GEOM, R.PAD)
    pad2 = R.nid(m, mujoco.mjtObj.mjOBJ_GEOM, R.PAD2)
    d = mujoco.MjData(m)
    d.qpos[:] = q0
    mujoco.mj_kinematics(m, d)
    return dict(q0=q0, pad0=d.geom_xpos[pad].copy(), pad20=d.geom_xpos[pad2].copy(),
                padr=float(m.geom_size[pad][0]), nom=R.face_of(m),
                depth=R.num(m, "brace_press_depth"))


def solve_family(m, g, hs):
    rows = []
    for h in hs:
        t = g["pad0"].copy();  t[2] += h - g["nom"]
        t2 = g["pad20"].copy(); t2[2] += h - g["nom"]
        q, err = R.solve(m, g["q0"], t, t2)
        dg = R.diagnose(m, q, h)
        qq = q[3:7]
        pitch = np.degrees(np.arcsin(np.clip(2*(qq[0]*qq[2]-qq[3]*qq[1]), -1, 1)))
        rows.append(dict(h=float(h), q=q, err=err, pitch=float(pitch),
                         base_z=float(q[2]), base_x=float(q[0]),
                         hip=float(np.degrees(q[8])), knee=float(np.degrees(q[10])),
                         com=float(dg["com_mid"]), sh=float(dg["sh_above"])))
    return rows


def save(fig, out, name, mode):
    suffix = ".png" if mode == "light" else ".dark.png"
    fig.savefig(os.path.join(out, name + suffix), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_keyframe(g, obs, out, mode):
    c = style(mode)
    hs = np.linspace(0.70, 1.20, 200)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0),
                           gridspec_kw=dict(width_ratios=[1.35, 1]))
    a = ax[0]
    tgt = hs - g["depth"]                       # brace_press_z, tracks the face
    pad = np.full_like(hs, g["pad0"][2])        # the keyframe, which does not
    a.fill_between(hs, pad, tgt, color=SEQ[1], alpha=0.55, lw=0,
                   label="what Posture and Brace Pos disagree about")
    a.plot(hs, tgt, color=SEQ[5], lw=2.0, label="Brace Pos target (tracks the slab)")
    a.plot(hs, pad, color=CAT["left wrist"], lw=2.0,
           label="forearm pad at the shipped keyframe")
    a.plot(hs, hs, color=c["muted"], lw=1.0, ls=":", label="slab face")
    a.axvline(g["nom"], color=c["mid"], lw=1.0, ls="--")
    a.annotate("compiled slab\n%.3f m" % g["nom"], (g["nom"], 0.735),
               xytext=(6, 0), textcoords="offset points", fontsize=8,
               color=c["mid"], va="bottom")
    for h, os_ in sorted(obs.items()):
        col = STATUS[max(set(os_), key=os_.count)]
        a.plot([h], [g["pad0"][2]], "o", ms=7, mfc=col, mec=c["surface"], mew=1.2,
               zorder=5)
    a.set_xlabel("slab face height (m)")
    a.set_ylabel("world z (m)")
    a.set_title("The brace posture keyframe solves one table height",
                loc="left", fontsize=11, color=c["fg"])
    a.grid(alpha=0.6, lw=0.6)
    a.legend(loc="upper left", fontsize=8, frameon=False)

    b = ax[1]
    gap = 1000.0 * (tgt - pad)
    b.axhline(0, color=c["axis"], lw=1.0)
    b.plot(hs, gap, color=SEQ[5], lw=2.0)
    for h, os_ in sorted(obs.items()):
        gg = 1000.0 * ((h - g["depth"]) - g["pad0"][2])
        o = max(set(os_), key=os_.count)
        b.plot([h], [gg], "o", ms=8, mfc=STATUS[o], mec=c["surface"], mew=1.2, zorder=5)
        b.annotate("%d/%d %s" % (os_.count(o), len(os_), o), (h, gg),
                   xytext=(0, 11 if gg >= 0 else -17), textcoords="offset points",
                   ha="center", fontsize=8, color=c["mid"])
    b.set_xlabel("slab face height (m)")
    b.set_ylabel("target − keyframe pad (mm)")
    b.set_title("and the disagreement is the height error",
                loc="left", fontsize=11, color=c["fg"])
    b.set_ylim(-225, 195)
    b.grid(alpha=0.6, lw=0.6)
    save(fig, out, "fig_keyframe", mode)


def fig_pose(rows, g, m, out, mode):
    c = style(mode)
    hs = np.array([r["h"] for r in rows])
    q0 = g["q0"]
    qq = q0[3:7]
    p0 = np.degrees(np.arcsin(np.clip(2*(qq[0]*qq[2]-qq[3]*qq[1]), -1, 1)))
    panels = [
        ("base pitch", [r["pitch"] for r in rows], "deg", p0),
        ("hip pitch (left)", [r["hip"] for r in rows], "deg", np.degrees(q0[8])),
        ("base height", [r["base_z"] for r in rows], "m", q0[2]),
        ("CoM ahead of midfoot", [1000*r["com"] for r in rows], "mm", None),
    ]
    fig, ax = plt.subplots(1, 4, figsize=(13.5, 3.2))
    for k, (name, vals, unit, ship) in enumerate(panels):
        a = ax[k]
        a.plot(hs, vals, color=SEQ[5], lw=2.0)
        if ship is not None:
            a.axhline(ship, color=CAT["left wrist"], lw=1.4, ls="--")
            a.annotate("shipped keyframe", (hs[0], ship), xytext=(2, 4),
                       textcoords="offset points", fontsize=8,
                       color=CAT["left wrist"])
        if name.startswith("CoM"):
            cap = 1000 * R.num(m, "com_cap_fwd")
            a.axhline(cap, color=CAT["pelvis"], lw=1.2, ls=":")
            a.annotate("com_cap_fwd %.0f mm" % cap, (hs[0], cap), xytext=(2, 3),
                       textcoords="offset points", fontsize=8, color=CAT["pelvis"])
            a.axhline(0, color=c["axis"], lw=1.0)
            a.set_ylim(-10, 170)
        a.axvline(g["nom"], color=c["mid"], lw=1.0, ls="--")
        a.set_title(name, loc="left", fontsize=10, color=c["fg"])
        a.set_xlabel("slab face (m)")
        a.set_ylabel(unit)
        a.grid(alpha=0.6, lw=0.6)
    fig.suptitle("The posture a slab requires, solved for each height "
                 "(feet held, both brace pads on the surface)",
                 x=0.005, ha="left", fontsize=11, color=c["fg"])
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, out, "fig_pose", mode)


def fig_mode(m, rows, g, out, mode):
    c = style(mode)
    D = []
    for r in rows:
        e = np.zeros(m.nv)
        mujoco.mj_differentiatePos(m, e, 1.0, g["q0"], r["q"])
        D.append(e[R.SEL])
    D = np.array(D)
    U, S, Vt = np.linalg.svd(D - D.mean(0), full_matrices=False)
    frac = 100 * S**2 / np.sum(S**2)
    lbl = {int(m.jnt_dofadr[j]): mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
           for j in range(1, m.njnt)}
    for i, n in enumerate(["base x", "base y", "base z", "base roll", "base pitch",
                           "base yaw"]):
        lbl[i] = n
    v = Vt[0] / np.linalg.norm(Vt[0])
    if v[R.SEL.index(4)] < 0:
        v = -v                                   # sign: more bow = positive
    order = np.argsort(-np.abs(v))[:10][::-1]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6),
                           gridspec_kw=dict(width_ratios=[1, 1.5]))
    a = ax[0]
    a.bar(np.arange(1, 7), frac[:6], color=[SEQ[5]] + [c["muted"]] * 5, width=0.62)
    a.set_xlabel("mode"); a.set_ylabel("% of variance")
    a.set_title("the requirement is one-dimensional", loc="left", fontsize=10,
                color=c["fg"])
    a.annotate("%.1f%%" % frac[0], (1, frac[0]), xytext=(0, 4),
               textcoords="offset points", ha="center", fontsize=9, color=c["fg"])
    a.set_ylim(0, 108); a.grid(axis="y", alpha=0.6, lw=0.6)
    b = ax[1]
    names = [lbl[R.SEL[i]].replace("_joint", "").replace("_", " ") for i in order]
    b.barh(np.arange(len(order)), v[order],
           color=[SEQ[5] if x > 0 else CAT["left wrist"] for x in v[order]],
           height=0.66)
    b.set_yticks(np.arange(len(order))); b.set_yticklabels(names, fontsize=8)
    b.axvline(0, color=c["axis"], lw=1.0)
    b.set_xlabel("component of the first mode (unit vector)")
    b.set_title("and it is one coordinated bow: hips, elbow, trunk pitch",
                loc="left", fontsize=10, color=c["fg"])
    b.grid(axis="x", alpha=0.6, lw=0.6)
    fig.tight_layout()
    save(fig, out, "fig_mode", mode)
    return dict(variance_first_mode=float(frac[0]),
                components={lbl[R.SEL[i]]: float(v[i])
                            for i in np.argsort(-np.abs(v))[:8]})


def fig_armreach(m, g, obs, out, mode):
    """What the arm can do on its own, with the trunk held at the shipped pose."""
    from probe_armreach import arm_only_seat
    c = style(mode)
    tgc = R.nid(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    ctr, half = d.geom_xpos[tgc].copy(), m.geom_size[tgc].copy()
    near, far = ctr[0] - half[0], ctr[0] + half[0]
    y_tgt = ctr[1] + R.num(m, "brace_lat_inset", 0.20)
    hs = np.round(np.arange(0.835, 1.1401, 0.0075), 4)
    err, px, py, seats = [], [], [], []
    for h in hs:
        q, e, p = arm_only_seat(m, g["q0"], g["pad0"][2] + (h - g["nom"]),
                                g["pad20"][2] + (h - g["nom"]))
        on = (near + 0.046 <= p[0] <= far) and abs(p[1] - ctr[1]) <= half[1]
        err.append(1000 * e); px.append(p[0]); py.append(p[1])
        seats.append(e < 1e-3 and on)
    err, py, seats = np.array(err), np.array(py), np.array(seats)
    lo = float(hs[seats][0]) if seats.any() else np.nan

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    a = ax[0]
    a.axvspan(lo, hs[-1], color=STATUS["complete"], alpha=0.10, lw=0)
    a.plot(hs, err, color=SEQ[5], lw=2.0)
    a.axvline(lo, color=STATUS["complete"], lw=1.4)
    a.annotate("arm alone cannot reach\nany slab below %.3f m" % lo, (lo, 22),
               xytext=(-10, 0), textcoords="offset points", ha="right", va="center",
               fontsize=8.5, color=c["mid"])
    a.axvline(g["nom"], color=c["mid"], lw=1.0, ls="--")
    for h, o in sorted(obs.items()):
        v = max(set(o), key=o.count)
        a.plot([h], [-4], "o", ms=7, mfc=STATUS[v], mec=c["surface"], mew=1.2,
               clip_on=False, zorder=6)
    a.set_xlabel("slab face height (m)"); a.set_ylabel("seat error (mm)")
    a.set_ylim(-8, 100)
    a.set_title("Trunk frozen: how far short the forearm falls",
                loc="left", fontsize=10.5, color=c["fg"])
    a.grid(alpha=0.6, lw=0.6)

    b = ax[1]
    b.plot(hs[seats], py[seats], color=SEQ[5], lw=2.0, label="where the arm can seat")
    b.axhline(y_tgt, color=CAT["pelvis"], lw=1.4, ls="--")
    b.annotate("Brace Pos lateral target %.2f m" % y_tgt, (hs[-1], y_tgt),
               xytext=(-4, 5), textcoords="offset points", ha="right", fontsize=8.5,
               color=CAT["pelvis"])
    b.plot([g["nom"]], [g["pad0"][1]], "o", ms=7, mfc=CAT["left wrist"],
           mec=c["surface"], mew=1.2, zorder=5)
    b.annotate("shipped keyframe", (g["nom"], g["pad0"][1]), xytext=(6, -12),
               textcoords="offset points", fontsize=8.5, color=CAT["left wrist"])
    b.axvline(g["nom"], color=c["mid"], lw=1.0, ls="--")
    b.set_xlabel("slab face height (m)"); b.set_ylabel("pad y at the seat (m)")
    b.set_xlim(hs[0], hs[-1])
    b.set_title("and how far inboard it has to come to do it",
                loc="left", fontsize=10.5, color=c["fg"])
    b.grid(alpha=0.6, lw=0.6)
    fig.tight_layout()
    save(fig, out, "fig_armreach", mode)
    return dict(arm_only_low_edge=lo, lat_target=float(y_tgt),
                pad_y_at=dict(zip([float(x) for x in hs], [float(v) for v in py])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=os.path.join(HERE, "runs/seeded"))
    ap.add_argument("--out", default=os.path.join(HERE, "figs"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m = pristine_model()
    g = geometry(m)
    obs = outcomes(a.baseline)
    hs = np.round(np.arange(0.685, 1.1851, 0.025), 3)
    rows = solve_family(m, g, hs)
    meta = arm = None
    for mode in ("light", "dark"):
        fig_keyframe(g, obs, a.out, mode)
        fig_pose(rows, g, m, a.out, mode)
        meta = fig_mode(m, rows, g, a.out, mode)
        arm = fig_armreach(m, g, obs, a.out, mode)
    out = dict(nominal_face=g["nom"], pad_z=float(g["pad0"][2]),
               pad_x=float(g["pad0"][0]), pad_y=float(g["pad0"][1]),
               press_depth=g["depth"],
               pad_clear_at_nominal_mm=1000*float(g["pad0"][2] - g["padr"] - g["nom"]),
               family=[{k: r[k] for k in
                        ("h", "pitch", "base_z", "base_x", "hip", "knee", "com",
                         "sh", "err")} for r in rows],
               mode=meta, armreach=arm)
    json.dump(out, open(os.path.join(a.out, "pose.json"), "w"), indent=1)
    print("wrote fig_keyframe / fig_pose / fig_mode and pose.json to", a.out)
    print("pad clear at nominal: %+.1f mm ; first mode %.1f%% of variance ; "
          "arm-only low edge %.3f m"
          % (out["pad_clear_at_nominal_mm"], meta["variance_first_mode"],
             arm["arm_only_low_edge"]))


if __name__ == "__main__":
    main()
