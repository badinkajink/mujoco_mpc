#!/usr/bin/env python3
"""Render a lean_bench rollout to mp4, annotated with phase, table height and load.

The bench dumps `--qpos_out` (t + full qpos, decimated) and `--out` (t + phase +
metrics). This replays the qpos through the SAME assembled model the bench ran --
with the SAME table height applied -- and burns the state into the frame, so a
claim about "it still braced at 0.785" is checkable in ten seconds rather than by
reading a CSV.

Replay only: no physics is stepped, so the video cannot disagree with the run.

The table move is duplicated here from lean.cc::TransitionLocked. It has to be:
the height is a runtime task parameter, so the compiled XML on disk always shows
the nominal 0.985 slab and a naive replay would draw the robot bracing on thin
air. Both copies derive the face from `geom_pos.z + geom_size.z`, never a literal.

usage:
  render_video.py --qpos R.qpos.csv --states R.csv --table_h 0.785 --out R.mp4
"""
import argparse, csv, math, os, sys

# GL backend on this box, measured 2026-09-04: EGL raises EGLError and OSMesa has
# no library, so glfw is the only one that works -- and it needs a real display.
# An agent/CI shell inherits an EMPTY $DISPLAY, where glfw fails with a confusing
# `Renderer has no attribute _mjr_context`. The user's X socket is
# /tmp/.X11-unix/X1, i.e. :1 (NOT :0), so fall back to that rather than making
# every caller remember. Override either variable to change backends.
os.environ.setdefault("MUJOCO_GL", "glfw")
if not os.environ.get("DISPLAY"):
    socks = sorted(f for f in os.listdir("/tmp/.X11-unix")
                   if f.startswith("X")) if os.path.isdir("/tmp/.X11-unix") else []
    if socks:
        os.environ["DISPLAY"] = ":" + socks[0][1:]
import numpy as np
import mujoco
import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "../.."))
MODEL = os.path.join(ROOT, "build_cmake/mjpc/tasks/humanoid_bench/lean/"
                           "Lean_H12_Magpie.xml")

# `reach_err` is deliberately NOT read from the CSV: lean::ComputeMetrics gates
# its reach block on `kf.name == "reach_to_target"`, which strategy 25's rung is
# not called, so the column is nan for the whole ladder. Computed here from the
# right gripper and the target mocap, both logged straight out of the model.
ANNOT = ("t", "phase_name", "face_z", "pad_clear", "f_forearm", "f_wrist",
         "f_torso", "f_pelvis", "com_beyond_foot_edge",
         "rhand_x", "rhand_y", "rhand_z", "tgt_x", "tgt_y", "tgt_z")


def set_table_height(model, face_z):
    """Mirror of lean.cc's Table H block: slab, legs, object, target mocap."""
    tb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    tg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    if tb < 0 or tg < 0 or not face_z:
        return 0.0
    slab_off = model.geom_pos[tg][2] + model.geom_size[tg][2]
    want = face_z - slab_off
    dz = want - model.body_pos[tb][2]
    model.body_pos[tb][2] = want
    under = model.geom_pos[tg][2] - model.geom_size[tg][2]
    for n in ("table_leg_1", "table_leg_2", "table_leg_3", "table_leg_4",
              "table_leg_1_collision", "table_leg_2_collision",
              "table_leg_3_collision", "table_leg_4_collision"):
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        if g < 0:
            continue
        half = max(0.01, 0.5 * (under + want))
        model.geom_size[g][2] = half
        model.geom_pos[g][2] = under - half
    tm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    if tm >= 0 and model.body_mocapid[tm] >= 0:
        model.body_pos[tm][2] += dz          # replay overwrites mocap from here
    return dz


def load_states(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            # The bench appends while it runs, so the final row can be torn.
            if any(r.get(k) in (None, "") for k in ANNOT):
                continue
            try:
                rows.append({k: (r[k] if k == "phase_name" else float(r[k]))
                             for k in ANNOT})
            except ValueError:
                continue
    return rows


def nearest(states, t):
    if not states:
        return None
    return min(states, key=lambda s: abs(s["t"] - t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpos", required=True)
    ap.add_argument("--states", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--table_h", type=float, default=0.0,
                    help="face z the run used; 0 = the compiled slab")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--title", default="")
    a = ap.parse_args()

    # The assembled XML sets meshdir relative to its own directory.
    cwd = os.getcwd()
    os.chdir(os.path.dirname(MODEL))
    model = mujoco.MjModel.from_xml_path(os.path.basename(MODEL))
    os.chdir(cwd)
    dz = set_table_height(model, a.table_h)
    # The task XML ships a 640x480 offscreen framebuffer; raise it in memory
    # rather than editing a shared model file. Multiple of 16 so ffmpeg does not
    # silently resize the frames underneath us.
    a.width = ((a.width + 15) // 16) * 16
    a.height = ((a.height + 15) // 16) * 16
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, a.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, a.height)
    data = mujoco.MjData(model)

    rows = []
    with open(a.qpos) as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            try:
                v = [float(x) for x in r]
            except ValueError:
                continue                      # torn final line, same reason
            if len(v) == 1 + model.nq:        # a short OR long row is unreplayable
                rows.append(v)
    if not rows:
        sys.exit("no qpos rows in %s" % a.qpos)
    states = load_states(a.states) if a.states else []

    t0, t1 = rows[0][0], rows[-1][0]
    n_frames = max(1, int((t1 - t0) * a.fps))
    idx = np.linspace(0, len(rows) - 1, n_frames).astype(int)

    # Framed on the slab edge and the robot's left side, where the brace lands.
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation, cam.distance = 168.0, -14.0, 3.0
    cam.lookat[:] = [0.62, -0.05, (a.table_h or 0.985) + 0.02]

    import cv2
    writer = imageio_ffmpeg.write_frames(a.out, (a.width, a.height),
                                         fps=a.fps, quality=7)
    writer.send(None)
    with mujoco.Renderer(model, height=a.height, width=a.width) as ren:
        for k in idx:
            row = rows[k]
            t = row[0]
            data.qpos[:] = row[1:1 + model.nq]
            mujoco.mj_forward(model, data)
            ren.update_scene(data, camera=cam)
            img = ren.render().copy()
            s = nearest(states, t)
            lines = [a.title or os.path.basename(a.qpos)]
            if s:
                lines += [
                    "t %6.2f s   phase %-22s table face %.3f m" %
                    (t, s["phase_name"], s["face_z"]),
                    "table load   forearm %4.0f N  wrist %4.0f N  torso %4.0f N  "
                    "pelvis %4.0f N" %
                    (s["f_forearm"], s["f_wrist"], s["f_torso"], s["f_pelvis"]),
                    "pad clearance %+5.0f mm   reach err %5.0f mm   "
                    "CoM past foot edge %+5.0f mm" %
                    (1000 * s["pad_clear"],
                     1000 * math.dist([s["rhand_x"], s["rhand_y"], s["rhand_z"]],
                                      [s["tgt_x"], s["tgt_y"], s["tgt_z"]]),
                     1000 * s["com_beyond_foot_edge"]),
                ]
            else:
                lines.append("t %6.2f s" % t)
            for i, txt in enumerate(lines):
                y = 26 + 24 * i
                sc = 0.62 if i == 0 else 0.50
                cv2.putText(img, txt, (14, y), cv2.FONT_HERSHEY_SIMPLEX, sc,
                            (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, txt, (14, y), cv2.FONT_HERSHEY_SIMPLEX, sc,
                            (255, 255, 255), 1, cv2.LINE_AA)
            writer.send(np.ascontiguousarray(img))
    writer.close()
    print("wrote %s  (%d frames, %.1f s of sim, table dz %+.3f m)"
          % (a.out, n_frames, t1 - t0, dz))


if __name__ == "__main__":
    main()
