#!/usr/bin/env python3
"""Render a lean_bench rollout to mp4, annotated with the phase it was in.

The bench dumps `--qpos_out` (t + full qpos, decimated) and `--out` (t + phase +
derived state). This replays the qpos through the SAME assembled model the bench
ran and burns the phase name, sim time and brace load into the frame, so a claim
about "the schedule got shorter and it still braced" is checkable in ten seconds
rather than by reading a CSV.

Replay only -- no physics is stepped, so the video cannot disagree with the run.

usage:
  render_video.py --qpos R.qpos.csv --states R.csv --out R.mp4 [--fps 25]
"""
import argparse, csv, os, sys

# EGL and OSMesa both fail on this box (EGLError / missing GL symbols); the glfw
# backend renders offscreen fine because there is a display. Override with
# MUJOCO_GL if that ever stops being true.
os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np
import mujoco
import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "../.."))
MODEL = os.path.join(ROOT, "build_cmake/mjpc/tasks/humanoid_bench/lean/"
                           "Lean_H12_Magpie.xml")


def load_states(path):
    """t -> (phase_name, f_elbow, f_forearm, f_palm, f_torso), for annotation."""
    out = []
    keys = ("t", "phase_name", "f_elbow", "f_forearm", "f_palm", "f_torso")
    with open(path) as f:
        for r in csv.DictReader(f):
            # The bench appends while it runs, so the final row can be torn.
            # Skip anything incomplete rather than dying on a partial line.
            if any(r.get(k) in (None, "") for k in keys):
                continue
            try:
                out.append((float(r["t"]), r["phase_name"],
                            float(r["f_elbow"]), float(r["f_forearm"]),
                            float(r["f_palm"]), float(r["f_torso"])))
            except ValueError:
                continue
    return out


def nearest(states, t):
    if not states:
        return ("?", 0.0, 0.0, 0.0, 0.0)
    i = min(range(len(states)), key=lambda k: abs(states[k][0] - t))
    return states[i][1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpos", required=True)
    ap.add_argument("--states", default="")
    ap.add_argument("--out", required=True)
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
    # The task XML ships the default 640x480 offscreen framebuffer; raise it in
    # memory rather than editing a shared model file. Round up to a multiple of
    # 16 so ffmpeg does not silently resize the frames underneath us.
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
            # A torn final line can be short OR long; only a full (t + nq) row
            # is replayable, so require exactly that.
            if len(v) == 1 + model.nq:
                rows.append(v)
    if not rows:
        sys.exit("no qpos rows in %s" % a.qpos)
    states = load_states(a.states) if a.states else []

    # Decimate the log down to the requested frame rate.
    t0, t1 = rows[0][0], rows[-1][0]
    n_frames = max(1, int((t1 - t0) * a.fps))
    idx = np.linspace(0, len(rows) - 1, n_frames).astype(int)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation, cam.distance = 168.0, -14.0, 3.0
    cam.lookat[:] = [0.62, -0.05, 0.98]

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
            ph, fe, ff, fp, ft = nearest(states, t)
            lines = [
                "%s" % (a.title or os.path.basename(a.qpos)),
                "t = %6.2f s    phase: %s" % (t, ph),
                "table load  elbow %5.0f N  forearm %5.0f N  palm %5.0f N  torso %5.0f N"
                % (fe, ff, fp, ft),
            ]
            for i, s in enumerate(lines):
                y = 26 + 24 * i
                cv2.putText(img, s, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.62 if i == 0 else 0.52, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, s, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.62 if i == 0 else 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            writer.send(np.ascontiguousarray(img))
    writer.close()
    print("wrote %s  (%d frames, %.1f s of sim)" % (a.out, n_frames, t1 - t0))


if __name__ == "__main__":
    main()
