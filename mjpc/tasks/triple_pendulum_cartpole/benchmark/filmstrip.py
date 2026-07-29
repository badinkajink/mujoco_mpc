#!/usr/bin/env python3
"""Render an MJPC planner rollout so it can be looked at, not inferred.

Aggregate cost is sign-blind and timing-blind. On this task MPPI records the
*lowest* average cost of any planner while leaving the pendulum hanging upside
down at the goal -- the number says what the planner was paid for, not what it
did. This script replays a `corridor_benchmark --dump` CSV through the real model
and writes:

  * a labelled filmstrip PNG (one tile per sampled step, annotated with the
    quantities that decide success), and
  * optionally an MP4 of the whole rollout,

alongside a per-step text trace, so the picture and the numbers can be checked
against each other.

Usage:
  MUJOCO_GL=egl python3 filmstrip.py --dump rollouts/mppi.csv \\
      --out /tmp/mppi.png --video /tmp/mppi.mp4

  # choose frames explicitly (defaults to event-aligned frames, see --at)
  MUJOCO_GL=egl python3 filmstrip.py --dump rollouts/mppi.csv \\
      --at 0,600,1200,2000,3000,4500,5999
"""
import argparse
import csv
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
from PIL import Image, ImageDraw


def default_xml():
    here = pathlib.Path(__file__).resolve()
    # .../triple_pendulum_cartpole/benchmark/ -> .../triple_pendulum_cartpole/
    return str(here.parents[1] / "task.xml")


def load_dump(path):
    """Read a corridor_benchmark --dump CSV into a dict of arrays."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"empty dump: {path}")
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    return cols


def pick_frames(d, n, explicit=None):
    """Event-aligned frame selection.

    A uniform sample of a 30 s rollout mostly shows the tail. The moments that
    decide the outcome are: the start, when the pendulum first leaves upright,
    the closest approach to an obstacle, the deepest contact, when the cart
    first crosses the corridor, the furthest right it reaches, and the end.
    """
    if explicit:
        return sorted({min(max(0, i), len(d["step"]) - 1) for i in explicit})

    N = len(d["step"])
    cos1 = np.cos(d["th1"])
    events = {0, N - 1}

    # first departure from upright
    left_upright = np.argmax(cos1 < 0.9)
    if cos1[left_upright] < 0.9:
        events.add(int(left_upright))

    # closest approach to an obstacle, and deepest penetration if any
    events.add(int(np.argmin(d["min_clearance"])))
    if d["ncon"].max() > 0:
        events.add(int(np.argmax(d["ncon"])))

    # first crossing of the corridor plane (x = 3) and the rightmost extent
    past = np.argmax(d["cart"] > 3.0)
    if d["cart"][past] > 3.0:
        events.add(int(past))
    events.add(int(np.argmax(d["cart"])))

    events = sorted(events)
    # top up to n with a uniform spread so the strip is not sparse
    if len(events) < n:
        filler = np.linspace(0, N - 1, n - len(events) + 2).astype(int)[1:-1]
        events = sorted(set(events) | set(int(i) for i in filler))
    return events[:n]


def render_frames(model, data, d, indices, width, height, camera, track=False):
    """Replay the dumped states through the real model and render each one.

    With track=True the camera follows the cart, which makes the pendulum
    legible at the cost of losing the obstacles as a fixed reference. Use the
    fixed camera to judge *where* the cart is, tracking to judge *what the
    pendulum is doing*.
    """
    frames = []
    with mujoco.Renderer(model, height, width) as renderer:
        for i in indices:
            if track:
                camera.lookat[0] = d["cart"][i]
            data.qpos[0] = d["cart"][i]
            data.qpos[1] = d["th1"][i]
            data.qpos[2] = d["th2"][i]
            data.qpos[3] = d["th3"][i]
            data.qvel[0] = d["dcart"][i]
            data.qvel[1] = d["dth1"][i]
            data.qvel[2] = d["dth2"][i]
            data.qvel[3] = d["dth3"][i]
            data.ctrl[0] = d["ctrl"][i]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())
    return frames


def make_camera(model, distance, azimuth, elevation, lookat):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.lookat[:] = lookat
    return cam


def annotate(frame, lines):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    # dark plate behind the text so it stays readable over the sky gradient
    pad, lh = 4, 12
    box_h = lh * len(lines) + 2 * pad
    draw.rectangle([0, 0, img.width, box_h], fill=(0, 0, 0))
    for k, line in enumerate(lines):
        draw.text((pad, pad + k * lh), line, fill=(255, 255, 255))
    return np.asarray(img)


def tile(frames, cols):
    rows = (len(frames) + cols - 1) // cols
    h, w, _ = frames[0].shape
    sheet = np.full((rows * h, cols * w, 3), 24, dtype=np.uint8)
    for k, f in enumerate(frames):
        r, c = divmod(k, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = f
    return sheet


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", required=True, help="corridor_benchmark --dump CSV")
    p.add_argument("--xml", default=default_xml())
    p.add_argument("--out", default=None, help="filmstrip PNG path")
    p.add_argument("--video", default=None, help="optional MP4 path")
    p.add_argument("--at", default=None,
                   help="comma-separated step indices; default is event-aligned")
    p.add_argument("--frames", type=int, default=8, help="tiles in the filmstrip")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--video-stride", type=int, default=4,
                   help="use every Nth step for the video")
    p.add_argument("--distance", type=float, default=6.5)
    p.add_argument("--azimuth", type=float, default=90.0,
                   help="90 puts the x-z plane of motion in the image plane")
    p.add_argument("--elevation", type=float, default=-8.0)
    p.add_argument("--lookat", default="3.0,0,0.1")
    p.add_argument("--from-step", type=int, default=0,
                   help="first step included in the video")
    p.add_argument("--to-step", type=int, default=-1,
                   help="last step included in the video (-1 = end)")
    p.add_argument("--track", action="store_true",
                   help="camera follows the cart (pendulum legible, obstacles "
                        "no longer a fixed reference)")
    args = p.parse_args()

    d = load_dump(args.dump)
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    lookat = [float(v) for v in args.lookat.split(",")]
    cam = make_camera(model, args.distance, args.azimuth, args.elevation, lookat)

    explicit = [int(v) for v in args.at.split(",")] if args.at else None
    indices = pick_frames(d, args.frames, explicit)

    # ---------- the numbers that pair with the picture ----------
    cosmin = np.minimum.reduce([np.cos(d["th1"]), np.cos(d["th2"]), np.cos(d["th3"])])
    print(f"dump: {args.dump}   {len(d['step'])} steps, "
          f"{d['time'][-1]:.2f} s simulated")
    print(f"  cart:  start {d['cart'][0]:+.3f}  max {d['cart'].max():+.3f}  "
          f"final {d['cart'][-1]:+.3f}")
    print(f"  worst cos(theta) at end: {cosmin[-1]:+.3f}   "
          f"(upright needs > +0.95)")
    print(f"  min clearance over run:  {d['min_clearance'].min():+.4f} m   "
          f"contact steps: {int((d['ncon'] > 0).sum())} "
          f"({100.0 * (d['ncon'] > 0).mean():.1f}%)")
    print(f"  |ctrl| max: {np.abs(d['ctrl']).max():.2f} N   "
          f"saturated (>19.9 N): {100.0 * (np.abs(d['ctrl']) > 19.9).mean():.1f}%")
    print(f"  cost: first {d['cost'][0]:.2f}  min {d['cost'].min():.2f}  "
          f"final {d['cost'][-1]:.2f}  mean {d['cost'].mean():.2f}")
    print("\n  frame trace:")
    print(f"    {'step':>6} {'t(s)':>6} {'cart':>7} {'cos_min':>8} "
          f"{'clear':>8} {'ctrl':>7} {'cost':>8}")
    for i in indices:
        print(f"    {int(d['step'][i]):6d} {d['time'][i]:6.2f} "
              f"{d['cart'][i]:+7.3f} {cosmin[i]:+8.3f} "
              f"{d['min_clearance'][i]:+8.3f} {d['ctrl'][i]:+7.2f} "
              f"{d['cost'][i]:8.2f}")

    # ---------- filmstrip ----------
    out = args.out or str(pathlib.Path(args.dump).with_suffix(".png"))
    frames = render_frames(model, data, d, indices, args.width, args.height,
                           cam, args.track)
    labelled = []
    for k, i in enumerate(indices):
        labelled.append(annotate(frames[k], [
            f"step {int(d['step'][i])}  t={d['time'][i]:.2f}s",
            f"cart {d['cart'][i]:+.2f}  cos_min {cosmin[i]:+.2f}",
            f"clear {d['min_clearance'][i]:+.3f}  u {d['ctrl'][i]:+.1f}",
        ]))
    Image.fromarray(tile(labelled, args.cols)).save(out)
    print(f"\nwrote filmstrip: {out}")

    # ---------- video ----------
    if args.video:
        import imageio.v2 as imageio
        stride = max(1, args.video_stride)
        lo = max(0, args.from_step)
        hi = len(d["step"]) if args.to_step < 0 else min(len(d["step"]),
                                                        args.to_step + 1)
        vid_idx = list(range(lo, hi, stride))
        vframes = render_frames(model, data, d, vid_idx, args.width,
                                args.height, cam, args.track)
        vlabelled = [
            annotate(f, [f"t={d['time'][i]:.2f}s  cart {d['cart'][i]:+.2f}  "
                         f"cos_min {cosmin[i]:+.2f}"])
            for f, i in zip(vframes, vid_idx)
        ]
        imageio.mimwrite(args.video, vlabelled, fps=args.fps // stride or 1,
                         macro_block_size=1)
        print(f"wrote video:     {args.video}  "
              f"({len(vlabelled)} frames @ {args.fps // stride or 1} fps)")


if __name__ == "__main__":
    main()
