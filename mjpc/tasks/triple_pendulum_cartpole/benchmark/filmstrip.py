#!/usr/bin/env python3
"""Render an MJPC planner rollout so it can be looked at, not inferred.

Aggregate cost is sign-blind and timing-blind. On this task a planner can record
the *lowest* average cost of any and still leave the pendulum hanging upside
down at the goal -- the number says what the planner was paid for, not what it
did. This script replays a `corridor_benchmark --dump` CSV through the real model
and writes:

  * a labelled filmstrip PNG (one tile per sampled step, annotated with the
    quantities that decide success), and
  * optionally an MP4 of the whole rollout,

alongside a per-step text trace, so the picture and the numbers can be checked
against each other.

Usage:
  MUJOCO_GL=egl python3 filmstrip.py --dump dumps/combined_predictive_sampling_0.csv \\
      --out /tmp/ps.png --video /tmp/ps.mp4

  # choose frames explicitly (defaults to event-aligned frames, see --at)
  MUJOCO_GL=egl python3 filmstrip.py --dump dumps/combined_predictive_sampling_0.csv \\
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
from PIL import Image, ImageDraw, ImageFont


def task_xml(task):
    """Model file for a dump's `# task=` header.

    The dump names the world it was run in, so a slalom rollout cannot be
    rendered against the single-corridor task.xml -- which would draw one
    bottleneck where the run had three and put the goal 5 m short.
    """
    here = pathlib.Path(__file__).resolve()
    # .../triple_pendulum_cartpole/benchmark/ -> .../triple_pendulum_cartpole/
    root = here.parents[1]
    return str(root / ("slalom.xml" if task == "slalom" else "task.xml"))


def load_dump(path):
    """Read a corridor_benchmark --dump CSV into (arrays, stage).

    The file may open with `# key=value` comment lines. corridor_benchmark
    writes the stage there, which is what lets the render reconstruct the world
    the run actually used rather than whatever task.xml says today.
    """
    meta = {}
    with open(path) as f:
        lines = []
        for line in f:
            if line.startswith("#"):
                k, _, v = line[1:].strip().partition("=")
                meta[k.strip()] = v.strip()
            else:
                lines.append(line)
        rows = list(csv.DictReader(lines))
    if not rows:
        sys.exit(f"empty dump: {path}")
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    return cols, meta


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


def gap_frames(d, gaps, radius):
    """Frames that show the run as a sequence of obstacle crossings.

    Each obstacle is a disc of the given radius standing in the xz plane, so it
    occupies x in [g - radius, g + radius]. The panel worth showing on the way
    in is the first frame at the near edge, and on the way out the first frame
    past the far edge; those two bracket the posture the cart had to hold to get
    through. The first and last frame of the run bookend them.

    Returns (indices, labels). An obstacle the cart never reaches contributes no
    panel, so a run that failed early yields a correspondingly shorter strip.
    """
    cart = d["cart"]

    def first_at(x):
        hit = np.flatnonzero(cart >= x)
        return int(hit[0]) if hit.size else None

    idx, labels = [0], ["start"]
    for k, g in enumerate(sorted(gaps), start=1):
        for tag, x in (("entering", g - radius), ("leaving", g + radius)):
            i = first_at(x)
            if i is not None and i not in idx:
                idx.append(i)
                labels.append(f"gap {k}, {tag}")
    last = len(cart) - 1
    if last not in idx:
        idx.append(last)
        labels.append("goal")
    order = sorted(range(len(idx)), key=lambda k: idx[k])
    return [idx[k] for k in order], [labels[k] for k in order]


def _title_font(size):
    """A scalable face for the stage caption, or None to fall back.

    PIL's built-in bitmap font is fixed at ~11 px, which is unreadable once a
    1920 px sheet is scaled into a two-column figure. Any of these DejaVu paths
    is enough; if none is present the caller draws with the default font and
    the caption is merely small, not missing.
    """
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"):
        if pathlib.Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return None


def annotate(frame, lines, title=None):
    """Draw the telemetry plate, and optionally a stage caption above it.

    The caption names what the panel is showing ("gap 2, entering"); the plate
    below carries the numbers that back it up. Keeping them separate lets the
    figure be read at a glance and still be checked against the dump.
    """
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    pad, lh = 4, 12
    title_h = 0
    font = _title_font(22) if title else None
    if title:
        title_h = (font.size + 10) if font else 20
    # dark plate behind the text so it stays readable over the sky gradient
    box_h = lh * len(lines) + 2 * pad + title_h
    draw.rectangle([0, 0, img.width, box_h], fill=(0, 0, 0))
    if title:
        draw.text((pad, pad), title, fill=(255, 214, 102), font=font)
    for k, line in enumerate(lines):
        draw.text((pad, pad + title_h + k * lh), line, fill=(255, 255, 255))
    return np.asarray(img)


def tile(frames, cols, gap=6):
    """Lay the panels out on a grid, with a hairline gutter between them.

    Without the gutter, adjacent panels share the same pale background and read
    as one continuous image; the gutter is what makes the panel count obvious.
    """
    rows = (len(frames) + cols - 1) // cols
    h, w, _ = frames[0].shape
    sheet = np.full((rows * h + (rows - 1) * gap,
                     cols * w + (cols - 1) * gap, 3), 24, dtype=np.uint8)
    for k, f in enumerate(frames):
        r, c = divmod(k, cols)
        y, x = r * (h + gap), c * (w + gap)
        sheet[y:y + h, x:x + w] = f
    return sheet


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", required=True, help="corridor_benchmark --dump CSV")
    p.add_argument("--xml", default=None,
                   help="model to render against. Defaults to the file named "
                        "by the dump's `# task=` header.")
    p.add_argument("--task", default=None, choices=["corridor", "slalom"],
                   help="world the dump came from. Read from the dump's "
                        "`# task=` header by default; pass this only to "
                        "override it or for dumps written before the header "
                        "existed.")
    p.add_argument("--out", default=None, help="filmstrip PNG path")
    p.add_argument("--video", default=None, help="optional MP4 path")
    p.add_argument("--at", default=None,
                   help="comma-separated step indices; default is event-aligned")
    p.add_argument("--frames", type=int, default=8, help="tiles in the filmstrip")
    p.add_argument("--labels", default=None,
                   help="'|'-separated stage captions, one per frame, drawn "
                        "above the telemetry plate. Fewer labels than frames "
                        "leaves the remaining panels uncaptioned.")
    p.add_argument("--gaps", default=None,
                   help="comma-separated obstacle x positions. With --at unset, "
                        "picks the frame entering and the frame leaving each "
                        "obstacle's x-extent, plus the first and last frame, "
                        "and captions them.")
    p.add_argument("--gap-radius", type=float, default=0.6,
                   help="half-extent in x of an obstacle disc, used by --gaps "
                        "to decide where entering and leaving begin.")
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
    p.add_argument("--stage", default=None,
                   choices=["corridor", "balance", "combined"],
                   help="stage the dump came from. Read from the dump's "
                        "`# stage=` header by default; pass this only to "
                        "override it or for dumps written before the header "
                        "existed.")
    args = p.parse_args()

    d, meta = load_dump(args.dump)
    stage = args.stage or meta.get("stage")
    task = args.task or meta.get("task", "corridor")
    xml = args.xml or task_xml(task)
    model = mujoco.MjModel.from_xml_path(xml)

    # corridor_benchmark's balance stage moves the disks out of the plane of
    # motion and clears their contact bits. filmstrip.py reloads task.xml from
    # disk and so would otherwise show them at their XML position -- a corridor
    # that was not in the simulation being replayed. The dumped min_clearance
    # (~1413 m) is the tell.
    if stage == "balance":
        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if name and name.startswith("obstacle"):
                model.geom_pos[g] = [1e3, 0.0, 1e3]
                model.geom_contype[g] = 0
                model.geom_conaffinity[g] = 0

    data = mujoco.MjData(model)
    lookat = [float(v) for v in args.lookat.split(",")]
    cam = make_camera(model, args.distance, args.azimuth, args.elevation, lookat)

    explicit = [int(v) for v in args.at.split(",")] if args.at else None
    labels = args.labels.split("|") if args.labels else None
    if args.gaps and explicit is None:
        explicit, labels = gap_frames(
            d, [float(v) for v in args.gaps.split(",")], args.gap_radius)
    indices = pick_frames(d, args.frames, explicit)

    # ---------- the numbers that pair with the picture ----------
    cosmin = np.minimum.reduce([np.cos(d["th1"]), np.cos(d["th2"]), np.cos(d["th3"])])
    print(f"dump: {args.dump}   {len(d['step'])} steps, "
          f"{d['time'][-1]:.2f} s simulated   task: {task}   "
          f"stage: {stage or 'unspecified'}")
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
        ], title=labels[k] if labels and k < len(labels) else None))
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
