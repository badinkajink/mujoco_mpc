#!/usr/bin/env python3
"""Render the brace posture at several slab heights: shipped keyframe vs re-solved.

One still per cell, assembled into a contact sheet. The point is to see, without
reading a number, that the shipped pose holds the same bow whatever the slab does
and that the re-solved pose folds to meet it.

usage: render_pose.py --heights 0.785,0.985,1.085 --out docs/lean/media/th/poses.png
"""
import argparse, os, sys
os.environ.setdefault("MUJOCO_GL", "glfw")
if not os.environ.get("DISPLAY"):
    socks = sorted(f for f in os.listdir("/tmp/.X11-unix")
                   if f.startswith("X")) if os.path.isdir("/tmp/.X11-unix") else []
    if socks:
        os.environ["DISPLAY"] = ":" + socks[0][1:]
import numpy as np
import mujoco
import imageio_ffmpeg  # noqa: F401  (pulls the same ffmpeg the videos use)
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import retarget as R
from analyze_pose import pristine_model, geometry

W, H = 620, 620


def shot(m, q, face, cam_az=118.0):
    from render_video import set_table_height
    set_table_height(m, face)
    d = mujoco.MjData(m)
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.55, 0.0, 0.85]
    cam.distance, cam.azimuth, cam.elevation = 2.45, cam_az, -8.0
    # the model ships a small offscreen framebuffer; asking for a bigger frame
    # raises inside Renderer.__init__ and surfaces as a missing _gl_context
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W)
    m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    with mujoco.Renderer(m, H, W) as ren:
        ren.update_scene(d, camera=cam)
        return Image.fromarray(ren.render())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heights", default="0.785,0.885,0.985,1.085")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    hs = [float(x) for x in a.heights.split(",")]
    base = pristine_model()
    g = geometry(base)

    cells, labels = [], []
    for row, kind in enumerate(("shipped", "re-solved")):
        for h in hs:
            m = pristine_model()
            if kind == "shipped":
                q = g["q0"].copy()
            else:
                t = g["pad0"].copy();  t[2] += h - g["nom"]
                t2 = g["pad20"].copy(); t2[2] += h - g["nom"]
                q, _ = R.solve(m, g["q0"], t, t2)
            cells.append(shot(m, q, h))
            labels.append("%s brace posture \u2014 slab %.3f m" % (kind, h))

    n = len(hs)
    bar = 34
    sheet = Image.new("RGB", (W * n, (H + bar) * 2), "#fcfcfb")
    dr = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 21)
    except OSError:
        font = ImageFont.load_default()
    for i, im in enumerate(cells):
        x, y = (i % n) * W, (i // n) * (H + bar)
        dr.text((x + 10, y + 7), labels[i], fill="#0b0b0b", font=font)
        sheet.paste(im, (x, y + bar))
        dr.rectangle([x, y + bar, x + W - 1, y + bar + H - 1], outline="#e1e0d9")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sheet.save(a.out)
    print("wrote", a.out, sheet.size)


if __name__ == "__main__":
    main()
