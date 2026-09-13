#!/usr/bin/env python3
"""Render the bench's qpos track (lean_bench --qpos_out, 50 Hz) to an MP4 per
run and a filmstrip PNG across runs.

    MUJOCO_GL=egl ./render.py --runs runs/video_spp15 --out ../../docs/lean/media/planner_ablation

The model is the compiled task XML under build_cmake (meshes resolve from
there); the deploy gains do not change geometry, so the XML plant renders the
deploy-plant run exactly.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import imageio
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "../../build_cmake/mjpc/tasks/humanoid_bench/lean/Lean_H12_Magpie.xml")
LABEL = {
    "cem": "CEM as shipped (elite mean, hold, 0.01 rad)",
    "icem": "iCEM as shipped (elite mean, hold + AR(1), 0.01 rad)",
    "ps": "predictive sampling as shipped (argmin, cubic, 0.03-0.36 rad)",
    "mppi": "MPPI as shipped (softmax 0.1, cubic, 0.03-0.36 rad)",
    "ps_raw01_zero": "predictive sampling, 0.01 rad, hold",
    "ps_raw01_cubic": "predictive sampling, 0.01 rad, cubic",
    "mppi_raw01_zero_l1": "MPPI, 0.01 rad, hold, lambda 1",
}
PH = ["stand", "lean", "reach", "release", "stand back 1", "stand back 2", "stand back 3", "stand back 4", "final stand"]


def font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def phase_at(metrics, t):
    i = np.searchsorted(metrics.t.values, t, side="right") - 1
    return int(metrics.phase.iloc[max(0, i)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--size", default="640x400")
    ap.add_argument("--strip_times", default="5,12.5,14.5,20,30,42")
    ap.add_argument("--no_video", action="store_true", help="only the filmstrip")
    ap.add_argument("--arms", default="")
    ap.add_argument("--seed", type=int, default=1, help="seed to render per arm")
    ap.add_argument("--override", default="", help="arm:seed,... exceptions to --seed")
    a = ap.parse_args()
    W, H = [int(x) for x in a.size.split("x")]
    os.makedirs(a.out, exist_ok=True)
    cwd = os.getcwd(); os.chdir(os.path.dirname(XML))
    model = mujoco.MjModel.from_xml_path(os.path.basename(XML)); os.chdir(cwd)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera(); cam.lookat[:] = [0.55, 0.0, 0.8]; cam.distance = 3.1; cam.azimuth = 125; cam.elevation = -10
    opt = mujoco.MjvOption()
    f_big, f_small = font(18), font(15)

    rows = [json.loads(l) for l in open(os.path.join(a.runs, "results.jsonl"))]
    want = {r["arm"]: a.seed for r in rows}
    for kv in [x for x in a.override.split(",") if x]:
        k, v = kv.split(":"); want[k] = int(v)
    rows = [r for r in rows if int(r["seed"]) == want.get(r["arm"], -1)]
    arms = [x for x in a.arms.split(",") if x] or list(dict.fromkeys(r["arm"] for r in rows))
    strip_t = [float(x) for x in a.strip_times.split(",")]
    strips = {}
    for r in rows:
        if r["arm"] not in arms:
            continue
        tag = "%s_s%s" % (r["arm"], r["seed"])
        qp = os.path.join(a.runs, tag + ".qpos.csv")
        if not os.path.exists(qp):
            print("no qpos track for", tag); continue
        q = pd.read_csv(qp); met = pd.read_csv(r["csv"])
        outcome = "completed at %.1f s" % float(r["t_complete"]) if r.get("complete") == "1" else \
                  ("fell at %.1f s" % float(r["t_end"]) if r.get("fell") == "1" else "did not complete by 75 s")
        vid = os.path.join(a.out, "vid_%s.mp4" % r["arm"])
        writer = None if a.no_video else imageio.get_writer(vid, fps=a.fps, codec="libx264", quality=7,
                                                             macro_block_size=8, ffmpeg_params=["-pix_fmt", "yuv420p"])
        step = max(1, int(round(50 / a.fps)))
        frames_for_strip = {}
        last = None
        for i in range(0, len(q), step):
            t = float(q.t.iloc[i])
            if a.no_video and not any(st not in frames_for_strip and t >= st - 1e-6 for st in strip_t) and i != len(q) - 1 and (i + step) < len(q):
                continue
            t = float(q.t.iloc[i])
            data.qpos[:] = q.iloc[i, 1:1 + model.nq].values
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, cam, opt)
            img = Image.fromarray(renderer.render())
            d = ImageDraw.Draw(img)
            ph = phase_at(met, t)
            d.rectangle([0, 0, W, 50], fill=(255, 255, 255))
            d.text((8, 4), LABEL.get(r["arm"], r["arm"]), fill=(20, 20, 20), font=f_big)
            d.text((8, 28), "t = %5.1f s   rung: %s   %s   33 plans/s, deploy gains, seed %s" % (t, PH[ph] if 0 <= ph < len(PH) else ph, outcome, r["seed"]),
                   fill=(80, 80, 80), font=f_small)
            if writer is not None:
                writer.append_data(np.asarray(img))
            last = (img.copy(), ph, t)
            for st in strip_t:
                if st not in frames_for_strip and t >= st - 1e-6:
                    frames_for_strip[st] = (img.copy(), ph, t)
        # a run that ended early keeps its last frame in the remaining slots, marked
        for st in strip_t:
            if st not in frames_for_strip:
                frames_for_strip[st] = ("ended", last)
        if writer is not None:
            writer.close()
        strips[r["arm"]] = (frames_for_strip, outcome)
        print("->", vid, "%d frames" % (len(q) // step))
    # filmstrip: rows = arms in the requested order, columns = times
    if strips:
        tw, th = W // 2, H // 2
        lab_w = 0
        rows_img = []
        for arm in arms:
            if arm not in strips:
                continue
            fr, outcome = strips[arm]
            row = Image.new("RGB", (tw * len(strip_t), th), (250, 250, 250))
            for j, st in enumerate(strip_t):
                cell = fr[st]
                ended = cell[0] == "ended"
                im0 = cell[1][0] if ended else cell[0]
                im = im0.crop((0, 50, W, H)).resize((tw, th), Image.LANCZOS)
                if ended:
                    im = Image.blend(im, Image.new("RGB", im.size, (255, 255, 255)), 0.35)
                    d2 = ImageDraw.Draw(im); d2.text((6, th - 22), "run ended at %.1f s" % cell[1][2], fill=(120, 30, 30), font=f_small)
                row.paste(im, (j * tw, 0))
            rows_img.append((arm, outcome, row))
        strip = Image.new("RGB", (tw * len(strip_t), (th + 26) * len(rows_img) + 24), (255, 255, 255))
        d = ImageDraw.Draw(strip)
        for j, st in enumerate(strip_t):
            d.text((j * tw + 6, 4), "t = %g s" % st, fill=(60, 60, 60), font=f_small)
        y = 24
        for arm, outcome, row in rows_img:
            d.text((6, y + 4), "%s — %s" % (LABEL.get(arm, arm), outcome), fill=(20, 20, 20), font=f_small)
            strip.paste(row, (0, y + 26)); y += th + 26
        sp = os.path.join(a.out, "frames.png"); strip.save(sp); print("->", sp, strip.size)


if __name__ == "__main__":
    main()
