#!/usr/bin/env python3
"""Render friction-sweep runs to MP4 with the contact state drawn on the robot.

    MUJOCO_GL=egl ./render.py --runs runs/grid1 --tags t1_f1_s3,t0.8_f0.4_s3 \
        --out ../../docs/lean/media/brace_friction/vid_s3.mp4 [--t0 20 --t1 40] [--speed 1]

One tag renders one run; two tags render them side by side on a shared clock
(same seed, two frictions). Drawn from the run's own records:
  * qpos track (<tag>.qpos.csv, 100 Hz) poses the model (compiled task XML, so
    geometry matches the plant; friction and gains do not change geometry);
  * contact record (<tag>.contact.csv, 500 Hz) gives the force arrows: the
    SHEAR (horizontal) part of the force the slab puts on the brace pads and the
    floor puts on each foot, drawn flat on that surface from the contact point,
    1.5 mm/N. Arrow colour = shear ratio |F_xy|/F_z against that surface's plant
    friction: green below 0.6 mu, amber 0.6-0.95 mu, red at the friction limit.
    Spheres trace the pad's contact line while loaded (blue holding, red
    sliding faster than 20 mm/s) and the soles wherever they slid;
  * strip chart underneath: pad and foot shear ratios (10 ms median) over a
    16 s window that scrolls with the frame, the two plant friction
    coefficients as reference lines, a cursor at the frame time.
"""
import argparse, json, os
import numpy as np
import pandas as pd
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "../../build_cmake/mjpc/tasks/humanoid_bench/lean/Lean_H12_Magpie.xml")
RUNG = ["stand", "lean", "reach", "release", "stand back 1", "stand back 2", "stand back 3",
        "stand back 4", "final stand"]
FACE_Z = 0.985
# reference palette (dataviz skill): series slots for identity, status for stick/slide
C_PAD, C_FL, C_FR = "#2a78d6", "#eb6834", "#1baf7a"
S_GOOD, S_WARN, S_CRIT = (12, 163, 12), (250, 178, 25), (208, 59, 59)
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def font(size, bold=False):
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for d in ["/usr/share/fonts/truetype/dejavu/", "/usr/share/fonts/dejavu/"]:
        for n in names:
            if os.path.exists(d + n):
                return ImageFont.truetype(d + n, size)
    return ImageFont.load_default()


def status(ratio, mu):
    if not np.isfinite(ratio) or mu <= 0:
        return S_GOOD
    x = ratio / mu
    return S_CRIT if x >= 0.95 else S_WARN if x >= 0.6 else S_GOOD


def add_arrow(scene, p0, vec, rgb, width=0.012):
    if scene.ngeom >= scene.maxgeom or np.linalg.norm(vec) < 1e-4:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                        np.array([rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, 1.0], dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(p0, float),
                         np.asarray(p0, float) + np.asarray(vec, float))
    scene.ngeom += 1


def add_dot(scene, p, rgb, r=0.006, alpha=1.0):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, r, r]), np.asarray(p, float),
                        np.eye(3).flatten(), np.array([rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, alpha],
                                                      dtype=np.float32))
    scene.ngeom += 1


class Run:
    def __init__(self, runs, tag):
        self.tag = tag
        rec = {}
        for line in open(os.path.join(runs, "results.jsonl")):
            r = json.loads(line)
            if r["tag"] == tag:
                rec = r
        self.rec = rec
        self.q = pd.read_csv(os.path.join(runs, tag + ".qpos.csv"))
        self.c = pd.read_csv(os.path.join(runs, tag + ".contact.csv"))
        self.mu_t = float(rec.get("plant_table_mu", 1.0))
        self.mu_f = float(rec.get("plant_foot_mu", 1.0))
        c = self.c
        self.t = c.t.to_numpy()
        def ratio(fx, fy, fz, floor):
            r = pd.Series(np.hypot(fx, fy) / np.maximum(fz, 1e-9)).rolling(5, center=True, min_periods=1)
            r = r.median().to_numpy().copy()
            r[fz.to_numpy() <= floor] = np.nan
            return r
        self.r_pad = ratio(c.pd_fx, c.pd_fy, c.pd_fz, 20.0)
        self.r_fl = ratio(c.fl_fx, c.fl_fy, c.fl_fz, 50.0)
        self.r_fr = ratio(c.fr_fx, c.fr_fy, c.fr_fz, 50.0)
        # running slide totals (mm), same definition as analyze.py
        self.slide_pad = np.cumsum(c.pd_slip.to_numpy() * (c.pd_fz.to_numpy() > 20)) * 2.0
        fl = np.cumsum(c.fl_slip.to_numpy() * (c.fl_fz.to_numpy() > 50)) * 2.0
        fr = np.cumsum(c.fr_slip.to_numpy() * (c.fr_fz.to_numpy() > 50)) * 2.0
        self.slide_foot = np.maximum(fl, fr)
        self.t_end = float(self.t[-1])
        if rec.get("complete") == "1":
            self.outcome = "completed at %.1f s" % float(rec["t_complete"])
        elif rec.get("fell") == "1":
            self.outcome = "fell at %.1f s" % float(rec["t_end"])
        else:
            self.outcome = "no completion by %.0f s" % float(rec.get("t_end", self.t_end))
        self.label = "slab μ %.2g, floor μ %.2g" % (self.mu_t, self.mu_f)
        if float(rec.get("planner_table_mu", 1.0)) != 1.0 or float(rec.get("planner_foot_mu", 1.0)) != 1.0:
            self.label += " (planner told)"
        else:
            self.label += " (planner assumes 1.0)"

    def idx(self, t):
        return int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))


class Strip:
    """Persistent strip chart: all data plotted once, window and cursor moved per frame."""
    WIN_BACK, WIN_FWD = 12.0, 4.0

    def __init__(self, runs, W, H):
        n = len(runs)
        self.fig, axes = plt.subplots(1, n, figsize=(W / 100, H / 100), dpi=100, squeeze=False)
        self.fig.patch.set_facecolor(SURF)
        self.axes, self.cursors = list(axes[0]), []
        for ax, r in zip(self.axes, runs):
            ax.set_facecolor(SURF)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(AXIS)
            ax.tick_params(colors=MUTED, labelsize=8, length=2)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.plot(r.t, r.r_pad, color=C_PAD, lw=1.8, label="Brace pads")
            ax.plot(r.t, r.r_fl, color=C_FL, lw=1.2, label="Left foot")
            ax.plot(r.t, r.r_fr, color=C_FR, lw=1.2, label="Right foot")
            ax.axhline(r.mu_t, color=C_PAD, lw=0.9, alpha=0.55)
            ax.axhline(r.mu_f, color=C_FL, lw=0.9, alpha=0.55)
            self.mu_labels = []
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("Shear / normal", fontsize=8, color=INK2)
            ax.set_xlabel("Time (s)", fontsize=8, color=INK2)
            if abs(r.mu_t - r.mu_f) < 0.02:
                ax.text(1.005, r.mu_t / 1.1, "slab & floor \u03bc", transform=ax.transAxes,
                        va="center", fontsize=7, color=INK2)
            else:
                ax.text(1.005, r.mu_t / 1.1, "slab \u03bc", transform=ax.transAxes, va="center", fontsize=7, color=INK2)
                ax.text(1.005, r.mu_f / 1.1, "floor \u03bc", transform=ax.transAxes, va="center", fontsize=7, color=INK2)
            self.cursors.append(ax.axvline(0, color=INK, lw=0.9))
        self.axes[0].legend(loc="upper left", fontsize=7, frameon=False, ncol=3)
        self.fig.subplots_adjust(left=0.06 if n == 1 else 0.05, right=0.93 if n == 1 else 0.95,
                                 bottom=0.3, top=0.93, wspace=0.22)

    def render(self, t):
        for ax, cur in zip(self.axes, self.cursors):
            ax.set_xlim(max(0.0, t - self.WIN_BACK), max(0.0, t - self.WIN_BACK) + self.WIN_BACK + self.WIN_FWD)
            cur.set_xdata([t, t])
        self.fig.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        return Image.frombuffer("RGBA", (w, h), self.fig.canvas.buffer_rgba()).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tags", required=True, help="one tag, or two comma-separated for side by side")
    ap.add_argument("--out", required=True)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=-1.0, help="-1 = end of the longest run")
    ap.add_argument("--speed", type=float, default=2.0, help="sim seconds per video second")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--size", default="", help="WxH of each 3D view (default 960x540, 640x480 paired)")
    ap.add_argument("--cam", default="0.42,0.12,0.58,2.4,300,-16", help="lookat x,y,z,distance,azimuth,elevation")
    ap.add_argument("--still", default="", help="comma list of times: write PNG stills instead of a video")
    a = ap.parse_args()
    runs = [Run(a.runs, t) for t in a.tags.split(",")]
    n = len(runs)
    W, H = [int(x) for x in (a.size or ("960x540" if n == 1 else "640x480")).split("x")]
    t1 = a.t1 if a.t1 > 0 else max(r.t_end for r in runs)
    cwd = os.getcwd()
    os.chdir(os.path.dirname(XML))
    model = mujoco.MjModel.from_xml_path(os.path.basename(XML))
    os.chdir(cwd)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, W)
    model.vis.global_.offheight = max(model.vis.global_.offheight, H)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, H, W, max_geom=10000)
    lx, ly, lz, dist, az, el = [float(x) for x in a.cam.split(",")]
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [lx, ly, lz]
    cam.distance, cam.azimuth, cam.elevation = dist, az, el
    opt = mujoco.MjvOption()
    f_big, f_small = font(17, True), font(14)
    HEAD, STRIP = 58, 150
    strip = Strip(runs, W * n, STRIP)
    frames_t = np.arange(a.t0, t1 + 1e-9, a.speed / a.fps)
    stills = [float(x) for x in a.still.split(",") if x]
    if stills:
        frames_t = np.array(stills)
    writer = None
    if not stills:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        writer = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=7, macro_block_size=8,
                                    ffmpeg_params=["-pix_fmt", "yuv420p"])
    for fi, t in enumerate(frames_t):
        canvas = Image.new("RGB", (W * n, HEAD + H + STRIP), SURF)
        for k, r in enumerate(runs):
            tt = min(t, r.t_end)
            qi = int(np.clip(np.searchsorted(r.q.t.to_numpy(), tt), 0, len(r.q) - 1))
            data.qpos[:] = r.q.iloc[qi, 1:1 + model.nq].to_numpy()
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, cam, opt)
            sc = renderer.scene
            ci = r.idx(tt)
            row = r.c.iloc[ci]
            # trails: pad contact line and soles while loaded (every 20 ms up to now)
            hist = r.c.iloc[max(0, r.idx(max(a.t0, tt - 25.0))):ci + 1:10]
            for _, h in hist.iterrows():
                if h.pd_fz > 20:
                    add_dot(sc, (h.pad_x, h.pad_y, FACE_Z + 0.003),
                            S_CRIT if h.pd_slip > 0.02 else (42, 120, 214), 0.005)
                for s, f in (("L", "fl"), ("R", "fr")):
                    if h[f + "_fz"] > 50 and h[f + "_slip"] > 0.02:
                        add_dot(sc, (h["sole" + s + "_x"], h["sole" + s + "_y"], 0.004), S_CRIT, 0.006)
            if row.pd_fz > 5:
                add_arrow(sc, (row.pad_x, row.pad_y, FACE_Z + 0.006), 0.0015 * np.array([row.pd_fx, row.pd_fy, 0.0]),
                          status(np.hypot(row.pd_fx, row.pd_fy) / max(row.pd_fz, 1e-9), r.mu_t), 0.01)
            for s, f in (("L", "fl"), ("R", "fr")):
                if row[f + "_fz"] > 5:
                    add_arrow(sc, (row["sole" + s + "_x"], row["sole" + s + "_y"], 0.006),
                              0.0015 * np.array([row[f + "_fx"], row[f + "_fy"], 0.0]),
                              status(np.hypot(row[f + "_fx"], row[f + "_fy"]) / max(row[f + "_fz"], 1e-9), r.mu_f), 0.01)
            img = Image.fromarray(renderer.render())
            if t > r.t_end + 1e-6:
                img = Image.blend(img, Image.new("RGB", img.size, (255, 255, 255)), 0.45)
            canvas.paste(img, (k * W, HEAD))
            d = ImageDraw.Draw(canvas)
            ph = int(row.phase)
            d.text((k * W + 10, 6), r.label, fill=INK, font=f_big)
            d.text((k * W + 10, 32), "t %5.2f s   rung: %s   %s" % (tt, RUNG[ph] if 0 <= ph < len(RUNG) else ph,
                                                                  r.outcome), fill=INK2, font=f_small)
            # live numbers on the render, bottom-left
            ypad = HEAD + H - 58
            d.rectangle([k * W + 8, ypad - 4, k * W + 330, ypad + 50], fill=(252, 252, 251))
            d.text((k * W + 14, ypad), "brace pad %4.0f N, shear/normal %s, slid %3.0f mm" % (
                row.pd_fz, ("%.2f" % (np.hypot(row.pd_fx, row.pd_fy) / row.pd_fz)) if row.pd_fz > 20 else " -  ",
                r.slide_pad[ci]), fill=INK, font=f_small)
            d.text((k * W + 14, ypad + 24), "feet slid up to %3.0f mm   pelvis %.2f m" % (r.slide_foot[ci], row.pelvis_z),
                   fill=INK, font=f_small)
        canvas.paste(strip.render(t), (0, HEAD + H))
        if writer is not None:
            writer.append_data(np.asarray(canvas))
        else:
            base, ext = os.path.splitext(a.out)
            canvas.save(f"{base}_t{t:05.2f}{ext or '.png'}")
    if writer is not None:
        writer.close()
    print("->", a.out, len(frames_t), "frames")


if __name__ == "__main__":
    main()
