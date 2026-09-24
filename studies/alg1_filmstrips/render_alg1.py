#!/usr/bin/env python3
"""Slide figures for Algorithm 1 (one CEM planning iteration) from a lean_bench
sample dump (`lean_bench --samples_out DIR --samples_at t,...`).

    MUJOCO_GL=egl ./render_alg1.py --tick runs/<run>/tick01_t13.500 --out figs/<name>

One dump directory holds one planning iteration: the centre spline and its
rollout, the sampling std and the drawn noise, every candidate spline with its
rollout and return, the elite order, and the folded spline with its rollout.
The figures follow the four highlighted stages of the algorithm box:

    A  center    theta_bar and its rollout over the horizon
    B  spread    sigma per parameter, and the N perturbed splines around theta_bar
    C  rollouts  every candidate's end state (fan) with its return J(k); four
                 candidates as filmstrips with their running cost
    D  fold      the elites, their mean, and the mean's rollout

Frames come from the compiled task XML under build_cmake (same geometry as the
deploy plant). All renders are the side view of the sagittal plane; the robot
is drawn in light grey for ghost overlays so a cost tint keeps its shading.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec, colors as mcolors
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "../../build_cmake/mjpc/tasks/humanoid_bench/lean/Lean_H12_Magpie.xml")

# joints shown in the spline panels: index into the 27 actuators
KEY_JOINTS = [(1, "hip pitch"), (3, "knee"), (4, "ankle pitch"), (12, "torso"),
              (13, "L shoulder pitch"), (16, "L elbow"), (20, "R shoulder pitch"), (23, "R elbow")]
ROBOT_BODIES = range(1, 32)   # pelvis .. right_magpie_gripper in the compiled model

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
    "savefig.dpi": 200, "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
})
CMAP = matplotlib.colormaps["viridis"]
C_NOMINAL = "#1b1b1b"
C_FOLD = "#c8102e"
C_ELITE = "#f28e2b"
C_GREY = "#b8b8b8"


# ----------------------------------------------------------------------------- data
def load_tick(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    center = pd.read_csv(os.path.join(d, "center.csv"))
    sigma = pd.read_csv(os.path.join(d, "sigma.csv"))
    noise = pd.read_csv(os.path.join(d, "noise.csv"))
    cand = pd.read_csv(os.path.join(d, "candidates.csv"))
    scores = pd.read_csv(os.path.join(d, "scores.csv")).sort_values("k").reset_index(drop=True)
    roll = pd.read_csv(os.path.join(d, "rollouts.csv"))
    fold = pd.read_csv(os.path.join(d, "fold.csv"))
    ucols = [c for c in center.columns if c.startswith("u")]
    qcols = [c for c in roll.columns if c.startswith("q")]
    rollouts = {int(k): g.sort_values("step").reset_index(drop=True) for k, g in roll.groupby("k")}
    return dict(meta=meta, center=center, sigma=sigma, noise=noise, cand=cand, scores=scores,
                rollouts=rollouts, fold=fold, ucols=ucols, qcols=qcols)


def spline_values(df, ucols, k=None):
    """knots x nu array for the centre/fold (k None) or candidate k."""
    g = df if k is None else df[df.k == k]
    g = g.sort_values("knot")
    return g.t.values, g[ucols].values


# ----------------------------------------------------------------------------- rendering
class Scene:
    def __init__(self, W=900, H=560, azimuth=90, elevation=-8, lookat=(0.42, 0.0, 0.72), distance=2.95):
        cwd = os.getcwd(); os.chdir(os.path.dirname(XML))
        self.m = mujoco.MjModel.from_xml_path(os.path.basename(XML)); os.chdir(cwd)
        self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, W)
        self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, H)
        self.d = mujoco.MjData(self.m)
        self.W, self.H = W, H
        self.r = mujoco.Renderer(self.m, H, W)
        self.rseg = mujoco.Renderer(self.m, H, W); self.rseg.enable_segmentation_rendering()
        self.cam = mujoco.MjvCamera(); self.cam.lookat[:] = lookat; self.cam.distance = distance
        self.cam.azimuth = azimuth; self.cam.elevation = elevation
        self.opt = mujoco.MjvOption()
        self.robot_geoms = np.array([g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] in ROBOT_BODIES])
        self.rgba0 = self.m.geom_rgba.copy()

    def _set(self, qpos):
        self.d.qpos[:] = qpos; mujoco.mj_forward(self.m, self.d)

    def frame(self, qpos):
        self._set(qpos); self.r.update_scene(self.d, self.cam, self.opt)
        return self.r.render().copy()

    def mask(self, qpos):
        self._set(qpos); self.rseg.update_scene(self.d, self.cam, self.opt)
        seg = self.rseg.render(); gid = seg[..., 0]
        body = self.m.geom_bodyid[np.clip(gid, 0, self.m.ngeom - 1)]
        return (gid >= 0) & (body >= 1) & (body <= 31)

    def ghost(self, qpos, rgb=(0.86, 0.86, 0.86)):
        """robot rendered light grey, plus its mask"""
        self.m.geom_rgba[self.robot_geoms, :3] = rgb
        img = self.frame(qpos)
        self.m.geom_rgba[:] = self.rgba0
        return img, self.mask(qpos)

    def project(self, pts):
        """world points (n,3) -> pixel (u, v) for the current camera; a pinhole at
        the mean of the two stereo eyes with the scene's frustum"""
        self.r.update_scene(self.d, self.cam, self.opt)
        c0, c1 = self.r.scene.camera[0], self.r.scene.camera[1]
        pos = 0.5 * (np.asarray(c0.pos) + np.asarray(c1.pos))
        fwd = np.asarray(c0.forward, dtype=float); up = np.asarray(c0.up, dtype=float)
        right = np.cross(fwd, up)
        rel = np.asarray(pts, dtype=float) - pos
        x = rel @ right; y = rel @ up; z = rel @ fwd
        half_h = 0.5 * (c0.frustum_top - c0.frustum_bottom)
        half_w = half_h * self.W / self.H
        u = self.W / 2 + (x * c0.frustum_near / z) / half_w * self.W / 2
        v = self.H / 2 - ((y * c0.frustum_near / z) - c0.frustum_center) / half_h * self.H / 2
        return np.c_[u, v]

    def body_path(self, tick, k, body, stride=2):
        """world position of `body` (or `geom:<name>`) along rollout k, every `stride` steps"""
        is_geom = body.startswith("geom:")
        bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM if is_geom else mujoco.mjtObj.mjOBJ_BODY, body[5:] if is_geom else body)
        r = tick["rollouts"][k]; Q = r[tick["qcols"]].values
        out = []
        for i in range(0, len(Q), stride):
            self.d.qpos[:] = Q[i]; mujoco.mj_forward(self.m, self.d)
            out.append((self.d.geom_xpos if is_geom else self.d.xpos)[bid].copy())
        return np.array(out)

    def pad_gap(self, tick, k):
        """gap between the lowest point of left_forearm_pad and the table face, per horizon step (m)"""
        pad = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "left_forearm_pad")
        tab = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        prad, half = self.m.geom_size[pad, 0], self.m.geom_size[pad, 1]
        Q = tick["rollouts"][k][tick["qcols"]].values
        out = np.empty(len(Q))
        for i, q in enumerate(Q):
            self.d.qpos[:] = q; mujoco.mj_forward(self.m, self.d)
            surf = self.d.geom_xpos[tab, 2] + self.m.geom_size[tab, 2]
            az = self.d.geom_xmat[pad].reshape(3, 3)[2, 2]
            out[i] = self.d.geom_xpos[pad, 2] - abs(half * az) - prad - surf
        return out

    def composite(self, base, overlays, alpha=0.5):
        """overlays: list of (ghost_img, mask, colour). Painted in order."""
        out = base.astype(np.float32) / 255.0
        for img, msk, col in overlays:
            tint = img.astype(np.float32) / 255.0 * np.asarray(col, dtype=np.float32)[:3] / 0.86
            tint = np.clip(tint, 0, 1)
            a = alpha * msk[..., None]
            out = out * (1 - a) + tint * a
        return (out * 255).astype(np.uint8)


def outline(mask, width=2):
    """boundary pixels of a mask: mask minus its erosion by `width`"""
    er = mask.copy()
    for _ in range(width):
        e = er.copy()
        e[1:, :] &= er[:-1, :]; e[:-1, :] &= er[1:, :]; e[:, 1:] &= er[:, :-1]; e[:, :-1] &= er[:, 1:]
        er = e
    return mask & ~er


def paint_outline(img, mask, colour, width=2):
    out = img.copy(); ol = outline(mask, width)
    out[ol] = (np.asarray(colour[:3]) * 255).astype(np.uint8)
    return out


def crop(img, box):
    """box = (x0, y0, x1, y1) as fractions of width/height"""
    H, W = img.shape[:2]
    return img[int(box[1] * H):int(box[3] * H), int(box[0] * W):int(box[2] * W)]


# ----------------------------------------------------------------------------- helpers
def horizon_steps(tick, n_frames):
    H = tick["meta"]["horizon_steps"]
    return [int(round(i * (H - 1) / (n_frames - 1))) for i in range(n_frames)]


def qpos_at(tick, k, step):
    r = tick["rollouts"][k]
    return r.loc[step, tick["qcols"]].values


def step_line(ax, t_knots, vals, T, marker=True, **kw):
    """zero-order hold spline over the horizon [0, T]: each knot holds until the
    next; the last knot sits at h = T and is drawn as a marker only."""
    h = t_knots - t_knots[0]
    inside = h < T - 1e-9
    tt = np.r_[h[inside], T]
    vv = np.r_[vals[inside], vals[inside][-1]]
    ax.step(tt, vv, where="post", **kw)
    if marker:
        mk = dict(kw); mk.pop("ls", None); mk.pop("alpha", None); mk.pop("label", None)
        ax.plot(h, vals, "o", ms=3.2, **{k: v for k, v in mk.items() if k in ("color",)})


CLEAN = False   # --clean: no title/subtitle text (slide versions)


def title_box(fig, text, sub=None, y=0.985):
    if CLEAN:
        return
    fig.text(0.012, y, text, ha="left", va="top", fontsize=15, fontweight="bold")
    if sub:
        fig.text(0.012, y - 0.045, sub, ha="left", va="top", fontsize=11, color="#444444")


def note(fig, x, y, text, **kw):
    if not CLEAN:
        fig.text(x, y, text, **kw)


def knot_ylim(ax, tick, j, T, extra=()):
    """y-range from the knots that act inside the horizon (h < T), over every
    candidate plus the centre and any extra rows; the h = T knot may clip"""
    tk, _ = spline_values(tick["center"], tick["ucols"])
    act = (tk - tk[0]) < T - 1e-9
    vals = [tick["cand"][tick["ucols"][j]].values.reshape(-1, len(tk))[:, act].ravel(),
            tick["center"][tick["ucols"][j]].values[act]]
    for df in extra:
        vals.append(df[tick["ucols"][j]].values[act])
    v = np.concatenate(vals); lo, hi = v.min(), v.max(); pad = 0.12 * (hi - lo + 1e-3)
    ax.set_ylim(lo - pad, hi + pad)


def rank_colour(rank, n):
    return CMAP(0.1 + 0.8 * (1 - rank / max(1, n - 1)))


def filmstrip_row(fig, gs_row, scene, tick, k, steps, box, label=None, label_col=C_NOMINAL, ghost_x0=True):
    """one row of frames for rollout k at the given horizon steps"""
    axes = []
    x0 = qpos_at(tick, -1, 0)
    g0, m0 = scene.ghost(x0) if ghost_x0 else (None, None)
    dt = tick["meta"]["dt"]
    for j, s in enumerate(steps):
        ax = fig.add_subplot(gs_row[j])
        img = scene.frame(qpos_at(tick, k, s))
        if ghost_x0 and s > 0:
            img = scene.composite(img, [(g0, m0, (0.6, 0.6, 0.6))], alpha=0.28)
        ax.imshow(crop(img, box)); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title("h = %.1f s" % (s * dt), fontsize=10, color="#555555", pad=3)
        axes.append(ax)
    if label:
        axes[0].text(0.02, 0.97, label, transform=axes[0].transAxes, ha="left", va="top", fontsize=10,
                     color=label_col, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=label_col, lw=1))
    return axes


# ----------------------------------------------------------------------------- figures
TRACE_BODIES = [("right_magpie_gripper", 2.0), ("left_magpie_gripper", 2.0), ("pelvis", 1.6)]


def draw_fan(ax, tick, scene, step, ks, colours, box, alpha=0.12, nominal=True, fold=False, traces=True,
             trace_bodies=TRACE_BODIES, outline_w=1):
    """imshow the fan at horizon `step` on ax, then the body traces from h = 0 to
    `step` as polylines in the same colours"""
    img = fan_image(tick, scene, step, ks, colours, alpha=alpha, include_nominal=nominal, fold=fold, outline_w=outline_w)
    H, W = img.shape[:2]; x0, y0 = int(box[0] * W), int(box[1] * H)
    ax.imshow(crop(img, box)); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    if not traces:
        return
    scene._set(qpos_at(tick, -1, 0))   # camera state for project()
    for body, lw in trace_bodies:
        for k, col in zip(ks, colours):
            P = scene.body_path(tick, k, body, stride=2)[: step // 2 + 1]
            uv = scene.project(P)
            ax.plot(uv[:, 0] - x0, uv[:, 1] - y0, color=col, lw=lw, alpha=0.85, solid_capstyle="round")
        if nominal:
            uv = scene.project(scene.body_path(tick, -1, body, stride=2)[: step // 2 + 1])
            ax.plot(uv[:, 0] - x0, uv[:, 1] - y0, color=C_NOMINAL, lw=lw + 0.6, alpha=0.9)
        if fold:
            uv = scene.project(scene.body_path(tick, -2, body, stride=2)[: step // 2 + 1])
            ax.plot(uv[:, 0] - x0, uv[:, 1] - y0, color=C_FOLD, lw=lw + 0.8, alpha=0.95)
    ax.set_xlim(0, int(box[2] * W) - x0); ax.set_ylim(int(box[3] * H) - y0, 0)


def fig_center(tick, scene, out, box, n_frames=6):
    meta = tick["meta"]
    steps = horizon_steps(tick, n_frames)
    fig = plt.figure(figsize=(16, 8.4))
    gs = gridspec.GridSpec(2, 1, figure=fig, left=0.045, right=0.99, top=0.865, bottom=0.07,
                           hspace=0.28, height_ratios=[1.9, 1])
    gs_top = gridspec.GridSpecFromSubplotSpec(1, n_frames, subplot_spec=gs[0], wspace=0.03)
    filmstrip_row(fig, gs_top, scene, tick, -1, steps, box, label="rollout of θ̄ from x₀")
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.16, width_ratios=[1, 1.9])
    axc = fig.add_subplot(gs_bot[0])
    r = tick["rollouts"][-1]
    hh = r.t.values - r.t.values[0]
    axc.plot(hh, r.cost.values, color=C_NOMINAL, lw=2)
    axc.set_xlim(0, hh[-1]); axc.set_xlabel("horizon time h (s)"); axc.set_ylabel("c(x_h, u_h)")
    axc.set_title("running cost along the rollout; J = mean = %.2f" % (r.cost.mean()), fontsize=10.5, loc="left")
    for st in steps:
        axc.axvline(st * meta["dt"], color="#cccccc", lw=0.8, zorder=0)
    tk, U = spline_values(tick["center"], tick["ucols"])
    T = (meta["horizon_steps"] - 1) * meta["dt"]
    sub = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs_bot[1], hspace=0.8, wspace=0.45)
    for i, (j, name) in enumerate(KEY_JOINTS):
        ax = fig.add_subplot(sub[i // 4, i % 4])
        step_line(ax, tk, U[:, j], T, color=C_NOMINAL, lw=2)
        ax.set_title(name, fontsize=9.5, loc="left", pad=2)
        ax.tick_params(labelsize=8); ax.set_xlim(-0.02, T + 0.02); knot_ylim(ax, tick, j, T)
        if i // 4 == 1: ax.set_xlabel("h (s)", fontsize=9)
    note(fig, 0.39, 0.385, "θ̄ itself: %d knots × %d joints (position targets, rad), zero-order hold; eight joints shown. "
             "The knot at h = %.1f s acts only on the last step." % (len(tk), U.shape[1], T), fontsize=10, color="#333333")
    title_box(fig, "1   θ̄ ← Center(θ)      what to sample around",
              "t = %.2f s, %s.  The spline kept by the previous iteration, resampled onto knots starting at t; "
              "rolled out once from the measured state x₀ through f." % (meta["t_plan"], meta["phase_name"]))
    fig.savefig(out + "_A_center.png"); fig.savefig(out + "_A_center.pdf"); plt.close(fig)


JOINT_GROUPS = [(0, 6, "left leg"), (6, 12, "right leg"), (12, 13, "torso"), (13, 20, "left arm"), (20, 27, "right arm")]
JOINT_SHORT = ["hip y", "hip p", "hip r", "knee", "ank p", "ank r"] * 2 + ["torso"] + \
              ["sh p", "sh r", "sh y", "elbow", "wr r", "wr p", "wr y"] * 2


def fig_spread(tick, scene, out):
    meta = tick["meta"]
    N = meta["n"]
    tk, U0 = spline_values(tick["center"], tick["ucols"])
    T = (meta["horizon_steps"] - 1) * meta["dt"]
    sig = tick["sigma"]
    nk, nu = meta["knots"], meta["nu"]
    S = sig.sigma_used.values.reshape(nk, nu)
    fig = plt.figure(figsize=(16, 6.6))
    gs = gridspec.GridSpec(1, 3, figure=fig, left=0.05, right=0.99, top=0.78, bottom=0.17, wspace=0.25,
                           width_ratios=[1.0, 0.05, 2.1])
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(S * 1e3, aspect="auto", cmap="magma", vmin=0, vmax=max(1e-9, S.max() * 1e3))
    ax.set_yticks(range(nk)); ax.set_yticklabels(["knot %d  (h = %.1f s)" % (i, tk[i] - tk[0]) for i in range(nk)])
    ax.set_xticks(range(nu)); ax.set_xticklabels(JOINT_SHORT, rotation=90, fontsize=7)
    for a, b, name in JOINT_GROUPS:
        ax.annotate(name, xy=((a + b - 1) / 2, nk - 0.5), xytext=(0, -46), textcoords="offset points",
                    ha="center", va="top", fontsize=9, annotation_clip=False)
        if a > 0: ax.axvline(a - 0.5, color="white", lw=1.2)
    ax.set_title("Σ: sampling std per parameter (mrad)\nmin %.0f, max %.0f; floor std_min = %.0f mrad" %
                 (S.min() * 1e3, S.max() * 1e3, meta["std_min_used"] * 1e3), fontsize=10.5, loc="left")
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 1])); cb.ax.tick_params(labelsize=8)
    sub = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[0, 2], hspace=0.55, wspace=0.38)
    for i, (j, name) in enumerate(KEY_JOINTS):
        axj = fig.add_subplot(sub[i // 4, i % 4])
        for k in range(N):
            _, Uk = spline_values(tick["cand"], tick["ucols"], k)
            step_line(axj, tk, Uk[:, j], T, color=C_GREY, lw=0.9, alpha=0.9, marker=False)
            axj.plot(tk - tk[0], Uk[:, j], "o", color=C_GREY, ms=2.2, alpha=0.7)
        step_line(axj, tk, U0[:, j], T, color=C_NOMINAL, lw=2.2)
        axj.set_title(name, fontsize=9.5, loc="left", pad=2); axj.tick_params(labelsize=8); axj.set_xlim(-0.02, T + 0.02)
        knot_ylim(axj, tick, j, T)
        if i // 4 == 1: axj.set_xlabel("h (s)", fontsize=9)
        if i % 4 == 0: axj.set_ylabel("target (rad)", fontsize=9)
    note(fig, 0.40, 0.845, "θ(k) = θ̄ + ε(k), k = 1…%d   (grey: the %d perturbed splines and their knots; black: θ̄)" % (N, N),
             fontsize=10.5, color="#333333")
    title_box(fig, "2–3   ε(k) ~ N(0, Σ),  θ(k) ← θ̄ + ε(k)      Spread",
              "t = %.2f s.  Σ is diagonal over the %d × %d = %d spline parameters: the elite variance from the last "
              "fold, floored at std_min.  Every knot of every joint moves by its own draw."
              % (meta["t_plan"], nk, nu, nk * nu))
    fig.savefig(out + "_B_spread.png"); fig.savefig(out + "_B_spread.pdf"); plt.close(fig)


def fan_image(tick, scene, step, ks, colours, alpha=0.12, include_nominal=True, nominal_col=None, fold=False,
              x0_ghost=True, outline_w=1):
    """the scene at x0 (robot as a faint grey ghost) with candidates ks painted as
    tinted silhouettes at horizon step; the nominal and the fold as outlines"""
    g0, m0 = scene.ghost(qpos_at(tick, -1, 0))
    base = scene.frame(qpos_at(tick, -1, 0))
    if x0_ghost:
        base = base.copy(); base[m0] = g0[m0]
        base = scene.composite(base, [(g0, m0, (0.95, 0.95, 0.95))], alpha=0.7)
    overlays = []; masks = []
    for k, col in zip(ks, colours):
        g, m = scene.ghost(qpos_at(tick, k, step))
        overlays.append((g, m, col)); masks.append((m, col))
    img = scene.composite(base, overlays, alpha=alpha)
    for m, col in masks:
        img = paint_outline(img, m, col, width=outline_w)
    if include_nominal:
        img = paint_outline(img, scene.mask(qpos_at(tick, -1, step)), nominal_col or (0.05, 0.05, 0.05), width=2)
    if fold:
        img = paint_outline(img, scene.mask(qpos_at(tick, -2, step)), mcolors.to_rgb(C_FOLD), width=3)
    return img


def fig_rollouts(tick, scene, out, box, fanbox, n_frames=6):
    meta = tick["meta"]; N = meta["n"]; ne = meta["n_elite"]
    sc = tick["scores"]
    order = sc.sort_values("rank").k.values
    J = sc.set_index("k").total_return
    col_of = {k: rank_colour(int(sc.loc[sc.k == k, "rank"].iloc[0]), N) for k in range(N)}
    H = meta["horizon_steps"] - 1
    # ---- C1: fan + returns
    fig = plt.figure(figsize=(16, 7.0))
    gs = gridspec.GridSpec(1, 2, figure=fig, left=0.01, right=0.98, top=0.83, bottom=0.10, wspace=0.10,
                           width_ratios=[1.45, 1])
    ax = fig.add_subplot(gs[0, 0])
    draw_fan(ax, tick, scene, H, list(order[::-1]), [col_of[k] for k in order[::-1]], fanbox)
    ax.set_title("all %d rollouts at h = %.1f s, tinted by J(k); traces: both hands and the pelvis over the horizon;\n"
                 "black = θ̄'s rollout; pale = x₀" % (N, H * meta["dt"]), fontsize=10.5, loc="left")
    axb = fig.add_subplot(gs[0, 1])
    ranks = np.arange(N)
    vals = J.loc[order].values
    axb.bar(ranks, vals, color=[col_of[k] for k in order], width=0.8)
    axb.axhline(tick["rollouts"][-1].cost.mean(), color=C_NOMINAL, lw=1.5, ls="--", label="J of θ̄'s rollout")
    axb.set_xticks(ranks); axb.set_xticklabels(["k=%d" % k for k in order], rotation=90, fontsize=8)
    axb.set_ylabel("J(k) = mean running cost over the horizon"); axb.set_xlabel("candidates, sorted by J")
    lo, hi = vals.min(), vals.max(); pad = 0.15 * (hi - lo + 1e-9)
    axb.set_ylim(lo - pad, hi + pad)
    axb.legend(loc="upper left", fontsize=9, frameon=False)
    axb.set_title("scores: best %.1f, worst %.1f over N = %d" % (lo, hi, N), fontsize=10.5, loc="left")
    title_box(fig, "5–6   u_t ← spline(θ(k); t), roll out f from x₀;  J(k) ← mean cost      Rollouts and scoring",
              "t = %.2f s.  Each of the %d splines is played open-loop through the MuJoCo model for %.1f s from the same x₀ "
              "and scored by the phase-scheduled cost." % (meta["t_plan"], N, meta["horizon_steps"] * meta["dt"]))
    fig.savefig(out + "_C1_rollouts_fan.png"); fig.savefig(out + "_C1_rollouts_fan.pdf"); plt.close(fig)
    # ---- C2: four candidate filmstrips (best, two middle, worst) with their running cost
    picks = [order[0], order[N // 3], order[2 * N // 3], order[-1]]
    labels = ["best  k=%d  J=%.1f", "rank %d  k=%%d  J=%%.1f" % (N // 3 + 1), "rank %d  k=%%d  J=%%.1f" % (2 * N // 3 + 1), "worst  k=%d  J=%.1f"]
    steps = horizon_steps(tick, n_frames)
    fig = plt.figure(figsize=(16, 2.9 * len(picks) + 1.0))
    gs = gridspec.GridSpec(len(picks), n_frames + 2, figure=fig, left=0.01, right=0.99, top=0.91, bottom=0.05,
                           wspace=0.04, hspace=0.30, width_ratios=[1] * n_frames + [0.12, 1.4])
    for i, (k, lab) in enumerate(zip(picks, labels)):
        gs_row = gridspec.GridSpecFromSubplotSpec(1, n_frames, subplot_spec=gs[i, :n_frames], wspace=0.04)
        filmstrip_row(fig, gs_row, scene, tick, int(k), steps, box, label=lab % (k, J[k]), label_col=mcolors.to_hex(col_of[k]))
        axc = fig.add_subplot(gs[i, n_frames + 1])
        r = tick["rollouts"][int(k)]; hh = r.t.values - r.t.values[0]
        r0 = tick["rollouts"][-1]
        axc.plot(hh, r0.cost.values, color=C_GREY, lw=1.2, label="θ̄")
        axc.plot(hh, r.cost.values, color=col_of[k], lw=2, label="k=%d" % k)
        axc.set_xlim(0, hh[-1]); axc.tick_params(labelsize=8)
        axc.set_ylabel("c(x_h,u_h)", fontsize=9)
        if i == len(picks) - 1: axc.set_xlabel("h (s)", fontsize=9)
        if i == 0: axc.legend(fontsize=8, frameon=False, loc="upper left")
    title_box(fig, "5–6   four of the %d rollouts, frame by frame, with their running cost" % N,
              "t = %.2f s.  Rows: best, two mid-ranked, worst by J(k).  Faint grey: x₀." % meta["t_plan"], y=0.99)
    fig.savefig(out + "_C2_rollouts_strips.png"); fig.savefig(out + "_C2_rollouts_strips.pdf"); plt.close(fig)
    # ---- C3: the fan as a filmstrip over the horizon
    fsteps = [int(round(f * H)) for f in (0.25, 0.5, 0.75, 1.0)]
    fig = plt.figure(figsize=(16, 5.9))
    gs = gridspec.GridSpec(1, len(fsteps), figure=fig, left=0.01, right=0.99, top=0.84, bottom=0.02, wspace=0.03)
    for j, st in enumerate(fsteps):
        ax = fig.add_subplot(gs[0, j])
        draw_fan(ax, tick, scene, st, list(order[::-1]), [col_of[k] for k in order[::-1]], fanbox)
        ax.set_title("h = %.2f s" % (st * meta["dt"]), fontsize=11, color="#444444", pad=4)
    title_box(fig, "5   all %d rollouts along the horizon" % N,
              "t = %.2f s.  Same x₀, %d splines, tinted by J(k) (yellow = lowest); black = θ̄'s rollout; traces: hands and pelvis."
              % (meta["t_plan"], N), y=0.985)
    fig.savefig(out + "_C3_fan_strip.png"); fig.savefig(out + "_C3_fan_strip.pdf"); plt.close(fig)


def fig_fold(tick, scene, out, box, fanbox, n_frames=6):
    meta = tick["meta"]; N = meta["n"]; ne = meta["n_elite"]
    sc = tick["scores"]; order = sc.sort_values("rank").k.values; J = sc.set_index("k").total_return
    elites = [int(k) for k in order[:ne]]
    col_of = {k: rank_colour(int(sc.loc[sc.k == k, "rank"].iloc[0]), N) for k in range(N)}
    H = meta["horizon_steps"] - 1
    tk, U0 = spline_values(tick["center"], tick["ucols"])
    _, Uf = spline_values(tick["fold"], tick["ucols"])
    T = (meta["horizon_steps"] - 1) * meta["dt"]
    sig = tick["sigma"]; nk, nu = meta["knots"], meta["nu"]
    Sn = sig.sigma_next.values.reshape(nk, nu); Su = sig.sigma_used.values.reshape(nk, nu)
    fig = plt.figure(figsize=(16, 7.6))
    gs = gridspec.GridSpec(1, 3, figure=fig, left=0.045, right=0.99, top=0.80, bottom=0.10, wspace=0.20,
                           width_ratios=[0.7, 1.55, 1.3])
    axb = fig.add_subplot(gs[0, 0])
    vals = J.loc[order].values; ranks = np.arange(N)
    cols = [col_of[k] if i < ne else C_GREY for i, k in enumerate(order)]
    axb.bar(ranks, vals, color=cols, width=0.8)
    axb.axvspan(-0.5, ne - 0.5, color=C_ELITE, alpha=0.12, lw=0)
    axb.set_xticks(ranks); axb.set_xticklabels(["%d" % k for k in order], fontsize=7); axb.set_xlabel("k, sorted by J")
    axb.set_ylabel("J(k)")
    lo, hi = vals.min(), vals.max(); pad = 0.15 * (hi - lo + 1e-9); axb.set_ylim(lo - pad, hi + pad)
    axb.set_title("the %d elites of %d" % (ne, N), fontsize=10.5, loc="left", color=C_ELITE)
    ax = fig.add_subplot(gs[0, 1])
    draw_fan(ax, tick, scene, H, elites[::-1], [col_of[k] for k in elites[::-1]], fanbox, alpha=0.25,
             nominal=False, fold=True, outline_w=1)
    ax.set_title("the %d elite rollouts at h = %.1f s, and the rollout of their mean θ (red), J = %.1f"
                 % (ne, H * meta["dt"], tick["rollouts"][-2].cost.mean()), fontsize=10.5, loc="left")
    sub = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[0, 2], hspace=0.6, wspace=0.4)
    for i, (j, name) in enumerate(KEY_JOINTS):
        axj = fig.add_subplot(sub[i // 4, i % 4])
        for k in range(N):
            if k in elites: continue
            _, Uk = spline_values(tick["cand"], tick["ucols"], k)
            step_line(axj, tk, Uk[:, j], T, color="#dddddd", lw=0.8, marker=False)
        for k in elites:
            _, Uk = spline_values(tick["cand"], tick["ucols"], k)
            step_line(axj, tk, Uk[:, j], T, color=col_of[k], lw=1.3, alpha=0.9, marker=False)
        step_line(axj, tk, U0[:, j], T, color=C_NOMINAL, lw=1.4, ls="--", marker=False)
        step_line(axj, tk, Uf[:, j], T, color=C_FOLD, lw=2.4)
        axj.set_title(name, fontsize=9.5, loc="left", pad=2); axj.tick_params(labelsize=8); axj.set_xlim(-0.02, T + 0.02)
        knot_ylim(axj, tick, j, T, extra=(tick["fold"],))
        if i // 4 == 1: axj.set_xlabel("h (s)", fontsize=9)
    note(fig, 0.665, 0.87, "red = mean of the %d elite splines; dashed black = θ̄;\ngrey = the %d dropped candidates" % (ne, N - ne),
         fontsize=10, color="#333333", va="top")
    title_box(fig, "8   θ ← Fold({θ(k)}, {J(k)})      how the winner is kept",
              "t = %.2f s.  CEM keeps the %d lowest-J candidates, sets θ to their mean and Σ to their variance (floored at "
              "%.0f mrad; refit mean %.1f → %.1f mrad); u₀ = spline(θ; 0) goes to the plant and θ shifts one step."
              % (meta["t_plan"], ne, meta["std_min"] * 1e3, Su.mean() * 1e3, Sn.mean() * 1e3))
    fig.savefig(out + "_D_fold.png"); fig.savefig(out + "_D_fold.pdf"); plt.close(fig)
    steps = horizon_steps(tick, n_frames)
    fig = plt.figure(figsize=(16, 7.2))
    gs = gridspec.GridSpec(2, 1, figure=fig, left=0.01, right=0.99, top=0.87, bottom=0.02, hspace=0.2)
    g0 = gridspec.GridSpecFromSubplotSpec(1, n_frames, subplot_spec=gs[0], wspace=0.03)
    g1 = gridspec.GridSpecFromSubplotSpec(1, n_frames, subplot_spec=gs[1], wspace=0.03)
    filmstrip_row(fig, g0, scene, tick, -1, steps, box, label="θ̄ (centre), J = %.1f" % tick["rollouts"][-1].cost.mean())
    filmstrip_row(fig, g1, scene, tick, -2, steps, box, label="θ after the fold, J = %.1f" % tick["rollouts"][-2].cost.mean(), label_col=C_FOLD)
    title_box(fig, "8   the folded plan against the centre it came from",
              "t = %.2f s.  Both rolled out from the same x₀.  The difference is one iteration of Algorithm 1." % meta["t_plan"])
    fig.savefig(out + "_D2_fold_strips.png"); fig.savefig(out + "_D2_fold_strips.pdf"); plt.close(fig)


def fig_overview(tick, scene, out, box, fanbox):
    """one 2x2 panel with the four stages side by side, for the summary slide"""
    meta = tick["meta"]; N = meta["n"]; ne = meta["n_elite"]
    sc = tick["scores"]; order = sc.sort_values("rank").k.values
    elites = [int(k) for k in order[:ne]]
    col_of = {k: rank_colour(int(sc.loc[sc.k == k, "rank"].iloc[0]), N) for k in range(N)}
    H = meta["horizon_steps"] - 1
    tk, U0 = spline_values(tick["center"], tick["ucols"])
    T = (meta["horizon_steps"] - 1) * meta["dt"]
    j, jname = KEY_JOINTS[0]
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.05, wspace=0.10, hspace=0.20)
    ax = axs[0, 0]
    draw_fan(ax, tick, scene, H, [-1], [(0.35, 0.35, 0.35)], fanbox, alpha=0.55, nominal=True, outline_w=1)
    ax.set_title("1  Center: θ̄ rolled out from x₀ (dark = h = %.1f s, with traces)" % (H * meta["dt"]), loc="left")
    ax = axs[0, 1]
    for k in range(N):
        _, Uk = spline_values(tick["cand"], tick["ucols"], k); step_line(ax, tk, Uk[:, j], T, color=C_GREY, lw=1, marker=False)
    step_line(ax, tk, U0[:, j], T, color=C_NOMINAL, lw=2.5); knot_ylim(ax, tick, j, T)
    ax.set_title("2–4  Spread: %d splines θ̄ + ε(k), %s target" % (N, jname), loc="left"); ax.set_xlabel("h (s)"); ax.set_ylabel("rad")
    ax = axs[1, 0]
    draw_fan(ax, tick, scene, H, list(order[::-1]), [col_of[k] for k in order[::-1]], fanbox)
    ax.set_title("5–6  Rollouts: %d end states and traces tinted by J(k)" % N, loc="left")
    ax = axs[1, 1]
    draw_fan(ax, tick, scene, H, elites[::-1], [col_of[k] for k in elites[::-1]], fanbox, alpha=0.25, nominal=False, fold=True)
    ax.set_title("8  Fold: the %d elites and the rollout of their mean (red)" % ne, loc="left")
    fig.suptitle("Algorithm 1, one iteration at t = %.2f s (%s): CEM, N = %d, k = %d, σ floor %.0f mrad, horizon %.1f s"
                 % (meta["t_plan"], meta["phase_name"], N, ne, meta["std_min_used"] * 1e3, T), x=0.04, ha="left", fontsize=13)
    fig.savefig(out + "_E_overview.png"); fig.savefig(out + "_E_overview.pdf"); plt.close(fig)


PAD_TRACE = [("geom:left_forearm_pad", 2.2), ("right_magpie_gripper", 1.4)]


def fig_contact(tick, scene, detail, out, fanbox, dbox, tol=0.003):
    """G: the brace-pad landing inside the rollouts. Detail camera on the pad,
    pad gap vs horizon per rollout, and the J bars split into landers / hoverers."""
    meta = tick["meta"]; N = meta["n"]; ne = meta["n_elite"]; H = meta["horizon_steps"] - 1; dt = meta["dt"]
    sc = tick["scores"]; order = sc.sort_values("rank").k.values; J = sc.set_index("k").total_return
    col_of = {k: rank_colour(int(sc.loc[sc.k == k, "rank"].iloc[0]), N) for k in range(N)}
    gaps = {k: detail.pad_gap(tick, k) for k in sorted(tick["rollouts"])}
    land = {k: (np.nonzero(g <= tol)[0][0] * dt if (g <= tol).any() else None) for k, g in gaps.items()}
    landers = [k for k in range(N) if land[k] is not None]; hoverers = [k for k in range(N) if land[k] is None]
    el_land = sum(1 for k in landers if k in set(order[:ne]))
    hh = np.arange(H + 1) * dt
    # ---- G1: detail fan + gap curves + J bars
    fig = plt.figure(figsize=(16, 7.2))
    gs = gridspec.GridSpec(1, 3, figure=fig, left=0.01, right=0.99, top=0.82, bottom=0.11, wspace=0.16,
                           width_ratios=[1.5, 1.05, 0.85])
    ax = fig.add_subplot(gs[0, 0])
    draw_fan(ax, tick, detail, H, list(order[::-1]), [col_of[k] for k in order[::-1]], dbox, alpha=0.10,
             nominal=True, fold=True, traces=True, trace_bodies=PAD_TRACE, outline_w=1)
    ax.set_title("the forearm pad at h = %.1f s, all %d rollouts tinted by J(k);\nblack outline = θ̄'s rollout, red = the fold's; traces = pad centre and right hand" % (H * dt, N), fontsize=10.5, loc="left")
    axg = fig.add_subplot(gs[0, 1])
    for k in order[::-1]:
        axg.plot(hh, gaps[k] * 1e3, color=col_of[k], lw=1.2, alpha=0.9)
    axg.plot(hh, gaps[-1] * 1e3, color=C_NOMINAL, lw=2.0, ls="--", label="θ̄ (centre)")
    axg.plot(hh, gaps[-2] * 1e3, color=C_FOLD, lw=2.4, label="θ after the fold")
    axg.axhline(0, color="#888888", lw=1); axg.axhspan(-30, tol * 1e3, color="#d9d3c7", alpha=0.5, lw=0)
    axg.set_xlim(0, H * dt); axg.set_ylim(min(-8, min(g.min() for g in gaps.values()) * 1e3 - 3), max(g.max() for g in gaps.values()) * 1e3 + 5)
    axg.set_xlabel("horizon time h (s)"); axg.set_ylabel("pad gap to the table face (mm)")
    axg.legend(fontsize=9, frameon=False, loc="upper right")
    axg.set_title("pad clearance along each rollout; %d of %d land inside the horizon\nθ̄: %s; fold: %s" % (
        len(landers), N,
        ("lands at h = %.2f s" % land[-1]) if land[-1] is not None else "hovers, %+.0f mm at the end" % (gaps[-1][-1] * 1e3),
        ("lands at h = %.2f s" % land[-2]) if land[-2] is not None else "hovers, %+.0f mm at the end" % (gaps[-2][-1] * 1e3)),
        fontsize=10.5, loc="left")
    axb = fig.add_subplot(gs[0, 2])
    ranks = np.arange(N); vals = J.loc[order].values
    for i, k in enumerate(order):
        axb.bar(i, vals[i], color=col_of[k], width=0.8, hatch=None if k in landers else "////", edgecolor="white" if k in landers else "#555555", lw=0.6)
    axb.axvspan(-0.5, ne - 0.5, color=C_ELITE, alpha=0.12, lw=0)
    axb.axhline(J_nom := tick["rollouts"][-1].cost.mean(), color=C_NOMINAL, lw=1.4, ls="--")
    axb.set_xticks(ranks); axb.set_xticklabels(["%d" % k for k in order], fontsize=7); axb.set_xlabel("k, sorted by J")
    lo, hi = vals.min(), vals.max(); pad_ = 0.15 * (hi - lo + 1e-9); axb.set_ylim(lo - pad_, hi + pad_)
    axb.set_ylabel("J(k)")
    axb.set_title("solid = lands the pad, hatched = hovers;\nelites (shaded): %d of %d are landers" % (el_land, ne), fontsize=10.5, loc="left")
    title_box(fig, "5–8   the brace contact inside the rollouts      what the sampler finds before the plant touches",
              "t = %.2f s, %s.  Pad at %+.0f mm above the slab at x₀.  Landed = pad within %.0f mm of the face." %
              (meta["t_plan"], meta["phase_name"], gaps[-1][0] * 1e3, tol * 1e3))
    fig.savefig(out + "_G1_contact.png"); fig.savefig(out + "_G1_contact.pdf"); plt.close(fig)
    # ---- G2: the detail fan as a filmstrip
    fsteps = [int(round(f * H)) for f in (0.25, 0.5, 0.75, 1.0)]
    fig = plt.figure(figsize=(16, 5.6))
    gs = gridspec.GridSpec(1, len(fsteps), figure=fig, left=0.01, right=0.99, top=0.84, bottom=0.02, wspace=0.03)
    for j, st in enumerate(fsteps):
        ax = fig.add_subplot(gs[0, j])
        draw_fan(ax, tick, detail, st, list(order[::-1]), [col_of[k] for k in order[::-1]], dbox, alpha=0.10,
                 nominal=True, fold=True, traces=True, trace_bodies=PAD_TRACE, outline_w=1)
        n_land = sum(1 for k in range(N) if land[k] is not None and land[k] <= st * dt + 1e-9)
        ax.set_title("h = %.2f s   pad down in %d of %d" % (st * dt, n_land, N), fontsize=11, color="#444444", pad=4)
    title_box(fig, "5   the pad landing along the horizon, all %d rollouts" % N,
              "t = %.2f s.  Tinted by J(k) (yellow = lowest); black outline = θ̄'s rollout, red = the fold's." % meta["t_plan"], y=0.985)
    fig.savefig(out + "_G2_contact_strip.png"); fig.savefig(out + "_G2_contact_strip.pdf"); plt.close(fig)
    return dict(landers=landers, hoverers=hoverers, elite_landers=el_land, land=land, gap0=float(gaps[-1][0]))


def fig_compare(ticks, labels, scene, out, fanbox):
    """the fan at h = T for several dumps side by side (e.g. three sigma settings)"""
    fig = plt.figure(figsize=(16, 6.2))
    gs = gridspec.GridSpec(1, len(ticks), figure=fig, left=0.01, right=0.99, top=0.84, bottom=0.02, wspace=0.03)
    for j, (tick, lab) in enumerate(zip(ticks, labels)):
        meta = tick["meta"]; N = meta["n"]; H = meta["horizon_steps"] - 1
        sc = tick["scores"]; order = sc.sort_values("rank").k.values
        col_of = {k: rank_colour(int(sc.loc[sc.k == k, "rank"].iloc[0]), N) for k in range(N)}
        ax = fig.add_subplot(gs[0, j])
        draw_fan(ax, tick, scene, H, list(order[::-1]), [col_of[k] for k in order[::-1]], fanbox)
        J = sc.total_return
        ax.set_title("%s\nJ: best %.0f, worst %.0f" % (lab, J.min(), J.max()), fontsize=11, color="#333333", pad=4)
    title_box(fig, "one planning iteration at three sampling widths",
              "separate runs of the same seed, each at t = %.2f s; all %d rollouts at h = %.1f s, tinted by J(k); black = θ̄'s rollout; pale = x₀."
              % (meta["t_plan"], N, H * meta["dt"]), y=0.985)
    fig.savefig(out + "_F_sigma_compare.png"); fig.savefig(out + "_F_sigma_compare.pdf"); plt.close(fig)


def spread_stats(tick, scene):
    """how different the rollouts are, in numbers: end-of-horizon spread of the pelvis and the two hands"""
    m = scene.m; d = scene.d
    H = tick["meta"]["horizon_steps"] - 1
    ids = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ["pelvis", "torso_link", "left_magpie_gripper", "right_magpie_gripper"]}
    rows = []
    for k in sorted(tick["rollouts"]):
        d.qpos[:] = qpos_at(tick, k, H); mujoco.mj_forward(m, d)
        rows.append(dict(k=k, **{n: d.xpos[i].copy() for n, i in ids.items()}))
    ref = rows[0]  # k = -2 (fold) sorts first; use nominal instead
    ref = [r for r in rows if r["k"] == -1][0]
    out = {}
    for n in ids:
        dev = np.array([np.linalg.norm(r[n] - ref[n]) for r in rows if r["k"] >= 0])
        out[n] = dict(mean_mm=float(dev.mean() * 1e3), max_mm=float(dev.max() * 1e3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tick", required=True, help="dump directory for one planning iteration")
    ap.add_argument("--out", required=True, help="output prefix (directory is created)")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--azimuth", type=float, default=90)
    ap.add_argument("--elevation", type=float, default=-8)
    ap.add_argument("--lookat", default="0.40,0,0.85")
    ap.add_argument("--distance", type=float, default=2.7)
    ap.add_argument("--box", default="0.20,0.03,0.72,0.99", help="filmstrip crop, fractions x0,y0,x1,y1")
    ap.add_argument("--fanbox", default="0.14,0.03,0.86,0.99", help="fan crop, fractions x0,y0,x1,y1")
    ap.add_argument("--only", default="", help="comma list of A,B,C,D,E to render (default all)")
    ap.add_argument("--clean", action="store_true", help="no title/subtitle text (slide versions)")
    ap.add_argument("--contact", action="store_true", help="also render the brace-contact figures (G1, G2) with a detail camera on the pad")
    ap.add_argument("--dbox", default="0.05,0.08,0.95,0.98", help="crop for the detail camera")
    ap.add_argument("--compare", default="", help="comma list of further tick dirs; renders the sigma comparison (F) of --tick + these")
    ap.add_argument("--labels", default="", help="labels for --compare (comma list, first = --tick)")
    a = ap.parse_args()
    global CLEAN; CLEAN = a.clean
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tick = load_tick(a.tick)
    scene = Scene(azimuth=a.azimuth, elevation=a.elevation, lookat=[float(x) for x in a.lookat.split(",")], distance=a.distance)
    box = [float(x) for x in a.box.split(",")]
    fanbox = [float(x) for x in a.fanbox.split(",")]
    st = spread_stats(tick, scene)
    print(json.dumps(dict(tick=a.tick, meta=tick["meta"], end_spread_vs_nominal=st,
                          J=dict(nominal=float(tick["rollouts"][-1].cost.mean()), fold=float(tick["rollouts"][-2].cost.mean()),
                                 best=float(tick["scores"].total_return.min()), worst=float(tick["scores"].total_return.max()))), indent=1))
    only = set(a.only.split(",")) if a.only else set("ABCDE")
    if "A" in only: fig_center(tick, scene, a.out, box, a.frames)
    if "B" in only: fig_spread(tick, scene, a.out)
    if "C" in only: fig_rollouts(tick, scene, a.out, box, fanbox, a.frames)
    if "D" in only: fig_fold(tick, scene, a.out, box, fanbox, a.frames)
    if "E" in only: fig_overview(tick, scene, a.out, box, fanbox)
    if a.contact:
        # detail camera: on the pad at x0, from the same side, 1.35 m away
        scene._set(qpos_at(tick, -1, 0))
        pad = mujoco.mj_name2id(scene.m, mujoco.mjtObj.mjOBJ_GEOM, "left_forearm_pad")
        look = scene.d.geom_xpos[pad].copy(); look[0] += 0.06; look[2] += 0.02
        detail = Scene(W=1200, H=750, azimuth=a.azimuth, elevation=-16, lookat=look, distance=1.7)
        dbox = [float(x) for x in a.dbox.split(",")]
        st = fig_contact(tick, scene, detail, a.out, fanbox, dbox)
        print(json.dumps(dict(contact=dict(landers=len(st["landers"]), elite_landers=st["elite_landers"], gap0_mm=round(st["gap0"] * 1e3, 1),
                                           land_nominal=st["land"][-1], land_fold=st["land"][-2]))))
    if a.compare:
        ticks = [tick] + [load_tick(d) for d in a.compare.split(",")]
        labels = a.labels.split(",") if a.labels else [a.tick] + a.compare.split(",")
        fig_compare(ticks, labels, scene, a.out, fanbox)
    print("->", a.out + "_{A_center,B_spread,C1_rollouts_fan,C2_rollouts_strips,C3_fan_strip,D_fold,D2_fold_strips,E_overview}.{png,pdf}")


if __name__ == "__main__":
    main()
