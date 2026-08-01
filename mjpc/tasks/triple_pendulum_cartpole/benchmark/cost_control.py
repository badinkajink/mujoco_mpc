#!/usr/bin/env python3
"""Cost-vs-control characterization for corridor_benchmark dumps.

A dump records the total cost per step but not its decomposition, and total cost
on this task is actively misleading -- driving through with the pendulum
whirling scores better than threading the corridor. This replays a dump through
the model, recomputes every cost TERM with the weights the run used, and puts
them next to the control signal, so "what was the planner paying for" and "what
did it do with the actuator" can be read off the same time axis.

Per dump it emits:
  <out>.png    four panels: stacked cost terms, control trace, cost-vs-control
               phase scatter, and the per-term share while inside a bottleneck
  <out>.csv    per-step term breakdown, for anything this script does not plot

Across dumps, --summary appends one row per run to a shared CSV: control
saturation, effort, reversal rate, and where the cost went. That table is the
thing to sort when comparing weight/planner configurations.

Usage:
  cost_control.py --dump renders/x_0.csv --weights 1,0,0.1,0.01,8000 \
      --out renders/x_0 --summary renders/cost_control_summary.csv --label w8000
"""
import argparse
import csv
import math
import os
import pathlib

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TERMS = ["Cart", "Upright", "Velocity", "Control", "Avoidance"]


def task_of(path):
    with open(path) as f:
        for line in f:
            if line.startswith("# task="):
                return line.split("=", 1)[1].strip()
            if not line.startswith("#"):
                break
    return "corridor"


def xml_for(task):
    here = pathlib.Path(__file__).resolve().parent.parent
    return str(here / ("slalom.xml" if task == "slalom" else "task.xml"))


def load(path):
    with open(path) as f:
        return [r for r in csv.DictReader(l for l in f if not l.startswith("#"))]


def analyse(dump, weights, margin):
    task = task_of(dump)
    m = mujoco.MjModel.from_xml_path(xml_for(task))
    d = mujoco.MjData(m)
    heads = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s)
             for s in ("head1", "head2", "tip")]
    head_r = [m.geom_size[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)][0]
              for g in ("head1", "head2", "head3")]
    obst = [g for g in range(m.ngeom)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
            .startswith("obstacle")]
    orad = [m.geom_size[g][0] for g in obst]
    gaps = sorted({float(m.geom_pos[g][0]) for g in obst})
    goal_i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, "residual_Goal")
    goal = float(m.numeric_data[m.numeric_adr[goal_i]])
    umax = float(m.actuator_ctrlrange[0, 1])

    rows = load(dump)
    out = {k: [] for k in TERMS}
    t, u, cart, clr = [], [], [], []
    for r in rows:
        d.qpos[:] = [float(r[k]) for k in ("cart", "th1", "th2", "th3")]
        d.qvel[:] = [float(r[k]) for k in ("dcart", "dth1", "dth2", "dth3")]
        ui = float(r["ctrl"])
        d.ctrl[0] = ui
        mujoco.mj_forward(m, d)

        out["Cart"].append(weights[0] * abs(d.qpos[0] - goal))
        out["Upright"].append(weights[1] * sum(
            abs(math.cos(d.qpos[1 + i]) - 1.0) for i in range(3)))
        out["Velocity"].append(weights[2] * float(np.abs(d.qvel).sum()))
        out["Control"].append(weights[3] * abs(ui))
        avoid = 0.0
        cmin = 1e9
        for h, hr in zip(heads, head_r):
            p = d.site_xpos[h]
            for g, rr in zip(obst, orad):
                c = d.geom_xpos[g]
                gap = math.hypot(p[0] - c[0], p[2] - c[2]) - rr - hr
                cmin = min(cmin, gap)
                avoid += max(0.0, margin - gap)
        out["Avoidance"].append(weights[4] * avoid)
        t.append(float(r["time"]))
        u.append(ui)
        cart.append(d.qpos[0])
        clr.append(cmin)

    return dict(task=task, t=np.array(t), u=np.array(u), cart=np.array(cart),
                clr=np.array(clr), terms={k: np.array(v) for k, v in out.items()},
                gaps=gaps, umax=umax, goal=goal)


def plot(a, out_png, title):
    t, u, terms = a["t"], a["u"], a["terms"]
    total = sum(terms.values())
    fig, ax = plt.subplots(2, 2, figsize=(13, 7.5))
    fig.suptitle(title, fontsize=11)

    # crossing times, for vertical rules on the time-axis panels
    xt = []
    for gx in a["gaps"]:
        i = np.argmax(a["cart"] >= gx) if (a["cart"] >= gx).any() else None
        if i:
            xt.append(t[i])

    ax0 = ax[0, 0]
    ax0.stackplot(t, [terms[k] for k in TERMS], labels=TERMS,
                  colors=["#2a78d6", "#eb6834", "#1baf7a", "#9467bd", "#d0402c"])
    ax0.set_yscale("symlog", linthresh=1.0)
    ax0.set_title("cost terms (weighted)")
    ax0.set_xlabel("s"); ax0.legend(fontsize=7, loc="upper right", ncol=2)

    ax1 = ax[0, 1]
    ax1.axhspan(-a["umax"], a["umax"], color="0.92", zorder=0)
    ax1.plot(t, u, lw=0.8, color="#11141a")
    ax1.set_title(f"control (|u| <= {a['umax']:.0f} N), "
                  f"saturated {100*np.mean(np.abs(u) >= 0.999*a['umax']):.0f}%")
    ax1.set_xlabel("s"); ax1.set_ylabel("N")

    ax2 = ax[1, 0]
    s = ax2.scatter(u, total, c=t, s=4, cmap="viridis")
    ax2.set_yscale("symlog", linthresh=1.0)
    ax2.set_xlabel("control (N)"); ax2.set_ylabel("total cost")
    ax2.set_title("cost vs control, coloured by time")
    fig.colorbar(s, ax=ax2, label="s")

    ax3 = ax[1, 1]
    ax3.plot(a["cart"], a["clr"], lw=0.8, color="#11141a")
    ax3.axhline(0, color="#d0402c", lw=1, ls="--")
    for gx in a["gaps"]:
        ax3.axvline(gx, color="0.7", lw=0.8)
    ax3.set_xlabel("cart x (m)"); ax3.set_ylabel("min clearance (m)")
    ax3.set_ylim(-0.15, 0.6)
    ax3.set_title("clearance along the course (dashed = contact)")

    for a_ in (ax0, ax1):
        for x in xt:
            a_.axvline(x, color="0.7", lw=0.8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", required=True)
    p.add_argument("--weights", required=True,
                   help="cart,upright,velocity,control,avoidance as run")
    p.add_argument("--margin", type=float, default=0.08)
    p.add_argument("--out", required=True, help="path prefix for .png and .csv")
    p.add_argument("--summary", default=None, help="append one row here")
    p.add_argument("--label", default="")
    args = p.parse_args()

    w = [float(x) for x in args.weights.split(",")]
    a = analyse(args.dump, w, args.margin)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot(a, args.out + ".png", f"{args.label}  weights={args.weights} "
                               f"margin={args.margin}")

    with open(args.out + ".csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["time", "cart", "ctrl", "min_clearance"] + TERMS + ["total"])
        tot = sum(a["terms"].values())
        for i in range(len(a["t"])):
            wr.writerow([f"{a['t'][i]:.4f}", f"{a['cart'][i]:.4f}",
                         f"{a['u'][i]:.4f}", f"{a['clr'][i]:.4f}"]
                        + [f"{a['terms'][k][i]:.4f}" for k in TERMS]
                        + [f"{tot[i]:.4f}"])

    u, t = a["u"], a["t"]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.005
    rev = int(np.sum(np.diff(np.sign(u)) != 0))
    tot = sum(a["terms"].values())
    row = dict(
        label=args.label, dump=os.path.basename(args.dump),
        weights=args.weights, margin=args.margin,
        sim_s=f"{t[-1]:.2f}", max_cart=f"{a['cart'].max():.3f}",
        min_clearance=f"{a['clr'].min():.4f}",
        saturated_pct=f"{100*np.mean(np.abs(u) >= 0.999*a['umax']):.1f}",
        mean_abs_u=f"{np.abs(u).mean():.3f}",
        effort_int_u2=f"{float(np.sum(u**2)*dt):.1f}",
        reversals_per_s=f"{rev/max(1e-9, t[-1]):.1f}",
        mean_total_cost=f"{tot.mean():.2f}",
        **{f"share_{k}": f"{100*a['terms'][k].sum()/max(1e-9, tot.sum()):.1f}"
           for k in TERMS})
    if args.summary:
        new = not os.path.exists(args.summary)
        with open(args.summary, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                wr.writeheader()
            wr.writerow(row)
    print("  ".join(f"{k}={v}" for k, v in row.items()))


if __name__ == "__main__":
    main()
