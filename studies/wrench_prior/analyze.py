#!/usr/bin/env python3
"""Tables and figures for the wrench-prior push sweeps (sweep.py output).

Reads <run>/results.jsonl, writes <run>/summary.json and the figures under
<out_media>. Colors follow the reference dataviz palette (blue sequential,
categorical slots in fixed order); text stays in ink, never in series color.
"""
import argparse, json, os, collections, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OBJ_ORDER = ["light", "nominal", "heavyish", "slippery", "grippy", "heavy"]
OBJ_LABEL = {"light": "0.2 kg, mu 0.4", "nominal": "0.5 kg, mu 0.4", "heavyish": "1.0 kg, mu 0.4",
             "slippery": "0.5 kg, mu 0.2", "grippy": "0.5 kg, mu 0.8", "heavy": "2.0 kg, mu 0.4"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130})


def load(run):
    rows = [json.loads(l) for l in open(os.path.join(run, "results.jsonl"))]
    return [r for r in rows if r.get("outcome") != "error"]


def cell_stats(rs):
    n = len(rs)
    succ = sum(r["outcome"] == "success" for r in rs)
    return {"n": n, "success": succ,
            "outcomes": dict(collections.Counter(r["outcome"] for r in rs)),
            "t_done_med": float(np.median([r["t_done"] for r in rs if r["outcome"] == "success"])) if succ else None,
            "final_err_med": float(np.median([r["final_err"] for r in rs])),
            "peak_f_lp_med": float(np.median([r["peak_f_lp"] for r in rs])),
            "peak_f_lp_max": float(np.max([r["peak_f_lp"] for r in rs])),
            "peak_box_v_med": float(np.median([r.get("peak_box_v", float("nan")) for r in rs]))}


def sweep_a(rows):
    A = [r for r in rows if r["sweep"] == "A"]
    ratios_m = sorted({round(r["m_hat"] / r["m"], 4) for r in A if abs(r["mu_hat"] - r["mu"]) < 1e-9})
    ratios_mu = sorted({round(r["mu_hat"] / r["mu"], 4) for r in A if abs(r["m_hat"] - r["m"]) > 1e-9 or abs(r["mu_hat"] - r["mu"]) > 1e-9} - {1.0})
    grid_m = {o: {} for o in OBJ_ORDER}
    grid_mu = {o: {} for o in OBJ_ORDER}
    for o in OBJ_ORDER:
        for rm in ratios_m:
            rs = [r for r in A if r["obj"] == o and abs(r["m_hat"] / r["m"] - rm) < 1e-6 and abs(r["mu_hat"] - r["mu"]) < 1e-9]
            if rs:
                grid_m[o][rm] = cell_stats(rs)
        for rmu in ratios_mu:
            rs = [r for r in A if r["obj"] == o and abs(r["mu_hat"] / r["mu"] - rmu) < 1e-6 and abs(r["m_hat"] - r["m"]) < 1e-9]
            if rs:
                grid_mu[o][rmu] = cell_stats(rs)
    return {"ratios_m": ratios_m, "ratios_mu": ratios_mu, "grid_m": grid_m, "grid_mu": grid_mu}


def sweep_b(rows):
    B = [r for r in rows if r["sweep"] == "B"]
    ratios = sorted({round(r["f_max"] / (r["mu"] * r["m"] * 9.81), 4) for r in B})
    grid = {o: {} for o in OBJ_ORDER}
    for o in OBJ_ORDER:
        for rc in ratios:
            rs = [r for r in B if r["obj"] == o and abs(r["f_max"] / (r["mu"] * r["m"] * 9.81) - rc) < 1e-6]
            if rs:
                grid[o][rc] = cell_stats(rs)
    return {"ratios": ratios, "grid": grid, "rows": [{k: r[k] for k in ("obj", "m", "mu", "f_max", "seed", "outcome",
                                                                     "t_done", "final_err", "peak_f_lp", "peak_box_v")} for r in B]}


def heatmap(ax, grid, cols, title, xlabel, fmt_col):
    objs = [o for o in OBJ_ORDER if grid.get(o)]
    M = np.full((len(objs), len(cols)), np.nan)
    N = np.zeros_like(M)
    for i, o in enumerate(objs):
        for j, c in enumerate(cols):
            st = grid[o].get(c)
            if st:
                M[i, j] = st["success"] / st["n"]
                N[i, j] = st["n"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#f4f8fe"] + SEQ)
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(objs)):
        for j in range(len(cols)):
            if not np.isnan(M[i, j]):
                st = grid[objs[i]][cols[j]]
                col = "#ffffff" if M[i, j] > 0.6 else INK
                ax.text(j, i, f"{st['success']}/{st['n']}", ha="center", va="center", color=col, fontsize=9)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([fmt_col(c) for c in cols])
    ax.set_yticks(range(len(objs)))
    ax.set_yticklabels([OBJ_LABEL[o] for o in objs])
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_success(a, b, out):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), gridspec_kw={"width_ratios": [7, 2, 6]})
    heatmap(axes[0], a["grid_m"], a["ratios_m"], "Successes / seeds vs believed mass",
            "believed mass / true mass (mu believed correctly)", lambda c: f"{c:g}x")
    heatmap(axes[1], a["grid_mu"], a["ratios_mu"], "vs believed friction",
            "believed mu / true mu", lambda c: f"{c:g}x")
    axes[1].set_yticklabels([])
    heatmap(axes[2], b["grid"], b["ratios"], "Successes / seeds vs wrench cap (belief correct)",
            "F_max / mu m g of the true object", lambda c: f"{c:g}x")
    axes[2].set_yticklabels([])
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_success.png"), bbox_inches="tight")
    plt.close(fig)


def fig_error_vs_ratio(a, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for k, o in enumerate(OBJ_ORDER):
        if not a["grid_m"].get(o):
            continue
        xs = [x for x in a["ratios_m"] if x in a["grid_m"][o]]
        ys = [a["grid_m"][o][x]["final_err_med"] * 100 for x in xs]
        axes[0].plot(xs, ys, "-o", color=CAT[k], ms=4, lw=1.6, label=OBJ_LABEL[o])
        fs = [a["grid_m"][o][x]["peak_box_v_med"] for x in xs]
        axes[1].plot(xs, fs, "-o", color=CAT[k], ms=4, lw=1.6, label=OBJ_LABEL[o])
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(a["ratios_m"])
        ax.set_xticklabels([f"{c:g}" for c in a["ratios_m"]])
        ax.set_xlabel("believed mass / true mass")
        ax.grid(True, color=GRID, lw=0.6)
    axes[0].axhline(3.0, color=MUTED, lw=1, ls="--")
    axes[0].text(a["ratios_m"][0], 3.2, "success tolerance 3 cm", color=MUTED, fontsize=8)
    axes[0].set_ylabel("Final box error, median (cm)")
    axes[0].set_title("Final error vs mass belief", loc="left", fontsize=10)
    axes[1].set_ylabel("Peak box speed, median (m/s)")
    axes[1].set_title("How hard the box is shoved vs mass belief", loc="left", fontsize=10)
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_error_vs_ratio.png"), bbox_inches="tight")
    plt.close(fig)


def fig_cap(b, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    rows = b["rows"]
    for k, o in enumerate(OBJ_ORDER):
        rs = [r for r in rows if r["obj"] == o]
        if not rs:
            continue
        fs = np.array([r["f_max"] for r in rs])
        pk = np.array([r["peak_f_lp"] for r in rs])
        ok = np.array([r["outcome"] == "success" for r in rs])
        axes[0].scatter(fs[ok], pk[ok], s=22, color=CAT[k], label=OBJ_LABEL[o], zorder=3, edgecolor="white", lw=0.6)
        axes[0].scatter(fs[~ok], pk[~ok], s=30, facecolor="none", edgecolor=CAT[k], lw=1.2, zorder=3, marker="s")
    lim = max(r["f_max"] for r in rows) * 1.05
    axes[0].plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--")
    axes[0].text(lim * 0.62, lim * 0.66, "peak = cap", color=MUTED, fontsize=8, rotation=38)
    axes[0].set_xlabel("Planner wrench cap F_max (N)")
    axes[0].set_ylabel("Peak 50 ms-filtered contact force (N)")
    axes[0].set_title("Peak contact force vs cap (filled = success)", loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[0].grid(True, color=GRID, lw=0.6)
    xs = b["ratios"]
    n_tot = [sum(1 for r in rows if abs(r["f_max"] / (r["mu"] * r["m"] * 9.81) - x) < 1e-6) for x in xs]
    n_unt = [sum(1 for r in rows if abs(r["f_max"] / (r["mu"] * r["m"] * 9.81) - x) < 1e-6 and r["peak_f_lp"] < 0.05) for x in xs]
    n_ok = [sum(1 for r in rows if abs(r["f_max"] / (r["mu"] * r["m"] * 9.81) - x) < 1e-6 and r["outcome"] == "success") for x in xs]
    axes[1].plot(xs, [u / t for u, t in zip(n_unt, n_tot)], "-o", color=CAT[1], ms=4, lw=1.6, label="never touched the box")
    axes[1].plot(xs, [u / t for u, t in zip(n_ok, n_tot)], "-o", color=CAT[0], ms=4, lw=1.6, label="success")
    for x, u, t in zip(xs, n_unt, n_tot):
        axes[1].annotate(f"{u}/{t}", (x, u / t), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.5, color=INK2)
    axes[1].set_xlabel("F_max / mu m g (all six objects pooled)")
    axes[1].set_ylabel("Fraction of episodes")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Untouched and successful episodes vs cap", loc="left", fontsize=10)
    axes[1].legend(frameon=False, fontsize=7.5, loc="center right")
    axes[1].grid(True, color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_cap.png"), bbox_inches="tight")
    plt.close(fig)


def fig_trace(run, tags, out, name="fig_trace.png", labels=None):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.0))
    for k, tag in enumerate(tags):
        p = os.path.join(run, "logs", tag + ".npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        lab = labels[k] if labels else tag
        axes[0].plot(z["t"], z["err"] * 100, color=CAT[k], lw=1.6, label=lab)
        axes[1].plot(z["t"], z["f_lp"], color=CAT[k], lw=1.6, label=lab)
        axes[2].plot(z["t"], 10 * np.linalg.norm(z["u"][:, :3], axis=1), color=CAT[k], lw=1.2, label=lab)
    axes[0].set_ylabel("Box error to target (cm)")
    axes[1].set_ylabel("Contact force on box, 50 ms filtered (N)")
    axes[2].set_ylabel("Commanded wrist force |F| (N)")
    for ax in axes:
        ax.set_xlabel("time since the push began (s)")
        ax.grid(True, color=GRID, lw=0.6)
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(HERE, "runs/sweep1"))
    ap.add_argument("--media", default=os.path.join(HERE, "../../docs/wrench_prior/media"))
    a = ap.parse_args()
    os.makedirs(a.media, exist_ok=True)
    rows = load(a.run)
    A, B = sweep_a(rows), sweep_b(rows)
    summary = {"n_rows": len(rows), "A": A, "B": B,
               "wall_s_median": float(np.median([r["wall_s"] for r in rows])),
               "plan_wall_mean": float(np.mean([r["plan_wall_mean"] for r in rows]))}
    json.dump(summary, open(os.path.join(a.run, "summary.json"), "w"), indent=1, default=str)
    fig_success(A, B, a.media)
    fig_error_vs_ratio(A, a.media)
    fig_cap(B, a.media)
    fig_trace(a.run, ["A_nominal_m1_s0", "A_nominal_m0.1_s0", "A_nominal_m10_s0"], a.media,
              labels=["belief = truth (0.5 kg)", "belief 0.05 kg (0.1x): slow", "belief 5 kg (10x): shoved off-axis"])
    fig_trace(a.run, ["B_nominal_cap0.5_s0", "B_nominal_cap1_s0", "B_nominal_cap3_s0"], a.media,
              name="fig_trace_cap.png", labels=["cap 0.5x mu m g", "cap 1x", "cap 3x"])
    # console summary
    print(f"{len(rows)} rows; median wall {summary['wall_s_median']:.0f} s; plan {summary['plan_wall_mean']*1000:.0f} ms")
    print("Sweep A (mass belief):  " + "  ".join(f"{r:g}x" for r in A["ratios_m"]))
    for o in OBJ_ORDER:
        if A["grid_m"].get(o):
            print(f"  {OBJ_LABEL[o]:16s} " + "  ".join(f"{A['grid_m'][o][r]['success']}/{A['grid_m'][o][r]['n']} ({A['grid_m'][o][r]['final_err_med']*100:4.1f}cm)" for r in A["ratios_m"] if r in A["grid_m"][o]))
    print("Sweep A (mu belief):  " + "  ".join(f"{r:g}x" for r in A["ratios_mu"]))
    for o in OBJ_ORDER:
        if A["grid_mu"].get(o):
            print(f"  {OBJ_LABEL[o]:16s} " + "  ".join(f"{A['grid_mu'][o][r]['success']}/{A['grid_mu'][o][r]['n']} ({A['grid_mu'][o][r]['final_err_med']*100:4.1f}cm)" for r in A["ratios_mu"] if r in A["grid_mu"][o]))
    print("Sweep B (cap):  " + "  ".join(f"{r:g}x" for r in B["ratios"]))
    for o in OBJ_ORDER:
        if B["grid"].get(o):
            print(f"  {OBJ_LABEL[o]:16s} " + "  ".join(f"{B['grid'][o][r]['success']}/{B['grid'][o][r]['n']} (pk {B['grid'][o][r]['peak_f_lp_med']:4.1f}N, {B['grid'][o][r]['final_err_med']*100:4.1f}cm)" for r in B["ratios"] if r in B["grid"][o]))


if __name__ == "__main__":
    main()
