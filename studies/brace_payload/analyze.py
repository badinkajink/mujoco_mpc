#!/usr/bin/env python3
"""Tables and figures for the payload-belief sweep (sweep.py output)."""
import argparse, json, os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130})


def load(run, attached_only=True):
    rows = []
    for line in open(os.path.join(run, "results.jsonl")):
        r = json.loads(line)
        for k in ("fell", "complete", "max_phase"):
            r[k] = int(r.get(k, -1))
        for k in ("peak_brace_carry", "min_pelvis_carry", "max_tilt_carry", "t_complete", "t_end", "payload_t"):
            r[k] = float(r.get(k, "nan"))
        r["attached"] = r["payload_t"] >= 0.0
        r["carried"] = int(r["attached"] and r["fell"] == 0 and r["max_phase"] >= 10)
        rows.append(r)
    return [r for r in rows if r["attached"]] if attached_only else rows


def grid(rows, arm):
    ms = sorted({r["m"] for r in rows}); bs = sorted({r["b"] for r in rows})
    out = {}
    for m in ms:
        for b in bs:
            rs = [r for r in rows if r["arm"] == arm and r["m"] == m and r["b"] == b]
            if not rs:
                continue
            out[(m, b)] = {"n": len(rs), "complete": sum(r["complete"] == 1 for r in rs),
                           "carried": sum(r["carried"] for r in rs),
                           "fell": sum(r["fell"] == 1 for r in rs),
                           "peak_brace_med": float(np.median([r["peak_brace_carry"] for r in rs])),
                           "min_pelvis_med": float(np.median([r["min_pelvis_carry"] for r in rs])),
                           "max_tilt_med": float(np.median([r["max_tilt_carry"] for r in rs])),
                           "max_phase_med": float(np.median([r["max_phase"] for r in rs]))}
    return ms, bs, out


def heat(ax, ms, bs, g, key, title, fmt, vmax):
    M = np.full((len(ms), len(bs)), np.nan)
    for i, m in enumerate(ms):
        for j, b in enumerate(bs):
            if (m, b) in g:
                M[i, j] = g[(m, b)][key] / (g[(m, b)]["n"] if key in ("complete", "fell", "carried") else 1.0)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#f4f8fe"] + SEQ)
    ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i, m in enumerate(ms):
        for j, b in enumerate(bs):
            if (m, b) in g:
                st = g[(m, b)]
                txt = fmt(st)
                col = "#ffffff" if (M[i, j] / vmax) > 0.6 else INK
                ax.text(j, i, txt, ha="center", va="center", color=col, fontsize=8.5)
    ax.set_xticks(range(len(bs))); ax.set_xticklabels([f"{b:g} kg" for b in bs])
    ax.set_yticks(range(len(ms))); ax.set_yticklabels([f"{m:g} kg" for m in ms])
    ax.set_xlabel("believed payload"); ax.set_ylabel("true payload")
    ax.set_title(title, loc="left", fontsize=10); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_grids(rows, media):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2))
    for k, arm in enumerate(["P", "M"]):
        ms, bs, g = grid(rows, arm)
        if not g:
            continue
        name = {"P": "Arm P: belief sets model + brace preload", "M": "Arm M: belief sets model only"}[arm]
        heat(axes[k, 0], ms, bs, g, "carried", f"{name}\ncarried through release / attached runs",
             lambda st: f"{st['carried']}/{st['n']}", 1.0)
        heat(axes[k, 1], ms, bs, g, "fell", "fell after the attach / attached runs  (median peak brace)",
             lambda st: f"{st['fell']}/{st['n']}\n{st['peak_brace_med']:.0f} N", 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(media, "fig_payload_grid.png"), bbox_inches="tight")
    plt.close(fig)


def fig_trace(run, tags, media, labels, name="fig_payload_trace.png"):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.0))
    for k, tag in enumerate(tags):
        p = os.path.join(run, tag + ".csv")
        if not os.path.exists(p):
            continue
        rs = list(csv.DictReader(open(p)))
        t = np.array([float(r["t"]) for r in rs])
        # align on the payload attach time (phase 5 entry)
        ph = np.array([int(r["phase"]) for r in rs])
        idx = np.where(ph >= 5)[0]
        t0 = t[idx[0]] if len(idx) else t[0]
        sel = (t >= t0 - 5) & (t <= t0 + 60)
        axes[0].plot(t[sel] - t0, [float(r["brace_normal_N"]) for r in np.array(rs, dtype=object)[sel]], color=CAT[k], lw=1.4, label=labels[k])
        axes[1].plot(t[sel] - t0, [float(r["com_beyond_foot_edge"]) * 100 for r in np.array(rs, dtype=object)[sel]], color=CAT[k], lw=1.4)
        axes[2].plot(t[sel] - t0, [float(r["pelvis_z"]) for r in np.array(rs, dtype=object)[sel]], color=CAT[k], lw=1.4)
    axes[0].set_ylabel("Brace normal force (N)"); axes[1].set_ylabel("CoM beyond foot edge (cm)"); axes[2].set_ylabel("Pelvis height (m)")
    for ax in axes:
        ax.set_xlabel("time since the payload attached (s)"); ax.grid(True, color=GRID, lw=0.6)
        ax.axvline(0, color=MUTED, lw=1, ls="--")
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(media, name), bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(HERE, "runs/sweep1"))
    ap.add_argument("--media", default=os.path.join(HERE, "../../docs/brace_payload/media"))
    a = ap.parse_args()
    os.makedirs(a.media, exist_ok=True)
    allrows = load(a.run, attached_only=False)
    rows = [r for r in allrows if r["attached"]]
    print(f"{len(allrows)} runs, {len(rows)} reached the attach; median wall {np.median([r['wall_s'] for r in allrows]):.0f} s")
    summary = {}
    for arm in ["P", "M"]:
        ms, bs, g = grid(rows, arm)
        if not g:
            continue
        summary[arm] = {f"{m:g}|{b:g}": v for (m, b), v in g.items()}
        print(f"Arm {arm}: rows = true payload, cols = believed payload  (completed/n, fell/n, med peak brace N, med min pelvis)")
        print("           " + "".join(f"{b:>22g} kg" for b in bs))
        for m in ms:
            line = f"  true {m:4g} kg "
            for b in bs:
                st = g.get((m, b))
                line += (f"  {st['complete']}/{st['n']} f{st['fell']} {st['peak_brace_med']:4.0f}N {st['min_pelvis_med']:.2f}" if st else " " * 25)
            print(line)
    json.dump(summary, open(os.path.join(a.run, "summary.json"), "w"), indent=1)
    fig_grids(rows, a.media)
    fig_trace(a.run, ["P_m0_b0_s0", "P_m4_b0_s0", "P_m1_b4_s1", "M_m4_b4_s0"], a.media,
              ["no payload, no belief", "4 kg carried, believed 0", "1 kg, believed 4 kg: fell at release",
               "4 kg, believed 4 kg: fell at release"])


if __name__ == "__main__":
    main()
