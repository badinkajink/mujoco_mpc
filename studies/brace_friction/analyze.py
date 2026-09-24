#!/usr/bin/env python3
"""Score friction-sweep runs from their per-step contact records.

For every run in <runs>/results.jsonl, reads <tag>.contact.csv (2 ms rows, see
lean_bench --contact_out) and writes one row of derived quantities to
<runs>/scored.jsonl, then prints a per-cell table.

Definitions (the page quotes these):
  loaded brace      brace-pad normal force pd_fz > 20 N (forearm + wrist pads on
                    the slab; the left gripper jaw rests on the slab through
                    stand-up and is counted only in the whole-arm br_* columns)
  touchdown         first row in phase >= 1 with pd_fn > 10 N
  touchdown speed   pad-centre speed averaged over the 20 ms before touchdown,
                    split into down (-v_z) and along-table (|v_xy|)
  shear ratio       |F_xy| / F_z of a surface's summed contact force on the robot
                    (brace: slab top; foot: floor). The friction coefficient the
                    contact needs to hold that load without sliding. Smoothed
                    with a 10 ms running median before any peak is taken.
  slide             integral of the contact-point slip speed over time while the
                    surface is loaded (brace F_z > 20 N, foot F_z > 50 N): the
                    distance the contact actually slid, in mm
  slip onset        first time the slip speed stays above 20 mm/s for 40 ms
                    while loaded
"""
import argparse, json, os
import numpy as np
import pandas as pd

F_BRACE, F_FOOT = 20.0, 50.0
V_SLIP, T_SLIP = 0.020, 0.040
DT = 0.002


def med10(x):
    return pd.Series(x).rolling(5, center=True, min_periods=1).median().to_numpy().copy()


def onset(t, v, mask):
    """First t where v > V_SLIP for T_SLIP while mask holds."""
    need = int(round(T_SLIP / DT))
    run = 0
    hit = (v > V_SLIP) & mask
    for i, h in enumerate(hit):
        run = run + 1 if h else 0
        if run >= need:
            return float(t[i - need + 1])
    return None


def score(tag, runs, rec):
    c = pd.read_csv(os.path.join(runs, tag + ".contact.csv"))
    t = c.t.to_numpy()
    out = {"tag": tag, "cell": rec["cell"], "seed": rec["seed"],
           "plant_table_mu": rec["plant_table_mu"], "plant_foot_mu": rec["plant_foot_mu"],
           "planner_table_mu": rec["planner_table_mu"], "planner_foot_mu": rec["planner_foot_mu"],
           "fell": int(rec.get("fell", 0) or 0), "complete": int(rec.get("complete", 0) or 0),
           "t_end": float(rec.get("t_end", "nan")), "t_complete": float(rec.get("t_complete", "nan"))}
    enter = [float(x) for x in rec.get("enter", "").split(":") if x]
    out["enter"] = enter
    ph = c.phase.to_numpy()
    # ---- brace (the pads)
    brz = c.pd_fz.to_numpy()
    loaded = brz > F_BRACE
    out["brace_loaded_s"] = float(loaded.sum() * DT)
    out["brace_peak_N"] = float(med10(brz).max()) if len(brz) else 0.0
    out["arm_peak_N"] = float(med10(c.br_fz.to_numpy()[ph >= 1]).max()) if (ph >= 1).any() else 0.0
    td = np.flatnonzero((ph >= 1) & (c.pd_fn.to_numpy() > 10.0))
    if len(td):
        i0 = td[0]
        out["t_touch"] = float(t[i0])
        w = slice(max(0, i0 - 10), max(1, i0))
        out["touch_v_down"] = float(-c.padv_z.to_numpy()[w].mean())
        out["touch_v_xy"] = float(np.hypot(c.padv_x.to_numpy()[w], c.padv_y.to_numpy()[w]).mean())
        out["touch_phase"] = int(ph[i0])
    else:
        out["t_touch"] = None
    shear_br = np.hypot(c.pd_fx, c.pd_fy).to_numpy()
    r_br = np.where(loaded, shear_br / np.maximum(brz, 1e-9), np.nan)
    r_br_s = med10(np.nan_to_num(r_br, nan=0.0))
    r_br_s[~loaded] = np.nan
    out["br_ratio_p95"] = float(np.nanpercentile(r_br_s, 95)) if loaded.any() else None
    out["br_ratio_max"] = float(np.nanmax(r_br_s)) if loaded.any() else None
    if out["t_touch"] is not None:
        win = (t >= out["t_touch"]) & (t <= out["t_touch"] + 0.3) & loaded
        out["br_ratio_impact"] = float(np.nanmax(r_br_s[win])) if win.any() else None
    else:
        out["br_ratio_impact"] = None
    for thr in (0.4, 0.6, 0.8):
        out[f"br_frac_over_{thr:g}"] = float(np.nanmean(r_br_s[loaded] > thr)) if loaded.any() else None
    brs = c.pd_slip.to_numpy()
    out["brace_slide_mm"] = float((brs * loaded).sum() * DT * 1000)
    out["brace_slip_peak"] = float(med10(brs * loaded).max())
    out["t_brace_slip"] = onset(t, brs, loaded)
    # ---- feet
    feet_slide, feet_ratio_max, feet_ratio_p95, t_fs = [], [], [], []
    for f in ("fl", "fr"):
        fz = c[f + "_fz"].to_numpy()
        ld = fz > F_FOOT
        sh = np.hypot(c[f + "_fx"], c[f + "_fy"]).to_numpy()
        r = med10(np.where(ld, sh / np.maximum(fz, 1e-9), 0.0))
        r[~ld] = np.nan
        after = t >= (enter[1] if len(enter) > 1 and enter[1] > 0 else 0.0)
        m = ld & after
        feet_ratio_max.append(float(np.nanmax(r[m])) if m.any() else 0.0)
        feet_ratio_p95.append(float(np.nanpercentile(r[m], 95)) if m.any() else 0.0)
        s = c[f + "_slip"].to_numpy()
        feet_slide.append(float((s * m).sum() * DT * 1000))
        t_fs.append(onset(t, s, m))
    out["foot_ratio_max"] = max(feet_ratio_max)
    out["foot_ratio_p95"] = max(feet_ratio_p95)
    out["foot_slide_mm"] = max(feet_slide)
    out["foot_slide_L_mm"], out["foot_slide_R_mm"] = feet_slide
    ts = [x for x in t_fs if x is not None]
    out["t_foot_slip"] = min(ts) if ts else None
    # sole displacement from the lean-rung entry to the end of the record
    i1 = np.searchsorted(t, enter[1]) if len(enter) > 1 and enter[1] > 0 else 0
    dL = np.hypot(c.soleL_x - c.soleL_x.iloc[i1], c.soleL_y - c.soleL_y.iloc[i1]).to_numpy()[i1:]
    dR = np.hypot(c.soleR_x - c.soleR_x.iloc[i1], c.soleR_y - c.soleR_y.iloc[i1]).to_numpy()[i1:]
    out["sole_disp_mm"] = float(max(dL.max(), dR.max()) * 1000) if len(dL) else 0.0
    # ---- event order before a fall
    if out["fell"]:
        seq = sorted([(k, v) for k, v in (("brace_slip", out["t_brace_slip"]),
                                         ("foot_slip", out["t_foot_slip"])) if v is not None],
                     key=lambda kv: kv[1])
        out["chain"] = ">".join(k for k, _ in seq) + (">" if seq else "") + "fall"
    else:
        out["chain"] = ""
    out["max_phase"] = int(ph.max())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(os.path.join(a.runs, "results.jsonl"))]
    rows = []
    for r in recs:
        if not os.path.exists(os.path.join(a.runs, r["tag"] + ".contact.csv")):
            continue
        rows.append(score(r["tag"], a.runs, r))
    with open(os.path.join(a.runs, "scored.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    df = pd.DataFrame(rows)
    if df.empty:
        print("no runs")
        return
    g = df.groupby("cell")
    tab = pd.DataFrame({
        "n": g.size(), "complete": g.complete.sum(), "fell": g.fell.sum(),
        "touch_v_down": g.touch_v_down.median(), "br_ratio_impact": g.br_ratio_impact.median(),
        "br_ratio_p95": g.br_ratio_p95.median(), "brace_slide_mm": g.brace_slide_mm.median(),
        "foot_ratio_max": g.foot_ratio_max.median(), "foot_slide_mm": g.foot_slide_mm.median(),
        "sole_disp_mm": g.sole_disp_mm.median()})
    pd.set_option("display.width", 200)
    print(tab.round(3).to_string())
    for _, r in df[df.fell == 1].iterrows():
        print(f"  fell {r.tag:22s} t_end={r.t_end:6.2f} touch={r.t_touch} brace_slip={r.t_brace_slip} "
              f"foot_slip={r.t_foot_slip} chain={r.chain} brace_slide={r.brace_slide_mm:.0f} mm "
              f"foot_slide={r.foot_slide_mm:.0f} mm")


if __name__ == "__main__":
    main()
