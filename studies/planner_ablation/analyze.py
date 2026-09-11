#!/usr/bin/env python3
"""Score planner-ablation runs: outcome counts per arm plus the plant-side
channels that separate 'worked' from 'did not' (executed-control jitter, phase
cost, brace load, reach error), all read from the per-run CSVs.

Outcome ladder per run (mutually exclusive, in this order):
  fell       pelvis below 0.5 m (bench stop)
  complete   entered the terminal rung and held it 3 s
  stalled    neither within total_time; max phase reached is recorded

Reach error: distance from the right jaw tip (lean.cc kGripperTipLocal, logged
as jaw_*) to strategy 25's rung-2 point (near edge + rtt[0], centre - rtt[1],
face + rtt[2]) -- the 70 mm gate the ladder advances on. Minimum over the run.

Executed-control jitter: from the 50 Hz state track, mean over rows of the RMS
across actuators of u_t - u_{t-1}. It is what the plant receives, planner
family aside, so it is the channel a noise-magnitude story has to show up in.
"""
import argparse, csv, json, math, os, statistics as st, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arms import ARMS  # noqa: E402

RTT = (0.55, 0.04, 0.15)
NEAR_EDGE_X, CENTRE_Y = 0.45, 0.0
PHASE_NAMES = ["stand_up", "brace_lean", "reach", "release", "standback_r1",
               "standback_r2", "standback_r3", "standback_r4", "stand_final"]


def fnum(x):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return float("nan")


def score_run(rec):
    out = {"arm": rec["arm"], "seed": int(rec["seed"]), "planner": rec["planner"],
           "rc": rec["rc"], "wall_s": rec["wall_s"], "csv": rec["csv"],
           "state": rec.get("state", "")}
    fell = rec.get("fell") == "1"
    complete = rec.get("complete") == "1"
    out["outcome"] = "fell" if fell else ("complete" if complete else "stalled")
    out["t_complete"] = fnum(rec.get("t_complete", "nan"))
    out["t_end"] = fnum(rec.get("t_end", "nan"))
    enter = [fnum(x) for x in rec.get("enter", "").split(":") if x]
    out["enter"] = enter
    out["max_phase"] = max([i for i, e in enumerate(enter) if e >= 0] or [-1])
    # rung durations from the entry times (nan if the rung was not left)
    names = ["stand_up", "brace_lean", "reach", "release", "standback_r1",
             "standback_r2", "standback_r3", "standback_r4", "stand_final"]
    out["rung_dur"] = {}
    for i in range(len(enter) - 1):
        if enter[i] >= 0 and enter[i + 1] >= 0:
            out["rung_dur"][names[i]] = enter[i + 1] - enter[i]
    out["phase_at_end"] = out["max_phase"]

    rows = list(csv.DictReader(open(rec["csv"]))) if os.path.exists(rec["csv"]) else []
    if not rows:
        return out
    face = fnum(rows[0]["face_z"])
    tgt = (NEAR_EDGE_X + RTT[0], CENTRE_Y - RTT[1], face + RTT[2])
    per_phase_cost = defaultdict(list)
    per_phase_brace = defaultdict(list)
    reach_min = float("inf")
    brace_peak = 0.0
    trunk_peak = 0.0
    seated = 0
    lean_rows = 0
    ess = []
    pelvis_min, tilt_max = float("inf"), 0.0
    for r in rows:
        if len(r) != len(rows[0]) or r.get("cost") in (None, ""):
            continue
        pelvis_min = min(pelvis_min, fnum(r["pelvis_z"]))
        tilt_max = max(tilt_max, fnum(r["torso_tilt_deg"]))
        ph = int(r["phase"]) if r["phase"] not in ("", "-1") else -1
        c = fnum(r["cost"])
        if ph >= 0 and not math.isnan(c):
            per_phase_cost[ph].append(c)
        bn = fnum(r["brace_normal_N"])
        tn = fnum(r["trunk_normal_N"])
        brace_peak = max(brace_peak, bn)
        trunk_peak = max(trunk_peak, tn)
        if ph in (1, 2, 3):
            per_phase_brace[ph].append(bn)
            lean_rows += 1
            if bn > 5.0:
                seated += 1
        if ph in (1, 2):
            d = math.dist((fnum(r["jaw_x"]), fnum(r["jaw_y"]), fnum(r["jaw_z"])), tgt)
            reach_min = min(reach_min, d)
        e = fnum(r["mppi_ess"])
        if not math.isnan(e):
            ess.append(e)
    out["pelvis_min_m"] = pelvis_min
    out["tilt_max_deg"] = tilt_max
    # The bench's fall stop is pelvis < 0.5 m. A robot draped over the slab with
    # the torso past horizontal never trips it, so score a collapse separately:
    # pelvis below 0.75 m or torso tilt past 60 deg at any logged step.
    out["collapsed"] = bool(pelvis_min < 0.75 or tilt_max > 60.0)
    if out["outcome"] == "stalled" and out["collapsed"]:
        out["outcome"] = "collapsed"
    out["cost_phase_mean"] = {PHASE_NAMES[p]: st.mean(v) for p, v in per_phase_cost.items() if v}
    out["cost_stand_mean"] = out["cost_phase_mean"].get("stand_up", float("nan"))
    out["brace_peak_N"] = brace_peak
    out["trunk_peak_N"] = trunk_peak
    out["brace_median_reach_N"] = st.median(per_phase_brace[2]) if per_phase_brace[2] else float("nan")
    out["seated_frac"] = seated / lean_rows if lean_rows else float("nan")
    out["reach_min_mm"] = 1000.0 * reach_min if reach_min < float("inf") else float("nan")
    out["mppi_ess_median"] = st.median(ess) if ess else float("nan")

    # executed-control jitter from the state track
    sp = rec.get("state", "")
    if sp and os.path.exists(sp):
        prev = None
        jit_all, jit_phase = [], defaultdict(list)
        t_prev, dts = None, []
        with open(sp) as f:
            rd = csv.reader(f)
            hdr = next(rd)
            ucols = [i for i, h in enumerate(hdr) if h.startswith("u")]
            pcol = hdr.index("phase")
            for row in rd:
                if len(row) != len(hdr):
                    continue  # partial last line of a run still in flight
                t_row = float(row[0])
                if t_prev is not None and len(dts) < 50:
                    dts.append(t_row - t_prev)
                t_prev = t_row
                u = [float(row[i]) for i in ucols]
                if prev is not None:
                    d = math.sqrt(sum((a - b) ** 2 for a, b in zip(u, prev)) / len(u))
                    jit_all.append(d)
                    ph = int(row[pcol])
                    if 0 <= ph < len(PHASE_NAMES):
                        jit_phase[PHASE_NAMES[ph]].append(d)
                prev = u
        out["jitter_rms_rad"] = st.mean(jit_all) if jit_all else float("nan")
        # the row interval the step was measured over: 0.020 s for the 50 Hz
        # logs, one plan (spp x 0.002 s) for --log_per_plan runs
        out["row_dt"] = st.median(dts) if dts else float("nan")
        out["jitter_phase"] = {k: st.mean(v) for k, v in jit_phase.items() if v}
        out["jitter_stand_rad"] = out["jitter_phase"].get("stand_up", float("nan"))
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def nanmedian(v):
    v = [x for x in v if not (isinstance(x, float) and math.isnan(x))]
    return st.median(v) if v else float("nan")


def aggregate(runs):
    by = defaultdict(list)
    for r in runs:
        by[r["arm"]].append(r)
    agg = {}
    for arm, rs in by.items():
        n = len(rs)
        k = sum(r["outcome"] == "complete" for r in rs)
        f = sum(r["outcome"] == "fell" for r in rs)
        c = sum(r["outcome"] == "collapsed" for r in rs)
        agg[arm] = {
            "planner": ARMS[arm][0], "numerics": ARMS[arm][1], "n": n,
            "complete": k, "fell": f, "collapsed": c, "stalled": n - k - f - c,
            "upright": n - f - c,
            "complete_wilson": wilson(k, n),
            "t_complete_median": nanmedian([r["t_complete"] for r in rs if r["outcome"] == "complete"]),
            "t_fall_median": nanmedian([r["t_end"] for r in rs if r["outcome"] == "fell"]),
            "pelvis_min_median": nanmedian([r.get("pelvis_min_m", float("nan")) for r in rs]),
            "tilt_max_median": nanmedian([r.get("tilt_max_deg", float("nan")) for r in rs]),
            "max_phase": sorted(r["max_phase"] for r in rs),
            "max_phase_median": st.median([r["max_phase"] for r in rs]),
            "cost_stand_median": nanmedian([r.get("cost_stand_mean", float("nan")) for r in rs]),
            "jitter_stand_median": nanmedian([r.get("jitter_stand_rad", float("nan")) for r in rs]),
            "jitter_all_median": nanmedian([r.get("jitter_rms_rad", float("nan")) for r in rs]),
            "brace_peak_median": nanmedian([r.get("brace_peak_N", float("nan")) for r in rs]),
            "reach_min_mm_median": nanmedian([r.get("reach_min_mm", float("nan")) for r in rs]),
            "seated_frac_median": nanmedian([r.get("seated_frac", float("nan")) for r in rs]),
            "mppi_ess_median": nanmedian([r.get("mppi_ess_median", float("nan")) for r in rs]),
            "seeds": [r["seed"] for r in rs],
        }
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs with results.jsonl")
    ap.add_argument("--out", default="")
    ap.add_argument("--order", default="", help="comma list of arms for the table order")
    a = ap.parse_args()
    # later --runs dirs override earlier ones for the same (arm, seed), so a
    # redo directory replaces the runs it redid
    by_key = {}
    for d in a.runs:
        p = os.path.join(d, "results.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            if r.get("summary"):
                by_key[(r["arm"], int(r["seed"]))] = r
    recs = list(by_key.values())
    runs = [score_run(r) for r in recs]
    agg = aggregate(runs)
    order = [x for x in a.order.split(",") if x] or list(agg.keys())
    print("%-24s %2s %4s %4s %4s %9s  %6s  %8s  %8s  %7s  %7s %6s %6s"
          % ("arm", "n", "cmpl", "fell", "clps", "phase", "t_cmpl", "jit_stnd", "cost_stnd", "brace_N", "reach_mm", "tiltmax", "ess"))
    for arm in order:
        if arm not in agg:
            continue
        g = agg[arm]
        print("%-24s %2d %4d %4d %4d %9s  %6.1f  %8.4f  %8.2f  %7.0f  %7.0f %6.0f %6.1f"
              % (arm, g["n"], g["complete"], g["fell"], g["collapsed"],
                 "/".join(str(x) for x in g["max_phase"]),
                 g["t_complete_median"], g["jitter_stand_median"], g["cost_stand_median"],
                 g["brace_peak_median"], g["reach_min_mm_median"], g["tilt_max_median"],
                 g["mppi_ess_median"]))
    if a.out:
        json.dump({"runs": runs, "agg": agg}, open(a.out, "w"), indent=1, default=str)
        print("->", a.out)


if __name__ == "__main__":
    main()
