#!/usr/bin/env python3
"""Payload-belief sweep for the braced retrieval (strategy 11 = strat 27 + timeout_advance).

One lean_bench run per (arm, true payload, believed payload, seed). At rung 5 (the
first rung after the grasp close) the plant gets `payload_true` kg on the right
gripper, ramped over 0.3 s; the planner's model gets `payload_belief` kg at once.
Arm P ("prior pipeline") also preloads the brace: brace_force_target_add =
K_LEVER * belief * g for rungs 5-8. Arm M ("model only") leaves the JSON targets.
Arm D is M with the planner's release plan dumped (--plan_out, phases >= 8).
Arms Smean / Smax / Smin are the scenario baseline: belief 0 on the nominal model,
iCEM-DR scoring every candidate over the mass set --scenario_masses (default
0,2,4 kg) with mean / max / min aggregation (sweep2, 2026-09-15).

Resumable: results.jsonl is appended and fsync'd per run; existing tags are
skipped. Run policy (memory: mjpc-sweep-cpu-budget): 2 jobs x 6 threads, nice 10,
systemd scope with CPUQuota and MemoryMax when available.
"""
import argparse, json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "../.."))
BIN = os.path.join(ROOT, "build_cmake/bin/lean_bench")
GUARD = os.path.expanduser("~/.claude/bin/resguard.sh")
SUMMARY = re.compile(r"\[bench-summary\] (.*)")
G = 9.81
K_LEVER = 2.0   # brace preload per kg believed: d_hand / d_brace from the toe, ~0.48 m / 0.24 m
TRUE = [0.0, 1.0, 2.0, 4.0]
BELIEF = [0.0, 1.0, 2.0, 4.0]


def conditions(a):
    seeds = [int(s) for s in a.seeds.split(",")]
    trues = [float(x) for x in a.trues.split(",")]
    beliefs = [float(x) for x in a.beliefs.split(",")]
    out = []
    for arm in a.arms.split(","):
        for seed in seeds:
            for m in trues:
                if arm.startswith("S"):
                    out.append(dict(tag=f"{arm}_m{m:g}_s{seed}", arm=arm, m=m, b=0.0, bft_add=0.0,
                                    seed=seed, scenario_agg=arm[1:], scenario_masses=a.scenario_masses))
                    continue
                for b in beliefs:
                    if a.diag_only and m != b:
                        continue
                    add = K_LEVER * b * G if arm == "P" else 0.0
                    out.append(dict(tag=f"{arm}_m{m:g}_b{b:g}_s{seed}", arm=arm, m=m, b=b,
                                    bft_add=round(add, 2), seed=seed))
    return out


def run_one(c, a):
    csv_path = os.path.join(a.out, c["tag"] + ".csv")
    log_path = os.path.join(a.out, c["tag"] + ".log")
    # Every job runs through the resource guard (memory: mjpc-sweep-cpu-budget): a
    # systemd scope with MemoryMax + CPUQuota + nice, a watchdog that kills the job
    # if MemAvailable collapses, and a launch gate (rc 3 = closed -> wait and retry).
    cmd = [GUARD, "run", "--mem", a.mem_max, "--cpu", str(a.cpu_quota), "--nice", str(a.nice), "--",
           BIN,
            "--task", "Lean H12 Magpie", "--strategy", str(a.slot), "--seed", str(c["seed"]),
            "--total_time", str(a.total_time), "--threads", str(a.threads), "--spp", str(a.spp),
            "--gains", a.gains, "--numeric", "agent_allocate_active_only=1",
            "--payload_true", f"{c['m']:g}", "--payload_belief", f"{c['b']:g}",
            "--bft_add", f"{c['bft_add']:g}", "--out", csv_path]
    if c.get("scenario_agg"):
        cmd += ["--scenario_masses", c["scenario_masses"], "--scenario_agg", c["scenario_agg"]]
    if c["arm"] == "D":
        cmd += ["--plan_out", os.path.join(a.out, c["tag"] + ".plan.csv"),
                "--plan_out_from_phase", "8", "--plan_out_stride", "5"]
    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write(" ".join(cmd) + "\n")
        for attempt in range(60):
            p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if p.returncode != 3:
                break
            lf.write(f"[sweep] launch gate closed (attempt {attempt}); waiting 60 s\n{p.stderr}")
            print(f"  gate closed for {c['tag']}: {p.stderr.strip().splitlines()[-1][:120]}", flush=True)
            time.sleep(60)
        lf.write(p.stderr)
    rec = dict(c, wall_s=round(time.time() - t0, 1), rc=p.returncode, csv=csv_path, summary="")
    for line in p.stderr.splitlines():
        g = SUMMARY.search(line)
        if g:
            rec["summary"] = g.group(1)
    for kv in rec["summary"].split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            if k not in ("task", "planner", "wall_s", "seed"):
                rec[k] = v
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="P,M")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--trues", default=",".join(f"{x:g}" for x in TRUE))
    ap.add_argument("--beliefs", default=",".join(f"{x:g}" for x in BELIEF))
    ap.add_argument("--scenario_masses", default="0,2,4")
    ap.add_argument("--diag_only", action="store_true")
    ap.add_argument("--slot", type=int, default=11)
    ap.add_argument("--total_time", type=float, default=130)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--spp", type=int, default=15)
    ap.add_argument("--gains", default="deploy")
    ap.add_argument("--cpu_quota", type=int, default=700)   # (threads + 1) x 100
    ap.add_argument("--mem_max", default="5G")              # measured RSS 3.7 GB x 1.3
    ap.add_argument("--nice", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res_path = os.path.join(a.out, "results.jsonl")
    done = set()
    if os.path.exists(res_path):
        for line in open(res_path):
            try:
                done.add(json.loads(line)["tag"])
            except Exception:
                pass
    conds = [c for c in conditions(a) if c["tag"] not in done]
    print(f"{len(conds)} runs to do ({len(done)} done)", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.jobs) as ex, open(res_path, "a") as f:
        futs = {ex.submit(run_one, c, a): c for c in conds}
        n = 0
        for fut in as_completed(futs):
            rec = fut.result()
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
            n += 1
            print(f"  [{n}/{len(conds)} {time.time() - t0:5.0f}s] {rec['tag']:16s} fell={rec.get('fell')} "
                  f"complete={rec.get('complete')} max_phase={rec.get('max_phase')} "
                  f"peak_brace={rec.get('peak_brace_carry')} min_pelvis={rec.get('min_pelvis_carry')} "
                  f"tilt={rec.get('max_tilt_carry')} wall={rec['wall_s']}", flush=True)


if __name__ == "__main__":
    main()
