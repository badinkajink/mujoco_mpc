#!/usr/bin/env python3
"""Payload-belief sweep for the braced retrieval (strategy 11 = strat 27 + timeout_advance).

One lean_bench run per (arm, true payload, believed payload, seed). At rung 5 (the
first rung after the grasp close) the plant gets `payload_true` kg on the right
gripper, ramped over 0.3 s; the planner's model gets `payload_belief` kg at once.
Arm P ("prior pipeline") also preloads the brace: brace_force_target_add =
K_LEVER * belief * g for rungs 5-8. Arm M ("model only") leaves the JSON targets.

Resumable: results.jsonl is appended and fsync'd per run; existing tags are
skipped. Run policy (memory: mjpc-sweep-cpu-budget): 2 jobs x 6 threads, nice 10,
systemd scope with CPUQuota and MemoryMax when available.
"""
import argparse, json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "../.."))
BIN = os.path.join(ROOT, "build_cmake/bin/lean_bench")
SUMMARY = re.compile(r"\[bench-summary\] (.*)")
G = 9.81
K_LEVER = 2.0   # brace preload per kg believed: d_hand / d_brace from the toe, ~0.48 m / 0.24 m
TRUE = [0.0, 1.0, 2.0, 4.0]
BELIEF = [0.0, 1.0, 2.0, 4.0]


def conditions(a):
    seeds = [int(s) for s in a.seeds.split(",")]
    out = []
    for arm in a.arms.split(","):
        for seed in seeds:
            for m in TRUE:
                for b in BELIEF:
                    if a.diag_only and m != b:
                        continue
                    add = K_LEVER * b * G if arm == "P" else 0.0
                    out.append(dict(tag=f"{arm}_m{m:g}_b{b:g}_s{seed}", arm=arm, m=m, b=b,
                                    bft_add=round(add, 2), seed=seed))
    return out


def run_one(c, a):
    csv_path = os.path.join(a.out, c["tag"] + ".csv")
    log_path = os.path.join(a.out, c["tag"] + ".log")
    cmd = []
    if a.cpu_quota > 0:
        cmd += ["systemd-run", "--user", "--scope", "--quiet",
                "-p", f"CPUQuota={a.cpu_quota}%", "-p", f"MemoryMax={a.mem_max}"]
    cmd += ["nice", "-n", str(a.nice), BIN,
            "--task", "Lean H12 Magpie", "--strategy", str(a.slot), "--seed", str(c["seed"]),
            "--total_time", str(a.total_time), "--threads", str(a.threads), "--spp", str(a.spp),
            "--gains", a.gains, "--numeric", "agent_allocate_active_only=1",
            "--payload_true", f"{c['m']:g}", "--payload_belief", f"{c['b']:g}",
            "--bft_add", f"{c['bft_add']:g}", "--out", csv_path]
    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write(" ".join(cmd) + "\n")
        p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
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
    ap.add_argument("--diag_only", action="store_true")
    ap.add_argument("--slot", type=int, default=11)
    ap.add_argument("--total_time", type=float, default=130)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--spp", type=int, default=15)
    ap.add_argument("--gains", default="deploy")
    ap.add_argument("--cpu_quota", type=int, default=700)
    ap.add_argument("--mem_max", default="5G")
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
