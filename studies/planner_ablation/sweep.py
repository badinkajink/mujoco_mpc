#!/usr/bin/env python3
"""Run planner-ablation arms x seeds through lean_bench, N at a time, resumable.

One `lean_bench` process per (arm, seed). The task, strategy JSON, model and
weights are the same in every arm; arms differ only in `agent_planner` and the
model <numeric> overrides listed in arms.py, all applied with --numeric before
Agent::Initialize. Every run writes the per-step CSV, a 50 Hz state track (for
the executed-control jitter analysis) and a log; the parsed [bench-summary] line
is appended to results.jsonl and fsync'd as it lands, so a crash loses at most
the run in flight. Re-running skips (arm, seed) pairs already in results.jsonl.

Run policy (memory: mjpc-sweep-cpu-budget): each process under a systemd scope
with a CPUQuota and a MemoryMax, nice 10, threads held at 6 across every arm.
With agent_allocate_active_only=1 a run peaks at ~3.7 GB RSS, so two jobs fit
the box the user leaves overnight; keep --jobs 1 while they are at the desk.
"""
import argparse, json, os, re, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arms import ARMS, BATCHES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "../.."))
BIN = os.path.join(ROOT, "build_cmake/bin/lean_bench")
MEM_MAX = os.environ.get("SWEEP_MEM_MAX", "5G")
SUMMARY = re.compile(r"\[bench-summary\] (.*)")


def run_one(arm, seed, a):
    planner, nums = ARMS[arm]
    tag = "%s_s%d" % (arm, seed)
    csv_path = os.path.join(a.out, tag + ".csv")
    state_path = os.path.join(a.out, tag + ".state.csv")
    log_path = os.path.join(a.out, tag + ".log")
    cmd = []
    if a.cpu_quota > 0:
        cmd += ["systemd-run", "--user", "--scope", "--quiet",
                "-p", "CPUQuota=%d%%" % a.cpu_quota, "-p", "MemoryMax=%s" % MEM_MAX]
    cmd += ["nice", "-n", str(a.nice), BIN,
            "--task", a.task, "--strategy", str(a.slot), "--seed", str(seed),
            "--total_time", str(a.total_time), "--threads", str(a.threads),
            "--spp", str(a.spp), "--out", csv_path, "--state_out", state_path,
            "--numeric", "agent_allocate_active_only=1",
            "--numeric", "agent_planner=%d" % planner]
    for k, v in nums.items():
        cmd += ["--numeric", "%s=%g" % (k, v)]
    if a.qpos:
        cmd += ["--qpos_out", os.path.join(a.out, tag + ".qpos.csv")]
    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write(" ".join(cmd) + "\n")
        p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           text=True)
        lf.write(p.stderr)
    wall = time.time() - t0
    m = None
    for line in p.stderr.splitlines():
        g = SUMMARY.search(line)
        if g:
            m = g.group(1)
    rec = {"arm": arm, "seed": seed, "planner": planner, "numerics": nums,
           "wall_s": round(wall, 1), "rc": p.returncode, "csv": csv_path,
           "state": state_path, "summary": m or ""}
    if m:
        for kv in m.split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                if k not in ("task", "planner"):
                    rec[k] = v
    print("  done %-26s rc=%3d wall=%5.0fs fell=%s complete=%s t_complete=%s enter=%s"
          % (tag, p.returncode, wall, rec.get("fell"), rec.get("complete"),
             rec.get("t_complete"), rec.get("enter")), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="", help="comma list of arm names")
    ap.add_argument("--batch", default="", help="comma list of batch letters from arms.py")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--task", default="Lean H12 Magpie")
    ap.add_argument("--slot", type=int, default=25)
    ap.add_argument("--total_time", type=float, default=75)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--cpu_quota", type=int, default=700)
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--spp", type=int, default=3)
    ap.add_argument("--qpos", action="store_true", help="also dump qpos for video")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    arms = [x for x in a.arms.split(",") if x]
    for b in [x for x in a.batch.split(",") if x]:
        arms += BATCHES[b]
    seeds = [int(s) for s in a.seeds.split(",")]
    unknown = [x for x in arms if x not in ARMS]
    if unknown:
        sys.exit("unknown arms: %s" % unknown)

    ncpu = os.cpu_count() or 4
    want = a.jobs * (a.threads + 1)
    if want > 16 and not a.force:
        sys.exit("refusing: jobs*(threads+1)=%d on %d cores" % (want, ncpu))
    load1 = os.getloadavg()[0]
    if load1 > ncpu / 2 and not a.force:
        sys.exit("refusing: 1-min load %.1f on %d cores" % (load1, ncpu))

    os.makedirs(a.out, exist_ok=True)
    res_path = os.path.join(a.out, "results.jsonl")
    done = set()
    if os.path.exists(res_path):
        for line in open(res_path):
            try:
                r = json.loads(line)
                if r.get("summary"):
                    done.add((r["arm"], r["seed"]))
            except json.JSONDecodeError:
                pass
    jobs = [(arm, s) for arm in arms for s in seeds if (arm, s) not in done]
    print("%d runs (%d arms x %d seeds, %d already done), %d at a time, "
          "%d threads each, spp=%d, MemoryMax=%s, CPUQuota=%d%%"
          % (len(jobs), len(arms), len(seeds), len(done), a.jobs, a.threads,
             a.spp, MEM_MAX, a.cpu_quota), flush=True)

    rf = open(res_path, "a")
    lock = threading.Lock()

    def run_and_record(job):
        r = run_one(job[0], job[1], a)
        with lock:
            rf.write(json.dumps(r) + "\n"); rf.flush(); os.fsync(rf.fileno())
        return r

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        list(ex.map(run_and_record, jobs))
    rf.close()
    print("sweep done in %.1f min -> %s" % ((time.time() - t0) / 60.0, res_path),
          flush=True)


if __name__ == "__main__":
    main()
