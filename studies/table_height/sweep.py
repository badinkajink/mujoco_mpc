#!/usr/bin/env python3
"""Sweep the lean pipeline over table height: heights x seeds, N runs at a time.

Each run is one `lean_bench` process with `--table_h <face z, m>`; the task is
otherwise untouched (same strategy JSON, same weights, same model). Nothing is
recompiled between points -- the height is task parameter index 7, applied on the
first Transition.

Concurrency: this box has 20 cores and has hard-frozen twice under memory
pressure, so `jobs * threads` is capped and every summary row is fsync'd as it
lands rather than at the end.
"""
import argparse, csv, os, re, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "../.."))
BIN = os.path.join(ROOT, "build_cmake/bin/lean_bench")
SUMMARY = re.compile(r"\[bench-summary\] (.*)")


def tag_for(h, seed):
    return "h%04d_s%d" % (round(h * 1000), seed)


def run_one(job):
    h, seed, outdir, task, slot, total_time, threads, video, spp, quota = job
    tag = tag_for(h, seed)
    csv_path = os.path.join(outdir, tag + ".csv")
    qpos_path = os.path.join(outdir, tag + ".qpos.csv") if video else ""
    # nice: MJPC saturates every planner thread it is given and this box is the
    # user's workstation. A sweep that starves the compositor hard-crashed it on
    # 2026-08-26 and took every partial result with it.
    cmd = []
    if quota > 0:
        # A hard CPU cap is the only thing that reliably keeps the compositor
        # responsive; `nice` does not when the contention is thread count.
        cmd += ["systemd-run", "--user", "--scope", "--quiet",
                "-p", "CPUQuota=%d%%" % quota, "-p", "MemoryMax=6G"]
    cmd += ["nice", "-n", "15",
           BIN, "--task", task, "--strategy", str(slot), "--seed", str(seed),
           "--table_h", "%.4f" % h,
           "--total_time", str(total_time), "--threads", str(threads),
           "--spp", str(spp), "--out", csv_path]
    if qpos_path:
        cmd += ["--qpos_out", qpos_path]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    m = None
    for line in p.stderr.splitlines():
        g = SUMMARY.search(line)
        if g:
            m = g.group(1)
    rec = {"table_h": h, "seed": seed, "wall_s": round(wall, 1),
           "rc": p.returncode, "summary": m or "", "csv": csv_path}
    if m:
        for kv in m.split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                if k != "task":
                    rec[k] = v
    print("  done %-14s rc=%d wall=%5.0fs %s"
          % (tag, p.returncode, wall, m or "(no summary)"), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--heights", default="0.735,0.785,0.835,0.885,0.935,0.985,1.035,1.085")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--task", default="Lean H12 Magpie")
    ap.add_argument("--slot", type=int, default=25)
    ap.add_argument("--total_time", type=float, default=110)
    # ★ 2026-09-04 SERIAL BY DEFAULT. jobs=3 x threads=4 stuttered the user's
    # desktop (load 27 on 20 cores) even under `nice -n 15` -- an MJPC process
    # costs threads+1, and the box is a workstation with an IDE and a browser on
    # it. One run at a time, 4 planner threads, under a CPUQuota scope.
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cpu_quota", type=int, default=450,
                    help="systemd CPUQuota %% for each run; 0 disables the scope. "
                         "nice alone does not protect the compositor when the "
                         "problem is thread count.")
    ap.add_argument("--spp", type=int, default=3,
                    help="steps per plan; agent_timestep is 0.010 so spp=3 -> 33 Hz, "
                         "the rate the deploy node actually runs at")
    ap.add_argument("--video_seed", type=int, default=0,
                    help="dump qpos for this seed only (video source)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    ncpu = os.cpu_count() or 4
    want = a.jobs * (a.threads + 1)      # +1: the sim/physics thread on top of the pool
    if want > 8 and not a.force:
        sys.exit("refusing: jobs*(threads+1)=%d on a %d-core workstation. The "
                 "ceiling is 8 while the user is at the desk (2026-09-04: 15 "
                 "threads under nice stuttered it). Lower --jobs/--threads, or "
                 "pass --force." % (want, ncpu))
    load1 = os.getloadavg()[0]
    if load1 > ncpu / 2 and not a.force:
        sys.exit("refusing: 1-min load average is %.1f on %d cores before we even "
                 "start. Wait for the box to quiet down, or pass --force."
                 % (load1, ncpu))
    print("core budget: %d threads on %d cores (load1 %.1f), CPUQuota=%d%%"
          % (want, ncpu, load1, a.cpu_quota), flush=True)

    os.makedirs(a.out, exist_ok=True)
    heights = [float(x) for x in a.heights.split(",")]
    jobs = [(h, s, a.out, a.task, a.slot, a.total_time, a.threads,
             s == a.video_seed, a.spp, a.cpu_quota)
            for h in heights for s in range(a.seeds)]
    print("%d runs (%d heights x %d seeds), %d at a time, %d threads each, "
          "spp=%d (plan %.0f Hz)"
          % (len(jobs), len(heights), a.seeds, a.jobs, a.threads, a.spp,
             1.0 / (a.spp * 0.010)), flush=True)

    keys = ["table_h", "seed", "wall_s", "rc", "fell", "complete", "t_complete",
            "t_end", "face_z", "phases", "enter", "csv", "summary"]
    sf = open(os.path.join(a.out, "summary.csv"), "w", newline="")
    w = csv.DictWriter(sf, fieldnames=keys, extrasaction="ignore")
    w.writeheader(); sf.flush(); os.fsync(sf.fileno())
    lock = threading.Lock()

    def run_and_record(job):
        r = run_one(job)
        with lock:
            w.writerow(r); sf.flush(); os.fsync(sf.fileno())
        return r

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        list(ex.map(run_and_record, jobs))
    sf.close()
    print("sweep done in %.1f min -> %s/summary.csv"
          % ((time.time() - t0) / 60.0, a.out), flush=True)


if __name__ == "__main__":
    main()
