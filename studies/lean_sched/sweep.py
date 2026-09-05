#!/usr/bin/env python3
"""Run the lean schedule sweep: variants x seeds, N at a time, and collect the
per-run summary into one CSV.

Each run is one `lean_bench` process with its own LEAN_STRATEGY_OVERRIDE, so the
variants do not contend for a strategy file. Concurrency x threads is sized to
the box (20 cores); the host has frozen twice under memory pressure, so this
never runs more than `--jobs` at a time and never oversubscribes the cores.
"""
import argparse, csv, os, re, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "../.."))
BIN = os.path.join(ROOT, "build_cmake/bin/lean_bench")
SUMMARY = re.compile(r"\[bench-summary\] (.*)")


def run_one(args):
    variant, seed, outdir, task, slot, total_time, threads, video, spp = args
    tag = "%s_seed%d" % (variant, seed)
    csv_path = os.path.join(outdir, tag + ".csv")
    qpos_path = os.path.join(outdir, tag + ".qpos.csv") if video else ""
    env = dict(os.environ)
    env["LEAN_STRATEGY_OVERRIDE"] = variant
    # nice: MJPC saturates every planner thread it is given and this box is the
    # user's workstation. A sweep that starves the compositor has already hard-
    # crashed it once (2026-08-26) and took every partial result with it.
    cmd = ["nice", "-n", "15",
           BIN, "--task", task, "--strategy", str(slot), "--seed", str(seed),
           "--total_time", str(total_time), "--threads", str(threads),
           "--spp", str(spp), "--out", csv_path]
    if qpos_path:
        cmd += ["--qpos_out", qpos_path]
    t0 = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall = time.time() - t0
    m = None
    for line in p.stderr.splitlines():
        g = SUMMARY.search(line)
        if g:
            m = g.group(1)
    rec = {"variant": variant, "seed": seed, "wall_s": round(wall, 1),
           "rc": p.returncode, "summary": m or "", "csv": csv_path}
    if m:
        for kv in m.split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                rec[k] = v
    print("  done %-34s rc=%d wall=%5.0fs %s"
          % (tag, p.returncode, wall, m or "(no summary)"), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="h12_recovery_noreach")
    ap.add_argument("--variants", default="base,sus50,both50,both33")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--task", default="Lean H12 Magpie")
    ap.add_argument("--slot", type=int, default=24)
    ap.add_argument("--total_time", type=float, default=220)
    ap.add_argument("--jobs", type=int, default=1,
                    help="concurrent runs; jobs*threads must stay well under nproc")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--force", action="store_true",
                    help="override the core-budget guard (do not)")
    # Plan rate = 1 / (spp * agent_timestep). Lean_H12_Magpie sets agent_timestep
    # 0.010, so spp=3 -> 33 Hz, spp=2 -> 50 Hz. The deploy node runs 34-45 Hz on
    # real; the old default spp=4 planned at 25 Hz, which made every variant look
    # more fragile than the robot actually is.
    ap.add_argument("--spp", type=int, default=3)
    ap.add_argument("--video_seed", type=int, default=0,
                    help="dump qpos for this seed only (video source)")
    a = ap.parse_args()

    # ---- CORE BUDGET GUARD. `jobs * threads` reaching nproc starves the desktop
    # and hard-crashed this box on 2026-08-26. Leave at least 6 cores free.
    ncpu = os.cpu_count() or 4
    want = a.jobs * a.threads
    if want > max(1, ncpu - 6) and not a.force:
        sys.exit("refusing: jobs*threads=%d on a %d-core box leaves %d cores free "
                 "(need >=6). Lower --jobs/--threads, or pass --force." %
                 (want, ncpu, ncpu - want))
    print("core budget: %d worker threads on %d cores (%d free)"
          % (want, ncpu, ncpu - want), flush=True)

    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for v in a.variants.split(","):
        name = "%s_%s" % (a.base, v)
        for s in range(a.seeds):
            jobs.append((name, s, a.out, a.task, a.slot, a.total_time,
                         a.threads, s == a.video_seed, a.spp))
    print("%d runs, %d at a time, %d threads each, spp=%d (plan %.0f Hz)"
          % (len(jobs), a.jobs, a.threads, a.spp, 1.0 / (a.spp * 0.010)),
          flush=True)
    keys = ["variant", "seed", "wall_s", "rc", "fell", "complete", "t_complete",
            "t_end", "phases", "enter", "csv", "summary"]
    # Append+flush+fsync EVERY row as it lands. The 2026-08-26 crash lost a whole
    # sweep because results were only written at the end.
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
        recs = list(ex.map(run_and_record, jobs))
    sf.close()
    print("sweep done in %.1f min -> %s/summary.csv"
          % ((time.time() - t0) / 60.0, a.out), flush=True)


if __name__ == "__main__":
    main()
