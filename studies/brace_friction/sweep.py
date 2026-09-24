#!/usr/bin/env python3
"""Friction sweep for the braced lean (strategy 25) at the deployed controller.

One lean_bench run per (cell, seed). A cell sets the sliding friction of the
robot<->table contacts (every <pair> naming the slab + the slab geoms, which
carry priority 1) and of the foot<->floor contacts on the PLANT, and optionally
on the PLANNER's model:

    table_mu:foot_mu               plant only, planner keeps 1.0 / 1.0
    table_mu:foot_mu:ptable:pfoot  planner model set too (belief)

The controller is the one the robot runs: CEM (agent_planner 5, shipped
std_min 0.01, 6 elites), 33 plans/s (--spp 15), deploy joint PD (--gains deploy).

Each run writes <tag>.csv (lean_bench metrics, 50 Hz), <tag>.contact.csv (per
2 ms step: brace / foot / other contact forces, slip speeds, pad and sole
positions, CoM), <tag>.qpos.csv (100 Hz, for video) and <tag>.log. The parsed
[bench-summary] line is appended to results.jsonl and fsync'd as it lands;
re-running skips tags already there.

Run policy (~/.claude/CLAUDE.md "Compute safety"): every job goes through
`resguard.sh run` (systemd scope, MemoryMax, CPUQuota, nice, watchdog; rc 3 =
launch gate closed -> wait and retry). Measured RSS 3.66 GB at 6 threads, so
--mem 5G and two jobs at >= 12 GB MemAvailable.

    ./sweep.py --out runs/grid --cells 1:1,0.8:0.4 --seeds 0-11 --jobs 2
"""
import argparse, json, os, re, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "../.."))
BIN = os.path.join(ROOT, "build_cmake/bin/lean_bench")
GUARD = os.path.expanduser("~/.claude/bin/resguard.sh")
SUMMARY = re.compile(r"\[bench-summary\] (.*)")


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def cell_tag(cell):
    t, f = cell[0], cell[1]
    tag = f"t{t:g}_f{f:g}"
    if len(cell) == 4:
        tag += f"_pt{cell[2]:g}_pf{cell[3]:g}"
    return tag


def run_one(cell, seed, a):
    tag = f"{cell_tag(cell)}_s{seed}"
    base = os.path.join(a.out, tag)
    cmd = [GUARD, "run", "--mem", a.mem_max, "--cpu", str(a.cpu_quota), "--nice", str(a.nice), "--",
           BIN, "--task", "Lean H12 Magpie", "--strategy", str(a.slot), "--seed", str(seed),
           "--total_time", str(a.total_time), "--threads", str(a.threads), "--spp", str(a.spp),
           "--gains", "deploy", "--numeric", "agent_allocate_active_only=1",
           "--numeric", "agent_planner=5",
           "--out", base + ".csv", "--contact_out", base + ".contact.csv",
           "--qpos_out", base + ".qpos.csv", "--video_hz", "100"]
    if cell[0] != 1.0:
        cmd += ["--plant_table_mu", f"{cell[0]:g}"]
    if cell[1] != 1.0:
        cmd += ["--plant_foot_mu", f"{cell[1]:g}"]
    if len(cell) == 4:
        cmd += ["--planner_table_mu", f"{cell[2]:g}", "--planner_foot_mu", f"{cell[3]:g}"]
    for kv in a.numeric:
        cmd += ["--numeric", kv]
    if a.stiff:
        cmd += ["--plant_table_stiff", "1"]
    if a.table_dz != 0.0:
        cmd += ["--plant_table_dz", f"{a.table_dz:g}"]
    if a.assist_fx != 0.0:
        cmd += ["--assist_fx", f"{a.assist_fx:g}", "--assist_from_phase", str(a.assist_from_phase),
                "--assist_until_phase", str(a.assist_until_phase), "--assist_gap", f"{a.assist_gap:g}"]
    t0 = time.time()
    with open(base + ".log", "w") as lf:
        lf.write(" ".join(cmd) + "\n")
        # rc 3 = the guard's launch gate is closed: wait for it without a retry
        # limit (a gate held by cold swap can stay shut for hours) and say so
        # once every 10 minutes rather than every attempt
        attempt = 0
        while True:
            p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if p.returncode != 3:
                break
            if attempt % 10 == 0:
                lf.write(f"[sweep] launch gate closed (attempt {attempt}); waiting\n{p.stderr}")
                lf.flush()
                print(f"  gate closed for {tag} (attempt {attempt}): "
                      f"{p.stderr.strip().splitlines()[-1][:120]}", flush=True)
            attempt += 1
            time.sleep(60)
        lf.write(p.stderr)
    rec = {"tag": tag, "cell": cell_tag(cell), "plant_table_mu": cell[0], "plant_foot_mu": cell[1],
           "planner_table_mu": cell[2] if len(cell) == 4 else 1.0,
           "planner_foot_mu": cell[3] if len(cell) == 4 else 1.0,
           "seed": seed, "numeric": a.numeric, "stiff": int(a.stiff), "table_dz": a.table_dz,
           "assist_fx": a.assist_fx, "assist_gap": a.assist_gap,
           "wall_s": round(time.time() - t0, 1),
           "rc": p.returncode, "summary": ""}
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
    ap.add_argument("--cells", required=True, help="t:f[:pt:pf],... (see module doc)")
    ap.add_argument("--seeds", default="0-11")
    ap.add_argument("--numeric", action="append", default=[], help="extra lean_bench --numeric k=v")
    ap.add_argument("--stiff", action="store_true", help="lean_bench --plant_table_stiff 1 (rigid slab on the plant)")
    ap.add_argument("--table_dz", type=float, default=0.0, help="lean_bench --plant_table_dz (m, plant slab only)")
    ap.add_argument("--assist_fx", type=float, default=0.0,
                    help="lean_bench --assist_fx (N on the torso, negative = pulled back from the table)")
    ap.add_argument("--assist_from_phase", type=int, default=1)
    ap.add_argument("--assist_until_phase", type=int, default=2)
    ap.add_argument("--assist_gap", type=float, default=0.15,
                    help="pull only while the pad is within this many m of the slab face")
    ap.add_argument("--slot", type=int, default=25)
    ap.add_argument("--total_time", type=float, default=75)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--spp", type=int, default=15)
    ap.add_argument("--cpu_quota", type=int, default=700)   # (threads + 1) x 100
    ap.add_argument("--mem_max", default="5G")              # measured RSS 3.66 GB x 1.3
    ap.add_argument("--nice", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cells = [tuple(float(x) for x in c.split(":")) for c in a.cells.split(",") if c]
    seeds = parse_seeds(a.seeds)
    res_path = os.path.join(a.out, "results.jsonl")
    done = set()
    if os.path.exists(res_path):
        for line in open(res_path):
            try:
                r = json.loads(line)
                if r.get("summary"):          # a run that produced a summary line
                    done.add(r["tag"])
            except Exception:
                pass
    # seed-major order: every cell gets seed 0 before any cell gets seed 1, so an
    # interrupted sweep leaves balanced cells
    todo = [(c, s) for s in seeds for c in cells if f"{cell_tag(c)}_s{s}" not in done]
    print(f"{len(todo)} runs ({len(cells)} cells x {len(seeds)} seeds, {len(done)} already done)",
          flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.jobs) as ex, open(res_path, "a") as f:
        futs = {ex.submit(run_one, c, s, a): (c, s) for c, s in todo}
        n = 0
        for fut in as_completed(futs):
            rec = fut.result()
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
            n += 1
            print(f"  [{n}/{len(todo)} {time.time() - t0:5.0f}s] {rec['tag']:22s} rc={rec['rc']} "
                  f"fell={rec.get('fell')} complete={rec.get('complete')} "
                  f"t_complete={rec.get('t_complete')} t_end={rec.get('t_end')} wall={rec['wall_s']}",
                  flush=True)


if __name__ == "__main__":
    main()
