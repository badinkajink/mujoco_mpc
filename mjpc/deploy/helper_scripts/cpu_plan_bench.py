#!/usr/bin/env python3
"""cpu_plan_bench.py -- portable MJPC planning-rate benchmark (Lean / stand, strategy 6).

WHY  We compare planning rate across machines: ~34/s deployed on the Ryzen AI 9
     HX 370 laptop vs ~24/s on the i7-14700F lab desktop. "Better machine" = higher
     planning rate. This script measures the CPU-bound workload the CEM planner is
     dominated by -- N independent rollouts of the stand pose through MuJoCo, threaded
     -- so you can rank ANY machine (Linux box or Mac) with one command.

TWO NUMBERS
  proxy plan rate  the native-threaded rollout rate on THIS machine. Runs anywhere
                   (`pip install mujoco`). This is the cross-machine index. It over-
                   states the deployed node rate by a constant-ish factor (no CEM
                   sampling / cost eval / spline overhead), but the RATIO between
                   machines tracks the real one, because both are bound by the same
                   thing: mj_step throughput on the slowest core.
  real plan rate   parsed from a live h12_control_node's `plan=NN/s` stderr line
                   (--from-node-log). The literal deployed number (the 34/24).

PREREQ  this repo git-pulled (for the model XML + meshes) and `pip install mujoco`.
        Optional: `pip install psutil` for richer CPU% / freq / temperature sampling
        (falls back to /proc + sysctl when absent).

TYPICAL
  # ** THE ONE ** best thread x trajectory config for THIS machine (standalone,
  # no robot), auto-exits, writes JSON. Run this on every machine and compare:
  python3 cpu_plan_bench.py --grid --tag lab-i7-14700F

  # 90 s steady benchmark at the deploy operating point (20 traj, hw-6 threads):
  python3 cpu_plan_bench.py --tag macbook-m3

  # 1-D thread sweep only (fixed 20 trajectories):
  python3 cpu_plan_bench.py --sweep --tag ryzen-hx370

  # literal deployed rate from a normal node run you tee'd to a file:
  #   ./h12_control_node ... 2> node.err
  python3 cpu_plan_bench.py --from-node-log node.err

  # calibrate proxy -> real using a known real rate measured on THIS machine:
  python3 cpu_plan_bench.py --grid --calibrate 34   # Ryzen deployed ~34/s

Collect the JSON from every machine and hand them over -- ranking is then trivial.
The grid's plan-iters/s is a faithful RELATIVE index (same CPU-bound workload as
the real CEM planner); multiply by the --calibrate factor to predict real /s.
"""
import argparse, json, os, platform, re, signal, subprocess, sys, time
from datetime import datetime, timezone

# ---- optional deps -------------------------------------------------------- #
try:
    import numpy as np
except Exception:
    print("FATAL: numpy is required (`pip install numpy`).", file=sys.stderr); sys.exit(2)
try:
    import mujoco
except Exception:
    print("FATAL: mujoco is required (`pip install mujoco`).", file=sys.stderr); sys.exit(2)
try:
    import psutil
except Exception:
    psutil = None

DEPLOY_TRAJECTORIES = 20      # --plan_trajectories default
DEPLOY_HORIZON_STEPS = 100    # 1.0 s horizon / agent_timestep 0.010
THREAD_RESERVE = 6            # node default: plan_threads = hw - 6


# ---- model discovery ------------------------------------------------------ #
def find_xml(explicit=None):
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    here = os.path.dirname(os.path.abspath(__file__))
    # walk up to the repo root, then look in build tree (preferred) then source
    roots = [here]
    p = here
    for _ in range(8):
        p = os.path.dirname(p)
        roots.append(p)
    globs = [
        "build/mjpc/tasks/humanoid_bench/lean/Lean_H12_Magpie.xml",
        "mjpc/tasks/humanoid_bench/lean/Lean_H12_Magpie.xml",
    ]
    for r in roots:
        for g in globs:
            cand = os.path.join(r, g)
            if os.path.isfile(cand):
                return cand
    # last resort: recursive search under the nearest repo root
    for r in roots:
        if os.path.isdir(os.path.join(r, "mjpc")) or os.path.isdir(os.path.join(r, "build")):
            for dp, _, fs in os.walk(r):
                if "Lean_H12_Magpie.xml" in fs:
                    return os.path.join(dp, "Lean_H12_Magpie.xml")
    return None


# ---- machine profile ------------------------------------------------------ #
def cpu_brand():
    s = platform.system()
    try:
        if s == "Linux":
            for line in open("/proc/cpuinfo"):
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif s == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "unknown"


def max_freq_mhz():
    try:
        if psutil:
            f = psutil.cpu_freq()
            if f and f.max:
                return round(f.max)
    except Exception:
        pass
    try:
        if platform.system() == "Linux":
            v = open("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq").read().strip()
            return round(int(v) / 1000)
        if platform.system() == "Darwin":
            hz = int(subprocess.check_output(["sysctl", "-n", "hw.cpufrequency_max"], text=True))
            return round(hz / 1e6)
    except Exception:
        pass
    return None


def governor():
    try:
        return open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip()
    except Exception:
        return None


def temperature_c():
    """Hottest core/package temp, or None. psutil first, then Linux /sys hwmon."""
    try:
        if psutil and hasattr(psutil, "sensors_temperatures"):
            t = psutil.sensors_temperatures()
            vals = [s.current for arr in t.values() for s in arr if s.current]
            if vals:
                return round(max(vals), 1)
    except Exception:
        pass
    try:
        import glob
        vals = []
        for f in glob.glob("/sys/class/hwmon/hwmon*/temp*_input"):
            try:
                vals.append(int(open(f).read().strip()) / 1000.0)
            except Exception:
                pass
        if vals:
            return round(max(vals), 1)
    except Exception:
        pass
    return None


def profile():
    logical = os.cpu_count() or 1
    physical = None
    if psutil:
        try:
            physical = psutil.cpu_count(logical=False)
        except Exception:
            pass
    return {
        "host": platform.node(),
        "os": platform.platform(),
        "cpu": cpu_brand(),
        "cores_physical": physical,
        "cores_logical": logical,
        "max_freq_mhz": max_freq_mhz(),
        "governor": governor(),
        "mujoco": mujoco.__version__,
        "python": platform.python_version(),
        "psutil": bool(psutil),
    }


# ---- rollout harness (native mujoco.rollout, ThreadPool fallback) --------- #
class Harness:
    def __init__(self, xml, ntraj, steps, threads, model=None, keyframe="stand"):
        self.m = model if model is not None else mujoco.MjModel.from_xml_path(xml)
        self.ntraj, self.steps, self.threads = ntraj, steps, threads
        kid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if kid < 0:
            kid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.kid = max(kid, 0)
        self.mode = "native"
        try:
            from mujoco import rollout as _r
            self._r = _r
            d0 = mujoco.MjData(self.m)
            if self.m.nkey > 0:
                mujoco.mj_resetDataKeyframe(self.m, d0, self.kid)
            mujoco.mj_forward(self.m, d0)
            ns = mujoco.mj_stateSize(self.m, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            s0 = np.zeros(ns)
            mujoco.mj_getState(self.m, d0, s0, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            self._init = np.tile(s0, (ntraj, 1))
            self._ctrl = np.zeros((ntraj, steps, self.m.nu))
            self._scratch = [mujoco.MjData(self.m) for _ in range(threads)]
            self._roll = _r.Rollout(nthread=threads)
            self._roll.rollout(self.m, self._scratch, self._init, control=self._ctrl)  # warm/validate
        except Exception:
            self.mode = "threadpool"
            from concurrent.futures import ThreadPoolExecutor
            self._datas = [mujoco.MjData(self.m) for _ in range(ntraj)]
            self._ex = ThreadPoolExecutor(max_workers=threads)

    def plan_iteration(self):
        """One planner iteration == ntraj rollouts of `steps`, barriered."""
        if self.mode == "native":
            self._roll.rollout(self.m, self._scratch, self._init, control=self._ctrl)
        else:
            def one(d):
                if self.m.nkey > 0:
                    mujoco.mj_resetDataKeyframe(self.m, d, self.kid)
                for _ in range(self.steps):
                    mujoco.mj_step(self.m, d)
            list(self._ex.map(one, self._datas))

    def close(self):
        try:
            if self.mode == "native":
                self._roll.close()
            else:
                self._ex.shutdown(wait=False)
        except Exception:
            pass


# ---- timed windows -------------------------------------------------------- #
def run_window(h, seconds, sample=True):
    """Run plan iterations for `seconds`; return (iters/s, samples[])."""
    if psutil:
        psutil.cpu_percent(None)  # prime
    t0 = time.perf_counter()
    last = t0
    iters = 0
    samples = []
    while True:
        h.plan_iteration()
        iters += 1
        now = time.perf_counter()
        if sample and now - last >= 1.0:
            samples.append({
                "t": round(now - t0, 1),
                "rate": round(iters / (now - t0), 2),
                "cpu_pct": psutil.cpu_percent(None) if psutil else None,
                "temp_c": temperature_c(),
            })
            last = now
        if now - t0 >= seconds:
            break
    dt = time.perf_counter() - t0
    return iters / dt, samples


def droop_pct(samples):
    """% the rate fell from the first fifth to the last fifth (thermal soak signature)."""
    rates = [s["rate"] for s in samples if s.get("rate")]
    if len(rates) < 5:
        return None
    k = max(1, len(rates) // 5)
    a = sum(rates[:k]) / k
    b = sum(rates[-k:]) / k
    return round(100 * (a - b) / a, 1) if a else None


# ---- node-log parser ------------------------------------------------------ #
def parse_node_log(path):
    src = sys.stdin if path == "-" else open(path, "r", errors="ignore")
    rates = [float(m) for line in src for m in re.findall(r"plan=([\d.]+)/s", line)]
    if path != "-":
        src.close()
    return rates


def stats(v):
    if not v:
        return {}
    a = sorted(v)
    n = len(a)
    q = lambda p: a[min(n - 1, int(p * n))]
    return {"n": n, "median": round(a[n // 2], 2), "mean": round(sum(a) / n, 2),
            "p10": round(q(0.10), 2), "p90": round(q(0.90), 2),
            "min": round(a[0], 2), "max": round(a[-1], 2)}


def _settled_contacts(xml, keyframe):
    try:
        m = mujoco.MjModel.from_xml_path(xml)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        d = mujoco.MjData(m)
        if kid >= 0:
            mujoco.mj_resetDataKeyframe(m, d, kid)
        for _ in range(60):
            mujoco.mj_step(m, d)
        return d.ncon
    except Exception:
        return None


# ---- main ----------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Portable MJPC planning-rate benchmark (Lean stand, strategy 6).")
    ap.add_argument("--duration", type=float, default=90, help="steady-run seconds (default 90)")
    ap.add_argument("--trajectories", type=int, default=DEPLOY_TRAJECTORIES,
                    help="rollouts per plan iteration (deploy default 20)")
    ap.add_argument("--horizon-steps", type=int, default=DEPLOY_HORIZON_STEPS,
                    help="physics steps per rollout (deploy: 1.0s/0.010 = 100)")
    ap.add_argument("--threads", type=int, default=0,
                    help="worker threads; 0 = hw_logical - 6 (deploy default)")
    ap.add_argument("--sweep", action="store_true",
                    help="1-D thread scaling sweep instead of a single steady run")
    ap.add_argument("--sweep-window", type=float, default=6, help="seconds per sweep point")
    ap.add_argument("--grid", action="store_true",
                    help="2-D sweep over threads x trajectories -> best config for this machine")
    ap.add_argument("--grid-threads", default="",
                    help="comma list of thread counts (default: a spread up to logical cores)")
    ap.add_argument("--grid-traj", default="8,12,16,20,24,32",
                    help="comma list of trajectory counts (default 8,12,16,20,24,32)")
    ap.add_argument("--grid-window", type=float, default=4, help="seconds per grid cell (default 4)")
    ap.add_argument("--deploy-traj", type=int, default=DEPLOY_TRAJECTORIES,
                    help="trajectory count to optimise THREADS at in --grid (default 20; "
                         "this is a search-quality choice, not tuned for speed)")
    ap.add_argument("--from-node-log", metavar="PATH",
                    help="parse `plan=NN/s` from a node stderr log ('-' = stdin); skips the proxy")
    ap.add_argument("--calibrate", type=float, metavar="REAL_RATE",
                    help="known real deployed rate on THIS machine; prints proxy->real factor")
    ap.add_argument("--xml", help="override model path (auto-found otherwise)")
    ap.add_argument("--keyframe", default="stand",
                    help="pose to roll out from: 'stand' (default; ~= sustained brace cost), "
                         "'forearm_brace_lean' / 'forearm_brace_reach' (braced phase), or "
                         "'forearm_brace_mid' (busy transient, ~30%% slower = conservative). "
                         "The header prints the settled contact count so you see the load.")
    ap.add_argument("--tag", default="", help="free-text machine label for the report")
    ap.add_argument("--out", help="JSON output path (default cpu_bench_<host>_<ts>.json in cwd)")
    ap.add_argument("--no-json", action="store_true", help="console only, no JSON file")
    args = ap.parse_args()

    prof = profile()
    logical = prof["cores_logical"]
    threads = args.threads if args.threads > 0 else max(1, logical - THREAD_RESERVE)

    # ---- node-log mode: literal deployed rate, no proxy --------------------
    if args.from_node_log:
        rates = parse_node_log(args.from_node_log)
        st = stats(rates)
        print("\n=== real node plan rate  (from %s) ===" % args.from_node_log)
        if not st:
            print("  no `plan=NN/s` lines found.")
        else:
            print("  samples %(n)d   median %(median)s/s   mean %(mean)s/s"
                  "   p10 %(p10)s   p90 %(p90)s   min %(min)s   max %(max)s" % st)
        rec = {"kind": "node_log", "source": args.from_node_log, "profile": prof,
               "real_plan_rate": st,
               "utc": datetime.now(timezone.utc).isoformat()}
        _emit_json(args, prof, rec)
        return

    xml = find_xml(args.xml)
    if not xml:
        print("FATAL: could not locate Lean_H12_Magpie.xml (repo not pulled?). "
              "Pass --xml PATH.", file=sys.stderr)
        sys.exit(2)

    # thread/trajectory lists for grid mode
    def _ints(s):
        return [int(x) for x in s.replace(",", " ").split() if x.strip()]
    grid_traj = _ints(args.grid_traj) or [DEPLOY_TRAJECTORIES]
    if args.grid_threads.strip():
        grid_threads = [t for t in _ints(args.grid_threads) if 1 <= t <= logical]
    else:
        cand = [8, 9, 10, 11, 12, 13, 14, 15, 16, 24]
        grid_threads = sorted({t for t in cand if 1 <= t <= logical})
        # keep a couple of low points on small machines so the curve isn't a dot
        for t in (2, 4, 6):
            if t <= logical and t not in grid_threads and len(grid_threads) < 4:
                grid_threads.append(t)
        grid_threads = sorted(set(grid_threads))

    # hard watchdog so it always auto-kills itself
    if args.grid:
        budget = args.grid_window * len(grid_threads) * len(grid_traj) + 60
    elif args.sweep:
        budget = args.sweep_window * 10 + 30
    else:
        budget = args.duration + 30
    try:
        signal.signal(signal.SIGALRM, lambda *_: os._exit(3))
        signal.alarm(int(budget))
    except Exception:
        pass  # Windows / no SIGALRM

    print("=" * 68)
    print(" MJPC planning-rate benchmark  (Lean / stand, strategy 6)")
    print("=" * 68)
    lbl = (args.tag + "  ") if args.tag else ""
    print(f" {lbl}{prof['cpu']}")
    print(f" cores {prof['cores_physical'] or '?'}p / {logical}l   "
          f"maxfreq {prof['max_freq_mhz'] or '?'} MHz   gov {prof['governor'] or 'n/a'}")
    print(f" model {os.path.basename(xml)}   workload {args.trajectories} rollouts "
          f"x {args.horizon_steps} steps")
    _ncon = _settled_contacts(xml, args.keyframe)
    print(f" pose {args.keyframe}"
          + (f"   ({_ncon} contacts active)" if _ncon is not None else ""))
    t_start = temperature_c()
    if t_start is not None:
        print(f" start temp {t_start} C")
    print("-" * 68)

    rec = {"kind": "proxy", "tag": args.tag, "profile": prof, "xml": xml,
           "trajectories": args.trajectories, "horizon_steps": args.horizon_steps,
           "utc": datetime.now(timezone.utc).isoformat()}

    if args.grid:
        shared = mujoco.MjModel.from_xml_path(xml)  # load once, reuse per cell
        print(f" GRID  threads {grid_threads}  x  trajectories {grid_traj}"
              f"   ({args.grid_window:.0f}s/cell)")
        print(f" cells report plan-iterations/s (one iteration = N rollouts x "
              f"{args.horizon_steps} steps, barriered)\n")
        # header row: trajectory counts
        print(" thr\\traj " + "".join(f"{n:>7}" for n in grid_traj))
        matrix = []
        by_traj = {n: (-1, None) for n in grid_traj}   # n -> (best_rate, best_threads)
        for t in grid_threads:
            row = []
            for j, n in enumerate(grid_traj):
                h = Harness(xml, n, args.horizon_steps, t, model=shared, keyframe=args.keyframe)
                rate, _ = run_window(h, args.grid_window, sample=False)
                h.close()
                row.append(round(rate, 1))
                if rate > by_traj[n][0]:
                    by_traj[n] = (round(rate, 1), t)
            matrix.append({"threads": t, "rates": row})
            print(f" {t:>4}    " + "".join(f"{v:>7.1f}" for v in row))
        # the recommendation optimises THREADS at the deploy trajectory count only.
        # (trajectory count is a SEARCH-QUALITY knob -- fewer always reads faster
        # but plans worse; the script cannot see quality, so it never picks it.)
        op = args.deploy_traj if args.deploy_traj in grid_traj else \
            min(grid_traj, key=lambda x: abs(x - args.deploy_traj))
        op_rate, op_threads = by_traj[op]
        rec["grid"] = {"threads": grid_threads, "trajectories": grid_traj,
                       "window_s": args.grid_window, "matrix": matrix,
                       "best_threads_by_traj": {n: {"rate": r, "threads": th}
                                                for n, (r, th) in by_traj.items()}}
        rec["best"] = {"trajectories": op, "threads": op_threads, "rate": op_rate,
                       "note": "threads optimised at the deploy trajectory count; "
                               "trajectory count is fixed by search quality, not speed"}
        rec["headline_proxy_rate"] = op_rate
        t_end = temperature_c()
        print("-" * 68)
        print(f" BEST THREADS at {op} trajectories (your deploy quality point):  "
              f"{op_threads} threads  ->  {op_rate} plan/s")
        print(f"   trajectory count is a SEARCH-QUALITY knob, NOT a speed dial:")
        print(f"   fewer trajectories read faster here only because they search less "
              f"-- keep 20; the dive needs the samples. This script cannot see plan")
        print(f"   quality, so it optimises threads ONLY, at your chosen trajectory count.")
        if t_start is not None and t_end is not None:
            print(f" temp {t_start} -> {t_end} C")
            if t_end - t_start > 20:
                print(f"   note: package warmed {t_end - t_start:.0f} C across the grid; "
                      f"later cells may read low from heat-soak. Re-run cool to confirm the peak.")
    elif args.sweep:
        seq, seen = [], set()
        for t in [1, 2, 4, 8, 12, 16, prof["cores_physical"] or 0,
                  logical - THREAD_RESERVE, logical]:
            if t and 1 <= t <= logical and t not in seen:
                seen.add(t); seq.append(t)
        seq.sort()
        print(f" thread sweep {seq}  ({args.sweep_window:.0f}s each)")
        print(f" {'thr':>4} {'plan/s':>8} {'mj_step/s':>12} {'cpu%':>6} {'temp':>6}")
        rows = []
        for t in seq:
            h = Harness(xml, args.trajectories, args.horizon_steps, t, keyframe=args.keyframe)
            rate, samp = run_window(h, args.sweep_window, sample=False)
            h.close()
            cpu = psutil.cpu_percent(None) if psutil else None
            temp = temperature_c()
            mjs = rate * args.trajectories * args.horizon_steps
            rows.append({"threads": t, "plan_rate": round(rate, 2),
                         "mj_step_per_s": round(mjs), "cpu_pct": cpu, "temp_c": temp})
            print(f" {t:>4} {rate:>8.1f} {mjs:>12,.0f} "
                  f"{(cpu if cpu is not None else 0):>6.0f} {(temp if temp else 0):>6.1f}")
        rec["sweep"] = rows
        best = max(rows, key=lambda r: r["plan_rate"])
        print("-" * 68)
        print(f" peak proxy rate {best['plan_rate']}/s at {best['threads']} threads")
        rec["headline_proxy_rate"] = best["plan_rate"]
    else:
        h = Harness(xml, args.trajectories, args.horizon_steps, threads, keyframe=args.keyframe)
        print(f" threads {threads}   harness {h.mode}   running {args.duration:.0f}s ...")
        rate, samples = run_window(h, args.duration)
        h.close()
        t_end = temperature_c()
        mjs = rate * args.trajectories * args.horizon_steps
        dp = droop_pct(samples)
        rec.update({"threads": threads, "harness": h.mode,
                    "headline_proxy_rate": round(rate, 2),
                    "mj_step_per_s": round(mjs), "samples": samples,
                    "temp_start_c": t_start, "temp_end_c": t_end, "droop_pct": dp})
        print("-" * 68)
        print(f" PROXY PLAN RATE  {rate:5.1f} /s        (mj_step {mjs:,.0f}/s)")
        if t_start is not None and t_end is not None:
            print(f" temp {t_start} -> {t_end} C")
        if dp is not None and dp >= 15:
            print(f" ** THERMAL DROOP: rate fell {dp}% over the run -- CPU heat-soak "
                  f"(the effect that tanked the Ryzen below 25/s). Let it cool and re-run.")
        if args.calibrate:
            factor = args.calibrate / rate
            rec["calibration"] = {"real_rate": args.calibrate, "proxy_rate": round(rate, 2),
                                  "factor": round(factor, 3)}
            print(f" calibration: real {args.calibrate}/s = {factor:.3f} x proxy  "
                  f"(apply this factor to other machines' proxy rate to predict their real rate)")

    print("=" * 68)
    _emit_json(args, prof, rec)


def _emit_json(args, prof, rec):
    if args.no_json:
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    host = re.sub(r"[^A-Za-z0-9_-]", "", prof["host"] or "host")
    out = args.out or os.path.join(os.getcwd(), f"cpu_bench_{host}_{ts}.json")
    try:
        json.dump(rec, open(out, "w"), indent=2)
        print(f" JSON -> {out}")
    except Exception as e:
        print(f" (could not write JSON: {e})", file=sys.stderr)


if __name__ == "__main__":
    main()
