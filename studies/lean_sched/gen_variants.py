#!/usr/bin/env python3
"""Generate schedule variants of a lean strategy, holding phase 0 fixed.

Phase 0 (`stand_up`, sustain 15 s + ramp 5 s) is the bring-up settle the real
robot needs and the user has ruled OUT of scope, so every variant copies it
byte-for-byte. Everything downstream is scaled.

The variants are written next to the strategies as `<base>_<tag>.json` and
selected at runtime by LEAN_STRATEGY_OVERRIDE, so a sweep runs them concurrently
without touching the slot's own file.
"""
import copy, json, os, sys

STRAT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../mjpc/tasks/humanoid_bench/lean/strategies")
STRAT_DIR = os.path.normpath(STRAT_DIR)
HELD = 9999.0

# tag -> (sustain scale, ramp scale) applied to every phase EXCEPT phase 0.
VARIANTS = {
    "base":        (1.0,     1.0),
    "sus50":       (0.5,     1.0),
    "both50":      (0.5,     0.5),
    "both33":      (1 / 3.0, 1 / 3.0),
}


def floor_of(phases):
    """Scheduled floor = sum of the FINITE sustains, and nothing else.

    MEASURED 2026-08-26 (lean_bench smoke, strategy 24): phase 0 carries
    sustain 15 + ramp 5 and advances at t=15.00, not t=20. The target ramp runs
    CONCURRENTLY with the sustain and does not gate the advance -- so ramp
    seconds are not additive wall-clock. Cutting a ramp makes the reference
    move faster inside the same window; only cutting a sustain shortens the
    window. An earlier version of this function summed both and overcounted."""
    return sum(p.get("success_sustain_time", 0.0) or 0.0
               for p in phases
               if (p.get("success_sustain_time", 0.0) or 0.0) < HELD)


def ramp_of(phases):
    return sum(p.get("target_ramp_sec", 0.0) or 0.0 for p in phases)


def main(base_name):
    src = json.load(open(os.path.join(STRAT_DIR, base_name + ".json")))
    print("%-28s n=%d sustain_floor=%.1f s ramp_total=%.1f s"
          % (base_name, len(src), floor_of(src), ramp_of(src)))
    for tag, (ss, rs) in VARIANTS.items():
        out = []
        for i, p in enumerate(src):
            q = copy.deepcopy(p)
            if i > 0:                                   # phase 0 is untouchable
                s = q.get("success_sustain_time")
                if s is not None and s < HELD:
                    q["success_sustain_time"] = round(s * ss, 3)
                r = q.get("target_ramp_sec")
                if r:
                    q["target_ramp_sec"] = round(r * rs, 3)
            out.append(q)
        name = "%s_%s" % (base_name, tag)
        json.dump(out, open(os.path.join(STRAT_DIR, name + ".json"), "w"), indent=4)
        print("  %-34s sustain_floor=%6.1f s (saves %5.1f s)  ramp_total=%5.1f s"
              % (name, floor_of(out), floor_of(src) - floor_of(out), ramp_of(out)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "h12_recovery_noreach")
