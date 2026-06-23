# Task 8 Report: Generator _gen_lean_pipeline.py

## Commit
Hash: 048e55d
Message: "lean: add 31-35 ladder generator + generated JSONs"

## Generator Code (as committed)

Path: `mjpc/tasks/humanoid_bench/lean/strategies/_gen_lean_pipeline.py`

Key deviation from spec: spec listed `P_brace = phase(brace, 3, "forearm_brace_lean", 9999.0, 15.0)` but verification step 3c requires `h12_lean_full.json`'s last phase sustain to be FINITE (not 9999). Fixed to `4.0` so that `held(P_brace)` in the brace truncation file still produces 9999 and `P_brace` directly in the full file is finite (4.0). All other code transcribed verbatim.

## core_check Results (all 10 files)

```
PASS {}  <- h12_lean_brace.json
PASS {}  <- h12_lean_counterbalance.json
PASS {}  <- h12_lean_full.json
PASS {}  <- h12_lean_reach.json
PASS {}  <- h12_lean_stand.json
PASS {}  <- h12_hands_lean_brace.json
PASS {}  <- h12_hands_lean_counterbalance.json
PASS {}  <- h12_hands_lean_full.json
PASS {}  <- h12_hands_lean_reach.json
PASS {}  <- h12_hands_lean_stand.json
```

## Phase Count Table

| File | Phases |
|------|--------|
| h12_lean_stand | 1 |
| h12_lean_reach | 2 |
| h12_lean_counterbalance | 3 |
| h12_lean_brace | 4 |
| h12_lean_full | 4 |

## Brace-Phase Spot-Check Output

```
h12_lean_brace.json last phase:
  name=forearm_brace_lean  ramp=15.0  sustain=9999.0
h12_lean_full.json last phase:
  name=forearm_brace_lean  ramp=15.0  sustain=4.0
ALL SPOT-CHECKS PASS
```

h12_lean_brace.json: last phase name=="forearm_brace_lean" PASS, target_ramp_sec==15.0 PASS, success_sustain_time==9999.0 PASS.
h12_lean_full.json: last phase success_sustain_time==4.0 (FINITE, not 9999) PASS.
