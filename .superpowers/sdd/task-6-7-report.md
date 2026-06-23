# Task 6 & 7 Implementation Report

## Task 6 — Register index 22 + raise slider, create strat-22 pre-lean JSON

### Files Created
- `mjpc/tasks/humanoid_bench/lean/strategies/h12_simple_forearm_brace.json` — single `forearm_brace_lean` phase, `target_ramp_sec: 15.0`, `brace_force_target: 15.0`, strat-6 leg-core + brace overrides
- `mjpc/tasks/humanoid_bench/lean/strategies/h12_hands_simple_forearm_brace.json` — identical copy for Hands variant

### Files Modified
- `mjpc/tasks/humanoid_bench/lean/lean.h` — base list: `"h12_simple_reach",` (was `};`) + new `"h12_simple_forearm_brace"};  // 22`; Hands list: same pattern with `h12_hands_` prefix
- `mjpc/tasks/humanoid_bench/lean/Lean_H12.xml` (line 247) — `data="5 0 21"` → `data="5 0 22"`
- `mjpc/tasks/humanoid_bench/lean/Lean_H12_Magpie.xml` (line 247) — same
- `mjpc/tasks/humanoid_bench/lean/Lean_H12_Hands.xml` (line 137) — same

### Static Checks
- `ninja mjpc` exit 0; last lines:
  ```
  [3/6] Building CXX object mjpc/CMakeFiles/libmjpc.dir/tasks/humanoid_bench/lean/lean.cc.o
  [5/6] Linking CXX static library lib/libmjpc.a
  [6/6] Linking CXX executable bin/mjpc
  ```
- JSON parse: `JSON OK`
- Build tree: `build/mjpc/tasks/humanoid_bench/lean/strategies/h12_simple_forearm_brace.json` exists

### Commit
`4b73ba6` — lean: add strat 22 pre-lean forearm-brace + register, slider->22

---

## Task 7 — Re-anchor strat 16 counterbalance leg-core to strat 6

### Script
`/tmp/core_check.py` created with exact CORE dict from spec.

### Pre-fix MISMATCH (h12_simple_counterbalance.json last phase `counterbalance_standing`)
```
MISMATCH {'Foot Stability': (60.0, 15.0), 'Angular Momentum': (14.0, 5.0)}
exit: 1
```

### Files Modified
- `mjpc/tasks/humanoid_bench/lean/strategies/h12_simple_counterbalance.json`
  - `Angular Momentum`: 14.0 → 5.0
  - `Foot Stability`: 60.0 → 15.0
  - (all other weights unchanged, lean knobs preserved)
- `mjpc/tasks/humanoid_bench/lean/strategies/h12_hands_simple_counterbalance.json`
  - Previous: minimal 7-key weight block; all CORE keys missing → all 0 (MISMATCH)
  - Fixed: added all 17 CORE keys with strat-6 values; lean knobs (Pelvis Tilt 50, Torso Forward Tilt 15, Object Dist 100, Reaching Hand Dist 80) preserved unchanged

### Post-fix core_check results
```
=== h12_simple_counterbalance ===
PASS {}
exit: 0
=== h12_hands_simple_counterbalance ===
PASS {}
exit: 0
=== h12_simple_forearm_brace ===
PASS {}
exit: 0
```

### Commit
`ba1b29f` — lean: re-anchor strat 16 counterbalance leg-core to strat 6
