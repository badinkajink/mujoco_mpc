# Tasks 3–5 Implementation Report

**Branch:** `lean-pipeline-31-35`
**File edited:** `mjpc/tasks/humanoid_bench/lean/lean.cc`
**Build result:** `ninja mjpc` exit 0 (clean)

---

## Task 3: Shared auto-arm selection

**Commit:** `8e3fa15` — "lean: shared auto-arm selection (reaching + opposite bracing hand)"

### Before (lines 425–432 in original):
```cpp
  // Right arm always braces on the table; left arm always reaches for the
  // object. body1=28 in the contact keyframes targets the right hand body, so
  // the reaching/bracing assignment must stay fixed — dynamic switching based
  // on object position causes both arms to be pulled toward the table
  // simultaneously, creating an irresolvable contradiction.
  constexpr bool left_reaches = true;
  double const *reaching_hand = left_hand_pos;
  double const *bracing_hand  = right_hand_pos;
```

And separately at ~line 446:
```cpp
  double brace_contact_force = left_reaches ? right_contact[0] : left_contact[0];
  double reach_contact_force = left_reaches ? left_contact[0] : right_contact[0];
```

### After:
```cpp
  // ----- Determine which hand reaches and which braces ----- //
  double const *left_hand_pos = SensorByName(model, data, "left_hand_pos");
  double const *right_hand_pos = SensorByName(model, data, "right_hand_pos");
  // torso_pos needed here for AUTO arm-side selection; also used below for
  // brace target computation. Declared once, reused in both places.
  double *torso_pos = SensorByName(model, data, "torso_position");

  // Auto-arm selection shared by counterbalance + forearm-brace phases (the
  // reach_to_target branch does its own identical pick). reach_hand numeric:
  // 0 = AUTO (mocap target y < torso y -> right hand reaches), 1 = force LEFT,
  // 2 = force RIGHT. The OTHER arm always braces/counterweights.
  int rh_id_sel = mj_name2id(model, mjOBJ_NUMERIC, "reach_hand");
  int rh_sel = (rh_id_sel >= 0)
      ? (int)std::lround(model->numeric_data[model->numeric_adr[rh_id_sel]])
      : 0;
  bool reach_right = (rh_sel == 2) ? true
                   : (rh_sel == 1) ? false
                   : (data->mocap_pos[1] < torso_pos[1]);
  double const *reaching_hand = reach_right ? right_hand_pos : left_hand_pos;
  double const *bracing_hand  = reach_right ? left_hand_pos  : right_hand_pos;
```

And at contact forces:
```cpp
  double brace_contact_force = reach_right ? left_contact[0] : right_contact[0];
  double reach_contact_force = reach_right ? right_contact[0] : left_contact[0];
```

**Scope note:** `torso_pos` was originally declared ~30 lines later (after `table_pos`). Since
`reach_right` depends on `torso_pos[1]`, the declaration was hoisted to before the arm-selection
block, and the original later declaration replaced with a comment. This is the only deviation
from the plan's verbatim replacement — necessary to avoid an "undeclared identifier" compile error.

---

## Task 4: Counterbalance branch uses live mocap target

**Commit:** `d33d2e1` — "lean: counterbalance reaches the live mocap object (no clamp -> leans)"

### Before:
```cpp
  else if (residual_keyframe_.name == "counterbalance_standing") {
    double const *fl = SensorByName(model, data, "foot_left_pos");
    double const *fr = SensorByName(model, data, "foot_right_pos");
    phase1_target_storage[0] = 0.5 * (fl[0] + fr[0]) + 0.70;
    phase1_target_storage[1] = 0.5 * (fl[1] + fr[1]) + 0.15;
    phase1_target_storage[2] = 0.75;
    reach_target = phase1_target_storage;
  }
```

### After:
```cpp
  else if (residual_keyframe_.name == "counterbalance_standing") {
    // Counterbalance (Strategy 16 pre-lean + pipeline stage 33): the reaching
    // arm pulls toward the LIVE mocap object (world-fixed, so the reach error
    // shrinks as the body bows in -> self-limiting, no runaway). NO sphere clamp
    // (unlike reach_to_target): an out-of-reach object is exactly what makes the
    // torso lean forward, with the free arm + hips swinging back to counterweight.
    // Lean depth is bounded by Pelvis Tilt / Torso Forward Tilt (JSON lean knobs).
    mju_copy3(phase1_target_storage, data->mocap_pos);
    reach_target = phase1_target_storage;
  }
```

---

## Task 5: Forearm brace lateral offset mirrored to bracing arm

**Commit:** `3b5a7f6` — "lean: mirror forearm brace point to the auto-picked bracing arm"

### Before (inside `ideal_brace[3]` initializer):
```cpp
  double ideal_brace[3] = {
      torso_pos[0] + 0.4 * torso_to_table_x,  // Partway between torso and far edge
      torso_pos[1] - 0.24,                     // under/just-right-of R shoulder joint
      ...
      table_pos[2] - 0.06
  };
```

### After:
```cpp
  double ideal_brace[3] = {
      torso_pos[0] + 0.4 * torso_to_table_x,  // Partway between torso and far edge
      // bracing arm = the OTHER arm (reach_right -> left arm braces, so +0.24).
      torso_pos[1] + (reach_right ? 0.24 : -0.24),
      ...
      table_pos[2] - 0.06
  };
```

**`reach_right` scope at Task 5 site:** REUSED. Both the arm-selection block (Task 3) and
`ideal_brace` (Task 5) are inside the same function `lean::ResidualFn::Residual()` (lines 124–2045).
`reach_right` declared at ~line 433 is fully visible at `ideal_brace` at ~line 640. No recomputation needed.

---

## Final build output (last 5 lines + exit code)

```
Hunk #7 succeeded at 261 with fuzz 2.
patching file .../h1_2_modified_magpie.xml
[3/5] Building CXX object mjpc/CMakeFiles/libmjpc.dir/tasks/humanoid_bench/lean/lean.cc.o
[4/5] Linking CXX static library lib/libmjpc.a
[5/5] Linking CXX executable bin/mjpc
EXIT:0
```

---

## Commit hashes

| Task | Hash | Message |
|------|------|---------|
| 3 | `8e3fa15` | lean: shared auto-arm selection (reaching + opposite bracing hand) |
| 4 | `d33d2e1` | lean: counterbalance reaches the live mocap object (no clamp -> leans) |
| 5 | `3b5a7f6` | lean: mirror forearm brace point to the auto-picked bracing arm |

---

## Review Fix — commit `96d16e9`

**Finding 1 — variable shadow (`reach_right`)**

- Before (~line 588): `bool reach_right = (rh_mode == 2) ? true : ...` (shadowed the outer scope `reach_right` at line 436)
- After: renamed to `bool reach_right_reach = ...`; two uses in the same block updated: `reaching_hand = reach_right_reach ? ...` and `torso_pos[1] + (reach_right_reach ? -0.148 : 0.148)`
- No logic change; outer `reach_right` (line 436) and all uses outside `reach_to_target` block untouched.

**Finding 2 — stale comment (`left_reaches`)**

- Before (~line 527): `// no other strategy's reach assignment is touched (left_reaches stays true everywhere else).`
- After: `// no other strategy's reach assignment is touched (auto arm-selection via the outer reach_right covers the other branches).`

**Build:** `ninja mjpc` exit 0 — `[5/5] Linking CXX executable bin/mjpc`
**Commit:** `96d16e9` — `lean: fix reach_right shadow + stale left_reaches comment (review)`
