# Task 9 & 10 Implementation Report

## Task 9: Register 31–35 + reserved 23–30, slider → 35

### lean.h changes

**Base list (lean class)** — after index 22 (`h12_simple_forearm_brace`), appended:
```cpp
"h12_simple_stand", "h12_simple_stand", "h12_simple_stand", "h12_simple_stand", // 23-26 reserved
"h12_simple_stand", "h12_simple_stand", "h12_simple_stand", "h12_simple_stand", // 27-30 reserved
"h12_lean_stand",            // 31
"h12_lean_reach",            // 32
"h12_lean_counterbalance",   // 33
"h12_lean_brace",            // 34
"h12_lean_full"};            // 35
```

**Hands override (Lean_H12_Hands)** — identical extension with `h12_hands_simple_stand` placeholders (23-30) and `h12_hands_lean_*` (31-35).

Both lists now contain 36 entries (indices 0–35).

### XML slider changes

All three task XMLs changed `residual_Strategy data="5 0 22"` → `data="5 0 35"`:
- `mjpc/tasks/humanoid_bench/lean/Lean_H12.xml:247`
- `mjpc/tasks/humanoid_bench/lean/Lean_H12_Magpie.xml:247`
- `mjpc/tasks/humanoid_bench/lean/Lean_H12_Hands.xml:137`

### Static verification outputs

**Build:** `ninja mjpc 2>&1 | tail -5`
```
patching file ... h1_2_modified_magpie.xml ...
[3/6] Building CXX object ... lean.cc.o
[4/6] Building CXX object ... tasks.cc.o
[5/6] Linking CXX static library lib/libmjpc.a
[6/6] Linking CXX executable bin/mjpc
```
Exit 0.

**JSON source check (5/5 OK):**
```
h12_lean_stand OK
h12_lean_reach OK
h12_lean_counterbalance OK
h12_lean_brace OK
h12_lean_full OK
```

**Build tree count:** `ls build/mjpc/tasks/humanoid_bench/lean/strategies/h12_lean_*.json | wc -l` → `5`

**Model load:**
```
model loaded, nq= 41 nu= 27
```
(run from `build/mjpc/tasks/humanoid_bench/lean/` with relative mesh paths)

### Commit

```
99d55dd lean: register strats 31-35 + reserved 23-30, slider->35
```

---

## Task 10: Hook generator into the build

### CMakeLists.txt addition

In `mjpc/tasks/CMakeLists.txt`, before `copy_resources` (line ~300), added:

```cmake
add_custom_target(gen_lean_pipeline ALL
        COMMAND ${Python_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/humanoid_bench/lean/strategies/_gen_lean_pipeline.py
        COMMENT "Generating lean pipeline strategies 31-35 from pre-lean twins")

add_dependencies(copy_resources gen_lean_pipeline)
```

`${Python_EXECUTABLE}` was already in scope (used at line 255 for manipulation merge script) — no additional `find_package` needed.

### Verification output

```
touch mjpc/tasks/humanoid_bench/lean/strategies/h12_simple_counterbalance.json
cd build && cmake .. && ninja gen_lean_pipeline 2>&1 | tail -3

[0/2] Re-checking globbed directories...
[1/1] Generating lean pipeline strategies 31-35 from pre-lean twins
generated 31-35 ladder JSONs
```

### Regenerated file diff check

`git status --short mjpc/tasks/humanoid_bench/lean/strategies/h12_lean_*.json` → no output (generator is deterministic; files unchanged after re-run).

### Commit

```
796dc6d build: regenerate lean pipeline 31-35 on every build
```

---

## Summary

| Check | Result |
|-------|--------|
| Build exit | 0 |
| Source JSONs (31-35) | 5/5 present |
| Build tree JSONs | 5 |
| Model load nq/nu | 41/27 |
| CMake hook fires | yes — "generated 31-35 ladder JSONs" |
| Regenerated files diff | clean (deterministic) |

No concerns.
