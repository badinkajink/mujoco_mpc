# Lean Pipeline 31-35 — Progress Ledger

Plan: docs/plans/2026-06-22-lean-pipeline-31-35.md
Branch: lean-pipeline-31-35
Policy: twin/agent_server validation (reach_validate.py/_twin_probe.py) DEFERRED to user. Subagents do edits + ninja builds + static headless checks only.

Task 1: complete (branch + ledger setup)
Task 2: complete (commit df469e6, gripper collision proxy + twin mirror, static check PASS, review clean)
Tasks 3-5: complete (commits 8e3fa15,d33d2e1,3b5a7f6 + fix 96d16e9; lean.cc auto-arm + mocap counterbalance + mirrored brace; build exit 0; review spec OK, shadow+comment fixed)
Tasks 6-7: complete (commits 4b73ba6,ba1b29f; strat22 forearm_brace JSON+register idx22+slider22; strat16 leg-core re-anchored; build exit0; core_check PASS x3)
Task 7 CORRECTED (controller, deterministic script): prior subagent over-reached (added stand_up phase + changed Pelvis Tilt 50->75, Torso Fwd Tilt 15->10, Object Dist 100->45, Reaching Hand 80->55). Redone: single phase, knobs/arm preserved, leg-core+standing-height (BaseH450/Height100/Knees40) anchored. NOTE: generator (T8) must enforce bucket A AND bucket B core.
Task 8: complete (commit 048e55d; _gen_lean_pipeline.py generates 10 ladder JSONs; core_check PASS x10; phase counts 1/2/3/4/4; brace ramp 15; strat35 finite; source-of-truth chain verified strat16 tuned knobs -> strat33). Implementer caught plan bug: P_brace base sustain 9999->4.0 so full(35) stays finite.
Tasks 9-10: complete (commits 99d55dd,796dc6d; register 31-35 + reserved 23-30 + slider->35; CMake gen_lean_pipeline hook; build exit0; model nq41/nu27; deterministic regen).
Task 11 (static only, twin DEFERRED to user): all 0-35 strategy JSONs present in build tree, model loads, no runtime File-not-exist. Behavioral faithful-twin validation = USER.
Task 12: memory updated (BUILT status). Final regression build exit0. Whole-branch review package written.
