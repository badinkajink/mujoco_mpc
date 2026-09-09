# Final work state — 2026-09-09

User requested wrap-up. All 33 randomized ablation trials and all 18 pitch
screen trials completed: 51 episodes, 5.13 hours of simulation-run wall time.
No further simulations are authorized by the current wrap-up request.

Worktree: /home/humanoid/Programs/Humanoid_Simulation/mujoco_mpc
Branch: wxie/table-height (study merged by 463e782f)
Study commits: bd682089 protocol/exact-engine audit; a1be9670 ablation/screen;
a6b5bce0 final report.
Final report: docs/lean/20260909-table_height_robustness.html

The screen rejected brace_pitch_track=1. Strict low/nominal/high counts:
reference 3/3,3/3,2/3; pitch tracking 1/3,3/3,3/3. Full-task reference
2/3,2/3,1/3; candidate1/3,3/3,1/3. No falls in either fresh screen arm.
Thus prevention of backward falls, the original hypothesis, was not shown
in this cohort; the tall-table difference was one precise-hold success.
Low failures: minimum error37.4mm (seed21), precise loaded dwell1.4s(seed22).
Both also lacked continuous upward brace loading early in the reach hold.

The earlier ablation reference was3/3,3/3,1/3; full2/3,3/3,0/3. Keep these
cohorts separate. No height-dependent mixture of favorable rows was tested.

Runner session57124 and watcher64181 have exited. The watcher applied the
frozen rule, wrote screen1_decision.json, and did NOT launch confirmation.
STOP is set and gitignored. Strategy matches baseline_strategy.json.
No controller/Crocoddyl/hardware/default changes were applied by this screen.
No second scalar or any of the reserved15 confirmation trials ran.

All51 evaluation objects are exported in results.json; raw states, models,
commands and traces stay in studies/table_height/runs/robustness_20260909.
final_integrity.json verifies their binary/state/model/strategy identities.
Native replay uses the EXACT benchmark MuJoCo revision088079ef. Python is
used for aggregation and geometry rendering only; its nominal3.2.3 package
has an incompatible model ABI. The passive recorder changes no controls.
Six protocol tests passed. Report links and baseline restoration checked.

For future work, read protocol.md and append-only decisions.md before any
new tests. One optional scalar hypothesis remains unused, but must be
predeclared if the user resumes. Do not reuse confirmation seeds100–102
for tuning; .885/1.035 remain untested in this session. Do not rerun the
watcher blindly: its decision already exists and it deliberately refuses
refreezing. Nothing is currently queued to start itself.
