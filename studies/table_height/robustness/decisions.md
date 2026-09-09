# Decision log — append only

## Before dynamic ablation

Protocol and 33-run randomized manifest committed at bd682089. The .08 s
smoke run is excluded. It discovered the MuJoCo ABI mismatch; native replay
was fixed before any new experimental episode. Six endpoint/manifest tests
passed. Compiled Python nominal-model checksum matches the prior Python
model; this does NOT establish native/Python engine equivalence.

The final-recovery scorer was corrected before the FIRST experimental
result: it uses elapsed time via `longest`, not a minimum sample count.
No threshold changed and no experiment had been scored at that point.

## After the first three experimental episodes (hypothesis generation only)

No ablation is stopped or selected based on these single trials.
- Low reference seed 10 reached and recovered, contrary to a categorical
  reading of the old 0/3 recovery result. Preserve both cohorts separately.
- Low two-second-gate seed 10 reached within 30 mm for only 1.10 s, but
  completed the ladder. Its 70 mm observed dwell is 1.98 s: the phase gate
  can leave one fewer complete 50 Hz interval than its continuous timer.
  Preserve the conservative frozen success flag and show a one-sample
  sensitivity count. Do not portray this as proof of physical <2 s holding.
- High no-retarget seed 10 held 30 mm for 4.32 s and completed the ladder,
  but final arm contact peaked near16 N while torso tilt was3–4 degrees.
  The strict unloaded-recovery flag is false; this is NOT an approach fall
  or an inability to stand. Distinguish terminal contact from a phase stall.

The timer comparison is an opportunity-to-settle effect, not an improvement
in optimizer quality. The target-depth contrast is a task-difficulty effect.
The remaining 30 ablations and fresh confirmation must determine whether
these patterns repeat.

After four completed episodes, added automatic evaluation-integrity guards: native FK/live difference must be <5 micrometres and p95 brace-load difference <0.01 N. These limits exceed the known text-rounding resolution; failure halts evaluation, never changes success thresholds. All four existing checks passed by large margins (FK <1 micrometre, force <0.001 N). Controller and runtime study configuration remain unchanged.

## 2026-09-09T07:47:37 — complete ablation; adaptive screen 1 frozen

All 33 ablation trials are scored. Reference strict counts low/nominal/high
are 3/3, 3/3, 1/3; full counts 2/3, 3/3, 0/3. Far-target strict counts are
3/3, 2/3, 2/3, so closer is not shown universally superior. No-pose gives
2/3 at both extremes. Short-hold gives 0/3 strict at both extremes. These
small conditional comparisons do not identify all interactions.

Failure localization: both reference tall-table failures occur in phase 1,
at 15.086 and 16.316 s, BEFORE reach. Seed 12 pelvis pitch moves from +0.5°
at 11.98 s to -10.0° at 13 s and -15.7° at 14 s; base x retreats from
0.134 to 0.032 m. Both arm contacts disappear. This is backward approach
instability, not a failed reach solve. The existing Pelvis Tilt residual has
a ±60° zero-cost band in brace phase; backward_tilt_gain multiplies that
zero. Thus increasing that gain alone would not supply a restoring gradient.

Hypothesis 1: activate existing `brace_pitch_track=1` (default 0), gain remains
1. This tracks the height-retargeted keyframe pitch in both lean rungs, giving
backward tilt a nonzero penalty while preserving height-specific forward bow.
This is ONE existing numeric; no new residual/code, no gate relaxation. Earlier
unsynchronized-planner tests do not answer this synchronized-model comparison.
It is a testable hypothesis, not proof that the static pitch is dynamically optimal.

Screen 1: reference vs pitch_track, at .785/.985/1.085, fresh reset seeds
20/21/22; 18 episodes randomized within seed blocks using Random(20260929+seed).
Everything else remains reference: sync1, pose1, depth.40, dwell5, spp3,
perturb.003, six threads, 75 s, unchanged binary and frozen endpoints.
Contemporaneous reference controls avoid relying solely on a previous batch.
Select provisionally only if high-table strict count improves over these
controls, with no lower strict or ladder count at nominal, and no lower
strict count at low. Otherwise reject. Report full recovery and falls
separately; no stopping after favorable individual outcomes. At most one
further scalar hypothesis remains after this screen.
