# Loco gait: commit-then-track + between-plan feedback (design, 2026-10-14)

## What the data says (realchain twin, est v3, leaner safety, 17 traj / 18 threads = 50-57 plans/s)
* Stand 2/2 (90 s), push recovery intact (fwd 50 N, back 20 N, lateral 80 N with catch steps).
* Drive: W 0.15 x 3.5 s 0/8, W 0.08 x 3 s 1/2, Q 0.25 x 3.5 s 2/5, E 0/2, A 0/1.
* Isolation: estimator off / safety relaxed / both -> same fall. Beginning task -> same fall. Not plumbing.
* Fall signature: within 2 s of `drive: WALK` the stance ankle pitch sits at 46-54 Nm (planner budget
  48.6, motor 60), pitch 5-10 deg forward, then collapse. Knees cycle (gait engages), weight never
  transfers onto one foot before the other lifts.
* July: survival is monotonic in plan rate; 50 Hz = 2/5 band, lockstep = 5/5. We cannot buy 100 Hz on CPU.
* Literature: CMU 2503.04613 trots a real H1 with iLQR @50 Hz **plus TV-LQR @300 Hz between solves**;
  FS-MPC 2608.19443 walks a real H1 by sampling **around a feedback policy** (plain MPPI failed).
  Both: the missing layer is fast whole-body feedback between plans, not a better optimizer.

## Method 2 first (cheaper, independent of planner): COMMIT the stance CoM plan, sampler TRACKS it
Today the swing foot is forced open-loop (ModifyControl) but the stance-side weight shift is a soft cost
("Lateral Center" rock, weight ~75-110) the sampler must discover every plan. Make it a committed reference:
1. New numeric `trot_com_commit` (0 = off, byte-identical). New ResidualFn member `com_ref_[2]`, `com_ref_ok_`.
2. In TransitionLocked at every gait phase edge (reuse the R7 contact/phase latch at loco.cc ~2538-2610):
   * stance foot = the foot that is NOT about to swing; p_st = its measured xy.
   * LIPM commit: omega = sqrt(g/z). Desired DCM at end of this stance = next frozen step target
     (step_frozen_ / Step Place target). CoM reference over the stance: x_ref(t) = p_st + (xi_0 - p_st) e^{omega t}
     clamped to the sole (|x_ref - p_st| <= 0.06 fore-aft, lateral target = p_st_y +/- 0.02 inward).
     Store (p_st, xi_0, t_edge) -> the residual evaluates x_ref(t) in rollouts (time-indexed, so rollouts see a
     consistent plan, exactly like drive_gait_amp_ is snapshotted today).
3. Residual: replace the Lateral Center target `midfoot_y + rock` by `com_ref_y(t)` when committed, and ADD the
   fore-aft error `subcom_x - com_ref_x(t)` into the existing "Velocity"/"Lateral Center" slot family
   (new 1-dim user sensor "Stance CoM X" if the XML term count must grow; strategies: weight 0 except drive/trot).
4. Weight: start at Lateral Center's weight x2 for drive; sweep {1x, 2x, 4x} on the twin qq / slowfwd plans, n=3.
   Success metric = yaw-burst survival and slowfwd 2nd-burst survival; secondary = ankle pitch p99 < 45 Nm.
Why this should move the needle: the sampler is good at tracking a smooth target (that is the stand); it is bad
at discovering a shift. The video walker's advantage is exactly this commitment.

## Method 1: between-plan whole-body feedback (the TV-LQR substitute)
ModifyControl already runs at 200 Hz on fresh state (ActionFromPolicy -> task->ModifyControl). Add a small
LINEAR stabilizer there, gated by is_trot and `trot_fb_gain` (0 = off):
* inputs: base tilt (up_x, up_y from qpos quat), base angular rate (qvel[3:5]), and the CoM-ref error from Method 2.
* outputs (additive, clamped +/-0.08 rad): stance hip pitch += -k1*tilt_x - k2*wy ; stance hip roll += -k1*tilt_y - k2*wx ;
  both shoulder pitch += +k3*wy (angular-momentum arm catch, the joint that trips today, now closed-loop).
* The residual must see the SAME offsets (two-reader rule from TrotRockHipRoll) or the next plan fights them:
  add the identical function of the rollout state to the Posture/Hip targets. This is the TV-LQR idea with a
  hand-picked gain instead of a Riccati gain; CMU's gains come from iLQR, ours from a 3-point sweep on the twin.
Order: Method 2, measure; then Method 1 on top, measure. Keep every knob default 0 so strat 6/20 stay byte-identical.

## Planner question (for the RL goal)
Real-time CEM walking on this robot is at the published limit of the planner class. For a policy, the planner is an
OFFLINE teacher: lockstep iCEM/CEM with 64+ samples in the torque-limited twin (no real-time constraint) to
generate demonstrations for behaviours RL finds hard (bracing, recovery), and PPO in the warp/mjlab twin for the
gait. Sampling MPC's remaining real-time niche here is what the paper used it for: contact-rich, cost-specified
behaviours from a stand.
