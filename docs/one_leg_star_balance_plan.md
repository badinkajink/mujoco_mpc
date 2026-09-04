# One-Legged "Star Pose" Balance for H1-2 (H12) — Design Plan

**Status:** Research / design only. No implementation yet.
**Branch context:** researched from origin/icra2026 (read-only worktree at ~/Desktop/MJPC/icra2026_ro); intended to land on the lace/icra2026 line.
**Robot:** Unitree H1-2 ("H12") with the articulated Magpie gripper model.
**Goal:** Robot stands on ONE leg in a "star" pose — arms extended fully out to the sides, torso leaning slightly toward the stance foot, free leg lifted/abducted out to the side — held with quasi-static, near-perfect stability.

## 1. Feasibility verdict
Feasible; ~80% of the machinery already exists in the `beginning` and `stabilize` tasks on icra2026. The star pose is primarily: (1) one new cost term (CoM/ICP over the single stance sole), (2) a new pose keyframe (arms out + free leg abducted), (3) a 4-phase entry sequence (shift -> unload -> lift -> hold).

## 2. Decisive hardware constraint — the ankle cannot balance you sideways
From mjpc/tasks/humanoid_bench/h1_2_base/h1_2_pos.xml:
- hip_yaw: Z, ±0.43 rad (±25°), 200 Nm — minor.
- hip_pitch: Y, -3.14..2.5, 300 Nm — strong.
- hip_roll: X, L -0.43..3.14 / R -3.14..0.43, 300 Nm — abduction (lifts free leg sideways; huge range). Adduction toward midline capped ~0.43 rad (~25°) -> limits pelvis shift over stance foot.
- knee: Y, -0.26..2.05, 300 Nm — strong.
- ankle_pitch: Y, -0.90..0.52 (-51°..+30°), 75 Nm — weak.
- ankle_roll: X, ±0.26 rad (±15° ONLY), 40-75 Nm — the lateral balance DoF, and it's tiny.
Model authors annotate the ankle as "the only balance actuator" with a "~5° ankle-authority cliff". Consequence: lateral one-leg balance MUST come from hip strategy (support-hip roll within ~25° adduction budget) + whole-body angular momentum (arms + raised leg as counterweights). That is WHY arms-out+leg-out is the right physical design: it maximizes rotational inertia so small joint moves shift CoM / make corrective momentum without ankle torque the robot lacks. Add an ankle-action tax so the planner prefers hip over ankle. Other facts: ~70 kg, ~1.78 m, pelvis z ~1.03 m (high CoM), small sole (~2-3 cm lateral margin), shoulder_roll abducts ~195° at modest torque (arms help via momentum, not brute force — keep arm posture weight low).

## 3. Physics
(1) Static: CoM ground-projection inside the single sole (few cm) — aim center. (2) Dynamic: ICP = CoM_xy + CoM_vel·sqrt(z/g), tau~0.3s, stay inside the sole — dominant cost, already computed. (3) Rotational: regulate centroidal angular momentum L_cm (subtree_angmom at pelvis) -> 0; on a tip throw counter-momentum with torso+arms. Static one-leg balance = no stepping + weak ankle -> hip- and momentum-dominant.

## 4. Which task to build on — IMPORTANT correction
Both beginning and stabilize are WHOLE-BODY and DO actuate the arms (the "stabilize is legs-only" label is WRONG). beginning: nu=27, nq=41; residual code touches shoulder/elbow/arm 403x; has free-space arm-pose strategies (arms_sideways, arms_overhead, single_arm_raise, counterbalance_standing). stabilize: also actuates arms (357 refs; comments describe swinging the heavy arm backward as a counterweight) BUT its arm use is brace/reach-against-a-surface inherited from the lean task; strategies are all leg/gait/recovery.
DECISION: Base = beginning (whole-body, native arms-out pose authoring, already has the balance residual stack). Port from stabilize the more-mature single-support LEG machinery: collapse-polygon-to-one-foot (stabilize.cc:1398-1631), readiness-gated leg lift (2032-2158), force-gated swing-foot unload (2795-2809), recovery tiers (2541-2665). Neither alone suffices.

## 5. Reuse vs build
Reusable: capture-point/ICP balance barrier (beginning Balance term ~1757-2220; stabilize collapses to one foot 1398-1631); Lateral CoM centering (beginning term 36 :3536-3577; stabilize 2672-2731 — retarget midfoot->stance sole); centroidal angular-momentum regulation (beginning term 35 :3527-3529; stabilize 2541-2665); pose = joint-space posture keyframe (new <key> named by phase drives Posture term 14 + Control term 16; arms-out = shoulder_roll ±1.5); dormant leg-lift pair (stabilize 2032-2158; dead in beginning via is_leg_lift_stage_early name-gate); swing-foot force unload; foot-flat/upright, phase-ramped weights, integral CoM trim.
MISSING (build): (1) a true CoM/ICP-over-SINGLE-stance-foot cost — both project onto the two-foot polygon and center on midfoot; retarget to the stance sole with a small lateral bias. (2) a sustained single-foot lift (live lift is cyclic gait SwingBell -> 0; hold needs the leg-lift gate re-enabled or pure posture-keyframe lift with stance-side balance re-centered). (3) Symmetry weight = 0. (4) stance anchors made single-support aware (zero swing-foot XY anchor).

## 6. Cost set (relative weights; norms in mjpc/norm.h: 0 Quadratic,2 L2,3 Cosh,6 SmoothAbs,8 Rectify)
Core: Balance/ICP-over-stance-sole SmoothAbs ~40-60; Lateral CoM centering (over stance foot) ~300; angular momentum ->0 (esp roll Lx) ~10-15; CoM vel ->0 ~10. Geometry: stance-foot flat/no-roll (foot_up->(0,0,1)) SmoothAbs ~2000; stance loaded>=mg (Rectify) & no-slip moderate-high; swing-foot height+lateral-abduction target (smooth bell) ~180; torso/pelvis upright + slight lean ~100; arms-out posture LOW (~0.015/joint); joint-posture reg per-joint ~20-100. Regularizers: joint-vel ~0.01, control ~0.02, joint/vel-limit barriers tight on ankle-roll & hip adduction, ankle-action tax.

## 7. Phased entry (never lift-then-balance)
1 Shift (double support): ramp CoM/ICP over the stance foot (hip adduction, budget ~25°); begin torso lean + arm abduction. 2 Unload: Rectify drives swing-foot contact force ->0; GATE advance on the touch sensor reaching ~zero. 3 Lift+abduct: swing-foot height+sideways target on a smooth bell; raise balance/angmom/foot-flat weights, release double-support terms off swing leg. 4 Hold: full arms-out+leg-out targets, CoM centered on sole w/ margin, L_cm->0, min control/vel, integral trim. 5 Recovery (optional, no stepping): on large capture excursion target counter angular momentum instead of zero (stabilize stand_recover_*, default off).

## 8. MuJoCo/MJPC notes
subtreecom/subtreelinvel/subtreeangmom sensors (root at pelvis); subtree_angmom[3*pelvis_id]=[Lx,Ly,Lz]. CoM ground proj = subtree_com[0:2]. Per-foot load: <touch> site, <force>/<torque>, cfrc_ext, or iterate data->contact[] + mj_contactForce (no foot force sensor exists today). ZMP/CoP: CoP_xy = sum(p_i·f_iz)/sum f_iz. Upright: framezaxis objtype=xbody ->(0,0,1). Cost model: l = sum w_i·norm_i(residual_i); one <user> sensor per block, residuals in same order, dims must match.

## 9. Open decisions (need user input before implementing)
1 Stance foot: default left/right, parameterized or hard-picked? 2 Scope: static hold only, or include push-recovery counter-momentum tier? 3 New task vs new strategy in beginning: add a star_pose strategy+<key> in beginning, or fork a dedicated onelegstand task borrowing beginning's body + stabilize's single-support residuals?

## 10. Key refs
Pratt Capture Point (Humanoids 2006); MPC Capture Point arXiv:2307.13243; Macchietto/Zordan Momentum Control (SIGGRAPH 2009); Horak & Nashner (1986); HuB Learning Extreme Humanoid Balance arXiv:2505.07294; One-Foot Balance via MPC PMC9775477; Unitree "About H1-2". Local: beginning/*, stabilize/*, h1_2_base/h1_2_pos.xml, norm.h.
