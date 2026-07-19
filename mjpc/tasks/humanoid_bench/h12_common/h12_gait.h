// H1-2 task-twin shared gait/capture primitives (stage 4b of the 2026-07-18
// reorg). These are the "MUST match" functions: the cost gait clock
// (Residual), the open-loop swing forcer (ModifyControl), and the
// TransitionLocked latches/governors of BOTH task twins must consume THESE
// implementations, never re-derived copies -- cost/forcer coherence (plan
// I10) is what makes the stepping family work at all. Header-only, pure.

#ifndef MJPC_TASKS_HUMANOID_BENCH_H12_COMMON_H12_GAIT_H_
#define MJPC_TASKS_HUMANOID_BENCH_H12_COMMON_H12_GAIT_H_

#include <mujoco/mujoco.h>

namespace mjpc {
namespace h12 {

// Swing-foot clearance bell (WSS "quiet stepping"). Replaces the sin(pi*s)
// half-sine the gait clock and ModifyControl originally used: sin() leaves the
// foot with a nonzero vertical RATE at touchdown (derivative -pi at s=1) so it
// arrives still moving down and slams; the smoothstepped triangle lands with
// ZERO velocity AND zero acceleration. Measured on lean: swing chatter
// -22-32%, +72% survival. Same peak height and mid-swing timing -- a drop-in
// for the sine, nothing retunes. Used by BOTH the cost gait clock
// (g_bump_l/r) and the ModifyControl swing forcer of both twins.
inline double SwingBell(double s) {
  s = s < 0.0 ? 0.0 : (s > 1.0 ? 1.0 : s);
  double t = (s <= 0.5) ? 2.0 * s : 2.0 - 2.0 * s;   // triangle ramp 0->1->0
  return t * t * (3.0 - 2.0 * t);                     // smoothstep of the ramp
}

// Signed capture-point excursion (linear inverted pendulum):
//   zc  = max(0.5, zc_raw)                    (height floor)
//   tau = sqrt(zc / g)                        (capture time constant)
//   ex  = zc * (tx_raw - lean_nominal_x) + tau * vx   (signed fore-aft)
//   ey  = zc *  ty                        + tau * vy   (signed lateral)
// SIGNED so a RECOVERING velocity shrinks the excursion (a settling lean
// already moving back is not "losing balance"). tx/ty are the torso up-axis
// tip components (sensor "torso_up" or quat-derived); lean_nominal_x recenters
// fore-aft about the task's designed steady lean so a back-push immediately
// reads ex < 0. Callers that want the rock-immune lateral term use zc * ty
// alone (returned zc). This one function replaces the five hand-synchronized
// copies per task file that the "MUST match" comments used to police --
// arithmetic order is exactly the original sites', so results are
// bit-identical.
struct CaptureExcursion {
  double zc;    // floored CoM/base height actually used
  double tau;   // sqrt(zc/g)
  double ex;    // signed fore-aft capture excursion
  double ey;    // signed lateral capture excursion (tilt + velocity lead)
};
inline CaptureExcursion CaptureExcursionFrom(double zc_raw, double tx_raw,
                                             double ty, double vx, double vy,
                                             double lean_nominal_x) {
  CaptureExcursion r;
  r.zc = mju_max(0.5, zc_raw);
  r.tau = mju_sqrt(r.zc / 9.81);
  const double tx = tx_raw - lean_nominal_x;
  r.ex = r.zc * tx + r.tau * vx;
  r.ey = r.zc * ty + r.tau * vy;
  return r;
}

}  // namespace h12
}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_BENCH_H12_COMMON_H12_GAIT_H_
