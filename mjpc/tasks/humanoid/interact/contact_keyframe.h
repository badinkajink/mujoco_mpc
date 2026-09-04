// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MJPC_TASKS_HUMANOID_INTERACT_CONTACT_KEYFRAME_H_
#define MJPC_TASKS_HUMANOID_INTERACT_CONTACT_KEYFRAME_H_

#include <map>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include "nlohmann/json.hpp"
using json = nlohmann::json;

namespace mjpc::humanoid {

// ---------- Constants ----------------- //
constexpr int kNotSelectedInteract = -1;
constexpr int kNumberOfContactPairsInteract = 5;

// ---------- Enums --------------------- //
enum ContactKeyframeErrorType : int {
  kMax = 0,
  kMean = 1,
  kSum = 2,
  kNorm = 3,
};

class ContactPair {
 public:
  int body1, body2, geom1, geom2;
  mjtNum local_pos1[3], local_pos2[3];

  ContactPair()
      : body1(kNotSelectedInteract),
        body2(kNotSelectedInteract),
        geom1(kNotSelectedInteract),
        geom2(kNotSelectedInteract),
        local_pos1{0.},
        local_pos2{0.} {}

  void Reset();

  // populates the distance vector between the two contact points
  void GetDistance(mjtNum distance[3], const mjData* data) const;
};

class ContactKeyframe {
 public:
  std::string name;
  ContactPair contact_pairs[kNumberOfContactPairsInteract];

  // the direction on the xy-plane for the torso to point towards
  std::vector<mjtNum> facing_target;

  // weight of all residual terms (name -> value map)
  std::map<std::string, mjtNum> weight;

  // Opt2Skill-style per-phase contact-force reference (arXiv 2409.20514).
  // Negative = sentinel "not specified" → task falls back to its built-in
  // default (e.g. 15 N when contact is active, 0 otherwise).
  // Positive = use this exact target in the Brace Force residual.
  mjtNum brace_force_target;

  // ★ 2026-08-22 TARGET PHASE (lean strat 25 "h12_brace_targeting"): when this
  // holds exactly 3 numbers [depth_in_from_near_edge, lateral_right_of_center,
  // height_above_face] (metres, TABLE frame), the lean task treats the phase as
  // a right-arm TARGET HOVER: the reach target is built from the table geom at
  // these offsets (+ the `target_col_y` numeric added to lateral), the right
  // arm is masked out of Posture, and phase advance switches to gripper-vs-
  // target distance (so target_distance_tolerance/success_sustain_time mean
  // "hand within tol for T s"). Empty (default) = normal phase, byte-identical.
  std::vector<mjtNum> reach_target_table;

  // ★ 2026-08-24 GRASP RUNG (lean strat 27 "h12_brace_retrieval"). false
  // (default) = the reach_target_table point is graded at the GRIPPER JAW TIP
  // (the 08-23 tip-targeting fix: `right_gripper_jaw_a`'s far corner, which is
  // the gripper's lowest point on a hover and therefore the right thing to
  // hover with). true = grade it at the GRASP CENTRE instead -- the midpoint
  // between the two jaw plates, where an object must sit to be closed on.
  // The two differ by 112 mm (106 mm of it ALONG THE JAW SEPARATION AXIS,
  // because the tip is one jaw's corner, not the gripper centreline), so a
  // grasp rung graded at the tip would park the object ~11 cm off the jaws.
  // Both the residual and the phase-advance test read this, so the cost and
  // the advance always grade the SAME point (the 08-23 lesson).
  bool grasp_center;

  // ★ 2026-08-24 CLOSE RUNG (lean strat 27). true on the ONE rung where the
  // object sits between the jaws, i.e. where the grasp gate should fire the
  // gripper close and hold the ladder for the relay's verdict. A per-keyframe
  // flag rather than a model-level rung INDEX on purpose: an index is a global
  // number that would silently fire on whatever unrelated rung happens to share
  // it in another strategy (strat 24's index 5 is a standback rung). Absent =
  // false = no gate = byte-identical.
  bool grasp_close;

  // ★ 2026-08-24 SERVO RUNG (lean strat 27). true = this rung's reach target
  // TRACKS the object seen by the gripper camera: the task adds a slew-limited,
  // clamped world-space correction (measured object minus the nominal the JSON
  // was authored around) to the target. false (default) = the JSON coordinates
  // are used verbatim, which is what every pre-27 strategy wants. Set it on the
  // rungs that APPROACH the object; leave it off for lift/retract/tuck, which
  // deliberately move AWAY from it and must not chase a stale detection.
  bool servo;

  // ★ 2026-09-03 SERVO-SETTLED HOLD (lean strat 9 h12_brace_servo_sweep).
  // true = this rung's success clock only runs while the servo correction is
  // SETTLED: at least one accepted detection on this pass and the slew has
  // reached its wanted value (frozen/stale counts as settled -- the jaws
  // occlude the tag in the last cm by design). Absent = false = the clock
  // runs on distance alone = byte-identical.
  bool servo_hold;

  // ★ 2026-08-26 TILTED APPROACH (lean strat 28 "h12_brace_vision_retrieval").
  // Per-keyframe pitch-down [deg] of the gripper approach axis for the
  // "Reach Level" cost. 0 (default) = use the model numeric
  // reach_level_pitch_deg (which ships 0 = level), so every existing strategy
  // is byte-identical. Set ~40 on the vision grasp rungs: tilting the approach
  // drops the grasp centre ~0.19*sin(pitch) below the wrist, which is what
  // lets the braced arm reach a table-height block WITHOUT the squat.
  mjtNum reach_pitch_deg;

  // ★ 2026-08-26 FAIL-SOFT TIMEOUT (lean strat 28). false (default) = a
  // time-limit expiry RESETS the ladder to keyframe 0 (the historical
  // behaviour -- right for bring-up phases, catastrophic from a deep braced
  // lean: v2/v3 collapsed to base_z ~0.5 regressing to stand mid-lean). true =
  // expiry ADVANCES to the NEXT keyframe instead: a stuck vision/grasp rung
  // aborts its attempt and flows forward into retract -> release -> the proven
  // standback recovery, empty-handed but upright.
  bool timeout_advance;

  ContactKeyframe()
      : name(""),
        contact_pairs{},
        facing_target(),
        weight(),
        brace_force_target(-1.),
        reach_target_table(),
        grasp_center(false),
        grasp_close(false),
        servo(false),
        servo_hold(false),
        reach_pitch_deg(0.),
        timeout_advance(false),
        time_limit(10.),
        success_sustain_time(2.),
        target_distance_tolerance(0.1),
        target_ramp_sec(-1.) {}

  void Reset();

  mjtNum time_limit;  // maximum time (in seconds) allowed for attempting a
                      // single keyframe before resetting
  mjtNum success_sustain_time;  // minimum time (in seconds) that the objective
                                // needs to be satisfied within the distance
                                // threshold to consider the keyframe successful
  mjtNum target_distance_tolerance;  // the proximity to the keyframe objective
                                     // that needs to be maintained for a
                                     // certain time
  mjtNum target_ramp_sec;  // per-phase override (s) for the planner's target-pose
                           // ramp when ENTERING this phase. <0 = use the lean
                           // global default (kPhaseRampSeconds); 0 = snap. Lets a
                           // cyclic strategy ease its target in slowly (e.g. the
                           // squatter stand_up) so it does not launch the body.
};

void to_json(json& j, const ContactPair& contact_pair);
void from_json(const json& j, ContactPair& contact_pair);
void to_json(json& j, const ContactKeyframe& keyframe);
void from_json(const json& j, ContactKeyframe& keyframe);

}  // namespace mjpc::humanoid

#endif  // MJPC_TASKS_HUMANOID_INTERACT_CONTACT_KEYFRAME_H_
