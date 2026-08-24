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

  ContactKeyframe()
      : name(""),
        contact_pairs{},
        facing_target(),
        weight(),
        brace_force_target(-1.),
        reach_target_table(),
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
