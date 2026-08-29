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

#include "mjpc/tasks/humanoid/interact/contact_keyframe.h"

namespace mjpc::humanoid {

void ContactPair::Reset() {
  body1 = kNotSelectedInteract;
  body2 = kNotSelectedInteract;
  geom1 = kNotSelectedInteract;
  geom2 = kNotSelectedInteract;
  for (int i = 0; i < 3; i++) {
    local_pos1[i] = 0.;
    local_pos2[i] = 0.;
  }
}

void ContactPair::GetDistance(mjtNum distance[3], const mjData* data) const {
  mjtNum selected_global_pos1[3] = {0.};
  mju_mulMatVec(selected_global_pos1, data->xmat + 9 * body1, local_pos1, 3, 3);
  mju_addTo3(selected_global_pos1, data->xpos + 3 * body1);

  mjtNum selected_global_pos2[3] = {0.};
  mju_mulMatVec(selected_global_pos2, data->xmat + 9 * body2, local_pos2, 3, 3);
  mju_addTo3(selected_global_pos2, data->xpos + 3 * body2);

  mju_sub3(distance, selected_global_pos1, selected_global_pos2);
}

void ContactKeyframe::Reset() {
  name.clear();
  for (auto& contact_pair : contact_pairs) contact_pair.Reset();

  facing_target.clear();
  weight.clear();
  brace_force_target = -1.;
  reach_target_table.clear();
  grasp_center = false;
  grasp_close = false;
  servo = false;
  reach_pitch_deg = 0.;
  timeout_advance = false;
}

void to_json(json& j, const ContactPair& contact_pair) {
  j = json{{"body1", contact_pair.body1},
           {"body2", contact_pair.body2},
           {"geom1", contact_pair.geom1},
           {"geom2", contact_pair.geom2},
           {"local_pos1", contact_pair.local_pos1},
           {"local_pos2", contact_pair.local_pos2}};
}

void from_json(const json& j, ContactPair& contact_pair) {
  j.at("body1").get_to(contact_pair.body1);
  j.at("body2").get_to(contact_pair.body2);
  j.at("geom1").get_to(contact_pair.geom1);
  j.at("geom2").get_to(contact_pair.geom2);
  j.at("local_pos1").get_to(contact_pair.local_pos1);
  j.at("local_pos2").get_to(contact_pair.local_pos2);
}

void to_json(json& j, const ContactKeyframe& keyframe) {
  j = json{{"name", keyframe.name},
           {"contacts", keyframe.contact_pairs},
           {"facing_target", keyframe.facing_target},
           {"time_limit", keyframe.time_limit},
           {"success_sustain_time", keyframe.success_sustain_time},
           {"target_distance_tolerance", keyframe.target_distance_tolerance},
           {"weight", keyframe.weight},
           {"brace_force_target", keyframe.brace_force_target},
           {"target_ramp_sec", keyframe.target_ramp_sec},
           {"reach_target_table", keyframe.reach_target_table},
           {"grasp_center", keyframe.grasp_center},
           {"grasp_close", keyframe.grasp_close},
           {"servo", keyframe.servo},
           {"reach_pitch_deg", keyframe.reach_pitch_deg},
           {"timeout_advance", keyframe.timeout_advance}};
}

void from_json(const json& j, ContactKeyframe& keyframe) {
  j.at("name").get_to(keyframe.name);
  j.at("contacts").get_to(keyframe.contact_pairs);
  j.at("facing_target").get_to(keyframe.facing_target);
  j.at("time_limit").get_to(keyframe.time_limit);
  j.at("success_sustain_time").get_to(keyframe.success_sustain_time);
  j.at("target_distance_tolerance").get_to(keyframe.target_distance_tolerance);
  j.at("weight").get_to(keyframe.weight);
  // Optional fields — older strategy JSONs without them get the sentinel -1.
  keyframe.brace_force_target = j.value("brace_force_target", -1.);
  keyframe.target_ramp_sec = j.value("target_ramp_sec", -1.);
  // ★ 2026-08-22 target-phase field (lean strat 25); absent = empty = normal.
  keyframe.reach_target_table =
      j.value("reach_target_table", std::vector<mjtNum>());
  // ★ 2026-08-24 grasp rung (lean strat 27); absent = false = grade at the tip.
  keyframe.grasp_center = j.value("grasp_center", false);
  keyframe.grasp_close = j.value("grasp_close", false);
  keyframe.servo = j.value("servo", false);
  // ★ 2026-08-26 tilted approach (lean strat 28); absent = 0 = level/global.
  keyframe.reach_pitch_deg = j.value("reach_pitch_deg", 0.);
  // ★ 2026-08-26 fail-soft timeout (lean strat 28); absent = false = reset.
  keyframe.timeout_advance = j.value("timeout_advance", false);
}
}  // namespace mjpc::humanoid
