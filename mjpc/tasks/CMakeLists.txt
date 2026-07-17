# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copy the model files to the binary directory to make them available to the
# built binary.
message(STATUS "h12_SOURCE_DIR = ${h12_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR = ${CMAKE_CURRENT_BINARY_DIR}")

add_custom_target(
  copy_model_resources ALL
  ## dm_control models
  # acrobot
  COMMAND ${CMAKE_COMMAND} -E copy
          ${dm_control_SOURCE_DIR}/dm_control/suite/acrobot.xml
          ${CMAKE_CURRENT_BINARY_DIR}/acrobot/acrobot.xml
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/acrobot/acrobot_modified.xml
          ${CMAKE_CURRENT_BINARY_DIR}/acrobot/acrobot.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/acrobot/acrobot.xml.patch
  # cartpole
  COMMAND ${CMAKE_COMMAND} -E copy
          ${dm_control_SOURCE_DIR}/dm_control/suite/cartpole.xml
          ${CMAKE_CURRENT_BINARY_DIR}/cartpole/cartpole.xml
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/cartpole/cartpole_modified.xml
          ${CMAKE_CURRENT_BINARY_DIR}/cartpole/cartpole.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/cartpole/cartpole.xml.patch
  # humanoid
  COMMAND ${CMAKE_COMMAND} -E copy
          ${dm_control_SOURCE_DIR}/dm_control/suite/humanoid.xml
          ${CMAKE_CURRENT_BINARY_DIR}/humanoid/humanoid.xml
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/humanoid/humanoid_modified.xml
          ${CMAKE_CURRENT_BINARY_DIR}/humanoid/humanoid.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/humanoid/humanoid.xml.patch
  # particle
  COMMAND ${CMAKE_COMMAND} -E copy
          ${dm_control_SOURCE_DIR}/dm_control/suite/point_mass.xml
          ${CMAKE_CURRENT_BINARY_DIR}/particle/particle.xml
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/particle/particle_modified.xml
          ${CMAKE_CURRENT_BINARY_DIR}/particle/particle.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/particle/particle.xml.patch
  # swimmer
  COMMAND ${CMAKE_COMMAND} -E copy
          ${dm_control_SOURCE_DIR}/dm_control/suite/swimmer.xml
          ${CMAKE_CURRENT_BINARY_DIR}/swimmer/swimmer.xml
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/swimmer/swimmer_modified.xml
          ${CMAKE_CURRENT_BINARY_DIR}/swimmer/swimmer.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/swimmer/swimmer.xml.patch
  # walker
  COMMAND ${CMAKE_COMMAND} -E copy
          ${dm_control_SOURCE_DIR}/dm_control/suite/walker.xml
          ${CMAKE_CURRENT_BINARY_DIR}/walker/walker.xml
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/walker/walker_modified.xml
          ${CMAKE_CURRENT_BINARY_DIR}/walker/walker.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/walker/walker.xml.patch

  ## Menagerie models
  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/shadow_hand/right_hand.xml
          ${CMAKE_CURRENT_BINARY_DIR}/shadow_reorient/right_hand.xml
  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/wonik_allegro/right_hand.xml
          ${CMAKE_CURRENT_BINARY_DIR}/allegro/right_hand.xml
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/shadow_hand/assets
          ${CMAKE_CURRENT_BINARY_DIR}/shadow_reorient/assets

  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/wonik_allegro/assets
          ${CMAKE_CURRENT_BINARY_DIR}/allegro/assets
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/allegro/right_hand_modified.xml
          ${CMAKE_CURRENT_BINARY_DIR}/allegro/right_hand.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/allegro/right_hand.xml.patch

  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/franka_emika_panda/panda.xml
          ${CMAKE_CURRENT_BINARY_DIR}/panda/panda.xml
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/franka_emika_panda/assets
          ${CMAKE_CURRENT_BINARY_DIR}/panda/assets
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/panda/panda_modified.xml
          ${CMAKE_CURRENT_BINARY_DIR}/panda/panda.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/panda/panda.xml.patch

  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/unitree_a1/a1.xml
          ${CMAKE_CURRENT_BINARY_DIR}/quadruped/a1.xml
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/unitree_a1/assets
          ${CMAKE_CURRENT_BINARY_DIR}/quadruped/assets
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/quadruped/a1_modified.xml
          ${CMAKE_CURRENT_BINARY_DIR}/quadruped/a1.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/quadruped/a1.xml.patch

  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/unitree_go1/assets
          ${CMAKE_CURRENT_BINARY_DIR}/go1/assets
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/franka_emika_panda
          ${CMAKE_CURRENT_BINARY_DIR}/manipulation
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/robotiq_2f85
          ${CMAKE_CURRENT_BINARY_DIR}/manipulation

  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/skydio_x2/x2.xml
          ${CMAKE_CURRENT_BINARY_DIR}/quadrotor/quadrotor.xml
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/skydio_x2/assets
          ${CMAKE_CURRENT_BINARY_DIR}/quadrotor/assets
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/quadrotor/quadrotor_modified.xml
          ${CMAKE_CURRENT_BINARY_DIR}/quadrotor/quadrotor.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/quadrotor/quadrotor.xml.patch

  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/universal_robots_ur5e/assets
          ${CMAKE_CURRENT_BINARY_DIR}/ur5/assets/ur5e
  COMMAND ${CMAKE_COMMAND} -E copy
        ${menagerie_SOURCE_DIR}/universal_robots_ur5e/ur5e.xml
        ${CMAKE_CURRENT_BINARY_DIR}/ur5/ur5e.xml
  COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${h12_SOURCE_DIR}/magpie/assets
        ${CMAKE_CURRENT_BINARY_DIR}/ur5/assets
  COMMAND ${CMAKE_COMMAND} -E copy
        ${h12_SOURCE_DIR}/magpie/magpie_gripper.xml
        ${CMAKE_CURRENT_BINARY_DIR}/ur5/magpie_gripper.xml
#   COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/ur5/ur5_magpie.xml
#           ${CMAKE_CURRENT_BINARY_DIR}/ur5/ur5e.xml
#           <${CMAKE_CURRENT_SOURCE_DIR}/ur5/ur5e_magpie.xml.patch
  COMMAND ${CMAKE_COMMAND} -E copy
        ${h12_SOURCE_DIR}/magpie/ur5_magpie.xml
        ${CMAKE_CURRENT_BINARY_DIR}/ur5/ur5_magpie.xml
  COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${h12_SOURCE_DIR}/inspire/assets
        ${CMAKE_CURRENT_BINARY_DIR}/ur5/assets
  COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${h12_SOURCE_DIR}/inspire/assets/visual
        ${CMAKE_CURRENT_BINARY_DIR}/ur5/assets/visual
  COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${h12_SOURCE_DIR}/inspire/assets/collision
        ${CMAKE_CURRENT_BINARY_DIR}/ur5/assets/collision
  COMMAND ${CMAKE_COMMAND} -E copy
        ${h12_SOURCE_DIR}/inspire/inspire_right_ur5.xml
        ${CMAKE_CURRENT_BINARY_DIR}/ur5/inspire_right_ur5.xml
#   COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/ur5/ur5_inspire.xml
#           ${CMAKE_CURRENT_BINARY_DIR}/ur5/ur5e.xml
#           <${CMAKE_CURRENT_SOURCE_DIR}/ur5/ur5e_inspire.xml.patch
  COMMAND ${CMAKE_COMMAND} -E copy
        ${h12_SOURCE_DIR}/inspire/ur5_inspire.xml
        ${CMAKE_CURRENT_BINARY_DIR}/ur5/ur5_inspire.xml

  COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${h12_SOURCE_DIR}/inspire/assets
        ${CMAKE_CURRENT_BINARY_DIR}/grasp/assets
  COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${h12_SOURCE_DIR}/inspire/assets/visual
        ${CMAKE_CURRENT_BINARY_DIR}/grasp/assets/visual
  COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${h12_SOURCE_DIR}/inspire/assets/collision
        ${CMAKE_CURRENT_BINARY_DIR}/grasp/assets/collision
  COMMAND ${CMAKE_COMMAND} -E copy
        ${h12_SOURCE_DIR}/inspire/inspire_right_floating.xml
        ${CMAKE_CURRENT_BINARY_DIR}/grasp/inspire.xml


  ## Cube reorientation
  # patch cube from common assets
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/common_assets/cube_modified_shadow_reorient.xml
          ${CMAKE_CURRENT_BINARY_DIR}/common_assets/reorientation_cube.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/common_assets/cube.xml.patch

  ## Cube solve task
  # copy cube model from MuJoCo
  COMMAND ${CMAKE_COMMAND} -E copy
          ${mujoco_SOURCE_DIR}/model/cube/cube_3x3x3.xml
          ${CMAKE_CURRENT_BINARY_DIR}/rubik/cube_3x3x3.xml
  # copy cube assets from MuJoCo
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${mujoco_SOURCE_DIR}/model/cube/assets
          ${CMAKE_CURRENT_BINARY_DIR}/rubik/assets
  # modified cube model for task
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/rubik/cube_3x3x3_modified.xml
          ${CMAKE_CURRENT_BINARY_DIR}/rubik/cube_3x3x3.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/rubik/cube_3x3x3.xml.patch
  # modified cube model to transition model for scramble mode
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/rubik/transition_model.xml
          ${CMAKE_CURRENT_BINARY_DIR}/rubik/cube_3x3x3.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/rubik/transition_model.xml.patch
  # copy hand model from Menagerie
  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/shadow_hand/right_hand.xml
          ${CMAKE_CURRENT_BINARY_DIR}/rubik/right_hand.xml
  # copy hand assets from Menagerie
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/shadow_hand/assets
          ${CMAKE_CURRENT_BINARY_DIR}/rubik/assets

  # ALOHA
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/aloha/assets
          ${CMAKE_CURRENT_BINARY_DIR}/bimanual/assets
  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/aloha/aloha.xml
          ${CMAKE_CURRENT_BINARY_DIR}/bimanual/aloha.xml
  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/aloha/filtered_cartesian_actuators.xml
          ${CMAKE_CURRENT_BINARY_DIR}/bimanual/filtered_cartesian_actuators.xml
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/bimanual/aloha_cartesian.xml
          ${CMAKE_CURRENT_BINARY_DIR}/bimanual/aloha.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/bimanual/aloha.patch

  # MJX ALOHA
  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/aloha/mjx_aloha.patch
          ${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_aloha.patch
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_aloha.xml
          ${CMAKE_CURRENT_BINARY_DIR}/bimanual/aloha.xml
          <${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_aloha.patch
  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/aloha/scene.xml
          ${CMAKE_CURRENT_BINARY_DIR}/bimanual/scene.xml
  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/aloha/mjx_scene.patch
          ${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_scene.patch
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_scene.xml
          ${CMAKE_CURRENT_BINARY_DIR}/bimanual/scene.xml
          <${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_scene.patch

  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/aloha/mjx_filtered_cartesian_actuators.patch
          ${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_filtered_cartesian_actuators.patch
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_filtered_cartesian_actuators.xml
          ${CMAKE_CURRENT_BINARY_DIR}/bimanual/filtered_cartesian_actuators.xml
          <${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_filtered_cartesian_actuators.patch

  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/robotis_op3/op3.xml
          ${CMAKE_CURRENT_BINARY_DIR}/op3/op3.xml
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/robotis_op3/assets
          ${CMAKE_CURRENT_BINARY_DIR}/op3/assets
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/op3/op3_modified.xml
          ${CMAKE_CURRENT_BINARY_DIR}/op3/op3.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/op3/op3.xml.patch

  COMMAND ${Python_EXECUTABLE}
          ${CMAKE_CURRENT_BINARY_DIR}/manipulation/merge_panda_robotiq.py
          ${CMAKE_CURRENT_BINARY_DIR}/manipulation/panda_robotiq.xml

  # Unitree H1
  COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${menagerie_SOURCE_DIR}/unitree_h1/assets
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/h1
  COMMAND ${CMAKE_COMMAND} -E copy
        ${menagerie_SOURCE_DIR}/unitree_h1/h1.xml
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/h1/h1.xml
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/h1/h1_modified.xml
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/h1/h1.xml
        <${CMAKE_CURRENT_SOURCE_DIR}/humanoid_bench/h1.xml.patch

#   # Unitree H1-2
  COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${h12_SOURCE_DIR}/unitree_robots/h1_2/meshes
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/h1_2
  COMMAND ${CMAKE_COMMAND} -E copy
        ${h12_SOURCE_DIR}/unitree_robots/h1_2/h1_2_handless.xml
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/h1_2/h1_2.xml
  # VENDORED planner base (2026-07-03): h1_2_pos.xml / h1_2_pos_hands.xml /
  # magpie_h12.stl come from humanoid_bench/h1_2_base/ in THIS repo, NOT from the
  # h12 FetchContent checkout. The validated planner base had drifted from the
  # pinned h1_mujoco ef83c84 as UNCOMMITTED edits inside the host build's
  # _deps/h12-src (all real-robot + twin tuning ran on it, and the magpie patch
  # below is regenerated against it) -- any fresh clone/container build fetched
  # the wrong robot model and the patch hunk-rejected. These vendored bytes ==
  # that validated _deps state, so every build everywhere stages the same model.
  COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/humanoid_bench/h1_2_base/magpie_h12.stl
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/h1_2/magpie_h12.stl
  COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/humanoid_bench/h1_2_base/h1_2_pos.xml
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/h1_2/h1_2_modified.xml
  # Magpie model (2026-07-02): COPY the gripperless base as the "magpie" variant FIRST,
  # so the lower-body stabilize model is derived below from the gripperless condim-4 base
  # (exactly what the stand / hip-throw tuning + the real deploy are validated on). The
  # FULL magpie patch (welded gripper bodies + forearm brace pads/sites) is RE-APPLIED to
  # this file AFTER the stabilize derivation (see below) so the Lean/reach tasks -- which
  # reference left/right_magpie_gripper bodies + forearm_brace sites -- get the complete
  # model, while the stabilize/deploy model does NOT silently gain ~1 kg of gripper mass
  # mid-tuning. (The old 934c59a "patch is stale, 2 hunks reject" note is obsolete: against
  # the re-synced pinned base the patch now applies cleanly -- Lean loads again.)
  COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/h1_2/h1_2_modified.xml
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/h1_2/h1_2_modified_magpie.xml
  # Lower-body "stabilize" robot: derive the nu=12 legs-only variant from the
  # (just-built) magpie robot -- drop the 15 upper-body actuators and
  # equality-lock those joints at home. Then mirror lean's mesh symlinks so the
  # stabilize model resolves the shared h1_2 STLs.
  COMMAND ${Python_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/humanoid_bench/stabilize/_gen_stabilize_model.py
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/h1_2/h1_2_modified_magpie.xml
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/h1_2/h1_2_stabilize_magpie.xml
  # Stand robot (2026-07-16): derive a stand-ONLY variant of the gripperless base
  # carrying the REAL robot constants -- PD gains from h12_ros2_controller's
  # safety_full.yaml, forcerange/ctrlrange from the h12_safety_layer estop ratios x
  # URDF torque, and joint damping restored to the upstream 0. Stand-scoped on purpose:
  # every other H1-2 task is tuned against the base's (wrong) numbers, so they keep
  # loading h1_2_modified.xml untouched. See _gen_stand_model.py for provenance.
  COMMAND ${Python_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/humanoid_bench/stand/_gen_stand_model.py
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/h1_2/h1_2_modified.xml
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/h1_2/h1_2_stand.xml
  # Legs-only stand robot (2026-07-17): EQUALITY-LOCK torso+arms at home (keep the 15
  # upper joints + DOF, drop only their actuators) by reusing the stabilize derivation
  # on the real-constants stand model. Eq-lock, NOT weld, because the DEPLOY legs-only
  # node (h12_lower_body_controller.cc) injects the 15 MEASURED arm angles into
  # qpos[19..33] via equality-lock retargeting (deploy_common.cc:713,942) so the
  # planner CoM tracks the real arms -- a welded model has no arm DOF and breaks that.
  # Using the SAME eq-locked model for the MJPC-only benchmark keeps the only
  # difference between MJPC-only and sim2sim the pipeline, not the robot. nq=34, nv=33,
  # nu=12; ~1.1x planner speedup over full (vs 2.17x for weld -- deploy compat wins).
  COMMAND ${Python_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/humanoid_bench/stabilize/_gen_stabilize_model.py
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/h1_2/h1_2_stand.xml
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/h1_2/h1_2_stand_lower.xml
  # Upper-body "upper" robot (2026-07-10): derive the nu=15 upper-only variant
  # from the SAME gripperless base (before the gripper re-patch below) -- keep
  # the 15 torso+arm actuators, eq-lock the 12 legs, weld the pelvis. Mirror of
  # the stabilize derivation; plan h12/upper_body_mpc_controller_plan_2026-07-09.md.
  COMMAND ${Python_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/humanoid_bench/upper/_gen_upper_model.py
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/h1_2/h1_2_modified_magpie.xml
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/h1_2/h1_2_upper_magpie.xml
  COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/upper/meshes
  COMMAND ${CMAKE_COMMAND} -E rm -f
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/upper/meshes/h1_2
  COMMAND ${CMAKE_COMMAND} -E create_symlink ../../h1_2
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/upper/meshes/h1_2
  COMMAND ${CMAKE_COMMAND} -E rm -f
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/upper/h1_2
  COMMAND ${CMAKE_COMMAND} -E create_symlink meshes/h1_2
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/upper/h1_2
  # NOW re-apply the magpie patch to the magpie model (grippers + forearm brace pads/
  # sites) for the Lean/reach tasks -- AFTER the gripperless stabilize model was derived
  # above, so the stabilize/deploy model stays gripperless (tuning-safe). The tolerant
  # cmake -P wrapper keeps a FUTURE re-staleness from aborting copy_model_resources (a
  # hard dep of libmjpc): worst case Lean would miss the pads, but the build still
  # succeeds. (Was `bash -c "... || true"`, but CMake 3.22's Ninja generator leaks the
  # `|| true` out of the -c string, so a patch failure aborted the whole target in the
  # HAMS ros container; the -P script is generator/shell-proof.) The reject file (if any
  # hunk rejects against a future base) is discarded -- harmless to Lean.
  COMMAND ${CMAKE_COMMAND}
        -DTARGET_FILE=${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/h1_2/h1_2_modified_magpie.xml
        -DPATCH_FILE=${CMAKE_CURRENT_SOURCE_DIR}/humanoid_bench/h1_2_modified_magpie.xml.patch
        -P ${CMAKE_CURRENT_SOURCE_DIR}/humanoid_bench/apply_patch_tolerant.cmake
  COMMAND ${CMAKE_COMMAND} -E rm -f ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/h1_2/h1_2_modified_magpie.xml.rej
  COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/stabilize/meshes
  COMMAND ${CMAKE_COMMAND} -E rm -f
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/stabilize/meshes/h1_2
  COMMAND ${CMAKE_COMMAND} -E create_symlink ../../h1_2
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/stabilize/meshes/h1_2
  COMMAND ${CMAKE_COMMAND} -E rm -f
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/stabilize/h1_2
  COMMAND ${CMAKE_COMMAND} -E create_symlink meshes/h1_2
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/stabilize/h1_2
  COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/humanoid_bench/h1_2_base/h1_2_pos_hands.xml
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/h1_2/h1_2_modified_hands.xml
  COMMAND ${CMAKE_COMMAND} -E copy
        ${h12_SOURCE_DIR}/unitree_robots/h1_2/h1_2_torso_skin.xml
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/h1_2/h1_2_modified_torso_skin.xml
  COMMAND ${CMAKE_COMMAND} -E copy
        ${h12_SOURCE_DIR}/unitree_robots/h1_2/h12_tof.xml
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/h1_2/h1_2_skin_tof.xml
  COMMAND ${CMAKE_COMMAND} -E copy
        ${h12_SOURCE_DIR}/unitree_robots/h1_2/h12_tof.xml
        ${CMAKE_CURRENT_BINARY_DIR}/humanoid_bench/h1_2/h1_2_skin_cap_tof.xml
  COMMENT "Copying Menagerie and dm_control assets into binary directory")

add_custom_target(gen_lean_pipeline ALL
        COMMAND ${Python_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/humanoid_bench/lean/strategies/_gen_lean_pipeline.py
        COMMENT "Generating lean pipeline strategies 31-35 from pre-lean twins")

add_custom_target(copy_resources ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Copying tasks into binary directory")

add_dependencies(copy_resources gen_lean_pipeline)

add_dependencies(copy_model_resources copy_resources)

add_dependencies(libmjpc copy_model_resources)
