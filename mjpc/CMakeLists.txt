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

add_library(threadpool STATIC)
target_sources(
  threadpool
  PUBLIC threadpool.h
  PRIVATE threadpool.cc
)
target_link_libraries(
  threadpool
  absl::base
)
target_include_directories(threadpool PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(
  libmjpc STATIC
  states/state.cc
  states/state.h
  agent.cc
  agent.h
  trajectory.cc
  trajectory.h
  utilities.cc
  utilities.h
  tasks/tasks.cc
  tasks/tasks.h
  tasks/acrobot/acrobot.cc
  tasks/acrobot/acrobot.h
  tasks/allegro/allegro.cc
  tasks/allegro/allegro.h
  tasks/bimanual/insert/insert.cc
  tasks/bimanual/insert/insert.h
  tasks/bimanual/handover/handover.cc
  tasks/bimanual/handover/handover.h
  tasks/bimanual/reorient/reorient.cc
  tasks/bimanual/reorient/reorient.h
  tasks/cartpole/cartpole.cc
  tasks/cartpole/cartpole.h
  tasks/fingers/fingers.cc
  tasks/fingers/fingers.h
  tasks/humanoid/interact/interact.cc
  tasks/humanoid/interact/interact.h
  tasks/humanoid/interact/contact_keyframe.cc
  tasks/humanoid/interact/contact_keyframe.h
  tasks/humanoid/interact/motion_strategy.cc
  tasks/humanoid/interact/motion_strategy.h
  tasks/humanoid/stand/stand.cc
  tasks/humanoid/stand/stand.h
  tasks/humanoid/tracking/tracking.cc
  tasks/humanoid/tracking/tracking.h
  tasks/humanoid/walk/walk.cc
  tasks/humanoid/walk/walk.h
  tasks/manipulation/common.cc
  tasks/manipulation/common.h
  tasks/manipulation/manipulation.cc
  tasks/manipulation/manipulation.h
  tasks/op3/stand.cc
  tasks/op3/stand.h
  tasks/panda/panda.cc
  tasks/panda/panda.h
  tasks/particle/particle.cc
  tasks/particle/particle.h
  tasks/rubik/solve.cc
  tasks/rubik/solve.h
  tasks/shadow_reorient/hand.cc
  tasks/shadow_reorient/hand.h
  tasks/quadrotor/quadrotor.cc
  tasks/quadrotor/quadrotor.h
  tasks/quadruped/quadruped.cc
  tasks/quadruped/quadruped.h
  tasks/go1/quadruped.cc
  tasks/go1/quadruped.h
  tasks/swimmer/swimmer.cc
  tasks/swimmer/swimmer.h
  tasks/ur5/ur5.cc
  tasks/ur5/ur5.h
  tasks/grasp/grasp.cc
  tasks/grasp/grasp.h
  tasks/walker/walker.cc
  tasks/walker/walker.h
  tasks/humanoid_bench/utility/dm_control_utils_rewards.cpp
  tasks/humanoid_bench/walk/walk.cc
  tasks/humanoid_bench/walk/walk.h
  tasks/humanoid_bench/walk_reward.cc
  tasks/humanoid_bench/walk_reward.h
  tasks/humanoid_bench/stand/stand.cc
  tasks/humanoid_bench/stand/stand.h
  tasks/humanoid_bench/push/push.cc
  tasks/humanoid_bench/push/push.h
  tasks/humanoid_bench/h12_common/h12_plan_snapshot.h
  tasks/humanoid_bench/lean/lean.cc
  tasks/humanoid_bench/lean/lean.h
  tasks/humanoid_bench/stabilize/stabilize.cc
  tasks/humanoid_bench/stabilize/stabilize.h
  tasks/humanoid_bench/upper/upper.cc
  tasks/humanoid_bench/upper/upper.h
  tasks/humanoid_bench/avoid/avoid.cc
  tasks/humanoid_bench/avoid/avoid.h
  tasks/humanoid_bench/avoid/offline_obstacles.h
  planners/planner.cc
  planners/planner.h
  planners/policy.h
  planners/include.cc
  planners/include.h
  planners/cost_derivatives.cc
  planners/cost_derivatives.h
  planners/model_derivatives.cc
  planners/model_derivatives.h
  planners/cross_entropy/planner.cc
  planners/cross_entropy/planner.h
  planners/icem/planner.cc
  planners/icem/planner.h
  planners/icem_dr/planner.cc
  planners/icem_dr/planner.h
  planners/robust/robust_planner.cc
  planners/robust/robust_planner.h
  planners/sample_gradient/planner.cc
  planners/sample_gradient/planner.h
  planners/sampling/planner.cc
  planners/sampling/planner.h
  planners/sampling/policy.cc
  planners/sampling/policy.h
  planners/gradient/gradient.cc
  planners/gradient/gradient.h
  planners/gradient/planner.cc
  planners/gradient/planner.h
  planners/gradient/policy.cc
  planners/gradient/policy.h
  planners/gradient/settings.h
  planners/gradient/spline_mapping.cc
  planners/gradient/spline_mapping.h
  planners/ilqg/backward_pass.cc
  planners/ilqg/backward_pass.h
  planners/ilqg/boxqp.h
  planners/ilqg/planner.cc
  planners/ilqg/planner.h
  planners/ilqg/policy.cc
  planners/ilqg/policy.h
  planners/ilqg/settings.h
  planners/ilqs/planner.cc
  planners/ilqs/planner.h
  estimators/batch.cc
  estimators/batch.h
  estimators/estimator.h
  estimators/include.cc
  estimators/include.h
  estimators/kalman.cc
  estimators/kalman.h
  estimators/unscented.cc
  estimators/unscented.h
  direct/direct.cc
  direct/direct.h
  direct/trajectory.h
  direct/model_parameters.cc
  direct/model_parameters.h
  spline/spline.cc
  spline/spline.h
  app.cc
  app.h
  norm.cc
  norm.h
  simulate.cc
  simulate.h
  task.cc
  task.h
  $<TARGET_OBJECTS:mujoco::platform_ui_adapter>
)
set_target_properties(libmjpc PROPERTIES OUTPUT_NAME mjpc)
target_compile_options(libmjpc PUBLIC ${MJPC_COMPILE_OPTIONS})
target_compile_definitions(libmjpc PRIVATE MJSIMULATE_STATIC)
target_link_libraries(
  libmjpc
  absl::any_invocable
  absl::check
  absl::flat_hash_map
  absl::log
  absl::random_random
  absl::span
  glfw
  lodepng
  mujoco::mujoco
  mujoco::platform_ui_adapter
  threadpool
  Threads::Threads
  nlohmann_json::nlohmann_json
)
target_include_directories(libmjpc
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_BINARY_DIR}/..
)

add_executable(
  mjpc
  main.cc
)
target_link_libraries(
  mjpc
  absl::flags
  absl::flags_parse
  absl::log
  absl::random_random
  absl::strings
  libmjpc
  mujoco::mujoco
  threadpool
  Threads::Threads
)
target_include_directories(mjpc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(mjpc PUBLIC ${MJPC_COMPILE_OPTIONS})
target_link_options(mjpc PRIVATE ${MJPC_LINK_OPTIONS})
target_compile_definitions(mjpc PRIVATE MJSIMULATE_STATIC)
if(APPLE)
  target_sources(mjpc PRIVATE macos_gui.mm)
  target_link_libraries(mjpc "-framework Cocoa")
endif()

# Wire gRPC into mjpc when the service is enabled
if(MJPC_BUILD_GRPC_SERVICE)
  target_compile_definitions(mjpc PRIVATE MJPC_BUILD_GRPC_SERVICE)
  target_link_libraries(
    mjpc
    mjpc_ui_agent_service
    ${_GRPC_GRPCPP}
    ${_PROTOBUF_LIBPROTOBUF}
  )
  target_include_directories(mjpc PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
  )
endif()

add_library(
  libtestspeed STATIC
  testspeed_app.cc
  testspeed.h
  testspeed.cc
)
target_link_libraries(
  libtestspeed
  absl::random_random
  absl::strings
  libmjpc
  mujoco::mujoco
  threadpool
  Threads::Threads
)
target_include_directories(libtestspeed PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_executable(
  testspeed
  testspeed_app.cc
)
target_link_libraries(
  testspeed
  libtestspeed
  absl::flags
  absl::flags_parse
)
target_include_directories(testspeed PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(testspeed PUBLIC ${MJPC_COMPILE_OPTIONS})
target_link_options(testspeed PRIVATE ${MJPC_LINK_OPTIONS})
target_compile_definitions(testspeed PRIVATE MJSIMULATE_STATIC)

# -----------------------------------------------------------------------------
# Embedded DDS control node for the Unitree H1-2 (digital twin + real robot).
# OFF by default so the normal build never needs unitree_sdk2.  Enable with:
#   cmake .. -DMJPC_BUILD_DEPLOY=ON
# unitree_sdk2 is auto-located at ~/unitree_install (override by adding your own
# prefix to -DCMAKE_PREFIX_PATH).  See mjpc/deploy/README.md.
# -----------------------------------------------------------------------------
option(MJPC_BUILD_DEPLOY "Build the embedded DDS control node (needs unitree_sdk2)" OFF)
if(MJPC_BUILD_DEPLOY)
  list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/unitree_install")
  find_package(unitree_sdk2 REQUIRED)
  # bundled CycloneDDS .so ships next to the SDK; bake its dir into the rpath so
  # libddsc/libddscxx resolve at runtime regardless of LD_LIBRARY_PATH.
  get_target_property(_ddsc_loc ddsc IMPORTED_LOCATION)
  get_filename_component(_dds_libdir "${_ddsc_loc}" DIRECTORY)

  # Shared deploy sources -- when stage 3 splits deploy_common.cc, extend THIS
  # list (and the consumer package's H12_DEPLOY_COMMON_SRCS): the deploy source
  # list lives in exactly 2 places (here + core_ws/src/h12_deploy_mjpc).
  set(H12_DEPLOY_COMMON_SRCS
      deploy/deploy_common.cc
      deploy/deploy_model.cc
      deploy/deploy_net.cc
      deploy/deploy_state.cc
      deploy/deploy_telemetry.cc)
  # target : thin main pairs. NOTE the split core is ALSO built by the consumer
  # package as mjpc_split_core (the production binary); building it here keeps
  # a fork-only checkout able to produce every core.
  set(H12_DEPLOY_NODES
      h12_control_node        deploy/h12_control_node.cc
      h12_lower_body_controller deploy/h12_lower_body_controller.cc
      h12_upper_body_controller deploy/h12_upper_body_controller.cc
      h12_split_controller    deploy/h12_split_controller.cc)
  list(LENGTH H12_DEPLOY_NODES _n)
  math(EXPR _last "${_n} - 1")
  foreach(_i RANGE 0 ${_last} 2)
    list(GET H12_DEPLOY_NODES ${_i} _node)
    math(EXPR _j "${_i} + 1")
    list(GET H12_DEPLOY_NODES ${_j} _main)
    add_executable(${_node} ${_main} ${H12_DEPLOY_COMMON_SRCS})
    target_link_libraries(
      ${_node}
      absl::flags
      absl::flags_parse
      absl::log
      libmjpc
      mujoco::mujoco
      threadpool
      Threads::Threads
      unitree_sdk2
    )
    target_include_directories(${_node} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
    target_compile_options(${_node} PUBLIC ${MJPC_COMPILE_OPTIONS})
    target_link_options(${_node} PRIVATE ${MJPC_LINK_OPTIONS})
    target_compile_definitions(${_node} PRIVATE MJSIMULATE_STATIC)
    set_target_properties(${_node} PROPERTIES
      BUILD_RPATH "${_dds_libdir}"
      INSTALL_RPATH "${_dds_libdir}")
    message(STATUS "MJPC deploy: ${_node} ENABLED (unitree_sdk2 at ${_dds_libdir})")
  endforeach()
endif()

add_subdirectory(tasks)

if(BUILD_TESTING AND MJPC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

if(MJPC_BUILD_GRPC_SERVICE)
  add_subdirectory(grpc)
endif()

include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

# Attach the in-process MJPC gRPC server to every embedded control node so the
# existing monitor can connect (view state + live Strategy switch; the upper
# node's goal-ingest seam). Done HERE, after add_subdirectory(grpc), so
# mjpc_agent_service (grpc_agent_util helpers + generated proto + grpc++) is
# already defined. Targets come from the H12_DEPLOY_NODES pair list above.
# NOTE: the production colcon cores (core_ws/src/h12_deploy_mjpc) never define
# H12_NODE_GRPC -- a fork-built core has the monitor, the container one does not.
if(MJPC_BUILD_DEPLOY AND MJPC_BUILD_GRPC_SERVICE)
  list(LENGTH H12_DEPLOY_NODES _n)
  math(EXPR _last "${_n} - 1")
  foreach(_i RANGE 0 ${_last} 2)
    list(GET H12_DEPLOY_NODES ${_i} _node)
    if(TARGET ${_node})
      target_link_libraries(${_node} mjpc_agent_service)
      target_compile_definitions(${_node} PRIVATE H12_NODE_GRPC)
      message(STATUS "MJPC deploy: ${_node} gRPC monitor server ENABLED")
    endif()
  endforeach()
endif()