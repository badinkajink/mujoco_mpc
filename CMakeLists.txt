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

cmake_minimum_required(VERSION 3.16)

# INTERPROCEDURAL_OPTIMIZATION is enforced when enabled.
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
# Default to GLVND if available.
set(CMAKE_POLICY_DEFAULT_CMP0072 NEW)

# This line has to appear before 'PROJECT' in order to be able to disable incremental linking
set(MSVC_INCREMENTAL_DEFAULT ON)

project(
  mujoco_mpc
  VERSION 0.1.0
  DESCRIPTION "MuJoCo predictive control"
  HOMEPAGE_URL "https://github.com/google-deepmind/mujoco_mpc"
)

enable_language(C)
enable_language(CXX)
if(APPLE)
  enable_language(OBJC)
  enable_language(OBJCXX)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(MpcOptions)
include(MujocoMacOS)

option(MJPC_BUILD_TESTS "Build tests for MJPC" ON)
option(MJPC_GRPC_BUILD_TESTS "Build tests for gRPC" ON)
option(MJPC_BUILD_GRPC_SERVICE "Build MJPC gRPC service." OFF)
option(PYMJPC_BUILD_TESTS "Build tests for Python bindings" ON)

include(FindOrFetch)

find_package(Python)
find_package(Threads REQUIRED)

set(MUJOCO_BUILD_EXAMPLES OFF)
set(MUJOCO_BUILD_SIMULATE ON)
set(MUJOCO_BUILD_TESTS OFF)
set(MUJOCO_TEST_PYTHON_UTIL OFF)

set(MUJOCO_MPC_MUJOCO_GIT_TAG
    088079eff0450e32b98ee743141780ed68307506
    CACHE STRING "Git revision for MuJoCo."
)

set(MUJOCO_MPC_MENAGERIE_GIT_TAG
    d82891f8b6c576f9f6a9df308f3a1155ebedbc40
    CACHE STRING "Git revision for MuJoCo Menagerie."
)

set(MUJOCO_MPC_DM_CONTROL_GIT_TAG
    774f46182140106e22725914aad3c6299ed91edd
    CACHE STRING "Git revision for dm_control."
)

# CL_Assets is the SOURCE OF TRUTH for the H1-2 robot (2026-08-03). It replaces
# correlllab/h1_mujoco@ef83c84, whose arm joint ranges predate the arm service and
# disagree with h12_safety_layer/core/joint_limits.py -- elbow was offset ~pi/2 and
# shoulder_yaw was left/right swapped. CL_Assets' h1_2_handless.xml matches that
# table exactly on all 27 joints (range AND actuatorfrcrange), and its meshes are
# byte-identical to h1_mujoco's (90/90), so this swap moves limits only.
set(MUJOCO_MPC_CL_ASSETS_GIT_TAG
    be4db826a3f76ce67793088b0d53297f23d3fa4f
    CACHE STRING "Git revision for CL_Assets (H1-2 robot source of truth)."
)

findorfetch(
  USE_SYSTEM_PACKAGE
  OFF
  PACKAGE_NAME
  mujoco
  LIBRARY_NAME
  mujoco
  GIT_REPO
  https://github.com/google-deepmind/mujoco.git
  GIT_TAG
  ${MUJOCO_MPC_MUJOCO_GIT_TAG}
  TARGETS
  mujoco::mujoco
  mujoco::platform_ui_adapter
  EXCLUDE_FROM_ALL
)

# TODO(nimrod): Update to the latest version of abseil, or use the one defined
# by MuJoCo, once grpc fix their build issues.
set(MUJOCO_DEP_VERSION_abseil
    fb3621f4f897824c0dbe0615fa94543df6192f30 # LTS 20230802.1
    CACHE STRING "Version of `abseil` to be fetched."
    FORCE
)

set(BUILD_SHARED_LIBS_OLD ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS
    OFF
    CACHE INTERNAL "Build SHARED libraries"
)

findorfetch(
  USE_SYSTEM_PACKAGE
  OFF
  PACKAGE_NAME
  GTest
  LIBRARY_NAME
  googletest
  GIT_REPO
  https://github.com/google/googletest.git
  GIT_TAG
  ${MUJOCO_DEP_VERSION_gtest}
  TARGETS
  gtest
  gmock
  gtest_main
  EXCLUDE_FROM_ALL
)

set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_BUILD_TESTING OFF)
# ABSL_ENABLE_INSTALL is needed for
# https://github.com/protocolbuffers/protobuf/issues/12185#issuecomment-1594685860
set(ABSL_ENABLE_INSTALL ON)
findorfetch(
  USE_SYSTEM_PACKAGE
  OFF
  PACKAGE_NAME
  absl
  LIBRARY_NAME
  abseil-cpp
  GIT_REPO
  https://github.com/abseil/abseil-cpp.git
  GIT_TAG
  ${MUJOCO_DEP_VERSION_abseil}
  TARGETS
  absl::core_headers
  EXCLUDE_FROM_ALL
)

set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_DOCS OFF)
set(GLFW_INSTALL OFF)

findorfetch(
  USE_SYSTEM_PACKAGE
  MJPC_USE_SYSTEM_GLFW
  PACKAGE_NAME
  glfw
  LIBRARY_NAME
  glfw
  GIT_REPO
  https://github.com/glfw/glfw.git
  GIT_TAG
  ${MUJOCO_DEP_VERSION_glfw3}
  TARGETS
  glfw
  EXCLUDE_FROM_ALL
)

set(BUILD_SHARED_LIBS
    ${BUILD_SHARED_LIBS_OLD}
    CACHE BOOL "Build SHARED libraries" FORCE
)
unset(BUILD_SHARED_LIBS_OLD)

# h1_mujoco is RETAINED, but demoted: it no longer supplies the H1-2 robot (see the
# cl_assets declaration below). It still supplies assets CL_Assets does not carry at all
# -- inspire/, magpie/ur5_magpie.xml, and unitree_robots/h1_2/{h1_2_torso_skin,h12_tof}.xml
# -- which back the ur5 / inspire / skin / ToF tasks. Repointing this wholesale would
# have broken all four, so the two coexist with a clean split of responsibility.
FetchContent_Declare(
  h12
  GIT_REPOSITORY https://github.com/correlllab/h1_mujoco.git
  GIT_TAG ef83c84 # <-- specify main instead of default master
)
FetchContent_GetProperties(h12)
if(NOT h12_POPULATED)
  FetchContent_Populate(h12)
endif()

# NOTE on clone size: CL_Assets carries ~59 MB of meshes plus a large isaac_assets/ USD
# tree, and we consume only mujoco_assets/h1_2_handless.xml + meshes/h1_2. GIT_SHALLOW is
# NOT set because CMake's shallow path clones with `--branch <GIT_TAG>`, which cannot take
# a raw commit SHA -- and the pin is worth more than the clone time. If container build
# time becomes a problem, the fix is a sparse checkout, not un-pinning.
FetchContent_Declare(
  cl_assets
  GIT_REPOSITORY https://github.com/correlllab/CL_Assets.git
  GIT_TAG ${MUJOCO_MPC_CL_ASSETS_GIT_TAG}
)
FetchContent_GetProperties(cl_assets)
if(NOT cl_assets_POPULATED)
  FetchContent_Populate(cl_assets)
endif()

FetchContent_Declare(
  menagerie
  GIT_REPOSITORY https://github.com/google-deepmind/mujoco_menagerie.git
  GIT_TAG ${MUJOCO_MPC_MENAGERIE_GIT_TAG}
)

FetchContent_GetProperties(menagerie)
if(NOT menagerie_POPULATED)
  FetchContent_Populate(menagerie)
endif()

FetchContent_Declare(
  dm_control
  GIT_REPOSITORY https://github.com/google-deepmind/dm_control.git
  GIT_TAG ${MUJOCO_MPC_DM_CONTROL_GIT_TAG}
)

FetchContent_GetProperties(dm_control)
if(NOT dm_control_POPULATED)
  FetchContent_Populate(dm_control)
endif()

if(NOT TARGET lodepng)
  FetchContent_Declare(
    lodepng
    GIT_REPOSITORY https://github.com/lvandeve/lodepng.git
    GIT_TAG ${MJPC_DEP_VERSION_lodepng}
  )

  FetchContent_GetProperties(lodepng)
  if(NOT lodepng_POPULATED)
    FetchContent_Populate(lodepng)
    # This is not a CMake project.
    set(LODEPNG_SRCS ${lodepng_SOURCE_DIR}/lodepng.cpp)
    set(LODEPNG_HEADERS ${lodepng_SOURCE_DIR}/lodepng.h)
    add_library(lodepng STATIC ${LODEPNG_HEADERS} ${LODEPNG_SRCS})
    target_compile_options(lodepng PRIVATE ${MUJOCO_MACOS_COMPILE_OPTIONS})
    target_link_options(lodepng PRIVATE ${MUJOCO_MACOS_LINK_OPTIONS})
    target_include_directories(lodepng PUBLIC ${lodepng_SOURCE_DIR})
  endif()
endif()

set(MJPC_COMPILE_OPTIONS "${AVX_COMPILE_OPTIONS}" "${EXTRA_COMPILE_OPTIONS}")
set(MJPC_LINK_OPTIONS "${EXTRA_LINK_OPTIONS}")

add_definitions(-DSOURCE_DIR=\"${CMAKE_SOURCE_DIR}\")
add_subdirectory(mjpc)
