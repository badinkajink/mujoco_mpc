#!/usr/bin/env bash
# macOS build script for MuJoCo MPC — humanoidbench fork (Apple Silicon + Intel)
#
# Extra dependencies over upstream:
#   - nlohmann/json 3.11.3  (fetched automatically via FetchContent)
#   - h1_mujoco @ ef83c84   (fetched automatically via FetchContent)
#   - Universal Robots UR5e, Unitree H1/H1-2 (from mujoco_menagerie, auto-fetched)
#
# Usage:
#   ./build_macos.sh            # Release build
#   BUILD_TYPE=Debug ./build_macos.sh
#   MJPC_GRPC=ON ./build_macos.sh
#   JOBS=4 ./build_macos.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_TYPE="${BUILD_TYPE:-Release}"
GRPC="${MJPC_GRPC:-OFF}"
JOBS="${JOBS:-$(sysctl -n hw.logicalcpu)}"

# ── prerequisites ──────────────────────────────────────────────────────────────
if ! command -v brew &>/dev/null; then
  echo "error: Homebrew is required. Install from https://brew.sh" >&2
  exit 1
fi

missing=()
for pkg in cmake ninja; do
  command -v "$pkg" &>/dev/null || missing+=("$pkg")
done
# zlib ships with Xcode; brew version is needed for cmake's find_package
brew list zlib &>/dev/null || missing+=("zlib")

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "Installing missing dependencies: ${missing[*]}"
  brew install "${missing[@]}"
fi

# patch is required for XML model patching in tasks/CMakeLists.txt.
# It ships with macOS via Xcode CLT; verify it's present.
if ! command -v patch &>/dev/null; then
  echo "Installing patch via Homebrew..."
  brew install gpatch
fi

# ── cmake ──────────────────────────────────────────────────────────────────────
ARCH="$(uname -m)"   # arm64 or x86_64

echo "==> Configuring (arch=${ARCH}, build=${BUILD_TYPE}, grpc=${GRPC})"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DMJPC_BUILD_GRPC_SERVICE:BOOL="${GRPC}" \
  -DCMAKE_OSX_ARCHITECTURES="${ARCH}" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5

echo "==> Building with ${JOBS} jobs"
cmake --build "${BUILD_DIR}" --config "${BUILD_TYPE}" -- -j"${JOBS}"

echo ""
echo "Build complete. Binaries are in ${BUILD_DIR}/bin/"
echo "Run:  ${BUILD_DIR}/bin/mjpc"
