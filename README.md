<h1>
  <a href="#"><img alt="MuJoCo MPC" src="docs/assets/banner.png" width="100%"></a>
</h1>

<p>
  <a href="https://github.com/google-deepmind/mujoco_mpc/actions/workflows/build.yml?query=branch%3Amain" alt="GitHub Actions">
    <img src="https://img.shields.io/github/actions/workflow/status/google-deepmind/mujoco_mpc/build.yml?branch=main">
  </a>
  <a href="https://github.com/google-deepmind/mujoco_mpc/blob/main/LICENSE" alt="License">
    <img src="https://img.shields.io/github/license/google-deepmind/mujoco_mpc">
  </a>
</p>

**MuJoCo MPC (MJPC)** is an interactive application and software framework for
real-time predictive control with [MuJoCo](https://mujoco.org/), developed by
Google DeepMind.

MJPC allows the user to easily author and solve complex robotics tasks, and
currently supports multiple shooting-based planners. Derivative-based methods include iLQG and
Gradient Descent, while derivative-free methods include a simple yet very competitive planner
called Predictive Sampling.

- [Python API](#python-api)
  - [Installation](#installation-1)
    - [Prerequisites](#prerequisites-1)
    - [Install API](#install-api)
    - [Python API Installation Issues](#python-api-installation-issues)
  - [Contributing](#contributing)
  - [Known Issues](#known-issues)
  - [Citation](#citation)
  - [Acknowledgments](#acknowledgments)
  - [License and Disclaimer](#license-and-disclaimer)

## Overview

To read the paper describing this software package, please see our
[preprint](https://arxiv.org/abs/2212.00541).

## Authoring Predictive Control Tasks

See the [Predictive Control](docs/OVERVIEW.md) documentation for more
information.

## Graphical User Interface

For a detailed dive of the graphical user interface, see the
[MJPC GUI](docs/GUI.md) documentation.

## Installation

This branch (`macos-humanoidbench`) is tested on **Apple Silicon (M1–M5) and Intel** Macs running macOS 13 Ventura through macOS 15 Sequoia and the macOS 26 (Tahoe) beta, and on Ubuntu 22.04.

### Prerequisites

#### macOS (Apple Silicon or Intel)

Install Xcode Command Line Tools if not already present:
```sh
xcode-select --install
```

Install build dependencies via [Homebrew](https://brew.sh):
```sh
brew install cmake ninja
```

> `zlib` ships with Xcode CLT, but Homebrew's copy is needed for CMake's `find_package`. The build script installs it automatically if missing.

#### Ubuntu 22.04
```sh
sudo apt-get update && sudo apt-get install \
  cmake libgl1-mesa-dev libxinerama-dev libxcursor-dev \
  libxrandr-dev libxi-dev ninja-build zlib1g-dev \
  clang-13 libc++-13-dev libc++abi-13-dev
```

### Clone

```sh
git clone https://github.com/badinkajink/mujoco_mpc
cd mujoco_mpc
git checkout macos-humanoidbench
```

> The first configure step fetches several dependencies via CMake FetchContent (MuJoCo, abseil, GLFW, nlohmann/json, h1\_mujoco, mujoco\_menagerie, dm\_control). Expect 5–15 minutes on the first run.

### Build — Option A: build script (macOS, recommended)

```sh
./build_macos.sh
```

Produces `./build/bin/mjpc`. Optional environment overrides:

| Variable | Default | Example |
|---|---|---|
| `BUILD_TYPE` | `Release` | `BUILD_TYPE=Debug ./build_macos.sh` |
| `JOBS` | logical CPU count | `JOBS=4 ./build_macos.sh` |

### Build — Option B: manual cmake

**macOS**
```sh
cmake -S . -B build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_OSX_ARCHITECTURES="$(uname -m)" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5

cmake --build build --config Release -j$(sysctl -n hw.logicalcpu)
```

**Ubuntu 22.04**
```sh
cmake -S . -B build \
  -G Ninja \
  -DCMAKE_C_COMPILER=clang-13 \
  -DCMAKE_CXX_COMPILER=clang++-13 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release
```

### Run

```sh
./build/bin/mjpc
```

### Run Avoid tasks

By default the Avoid H12 task runs with both capacitance and ToF sensing. To toggle a sensing modality, set its task weight to 0 in the GUI. Run the simulation at **5% speed**, use the **iLQR planner**, and reset the Agent planner each time the obstacle resets. To generate new random obstacles, open `mjpc/tasks/humanoid_bench/avoid/Avoid_H12_ToF_Cap.xml` and set `use_offline_obstacles` to `0`.

### Headless testing

```sh
cd build/mjpc/test
ctest -C Release --output-on-failure -j$(sysctl -n hw.logicalcpu)
```

All 88 tests should pass.

### Build with VSCode (macOS)

Install the [CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools) and [C/C++](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools) extensions.

1. Open the cloned `mujoco_mpc` folder in VSCode.
2. **Cmd+Shift+P → "CMake: Select Configure Preset"** → pick **`macOS Release (Apple Clang, arm64)`**.
3. **Cmd+Shift+P → "CMake: Configure"** (runs once; downloads all dependencies).
4. **Cmd+Shift+P → "CMake: Build"**.
5. **Cmd+Shift+P → "CMake: Run Without Debugging"** → select **`mjpc`**.

No kit selection is needed — the preset supplies the compiler and all required flags automatically.

### Build issues

If you see `Failed to get the hash for HEAD` after a dependency version change, delete the stale FetchContent cache entry and reconfigure:

```sh
rm -rf ~/.cmake_fc_cache/abseil-cpp-*
cmake -S . -B build ...
```

For other issues, see the [GitHub Actions configuration](https://github.com/google-deepmind/mujoco_mpc/blob/main/.github/workflows/build.yml) for the upstream CI setup.

# Python API
We provide a simple Python API for MJPC. This API is still experimental and expects some more experience from its users. For example, the correct usage requires that the model (defined in Python) and the MJPC task (i.e., the residual and transition functions defined in C++) are compatible with each other. Currently, the Python API does not provide any particular error handling for verifying this compatibility and may be difficult to debug without more in-depth knowledge about MuJoCo and MJPC.

## Installation

### Prerequisites
1. Build MJPC (see instructions above).

2. Python 3.10

3. (Optionally) Create a conda environment with **Python 3.10**:
```sh
conda create -n mjpc python=3.10
conda activate mjpc
```

4. Install MuJoCo
```sh
pip install mujoco
```

### Install API
Next, change to the python directory:
```sh
cd python
```

Install the Python module:
```sh
python setup.py install
```

Test that installation was successful:
```sh
python "mujoco_mpc/agent_test.py"
```

Example scripts are found in `python/mujoco_mpc/demos`. For example from `python/`:
```sh
python mujoco_mpc/demos/agent/cartpole_gui.py
```
will run the MJPC GUI application using MuJoCo's passive viewer via Python.

### Python API Installation Issues
If your installation fails or is terminated prematurely, we recommend deleting the MJPC build directory and starting from scratch as the build will likely be corrupted. Additionally, delete the files generated during the installation process from the `python/` directory.

## Contributing

See the [Contributing](docs/CONTRIBUTING.md) documentation for more information.

## Known Issues

MJPC is not production-quality software, it is a **research prototype**. There
are likely to be missing features and outright bugs. If you find any, please
report them in the [issue tracker](https://github.com/google-deepmind/mujoco_mpc/issues).
Below we list some known issues, including items that we are actively working
on.

- We have not tested MJPC on Windows, but there should be no issues in
  principle.
- Task specification, in particular the setting of norms and their parameters in
  XML, is a bit clunky. We are still iterating on the design.
- The Gradient Descent search step is proportional to the scale of the cost
  function and requires per-task tuning in order to work well. This is not a bug
  but a property of vanilla gradient descent. It might be possible to ameliorate
  this with some sort of gradient normalisation, but we have not investigated
  this thoroughly.

## Citation

If you use MJPC in your work, please cite our accompanying [preprint](https://arxiv.org/abs/2212.00541):

```bibtex
@article{howell2022,
  title={{Predictive Sampling: Real-time Behaviour Synthesis with MuJoCo}},
  author={Howell, Taylor and Gileadi, Nimrod and Tunyasuvunakool, Saran and Zakka, Kevin and Erez, Tom and Tassa, Yuval},
  archivePrefix={arXiv},
  eprint={2212.00541},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2212.00541},
  doi={10.48550/arXiv.2212.00541},
  year={2022},
  month={dec}
}
```

## Acknowledgments

The main effort required to make this repository publicly available was
undertaken by [Taylor Howell](https://thowell.github.io/) and the Google
DeepMind Robotics Simulation team.

## License and Disclaimer

All other content is Copyright 2022 DeepMind Technologies Limited and licensed
under the Apache License, Version 2.0. A copy of this license is provided in the
top-level LICENSE file in this repository. You can also obtain it from
https://www.apache.org/licenses/LICENSE-2.0.

This is not an officially supported Google product.
