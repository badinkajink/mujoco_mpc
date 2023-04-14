# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Install script for MuJoCo MPC."""

import os
import pathlib
import platform
import setuptools
from setuptools.command import build_py
from setuptools.command import build_ext
import shutil
import subprocess


Path = pathlib.Path


class GenerateProtoGrpcCommand(setuptools.Command):
  """Specialized setup command to handle agent proto compilation.

  Generates the `agent_pb2{_grpc}.py` files from `agent_proto`. Assumes that
  `grpc_tools.protoc` is installed.
  """

  description = "Generate `.proto` files to Python protobuf and gRPC files."
  user_options = []

  def initialize_options(self):
    self.build_lib = None

  def finalize_options(self):
    self.set_undefined_options("build_py", ("build_lib", "build_lib"))

  def run(self):
    """Generate `agent.proto` into `agent_pb2{_grpc}.py`.

    This function looks more complicated than what it has to be because the
    `protoc` generator is very particular in the way it generates the imports
    for the generated `agent_pb2_grpc.py` file. The final argument of the
    `protoc` call has to be "mujoco_mpc/agent.proto" in order for the import to
    become `from mujoco_mpc import [agent_pb2_proto_import]` instead of just
    `import [agent_pb2_proto_import]`. The latter would fail because the name is
    meant to be relative but python3 interprets it as an absolute import.
    """
    agent_proto_filename = "agent.proto"
    agent_proto_source_path = Path("..", "grpc", agent_proto_filename).resolve()
    assert self.build_lib is not None
    build_lib_path = Path(self.build_lib).resolve()
    proto_module_relative_path = Path(
      "mujoco_mpc", "proto", agent_proto_filename)
    agent_proto_destination_path = Path(
        build_lib_path, proto_module_relative_path
    )
    agent_proto_destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Copy `agent_proto_filename` into current source.
    shutil.copy(agent_proto_source_path, agent_proto_destination_path)

    protoc_command_parts = [
        f"-I{build_lib_path}",
        f"--python_out={build_lib_path}",
        f"--grpc_python_out={build_lib_path}",
        agent_proto_destination_path
    ]
    # Instead of `self.spawn`, this should be runnable directly as
    # `grpc_tools.protoc.main(protoc_command_parts)`, but that seems to fail
    # on MacOS for some reason, most likely because of the lack of explicit
    # `protoc.py` included by the script-version of `protoc`.
    self.spawn([
      "python", "-m", "grpc_tools.protoc", *protoc_command_parts
    ])
    self.spawn([
      "touch", str(agent_proto_destination_path.parent / "__init__.py")
    ])


class CopyAgentServiceBinaryCommand(setuptools.Command):
  """Specialized setup command to copy `agent_service` next to `agent.py`.

  Assumes that the C++ gRPC `agent_service` binary has been manually built and
  and located in the default `mujoco_mpc/build/bin` folder.
  """

  description = "Copy `agent_service` next to `agent.py`."
  user_options = []

  def initialize_options(self):
    self.build_lib = None

  def finalize_options(self):
    self.set_undefined_options("build_py", ("build_lib", "build_lib"))

  def run(self):
    source_path = Path("../build/bin/agent_service")
    if not source_path.exists():
      raise ValueError(
          f"Cannot find `agent_service` binary from {source_path}. Please build"
          " the `agent_service` C++ gRPC service."
      )
    assert self.build_lib is not None
    build_lib_path = Path(self.build_lib).resolve()
    destination_path = Path(
      build_lib_path, "mujoco_mpc", "mjpc", "agent_service"
    )

    self.announce(f"{source_path.resolve()=}")
    self.announce(f"{destination_path.resolve()=}")

    destination_path.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy(source_path, destination_path)


class CopyTaskAssetsCommand(setuptools.Command):
  """Specialized setup command to copy `agent_service` next to `agent.py`.

  Assumes that the C++ gRPC `agent_service` binary has been manually built and
  and located in the default `mujoco_mpc/build/bin` folder.
  """

  description = (
      "Copy task assets over to python source to make them accessible by"
      " `Agent`."
  )
  user_options = []

  def initialize_options(self):
    self.build_lib = None

  def finalize_options(self):
    self.set_undefined_options("build_ext", ("build_lib", "build_lib"))

  def run(self):
    mjpc_tasks_path = Path(__file__).parent.parent / "mjpc" / "tasks"
    source_paths = tuple(mjpc_tasks_path.rglob("*.xml"))
    relative_source_paths = tuple(
        p.relative_to(mjpc_tasks_path) for p in source_paths
    )
    assert self.build_lib is not None
    build_lib_path = Path(self.build_lib).resolve()
    destination_dir_path = Path(build_lib_path, "mujoco_mpc", "mjpc", "tasks")
    self.announce(
        f"Copying assets {relative_source_paths} from"
        f" {mjpc_tasks_path} over to {destination_dir_path}."
    )

    for source_path, relative_source_path in zip(
        source_paths, relative_source_paths
    ):
      destination_path = destination_dir_path / relative_source_path
      destination_path.parent.mkdir(exist_ok=True, parents=True)
      shutil.copy(source_path, destination_path)


class BuildPyCommand(build_py.build_py):
  """Specialized Python builder to handle agent service dependencies.

  During build, this will generate the `agent_pb2{_grpc}.py` files and copy
  `agent_service` binary next to `agent.py`.
  """

  user_options = build_py.build_py.user_options

  def run(self):
    self.run_command("generate_proto_grpc")
    self.run_command("copy_task_assets")
    super().run()


class CMakeExtension(setuptools.Extension):
  """A Python extension that has been prebuilt by CMake.

  We do not want distutils to handle the build process for our extensions, so
  so we pass an empty list to the super constructor.
  """

  def __init__(self, name):
    super().__init__(name, sources=[])


class BuildCMakeExtension(build_ext.build_ext):
  """Uses CMake to build extensions."""

  def run(self):
    self._configure_and_build_agent_service()
    self.run_command("copy_agent_service_binary")

  def _configure_and_build_agent_service(self):
    """Check for CMake."""
    cmake_command = "cmake"
    build_cfg = "Debug"
    mujoco_mpc_root = Path(__file__).parent.parent
    mujoco_mpc_build_dir = mujoco_mpc_root / "build"
    cmake_configure_args = [
      "-DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE",
      f"-DCMAKE_BUILD_TYPE:STRING={build_cfg}",
      "-DBUILD_TESTING:BOOL=OFF"
    ]

    if platform.system() == "Darwin" and "ARCHFLAGS" in os.environ:
      osx_archs = []
      if "-arch x86_64" in os.environ["ARCHFLAGS"]:
        osx_archs.append("x86_64")
      if "-arch arm64" in os.environ["ARCHFLAGS"]:
        osx_archs.append("arm64")
      cmake_configure_args.append(
        f"-DCMAKE_OSX_ARCHITECTURES={';'.join(osx_archs)}")

    # TODO(hartikainen): We currently configure the builds into
    # `mujoco_mpc/build`. This should use `self.build_{temp,lib}` instead, to
    # isolate the Python builds from the C++ builds.
    print("Configuring CMake with the following arguments:")
    for arg in cmake_configure_args:
      print(f"  {arg}")
    subprocess.check_call([
      cmake_command,
      *cmake_configure_args,
      f"-S{mujoco_mpc_root.resolve()}",
      f"-B{mujoco_mpc_build_dir.resolve()}",
    ], cwd=mujoco_mpc_root)

    print("Building `agent_service` with CMake")
    subprocess.check_call([
      cmake_command,
      "--build",
      mujoco_mpc_build_dir.resolve(),
      "--target",
      "agent_service",
      f"-j{os.cpu_count()}",
      "--config",
      build_cfg,
    ], cwd=mujoco_mpc_root)


setuptools.setup(
    name="mujoco_mpc",
    version="0.1.0",
    author="DeepMind",
    author_email="mujoco@deepmind.com",
    description="MuJoCo MPC (MJPC)",
    url="https://github.com/deepmind/mujoco_mpc",
    license="MIT",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "grpcio-tools >= 1.53.0",
        "grpcio >= 1.53.0",
    ],
    extras_require={
        "test": [
            "absl-py",
            "mujoco >= 2.3.3",
        ],
    },
    ext_modules=[CMakeExtension("agent_service")],
    cmdclass={
        "build_py": BuildPyCommand,
        "build_ext": BuildCMakeExtension,
        "generate_proto_grpc": GenerateProtoGrpcCommand,
        "copy_agent_service_binary": CopyAgentServiceBinaryCommand,
        "copy_task_assets": CopyTaskAssetsCommand,
    },
    package_data={
        "": ["mjpc/agent_service", "mjpc/tasks/**/*.xml"],
    },
)
