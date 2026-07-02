from setuptools import find_packages, setup

package_name = "reach_target_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Allen Devaraj",
    maintainer_email="allendevaraj33333@gmail.com",
    description="HAMS perception -> MJPC-MPC live reach target bridge (gRPC).",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "bridge = reach_target_bridge.bridge_node:main",
        ],
    },
)
