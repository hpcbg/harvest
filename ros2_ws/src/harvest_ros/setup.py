import os
from glob import glob

from setuptools import setup

package_name = "harvest_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         [os.path.join("resource", package_name)]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="HPC",
    maintainer_email="office@hpc.bg",
    description="ROS 2 bridge for the HARVEST farm energy-management system",
    license="MIT",
    entry_points={
        "console_scripts": [
            "fleet_bridge = harvest_ros.bridge:main",
            "isaac_bridge = harvest_ros.isaac_bridge:main",
        ],
    },
)
