from setup_utils import include_directory
from setuptools import find_packages, setup

package_name = "object_detection_lidar"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        *include_directory(
            install_path=f"share/{package_name}/config",
            source_path="config",
        ),
        *include_directory(
            install_path=f"share/{package_name}/launch",
            source_path="launch",
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Smart Rollerz",
    maintainer_email="info@dhbw-smartrollerz.org",
    description="LiDAR-based obstacle detection for the Smarty vehicle",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "object_detection_lidar_node = "
            f"{package_name}.object_detection_lidar_node:main",
        ],
    },
)
