# Object Detection LiDAR

[![Build](https://github.com/DHBW-Smart-Rollerz/object_detection_lidar/actions/workflows/build.yaml/badge.svg)](https://github.com/DHBW-Smart-Rollerz/object_detection_lidar/actions/workflows/build.yaml) [![Test](https://github.com/DHBW-Smart-Rollerz/object_detection_lidar/actions/workflows/test.yaml/badge.svg)](https://github.com/DHBW-Smart-Rollerz/object_detection_lidar/actions/workflows/test.yaml) [![Lint](https://github.com/DHBW-Smart-Rollerz/object_detection_lidar/actions/workflows/lint.yaml/badge.svg)](https://github.com/DHBW-Smart-Rollerz/object_detection_lidar/actions/workflows/lint.yaml)

ROS 2 package (Python) for LiDAR-based obstacle detection on the Smarty vehicle (RPLIDAR A2M12, input: /scan).

## Usage

Clone the package into `~/smarty_workspace/src`, then build and launch it:

```bash
make build PKG=object_detection_lidar
ros2 launch object_detection_lidar object_detection_lidar.launch.py
```

## Node

`object_detection_lidar_node` subscribes to the 2D laser scan, drops invalid points and points outside `[min_range, max_range]` and publishes the remaining points as point cloud in the frame of the scan. The point cloud can be shown in RViz (display type `PointCloud2`).

| Parameter | Default | Description |
| --- | --- | --- |
| `scan_subscriber` | `/scan` | Input topic (`sensor_msgs/LaserScan`) |
| `points_publisher` | `/object_detection_lidar/points` | Output topic (`sensor_msgs/PointCloud2`) |
| `min_range` | `0.15` | Closer points are ignored, in m |
| `max_range` | `4.0` | Farther points are ignored, in m |
| `debug` | `False` | Logs the number of valid points and the processing times |

## Structure

- `.github/workflows/`: CI pipelines, which call the shared workflows of the [smarty_workspace](https://github.com/DHBW-Smart-Rollerz/smarty_workspace/tree/jazzy/.github/workflows):
  - `build.yaml`: Builds the package
  - `test.yaml`: Runs the tests of the package
  - `lint.yaml`: Checks the code with ruff (Python) and ament_uncrustify (C++)
- `config/`: All configurations (most of the time yaml files)
- `launch/`: Contains all launch files. Launch files can start multiple nodes with yaml-configurations
- `resource/`: Contains the package name (required to build with colcon)
- `object_detection_lidar`: Contains all nodes and sources for the ros package
- `test/`: Contains all tests
- `package.xml`: Contains metadata about the package
- `setup.py`: Used for Python package configuration
- `setup.cfg`: Additional configuration for the package
- `requirements.txt`: Python dependencies

## Contributing

Thank you for considering contributing to this repository! Here are a few guidelines to get you started:

1. Fork the repository and clone it locally.
2. Create a new branch for your contribution.
3. Make your changes and ensure they are properly tested.
4. Commit your changes and push them to your forked repository.
5. Submit a pull request with a clear description of your changes.

We appreciate your contributions and look forward to reviewing them!

## License

This repository is licensed under the MIT license. See [LICENSE](LICENSE) for details.
