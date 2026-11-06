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

`object_detection_lidar_node` subscribes to the 2D laser scan and processes every scan in four steps:

1. Drop invalid points and points outside `[min_range, max_range]`, publish the rest as point cloud in the frame of the scan
2. Cluster the points with DBSCAN (`clustering.py`)
3. Drop objects with a diameter above `max_object_diameter` (walls)
4. Classify every remaining object as box or other by the shape of its visible contour, one straight side or a right-angled corner (`box_classification.py`)

All topics can be shown in RViz (`PointCloud2`, `MarkerArray`).

| Parameter | Default | Description |
| --- | --- | --- |
| `scan_subscriber` | `/scan` | Input topic (`sensor_msgs/LaserScan`) |
| `points_publisher` | `/object_detection_lidar/points` | Valid scan points (`sensor_msgs/PointCloud2`) |
| `markers_publisher` | `/object_detection_lidar/markers` | One cube per object, boxes red (namespace `boxes`), others blue (`objects`) (`visualization_msgs/MarkerArray`) |
| `objects_publisher` | `/object_detection_lidar/objects` | One point per object (centroid) with the fields `extent_x`, `extent_y`, `diameter` and `is_box` (1.0/0.0) (`sensor_msgs/PointCloud2`) |
| `min_range` | `0.15` | Closer points are ignored, in m |
| `max_range` | `4.0` | Farther points are ignored, in m |
| `eps` | `0.10` | Max. distance of neighboring points (DBSCAN), in m |
| `min_samples` | `3` | Min. number of points of a core point (DBSCAN) |
| `max_object_diameter` | `0.6` | Larger objects are dropped (walls), in m |
| `box_line_tolerance` | `0.0023` | Max. RMS distance of the points to a box side at 0 m, in m |
| `box_line_tolerance_per_m` | `0.0015` | Increase of the line tolerance per m of distance, in m/m |
| `box_angle_tolerance` | `10.0` | Max. deviation of a box corner from 90°, in deg |
| `box_corner_tolerance_factor` | `2.5` | Factor on the line tolerance for the sides of a corner |
| `box_min_size` | `0.08` | Min. distance between the outermost points of a box, in m |
| `box_min_fill_ratio` | `0.7` | Min. ratio of points to beams within the angular span of a box |
| `debug` | `False` | Logs the objects with their class and the processing times |

The values are explained in `config/ros_params.yaml`. To play the rosbags of 03/11/2026, remap the scan topic: `ros2 bag play <bag> --topics /lidar/scan --remap /lidar/scan:=/scan`.

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
