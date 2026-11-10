# Object Detection LiDAR

[![Build](https://github.com/DHBW-Smart-Rollerz/object_detection_lidar/actions/workflows/build.yaml/badge.svg)](https://github.com/DHBW-Smart-Rollerz/object_detection_lidar/actions/workflows/build.yaml) [![Test](https://github.com/DHBW-Smart-Rollerz/object_detection_lidar/actions/workflows/test.yaml/badge.svg)](https://github.com/DHBW-Smart-Rollerz/object_detection_lidar/actions/workflows/test.yaml) [![Lint](https://github.com/DHBW-Smart-Rollerz/object_detection_lidar/actions/workflows/lint.yaml/badge.svg)](https://github.com/DHBW-Smart-Rollerz/object_detection_lidar/actions/workflows/lint.yaml)

ROS 2 package (Python) for obstacle detection from a 360° 2D LiDAR scan (`sensor_msgs/LaserScan`). It segments the scan into objects and publishes their position and visible width. The LiDAR does not classify obstacles (car, pedestrian): this is left to the camera and the sensor fusion, which reads the LiDAR objects in the same format as the camera objects.

## Usage

Clone the package into the `src` folder of the [smarty_workspace](https://github.com/DHBW-Smart-Rollerz/smarty_workspace), then build and launch it:

```bash
make build PKG=object_detection_lidar
ros2 launch object_detection_lidar object_detection_lidar.launch.py
```

The launch file accepts `debug` (default `False`) and `params_file` (default `config/ros_params.yaml` of the package).

## Pipeline

Every scan is processed by `detect_objects` (`pipeline.py`, independent of ROS):

1. **Points**: drop invalid points and points outside of `[min_range, max_range]` (`scan_processing.py`)
2. **Segmentation** (`segmentation: abd`, default): adaptive breakpoint detection on the scan ordered by beam; split between consecutive points at range jumps or gaps of foreign beams, beams without a return count as gap (`breakpoint_detection.py`)
3. **Optional DBSCAN** (`segmentation: dbscan_abd`): DBSCAN on the points (`clustering.py`), then the clusters are split with the same breakpoint criteria
4. **Object extraction**: centroid, extents, diameter and the two outermost visible points (left/right endpoint) of each object
5. **Wall filter**: drop objects with a diameter above `max_object_diameter`
6. **Region of interest** (`roi_enabled`, on by default): keep only objects whose centroid lies in a rectangle in the vehicle frame around the track and its near surroundings ahead of and next to the vehicle, objects away from the track are dropped
7. **Shape**: classify objects as box or other by their visible contour (`box_classification.py`); optionally drop non-box objects with a round contour, e.g. legs or bottles (`drop_round_objects`, `shape.py`)

## Topics

| Direction | Topic (parameter) | Type | Content |
| --- | --- | --- | --- |
| Sub | `/scan` (`scan_subscriber`) | `sensor_msgs/LaserScan` | Input scan |
| Pub | `/object_detection_lidar/points` (`points_publisher`) | `sensor_msgs/PointCloud2` | Valid scan points, frame of the scan |
| Pub | `/object_detection_lidar/markers` (`markers_publisher`) | `visualization_msgs/MarkerArray` | One cube per object, frame of the scan: boxes red (namespace `boxes`), others blue (`objects`), dropped round objects gray (`round`) |
| Pub | `/object_detection_lidar/objects` (`objects_publisher`) | `sensor_msgs/PointCloud2` | One point per object (centroid) with the fields `extent_x`, `extent_y`, `diameter`, `is_box`, `left_x`, `left_y`, `right_x`, `right_y`, `width`, in m, frame `vehicle` |
| Pub | `/object_detection_lidar/object` (`object_publisher`) | `std_msgs/Float32MultiArray` | Objects in the camera format, see below |

**Vehicle frame**: x forward, y left, origin at the center of the front axle on the ground (same frame as the camera objects). Scan points are transformed with `p_vehicle = R(laser_yaw) · p_laser + (lidar_offset_x, lidar_offset_y)`. No tf is published.

**Camera format** (`/object_detection_lidar/object`, same layout as `/object_detection/object`): flat array without layout, 6 values per object `[cls, BLx, BLy, BRx, BRy, score]`, empty if there is no object. Positions in mm in the vehicle frame.

| Value | LiDAR |
| --- | --- |
| `cls` | `-1` (unknown) |
| `BLx`, `BLy` | Left endpoint (largest bearing as seen from the LiDAR) |
| `BRx`, `BRy` | Right endpoint (smallest bearing) |
| `score` | `1.0` |

`width` is the distance between the endpoints, i.e. the visible width of the object, not its real size.

## Parameters

All parameters with their defaults and explanations are in [`config/ros_params.yaml`](config/ros_params.yaml). The detection defaults are defined in `PipelineParams` (`pipeline.py`); a test checks that they match the yaml file. The most important ones:

| Parameter | Default | Description |
| --- | --- | --- |
| `min_range`, `max_range` | `0.15`, `4.0` | Valid range of the scan points, in m |
| `segmentation` | `abd` | `abd`: breakpoint segmentation of the ordered scan; `dbscan_abd`: DBSCAN, then breakpoint split |
| `eps`, `min_samples` | `0.10`, `3` | DBSCAN neighborhood radius (m, `dbscan_abd` only) and min. points of an object |
| `breakpoint_enabled` | `true` | `dbscan_abd` only: `false` keeps the DBSCAN clusters unsplit |
| `breakpoint_lambda_deg`, `breakpoint_sigma_r` | `10.0`, `0.05` | ABD parameters: smallest angle of incidence (deg), range error (m) |
| `breakpoint_max_beam_gap` | `2` | Max. number of foreign beams between two consecutive points of an object |
| `breakpoint_count_invalid` | `true` | Beams without a return count as foreign beams |
| `max_object_diameter` | `0.6` | Wall filter, in m |
| `box_*` | | Tolerances of the box classification |
| `laser_yaw`, `lidar_offset_x`, `lidar_offset_y` | `90.0`, `0.0`, `0.0` | Mounting of the LiDAR in the vehicle frame (deg, m) |
| `roi_enabled`, `roi_x_min` … `roi_y_max` | `true`, `-0.5` … `3.0` | Region of interest (rectangle in the vehicle frame, m) |
| `drop_round_objects`, `round_*` | `true` | Filter for round objects and its thresholds |
| `debug` | `False` | Log the objects and the runtime of each step |

## Tests

```bash
# in the workspace
make test  # ruff, pyright and the tests of all packages
colcon test --packages-select object_detection_lidar --pytest-args -m "not flake8 and not pep257"
colcon test-result --verbose
```

## Evaluation

`tools/evaluate.py` runs every scan of one or more rosbags offline through `detect_objects` (no ROS node, no playback) and prints metrics per bag: runtime per step, objects per scan, class flicker, scatter with a standing vehicle and, if hand labels exist, object recall and box precision/recall. It uses the package next to it (working tree) and `config/ros_params.yaml`. ROS has to be sourced (`rosbag2_py`).

```bash
python3 tools/evaluate.py <bag> [<bag> ...] [--csv metrics.csv]
```

| Option | Default | Description |
| --- | --- | --- |
| `--topic` | `/lidar/scan` | `LaserScan` topic in the bag |
| `--params` | `config/ros_params.yaml` | Parameter file |
| `--labels-dir` | `tools/labels` | Hand labels, one `<bag directory name>.yaml` per bag |
| `--match-radius` | `0.10` | Max. distance between a labelled box and a detected centroid, in m |
| `--flicker-radius` | `0.05` | Max. centroid shift between two scans to count as the same object, in m |
| `--static START:END` | from the labels | Scan range with standing vehicle, repeatable |
| `--fov-half-angle DEG` | - | Additionally score only labels within ±DEG around forward (e.g. camera field of view) |
| `--set NAME=VALUE` | - | Override a detection parameter, repeatable |
| `--csv` | - | Also write the metrics as CSV (`bag,metric,value`) |

To play a bag with a different scan topic on the node, remap it, e.g. `ros2 bag play <bag> --remap /lidar/scan:=/scan`.

## Structure

- `.github/workflows/`: CI (build, test, lint with ruff), using the shared workflows of the [smarty_workspace](https://github.com/DHBW-Smart-Rollerz/smarty_workspace/tree/jazzy/.github/workflows)
- `config/`: Parameter file of the node
- `launch/`: Launch file
- `object_detection_lidar/`: Node and detection pipeline
- `tools/`: Offline evaluation (`evaluate.py`) and hand labels (`labels/`)
- `test/`: Tests
- `package.xml`, `setup.py`, `setup.cfg`, `requirements.txt`: Package metadata and dependencies

## References

- M. Ester, H.-P. Kriegel, J. Sander, X. Xu: A density-based algorithm for discovering clusters in large spatial databases with noise. KDD, 1996 (DBSCAN)
- G. A. Borges, M.-J. Aldon: Line extraction in 2D range images for mobile robotics. Journal of Intelligent and Robotic Systems 40, 2004 (Adaptive Breakpoint Detector)
- G. Taubin: Estimation of planar curves, surfaces, and nonplanar space curves defined by implicit equations. IEEE TPAMI 13, 1991 (circle fit in `shape.py`)

## License

This repository is licensed under the MIT license. See [LICENSE](LICENSE) for details.
