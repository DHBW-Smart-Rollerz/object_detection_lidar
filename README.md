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

`object_detection_lidar_node` subscribes to the 2D laser scan and processes every scan in four steps. The steps are combined in `detect_objects` (`pipeline.py`, independent of ROS), which the evaluation tool uses as well; the node only reads its parameters and publishes the result:

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

The values are explained in `config/ros_params.yaml`. The defaults of the detection parameters are defined once in `PipelineParams` (`pipeline.py`), a test checks that they match the yaml file. With `debug` the node logs the runtime of each step (`scan_to_points`, `cluster_points`, `extract_objects`, `filter_objects`, `is_box`) and of publishing. To play the rosbags of 03/11/2026, remap the scan topic: `ros2 bag play <bag> --topics /lidar/scan --remap /lidar/scan:=/scan`.

## Evaluation

`tools/evaluate.py` runs every scan of one or more rosbags through `detect_objects` (no ROS node, no playback) and prints metrics per bag. The rosbags stay outside of the repository. ROS has to be sourced (`rosbag2_py`), the Python environment needs `scikit-learn` like the node:

```bash
source /opt/ros/jazzy/setup.zsh
python3 tools/evaluate.py <bag> [<bag> ...] [--csv metrics.csv]
# e.g. both rosbags of 03/11/2026
python3 tools/evaluate.py ~/smarty_workspace/bags/ROS_Bags_2026-11-03_lidar/rosbag2_2026_11_03-10_{08_07,10_39}
```

The tool always evaluates the package next to it (working tree) with the parameters of `config/ros_params.yaml`.

| Option | Default | Description |
| --- | --- | --- |
| `--topic` | `/lidar/scan` | `sensor_msgs/LaserScan` topic in the bag |
| `--params` | `config/ros_params.yaml` | Parameter file of the node |
| `--labels-dir` | `tools/labels` | Hand labels, one file `<bag directory name>.yaml` per bag |
| `--match-radius` | `0.10` | Max. distance between a labelled box and a detected centroid, in m |
| `--flicker-radius` | `0.05` | Max. centroid shift between two scans to count as the same object, in m |
| `--static START:END` | from the labels | Scan range with standing vehicle (END excluded), repeatable |
| `--csv` | - | Also write the metrics as table `bag,metric,value` |

| Metric | Meaning |
| --- | --- |
| `time <step> [ms]` | Runtime per scan as median / 95th percentile / max, for the whole detection (`total`), each step and `clustering` (sum of `cluster_points`, `extract_objects`, `filter_objects`). Measured offline on the current machine, without ROS overhead |
| `objects/scan`, `boxes/scan` | Objects after the wall filter and objects classified as box, mean and median / 95th percentile / max |
| `flicker [%]` | Every object is assigned to the nearest object of the previous scan. If its centroid moved less than `--flicker-radius`, it counts as found again (`flicker pairs`). Flicker is the share of these pairs whose class (box/other) changed. No ego motion compensation, so with a moving vehicle mostly far or static objects are found again |
| `static objects`, `static scatter [mm]` | Only for scan ranges with standing vehicle: objects are followed over the range (nearest track mean within 8 cm), objects found in at least 80 % of the scans are kept. Scatter is the norm of the standard deviations of the centroid in x and y (median / 90th percentile), i.e. the precision, not the absolute accuracy. The main table pools all phases of a bag, a second table lists each phase (scan range, objects, median / 90th percentile) |
| `TP / FN / missed / FP` | Only with labels: labelled boxes matched one to one (nearest first, within `--match-radius`) to the objects in the labelled region. TP: classified as box, FN: not classified as box, missed: no object (labelled without cluster or no match), FP: box in the region matching no labelled box |
| `precision`, `recall`, `recall incl. missed` | TP / (TP + FP), TP / (TP + FN), TP / (TP + FN + missed) |

### Labels

Hand labels are small yaml files in `tools/labels/`, one per bag (file name = name of the bag directory). The scan index counts the messages of the topic from 0. Positions are centroids of the box clusters in the scan frame. Within `region` all clusters were reviewed, so every detected box there that matches no label is a false positive:

```yaml
bag: rosbag2_2026_11_03-10_08_07
frame: laser
region:
  heading: -90.0  # in deg, direction of the labelled sector (forward = -y)
  half_angle: 75.0  # in deg
  max_range: 4.0  # in m
static_scans:  # scan ranges [start, end) with the vehicle standing
  - [0, 6]
scans:
  - scan: 10
    boxes:  # centroids of the box clusters, in m
      - [0.3528, -1.1788]
    missed_boxes: 0  # boxes without a cluster
```

The labels of the rosbags of 03/11/2026 cover 25 moments (camera image and nearest scan) and were made by one person, some of them are uncertain.

## Structure

- `.github/workflows/`: CI pipelines, which call the shared workflows of the [smarty_workspace](https://github.com/DHBW-Smart-Rollerz/smarty_workspace/tree/jazzy/.github/workflows):
  - `build.yaml`: Builds the package
  - `test.yaml`: Runs the tests of the package
  - `lint.yaml`: Checks the code with ruff (Python) and ament_uncrustify (C++)
- `config/`: All configurations (most of the time yaml files)
- `launch/`: Contains all launch files. Launch files can start multiple nodes with yaml-configurations
- `resource/`: Contains the package name (required to build with colcon)
- `object_detection_lidar`: Contains all nodes and sources for the ros package
- `tools/`: Offline evaluation on rosbags (`evaluate.py`) and the hand labels (`labels/`)
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
