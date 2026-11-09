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

`object_detection_lidar_node` subscribes to the 2D laser scan and processes every scan in six steps. The steps are combined in `detect_objects` (`pipeline.py`, independent of ROS), which the evaluation tool uses as well; the node only reads its parameters and publishes the result:

1. Drop invalid points and points outside `[min_range, max_range]`, publish the rest as point cloud in the frame of the scan
2. Cluster the points with DBSCAN (`clustering.py`)
3. Split the clusters at breakpoints between consecutive beams, see [Breakpoint detection](#breakpoint-detection) (`breakpoint_detection.py`). With `segmentation: abd` steps 2 and 3 are replaced by a segmentation of all points at breakpoints (no DBSCAN), see [ABD-only segmentation](#abd-only-segmentation)
4. Drop objects with a diameter above `max_object_diameter` (walls)
5. Keep only objects whose centroid lies in the region of interest, a fixed corridor around the vehicle (see below)
6. Classify every remaining object as box or other by the shape of its visible contour, one straight side or a right-angled corner (`box_classification.py`)
7. Drop the objects that are no box and whose visible contour is round (`shape.py`, see [Round objects](#round-objects))

All topics can be shown in RViz (`PointCloud2`, `MarkerArray`).

| Parameter | Default | Description |
| --- | --- | --- |
| `scan_subscriber` | `/scan` | Input topic (`sensor_msgs/LaserScan`) |
| `points_publisher` | `/object_detection_lidar/points` | Valid scan points (`sensor_msgs/PointCloud2`) |
| `markers_publisher` | `/object_detection_lidar/markers` | One cube per object, boxes red (namespace `boxes`), others blue (`objects`), dropped round objects gray (`round`) (`visualization_msgs/MarkerArray`) |
| `objects_publisher` | `/object_detection_lidar/objects` | One point per object (centroid) with the fields `extent_x`, `extent_y`, `diameter`, `is_box` (1.0/0.0) and `left_x`, `left_y`, `right_x`, `right_y`, `width`, all in the vehicle frame (`frame_id` `vehicle`, extents along the vehicle axes) (`sensor_msgs/PointCloud2`) |
| `object_publisher` | `/object_detection_lidar/object` | Objects in the format of the camera, see [Output](#output) (`std_msgs/Float32MultiArray`) |
| `min_range` | `0.15` | Closer points are ignored, in m |
| `max_range` | `4.0` | Farther points are ignored, in m |
| `segmentation` | `dbscan_abd` | `dbscan_abd`: DBSCAN, then splitting at breakpoints; `abd`: breakpoints only (no DBSCAN, experimental) |
| `eps` | `0.10` | Max. distance of neighboring points (DBSCAN), in m |
| `min_samples` | `3` | Min. number of points of a core point (DBSCAN) |
| `breakpoint_enabled` | `True` | `False` keeps the DBSCAN clusters (no splitting) |
| `breakpoint_lambda_deg` | `10.0` | Smallest angle of incidence of a surface that is not split, in deg |
| `breakpoint_sigma_r` | `0.05` | Range error of a single point (noise and local artifacts), in m |
| `breakpoint_max_beam_gap` | `3` | Max. number of foreign beams between two consecutive points of an object |
| `breakpoint_count_invalid` | `False` | `True` also counts beams without a return as foreign beams |
| `max_object_diameter` | `0.6` | Larger objects are dropped (walls), in m |
| `box_line_tolerance` | `0.0023` | Max. RMS distance of the points to a box side at 0 m, in m |
| `box_line_tolerance_per_m` | `0.0015` | Increase of the line tolerance per m of distance, in m/m |
| `box_angle_tolerance` | `10.0` | Max. deviation of a box corner from 90°, in deg |
| `box_corner_tolerance_factor` | `2.5` | Factor on the line tolerance for the sides of a corner |
| `box_min_size` | `0.08` | Min. distance between the outermost points of a box, in m |
| `box_min_fill_ratio` | `0.7` | Min. ratio of points to beams within the angular span of a box |
| `laser_yaw` | `90.0` | Yaw of the laser frame in the vehicle frame (like a tf `base_link` -> `laser`), in deg |
| `lidar_offset_x` | `0.0` | Position of the LiDAR along x in the vehicle frame, in m (placeholder until measured) |
| `lidar_offset_y` | `0.0` | Position of the LiDAR along y in the vehicle frame, in m (placeholder until measured) |
| `roi_x_min` | `-1.5` | Rear border of the region of interest (vehicle frame, x forward), in m |
| `roi_x_max` | `4.0` | Front border of the region of interest, in m |
| `roi_y_min` | `-0.8` | Right border of the region of interest (vehicle frame, y left), in m |
| `roi_y_max` | `1.6` | Left border of the region of interest, in m |
| `roi_enabled` | `True` | `False` keeps all objects (no region of interest) |
| `drop_round_objects` | `True` | `False` keeps round objects |
| `round_min_points` | `12` | Objects with fewer points are never round |
| `round_max_radius` | `0.15` | Max. radius of a round object, in m |
| `round_min_arc_deg` | `60.0` | Min. visible arc of a round object (seen from its center), in deg |
| `round_max_circle_rms_factor` | `1.5` | Max. RMS of the circle as multiple of the noise (`box_line_tolerance` + `box_line_tolerance_per_m` * distance) |
| `round_min_line_ratio` | `2.0` | Min. ratio of the line RMS to the circle RMS |
| `round_min_corner_ratio` | `0.8` | Min. ratio of the corner RMS (two lines) to the circle RMS |
| `round_right_angle_tolerance` | `15.0` | A corner within 90° ± this needs `round_min_right_corner_ratio`, in deg |
| `round_min_right_corner_ratio` | `1.5` | Min. ratio of the corner RMS to the circle RMS for a right corner |
| `debug` | `False` | Logs the objects with their class and the processing times |

The values are explained in `config/ros_params.yaml`. The defaults of the detection parameters are defined once in `PipelineParams` (`pipeline.py`), a test checks that they match the yaml file. With `debug` the node logs the runtime of each step (`scan_to_points`, `cluster_points`, `split_clusters`, `extract_objects`, `filter_objects`, `roi`, `is_box`, `drop_round`) and of publishing. To play the rosbags of 03/11/2026, remap the scan topic: `ros2 bag play <bag> --topics /lidar/scan --remap /lidar/scan:=/scan`.

### Output

Every object has two endpoints, the outermost points of its visible contour as seen from the LiDAR: `left` has the largest bearing (angle around the LiDAR, counterclockwise), `right` the smallest (`contour_endpoints` in `clustering.py`). The bearings are taken relative to the mean point of the object, so objects behind the LiDAR (around ±180°) work as well. `width` is the distance between the endpoints, i.e. the visible width, not the real size of the object (the back side is hidden).

The endpoints are transformed into the vehicle frame of the camera (`to_vehicle_frame` in `pipeline.py`): x forward, y left, origin at the center of the front axle on the ground (position board of `camera_preprocessing`, `img/position/real_coordinates.pdf` and `config/config.yaml`), p_vehicle = R(`laser_yaw`) · p_laser + (`lidar_offset_x`, `lidar_offset_y`). The offset is not measured yet (0.0), so the positions are currently relative to the LiDAR. `/object_detection_lidar/objects` is completely in this frame (`frame_id` `vehicle`), points and markers stay in the frame `laser`. No tf `laser` -> `vehicle` is published, so rviz with the fixed frame `laser` cannot show the objects cloud (it would need a static transform with `laser_yaw` and the offset).

`/object_detection_lidar/object` has the same format as the camera objects (`/object_detection/object` of the `object_detection` package), so the fusion can read both alike: a flat `Float32MultiArray` without layout, 6 values per object `[cls, BLx, BLy, BRx, BRy, score]`, empty for no object.

| Value | Camera | LiDAR |
| --- | --- | --- |
| `cls` | `OBJECTS` of `smarty_utils.enums` (2 vehicle, 10 pedestrian) | `-1` (unknown, the LiDAR cannot classify cars and pedestrians) |
| `BLx`, `BLy` | Bottom left corner of the bounding box (left in the image) | Left endpoint |
| `BRx`, `BRy` | Bottom right corner of the bounding box | Right endpoint |
| `score` | Confidence of the network | `1.0` (a measured cluster that passed the filters, no probability) |

All positions are in mm in the vehicle frame, rounded to whole mm like the camera. Only the objects in the region of interest are published (all with `roi_enabled: false`), boxes and others alike. For objects ahead the left endpoint has the larger y, like BL of the camera; for objects behind the vehicle it is the other way round (left as seen from the LiDAR).

### Breakpoint detection

DBSCAN only looks at the distances between the points, so two obstacles closer than `eps` (0.1 m) become one object, even if the beams between them hit the background. The scan contains more information than the point set: the order of the beams. `split_clusters` (`breakpoint_detection.py`) sorts the points of each cluster by beam and splits it between two consecutive points if

- their distance is larger than D_max = r · sin(Δφ) / sin(λ − Δφ) + 3σ_r (Adaptive Breakpoint Detector, G. A. Borges and M.-J. Aldon: Line extraction in 2D range images for mobile robotics, Journal of Intelligent and Robotic Systems 40, 2004). r is the range of the first point, Δφ the actual angle between the two beams (skipped beams enlarge it), λ = `breakpoint_lambda_deg` the smallest angle of incidence of a surface that stays in one piece and σ_r = `breakpoint_sigma_r` the range error. D_max grows with the range, so the same parameters work near and far. For Δφ ≥ λ the two points are always split.
- more than `breakpoint_max_beam_gap` foreign beams lie between them, i.e. beams that hit something outside of the cluster (background, also beyond `max_range`, other objects, noise): something else is seen between the two points.

Beams without a return (inf, nan) are bridged and do not count (`breakpoint_count_invalid: false`), because dark or shiny spots on an obstacle return nothing. The beams form a circle: a cluster across the seam of the scan (around ±180°) is walked across it. Pieces with fewer than `min_samples` points become noise, like in DBSCAN. The split pieces then pass through the wall filter like any other object.

Results (09/11/2026): two obstacles 5 cm apart at 1.3 m (book 140 mm, round tin Ø 135 mm) are found as two stable objects with widths of 143 and 157 mm instead of one 345 mm object. On the rosbags of 03/11/2026 the box classification and the object recall are unchanged, there are ~0.5 more objects per scan.

The jump criterion is set loosely on purpose (σ_r = 0.05 m, i.e. 3σ_r = 15 cm, above eps): the A2M12 measures flat surfaces with local errors of up to ±7 cm and jumps of up to 14 cm between neighboring beams (seen on a carton at 0.9 m), which a tight σ_r splits into pieces. Therefore an object partly hidden behind another one is only split if the beams between them hit the background; a depth jump alone below ~15 cm does not split it. The values of the parameters are explained in `config/ros_params.yaml`.

### ABD-only segmentation

With `segmentation: abd` the points are not clustered by DBSCAN. `segment_scan` (`breakpoint_detection.py`) walks all valid points in the order of their beams around the full circle (across the seam at ±180°) and splits them with the same criteria and parameters as `split_clusters`; segments with fewer than `min_samples` points are noise, `eps` and `breakpoint_enabled` are not used. Since every valid point takes part, the beams between two consecutive points are only beams without a return (bridged with `breakpoint_count_invalid: false`) or beams whose return lies outside of `[min_range, max_range]`. `breakpoint_max_beam_gap` therefore only finds gaps to the far background; gaps to background within `max_range` appear as two jumps in depth and are found by D_max.

Differences to `dbscan_abd` (comparison of 09/11/2026 on the rosbags of 03/11 and 09/11/2026): without the limit of `eps` (0.1 m) on the distance of neighboring points, two things are merged that DBSCAN separates: points left and right of a dropout of many beams (D_max grows with the bridged angle, e.g. 30 beams at 2.5 m allow ~4 m) and objects less than ~15–19 cm in front of another surface (3σ_r = 15 cm). Labelled boxes then merge with walls and are dropped by the wall filter. A single stray point behind a surface splits it (DBSCAN treats it as noise). The segmentation itself is ~8x faster (~0.6 instead of ~5 ms). On the labelled rosbags of 03/11/2026 `abd` reaches an object recall of 80/88 % instead of 90/94 %, so `dbscan_abd` was chosen as default (09/11/2026).

### Region of interest

Only objects whose centroid lies in a fixed rectangle around the vehicle are kept and classified. The centroids are transformed into the vehicle frame (`to_vehicle_frame`, see [Output](#output)) and compared with `roi_x_min..roi_x_max` and `roi_y_min..roi_y_max` (borders included). The borders were chosen with the origin at the LiDAR (offset 0.0) and have to be checked once the offset is measured. The objects keep all their points, also those outside of the rectangle (markers in the frame `laser`, `/object_detection_lidar/objects` in the vehicle frame). The points topic contains all valid points.

The filter works on the objects, not on the points: all valid points are clustered and the wall filter sees the complete objects. Clipping the points before the clustering would cut barriers and walls at the border into short straight pieces, which are shorter than `max_object_diameter` and look like boxes (3 false boxes per 14 labelled scans in the rosbags of 03/11/2026). The centroid is the mean of the visible points of the object (not the center of the real object).

In the rosbags of 03/11/2026 the vehicle moves along −y of the laser frame and +x points to the left (ego motion by ICP, turn direction checked against the camera), so `laser_yaw` is 90°. The default corridor reaches 4 m ahead (as far as `max_range`), 1.5 m behind, 0.8 m to the right and 1.6 m to the left (reasons in `config/ros_params.yaml`). It covers the area behind and next to the vehicle on purpose, which the camera cannot see: merging back after overtaking within 2 m, parking spots, crossing traffic.

Limits:

- The corridor is fixed and does not follow curves. In a curve, obstacles on the track can leave it and objects next to the track can enter it.
- Crossing traffic at intersections is only partly covered, only within 0.8 m to the right and 1.6 m to the left.
- An object counts by its centroid only: an obstacle with its centroid outside of the corridor is dropped even if it reaches into it, one with its centroid inside is kept completely.
- Clustering and the wall filter still run on all points, so the corridor saves only the runtime of `is_box`.
- `roi_x_max` equals `max_range` (4 m), so the corridor drops nothing ahead that the range limit keeps. A shorter corridor finds obstacles ahead later (in the rosbags of 03/11/2026 with 3 m: object recall incl. missed 80.6 % instead of 91.7 %).

### Round objects

All obstacles of the CAuDri track, including the pedestrian figures, are angular boxes. Round contours come from things next to or on the track that are no obstacles: legs of examiners and spectators, bottles, chair legs. `is_round` (`shape.py`) fits a circle (algebraic fit of Taubin, nearly unbiased for short arcs, no iteration), a line and the best split into two lines (corner) to the visible contour. An object that is no box is dropped if all of these hold: at least `round_min_points` points, radius 1 cm to `round_max_radius`, the arc bulges toward the sensor and covers at least `round_min_arc_deg`, the circle fits within the noise and clearly better than a line, and no corner fits clearly better (stricter for right-angled corners, which is how the corner of a small far box looks). Otherwise the object is kept (recall first). Dropped objects are not published on `object_publisher` and `objects_publisher`, only as gray markers.

Evaluation of 09/11/2026:

- Rosbags of 03/11/2026 (`tools/evaluate.py`, with and without `--fov-half-angle 52`): object recall and box TP/FN/FP unchanged, ~0.17 round objects per scan dropped (objects/scan 10.4 → 10.2 and 9.2 → 9.0). None of the 1171 object scans of the labelled boxes (tracked ±30 scans around each labelled scan) was dropped. Runtime of `drop_round` ~1 ms (median).
- Home rosbags of 09/11/2026 (objects at ~0.7–1.3 m): toilet roll, vase and Haribo tin round in 100 % of the scans, kitchen roll in 92–96 %; carton, two books and a letter in 0 %.
- Synthetic scans (`test/test_shape.py`): cylinders of 50–150 mm are round as long as they have enough points (100 mm up to ~2.5 m, 150 mm up to ~3.5 m), boxes (side, corner, oblique, pedestrian figure, 100 mm box up to 4 m) never, also with twice the noise.

Limits: far round objects with fewer than `round_min_points` points are kept. Thin objects (chair legs < 3 cm) only have enough points close to the sensor.

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
| `--fov-half-angle DEG` | - | Additionally score the labels only within ±DEG around forward of the vehicle (camera field of view, see below), rows (objects and boxes) prefixed with `fov` |
| `--set NAME=VALUE` | - | Override a detection parameter of the yaml file, e.g. `--set roi_enabled=false`, repeatable |
| `--csv` | - | Also write the metrics as table `bag,metric,value` |

| Metric | Meaning |
| --- | --- |
| `time <step> [ms]` | Runtime per scan as median / 95th percentile / max, for the whole detection (`total`), each step and `segmentation` (sum of `cluster_points`, `split_clusters`, `extract_objects`, `filter_objects`, i.e. DBSCAN, breakpoint splitting or ABD alone, and building and filtering the objects). Measured offline on the current machine, without ROS overhead |
| `objects/scan`, `boxes/scan` | Objects after the wall filter and objects classified as box, mean and median / 95th percentile / max |
| `flicker [%]` | Every object is assigned to the nearest object of the previous scan. If its centroid moved less than `--flicker-radius`, it counts as found again (`flicker pairs`). Flicker is the share of these pairs whose class (box/other) changed. No ego motion compensation, so with a moving vehicle mostly far or static objects are found again |
| `static objects`, `static scatter [mm]` | Only for scan ranges with standing vehicle: objects are followed over the range (nearest track mean within 8 cm), objects found in at least 80 % of the scans are kept. Scatter is the norm of the standard deviations of the centroid in x and y (median / 90th percentile), i.e. the precision, not the absolute accuracy. The main table pools all phases of a bag, a second table lists each phase (scan range, objects, median / 90th percentile) |
| `static width / endpoint scatter [mm]` | Same tracks: median of the per-track median `width`, and the endpoint scatter (mean of the norms of the standard deviations of the left and of the right endpoint, median / 95th percentile over the tracks). Contains the beam spacing (the outermost beam on the object changes from scan to scan). Also listed per phase in the second table |
| `objects found / not found / missed` | **Primary metric**, only with labels: labelled boxes matched one to one (nearest first, within `--match-radius`) to all objects in the labelled region, regardless of their class. found: matched by an object (box or other), not found: labelled with a cluster but no object within the radius (e.g. dropped by the region of interest or the wall filter, merged with a neighbor, centroid shifted), missed: labelled without cluster |
| `objects recall`, `objects recall incl. missed` | found / (found + not found), found / (found + not found + missed) |
| `TP / FN / missed / FP` | Secondary (box classification), only with labels: labelled boxes matched one to one (nearest first, within `--match-radius`) to the objects in the labelled region. TP: classified as box, FN: not classified as box, missed: no object (labelled without cluster or no match), FP: box in the region matching no labelled box |
| `precision`, `recall`, `recall incl. missed` | TP / (TP + FP), TP / (TP + FN), TP / (TP + FN + missed) |

The main question of the evaluation is whether the LiDAR finds the obstacle at all and where it is: the class of an obstacle comes from the camera in the sensor fusion. The object recall is therefore the primary metric. `is_box` is only additional information (e.g. to support the camera or to filter clutter), so box precision and recall are secondary metrics. Both are scored with the same matching, region and field of view.

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

### Camera field of view

The labels were made with the camera image, objects outside of the image could not be checked. `--fov-half-angle` therefore also scores the labels in the intersection of the labelled region and a sector of ±DEG around forward (from `laser_yaw`). Detected objects and labelled boxes outside of it are ignored, labelled boxes without a cluster have no position and always count.

For the camera of the rosbags of 03/11/2026 use `--fov-half-angle 52`:

- The image (2064 × 1544 px) comes from a fisheye lens. The image circle is cut at the top and bottom and horizontally reaches from px 65 to 2049, centered at px ~1057.
- `camera_preprocessing` contains no stored calibration (`calib.bin` is missing). Calibrating the chessboard images in `camera_preprocessing/img/calib/new_lens` (16 images, 6 × 6) with the pinhole model of the package gives fx = 1158 px, cx = 1055 px (matches the center of the image circle) and k1 = −0.37, RMS 0.4 px. The edges of the image circle are then at −51.6° and +51.8° (OpenCV fisheye model: ±51.8°, but with RMS 15.6 px). The chessboard corners reach up to 47.5°, so only a few degrees are extrapolated. The focal length in `config.yaml` (2 mm) does not match fx (≈ 4.0 mm at 3.45 µm pixels).
- Checked in the rosbags by drawing the bearing of the LiDAR objects into the camera image: near and medium boxes line up if the camera is assumed ~0.3 m behind the LiDAR.

Limits:

- The sector is centered at the LiDAR, the camera sits behind it (~0.3 m, estimated from the images, not measured). Near objects are seen by the camera at a much smaller angle than by the LiDAR (e.g. 0.4 m ahead and 0.3 m to the side: 60° from the LiDAR, 33° from the camera). In the rosbags of 03/11/2026, 6 labelled boxes at 0.35-1.45 m are therefore outside of ±52° although they are in the image.
- The camera is assumed to look straight forward (yaw offset to the vehicle not known).
- The image circle is also cut at the top and bottom, near objects low in front of the vehicle are hidden by the vehicle itself.

The labels of the rosbags of 03/11/2026 cover 25 moments (camera image and nearest scan) and were made by one person, some of them are uncertain. 5 objects next to the track (furniture of the room, not obstacles) were removed from the labels on 09/11/2026 after checking the camera images.

## Structure

- `.github/workflows/`: CI pipelines, which call the shared workflows of the [smarty_workspace](https://github.com/DHBW-Smart-Rollerz/smarty_workspace/tree/jazzy/.github/workflows):
  - `build.yaml`: Builds the package
  - `test.yaml`: Runs the tests of the package
  - `lint.yaml`: Checks the code with ruff
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

## License

This repository is licensed under the MIT license. See [LICENSE](LICENSE) for details.
