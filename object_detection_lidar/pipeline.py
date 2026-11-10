"""Complete object detection on one 2D laser scan (independent of ROS).

The node and the offline evaluation (tools/evaluate.py) both call detect_objects,
so they run exactly the same steps with the same parameters.
"""

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field

import numpy as np

from object_detection_lidar.box_classification import is_box
from object_detection_lidar.breakpoint_detection import segment_scan, split_clusters
from object_detection_lidar.clustering import (
    DetectedObject,
    cluster_points,
    extract_objects,
    filter_objects,
)
from object_detection_lidar.scan_processing import scan_to_points, valid_beams
from object_detection_lidar.shape import is_round

# Names of the processing steps in the order they are executed (step_timer)
STEPS = (
    "scan_to_points",
    "cluster_points",
    "split_clusters",
    "extract_objects",
    "filter_objects",
    "roi",
    "is_box",
    "drop_round",
)

# Values of PipelineParams.segmentation: DBSCAN, then split_clusters (dbscan_abd),
# or segment_scan on all points without DBSCAN (abd)
SEGMENTATIONS = ("dbscan_abd", "abd")

# Returns a context manager that measures one step, called with the step name
StepTimer = Callable[[str], AbstractContextManager]


@dataclass(frozen=True)
class PipelineParams:
    """Parameters of the object detection.

    The field names are the names of the ROS parameters of the node, the defaults
    are the values of config/ros_params.yaml (explained there).
    """

    # Segmentation of the points into objects, one of SEGMENTATIONS
    segmentation: str = "dbscan_abd"
    min_range: float = 0.15  # in m, closer points are ignored (e.g. car body)
    max_range: float = 4.0  # in m, farther points are ignored
    eps: float = 0.10  # in m, max. distance of neighboring points (DBSCAN)
    min_samples: int = 3  # min. number of points of a core point (DBSCAN)
    # Splitting of the clusters at breakpoints (split_clusters)
    breakpoint_enabled: bool = True  # False keeps the DBSCAN clusters
    breakpoint_lambda_deg: float = 10.0  # in deg, min. angle of incidence
    breakpoint_sigma_r: float = 0.05  # in m, range error (incl. edge artifacts)
    breakpoint_max_beam_gap: int = 3  # max. number of foreign beams in a cluster
    breakpoint_count_invalid: bool = False  # beams without return count as foreign
    max_object_diameter: float = 0.6  # in m, larger objects are dropped (walls)
    box_line_tolerance: float = 0.0023  # in m, max. RMS distance to a box side
    box_line_tolerance_per_m: float = 0.0015  # in m/m, added per m of distance
    box_angle_tolerance: float = 10.0  # in deg, max. deviation of a corner from 90°
    box_corner_tolerance_factor: float = 2.5  # line tolerance factor (corner)
    box_min_size: float = 0.08  # in m, obstacles are at least 0.1 m wide
    box_min_fill_ratio: float = 0.7  # min. ratio of points to beams on a box
    # Vehicle frame of the camera: x forward, y left, origin at the center of the
    # front axle on the ground (see to_vehicle_frame)
    laser_yaw: float = 90.0  # in deg, yaw of the laser frame in the vehicle frame
    lidar_offset_x: float = 0.0  # in m, placeholder until measured
    lidar_offset_y: float = 0.0  # in m, placeholder until measured
    # Region of interest: fixed rectangle in the vehicle frame
    roi_x_min: float = -1.5  # in m, behind the vehicle
    roi_x_max: float = 4.0  # in m, ahead of the vehicle
    roi_y_min: float = -0.8  # in m, right of the vehicle
    roi_y_max: float = 1.6  # in m, left of the vehicle
    roi_enabled: bool = False  # True keeps only objects in the corridor
    # Round objects (no CAuDri obstacle, see shape.py), the noise of the circle
    # fit is box_line_tolerance + box_line_tolerance_per_m * distance
    drop_round_objects: bool = True  # False keeps round objects
    round_min_points: int = 12  # fewer points are never round
    round_max_radius: float = 0.15  # in m, larger circles are never round
    round_min_arc_deg: float = 60.0  # in deg, min. visible arc (from the center)
    round_max_circle_rms_factor: float = 1.5  # max. circle RMS / noise
    round_min_line_ratio: float = 2.0  # min. line RMS / circle RMS
    round_min_corner_ratio: float = 0.8  # min. corner RMS / circle RMS
    round_right_angle_tolerance: float = 15.0  # in deg, right corner: 90° +- this
    round_min_right_corner_ratio: float = 1.5  # min. right corner RMS / circle RMS

    def __post_init__(self) -> None:
        """Check the values that have no meaningful fallback."""
        if self.segmentation not in SEGMENTATIONS:
            raise ValueError(
                f"segmentation must be one of {SEGMENTATIONS}, "
                f"got {self.segmentation!r}"
            )


@dataclass(frozen=True)
class PipelineResult:
    """Result of the object detection on one scan.

    Attributes:
        points: Array of shape (N, 2) with all valid scan points in m (scan frame)
        objects: The detected objects after the wall filter whose centroid lies in
            the region of interest (all of them if it is disabled) and that are
            not round (if drop_round_objects)
        boxes: Whether the object with the same index is a box
        round_objects: The objects dropped as round (e.g. for debug markers)
    """

    points: np.ndarray
    objects: list[DetectedObject]
    boxes: list[bool]
    round_objects: list[DetectedObject] = field(default_factory=list)


# Class id of the LiDAR objects in the camera format: the LiDAR cannot tell a car
# from a pedestrian, the camera only uses OBJECTS.VEHICLE (2) and PEDESTRIAN (10)
# of smarty_utils.enums, so -1 is free and marks "unknown"
UNKNOWN_CLASS = -1.0
# Score of the LiDAR objects in the camera format: every published object is a
# measured cluster that passed the filters, the detection has no probability, so
# the score is fixed (the fusion can still weight the sensors itself)
LIDAR_SCORE = 1.0


def to_vehicle_frame(
    points: np.ndarray, laser_yaw: float, offset_x: float, offset_y: float
) -> np.ndarray:
    """Transform points from the laser frame into the vehicle frame.

    The vehicle frame is the frame of the camera objects (camera_preprocessing,
    position board): x forward, y left, origin at the center of the front axle on
    the ground. p_vehicle = R(laser_yaw) * p_laser + (offset_x, offset_y), like a
    tf from the vehicle frame to the laser frame (2D, the height is ignored).

    Args:
        points: Array of shape (N, 2) with x/y coordinates in m (laser frame)
        laser_yaw: Yaw of the laser frame in the vehicle frame in deg
        offset_x: Position of the LiDAR along x in the vehicle frame in m
        offset_y: Position of the LiDAR along y in the vehicle frame in m

    Returns:
        Array of shape (N, 2) with x/y coordinates in m (vehicle frame).
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    yaw = np.radians(laser_yaw)
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    return points @ rotation.T + np.array([offset_x, offset_y])


def params_to_vehicle_frame(points: np.ndarray, params: PipelineParams) -> np.ndarray:
    """Apply to_vehicle_frame with laser_yaw and the LiDAR offset of params."""
    return to_vehicle_frame(
        points, params.laser_yaw, params.lidar_offset_x, params.lidar_offset_y
    )


def roi_mask(
    points: np.ndarray,
    laser_yaw: float,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> np.ndarray:
    """Check which points lie in a rectangle of the vehicle frame.

    The points are transformed with to_vehicle_frame. The rectangle borders count
    as inside.

    Args:
        points: Array of shape (N, 2) with x/y coordinates in m (laser frame)
        laser_yaw: Yaw of the laser frame in the vehicle frame in deg
        x_min: Rear border of the rectangle in m (vehicle frame)
        x_max: Front border of the rectangle in m (vehicle frame)
        y_min: Right border of the rectangle in m (vehicle frame)
        y_max: Left border of the rectangle in m (vehicle frame)
        offset_x: Position of the LiDAR along x in the vehicle frame in m
        offset_y: Position of the LiDAR along y in the vehicle frame in m

    Returns:
        Array of shape (N,), True for the points inside the rectangle.
    """
    vehicle = to_vehicle_frame(points, laser_yaw, offset_x, offset_y)
    x, y = vehicle[:, 0], vehicle[:, 1]
    return (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)


def vehicle_endpoints(
    objects: list[DetectedObject], params: PipelineParams
) -> np.ndarray:
    """Return the endpoints of the objects in the vehicle frame.

    Args:
        objects: The detected objects (laser frame)
        params: Parameters with laser_yaw and the LiDAR offset

    Returns:
        Array of shape (M, 4) with left_x, left_y, right_x, right_y in m.
    """
    left = np.array([(o.left_x, o.left_y) for o in objects], dtype=np.float64)
    right = np.array([(o.right_x, o.right_y) for o in objects], dtype=np.float64)
    return np.column_stack(
        (
            params_to_vehicle_frame(left, params),
            params_to_vehicle_frame(right, params),
        )
    )


def vehicle_centroids(
    objects: list[DetectedObject], params: PipelineParams
) -> np.ndarray:
    """Return the centroids of the objects in the vehicle frame.

    Args:
        objects: The detected objects (laser frame)
        params: Parameters with laser_yaw and the LiDAR offset

    Returns:
        Array of shape (M, 2) with x/y in m.
    """
    centroids = [(o.centroid_x, o.centroid_y) for o in objects]
    return params_to_vehicle_frame(np.array(centroids, dtype=np.float64), params)


def vehicle_extents(
    objects: list[DetectedObject], params: PipelineParams
) -> np.ndarray:
    """Return the size of the axis-aligned box of the objects in the vehicle frame.

    The box is aligned with the axes of the vehicle frame, so it is computed from
    the transformed points of the cluster (for laser_yaw 90° the extents of the
    laser frame swap places).

    Args:
        objects: The detected objects (laser frame)
        params: Parameters with laser_yaw and the LiDAR offset

    Returns:
        Array of shape (M, 2) with the extent along x and y in m.
    """
    extents = [
        np.ptp(params_to_vehicle_frame(o.points, params), axis=0) for o in objects
    ]
    return np.array(extents, dtype=np.float64).reshape(-1, 2)


def camera_format(objects: list[DetectedObject], params: PipelineParams) -> np.ndarray:
    """Convert the objects into the format of the camera object detection.

    The camera (object_detection, topic /object_detection/object) publishes a flat
    Float32MultiArray with 6 values per object: [cls, BLx, BLy, BRx, BRy, score].
    BL/BR are the bottom left/right corners of the bounding box in mm in the
    vehicle frame (rounded to whole mm), BL is the left side in the image. The
    LiDAR counterpart are the left/right endpoints of the visible contour, both as
    seen from the sensor.

    Args:
        objects: The detected objects (laser frame)
        params: Parameters with laser_yaw and the LiDAR offset

    Returns:
        Array of shape (6 * M,) with float32 values.
    """
    endpoints_mm = np.round(1e3 * vehicle_endpoints(objects, params))
    rows = np.column_stack(
        (
            np.full(len(objects), UNKNOWN_CLASS),
            endpoints_mm,
            np.full(len(objects), LIDAR_SCORE),
        )
    )
    return rows.astype(np.float32).reshape(-1)


def detect_objects(
    ranges: np.ndarray,
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    *,
    params: PipelineParams,
    step_timer: StepTimer | None = None,
) -> PipelineResult:
    """Detect and classify the objects in one 2D laser scan.

    The steps are scan_to_points, cluster_points, split_clusters, extract_objects,
    filter_objects, roi, is_box and drop_round (see STEPS). With
    params.segmentation "abd" cluster_points does nothing and split_clusters
    segments all points with segment_scan (breakpoint_enabled is then ignored).
    The valid range is the intersection of the range of the sensor (from the
    scan) and [params.min_range, params.max_range].

    All valid points are clustered, the region of interest is applied to the
    centroids of the objects afterwards. Clipping the points before the clustering
    would cut walls at the border of the region into short straight pieces, which
    pass the wall filter and look like boxes. The objects stay in the laser frame
    (vehicle frame: params_to_vehicle_frame, vehicle_endpoints, camera_format).

    Args:
        ranges: Measured distances in m, one per beam
        angle_min: Angle of the first beam in rad
        angle_increment: Angle between two beams in rad
        range_min: Minimum range of the sensor in m (from the scan)
        range_max: Maximum range of the sensor in m (from the scan)
        params: Parameters of the detection
        step_timer: Called with the name of each step, the returned context
            manager encloses the step (e.g. for runtime measurements)

    Returns:
        The valid points, the objects and their classes.
    """
    timed = step_timer or (lambda _: nullcontext())

    ranges = np.asarray(ranges, dtype=np.float64)
    valid_range = (max(range_min, params.min_range), min(range_max, params.max_range))
    with timed("scan_to_points"):
        points = scan_to_points(ranges, angle_min, angle_increment, *valid_range)
        beam_indices = np.flatnonzero(valid_beams(ranges, *valid_range))

    breakpoint_args = {
        "num_beams": ranges.size,
        "angle_increment": angle_increment,
        "lambda_angle": np.radians(params.breakpoint_lambda_deg),
        "sigma_r": params.breakpoint_sigma_r,
        "max_beam_gap": params.breakpoint_max_beam_gap,
        "min_samples": params.min_samples,
        # Beams with a return of the sensor (also beyond max_range)
        "returns": None
        if params.breakpoint_count_invalid
        else valid_beams(ranges, range_min, range_max),
    }
    abd = params.segmentation == "abd"

    with timed("cluster_points"):
        # ABD: no clusters, segment_scan walks all points
        labels = (
            np.zeros(len(points), dtype=int)
            if abd
            else cluster_points(points, params.eps, params.min_samples)
        )

    with timed("split_clusters"):
        if abd:
            labels = segment_scan(points, beam_indices, **breakpoint_args)
        elif params.breakpoint_enabled:
            labels = split_clusters(points, beam_indices, labels, **breakpoint_args)

    with timed("extract_objects"):
        objects = extract_objects(points, labels)

    with timed("filter_objects"):
        objects = filter_objects(objects, params.max_object_diameter)

    with timed("roi"):
        if params.roi_enabled:
            centroids = np.array(
                [(obj.centroid_x, obj.centroid_y) for obj in objects], dtype=np.float64
            ).reshape(-1, 2)
            inside = roi_mask(
                centroids,
                params.laser_yaw,
                x_min=params.roi_x_min,
                x_max=params.roi_x_max,
                y_min=params.roi_y_min,
                y_max=params.roi_y_max,
                offset_x=params.lidar_offset_x,
                offset_y=params.lidar_offset_y,
            )
            objects = [obj for obj, keep in zip(objects, inside, strict=True) if keep]

    with timed("is_box"):
        boxes = [
            is_box(
                obj.points,
                angle_increment,
                line_tolerance=params.box_line_tolerance,
                line_tolerance_per_m=params.box_line_tolerance_per_m,
                angle_tolerance=params.box_angle_tolerance,
                corner_tolerance_factor=params.box_corner_tolerance_factor,
                min_size=params.box_min_size,
                min_fill_ratio=params.box_min_fill_ratio,
            )
            for obj in objects
        ]

    with timed("drop_round"):
        # Boxes are never round, so only the other objects are checked
        round_flags = [
            params.drop_round_objects
            and not box
            and is_round(
                obj.points,
                min_points=params.round_min_points,
                max_radius=params.round_max_radius,
                min_arc_deg=params.round_min_arc_deg,
                noise=params.box_line_tolerance,
                noise_per_m=params.box_line_tolerance_per_m,
                max_circle_rms_factor=params.round_max_circle_rms_factor,
                min_line_ratio=params.round_min_line_ratio,
                min_corner_ratio=params.round_min_corner_ratio,
                right_angle_tolerance=params.round_right_angle_tolerance,
                min_right_corner_ratio=params.round_min_right_corner_ratio,
            )
            for obj, box in zip(objects, boxes, strict=True)
        ]
        round_objects = [
            obj for obj, rnd in zip(objects, round_flags, strict=True) if rnd
        ]
        kept = [i for i, rnd in enumerate(round_flags) if not rnd]
        objects = [objects[i] for i in kept]
        boxes = [boxes[i] for i in kept]

    return PipelineResult(
        points=points, objects=objects, boxes=boxes, round_objects=round_objects
    )
