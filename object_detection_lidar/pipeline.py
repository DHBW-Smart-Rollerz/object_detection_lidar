"""Complete object detection on one 2D laser scan (independent of ROS).

The node and the offline evaluation (tools/evaluate.py) both call detect_objects,
so they run exactly the same steps with the same parameters.
"""

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import numpy as np

from object_detection_lidar.box_classification import is_box
from object_detection_lidar.clustering import (
    DetectedObject,
    cluster_points,
    extract_objects,
    filter_objects,
)
from object_detection_lidar.scan_processing import scan_to_points

# Names of the processing steps in the order they are executed (step_timer)
STEPS = (
    "scan_to_points",
    "cluster_points",
    "extract_objects",
    "filter_objects",
    "roi",
    "is_box",
)

# Returns a context manager that measures one step, called with the step name
StepTimer = Callable[[str], AbstractContextManager]


@dataclass(frozen=True)
class PipelineParams:
    """Parameters of the object detection.

    The field names are the names of the ROS parameters of the node, the defaults
    are the values of config/ros_params.yaml (explained there).
    """

    min_range: float = 0.15  # in m, closer points are ignored (e.g. car body)
    max_range: float = 4.0  # in m, farther points are ignored
    eps: float = 0.10  # in m, max. distance of neighboring points (DBSCAN)
    min_samples: int = 3  # min. number of points of a core point (DBSCAN)
    max_object_diameter: float = 0.6  # in m, larger objects are dropped (walls)
    box_line_tolerance: float = 0.0023  # in m, max. RMS distance to a box side
    box_line_tolerance_per_m: float = 0.0015  # in m/m, added per m of distance
    box_angle_tolerance: float = 10.0  # in deg, max. deviation of a corner from 90°
    box_corner_tolerance_factor: float = 2.5  # line tolerance factor (corner)
    box_min_size: float = 0.08  # in m, obstacles are at least 0.1 m wide
    box_min_fill_ratio: float = 0.7  # min. ratio of points to beams on a box
    # Region of interest: fixed rectangle in the vehicle frame (x forward, y left,
    # origin at the center of the LiDAR)
    laser_yaw: float = 90.0  # in deg, yaw of the laser frame in the vehicle frame
    roi_x_min: float = -1.5  # in m, behind the vehicle
    roi_x_max: float = 3.0  # in m, ahead of the vehicle
    roi_y_min: float = -0.8  # in m, right of the vehicle
    roi_y_max: float = 1.6  # in m, left of the vehicle
    roi_enabled: bool = True  # False keeps all objects


@dataclass(frozen=True)
class PipelineResult:
    """Result of the object detection on one scan.

    Attributes:
        points: Array of shape (N, 2) with all valid scan points in m (scan frame)
        objects: The detected objects after the wall filter whose centroid lies in
            the region of interest (all of them if it is disabled)
        boxes: Whether the object with the same index is a box
    """

    points: np.ndarray
    objects: list[DetectedObject]
    boxes: list[bool]


def roi_mask(
    points: np.ndarray,
    laser_yaw: float,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    """Check which points lie in a rectangle of the vehicle frame.

    The points are rotated from the laser frame into the vehicle frame (x forward,
    y left) by p_vehicle = R(laser_yaw) * p_laser, like a tf from the vehicle
    frame to the laser frame without translation. The rectangle borders count as
    inside.

    Args:
        points: Array of shape (N, 2) with x/y coordinates in m (laser frame)
        laser_yaw: Yaw of the laser frame in the vehicle frame in deg
        x_min: Rear border of the rectangle in m (vehicle frame)
        x_max: Front border of the rectangle in m (vehicle frame)
        y_min: Right border of the rectangle in m (vehicle frame)
        y_max: Left border of the rectangle in m (vehicle frame)

    Returns:
        Array of shape (N,), True for the points inside the rectangle.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    yaw = np.radians(laser_yaw)
    c, s = np.cos(yaw), np.sin(yaw)
    x = c * points[:, 0] - s * points[:, 1]
    y = s * points[:, 0] + c * points[:, 1]
    return (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)


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

    The steps are scan_to_points, cluster_points, extract_objects, filter_objects,
    roi and is_box (see STEPS). The valid range is the intersection of the range
    of the sensor (from the scan) and [params.min_range, params.max_range].

    All valid points are clustered, the region of interest is applied to the
    centroids of the objects afterwards. Clipping the points before the clustering
    would cut walls at the border of the region into short straight pieces, which
    pass the wall filter and look like boxes. The objects stay in the laser frame.

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

    with timed("scan_to_points"):
        points = scan_to_points(
            np.asarray(ranges),
            angle_min=angle_min,
            angle_increment=angle_increment,
            range_min=max(range_min, params.min_range),
            range_max=min(range_max, params.max_range),
        )

    with timed("cluster_points"):
        labels = cluster_points(points, params.eps, params.min_samples)

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

    return PipelineResult(points=points, objects=objects, boxes=boxes)
