from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pytest
import yaml
from object_detection_lidar.box_classification import is_box
from object_detection_lidar.clustering import (
    cluster_points,
    extract_objects,
    filter_objects,
)
from object_detection_lidar.pipeline import (
    STEPS,
    PipelineParams,
    detect_objects,
    roi_mask,
)
from object_detection_lidar.scan_processing import scan_to_points
from test_box_classification import add_noise, circle, rectangle

CONFIG = Path(__file__).resolve().parents[1] / "config" / "ros_params.yaml"

# Beam layout of the RPLIDAR A2M12 in the rosbags of 03/11/2026
NUM_BEAMS = 1800
ANGLE_MIN = -np.pi  # in rad
ANGLE_INCREMENT = 2 * np.pi / NUM_BEAMS  # in rad
RANGE_MIN = 0.15  # in m, of the sensor
RANGE_MAX = 12.0  # in m, of the sensor
# Behavior without region of interest (all valid points are clustered)
NO_ROI = PipelineParams(roi_enabled=False)
# Default corridor of the node, in the rosbags forward is -y and left is +x
ROI = {"x_min": -1.5, "x_max": 3.0, "y_min": -0.8, "y_max": 1.6}


def scan_ranges(segments: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Return the range of every beam to the first hit segment (inf for no hit)."""
    angles = ANGLE_MIN + np.arange(NUM_BEAMS) * ANGLE_INCREMENT
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    ranges = np.full(NUM_BEAMS, np.inf)
    for start, end in segments:
        edge = end - start
        denominator = directions[:, 1] * edge[0] - directions[:, 0] * edge[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (start[1] * edge[0] - start[0] * edge[1]) / denominator
            s = (directions[:, 0] * start[1] - directions[:, 1] * start[0]) / (
                denominator
            )
        hit = (t > 0) & (s >= 0) & (s <= 1)
        ranges[hit] = np.minimum(ranges[hit], t[hit])
    return ranges


def noisy_scene() -> np.ndarray:
    """Return the ranges of a scan with a box, a cylinder and a wall (with noise)."""
    segments = [
        *rectangle((1.5, 0.5), 0.3, 0.3, np.radians(20)),
        *circle((-1.0, 1.0), 0.1),
        (np.array([-2.0, -2.5]), np.array([2.0, -2.5])),
    ]
    ranges = scan_ranges(segments)
    valid = np.isfinite(ranges)
    angles = ANGLE_MIN + np.arange(NUM_BEAMS)[valid] * ANGLE_INCREMENT
    points = ranges[valid, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    ranges[valid] = np.linalg.norm(add_noise(points), axis=1)
    return ranges


def run(ranges: np.ndarray, params: PipelineParams, **kwargs: object):
    return detect_objects(
        ranges,
        ANGLE_MIN,
        ANGLE_INCREMENT,
        RANGE_MIN,
        RANGE_MAX,
        params=params,
        **kwargs,  # type: ignore
    )


def test_same_result_as_single_steps():
    # Without region of interest the result is the one of the steps before
    ranges = noisy_scene()
    params = NO_ROI

    result = run(ranges, params)

    points = scan_to_points(
        ranges, ANGLE_MIN, ANGLE_INCREMENT, params.min_range, params.max_range
    )
    objects = filter_objects(
        extract_objects(points, cluster_points(points, params.eps, params.min_samples)),
        params.max_object_diameter,
    )
    boxes = [
        is_box(
            obj.points,
            ANGLE_INCREMENT,
            line_tolerance=params.box_line_tolerance,
            line_tolerance_per_m=params.box_line_tolerance_per_m,
            angle_tolerance=params.box_angle_tolerance,
            corner_tolerance_factor=params.box_corner_tolerance_factor,
            min_size=params.box_min_size,
            min_fill_ratio=params.box_min_fill_ratio,
        )
        for obj in objects
    ]
    np.testing.assert_array_equal(result.points, points)
    assert result.objects == objects
    assert result.boxes == boxes


def test_scene_is_classified():
    # The wall (4 m long) is filtered, the box and the cylinder remain
    result = run(noisy_scene(), NO_ROI)

    by_x = sorted(
        zip(result.objects, result.boxes, strict=True),
        key=lambda pair: pair[0].centroid_x,
    )
    assert len(by_x) == 2
    assert by_x[0][0].centroid_x < 0  # cylinder
    assert by_x[0][1] is False
    assert by_x[1][0].centroid_x > 0  # box
    assert by_x[1][1] is True


@pytest.mark.parametrize(
    ("range_min", "range_max", "expected"),
    [
        # The parameters limit the range of the sensor
        (0.15, 12.0, [1.0, 2.0]),
        # The sensor limits the parameters
        (1.5, 1.8, []),
        (0.15, 1.5, [1.0]),
    ],
)
def test_range_is_intersection_of_sensor_and_params(
    range_min: float, range_max: float, expected: list[float]
):
    ranges = np.full(NUM_BEAMS, np.inf)
    ranges[[0, 10, 20]] = [0.5, 1.0, 2.0]  # three beams far apart
    params = PipelineParams(
        min_range=0.8, max_range=3.0, min_samples=1, roi_enabled=False
    )

    result = detect_objects(
        ranges, ANGLE_MIN, ANGLE_INCREMENT, range_min, range_max, params=params
    )

    np.testing.assert_allclose(np.linalg.norm(result.points, axis=1), expected)


@pytest.mark.parametrize(
    "ranges", [np.array([]), np.full(NUM_BEAMS, np.inf), np.full(NUM_BEAMS, np.nan)]
)
def test_no_valid_points(ranges: np.ndarray):
    result = run(ranges, PipelineParams())

    assert result.points.shape == (0, 2)
    assert result.objects == []
    assert result.boxes == []


def test_step_timer_encloses_every_step():
    calls = []

    class Recorder:
        def __init__(self, name: str):
            self.name = name

        def __enter__(self):
            calls.append(("enter", self.name))

        def __exit__(self, *args: object):
            calls.append(("exit", self.name))

    run(noisy_scene(), PipelineParams(), step_timer=Recorder)

    expected = [(event, name) for name in STEPS for event in ("enter", "exit")]
    assert calls == expected


@pytest.mark.parametrize(
    ("point", "inside"),
    [
        ((0.0, -1.0), True),  # 1 m ahead
        ((0.0, -3.0), True),  # front border (inclusive)
        ((0.0, -3.1), False),  # too far ahead
        ((0.0, 1.0), True),  # 1 m behind
        ((0.0, 1.5), True),  # rear border (inclusive)
        ((0.0, 2.0), False),  # too far behind
        ((1.5, -1.0), True),  # 1.5 m to the left
        ((1.7, -1.0), False),  # too far to the left
        ((-0.8, 1.0), True),  # right border (inclusive), behind
        ((-1.0, -1.0), False),  # too far to the right
    ],
)
def test_roi_mask(point: tuple[float, float], inside: bool):
    mask = roi_mask(np.array([point]), 90.0, **ROI)

    assert mask.tolist() == [inside]


@pytest.mark.parametrize("laser_yaw", [0.0, 90.0, 180.0, -90.0])
def test_roi_mask_rotation(laser_yaw: float):
    # Points around a vehicle point 2 m ahead and 1 m left (inside), and 2 m ahead
    # and 1 m right (outside), expressed in a laser frame rotated by laser_yaw
    vehicle = np.array([[2.0, 1.0], [2.0, -1.0]])
    yaw = np.radians(laser_yaw)
    c, s = np.cos(yaw), np.sin(yaw)
    # Inverse rotation (row vectors times R equals R transposed times columns)
    laser = vehicle @ np.array([[c, -s], [s, c]])

    mask = roi_mask(laser, laser_yaw, **ROI)

    assert mask.tolist() == [True, False]


def test_roi_mask_empty():
    assert roi_mask(np.empty((0, 2)), 90.0, **ROI).shape == (0,)


def test_scene_with_roi():
    # Boxes in the laser frame: forward is -y, left is +x
    segments = [
        *rectangle((0.0, -2.0), 0.2, 0.2, 0.0),  # 2 m ahead: kept
        *rectangle((0.3, 1.0), 0.2, 0.2, 0.0),  # 1 m behind, left: kept
        *rectangle((-2.0, -1.0), 0.2, 0.2, 0.0),  # 2 m to the right: dropped
        *rectangle((0.5, -3.6), 0.2, 0.2, 0.0),  # 3.6 m ahead: dropped
    ]
    ranges = scan_ranges(segments)

    result = run(ranges, PipelineParams())

    # The points are not filtered, only the objects
    np.testing.assert_array_equal(result.points, run(ranges, NO_ROI).points)
    # Centroids of the visible sides facing the sensor
    centroids = sorted((obj.centroid_x, obj.centroid_y) for obj in result.objects)
    assert len(centroids) == 2
    np.testing.assert_allclose(centroids, [(0.0, -1.9), (0.3, 0.9)], atol=0.03)
    # Without region of interest all four boxes are found
    assert len(run(ranges, NO_ROI).objects) == 4


def vehicle_to_laser(x: float, y: float) -> np.ndarray:
    """Return a point of the vehicle frame in the laser frame (laser_yaw 90°)."""
    return np.array([y, -x])


def test_barrier_crossing_the_border_is_no_box():
    # Straight barrier right of the track, slightly diagonal: from 1 m ahead and
    # 0.72 m right to 3 m ahead and 1 m right. It crosses the right border
    # (0.8 m) 1.57 m ahead, the part inside is 0.57 m long.
    barrier = [(vehicle_to_laser(1.0, -0.72), vehicle_to_laser(3.0, -1.0))]
    ranges = scan_ranges(barrier)
    params = PipelineParams()

    result = run(ranges, params)

    # The whole barrier (2 m) is one object, dropped by the wall filter
    assert result.objects == []
    assert run(ranges, NO_ROI).objects == []
    # Clipping the points first would leave a short straight piece, a box
    points = result.points
    inside = roi_mask(
        points,
        params.laser_yaw,
        x_min=params.roi_x_min,
        x_max=params.roi_x_max,
        y_min=params.roi_y_min,
        y_max=params.roi_y_max,
    )
    piece = extract_objects(points[inside], np.zeros(inside.sum(), dtype=int))[0]
    assert piece.diameter < params.max_object_diameter
    assert is_box(
        piece.points,
        ANGLE_INCREMENT,
        line_tolerance=params.box_line_tolerance,
        line_tolerance_per_m=params.box_line_tolerance_per_m,
        angle_tolerance=params.box_angle_tolerance,
        corner_tolerance_factor=params.box_corner_tolerance_factor,
        min_size=params.box_min_size,
        min_fill_ratio=params.box_min_fill_ratio,
    )


def test_object_is_filtered_by_its_centroid():
    # Two boxes 0.3 x 0.3 m, the centroid is the one of the visible contour. The
    # first one is at 0.7 m right (inside) with points beyond the right border
    # (0.8 m), the second one at ~1.75 m left (outside, border at 1.6 m).
    segments = [
        *rectangle(tuple(vehicle_to_laser(1.5, -0.7)), 0.3, 0.3, 0.0),
        *rectangle(tuple(vehicle_to_laser(2.0, 1.9)), 0.3, 0.3, 0.0),
    ]
    ranges = scan_ranges(segments)

    result = run(ranges, PipelineParams())
    complete = run(ranges, NO_ROI)

    assert len(complete.objects) == 2
    assert len(result.objects) == 1
    # The kept object is complete, including its points outside the corridor
    kept = next(obj for obj in complete.objects if obj.centroid_x < 0)  # right
    assert result.objects[0] == kept
    assert kept.min_x < -0.8  # laser x = vehicle y
    np.testing.assert_array_equal(result.objects[0].points, kept.points)


def test_defaults_match_config():
    config = yaml.safe_load(CONFIG.read_text())
    ros_params = config["object_detection_lidar_node"]["ros__parameters"]

    defaults = asdict(PipelineParams())

    assert {f.name for f in fields(PipelineParams)} <= ros_params.keys()
    assert defaults == {name: ros_params[name] for name in defaults}
    # ROS declares the parameter type from the value, e.g. 1 instead of 1.0 in the
    # yaml would make a double parameter an integer one
    for f in fields(PipelineParams):
        assert type(ros_params[f.name]) is f.type, f.name
