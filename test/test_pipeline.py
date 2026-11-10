from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from object_detection_lidar.box_classification import is_box
from object_detection_lidar.breakpoint_detection import segment_scan, split_clusters
from object_detection_lidar.clustering import (
    DetectedObject,
    cluster_points,
    extract_objects,
    filter_objects,
)
from object_detection_lidar.pipeline import (
    STEPS,
    PipelineParams,
    camera_format,
    detect_objects,
    roi_mask,
    to_vehicle_frame,
    vehicle_centroids,
    vehicle_endpoints,
    vehicle_extents,
)
from object_detection_lidar.scan_processing import scan_to_points, valid_beams
from test_box_classification import add_noise, circle, rectangle

CONFIG = Path(__file__).resolve().parents[1] / "config" / "ros_params.yaml"

# Beam layout of the RPLIDAR A2M12 (typical values from recorded scans)
NUM_BEAMS = 1800
ANGLE_MIN = -np.pi  # in rad
ANGLE_INCREMENT = 2 * np.pi / NUM_BEAMS  # in rad
RANGE_MIN = 0.15  # in m, of the sensor
RANGE_MAX = 12.0  # in m, of the sensor
# Behavior without region of interest (all valid points are clustered) and
# without dropping round objects (the scenes contain cylinders)
NO_ROI = PipelineParams(roi_enabled=False, drop_round_objects=False)
# Defaults with the optional region of interest switched on
WITH_ROI = PipelineParams(roi_enabled=True)
# Default corridor of the node, in the laser frame forward is -y and left is +x
# Values per object in the camera format [cls, BLx, BLy, BRx, BRy, score]
CAMERA_FORMAT_SIZE = 6

ROI = {"x_min": -1.5, "x_max": 3.0, "y_min": -0.8, "y_max": 1.6}


def scan_ranges(
    segments: list[tuple[np.ndarray, np.ndarray]],
    angle_min: float = ANGLE_MIN,
    angle_increment: float = ANGLE_INCREMENT,
    num_beams: int = NUM_BEAMS,
) -> np.ndarray:
    """Return the range of every beam to the first hit segment (inf for no hit)."""
    angles = angle_min + np.arange(num_beams) * angle_increment
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    ranges = np.full(num_beams, np.inf)
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
    labels = split_clusters(
        points,
        np.flatnonzero(valid_beams(ranges, params.min_range, params.max_range)),
        cluster_points(points, params.eps, params.min_samples),
        num_beams=NUM_BEAMS,
        angle_increment=ANGLE_INCREMENT,
        lambda_angle=np.radians(params.breakpoint_lambda_deg),
        sigma_r=params.breakpoint_sigma_r,
        max_beam_gap=params.breakpoint_max_beam_gap,
        min_samples=params.min_samples,
    )
    objects = filter_objects(
        extract_objects(points, labels), params.max_object_diameter
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


def test_abd_same_result_as_segment_scan():
    # Segmentation "abd": segment_scan on all points replaces DBSCAN
    ranges = noisy_scene()
    params = replace(NO_ROI, segmentation="abd")

    result = run(ranges, params)

    points = scan_to_points(
        ranges, ANGLE_MIN, ANGLE_INCREMENT, params.min_range, params.max_range
    )
    labels = segment_scan(
        points,
        np.flatnonzero(valid_beams(ranges, params.min_range, params.max_range)),
        num_beams=NUM_BEAMS,
        angle_increment=ANGLE_INCREMENT,
        lambda_angle=np.radians(params.breakpoint_lambda_deg),
        sigma_r=params.breakpoint_sigma_r,
        max_beam_gap=params.breakpoint_max_beam_gap,
        min_samples=params.min_samples,
        returns=valid_beams(ranges, RANGE_MIN, RANGE_MAX),
    )
    objects = filter_objects(
        extract_objects(points, labels), params.max_object_diameter
    )
    assert result.objects == objects
    assert len(objects) == 2  # box and cylinder, the wall is filtered


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


def test_round_object_is_dropped():
    # Same scene as test_scene_is_classified: the cylinder (0.2 m) is dropped
    ranges = noisy_scene()

    kept = run(ranges, NO_ROI)
    result = run(ranges, replace(NO_ROI, drop_round_objects=True))

    assert len(result.objects) == 1
    assert result.objects[0].centroid_x > 0  # box
    assert result.boxes == [True]
    assert len(result.round_objects) == 1
    cylinder = next(obj for obj in kept.objects if obj.centroid_x < 0)
    assert result.round_objects[0] == cylinder
    # camera format and objects topic only see the kept objects
    assert camera_format(result.objects, NO_ROI).size == 6


def test_no_round_objects_without_dropping():
    assert run(noisy_scene(), NO_ROI).round_objects == []


def test_box_is_never_dropped_as_round(monkeypatch: pytest.MonkeyPatch):
    # Only objects that are no box are checked for a round shape
    checked = []

    def fake_is_round(points: np.ndarray, **_: object) -> bool:
        checked.append(len(points))
        return True

    monkeypatch.setattr("object_detection_lidar.pipeline.is_round", fake_is_round)
    result = run(noisy_scene(), replace(NO_ROI, drop_round_objects=True))

    assert result.boxes == [True]
    assert len(checked) == 1  # only the cylinder


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
        *rectangle((-0.3, 2.2), 0.2, 0.2, 0.0),  # 2.1 m behind: dropped
    ]
    ranges = scan_ranges(segments)

    result = run(ranges, WITH_ROI)

    # The points are not filtered, only the objects
    np.testing.assert_array_equal(result.points, run(ranges, NO_ROI).points)
    # Centroids of the visible sides facing the sensor
    centroids = sorted((obj.centroid_x, obj.centroid_y) for obj in result.objects)
    assert len(centroids) == 2
    np.testing.assert_allclose(centroids, [(0.0, -1.9), (0.3, 0.9)], atol=0.03)
    # Without region of interest all four boxes are found
    assert len(run(ranges, NO_ROI).objects) == 4


def test_roi_disabled_by_default():
    # Same scene: by default (roi_enabled false) all four boxes are kept, also
    # the ones to the side and behind; switched on, the corridor still filters
    segments = [
        *rectangle((0.0, -2.0), 0.2, 0.2, 0.0),  # 2 m ahead
        *rectangle((0.3, 1.0), 0.2, 0.2, 0.0),  # 1 m behind, left
        *rectangle((-2.0, -1.0), 0.2, 0.2, 0.0),  # 2 m to the right
        *rectangle((-0.3, 2.2), 0.2, 0.2, 0.0),  # 2.1 m behind
    ]
    ranges = scan_ranges(segments)

    assert PipelineParams().roi_enabled is False
    default = run(ranges, PipelineParams())
    assert len(default.objects) == 4
    assert len(run(ranges, WITH_ROI).objects) == 2


def vehicle_to_laser(x: float, y: float) -> np.ndarray:
    """Return a point of the vehicle frame in the laser frame (laser_yaw 90°)."""
    return np.array([y, -x])


def test_barrier_crossing_the_border_is_no_box():
    # Straight barrier right of the track, slightly diagonal: from 1 m ahead and
    # 0.72 m right to 3 m ahead and 1 m right. It crosses the right border
    # (0.8 m) 1.57 m ahead, the part inside is 0.57 m long.
    barrier = [(vehicle_to_laser(1.0, -0.72), vehicle_to_laser(3.0, -1.0))]
    ranges = scan_ranges(barrier)
    params = WITH_ROI

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

    result = run(ranges, WITH_ROI)
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


@pytest.mark.parametrize(
    ("laser", "vehicle"),
    [
        ((0.0, -1.0), (1.0, 0.0)),  # laser -y is forward
        ((1.0, 0.0), (0.0, 1.0)),  # laser +x is left
        ((0.0, 0.0), (0.0, 0.0)),  # LiDAR origin
    ],
)
def test_to_vehicle_frame_rotation(
    laser: tuple[float, float], vehicle: tuple[float, float]
):
    result = to_vehicle_frame(np.array([laser]), 90.0, 0.0, 0.0)

    np.testing.assert_allclose(result, [vehicle], atol=1e-12)


def test_to_vehicle_frame_offset_after_rotation():
    # LiDAR 0.2 m behind the front axle and 0.05 m to the right
    result = to_vehicle_frame(np.array([[0.0, -1.0], [0.0, 0.0]]), 90.0, -0.2, -0.05)

    np.testing.assert_allclose(result, [[0.8, -0.05], [-0.2, -0.05]], atol=1e-12)


def test_to_vehicle_frame_empty():
    assert to_vehicle_frame(np.empty((0, 2)), 90.0, 0.1, 0.2).shape == (0, 2)


def test_roi_mask_uses_offset():
    # 3.9 m ahead of the LiDAR: inside, but outside once the LiDAR is 0.2 m ahead
    # of the vehicle origin
    point = np.array([[0.0, -3.9]])
    roi = {**ROI, "x_max": 4.0}

    assert roi_mask(point, 90.0, **roi).tolist() == [True]
    assert roi_mask(point, 90.0, **roi, offset_x=0.2).tolist() == [False]


def line_object(x: float, y_right: float, y_left: float) -> DetectedObject:
    """Return the object of a line across x from y_right to y_left (laser frame)."""
    ys = np.linspace(y_right, y_left, 11)
    points = np.column_stack((np.full_like(ys, x), ys))
    (obj,) = extract_objects(points, np.zeros(len(points), dtype=int))
    return obj


def test_vehicle_endpoints():
    # Laser frame: x = 1 m from y = -0.2 to 0.3, i.e. in the vehicle frame (laser
    # x is left, laser y is backward) a line 1 m to the left from 0.2 m ahead
    # (bearing in the laser frame is smaller) to 0.3 m behind
    obj = line_object(1.0, -0.2, 0.3)
    params = PipelineParams(lidar_offset_x=0.1, lidar_offset_y=0.0)

    endpoints = vehicle_endpoints([obj], params)

    # left = largest bearing in the laser frame = laser (1.0, 0.3)
    np.testing.assert_allclose(endpoints, [[-0.2, 1.0, 0.3, 1.0]], atol=1e-12)


def test_vehicle_centroids():
    # Line 1 m ahead of the LiDAR (laser -y), from 0.2 m right to 0.3 m left
    # (laser x), the LiDAR sits 0.1 m ahead of the front axle
    xs = np.linspace(-0.2, 0.3, 11)
    (obj,) = extract_objects(
        np.column_stack((xs, np.full_like(xs, -1.0))), np.zeros(11, dtype=int)
    )
    params = PipelineParams(lidar_offset_x=0.1, lidar_offset_y=0.0)

    centroids = vehicle_centroids([obj], params)

    np.testing.assert_allclose(centroids, [[1.1, 0.05]], atol=1e-12)
    assert centroids[0, 0] > 0  # ahead of the vehicle


def test_vehicle_extents():
    # Line along laser y (0.5 m long), i.e. along vehicle x for laser_yaw 90°
    obj = line_object(1.0, -0.2, 0.3)
    assert (obj.extent_x, obj.extent_y) == pytest.approx((0.0, 0.5))

    extents = vehicle_extents([obj], PipelineParams(lidar_offset_x=0.1))

    np.testing.assert_allclose(extents, [[0.5, 0.0]], atol=1e-12)


def test_vehicle_frame_empty():
    params = PipelineParams()

    assert vehicle_centroids([], params).shape == (0, 2)
    assert vehicle_extents([], params).shape == (0, 2)
    assert vehicle_endpoints([], params).shape == (0, 4)


def test_camera_format_layout():
    # Two lines ahead of the vehicle (forward is laser -y, left is laser +x)
    ys = np.full(11, -1.0)
    xs = np.linspace(-0.1, 0.15, 11)  # from 0.1 m right to 0.15 m left
    near = extract_objects(np.column_stack((xs, ys)), np.zeros(11, dtype=int))[0]
    far = extract_objects(
        np.column_stack((np.linspace(0.3, 0.5, 11), np.full(11, -2.0))),
        np.zeros(11, dtype=int),
    )[0]
    params = PipelineParams(lidar_offset_x=-0.2, lidar_offset_y=0.0)

    data = camera_format([near, far], params)

    assert data.dtype == np.float32
    assert data.shape == (2 * CAMERA_FORMAT_SIZE,)
    # [cls, BLx, BLy, BRx, BRy, score] in mm, BL = left = larger y
    np.testing.assert_array_equal(
        data.reshape(-1, CAMERA_FORMAT_SIZE),
        [
            [-1.0, 800.0, 150.0, 800.0, -100.0, 1.0],
            [-1.0, 1800.0, 500.0, 1800.0, 300.0, 1.0],
        ],
    )


def test_camera_format_empty():
    data = camera_format([], PipelineParams())

    assert data.dtype == np.float32
    assert data.shape == (0,)
