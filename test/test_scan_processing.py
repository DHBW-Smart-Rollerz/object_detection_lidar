import math

import numpy as np
import pytest
from object_detection_lidar.scan_processing import scan_to_points


def test_points_on_axes():
    # Beams at 0, 90, 180 and 270 degrees, each 1 m away
    points = scan_to_points(
        np.ones(4),
        angle_min=0.0,
        angle_increment=math.pi / 2,
        range_min=0.1,
        range_max=10.0,
    )

    expected = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    np.testing.assert_allclose(points, expected, atol=1e-9)


def test_invalid_ranges_are_dropped():
    ranges = np.array([math.inf, math.nan, 0.05, 1.0, 20.0])
    points = scan_to_points(
        ranges,
        angle_min=0.0,
        angle_increment=0.0,
        range_min=0.1,
        range_max=10.0,
    )

    np.testing.assert_allclose(points, [[1.0, 0.0]])


def test_angle_is_kept_for_remaining_points():
    # The first beam is invalid, the second one has to keep its own angle
    points = scan_to_points(
        np.array([math.inf, 2.0]),
        angle_min=-math.pi / 2,
        angle_increment=math.pi / 2,
        range_min=0.1,
        range_max=10.0,
    )

    np.testing.assert_allclose(points, [[2.0, 0.0]], atol=1e-9)


@pytest.mark.parametrize("ranges", [np.array([]), np.full(3, math.inf)])
def test_no_valid_points(ranges: np.ndarray):
    points = scan_to_points(
        ranges,
        angle_min=0.0,
        angle_increment=0.1,
        range_min=0.1,
        range_max=10.0,
    )

    assert points.shape == (0, 2)
