import numpy as np
import pytest
from object_detection_lidar.box_classification import (
    fit_line,
    is_box,
    is_straight,
    order_by_angle,
    split_at_corner,
)

# Beam spacing of the RPLIDAR A2M12 (1800 beams per turn)
ANGLE_INCREMENT = 2 * np.pi / 1800
# Default parameters of the node
PARAMS = {
    "line_tolerance": 0.0023,
    "line_tolerance_per_m": 0.0015,
    "angle_tolerance": 10.0,
    "corner_tolerance_factor": 2.5,
    "min_size": 0.08,
    "min_fill_ratio": 0.7,
}


def raycast(segments: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Return the points where the beams of a 360° scan hit the segments first.

    Args:
        segments: Line segments (start, end) with x/y coordinates in m

    Returns:
        Array of shape (N, 2) with the hit points, one per beam that hits.
    """
    angles = np.arange(1800) * ANGLE_INCREMENT - np.pi
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    ranges = np.full(len(angles), np.inf)
    for start, end in segments:
        edge = end - start
        # Solve t * direction = start + s * edge for t (range) and s (on segment)
        denominator = directions[:, 1] * edge[0] - directions[:, 0] * edge[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (start[1] * edge[0] - start[0] * edge[1]) / denominator
            s = (directions[:, 0] * start[1] - directions[:, 1] * start[0]) / (
                denominator
            )
        hit = (t > 0) & (s >= 0) & (s <= 1)
        ranges[hit] = np.minimum(ranges[hit], t[hit])
    valid = np.isfinite(ranges)
    return directions[valid] * ranges[valid, None]


def rectangle(
    center: tuple[float, float], width: float, length: float, yaw: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the four sides of a rectangle rotated by yaw (in rad)."""
    corners = np.array(
        [
            [-length / 2, -width / 2],
            [length / 2, -width / 2],
            [length / 2, width / 2],
            [-length / 2, width / 2],
        ]
    )
    c, s = np.cos(yaw), np.sin(yaw)
    corners = corners @ np.array([[c, s], [-s, c]]) + center
    return [(corners[i], corners[(i + 1) % 4]) for i in range(4)]


def circle(
    center: tuple[float, float], radius: float, num: int = 200
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a circle approximated by num segments."""
    angles = np.linspace(0.0, 2 * np.pi, num + 1)
    points = np.column_stack(
        (center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles))
    )
    return [(points[i], points[i + 1]) for i in range(num)]


def add_noise(points: np.ndarray, seed: int = 0) -> np.ndarray:
    """Add range noise like in the rosbags (0.5 mm + 1.5 mm per m distance)."""
    ranges = np.linalg.norm(points, axis=1)
    sigma = 0.0005 + 0.0015 * ranges
    noisy = ranges + np.random.default_rng(seed).normal(0.0, 1.0, len(ranges)) * sigma
    return points * (noisy / ranges)[:, None]


def classify(points: np.ndarray) -> bool:
    return is_box(points, ANGLE_INCREMENT, **PARAMS)


def test_fit_line_of_exact_line():
    points = np.column_stack((np.linspace(0.0, 1.0, 11), np.linspace(0.0, 1.0, 11)))

    direction, rms = fit_line(points)

    assert abs(direction @ np.array([1.0, 1.0]) / np.sqrt(2)) == pytest.approx(1.0)
    assert rms == pytest.approx(0.0, abs=1e-12)


def test_fit_line_rms():
    # Points alternate +-1 mm around the x axis
    points = np.column_stack((np.linspace(0.0, 1.0, 20), np.tile([0.001, -0.001], 10)))

    _, rms = fit_line(points)

    assert rms == pytest.approx(0.001, rel=0.05)


def test_order_by_angle_across_seam():
    # Points behind the sensor, the beam angles jump from +pi to -pi in between
    angles = np.linspace(np.pi - 0.1, np.pi + 0.1, 21)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    shuffled = points[np.random.default_rng(0).permutation(len(points))]

    ordered, relative_angles = order_by_angle(shuffled)

    assert np.all(np.diff(relative_angles) > 0)
    # Ordered along the contour: neighbors are one beam apart
    steps = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
    np.testing.assert_allclose(steps, steps[0], rtol=1e-6)


def test_is_straight_rejects_arc_with_small_rms():
    # Flat arc: the RMS distance to the line is below the tolerance, the bulge not
    angles = np.linspace(-0.3, 0.3, 50)
    radius = 0.5
    points = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    _, rms = fit_line(points)
    tolerance = 0.007

    straight, _ = is_straight(points, tolerance)

    assert rms < tolerance
    assert not straight


def test_split_at_corner():
    first = np.column_stack((np.linspace(0.0, 0.2, 21), np.zeros(21)))
    second = np.column_stack((np.full(20, 0.2), np.linspace(0.01, 0.2, 20)))
    points = np.vstack((first, second)) + np.array([1.0, 0.0])

    sides = split_at_corner(points)

    assert sides is not None
    for side in sides:
        assert fit_line(side)[1] == pytest.approx(0.0, abs=1e-9)
    # The points within EDGE_MARGIN around the corner are dropped
    corner = np.array([1.2, 0.0])
    assert min(np.linalg.norm(np.vstack(sides) - corner, axis=1)) > 0.02


@pytest.mark.parametrize("distance", [1.0, 2.0, 3.0, 3.8])
@pytest.mark.parametrize("yaw", [0.0, 20.0, 45.0])
def test_box_is_box(distance: float, yaw: float):
    points = add_noise(raycast(rectangle((distance, 0.0), 0.3, 0.3, np.radians(yaw))))

    assert classify(points)


@pytest.mark.parametrize("distance", [1.0, 2.0, 3.0, 3.8])
def test_pedestrian_box_is_box(distance: float):
    # Smallest obstacle of the CAuDri: 100 x 150 mm pedestrian box, frontal
    points = add_noise(raycast(rectangle((distance, 0.0), 0.15, 0.10, 0.0)))

    assert classify(points)


def test_box_across_seam_is_box():
    points = add_noise(raycast(rectangle((-1.5, 0.0), 0.3, 0.3, np.radians(30))))

    assert classify(points)


@pytest.mark.parametrize("radius", [0.05, 0.1, 0.2, 0.3])
@pytest.mark.parametrize("distance", [1.0, 2.0, 3.0])
def test_cylinder_is_no_box(radius: float, distance: float):
    points = add_noise(raycast(circle((distance, 0.0), radius)))

    assert not classify(points)


@pytest.mark.parametrize("apex_angle", [60.0, 120.0])
def test_non_right_corner_is_no_box(apex_angle: float):
    # Wedge with its apex towards the sensor, only the two flanks are visible
    half = np.radians(apex_angle / 2)
    apex = np.array([1.5, 0.0])
    depth = 0.15 / np.tan(half)
    left = apex + np.array([depth, 0.15])
    right = apex + np.array([depth, -0.15])
    points = add_noise(raycast([(apex, left), (left, right), (right, apex)]))

    assert not classify(points)


def test_small_fragment_is_no_box():
    # 5 cm long straight fragment, below min_size
    points = raycast([(np.array([1.0, -0.025]), np.array([1.0, 0.025]))])

    assert not classify(points)


def test_sparse_cluster_is_no_box():
    # Straight, but only every third beam hits (e.g. a fence), fill ratio ~0.33
    points = raycast([(np.array([1.0, -0.15]), np.array([1.0, 0.15]))])[::3]

    assert not classify(points)


def test_too_few_points_is_no_box():
    points = raycast([(np.array([1.0, -0.15]), np.array([1.0, 0.15]))])[:5]

    assert not classify(points)
