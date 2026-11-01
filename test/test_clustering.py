import numpy as np
import pytest
from object_detection_lidar.clustering import (
    cluster_points,
    extract_objects,
    filter_objects,
)

EPS = 0.10
MIN_SAMPLES = 3
OFFSET = np.array([1.0, 0.0])  # moves rotated test shapes away from the origin


def make_group(center_x: float, center_y: float, num: int = 5) -> np.ndarray:
    """Return num points on a 4 cm long line along y around the center."""
    ys = center_y + np.linspace(-0.02, 0.02, num)
    return np.column_stack((np.full(num, center_x), ys))


def test_two_separated_groups():
    points = np.vstack((make_group(1.0, 0.0), make_group(1.0, 1.0)))

    labels = cluster_points(points, EPS, MIN_SAMPLES)
    objects = extract_objects(points, labels)

    assert len(objects) == 2
    centroids = sorted((obj.centroid_x, obj.centroid_y) for obj in objects)
    np.testing.assert_allclose(centroids, [[1.0, 0.0], [1.0, 1.0]], atol=1e-9)
    assert [obj.num_points for obj in objects] == [5, 5]


def test_single_point_is_noise():
    points = np.vstack((make_group(1.0, 0.0), [[2.0, 2.0]]))

    labels = cluster_points(points, EPS, MIN_SAMPLES)
    objects = extract_objects(points, labels)

    assert labels[-1] == -1
    assert len(objects) == 1
    assert objects[0].num_points == 5


def test_empty_input():
    points = np.empty((0, 2))

    labels = cluster_points(points, EPS, MIN_SAMPLES)
    objects = extract_objects(points, labels)

    assert labels.shape == (0,)
    assert objects == []
    assert filter_objects(objects, 0.6) == []


def test_long_line_is_filtered():
    # 2 m long wall with 1 cm point spacing and a small obstacle next to it
    wall = np.column_stack((np.full(201, 2.0), np.linspace(-1.0, 1.0, 201)))
    points = np.vstack((wall, make_group(1.0, 0.0)))

    labels = cluster_points(points, EPS, MIN_SAMPLES)
    objects = extract_objects(points, labels)
    filtered = filter_objects(objects, 0.6)

    assert len(objects) == 2
    assert len(filtered) == 1
    np.testing.assert_allclose(
        (filtered[0].centroid_x, filtered[0].centroid_y), (1.0, 0.0), atol=1e-9
    )


@pytest.mark.parametrize(
    ("distance", "expected_labels"),
    [(EPS * 0.99, [0, 0]), (EPS * 1.01, [-1, -1])],
)
def test_distance_around_eps(distance: float, expected_labels: list[int]):
    points = np.array([[1.0, 0.0], [1.0, distance]])

    labels = cluster_points(points, EPS, min_samples=2)

    assert labels.tolist() == expected_labels


def rotate(points: np.ndarray, angle: float) -> np.ndarray:
    """Rotate the points around the origin by angle (in rad)."""
    c, s = np.cos(angle), np.sin(angle)
    return points @ np.array([[c, s], [-s, c]])


def test_rotated_line_is_filtered():
    # 0.8 m long wall piece under 45°, the bounding box is only 0.57 x 0.57 m
    line = np.column_stack((np.linspace(0.0, 0.8, 81), np.zeros(81)))
    points = rotate(line, np.pi / 4) + OFFSET

    objects = extract_objects(points, cluster_points(points, EPS, MIN_SAMPLES))

    assert len(objects) == 1
    assert objects[0].extent_x == pytest.approx(0.8 / np.sqrt(2))
    assert objects[0].extent_y == pytest.approx(0.8 / np.sqrt(2))
    assert objects[0].diameter == pytest.approx(0.8)
    assert filter_objects(objects, 0.6) == []


def test_rotated_square_is_kept():
    # Outline of a 0.3 x 0.3 m box under 45°, points every 1 cm
    side = np.linspace(0.0, 0.3, 31)
    outline = np.vstack(
        (
            np.column_stack((side, np.zeros(31))),
            np.column_stack((side, np.full(31, 0.3))),
            np.column_stack((np.zeros(31), side)),
            np.column_stack((np.full(31, 0.3), side)),
        )
    )
    points = rotate(outline, np.pi / 4) + OFFSET

    objects = extract_objects(points, cluster_points(points, EPS, MIN_SAMPLES))
    filtered = filter_objects(objects, 0.6)

    assert len(filtered) == 1
    assert filtered[0].diameter == pytest.approx(0.3 * np.sqrt(2))


def test_diameter_of_two_points():
    points = np.array([[1.0, 0.0], [1.3, 0.4]])

    objects = extract_objects(points, cluster_points(points, eps=0.6, min_samples=2))

    assert len(objects) == 1
    assert objects[0].diameter == pytest.approx(0.5)


def test_diameter_of_collinear_points():
    points = np.array([[1.0, 0.0], [1.03, 0.04], [1.06, 0.08]])

    objects = extract_objects(points, cluster_points(points, EPS, MIN_SAMPLES))

    assert len(objects) == 1
    assert objects[0].diameter == pytest.approx(0.1)


def test_single_point_cluster():
    # With min_samples = 1 every isolated point is its own cluster
    points = np.vstack((make_group(1.0, 0.0), [[2.0, 2.0]]))

    objects = extract_objects(points, cluster_points(points, EPS, min_samples=1))

    single = [obj for obj in objects if obj.num_points == 1]
    assert len(objects) == 2
    assert len(single) == 1
    assert single[0].diameter == 0.0
    assert filter_objects(objects, 0.6) == objects
