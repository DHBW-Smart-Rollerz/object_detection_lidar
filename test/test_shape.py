import numpy as np
import pytest
from object_detection_lidar.pipeline import PipelineParams
from object_detection_lidar.shape import arc_angle, best_corner, fit_circle, is_round
from test_box_classification import circle, raycast, rectangle

DEFAULTS = PipelineParams()
# Parameters of the node
PARAMS = {
    "min_points": DEFAULTS.round_min_points,
    "max_radius": DEFAULTS.round_max_radius,
    "min_arc_deg": DEFAULTS.round_min_arc_deg,
    "noise": DEFAULTS.box_line_tolerance,
    "noise_per_m": DEFAULTS.box_line_tolerance_per_m,
    "max_circle_rms_factor": DEFAULTS.round_max_circle_rms_factor,
    "min_line_ratio": DEFAULTS.round_min_line_ratio,
    "min_corner_ratio": DEFAULTS.round_min_corner_ratio,
    "right_angle_tolerance": DEFAULTS.round_right_angle_tolerance,
    "min_right_corner_ratio": DEFAULTS.round_min_right_corner_ratio,
}
SEEDS = range(10)


def scan(
    segments: list[tuple[np.ndarray, np.ndarray]], seed: int, noise_factor: float = 1.0
) -> np.ndarray:
    """Raycast the segments and add range noise (0.5 mm + 1.5 mm/m, rosbags)."""
    points = raycast(segments)
    ranges = np.linalg.norm(points, axis=1)
    sigma = noise_factor * (0.0005 + 0.0015 * ranges)
    noisy = ranges + np.random.default_rng(seed).normal(0.0, 1.0, len(ranges)) * sigma
    return points * (noisy / ranges)[:, None]


def cylinder(diameter: float, distance: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a cylinder at the distance, slightly off the x axis."""
    bearing = 0.3
    return circle(
        (distance * np.cos(bearing), distance * np.sin(bearing)), diameter / 2
    )


def classify(points: np.ndarray) -> bool:
    return is_round(points, **PARAMS)


def test_fit_circle_exact():
    angles = np.linspace(0.0, np.pi, 20)
    points = np.column_stack((0.05 * np.cos(angles) + 1.0, 0.05 * np.sin(angles)))

    center, radius, rms = fit_circle(points)

    np.testing.assert_allclose(center, [1.0, 0.0], atol=1e-9)
    assert radius == pytest.approx(0.05)
    assert rms == pytest.approx(0.0, abs=1e-9)


def test_fit_circle_short_noisy_arc_is_nearly_unbiased():
    # 90° arc with 1 mm noise, the fit of Kasa would shrink the radius noticeably
    rng = np.random.default_rng(0)
    angles = np.linspace(0.0, np.pi / 2, 30)
    radius = 0.06 + rng.normal(0.0, 0.001, len(angles))
    points = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))

    _, fitted, rms = fit_circle(points)

    assert fitted == pytest.approx(0.06, rel=0.05)
    assert rms == pytest.approx(0.001, rel=0.3)


def test_fit_circle_of_collinear_points():
    points = np.column_stack((np.linspace(0.0, 1.0, 10), np.zeros(10)))

    _, radius, _ = fit_circle(points)

    assert radius > 1e6 or not np.isfinite(radius)


def test_best_corner_of_right_angle():
    first = np.column_stack((np.linspace(0.0, 0.2, 21), np.zeros(21)))
    second = np.column_stack((np.full(20, 0.2), np.linspace(0.01, 0.2, 20)))

    rms, angle = best_corner(np.vstack((first, second)))

    assert rms == pytest.approx(0.0, abs=1e-9)
    assert angle == pytest.approx(90.0)


def test_best_corner_too_few_points():
    assert best_corner(np.zeros((4, 2))) == (np.inf, 0.0)


def test_arc_angle_across_seam():
    # Arc around the -x direction of the center, the angles jump from +pi to -pi
    angles = np.linspace(np.pi - 1.0, np.pi + 1.0, 21)
    points = np.column_stack((np.cos(angles), np.sin(angles)))

    assert arc_angle(points, np.zeros(2)) == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("diameter", "distance"),
    [
        (0.05, 0.5),
        (0.08, 0.5),
        (0.08, 1.0),
        (0.1, 0.5),
        (0.1, 1.0),
        (0.1, 1.5),
        (0.15, 0.5),
        (0.15, 1.0),
        (0.15, 2.0),
        (0.15, 2.5),
    ],
)
def test_cylinder_is_round(diameter: float, distance: float):
    segments = cylinder(diameter, distance)

    assert all(classify(scan(segments, seed)) for seed in SEEDS)


@pytest.mark.parametrize(
    ("diameter", "distance"), [(0.05, 2.0), (0.1, 3.0), (0.1, 4.0), (0.15, 4.0)]
)
def test_far_cylinder_with_few_points_is_kept(diameter: float, distance: float):
    # Fewer than round_min_points points: never round (recall first)
    points = scan(cylinder(diameter, distance), 0)

    assert len(points) < PARAMS["min_points"]
    assert not classify(points)


@pytest.mark.parametrize("distance", [0.5, 1.0, 2.0, 3.0, 4.0])
@pytest.mark.parametrize("yaw", [0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 80.0])
@pytest.mark.parametrize(
    ("width", "length"),
    [(0.3, 0.4), (0.1, 0.2), (0.1, 0.1)],  # box, pedestrian figure, small box
)
def test_box_is_not_round(distance: float, yaw: float, width: float, length: float):
    # yaw 0: one side (plus a grazing second side, the box is off the axis),
    # 45: symmetric corner, others: oblique side and corner
    segments = rectangle((distance, 0.1), width, length, np.radians(yaw))

    assert not any(classify(scan(segments, seed)) for seed in SEEDS)


@pytest.mark.parametrize("distance", [1.0, 2.0, 3.0, 4.0])
@pytest.mark.parametrize("yaw", [0.0, 20.0, 45.0, 70.0])
def test_box_is_not_round_with_double_noise(distance: float, yaw: float):
    segments = rectangle((distance, 0.1), 0.1, 0.2, np.radians(yaw))

    assert not any(classify(scan(segments, seed, 2.0)) for seed in SEEDS)


@pytest.mark.parametrize(("diameter", "distance"), [(0.1, 0.5), (0.15, 1.0)])
def test_cylinder_is_round_with_double_noise(diameter: float, distance: float):
    segments = cylinder(diameter, distance)

    assert all(classify(scan(segments, seed, 2.0)) for seed in SEEDS)


def test_wall_is_not_round():
    segments = [(np.array([1.0, -0.3]), np.array([1.0, 0.3]))]

    assert not classify(scan(segments, 0))


def test_large_cylinder_is_not_round():
    # Radius above round_max_radius (e.g. a pillar or a large bin)
    assert not classify(scan(cylinder(0.5, 1.5), 0))


def test_concave_arc_is_not_round():
    # Inside of a curve (center in front of the contour, e.g. a curved barrier)
    angles = np.linspace(-0.6, 0.6, 40)
    points = np.column_stack((0.8 + 0.1 * np.cos(angles), 0.1 * np.sin(angles)))

    assert not classify(points)


def test_too_few_points_is_not_round():
    angles = np.linspace(0.0, np.pi, 7)
    points = np.column_stack((1.0 - 0.05 * np.sin(angles), 0.05 * np.cos(angles)))

    assert not classify(points)
