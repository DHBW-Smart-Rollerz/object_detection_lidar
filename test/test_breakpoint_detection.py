from dataclasses import replace

import numpy as np
import pytest
from object_detection_lidar.breakpoint_detection import (
    breakpoint_distance,
    segment_scan,
    split_clusters,
)
from object_detection_lidar.pipeline import PipelineParams, detect_objects
from test_box_classification import add_noise, circle, rectangle
from test_pipeline import (
    ANGLE_INCREMENT,
    ANGLE_MIN,
    NUM_BEAMS,
    RANGE_MAX,
    RANGE_MIN,
    scan_ranges,
)

LAMBDA = np.radians(10.0)
SIGMA_R = 0.01
# The round objects (cylinders) of the scenes are kept, only the segmentation
# is tested here
NO_ROI = PipelineParams(roi_enabled=False, drop_round_objects=False)
NO_SPLIT = replace(NO_ROI, breakpoint_enabled=False)
# Segmentation by breakpoints only (no DBSCAN)
ABD = replace(NO_ROI, segmentation="abd")
# Beam layout of the RPLIDAR A2M12 as in recorded scans: the
# scan does not close the circle exactly, the last beam is ~1° before the first
REAL_ANGLE_MIN = -3.1241390705108643
REAL_ANGLE_INCREMENT = 0.0034828970674425364


def noisy(ranges: np.ndarray, angle_min: float, angle_increment: float, seed: int):
    """Add range noise typical of recorded scans to the valid beams."""
    ranges = ranges.copy()
    valid = np.isfinite(ranges)
    angles = angle_min + np.flatnonzero(valid) * angle_increment
    points = ranges[valid, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    ranges[valid] = np.linalg.norm(add_noise(points, seed), axis=1)
    return ranges


def detect(
    segments: list[tuple[np.ndarray, np.ndarray]],
    params: PipelineParams = NO_ROI,
    angle_min: float = ANGLE_MIN,
    angle_increment: float = ANGLE_INCREMENT,
    seed: int = 0,
):
    ranges = scan_ranges(segments, angle_min, angle_increment, NUM_BEAMS)
    ranges = noisy(ranges, angle_min, angle_increment, seed)
    return detect_objects(
        ranges, angle_min, angle_increment, RANGE_MIN, RANGE_MAX, params=params
    ).objects


def polar(distance: float, bearing_deg: float) -> tuple[float, float]:
    bearing = np.radians(bearing_deg)
    return distance * np.cos(bearing), distance * np.sin(bearing)


def background() -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a closed room of 10 x 10 m (walls beyond max_range, not clustered)."""
    corners = np.array([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]])
    return [(corners[i], corners[(i + 1) % 4]) for i in range(4)]


def split(points: np.ndarray, indices: np.ndarray, **kwargs: object) -> np.ndarray:
    params: dict = {
        "num_beams": NUM_BEAMS,
        "angle_increment": ANGLE_INCREMENT,
        "lambda_angle": LAMBDA,
        "sigma_r": SIGMA_R,
        "max_beam_gap": 3,
        "min_samples": 3,
    } | kwargs
    return split_clusters(points, indices, np.zeros(len(points), dtype=int), **params)


def test_breakpoint_distance():
    # Borges and Aldon: r * sin(dphi) / sin(lambda - dphi) + 3 * sigma_r
    dphi = np.radians(0.2)
    expected = 2.0 * np.sin(dphi) / np.sin(LAMBDA - dphi) + 3 * SIGMA_R

    distance = breakpoint_distance(np.array([2.0]), np.array([dphi]), LAMBDA, SIGMA_R)

    np.testing.assert_allclose(distance, [expected])


def test_breakpoint_distance_grows_with_range_and_gap():
    distance = breakpoint_distance(
        np.array([1.0, 2.0, 1.0]), np.radians([0.2, 0.2, 0.4]), LAMBDA, SIGMA_R
    )

    assert distance[1] > distance[0]
    assert distance[2] > distance[0]


def test_breakpoint_distance_beyond_lambda_is_always_a_breakpoint():
    distance = breakpoint_distance(
        np.array([1.0, 1.0]), np.array([LAMBDA, 2 * LAMBDA]), LAMBDA, SIGMA_R
    )

    np.testing.assert_array_equal(distance, [-np.inf, -np.inf])


@pytest.mark.parametrize("gap", [0.03, 0.05, 0.08])
def test_two_objects_with_background_in_the_gap_are_split(gap: float):
    # Two boxes 14 cm wide side by side at 1.3 m, closer than eps: the beams in
    # the gap hit the far background
    width = 0.14
    offset = (width + gap) / 2
    segments = [
        *rectangle((1.3, offset), width, 0.1, 0.0),
        *rectangle((1.3, -offset), width, 0.1, 0.0),
        *background(),
    ]

    merged = detect(segments, NO_SPLIT)
    objects = detect(segments)

    assert len(merged) == 1
    assert len(objects) == 2
    centers = sorted(obj.centroid_y for obj in objects)
    np.testing.assert_allclose(centers, [-offset, offset], atol=0.01)


@pytest.mark.parametrize(("sigma_r", "num_objects"), [(0.01, 2), (0.05, 1)])
def test_depth_jump_splits_with_a_tight_range_error(sigma_r: float, num_objects: int):
    # A box partly hidden behind a closer box (overlapping bearings, no gap of
    # beams): the range jumps by 8 cm between two neighboring beams, less than eps.
    # Only a tight sigma_r splits it (breakpoint distance ~5 cm at 1 m), the
    # default (3 * sigma_r = 15 cm) keeps jumps below eps together
    segments = [
        *rectangle((1.0, 0.08), 0.16, 0.1, 0.0),
        *rectangle((1.08, -0.06), 0.2, 0.1, 0.0),
    ]
    params = replace(NO_ROI, breakpoint_sigma_r=sigma_r)

    assert len(detect(segments, NO_SPLIT)) == 1
    assert len(detect(segments, params)) == num_objects


def test_range_artifact_on_a_flat_face_is_not_split():
    # Flat box of 0.3 m at 0.9 m with a typical sensor artifact: on 20 beams in
    # the middle of the face the range is off by up to +6 cm and then -7 cm, with a
    # jump of 13 cm between two neighboring beams
    segments = [*rectangle((0.94, 0.0), 0.3, 0.08, 0.0), *background()]
    ranges = noisy(scan_ranges(segments), ANGLE_MIN, ANGLE_INCREMENT, 0)
    hit = np.flatnonzero(np.isfinite(ranges) & (ranges < 2.0))
    middle = hit[len(hit) // 2]
    ranges[middle - 10 : middle] += np.linspace(0.01, 0.06, 10)
    ranges[middle : middle + 10] -= np.linspace(0.07, 0.01, 10)

    result = detect_objects(
        ranges, ANGLE_MIN, ANGLE_INCREMENT, RANGE_MIN, RANGE_MAX, params=NO_ROI
    )

    assert len(result.objects) == 1
    assert result.objects[0].width == pytest.approx(0.3, abs=0.02)


@pytest.mark.parametrize("distance", [0.5, 1.0, 2.0, 3.0, 3.8])
@pytest.mark.parametrize("bearing", [-150.0, -60.0, 0.0, 45.0, 120.0])
@pytest.mark.parametrize(
    "shape",
    [
        lambda c: rectangle(c, 0.3, 0.3, 0.0),  # one side visible
        lambda c: rectangle(c, 0.3, 0.2, np.radians(35)),  # corner visible
        lambda c: rectangle(c, 0.4, 0.1, np.radians(60)),  # oblique flat box
        lambda c: circle(c, 0.07),  # round obstacle
    ],
    ids=["box", "corner", "oblique", "round"],
)
def test_single_object_is_not_split(distance: float, bearing: float, shape):
    segments = [*shape(polar(distance, bearing)), *background()]

    objects = detect(segments, seed=int(distance * 10 + bearing + 360))

    assert len(objects) == 1


@pytest.mark.parametrize("incidence", [20.0, 30.0, 60.0, 90.0])
def test_oblique_wall_is_not_shattered(incidence: float):
    # A 1.5 m wall piece centered 2 m away at 30°, seen under the given angle of
    # incidence at its center (the incidence varies along the wall): it must stay
    # in one piece, so the wall filter (max_object_diameter) still drops it
    center = np.array(polar(2.0, 30.0))
    direction = np.radians(30.0 + 180.0 - incidence)
    along = np.array([np.cos(direction), np.sin(direction)])
    start, end = center - 0.75 * along, center + 0.75 * along
    ranges = noisy(scan_ranges([(start, end)]), ANGLE_MIN, ANGLE_INCREMENT, 0)
    valid = np.isfinite(ranges) & (ranges <= 4.0)
    indices = np.flatnonzero(valid)
    angles = ANGLE_MIN + indices * ANGLE_INCREMENT
    points = ranges[valid, None] * np.column_stack((np.cos(angles), np.sin(angles)))

    labels = split(points, indices)

    np.testing.assert_array_equal(labels, 0)


@pytest.mark.parametrize(
    ("angle_min", "angle_increment"),
    [
        (ANGLE_MIN, ANGLE_INCREMENT),  # seam exactly at ±pi
        (REAL_ANGLE_MIN, REAL_ANGLE_INCREMENT),  # seam with ~1° without beams
        (0.0, ANGLE_INCREMENT),  # seam at 0, object at ±pi in the middle
    ],
)
def test_object_across_the_seam_is_not_split(angle_min: float, angle_increment: float):
    segments = [*rectangle((-1.5, 0.0), 0.3, 0.2, 0.0), *background()]

    objects = detect(segments, angle_min=angle_min, angle_increment=angle_increment)

    assert len(objects) == 1
    assert objects[0].centroid_x == pytest.approx(-1.4, abs=0.01)
    assert objects[0].centroid_y == pytest.approx(0.0, abs=0.01)
    assert objects[0].width == pytest.approx(0.3, abs=0.02)


def test_objects_on_both_sides_of_the_seam_are_split():
    # Gap of 5 cm with background at ±pi between two boxes
    segments = [
        *rectangle((-1.3, 0.095), 0.14, 0.1, 0.0),
        *rectangle((-1.3, -0.095), 0.14, 0.1, 0.0),
        *background(),
    ]

    objects = detect(
        segments, angle_min=REAL_ANGLE_MIN, angle_increment=REAL_ANGLE_INCREMENT
    )

    assert len(objects) == 2
    assert sorted(obj.centroid_y > 0 for obj in objects) == [False, True]


def test_dropout_on_an_object_is_bridged():
    # A box at 0.35 m with a dark stripe: 11 beams in its middle return nothing
    segments = [*rectangle(polar(0.4, -30.0), 0.2, 0.1, np.radians(-30.0))]
    ranges = noisy(scan_ranges(segments), ANGLE_MIN, ANGLE_INCREMENT, 0)
    hit = np.flatnonzero(np.isfinite(ranges))
    middle = hit[len(hit) // 2]
    ranges[middle - 5 : middle + 6] = np.inf
    counted = replace(NO_ROI, breakpoint_count_invalid=True)

    def count(params: PipelineParams) -> int:
        result = detect_objects(
            ranges, ANGLE_MIN, ANGLE_INCREMENT, RANGE_MIN, RANGE_MAX, params=params
        )
        return len(result.objects)

    assert count(NO_ROI) == 1
    assert count(counted) == 2


def test_disabled_keeps_the_dbscan_clusters():
    segments = [
        *rectangle((1.3, 0.095), 0.14, 0.1, 0.0),
        *rectangle((1.3, -0.095), 0.14, 0.1, 0.0),
        *background(),
    ]

    assert len(detect(segments, NO_SPLIT)) == 1


@pytest.mark.parametrize(("missing", "segments"), [(3, 1), (4, 2)])
def test_invalid_beams_up_to_max_beam_gap_are_bridged(missing: int, segments: int):
    # 10 points, then `missing` invalid beams, then 10 points of the same surface
    indices = np.concatenate((np.arange(10), np.arange(10 + missing, 20 + missing)))
    angles = ANGLE_MIN + indices * ANGLE_INCREMENT
    points = np.column_stack((np.cos(angles), np.sin(angles)))  # circle, r = 1 m

    labels = split(points, indices)

    assert len(np.unique(labels)) == segments
    assert (labels >= 0).all()


@pytest.mark.parametrize(("returns", "segments"), [(False, 1), (True, 2)])
def test_only_beams_with_a_return_are_foreign(returns: bool, segments: int):
    # 10 points, 6 beams between them with (background) or without a return
    indices = np.concatenate((np.arange(10), np.arange(16, 26)))
    angles = ANGLE_MIN + indices * ANGLE_INCREMENT
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    has_return = np.zeros(NUM_BEAMS, dtype=bool)
    has_return[indices] = True
    has_return[10:16] = returns

    labels = split(points, indices, returns=has_return)

    assert len(np.unique(labels)) == segments


def test_returns_across_the_seam():
    # Gap of 6 beams with returns across the seam (beams 1797-1799 and 0-2)
    indices = np.concatenate((np.arange(1787, 1797), np.arange(3, 13)))
    angles = ANGLE_MIN + indices * ANGLE_INCREMENT
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    has_return = np.ones(NUM_BEAMS, dtype=bool)

    labels = split(points, indices, returns=has_return)

    np.testing.assert_array_equal(labels, [1] * 10 + [0] * 10)


def test_short_segments_become_noise():
    # A 2-point piece behind a gap is dropped like DBSCAN noise (min_samples = 3)
    indices = np.concatenate((np.arange(10), [20, 21]))
    angles = ANGLE_MIN + indices * ANGLE_INCREMENT
    points = np.column_stack((np.cos(angles), np.sin(angles)))

    labels = split(points, indices)

    np.testing.assert_array_equal(labels[:10], 0)
    np.testing.assert_array_equal(labels[10:], -1)


def test_noise_and_labels_are_kept():
    # Noise stays noise, the clusters are renumbered in the order of the labels
    angles = ANGLE_MIN + np.arange(30) * ANGLE_INCREMENT
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    labels = np.array([5] * 10 + [-1] * 10 + [2] * 10)

    new_labels = split_clusters(
        points,
        np.arange(30),
        labels,
        num_beams=NUM_BEAMS,
        angle_increment=ANGLE_INCREMENT,
        lambda_angle=LAMBDA,
        sigma_r=SIGMA_R,
        max_beam_gap=3,
        min_samples=3,
    )

    np.testing.assert_array_equal(new_labels, [1] * 10 + [-1] * 10 + [0] * 10)


def test_cluster_across_the_seam_keeps_its_own_label():
    # Cluster 0 crosses the seam (beams 1795-1799 and 0-4), cluster 1 follows it
    indices = np.concatenate((np.arange(5), np.arange(1795, 1800), np.arange(100, 110)))
    angles = ANGLE_MIN + indices * ANGLE_INCREMENT
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    labels = np.array([0] * 10 + [1] * 10)

    new_labels = split_clusters(
        points,
        indices,
        labels,
        num_beams=NUM_BEAMS,
        angle_increment=ANGLE_INCREMENT,
        lambda_angle=LAMBDA,
        sigma_r=SIGMA_R,
        max_beam_gap=3,
        min_samples=3,
    )

    np.testing.assert_array_equal(new_labels, labels)


def test_empty_input():
    labels = split(np.empty((0, 2)), np.empty(0, dtype=int))

    assert labels.shape == (0,)


def segment(points: np.ndarray, indices: np.ndarray, **kwargs: object) -> np.ndarray:
    params: dict = {
        "num_beams": NUM_BEAMS,
        "angle_increment": ANGLE_INCREMENT,
        "lambda_angle": LAMBDA,
        "sigma_r": SIGMA_R,
        "max_beam_gap": 3,
        "min_samples": 3,
    } | kwargs
    return segment_scan(points, indices, **params)


def unit_circle(indices: np.ndarray) -> np.ndarray:
    angles = ANGLE_MIN + indices * ANGLE_INCREMENT
    return np.column_stack((np.cos(angles), np.sin(angles)))


@pytest.mark.parametrize("gap", [0.03, 0.05, 0.08])
def test_abd_two_objects_with_background_in_the_gap_are_split(gap: float):
    width = 0.14
    offset = (width + gap) / 2
    segments = [
        *rectangle((1.3, offset), width, 0.1, 0.0),
        *rectangle((1.3, -offset), width, 0.1, 0.0),
        *background(),
    ]

    objects = detect(segments, ABD)

    assert len(objects) == 2
    centers = sorted(obj.centroid_y for obj in objects)
    np.testing.assert_allclose(centers, [-offset, offset], atol=0.01)


@pytest.mark.parametrize("distance", [0.5, 1.0, 2.0, 3.8])
@pytest.mark.parametrize("bearing", [-150.0, 0.0, 120.0])
@pytest.mark.parametrize(
    "shape",
    [
        lambda c: rectangle(c, 0.3, 0.3, 0.0),  # one side visible
        lambda c: rectangle(c, 0.3, 0.2, np.radians(35)),  # corner visible
        lambda c: rectangle(c, 0.4, 0.1, np.radians(60)),  # oblique flat box
        lambda c: circle(c, 0.07),  # round obstacle
    ],
    ids=["box", "corner", "oblique", "round"],
)
def test_abd_single_object_is_not_split(distance: float, bearing: float, shape):
    segments = [*shape(polar(distance, bearing)), *background()]

    objects = detect(segments, ABD, seed=int(distance * 10 + bearing + 360))

    assert len(objects) == 1


@pytest.mark.parametrize("incidence", [20.0, 30.0, 60.0, 90.0])
def test_abd_oblique_wall_is_not_shattered(incidence: float):
    # Same wall as for split_clusters: one segment, the wall filter drops it
    center = np.array(polar(2.0, 30.0))
    direction = np.radians(30.0 + 180.0 - incidence)
    along = np.array([np.cos(direction), np.sin(direction)])
    segments = [(center - 0.75 * along, center + 0.75 * along), *background()]
    ranges = noisy(scan_ranges(segments), ANGLE_MIN, ANGLE_INCREMENT, 0)
    valid = np.isfinite(ranges) & (ranges <= 4.0)
    indices = np.flatnonzero(valid)
    angles = ANGLE_MIN + indices * ANGLE_INCREMENT
    points = ranges[valid, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    returns = np.isfinite(ranges)

    labels = segment(points, indices, returns=returns)

    np.testing.assert_array_equal(labels, 0)
    assert detect(segments, ABD) == []


def test_abd_closed_room_is_one_segment():
    # A room within max_range is one closed contour without any breakpoint
    corners = np.array([[-1.5, -1.5], [1.5, -1.5], [1.5, 1.5], [-1.5, 1.5]])
    segments = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    ranges = scan_ranges(segments)
    indices = np.arange(NUM_BEAMS)
    points = ranges[:, None] * unit_circle(indices)

    labels = segment(points, indices, sigma_r=0.05)

    np.testing.assert_array_equal(labels, 0)


def test_abd_box_in_front_of_a_wall_is_split():
    # A box 0.5 m in front of a wall within max_range: two jumps in depth
    segments = [
        *rectangle((1.0, 0.0), 0.2, 0.1, 0.0),
        (np.array([1.5, -0.2]), np.array([1.5, 0.2])),
    ]

    objects = detect(segments, ABD)

    assert len(objects) == 3
    box = min(objects, key=lambda obj: obj.centroid_x)
    assert box.width == pytest.approx(0.2, abs=0.02)


@pytest.mark.parametrize(
    ("angle_min", "angle_increment"),
    [
        (ANGLE_MIN, ANGLE_INCREMENT),
        (REAL_ANGLE_MIN, REAL_ANGLE_INCREMENT),
        (0.0, ANGLE_INCREMENT),
    ],
)
def test_abd_object_across_the_seam_is_not_split(
    angle_min: float, angle_increment: float
):
    segments = [*rectangle((-1.5, 0.0), 0.3, 0.2, 0.0), *background()]

    objects = detect(
        segments, ABD, angle_min=angle_min, angle_increment=angle_increment
    )

    assert len(objects) == 1
    assert objects[0].centroid_x == pytest.approx(-1.4, abs=0.01)
    assert objects[0].centroid_y == pytest.approx(0.0, abs=0.01)
    assert objects[0].width == pytest.approx(0.3, abs=0.02)


def test_abd_objects_on_both_sides_of_the_seam_are_split():
    segments = [
        *rectangle((-1.3, 0.095), 0.14, 0.1, 0.0),
        *rectangle((-1.3, -0.095), 0.14, 0.1, 0.0),
        *background(),
    ]

    objects = detect(
        segments, ABD, angle_min=REAL_ANGLE_MIN, angle_increment=REAL_ANGLE_INCREMENT
    )

    assert len(objects) == 2
    assert sorted(obj.centroid_y > 0 for obj in objects) == [False, True]


def test_abd_segment_across_the_seam_keeps_one_label():
    # Beams 1795-1799 and 0-4 are neighbors, beams 100-109 are a second segment
    indices = np.concatenate((np.arange(5), np.arange(100, 110), np.arange(1795, 1800)))
    labels = segment(unit_circle(indices), indices, returns=np.ones(NUM_BEAMS, bool))

    np.testing.assert_array_equal(labels, [1] * 5 + [0] * 10 + [1] * 5)


def test_abd_noise_points_are_dropped():
    # Single stray points and pairs (fewer than min_samples) between the beams of
    # a surface at 1 m are noise. Unlike DBSCAN + split_clusters (stray points
    # are DBSCAN noise there and bridged as foreign beams), each of them splits
    # the surface: in the walk they are two jumps in depth
    indices = np.arange(200, 300)
    points = unit_circle(indices)
    stray = np.array([220, 250, 251, 280])
    points[stray - 200] *= 3.0  # 3 m, far behind the surface
    returns = np.ones(NUM_BEAMS, dtype=bool)

    labels = segment(points, indices, returns=returns, sigma_r=0.05)

    np.testing.assert_array_equal(labels[stray - 200], -1)
    surface = np.delete(labels, stray - 200)
    assert len(np.unique(surface)) == 4  # split at each stray group
    assert (surface >= 0).all()


def test_abd_dropout_on_an_object_is_bridged():
    segments = [*rectangle(polar(0.4, -30.0), 0.2, 0.1, np.radians(-30.0))]
    ranges = noisy(scan_ranges(segments), ANGLE_MIN, ANGLE_INCREMENT, 0)
    hit = np.flatnonzero(np.isfinite(ranges))
    middle = hit[len(hit) // 2]
    ranges[middle - 5 : middle + 6] = np.inf

    result = detect_objects(
        ranges, ANGLE_MIN, ANGLE_INCREMENT, RANGE_MIN, RANGE_MAX, params=ABD
    )

    assert len(result.objects) == 1


def test_abd_without_breakpoint_is_one_segment():
    indices = np.arange(NUM_BEAMS)

    labels = segment(unit_circle(indices), indices)

    np.testing.assert_array_equal(labels, 0)


def test_abd_empty_and_single_point():
    assert segment(np.empty((0, 2)), np.empty(0, dtype=int)).shape == (0,)
    np.testing.assert_array_equal(
        segment(unit_circle(np.array([5])), np.array([5]), min_samples=1), [0]
    )


def test_invalid_segmentation_is_rejected():
    with pytest.raises(ValueError, match="segmentation"):
        PipelineParams(segmentation="kmeans")
