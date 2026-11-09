import numpy as np
import pytest
from object_detection_lidar.clustering import extract_objects
from object_detection_lidar.evaluation import (
    Labels,
    Region,
    ScanObjects,
    Stats,
    count_stats,
    flicker,
    match_nearest,
    score_labels,
    static_scatter,
    timing_stats,
)
from object_detection_lidar.pipeline import PipelineResult

# Region of the hand labels: forward is -y of the laser frame
FRONT = {"heading": -90.0, "half_angle": 75.0, "max_range": 4.0}


def scan(*objects: tuple[float, float, bool]) -> ScanObjects:
    """Return a scan with objects given as (x, y, is_box)."""
    return ScanObjects(
        centroids=np.array([obj[:2] for obj in objects], dtype=float).reshape(-1, 2),
        boxes=np.array([obj[2] for obj in objects], dtype=bool),
    )


def test_scan_objects_from_result():
    points = np.array([[1.0, 0.0], [1.0, 0.2], [2.0, 2.0], [2.0, 2.4]])
    objects = extract_objects(points, np.array([0, 0, 1, 1]))
    result = PipelineResult(points=points, objects=objects, boxes=[True, False])

    scan_objects = ScanObjects.from_result(result)

    np.testing.assert_allclose(scan_objects.centroids, [[1.0, 0.1], [2.0, 2.2]])
    assert scan_objects.boxes.tolist() == [True, False]


def test_scan_objects_from_empty_result():
    result = PipelineResult(points=np.empty((0, 2)), objects=[], boxes=[])

    scan_objects = ScanObjects.from_result(result)

    assert scan_objects.centroids.shape == (0, 2)
    assert scan_objects.boxes.shape == (0,)


def test_stats():
    stats = Stats.of(np.arange(101.0))

    assert stats == Stats(median=50.0, p95=95.0, max=100.0)
    assert np.isnan(Stats.of([]).median)


def test_timing_stats_in_ms_with_clustering_sum():
    times = [
        {
            "cluster_points": 0.001 * i,
            "extract_objects": 0.002,
            "filter_objects": 0.0,
            "total": 0.01,
        }
        for i in range(1, 4)
    ]

    stats = timing_stats(times)

    assert stats["total"].median == pytest.approx(10.0)
    assert stats["cluster_points"].max == pytest.approx(3.0)
    assert stats["clustering"].median == pytest.approx(4.0)


def test_timing_stats_without_clustering_steps():
    stats = timing_stats([{"total": 0.01}])

    assert set(stats) == {"total"}


def test_count_stats():
    scans = [scan((1, 0, True), (2, 0, False)), scan(), scan((1, 0, True))]

    objects, boxes, mean_objects, mean_boxes = count_stats(scans)

    assert objects.median == 1.0
    assert objects.max == 2.0
    assert boxes.max == 1.0
    assert mean_objects == pytest.approx(1.0)
    assert mean_boxes == pytest.approx(2 / 3)


def test_flicker_counts_class_changes_of_found_objects():
    scans = [
        scan((1.0, 0.0, True), (3.0, 0.0, False)),
        # first moved 2 cm and changed class, second moved 10 cm (not found again)
        scan((1.02, 0.0, False), (3.1, 0.0, False)),
        # first moved 1 cm, same class
        scan((1.03, 0.0, False)),
    ]

    result = flicker(scans, radius=0.05)

    assert result.pairs == 2
    assert result.flips == 1
    assert result.rate == pytest.approx(0.5)


def test_flicker_after_empty_scan():
    result = flicker([scan(), scan((1.0, 0.0, True))], radius=0.05)

    assert result.pairs == 0
    assert np.isnan(result.rate)


def test_static_scatter():
    # One object jitters +-1 cm along x, one appears only in 1 of 4 scans
    scans = [
        scan((1.01, 0.0, True)),
        scan((0.99, 0.0, False), (2.0, 2.0, False)),
        scan((1.01, 0.0, True)),
        scan((0.99, 0.0, True)),
    ]

    tracks = static_scatter(scans)

    assert len(tracks) == 1
    assert tracks[0].distance == pytest.approx(1.0)
    assert tracks[0].scatter == pytest.approx(0.01)
    assert tracks[0].box_ratio == pytest.approx(0.75)
    assert tracks[0].num_scans == 4


def test_region_contains_front_sector():
    region = Region(**FRONT)

    points = np.array(
        [
            [0.0, -1.0],  # straight ahead
            [-2.0, -0.6],  # 73° to the side
            [-2.0, -0.5],  # 76° to the side
            [0.0, 1.0],  # behind
            [0.0, -4.5],  # too far
        ]
    )

    assert region.contains(points).tolist() == [True, True, False, False, False]


def test_match_nearest_is_one_to_one():
    first = np.array([[0.0, 0.0], [0.05, 0.0]])
    second = np.array([[0.04, 0.0], [1.0, 0.0]])

    # Both points of first are closest to the same point, only the nearer one wins
    assert match_nearest(first, second, radius=0.1) == [(1, 0)]
    assert match_nearest(first, np.empty((0, 2)), radius=0.1) == []


def labels(*scans: dict) -> Labels:
    return Labels.from_dict({"region": FRONT, "scans": list(scans)})


def test_score_labels():
    scans = [
        scan(
            (0.0, -1.0, True),  # labelled box, found: TP
            (0.5, -2.0, False),  # labelled box, not classified: FN
            (-0.5, -2.0, True),  # box without label: FP
            (0.0, 1.0, True),  # box behind the vehicle, outside the region: ignored
        ),
        scan((0.0, -1.0, True)),  # not labelled: ignored
    ]
    labelled = labels(
        {
            "scan": 0,
            "boxes": [[0.02, -1.0], [0.5, -2.0], [1.0, -3.0]],  # last: no object
            "missed_boxes": 2,
        }
    )

    score = score_labels(scans, labelled, match_radius=0.1)

    assert score.true_positives == 1
    assert score.false_negatives == 1
    assert score.missed == 3
    assert score.false_positives == 1
    assert score.precision == pytest.approx(0.5)
    assert score.recall == pytest.approx(0.5)
    assert score.recall_total == pytest.approx(1 / 5)


def test_score_labels_match_radius():
    scans = [scan((0.0, -1.0, True))]
    labelled = labels({"scan": 0, "boxes": [[0.0, -1.08]]})

    near = score_labels(scans, labelled, match_radius=0.1)
    far = score_labels(scans, labelled, match_radius=0.05)

    assert (near.true_positives, near.false_positives, near.missed) == (1, 0, 0)
    assert (far.true_positives, far.false_positives, far.missed) == (0, 1, 1)


def test_score_labels_without_boxes():
    score = score_labels([scan((0.0, -1.0, False))], labels({"scan": 0}), 0.1)

    assert score.true_positives == score.false_positives == 0
    assert np.isnan(score.precision)
    assert np.isnan(score.recall)


def test_score_labels_scan_outside_bag():
    with pytest.raises(ValueError, match="not in the bag"):
        score_labels([scan()], labels({"scan": 5, "boxes": []}), 0.1)


def test_labels_from_dict():
    data = {
        "region": FRONT,
        "static_scans": [[0, 6], [10, 20]],
        "scans": [{"scan": 3, "boxes": [[1.0, 2.0]], "missed_boxes": 1}],
    }

    parsed = Labels.from_dict(data)

    assert parsed.region == Region(**FRONT)
    assert parsed.static_scans == [(0, 6), (10, 20)]
    assert parsed.scans[0].scan == 3
    np.testing.assert_allclose(parsed.scans[0].boxes, [[1.0, 2.0]])
    assert parsed.scans[0].missed_boxes == 1


# Camera field of view: +-52° around forward (-y of the laser frame)
CAMERA = Region(heading=-90.0, half_angle=52.0, max_range=np.inf)


def test_field_of_view_region():
    points = np.array(
        [
            [0.0, -1.0],  # straight ahead
            [1.2, -1.0],  # 50° to the left
            [-1.4, -1.0],  # 54° to the right
            [0.0, -10.0],  # far ahead, no range limit
        ]
    )

    assert CAMERA.contains(points).tolist() == [True, True, False, True]


def test_score_labels_in_field_of_view():
    scans = [
        scan(
            (0.0, -1.0, True),  # labelled box in view: TP
            (-1.0, -1.2, False),  # labelled box in view (40° right), not a box: FN
            (2.0, -1.0, True),  # labelled box at 63° left: outside of the view
            (-2.0, -1.5, True),  # box without label at 53° right: FP only without
        )
    ]
    labelled = labels(
        {
            "scan": 0,
            "boxes": [[0.0, -1.0], [-1.0, -1.2], [2.0, -1.0]],
            "missed_boxes": 1,
        }
    )

    full = score_labels(scans, labelled, match_radius=0.1)
    in_view = score_labels(scans, labelled, match_radius=0.1, field_of_view=CAMERA)

    assert (full.true_positives, full.false_negatives) == (2, 1)
    assert (full.missed, full.false_positives) == (1, 1)
    # The labelled box and the false box outside of the view are ignored, the box
    # labelled without a cluster has no position and still counts
    assert (in_view.true_positives, in_view.false_negatives) == (1, 1)
    assert (in_view.missed, in_view.false_positives) == (1, 0)


def test_field_of_view_does_not_widen_the_region():
    # The labelled region (+-75°, 4 m) still limits the range
    scans = [scan((0.0, -5.0, True))]
    labelled = labels({"scan": 0, "boxes": []})

    score = score_labels(scans, labelled, match_radius=0.1, field_of_view=CAMERA)

    assert score.false_positives == 0
