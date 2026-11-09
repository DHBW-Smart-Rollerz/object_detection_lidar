"""Metrics for the offline evaluation of the object detection (independent of ROS).

Used by tools/evaluate.py, which reads the scans from rosbags and runs them
through detect_objects. All positions are in m in the scan frame.
"""

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from object_detection_lidar.pipeline import PipelineResult


@dataclass(frozen=True)
class ScanObjects:
    """Centroids and classes of the objects of one scan.

    Attributes:
        centroids: Array of shape (M, 2) with the object centroids in m
        boxes: Array of shape (M,), True for boxes
    """

    centroids: np.ndarray
    boxes: np.ndarray

    @classmethod
    def from_result(cls, result: PipelineResult) -> "ScanObjects":
        """Take the centroids and classes from a pipeline result."""
        centroids = np.array(
            [(obj.centroid_x, obj.centroid_y) for obj in result.objects],
            dtype=np.float64,
        ).reshape(-1, 2)
        return cls(centroids=centroids, boxes=np.array(result.boxes, dtype=bool))


@dataclass(frozen=True)
class Stats:
    """Median, 95th percentile and maximum of a series of values."""

    median: float
    p95: float
    max: float

    @classmethod
    def of(cls, values: Sequence[float] | np.ndarray) -> "Stats":
        """Compute the statistics, NaN for an empty series."""
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return cls(np.nan, np.nan, np.nan)
        return cls(
            median=float(np.median(values)),
            p95=float(np.percentile(values, 95)),
            max=float(values.max()),
        )


# Steps whose runtimes are added up as "clustering" (as measured in the review of
# 06/11/2026, where these steps were timed together)
CLUSTERING_STEPS = ("cluster_points", "extract_objects", "filter_objects")


def timing_stats(step_times: Sequence[Mapping[str, float]]) -> dict[str, Stats]:
    """Compute the runtime statistics per step.

    Args:
        step_times: Runtime per step in s, one mapping per scan. The step
            "total" is the runtime of the whole detection.

    Returns:
        Statistics in ms per step, plus "clustering" (sum of CLUSTERING_STEPS) if
        all of these steps were measured.
    """
    names = list(dict.fromkeys(name for times in step_times for name in times))
    stats = {
        name: Stats.of([1e3 * times[name] for times in step_times if name in times])
        for name in names
    }
    if all(name in names for name in CLUSTERING_STEPS):
        stats["clustering"] = Stats.of(
            [
                1e3 * sum(times[name] for name in CLUSTERING_STEPS)
                for times in step_times
            ]
        )
    return stats


def count_stats(scans: Sequence[ScanObjects]) -> tuple[Stats, Stats, float, float]:
    """Count the objects and boxes per scan.

    Args:
        scans: The objects of each scan

    Returns:
        Statistics of the objects per scan, of the boxes per scan and the mean
        number of objects and boxes per scan.
    """
    objects = np.array([len(scan.boxes) for scan in scans], dtype=np.float64)
    boxes = np.array([scan.boxes.sum() for scan in scans], dtype=np.float64)
    mean_objects = float(objects.mean()) if objects.size else np.nan
    mean_boxes = float(boxes.mean()) if boxes.size else np.nan
    return Stats.of(objects), Stats.of(boxes), mean_objects, mean_boxes


@dataclass(frozen=True)
class FlickerResult:
    """Class changes of objects found again in the next scan."""

    pairs: int  # objects found again in the next scan
    flips: int  # of those, objects whose class changed

    @property
    def rate(self) -> float:
        """Return the share of flips among the pairs (NaN without pairs)."""
        return self.flips / self.pairs if self.pairs else np.nan


def flicker(scans: Sequence[ScanObjects], radius: float) -> FlickerResult:
    """Count how often the class of an object changes from one scan to the next.

    Every object is assigned to the nearest object of the previous scan. It counts
    as found again if its centroid moved by less than radius. No ego motion
    compensation is done, so with a moving vehicle mostly slow relative motion
    (far or static objects) is found again.

    Args:
        scans: The objects of consecutive scans
        radius: Max. shift of the centroid between two scans in m

    Returns:
        The number of objects found again and of class changes among them.
    """
    pairs = flips = 0
    for previous, current in itertools.pairwise(scans):
        if len(previous.boxes) == 0:
            continue
        for centroid, box in zip(current.centroids, current.boxes, strict=True):
            distances = np.linalg.norm(previous.centroids - centroid, axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] >= radius:
                continue
            pairs += 1
            flips += int(bool(box) != bool(previous.boxes[nearest]))
    return FlickerResult(pairs=pairs, flips=flips)


@dataclass(frozen=True)
class TrackScatter:
    """Scatter of one object over the scans of a static phase."""

    distance: float  # in m, distance of the mean centroid from the sensor
    scatter: float  # in m, norm of the standard deviations of x and y
    box_ratio: float  # share of the scans with the object classified as box
    num_scans: int  # scans in which the object was found


def static_scatter(
    scans: Sequence[ScanObjects], radius: float = 0.08, min_presence: float = 0.8
) -> list[TrackScatter]:
    """Measure the centroid scatter of objects while the vehicle is standing.

    The objects are followed greedily: each object joins the track whose mean
    centroid is nearest, if it is closer than radius, otherwise it starts a new
    track. Only tracks found in at least min_presence of the scans are kept. The
    scatter is the precision of the centroid, not its absolute accuracy.

    Args:
        scans: The objects of consecutive scans of one static phase
        radius: Max. distance of an object to the mean of its track in m
        min_presence: Min. share of the scans an object has to be found in

    Returns:
        The scatter of each kept track, ordered by distance.
    """
    tracks: list[list[tuple[float, float, bool]]] = []
    means: list[np.ndarray] = []
    for scan in scans:
        for centroid, box in zip(scan.centroids, scan.boxes, strict=True):
            distances = [np.linalg.norm(mean - centroid) for mean in means]
            best = int(np.argmin(distances)) if distances else -1
            if best >= 0 and distances[best] < radius:
                tracks[best].append((centroid[0], centroid[1], bool(box)))
                means[best] = np.mean([p[:2] for p in tracks[best]], axis=0)
            else:
                tracks.append([(centroid[0], centroid[1], bool(box))])
                means.append(np.array(centroid, dtype=np.float64))

    result = []
    for track in tracks:
        if len(track) < min_presence * len(scans):
            continue
        xy = np.array([p[:2] for p in track])
        result.append(
            TrackScatter(
                distance=float(np.linalg.norm(xy.mean(axis=0))),
                scatter=float(np.linalg.norm(xy.std(axis=0))),
                box_ratio=float(np.mean([p[2] for p in track])),
                num_scans=len(track),
            )
        )
    return sorted(result, key=lambda track: track.distance)


@dataclass(frozen=True)
class Region:
    """Circular sector of the scan frame in which all clusters were labelled."""

    heading: float  # in deg, direction of the center of the sector
    half_angle: float  # in deg, half opening angle
    max_range: float  # in m

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Return for each point of shape (N, 2) whether it lies in the sector."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        bearing = np.degrees(np.arctan2(points[:, 1], points[:, 0])) - self.heading
        bearing = (bearing + 180.0) % 360.0 - 180.0
        return (np.abs(bearing) < self.half_angle) & (
            np.linalg.norm(points, axis=1) <= self.max_range
        )


@dataclass(frozen=True)
class LabelledScan:
    """Hand labels of one scan."""

    scan: int  # index of the scan in the bag (0 = first scan of the topic)
    boxes: np.ndarray  # array of shape (K, 2), centroids of the box clusters in m
    missed_boxes: int = 0  # boxes in the region without a cluster


@dataclass(frozen=True)
class Labels:
    """Hand labels of one bag (format see README)."""

    region: Region
    scans: list[LabelledScan]
    # Scan index ranges [start, end) with the vehicle standing
    static_scans: list[tuple[int, int]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping) -> "Labels":
        """Build the labels from the parsed label file."""
        return cls(
            region=Region(**data["region"]),
            scans=[
                LabelledScan(
                    scan=int(scan["scan"]),
                    boxes=np.array(scan.get("boxes") or [], dtype=np.float64).reshape(
                        -1, 2
                    ),
                    missed_boxes=int(scan.get("missed_boxes", 0)),
                )
                for scan in data.get("scans") or []
            ],
            static_scans=[
                (int(start), int(end)) for start, end in data.get("static_scans") or []
            ],
        )


@dataclass(frozen=True)
class LabelScore:
    """Result of the box classification compared to the hand labels."""

    true_positives: int  # labelled boxes classified as box
    false_negatives: int  # labelled boxes with a cluster, not classified as box
    missed: int  # labelled boxes without a cluster (labelled or not matched)
    false_positives: int  # boxes in the region matching no labelled box

    @property
    def precision(self) -> float:
        """Return the share of correct boxes among all boxes in the region."""
        found = self.true_positives + self.false_positives
        return self.true_positives / found if found else np.nan

    @property
    def recall(self) -> float:
        """Return the share of box clusters classified as box."""
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total else np.nan

    @property
    def recall_total(self) -> float:
        """Return the share of all labelled boxes classified as box."""
        total = self.true_positives + self.false_negatives + self.missed
        return self.true_positives / total if total else np.nan


def match_nearest(
    first: np.ndarray, second: np.ndarray, radius: float
) -> list[tuple[int, int]]:
    """Match two point sets one to one, greedily by ascending distance.

    Args:
        first: Array of shape (N, 2) in m
        second: Array of shape (M, 2) in m
        radius: Max. distance of a matched pair in m

    Returns:
        Index pairs (index in first, index in second) of the matched points.
    """
    if len(first) == 0 or len(second) == 0:
        return []
    distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    pairs = []
    used_first: set[int] = set()
    used_second: set[int] = set()
    for flat in np.argsort(distances, axis=None):
        i, j = np.unravel_index(flat, distances.shape)
        if distances[i, j] >= radius:
            break
        if i in used_first or j in used_second:
            continue
        pairs.append((int(i), int(j)))
        used_first.add(int(i))
        used_second.add(int(j))
    return pairs


def score_labels(
    scans: Sequence[ScanObjects],
    labels: Labels,
    match_radius: float,
    field_of_view: Region | None = None,
) -> LabelScore:
    """Compare the box classification with the hand labels.

    The labelled boxes of a scan are matched one to one to the objects in the
    labelled region. A matched object classified as box is a true positive,
    otherwise a false negative. Labelled boxes without a matching object count as
    missed (like the boxes labelled without a cluster). Every box in the region
    that matches no labelled box is a false positive.

    With field_of_view, the region is the intersection of the labelled region and
    the field of view: detected objects and labelled boxes outside of it are
    ignored. Labelled boxes without a cluster have no position and always count.

    Args:
        scans: The objects of all scans of the bag
        labels: The hand labels of the bag
        match_radius: Max. distance between a labelled and a detected centroid in m
        field_of_view: Additional restriction of the region, e.g. the field of
            view of the camera the labels were made with

    Returns:
        The counts of the comparison.
    """
    tp = fn = missed = fp = 0
    for labelled in labels.scans:
        if labelled.scan >= len(scans):
            raise ValueError(f"Labelled scan {labelled.scan} is not in the bag")
        scan = scans[labelled.scan]
        in_region = labels.region.contains(scan.centroids)
        labelled_boxes = labelled.boxes
        if field_of_view is not None:
            in_region &= field_of_view.contains(scan.centroids)
            labelled_boxes = labelled_boxes[field_of_view.contains(labelled_boxes)]
        centroids = scan.centroids[in_region]
        boxes = scan.boxes[in_region]

        pairs = match_nearest(labelled_boxes, centroids, match_radius)
        matched = {j for _, j in pairs}
        hits = sum(bool(boxes[j]) for _, j in pairs)
        tp += hits
        fn += len(pairs) - hits
        missed += labelled.missed_boxes + len(labelled_boxes) - len(pairs)
        fp += sum(bool(box) for j, box in enumerate(boxes) if j not in matched)
    return LabelScore(
        true_positives=tp, false_negatives=fn, missed=missed, false_positives=fp
    )
