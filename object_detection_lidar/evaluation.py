"""Metrics for the offline evaluation of the object detection (independent of ROS).

Used by tools/evaluate.py, which reads the scans from rosbags and runs them
through detect_objects. All positions are in m in the scan frame.
"""

import itertools
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from object_detection_lidar.pipeline import PipelineResult


@dataclass(frozen=True)
class ScanObjects:
    """Centroids, classes and endpoints of the objects of one scan.

    Attributes:
        centroids: Array of shape (M, 2) with the object centroids in m
        boxes: Array of shape (M,), True for boxes
        endpoints: Array of shape (M, 4) with left_x, left_y, right_x, right_y of
            the visible contour in m
    """

    centroids: np.ndarray
    boxes: np.ndarray
    endpoints: np.ndarray

    @classmethod
    def from_result(cls, result: PipelineResult) -> "ScanObjects":
        """Take the centroids, classes and endpoints from a pipeline result."""
        centroids = np.array(
            [(obj.centroid_x, obj.centroid_y) for obj in result.objects],
            dtype=np.float64,
        ).reshape(-1, 2)
        endpoints = np.array(
            [
                (obj.left_x, obj.left_y, obj.right_x, obj.right_y)
                for obj in result.objects
            ],
            dtype=np.float64,
        ).reshape(-1, 4)
        return cls(
            centroids=centroids,
            boxes=np.array(result.boxes, dtype=bool),
            endpoints=endpoints,
        )


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


# Steps whose runtimes are added up as "segmentation": DBSCAN, breakpoint
# splitting (or ABD alone) and building and filtering the objects
SEGMENTATION_STEPS = (
    "cluster_points",
    "split_clusters",
    "extract_objects",
    "filter_objects",
)


def timing_stats(step_times: Sequence[Mapping[str, float]]) -> dict[str, Stats]:
    """Compute the runtime statistics per step.

    Args:
        step_times: Runtime per step in s, one mapping per scan. The step
            "total" is the runtime of the whole detection.

    Returns:
        Statistics in ms per step, plus "segmentation" (sum of SEGMENTATION_STEPS)
        if all of these steps were measured.
    """
    names = list(dict.fromkeys(name for times in step_times for name in times))
    stats = {
        name: Stats.of([1e3 * times[name] for times in step_times if name in times])
        for name in names
    }
    if all(name in names for name in SEGMENTATION_STEPS):
        stats["segmentation"] = Stats.of(
            [
                1e3 * sum(times[name] for name in SEGMENTATION_STEPS)
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
    width: float  # in m, median distance of the endpoints
    # in m, scatter of the endpoints: mean of the norms of the standard deviations
    # of the left and of the right endpoint
    endpoint_scatter: float


def static_scatter(
    scans: Sequence[ScanObjects], radius: float = 0.08, min_presence: float = 0.8
) -> list[TrackScatter]:
    """Measure the centroid scatter of objects while the vehicle is standing.

    The objects are followed greedily: each object joins the track whose mean
    centroid is nearest, if it is closer than radius, otherwise it starts a new
    track. Only tracks found in at least min_presence of the scans are kept. The
    scatter is the precision of the centroid, not its absolute accuracy. The same
    holds for the endpoints, whose scatter also contains the beam spacing (the
    outermost beam that hits the object can change from scan to scan).

    Args:
        scans: The objects of consecutive scans of one static phase
        radius: Max. distance of an object to the mean of its track in m
        min_presence: Min. share of the scans an object has to be found in

    Returns:
        The scatter of each kept track, ordered by distance.
    """
    # Per track: centroid x, y, is_box and the endpoints (4 values)
    tracks: list[list[tuple[float, float, bool, np.ndarray]]] = []
    means: list[np.ndarray] = []
    for scan in scans:
        for centroid, box, ends in zip(
            scan.centroids, scan.boxes, scan.endpoints, strict=True
        ):
            entry = (centroid[0], centroid[1], bool(box), np.asarray(ends))
            distances = [np.linalg.norm(mean - centroid) for mean in means]
            best = int(np.argmin(distances)) if distances else -1
            if best >= 0 and distances[best] < radius:
                tracks[best].append(entry)
                means[best] = np.mean([p[:2] for p in tracks[best]], axis=0)
            else:
                tracks.append([entry])
                means.append(np.array(centroid, dtype=np.float64))

    result = []
    for track in tracks:
        if len(track) < min_presence * len(scans):
            continue
        xy = np.array([p[:2] for p in track])
        ends = np.array([p[3] for p in track])
        has_ends = not np.isnan(ends).any()
        result.append(
            TrackScatter(
                distance=float(np.linalg.norm(xy.mean(axis=0))),
                scatter=float(np.linalg.norm(xy.std(axis=0))),
                box_ratio=float(np.mean([p[2] for p in track])),
                num_scans=len(track),
                width=float(
                    np.median(np.hypot(*(ends[:, :2] - ends[:, 2:]).T))
                    if has_ends
                    else np.nan
                ),
                endpoint_scatter=float(
                    np.mean(
                        [
                            np.linalg.norm(ends[:, :2].std(axis=0)),
                            np.linalg.norm(ends[:, 2:].std(axis=0)),
                        ]
                    )
                    if has_ends
                    else np.nan
                ),
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


def _labelled_scans(
    scans: Sequence[ScanObjects],
    labels: Labels,
    field_of_view: Region | None,
) -> Iterator[tuple[LabelledScan, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield each labelled scan restricted to the labelled region.

    With field_of_view, the region is the intersection of the labelled region and
    the field of view.

    Yields:
        The labelled scan, its labelled boxes in the region and the centroids and
        classes of the detected objects in the region.
    """
    for labelled in labels.scans:
        if labelled.scan >= len(scans):
            raise ValueError(f"Labelled scan {labelled.scan} is not in the bag")
        scan = scans[labelled.scan]
        in_region = labels.region.contains(scan.centroids)
        labelled_boxes = labelled.boxes
        if field_of_view is not None:
            in_region &= field_of_view.contains(scan.centroids)
            labelled_boxes = labelled_boxes[field_of_view.contains(labelled_boxes)]
        yield labelled, labelled_boxes, scan.centroids[in_region], scan.boxes[in_region]


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
    for labelled, labelled_boxes, centroids, boxes in _labelled_scans(
        scans, labels, field_of_view
    ):
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


@dataclass(frozen=True)
class ObjectScore:
    """Result of the object detection (any class) compared to the hand labels."""

    found: int  # labelled boxes matched by a detected object (box or other)
    not_found: int  # labelled boxes with a position, no object within the radius
    missed: int  # labelled boxes without a cluster (no position)

    @property
    def recall(self) -> float:
        """Return the share of the labelled box clusters found as object."""
        total = self.found + self.not_found
        return self.found / total if total else np.nan

    @property
    def recall_total(self) -> float:
        """Return the share of all labelled boxes found as object."""
        total = self.found + self.not_found + self.missed
        return self.found / total if total else np.nan


def score_objects(
    scans: Sequence[ScanObjects],
    labels: Labels,
    match_radius: float,
    field_of_view: Region | None = None,
) -> ObjectScore:
    """Check whether the labelled boxes were detected as objects at all.

    Like score_labels, but the class is ignored: the labelled boxes of a scan are
    matched one to one to all detected objects in the labelled region (optionally
    intersected with field_of_view). A matched labelled box counts as found, also
    if its object is classified as other. The class comes from the camera, the
    LiDAR has to find the obstacle and measure its position.

    Args:
        scans: The objects of all scans of the bag (after the region of interest)
        labels: The hand labels of the bag
        match_radius: Max. distance between a labelled and a detected centroid in m
        field_of_view: Additional restriction of the region, see score_labels

    Returns:
        The counts of the comparison.
    """
    found = not_found = missed = 0
    for labelled, labelled_boxes, centroids, _ in _labelled_scans(
        scans, labels, field_of_view
    ):
        pairs = match_nearest(labelled_boxes, centroids, match_radius)
        found += len(pairs)
        not_found += len(labelled_boxes) - len(pairs)
        missed += labelled.missed_boxes
    return ObjectScore(found=found, not_found=not_found, missed=missed)
