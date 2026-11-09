"""Evaluate the object detection offline on rosbags (no ROS node).

Every scan of the bag runs through detect_objects with the parameters of the
node (config/ros_params.yaml). The metrics are explained in the README.

Usage (ROS sourced):
    python3 tools/evaluate.py BAG [BAG ...] [--csv metrics.csv]
"""

import argparse
import csv
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import yaml

# Evaluate the package next to this tool (the working tree), not an installed copy
PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR))

from object_detection_lidar.evaluation import (  # noqa: E402
    Labels,
    LabelScore,
    Region,
    ScanObjects,
    TrackScatter,
    count_stats,
    flicker,
    score_labels,
    static_scatter,
    timing_stats,
)
from object_detection_lidar.pipeline import (  # noqa: E402
    STEPS,
    PipelineParams,
    StepTimer,
    detect_objects,
)

NODE_NAME = "object_detection_lidar_node"


@dataclass(frozen=True)
class Scan:
    """The fields of a LaserScan message needed by detect_objects."""

    ranges: np.ndarray  # in m
    angle_min: float  # in rad
    angle_increment: float  # in rad
    range_min: float  # in m
    range_max: float  # in m


@dataclass(frozen=True)
class Metric:
    """One row of the output table (several values, e.g. median / p95 / max)."""

    name: str
    values: tuple[float, ...]
    suffixes: tuple[str, ...] = ("",)
    precision: int = 1
    # Scan range [start, end) of a static phase, these rows are printed in a
    # separate table
    phase: tuple[int, int] | None = None


def read_scans(bag: Path, topic: str) -> Iterator[Scan]:
    """Read all laser scans of one topic from a rosbag.

    Args:
        bag: Directory of the bag (with metadata.yaml)
        topic: Topic of the sensor_msgs/LaserScan messages

    Yields:
        The scans in the order of the bag.
    """
    # Imported here, so the rest of the tool can be used without ROS
    import rosbag2_py  # noqa: PLC0415
    from rclpy.serialization import deserialize_message  # noqa: PLC0415
    from sensor_msgs.msg import LaserScan  # noqa: PLC0415

    reader = rosbag2_py.SequentialReader()
    # An empty storage id lets rosbag2 detect the format (mcap, sqlite3)
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id=""),
        rosbag2_py.ConverterOptions("", ""),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    while reader.has_next():
        _, data, _ = reader.read_next()
        msg = deserialize_message(data, LaserScan)
        yield Scan(
            ranges=np.asarray(msg.ranges, dtype=np.float64),
            angle_min=msg.angle_min,
            angle_increment=msg.angle_increment,
            range_min=msg.range_min,
            range_max=msg.range_max,
        )


def load_params(path: Path) -> PipelineParams:
    """Load the detection parameters of the node from its yaml file.

    Parameters missing in the file keep the defaults of PipelineParams.
    """
    data = yaml.safe_load(path.read_text())[NODE_NAME]["ros__parameters"]
    names = {f.name for f in fields(PipelineParams)}
    return PipelineParams(**{k: v for k, v in data.items() if k in names})


def parse_override(text: str) -> tuple[str, object]:
    """Parse a parameter override NAME=VALUE, the value is parsed as yaml."""
    name, sep, value = text.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {text!r}")
    return name.strip(), yaml.safe_load(value)


def override_params(
    params: PipelineParams, overrides: list[tuple[str, object]]
) -> PipelineParams:
    """Replace single parameters, e.g. [("roi_enabled", False)].

    Integers are accepted for float parameters, otherwise the type has to match.
    """
    types = {f.name: f.type for f in fields(PipelineParams)}
    changes = {}
    for name, raw in overrides:
        if name not in types:
            raise SystemExit(f"--set: unknown parameter {name!r}")
        expected = types[name]
        value = float(raw) if expected is float and type(raw) is int else raw
        if type(value) is not expected:
            raise SystemExit(f"--set: {name} expects {expected}, got {value!r}")
        changes[name] = value
    return replace(params, **changes)


def load_labels(path: Path) -> Labels | None:
    """Load the hand labels of a bag, None if the file does not exist."""
    if not path.is_file():
        return None
    return Labels.from_dict(yaml.safe_load(path.read_text()))


def parse_range(text: str) -> tuple[int, int]:
    """Parse a scan index range START:END (END excluded)."""
    start, end = text.split(":")
    return int(start), int(end)


def make_step_timer(times: dict[str, float]) -> StepTimer:
    """Return a step_timer that writes the runtime of each step in s to times."""

    @contextmanager
    def step_timer(name: str) -> Iterator[None]:
        start = time.perf_counter()
        yield
        times[name] = time.perf_counter() - start

    return step_timer


def scatter_stats(tracks: list[TrackScatter]) -> tuple[float, float]:
    """Return median and 90th percentile of the scatter of the tracks in mm."""
    if not tracks:
        return np.nan, np.nan
    scatter = np.array([1e3 * track.scatter for track in tracks])
    return float(np.median(scatter)), float(np.percentile(scatter, 90))


def score_metrics(score: LabelScore, prefix: str = "") -> list[Metric]:
    """Return the table rows of a comparison with the hand labels."""
    return [
        Metric(
            f"{prefix}TP / FN / missed / FP",
            (
                score.true_positives,
                score.false_negatives,
                score.missed,
                score.false_positives,
            ),
            ("tp", "fn", "missed", "fp"),
            0,
        ),
        Metric(f"{prefix}precision [%]", (100 * score.precision,)),
        Metric(f"{prefix}recall [%]", (100 * score.recall,)),
        Metric(f"{prefix}recall incl. missed [%]", (100 * score.recall_total,)),
    ]


def evaluate_bag(
    bag: Path, args: argparse.Namespace, params: PipelineParams
) -> list[Metric]:
    """Run the detection on all scans of a bag and compute the metrics."""
    labels_path = args.labels_dir / f"{bag.name}.yaml"
    labels = load_labels(labels_path)

    scans: list[ScanObjects] = []
    step_times: list[dict[str, float]] = []
    for scan in read_scans(bag, args.topic):
        times: dict[str, float] = {}
        start = time.perf_counter()
        result = detect_objects(
            scan.ranges,
            scan.angle_min,
            scan.angle_increment,
            scan.range_min,
            scan.range_max,
            params=params,
            step_timer=make_step_timer(times),
        )
        times["total"] = time.perf_counter() - start
        step_times.append(times)
        scans.append(ScanObjects.from_result(result))
    if not scans:
        raise SystemExit(f"{bag}: no messages on {args.topic}")

    stats_suffixes = ("median", "p95", "max")
    metrics = [Metric("scans", (len(scans),), precision=0)]
    timing = timing_stats(step_times)
    for name in ("total", *STEPS, "clustering"):
        s = timing[name]
        metrics.append(
            Metric(f"time {name} [ms]", (s.median, s.p95, s.max), stats_suffixes, 2)
        )

    objects, boxes, mean_objects, mean_boxes = count_stats(scans)
    metrics += [
        Metric("objects/scan (mean)", (mean_objects,)),
        Metric(
            "objects/scan",
            (objects.median, objects.p95, objects.max),
            stats_suffixes,
            0,
        ),
        Metric("boxes/scan (mean)", (mean_boxes,)),
        Metric("boxes/scan", (boxes.median, boxes.p95, boxes.max), stats_suffixes, 0),
    ]

    flicker_result = flicker(scans, args.flicker_radius)
    metrics += [
        Metric("flicker pairs", (flicker_result.pairs,), precision=0),
        Metric("flicker [%]", (100 * flicker_result.rate,)),
    ]

    static_ranges = args.static or (labels.static_scans if labels else [])
    tracks = []
    for start, end in static_ranges:
        phase_tracks = static_scatter(scans[start:end])
        tracks += phase_tracks
        metrics.append(
            Metric(
                f"static {start}:{end}",
                (len(phase_tracks), *scatter_stats(phase_tracks)),
                ("objects", "median [mm]", "p90 [mm]"),
                phase=(start, end),
            )
        )
    metrics += [
        Metric("static objects", (len(tracks),), precision=0),
        Metric("static scatter [mm]", scatter_stats(tracks), ("median", "p90")),
    ]

    if labels is not None:
        metrics.append(Metric("labelled scans", (len(labels.scans),), precision=0))
        metrics += score_metrics(score_labels(scans, labels, args.match_radius))
        if args.fov_half_angle is not None:
            # Forward of the vehicle in the laser frame (inverse of laser_yaw)
            field_of_view = Region(
                heading=-params.laser_yaw,
                half_angle=args.fov_half_angle,
                max_range=np.inf,
            )
            score = score_labels(scans, labels, args.match_radius, field_of_view)
            metrics += score_metrics(score, f"fov ±{args.fov_half_angle:g}° ")
    return metrics


def format_metric(metric: Metric) -> str:
    """Format the values of a metric for the terminal, e.g. '3.10 / 4.20 / 9.00'."""
    return " / ".join(
        "-" if np.isnan(value) else f"{value:.{metric.precision}f}"
        for value in metric.values
    )


def print_table(results: dict[str, list[Metric]]) -> None:
    """Print one row per metric and one column per bag (without the phases)."""
    results = {
        bag: [m for m in metrics if m.phase is None] for bag, metrics in results.items()
    }
    names = list(dict.fromkeys(m.name for metrics in results.values() for m in metrics))
    cells = {
        bag: {m.name: format_metric(m) for m in metrics}
        for bag, metrics in results.items()
    }
    first = max(len(name) for name in names)
    widths = {
        bag: max(len(bag), *(len(v) for v in cells[bag].values())) for bag in results
    }
    print(f"{'metric':<{first}}  " + "  ".join(f"{b:>{widths[b]}}" for b in results))
    print("-" * (first + sum(w + 2 for w in widths.values())))
    for name in names:
        row = "  ".join(f"{cells[b].get(name, ''):>{widths[b]}}" for b in results)
        print(f"{name:<{first}}  {row}")


def print_static_table(results: dict[str, list[Metric]]) -> None:
    """Print the scatter of each static phase, one row per bag and phase."""
    rows = [
        (
            bag,
            f"[{m.phase[0]}, {m.phase[1]})",
            f"{m.values[0]:.0f}",
            *("-" if np.isnan(v) else f"{v:.1f}" for v in m.values[1:]),
        )
        for bag, metrics in results.items()
        for m in metrics
        if m.phase is not None
    ]
    if not rows:
        return
    header = ("bag", "scans", "objects", "median [mm]", "p90 [mm]")
    widths = [max(len(str(r[i])) for r in (header, *rows)) for i in range(5)]
    print()
    print(
        f"{header[0]:<{widths[0]}}  "
        + "  ".join(f"{h:>{w}}" for h, w in zip(header[1:], widths[1:], strict=True))
    )
    print("-" * (sum(widths) + 2 * 4))
    for row in rows:
        print(
            f"{row[0]:<{widths[0]}}  "
            + "  ".join(f"{c:>{w}}" for c, w in zip(row[1:], widths[1:], strict=True))
        )


def write_csv(path: Path, results: dict[str, list[Metric]]) -> None:
    """Write the metrics as long table with the columns bag, metric, value."""
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("bag", "metric", "value"))
        for bag, metrics in results.items():
            for metric in metrics:
                for suffix, value in zip(metric.suffixes, metric.values, strict=True):
                    name = f"{metric.name} {suffix}".strip()
                    writer.writerow((bag, name, value))


def main(argv: list[str] | None = None) -> None:
    """Parse the arguments, evaluate all bags and print the metrics."""
    parser = argparse.ArgumentParser(
        description="Evaluate the object detection offline on rosbags."
    )
    parser.add_argument("bags", nargs="+", type=Path, help="rosbag directories")
    parser.add_argument("--topic", default="/lidar/scan", help="LaserScan topic")
    parser.add_argument(
        "--params",
        type=Path,
        default=PACKAGE_DIR / "config" / "ros_params.yaml",
        help="parameter file of the node",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=PACKAGE_DIR / "tools" / "labels",
        help="directory with the hand labels <bag name>.yaml",
    )
    parser.add_argument(
        "--match-radius",
        type=float,
        default=0.10,
        help="max. distance labelled box to detected centroid in m",
    )
    parser.add_argument(
        "--fov-half-angle",
        type=float,
        metavar="DEG",
        help="also score the labels only within +-DEG around forward of the vehicle "
        "(e.g. field of view of the camera), intersected with the labelled region",
    )
    parser.add_argument(
        "--flicker-radius",
        type=float,
        default=0.05,
        help="max. centroid shift between two scans in m",
    )
    parser.add_argument(
        "--static",
        type=parse_range,
        action="append",
        metavar="START:END",
        help="scan range with standing vehicle (overrides the labels), repeatable",
    )
    parser.add_argument(
        "--set",
        type=parse_override,
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override a detection parameter, e.g. roi_enabled=false, repeatable",
    )
    parser.add_argument("--csv", type=Path, help="also write the metrics to a CSV")
    args = parser.parse_args(argv)

    params = override_params(load_params(args.params), args.set)
    results = {}
    for bag in args.bags:
        print(f"Evaluating {bag} ...", file=sys.stderr)
        results[bag.name] = evaluate_bag(bag, args, params)
    print_table(results)
    print_static_table(results)
    if args.csv:
        write_csv(args.csv, results)


if __name__ == "__main__":
    main()
