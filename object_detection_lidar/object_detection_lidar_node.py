import numpy as np
import rclpy
import rclpy.utilities
import sensor_msgs.msg
import std_msgs.msg
import visualization_msgs.msg
from sensor_msgs_py import point_cloud2
from smarty_utils.enums import NodeState
from smarty_utils.smarty_node import SmartyNode
from timing import timer

from object_detection_lidar.clustering import (
    DetectedObject,
    cluster_points,
    extract_objects,
)
from object_detection_lidar.scan_processing import scan_to_points

# Only used for the visualization, the object data is not changed
MARKER_MIN_SIZE = 0.02  # in m, keeps thin clusters (e.g. one line) visible
MARKER_HEIGHT = 0.1  # in m, the scan is 2D, so the objects have no real height

# Min. number of points of one line, fewer points always fit a line
MIN_LINE_POINTS = 3
# in m, points this close to a corner or to the outer edges of a contour are not
# used for the line fits (rounded edges and mixed measurements of both sides or
# of the object and the background)
EDGE_MARGIN = 0.02
# Max. bulge of a box side relative to its length. Long, flat sides show a small
# systematic bulge in the scan (~1-2 % in recorded scans), the bulge of round or
# irregular objects is much larger (~8 % and more)
MAX_RELATIVE_BULGE = 0.03
# in m, min. length of both sides of a corner (without the corner margin)
MIN_CORNER_SIDE = 0.035
# Factor on the corner tolerance for the side of a corner that is hit at a
# grazing angle (step-like measurements with gaps)
GRAZING_TOLERANCE_FACTOR = 1.6


def order_by_angle(cluster: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort the points of a cluster by their beam angle seen from the sensor.

    The angles are taken relative to the centroid direction, so clusters lying
    across the +-pi seam of the scan are ordered correctly as well.

    Args:
        cluster: Array of shape (N, 2) with x/y coordinates in m

    Returns:
        The points ordered along the visible contour of the object and their
        angles relative to the centroid direction in rad (ascending).
    """
    centroid = cluster.mean(axis=0)
    angles = np.arctan2(cluster[:, 1], cluster[:, 0]) - np.arctan2(
        centroid[1], centroid[0]
    )
    angles = (angles + np.pi) % (2 * np.pi) - np.pi
    order = np.argsort(angles)
    return cluster[order], angles[order]


def fit_line(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a line through the points (total least squares).

    Args:
        points: Array of shape (N, 2) with x/y coordinates in m

    Returns:
        Unit direction vector of the line and the RMS of the perpendicular
        distances of the points to the line in m.
    """
    centered = points - points.mean(axis=0)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    # The smallest singular value is the norm of the residuals normal to the line
    rms = float(singular_values[-1] / np.sqrt(len(points)))
    return vh[0], rms


def is_straight(points: np.ndarray, tolerance: float) -> tuple[bool, np.ndarray]:
    """Check whether the points lie on a straight line.

    The RMS distance to the line alone lets slightly curved contours pass (the
    RMS of an arc is only ~30 % of its bulge). Therefore a parabola is fitted to
    the distances along the line as well and its bulge must not exceed the
    tolerance or MAX_RELATIVE_BULGE of the line length, whichever is larger.

    Args:
        points: Array of shape (N, 2) with x/y coordinates in m
        tolerance: Max. RMS distance to the line and min. allowed bulge in m

    Returns:
        Whether the points are straight and the unit direction vector of the line.
    """
    direction, rms = fit_line(points)
    if rms > tolerance:
        return False, direction

    centered = points - points.mean(axis=0)
    along = centered @ direction
    across = centered @ np.array([-direction[1], direction[0]])
    length = np.ptp(along)
    curvature = np.polyfit(along, across, 2)[0]
    bulge = abs(curvature) * (length / 2) ** 2
    return bool(bulge <= max(tolerance, MAX_RELATIVE_BULGE * length)), direction


def split_at_corner(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Split an L-shaped contour into its two sides.

    The point farthest from the chord between the first and the last point is a
    first guess of the corner. It is refined in a window around it to the split
    with the smallest total squared distance of both sides to their lines.

    Args:
        points: Array of shape (N, 2) with x/y coordinates in m, ordered along
            the contour

    Returns:
        The points of both sides without the points close to the corner, or None
        if one side has too few points.
    """
    # Can be empty after the outer edges were removed
    if len(points) < 2 * MIN_LINE_POINTS:
        return None

    chord = points[-1] - points[0]
    chord_length = np.linalg.norm(chord)
    if chord_length == 0:
        return None
    normal = np.array([-chord[1], chord[0]]) / chord_length
    guess = int(np.argmax(np.abs((points - points[0]) @ normal)))

    window = max(2, len(points) // 10)
    candidates = range(
        max(MIN_LINE_POINTS - 1, guess - window),
        min(len(points) - MIN_LINE_POINTS, guess + window) + 1,
    )
    if not candidates:
        return None

    def split_error(corner: int) -> float:
        _, rms_first = fit_line(points[: corner + 1])
        _, rms_second = fit_line(points[corner:])
        return rms_first**2 * (corner + 1) + rms_second**2 * (len(points) - corner)

    corner = min(candidates, key=split_error)

    far = np.linalg.norm(points - points[corner], axis=1) > EDGE_MARGIN
    first = points[:corner][far[:corner]]
    second = points[corner + 1 :][far[corner + 1 :]]
    if len(first) < MIN_LINE_POINTS or len(second) < MIN_LINE_POINTS:
        return None
    return first, second


def is_box(
    cluster: np.ndarray,
    angle_increment: float,
    *,
    line_tolerance: float,
    line_tolerance_per_m: float,
    angle_tolerance: float,
    corner_tolerance_factor: float,
    min_size: float,
    min_fill_ratio: float,
) -> bool:
    """Check whether the visible contour of a cluster is that of a box.

    The scan sees either one side of a box (one line) or a corner (two lines at
    a right angle). Round or irregular objects do not fit these shapes. Walls are
    straight as well, they have to be dropped beforehand (diameter filter).

    Small fragments almost always fit a line, so the visible contour has to be at
    least min_size long. A box is a closed surface, so (nearly) every beam within
    its angular span hits it. Sparse clusters with gaps are therefore no boxes.

    The sides of real boxes are very straight (recorded scans: RMS ~1-4 mm up to
    2 m, ~4-8 mm at 3-4 m), so the line tolerance grows with the distance. A
    single line is checked strictly, because small fragments of other objects are
    often nearly straight. A right-angled corner rarely occurs by chance, so its
    sides are checked with a larger tolerance.

    Args:
        cluster: Array of shape (N, 2) with x/y coordinates in m
        angle_increment: Angle between two beams of the scan in rad
        line_tolerance: Max. RMS distance of the points to a line at 0 m in m
        line_tolerance_per_m: Increase of the line tolerance per m of distance
        angle_tolerance: Max. deviation of a corner from 90° in deg
        corner_tolerance_factor: Factor on the line tolerance for the sides of a
            corner
        min_size: Min. distance between the outermost points in m
        min_fill_ratio: Min. ratio of points to beams within the angular span

    Returns:
        True if the cluster is a box.
    """
    if len(cluster) < 2 * MIN_LINE_POINTS:
        return False

    points, angles = order_by_angle(cluster)
    distance = float(np.linalg.norm(points.mean(axis=0)))
    tolerance = line_tolerance + line_tolerance_per_m * distance

    # Distance between the outermost points, for a corner this is the diagonal
    chord = points[-1] - points[0]
    chord_length = np.linalg.norm(chord)
    num_beams = (angles[-1] - angles[0]) / abs(angle_increment) + 1
    if chord_length < min_size or len(points) / num_beams < min_fill_ratio:
        return False

    # One side of the box is visible (strict, all points are used)
    straight, _ = is_straight(points, tolerance)
    if straight:
        return True

    # A corner is visible. The outer edges are not used for the line fits (mixed
    # measurements of the object and the background)
    inner = points[
        (np.linalg.norm(points - points[0], axis=1) > EDGE_MARGIN)
        & (np.linalg.norm(points - points[-1], axis=1) > EDGE_MARGIN)
    ]
    sides = split_at_corner(inner)
    if sides is None:
        return False
    first, second = sides

    direction_first, _ = fit_line(first)
    direction_second, _ = fit_line(second)
    # Directions have an arbitrary sign, so the angle is folded into [0°, 90°]
    cos_angle = abs(float(direction_first @ direction_second))
    angle = np.degrees(np.arccos(np.clip(cos_angle, 0.0, 1.0)))
    side_lengths = [np.linalg.norm(side[-1] - side[0]) for side in sides]
    if 90.0 - angle > angle_tolerance or min(side_lengths) < MIN_CORNER_SIDE:
        return False

    # A right angle between two sides is a strong hint for a box, so the sides
    # may deviate more from a line (noisy measurements at the corner and at
    # grazing incidence at the far end of a side). Usually one side faces the
    # sensor and is clean, only the other side is hit at a grazing angle. Both
    # sides being noisy hints at an irregular object instead.
    corner_tolerance = corner_tolerance_factor * tolerance
    clean = [is_straight(side, corner_tolerance)[0] for side in sides]
    grazing = [
        is_straight(side, GRAZING_TOLERANCE_FACTOR * corner_tolerance)[0]
        for side in sides
    ]
    return any(clean) and all(grazing)


class ObjectDetectionLidarNode(SmartyNode):
    """Detects obstacles by clustering the points of the 2D LiDAR scan."""

    def __init__(self):
        """Initialize the ObjectDetectionLidarNode."""
        super().__init__(
            "object_detection_lidar_node",
            "object_detection_lidar",
            node_parameters={
                # Subscriber topics
                "scan_subscriber": "/scan",
                # Publisher topics
                "points_publisher": "/object_detection_lidar/points",
                "markers_publisher": "/object_detection_lidar/markers",
                # Parameters
                "min_range": 0.15,  # in m, closer points are ignored (e.g. car body)
                "max_range": 4.0,  # in m, farther points are ignored
                "eps": 0.10,  # in m, max. distance of neighboring points (DBSCAN)
                "min_samples": 3,  # min. number of points of a core point (DBSCAN)
                "max_object_diameter": 0.6,  # in m, larger objects are dropped (walls)
                "box_line_tolerance": 0.0023,  # in m, max. RMS distance to a box side
                "box_line_tolerance_per_m": 0.0015,  # in m/m, added per m distance
                "box_angle_tolerance": 10.0,  # in deg, max. deviation of a corner
                "box_corner_tolerance_factor": 2.5,  # line tolerance factor (corner)
                "box_min_size": 0.08,  # in m, obstacles are at least 0.1 m wide
                "box_min_fill_ratio": 0.7,  # min. ratio of points to beams on a box
                "state": NodeState.ACTIVE.value,
                "debug": False,
            },
            subscribed_topics={
                "scan_subscriber": (
                    sensor_msgs.msg.LaserScan,
                    self.scan_callback,
                    1,
                ),
            },
            published_topics={
                "points_publisher": (sensor_msgs.msg.PointCloud2, 1),
                "markers_publisher": (visualization_msgs.msg.MarkerArray, 1),
            },
        )

        self.get_logger().info("ObjectDetectionLidarNode initialized")

    @property
    def min_range(self) -> float:
        """Return the minimum range in m."""
        return self.get_parameter("min_range").value  # type: ignore

    @property
    def max_range(self) -> float:
        """Return the maximum range in m."""
        return self.get_parameter("max_range").value  # type: ignore

    @property
    def eps(self) -> float:
        """Return the maximum distance of neighboring points in m (DBSCAN)."""
        return self.get_parameter("eps").value  # type: ignore

    @property
    def min_samples(self) -> int:
        """Return the minimum number of points of a core point (DBSCAN)."""
        return self.get_parameter("min_samples").value  # type: ignore

    @property
    def max_object_diameter(self) -> float:
        """Return the maximum object diameter in m."""
        return self.get_parameter("max_object_diameter").value  # type: ignore

    @property
    def box_line_tolerance(self) -> float:
        """Return the max. RMS distance of the points to a box side at 0 m in m."""
        return self.get_parameter("box_line_tolerance").value  # type: ignore

    @property
    def box_line_tolerance_per_m(self) -> float:
        """Return the increase of the box line tolerance per m of distance."""
        return self.get_parameter("box_line_tolerance_per_m").value  # type: ignore

    @property
    def box_angle_tolerance(self) -> float:
        """Return the max. deviation of a box corner from 90° in deg."""
        return self.get_parameter("box_angle_tolerance").value  # type: ignore

    @property
    def box_corner_tolerance_factor(self) -> float:
        """Return the factor on the line tolerance for the sides of a corner."""
        return self.get_parameter("box_corner_tolerance_factor").value  # type: ignore

    @property
    def box_min_size(self) -> float:
        """Return the min. size of the visible contour of a box in m."""
        return self.get_parameter("box_min_size").value  # type: ignore

    @property
    def box_min_fill_ratio(self) -> float:
        """Return the min. ratio of points to beams within the span of a box."""
        return self.get_parameter("box_min_fill_ratio").value  # type: ignore

    def scan_callback(self, msg: sensor_msgs.msg.LaserScan) -> None:
        """Executed by the ROS2 system whenever a new scan is received."""
        if not self.active:
            return
        self.process_scan(msg)

    @timer.Timer(name="total", filter_strength=40)
    def process_scan(self, msg: sensor_msgs.msg.LaserScan) -> None:
        """Convert the scan into points, detect objects and publish both.

        Args:
            msg: The laser scan message.
        """
        with timer.Timer(name="scan_to_points", filter_strength=40):
            points = scan_to_points(
                np.asarray(msg.ranges),
                angle_min=msg.angle_min,
                angle_increment=msg.angle_increment,
                range_min=max(msg.range_min, self.min_range),
                range_max=min(msg.range_max, self.max_range),
            )

        with timer.Timer(name="publish", filter_strength=40):
            # The scan is 2D, so all points lie in the plane z = 0 of the scan frame
            points_xyz = np.column_stack((points, np.zeros(len(points))))
            cloud = point_cloud2.create_cloud_xyz32(msg.header, points_xyz.tolist())
            self.points_publisher.publish(cloud)  # type: ignore

        with timer.Timer(name="clustering", filter_strength=40):
            labels = cluster_points(points, self.eps, self.min_samples)
            objects = extract_objects(points, labels)
            # Same order as the objects (sorted labels without noise)
            clusters = [
                points[labels == label] for label in np.unique(labels) if label != -1
            ]

            # Same criterion as filter_objects, but the clusters are kept as well
            kept = [
                (obj, cluster)
                for obj, cluster in zip(objects, clusters, strict=True)
                if obj.diameter <= self.max_object_diameter
            ]
            objects = [obj for obj, _ in kept]

        with timer.Timer(name="classification", filter_strength=40):
            boxes = [
                is_box(
                    cluster,
                    msg.angle_increment,
                    line_tolerance=self.box_line_tolerance,
                    line_tolerance_per_m=self.box_line_tolerance_per_m,
                    angle_tolerance=self.box_angle_tolerance,
                    corner_tolerance_factor=self.box_corner_tolerance_factor,
                    min_size=self.box_min_size,
                    min_fill_ratio=self.box_min_fill_ratio,
                )
                for _, cluster in kept
            ]

        with timer.Timer(name="publish_markers", filter_strength=40):
            markers = self.create_markers(msg.header, objects, boxes)
            self.markers_publisher.publish(markers)  # type: ignore

        if self._debug:
            self.get_logger().info(f"Valid points: {len(points)}/{len(msg.ranges)}")
            centroids = ", ".join(
                f"({obj.centroid_x:.2f}, {obj.centroid_y:.2f}, "
                f"{'box' if box else 'other'})"
                for obj, box in zip(objects, boxes, strict=True)
            )
            self.get_logger().info(
                f"Objects: {len(objects)}, boxes: {sum(boxes)} [{centroids}]"
            )
            timer.Timer(logger=self.get_logger().info).print()  # type: ignore

    @staticmethod
    def create_markers(
        header: std_msgs.msg.Header, objects: list[DetectedObject], boxes: list[bool]
    ) -> visualization_msgs.msg.MarkerArray:
        """Create one box marker per object for the visualization in rviz.

        Boxes are red, all other objects are blue. The first marker deletes all
        markers of the previous scan.

        Args:
            header: Header of the scan (frame and stamp of the markers)
            objects: The detected objects
            boxes: Whether the object with the same index is a box

        Returns:
            The marker array.
        """
        delete_all = visualization_msgs.msg.Marker()
        delete_all.header = header
        delete_all.action = visualization_msgs.msg.Marker.DELETEALL
        markers = [delete_all]

        for i, (obj, box) in enumerate(zip(objects, boxes, strict=True)):
            marker = visualization_msgs.msg.Marker()
            marker.header = header
            # Separate namespaces, so both classes can be toggled in rviz
            marker.ns = "boxes" if box else "objects"
            marker.id = i
            marker.type = visualization_msgs.msg.Marker.CUBE
            marker.action = visualization_msgs.msg.Marker.ADD
            marker.pose.position.x = (obj.min_x + obj.max_x) / 2
            marker.pose.position.y = (obj.min_y + obj.max_y) / 2
            marker.pose.orientation.w = 1.0
            marker.scale.x = max(obj.extent_x, MARKER_MIN_SIZE)
            marker.scale.y = max(obj.extent_y, MARKER_MIN_SIZE)
            marker.scale.z = MARKER_HEIGHT
            if box:
                marker.color.r = 1.0
            else:
                marker.color.b = 1.0
            marker.color.a = 0.5
            markers.append(marker)

        return visualization_msgs.msg.MarkerArray(markers=markers)


def main(args: list[str] | None = None) -> None:
    """Main function to start the ObjectDetectionLidarNode.

    Args:
        args: Launch arguments (default: None)
    """
    rclpy.init(args=args)
    node = ObjectDetectionLidarNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        # Shutdown if not already done by the ROS2 launch system
        if rclpy.utilities.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
