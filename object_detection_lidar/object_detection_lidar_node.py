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

from object_detection_lidar.box_classification import is_box
from object_detection_lidar.clustering import (
    DetectedObject,
    cluster_points,
    extract_objects,
    filter_objects,
)
from object_detection_lidar.scan_processing import scan_to_points

# Only used for the visualization, the object data is not changed
MARKER_MIN_SIZE = 0.02  # in m, keeps thin clusters (e.g. one line) visible
MARKER_HEIGHT = 0.1  # in m, the scan is 2D, so the objects have no real height

# Fields of the objects point cloud: centroid (x, y, z = 0), axis-aligned extent and
# diameter in m in the scan frame, is_box is 1.0 for boxes and 0.0 otherwise
OBJECT_FIELDS = [
    sensor_msgs.msg.PointField(
        name=name, offset=4 * i, datatype=sensor_msgs.msg.PointField.FLOAT32, count=1
    )
    for i, name in enumerate(
        ("x", "y", "z", "extent_x", "extent_y", "diameter", "is_box")
    )
]


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
                "objects_publisher": "/object_detection_lidar/objects",
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
                "objects_publisher": (sensor_msgs.msg.PointCloud2, 1),
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
            objects = filter_objects(objects, self.max_object_diameter)

        with timer.Timer(name="classification", filter_strength=40):
            boxes = [
                is_box(
                    obj.points,
                    msg.angle_increment,
                    line_tolerance=self.box_line_tolerance,
                    line_tolerance_per_m=self.box_line_tolerance_per_m,
                    angle_tolerance=self.box_angle_tolerance,
                    corner_tolerance_factor=self.box_corner_tolerance_factor,
                    min_size=self.box_min_size,
                    min_fill_ratio=self.box_min_fill_ratio,
                )
                for obj in objects
            ]

        with timer.Timer(name="publish_markers", filter_strength=40):
            markers = self.create_markers(msg.header, objects, boxes)
            self.markers_publisher.publish(markers)  # type: ignore

        with timer.Timer(name="publish_objects", filter_strength=40):
            objects_cloud = self.create_objects_cloud(msg.header, objects, boxes)
            self.objects_publisher.publish(objects_cloud)  # type: ignore

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
    def create_objects_cloud(
        header: std_msgs.msg.Header, objects: list[DetectedObject], boxes: list[bool]
    ) -> sensor_msgs.msg.PointCloud2:
        """Create a point cloud with one point per object for further processing.

        Each point is the centroid of an object (z = 0, so it can be shown in rviz)
        with the fields OBJECT_FIELDS. A point cloud is used, because it has a
        header (frame and stamp of the scan, needed for the fusion) and named
        fields without a custom message.

        Args:
            header: Header of the scan (frame and stamp of the objects)
            objects: The detected objects
            boxes: Whether the object with the same index is a box

        Returns:
            The point cloud.
        """
        rows = [
            (
                obj.centroid_x,
                obj.centroid_y,
                0.0,
                obj.extent_x,
                obj.extent_y,
                obj.diameter,
                float(box),
            )
            for obj, box in zip(objects, boxes, strict=True)
        ]
        return point_cloud2.create_cloud(header, OBJECT_FIELDS, rows)

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
