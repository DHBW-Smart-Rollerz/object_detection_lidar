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
    filter_objects,
)
from object_detection_lidar.scan_processing import scan_to_points

# Only used for the visualization, the object data is not changed
MARKER_MIN_SIZE = 0.02  # in m, keeps thin clusters (e.g. one line) visible
MARKER_HEIGHT = 0.1  # in m, the scan is 2D, so the objects have no real height


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

        with timer.Timer(name="publish_markers", filter_strength=40):
            markers = self.create_markers(msg.header, objects)
            self.markers_publisher.publish(markers)  # type: ignore

        if self._debug:
            self.get_logger().info(f"Valid points: {len(points)}/{len(msg.ranges)}")
            centroids = ", ".join(
                f"({obj.centroid_x:.2f}, {obj.centroid_y:.2f})" for obj in objects
            )
            self.get_logger().info(f"Objects: {len(objects)} [{centroids}]")
            timer.Timer(logger=self.get_logger().info).print()  # type: ignore

    @staticmethod
    def create_markers(
        header: std_msgs.msg.Header, objects: list[DetectedObject]
    ) -> visualization_msgs.msg.MarkerArray:
        """Create one box marker per object for the visualization in rviz.

        The first marker deletes all markers of the previous scan.

        Args:
            header: Header of the scan (frame and stamp of the markers)
            objects: The detected objects

        Returns:
            The marker array.
        """
        delete_all = visualization_msgs.msg.Marker()
        delete_all.header = header
        delete_all.action = visualization_msgs.msg.Marker.DELETEALL
        markers = [delete_all]

        for i, obj in enumerate(objects):
            marker = visualization_msgs.msg.Marker()
            marker.header = header
            marker.ns = "objects"
            marker.id = i
            marker.type = visualization_msgs.msg.Marker.CUBE
            marker.action = visualization_msgs.msg.Marker.ADD
            marker.pose.position.x = (obj.min_x + obj.max_x) / 2
            marker.pose.position.y = (obj.min_y + obj.max_y) / 2
            marker.pose.orientation.w = 1.0
            marker.scale.x = max(obj.extent_x, MARKER_MIN_SIZE)
            marker.scale.y = max(obj.extent_y, MARKER_MIN_SIZE)
            marker.scale.z = MARKER_HEIGHT
            marker.color.r = 1.0
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
