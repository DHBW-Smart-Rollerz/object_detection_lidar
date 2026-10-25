import numpy as np
import rclpy
import rclpy.utilities
import sensor_msgs.msg
from sensor_msgs_py import point_cloud2
from smarty_utils.enums import NodeState
from smarty_utils.smarty_node import SmartyNode
from timing import timer

from object_detection_lidar.scan_processing import scan_to_points


class ObjectDetectionLidarNode(SmartyNode):
    """Converts the 2D LiDAR scan into points as base for the obstacle detection."""

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
                # Parameters
                "min_range": 0.15,  # in m, closer points are ignored (e.g. car body)
                "max_range": 4.0,  # in m, farther points are ignored
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

    def scan_callback(self, msg: sensor_msgs.msg.LaserScan) -> None:
        """Executed by the ROS2 system whenever a new scan is received."""
        if not self.active:
            return
        self.process_scan(msg)

    @timer.Timer(name="total", filter_strength=40)
    def process_scan(self, msg: sensor_msgs.msg.LaserScan) -> None:
        """Convert the scan into points and publish them as point cloud.

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

        if self._debug:
            self.get_logger().info(f"Valid points: {len(points)}/{len(msg.ranges)}")
            timer.Timer(logger=self.get_logger().info).print()  # type: ignore


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
