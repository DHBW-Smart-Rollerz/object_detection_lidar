from dataclasses import asdict, fields

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

from object_detection_lidar.clustering import DetectedObject
from object_detection_lidar.pipeline import (
    PipelineParams,
    camera_format,
    detect_objects,
    vehicle_centroids,
    vehicle_endpoints,
    vehicle_extents,
)

# Only used for the visualization, the object data is not changed
MARKER_MIN_SIZE = 0.02  # in m, keeps thin clusters (e.g. one line) visible
MARKER_HEIGHT = 0.1  # in m, the scan is 2D, so the objects have no real height

# Frame of the objects point cloud: the vehicle frame of the camera objects (x
# forward, y left, origin at the center of the front axle, see to_vehicle_frame).
# There is no tf laser -> vehicle, so rviz with the fixed frame laser cannot show it.
VEHICLE_FRAME = "vehicle"

# Fields of the objects point cloud, all in m in the vehicle frame: centroid (x, y,
# z = 0), extent of the box aligned with the vehicle axes and diameter, is_box is 1.0
# for boxes and 0.0 otherwise, the endpoints of the visible contour (left_x, left_y,
# right_x, right_y) and their distance (width), like the camera objects.
OBJECT_FIELDS = [
    sensor_msgs.msg.PointField(
        name=name, offset=4 * i, datatype=sensor_msgs.msg.PointField.FLOAT32, count=1
    )
    for i, name in enumerate(
        (
            "x",
            "y",
            "z",
            "extent_x",
            "extent_y",
            "diameter",
            "is_box",
            "left_x",
            "left_y",
            "right_x",
            "right_y",
            "width",
        )
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
                "object_publisher": "/object_detection_lidar/object",
                # Parameters of the detection, defaults from PipelineParams
                **asdict(PipelineParams()),
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
                "object_publisher": (std_msgs.msg.Float32MultiArray, 1),
            },
        )

        self.get_logger().info("ObjectDetectionLidarNode initialized")

    @property
    def pipeline_params(self) -> PipelineParams:
        """Return the current parameters of the detection."""
        return PipelineParams(
            **{  # type: ignore
                field.name: self.get_parameter(field.name).value
                for field in fields(PipelineParams)
            }
        )

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
        params = self.pipeline_params
        result = detect_objects(
            np.asarray(msg.ranges),
            angle_min=msg.angle_min,
            angle_increment=msg.angle_increment,
            range_min=msg.range_min,
            range_max=msg.range_max,
            params=params,
            step_timer=lambda name: timer.Timer(name=name, filter_strength=40),
        )
        points, objects, boxes = result.points, result.objects, result.boxes

        with timer.Timer(name="publish", filter_strength=40):
            # The scan is 2D, so all points lie in the plane z = 0 of the scan frame
            points_xyz = np.column_stack((points, np.zeros(len(points))))
            cloud = point_cloud2.create_cloud_xyz32(msg.header, points_xyz.tolist())
            self.points_publisher.publish(cloud)  # type: ignore

        with timer.Timer(name="publish_markers", filter_strength=40):
            markers = self.create_markers(
                msg.header, objects, boxes, result.round_objects
            )
            self.markers_publisher.publish(markers)  # type: ignore

        with timer.Timer(name="publish_objects", filter_strength=40):
            objects_cloud = self.create_objects_cloud(
                msg.header, objects, boxes, params
            )
            self.objects_publisher.publish(objects_cloud)  # type: ignore

        with timer.Timer(name="publish_object", filter_strength=40):
            # Same format as the camera (/object_detection/object), so the fusion
            # can read both sensors alike. All objects (only those in the region
            # of interest with roi_enabled true), an empty array for no object.
            object_msg = std_msgs.msg.Float32MultiArray()
            object_msg.data = camera_format(objects, params).tolist()
            self.object_publisher.publish(object_msg)  # type: ignore

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
        header: std_msgs.msg.Header,
        objects: list[DetectedObject],
        boxes: list[bool],
        params: PipelineParams,
    ) -> sensor_msgs.msg.PointCloud2:
        """Create a point cloud with one point per object for further processing.

        Each point is the centroid of an object (z = 0) with the fields
        OBJECT_FIELDS, all transformed into the vehicle frame (vehicle_centroids,
        vehicle_extents, vehicle_endpoints, frame_id VEHICLE_FRAME). A point
        cloud is used, because it has a header (stamp of the scan, needed for the
        fusion) and named fields without a custom message.

        Args:
            header: Header of the scan (stamp of the objects)
            objects: The detected objects
            boxes: Whether the object with the same index is a box
            params: Parameters with laser_yaw and the LiDAR offset

        Returns:
            The point cloud.
        """
        centroids = vehicle_centroids(objects, params)
        extents = vehicle_extents(objects, params)
        endpoints = vehicle_endpoints(objects, params)
        rows = [
            (
                *(float(v) for v in centroid),
                0.0,
                *(float(v) for v in extent),
                obj.diameter,
                float(box),
                *(float(v) for v in ends),
                obj.width,
            )
            for obj, box, centroid, extent, ends in zip(
                objects, boxes, centroids, extents, endpoints, strict=True
            )
        ]
        vehicle_header = std_msgs.msg.Header(stamp=header.stamp, frame_id=VEHICLE_FRAME)
        return point_cloud2.create_cloud(vehicle_header, OBJECT_FIELDS, rows)

    @staticmethod
    def create_markers(
        header: std_msgs.msg.Header,
        objects: list[DetectedObject],
        boxes: list[bool],
        round_objects: list[DetectedObject] | None = None,
    ) -> visualization_msgs.msg.MarkerArray:
        """Create one box marker per object for the visualization in rviz.

        Boxes are red, all other objects are blue, the objects dropped as round
        are gray (namespace "round", for debugging, they are not published
        anywhere else). The first marker deletes all markers of the previous scan.

        Args:
            header: Header of the scan (frame and stamp of the markers)
            objects: The detected objects
            boxes: Whether the object with the same index is a box
            round_objects: The objects dropped as round

        Returns:
            The marker array.
        """
        delete_all = visualization_msgs.msg.Marker()
        delete_all.header = header
        delete_all.action = visualization_msgs.msg.Marker.DELETEALL
        markers = [delete_all]

        # 1 box, 0 other object, None round (dropped)
        classes = [*boxes, *([None] * len(round_objects or []))]
        all_objects = [*objects, *(round_objects or [])]
        for i, (obj, box) in enumerate(zip(all_objects, classes, strict=True)):
            marker = visualization_msgs.msg.Marker()
            marker.header = header
            # Separate namespaces, so the classes can be toggled in rviz
            marker.ns = "round" if box is None else "boxes" if box else "objects"
            marker.id = i
            marker.type = visualization_msgs.msg.Marker.CUBE
            marker.action = visualization_msgs.msg.Marker.ADD
            marker.pose.position.x = (obj.min_x + obj.max_x) / 2
            marker.pose.position.y = (obj.min_y + obj.max_y) / 2
            marker.pose.orientation.w = 1.0
            marker.scale.x = max(obj.extent_x, MARKER_MIN_SIZE)
            marker.scale.y = max(obj.extent_y, MARKER_MIN_SIZE)
            marker.scale.z = MARKER_HEIGHT
            if box is None:
                marker.color.r = marker.color.g = marker.color.b = 0.6
            elif box:
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
