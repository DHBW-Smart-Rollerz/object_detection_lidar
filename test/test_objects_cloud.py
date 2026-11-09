import numpy as np
import std_msgs.msg
from object_detection_lidar.clustering import DetectedObject, extract_objects
from object_detection_lidar.object_detection_lidar_node import (
    OBJECT_FIELDS,
    VEHICLE_FRAME,
    ObjectDetectionLidarNode,
)
from object_detection_lidar.pipeline import PipelineParams
from sensor_msgs_py import point_cloud2


def test_objects_cloud_in_vehicle_frame():
    # Line 1 m ahead of the LiDAR (laser -y is forward), across the vehicle
    obj = line_ahead()
    params = PipelineParams(lidar_offset_x=0.1, lidar_offset_y=0.0)
    header = std_msgs.msg.Header(frame_id="laser")
    header.stamp.sec = 42

    cloud = ObjectDetectionLidarNode.create_objects_cloud(
        header,
        [obj],
        [True],
        params,
    )

    assert cloud.header.frame_id == VEHICLE_FRAME
    assert cloud.header.stamp.sec == 42
    assert header.frame_id == "laser"  # the scan header is not changed
    names = [f.name for f in OBJECT_FIELDS]
    (row,) = point_cloud2.read_points_list(cloud, field_names=names)
    values = dict(zip(names, row, strict=True))
    # Centroid 1.1 m ahead of the front axle, centered
    assert values["x"] == np.float32(1.1)
    assert abs(values["y"]) < 1e-6
    assert values["z"] == 0.0
    # Across the vehicle: 0.5 m wide along vehicle y, thin along x
    assert values["extent_x"] < 1e-6
    np.testing.assert_allclose(values["extent_y"], 0.5, rtol=1e-6)
    assert values["is_box"] == 1.0
    # Endpoints ahead, left (larger y) and right, like the camera
    np.testing.assert_allclose(
        [values[n] for n in ("left_x", "left_y", "right_x", "right_y")],
        [1.1, 0.25, 1.1, -0.25],
        rtol=1e-6,
    )
    np.testing.assert_allclose(values["width"], 0.5, rtol=1e-6)


def line_ahead() -> DetectedObject:
    """Return a line 1 m ahead of the LiDAR from 0.25 m right to 0.25 m left."""
    xs = np.linspace(-0.25, 0.25, 11)  # laser x is left
    points = np.column_stack((xs, np.full_like(xs, -1.0)))
    (obj,) = extract_objects(points, np.zeros(len(points), dtype=int))
    return obj
