"""Clustering of 2D points into obstacle candidates (independent of ROS)."""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN


@dataclass(frozen=True)
class DetectedObject:
    """Centroid, axis-aligned bounding box, diameter and endpoints of a cluster.

    All values are in m and given in the scan frame. The diameter is the largest
    distance between two points of the cluster and, unlike the box, does not depend
    on the rotation of the object. The endpoints are the outermost points of the
    visible contour as seen from the sensor (see contour_endpoints), the width is
    their distance. The points of the cluster are kept, so later steps (e.g. the
    box classification) work on exactly the same cluster.
    """

    centroid_x: float
    centroid_y: float
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    diameter: float
    left_x: float  # endpoint with the largest bearing (left as seen from sensor)
    left_y: float
    right_x: float  # endpoint with the smallest bearing (right as seen from sensor)
    right_y: float
    # Array of shape (N, 2), not compared, so objects can still be compared
    # by their values (numpy arrays have no single truth value)
    points: np.ndarray = field(compare=False, repr=False)

    @property
    def extent_x(self) -> float:
        """Return the size of the box along x in m."""
        return self.max_x - self.min_x

    @property
    def extent_y(self) -> float:
        """Return the size of the box along y in m."""
        return self.max_y - self.min_y

    @property
    def width(self) -> float:
        """Return the distance between the two endpoints in m."""
        return float(np.hypot(self.left_x - self.right_x, self.left_y - self.right_y))


def contour_endpoints(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find the outermost points of a cluster as seen from the sensor.

    The bearing (angle around the sensor origin) of each point is taken relative
    to the bearing of the mean point and wrapped to [-pi, pi). This avoids the jump
    from +pi to -pi for objects behind the sensor (around ±180°). An object spans
    much less than 180°, so the relative bearing is unique.

    The left endpoint has the largest bearing (counterclockwise, left when looking
    from the sensor at the object), the right endpoint the smallest. Like the
    bottom corners of a camera bounding box, they bound the visible part of the
    object, the hidden back side is not measured.

    Args:
        points: Array of shape (N, 2) with x/y coordinates in m (N >= 1)

    Returns:
        The left and the right endpoint, each an array of shape (2,).
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    mean = points.mean(axis=0)
    reference = np.arctan2(mean[1], mean[0])
    bearings = np.arctan2(points[:, 1], points[:, 0]) - reference
    relative = (bearings + np.pi) % (2 * np.pi) - np.pi
    return points[int(np.argmax(relative))], points[int(np.argmin(relative))]


def cluster_points(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Cluster the points with DBSCAN.

    Two points are neighbors if their distance is <= eps. A point is a core point
    if it has at least min_samples neighbors, the point itself included (sklearn).

    Args:
        points: Array of shape (N, 2) with x/y coordinates in m
        eps: Maximum distance between two neighboring points in m
        min_samples: Minimum number of points in the neighborhood of a core point

    Returns:
        Array of shape (N,) with the cluster label per point, -1 marks noise.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        # sklearn raises an error for empty input
        return np.empty(0, dtype=int)

    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)


def extract_objects(points: np.ndarray, labels: np.ndarray) -> list[DetectedObject]:
    """Build one object per cluster, noise points (label -1) are ignored.

    Args:
        points: Array of shape (N, 2) with x/y coordinates in m
        labels: Array of shape (N,) with the cluster label per point

    Returns:
        List of the detected objects, ordered by cluster label.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    labels = np.asarray(labels)

    objects = []
    for label in np.unique(labels):
        if label == -1:
            continue
        cluster = points[labels == label]
        centroid = cluster.mean(axis=0)
        minimum = cluster.min(axis=0)
        maximum = cluster.max(axis=0)
        left, right = contour_endpoints(cluster)
        objects.append(
            DetectedObject(
                centroid_x=float(centroid[0]),
                centroid_y=float(centroid[1]),
                min_x=float(minimum[0]),
                max_x=float(maximum[0]),
                min_y=float(minimum[1]),
                max_y=float(maximum[1]),
                # Single-point clusters are possible with min_samples = 1, pdist
                # then returns an empty array and the diameter is 0
                diameter=float(pdist(cluster).max(initial=0.0)),
                left_x=float(left[0]),
                left_y=float(left[1]),
                right_x=float(right[0]),
                right_y=float(right[1]),
                points=cluster,
            )
        )
    return objects


def filter_objects(
    objects: list[DetectedObject], max_diameter: float
) -> list[DetectedObject]:
    """Drop objects that are too large to be an obstacle (e.g. walls).

    The diameter is used instead of the bounding box, because the box of a rotated
    object is smaller than its length (e.g. 0.57 x 0.57 m for a 0.8 m line at 45°).

    Args:
        objects: The detected objects
        max_diameter: Maximum diameter of an object in m

    Returns:
        The objects whose diameter is <= max_diameter.
    """
    return [obj for obj in objects if obj.diameter <= max_diameter]
