"""Clustering of 2D points into obstacle candidates (independent of ROS)."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN


@dataclass(frozen=True)
class DetectedObject:
    """Centroid, axis-aligned bounding box and diameter of a point cluster.

    All values are in m and given in the scan frame. The diameter is the largest
    distance between two points of the cluster and, unlike the box, does not depend
    on the rotation of the object.
    """

    centroid_x: float
    centroid_y: float
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    num_points: int
    diameter: float

    @property
    def extent_x(self) -> float:
        """Return the size of the box along x in m."""
        return self.max_x - self.min_x

    @property
    def extent_y(self) -> float:
        """Return the size of the box along y in m."""
        return self.max_y - self.min_y


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
        objects.append(
            DetectedObject(
                centroid_x=float(centroid[0]),
                centroid_y=float(centroid[1]),
                min_x=float(minimum[0]),
                max_x=float(maximum[0]),
                min_y=float(minimum[1]),
                max_y=float(maximum[1]),
                num_points=len(cluster),
                # Single-point clusters are possible with min_samples = 1, pdist
                # then returns an empty array and the diameter is 0
                diameter=float(pdist(cluster).max(initial=0.0)),
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
