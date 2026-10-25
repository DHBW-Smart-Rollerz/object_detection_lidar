"""Conversion of 2D LiDAR scans into Cartesian points (independent of ROS)."""

import numpy as np


def scan_to_points(
    ranges: np.ndarray,
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
) -> np.ndarray:
    """Convert the ranges of a 2D laser scan into x/y points in the sensor frame.

    Invalid measurements (inf, nan) and ranges outside [range_min, range_max] are
    dropped.

    Args:
        ranges: Measured distances in m, one per beam
        angle_min: Angle of the first beam in rad
        angle_increment: Angle between two beams in rad
        range_min: Minimum valid range in m
        range_max: Maximum valid range in m

    Returns:
        Array of shape (N, 2) with the x/y coordinates of the valid points in m.
    """
    ranges = np.asarray(ranges, dtype=np.float64)
    angles = angle_min + np.arange(ranges.size) * angle_increment

    valid = np.isfinite(ranges) & (ranges >= range_min) & (ranges <= range_max)
    ranges = ranges[valid]
    angles = angles[valid]

    return np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))
