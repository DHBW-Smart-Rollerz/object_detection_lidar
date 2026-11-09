"""Splitting of the scan at breakpoints (independent of ROS).

DBSCAN only looks at the distances between points. Two obstacles closer than eps
end up in one cluster, even if the scan clearly shows a gap between them (the beams
in between hit the background) or a jump in depth. The breakpoint detection walks
along the beams of each cluster and splits it where two consecutive points cannot
belong to the same surface: Adaptive Breakpoint Detector of Borges and Aldon,
"Line extraction in 2D range images for mobile robotics", Journal of Intelligent
and Robotic Systems 40, 2004 (split_clusters). segment_scan applies the same
criteria to all points of the scan and replaces DBSCAN (segmentation "abd").
"""

import numpy as np


def breakpoint_distance(
    previous_range: np.ndarray,
    angle_gap: np.ndarray,
    lambda_angle: float,
    sigma_r: float,
) -> np.ndarray:
    """Return the max. distance of two consecutive points of the same surface.

    Adaptive Breakpoint Detector (Borges and Aldon 2004): a surface seen under an
    angle of incidence of at least lambda_angle produces consecutive points at most
    D_max = r * sin(dphi) / sin(lambda - dphi) + 3 * sigma_r apart, r being the
    range of the first point and dphi the angle between the two beams. Points
    farther apart belong to different surfaces (or to a surface seen so obliquely
    that it is not measured reliably). D_max grows with the range, so the same
    parameters work near and far. For dphi >= lambda the formula has no meaning
    (the distance would be unbounded), the result is then -inf (always a
    breakpoint).

    Args:
        previous_range: Range of the first point of each pair in m
        angle_gap: Angle between the two beams of each pair in rad (>= 0)
        lambda_angle: Smallest angle of incidence of a surface in rad
        sigma_r: Standard deviation of the range measurement in m

    Returns:
        Array with D_max in m per pair, same shape as the inputs.
    """
    previous_range = np.asarray(previous_range, dtype=np.float64)
    angle_gap = np.asarray(angle_gap, dtype=np.float64)
    valid = angle_gap < lambda_angle
    # The denominator is only evaluated where it is positive
    denominator = np.sin(np.where(valid, lambda_angle - angle_gap, 1.0))
    distance = previous_range * np.sin(angle_gap) / denominator + 3.0 * sigma_r
    return np.where(valid, distance, -np.inf)


def pair_breaks(
    points: np.ndarray,
    beam_indices: np.ndarray,
    current: np.ndarray,
    after: np.ndarray,
    *,
    num_beams: int,
    angle_increment: float,
    lambda_angle: float,
    sigma_r: float,
    max_beam_gap: int,
    returns: np.ndarray | None,
    single: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Check pairs of consecutive points (by beam) for a breakpoint.

    A pair is separated by a breakpoint if the distance of its points is above
    breakpoint_distance (with the actual angle between their beams) or if more
    than max_beam_gap beams lie between them. With returns only the beams with a
    return count, otherwise every beam. The beams form a circle: the angle and the
    beams between a point and a point on an earlier beam are counted across the
    seam of the scan.

    Args:
        points: Array of shape (N, 2) with x/y coordinates in m
        beam_indices: Array of shape (N,) with the beam index of each point
        current: Array of shape (P,) with the index of the first point per pair
        after: Array of shape (P,) with the index of the second point per pair
        num_beams: Number of beams of the scan
        angle_increment: Angle between two beams in rad
        lambda_angle: Smallest angle of incidence of a surface in rad
        sigma_r: Standard deviation of the range measurement in m
        max_beam_gap: Max. number of beams between the two points of a pair
        returns: Array of shape (num_beams,), True for the beams that count, None
            counts every beam
        single: Array of shape (P,), True for pairs of a point with itself (the
            full circle lies in between)

    Returns:
        The breakpoints (True) and the angles between the beams in rad
        ([0, 2 pi)), both arrays of shape (P,).
    """
    direction = 1.0 if angle_increment >= 0 else -1.0
    angle_gaps = (
        direction * angle_increment * (beam_indices[after] - beam_indices[current])
    ) % (2 * np.pi)
    if returns is None:
        between = (beam_indices[after] - beam_indices[current]) % num_beams - 1
    else:
        counts = np.r_[0, np.cumsum(np.asarray(returns, dtype=bool))]
        a, b = beam_indices[current], beam_indices[after]
        between = np.where(
            b > a, counts[b] - counts[a + 1], counts[-1] - counts[a + 1] + counts[b]
        )
    angle_gaps[single] = 2 * np.pi

    ranges = np.hypot(points[current, 0], points[current, 1])
    distances = np.hypot(*(points[after] - points[current]).T)
    max_distances = breakpoint_distance(ranges, angle_gaps, lambda_angle, sigma_r)
    breaks = (distances > max_distances) | (between > max_beam_gap)
    return breaks, angle_gaps


def split_clusters(
    points: np.ndarray,
    beam_indices: np.ndarray,
    labels: np.ndarray,
    *,
    num_beams: int,
    angle_increment: float,
    lambda_angle: float,
    sigma_r: float,
    max_beam_gap: int,
    min_samples: int,
    returns: np.ndarray | None = None,
) -> np.ndarray:
    """Split the clusters at breakpoints between consecutive beams.

    The points of each cluster are sorted by their beam. Two consecutive points are
    separated by a breakpoint if

    - their distance is above breakpoint_distance (jump in depth), with the actual
      angle between their beams, so skipped beams enlarge the allowed distance, or
    - more than max_beam_gap foreign beams lie between them. These beams hit points
      outside of the cluster (background, other objects, noise), so something
      else is seen between the two points. If returns is given, only beams with a
      return count, beams without one (inf, nan) are bridged: dark or shiny spots
      on an obstacle return nothing, and the beams between two obstacles hit the
      background. Without returns all beams between the two points count.

    The beams form a circle: a cluster across the seam of the scan (the last and
    the first beam, around ±pi) is walked across the seam, the beams there are
    neighbors (the seam itself is no gap of beams, only of angle). The walk starts
    after the largest angle between two points of the cluster. Segments with fewer
    than min_samples points become noise, like small clusters in DBSCAN.

    Args:
        points: Array of shape (N, 2) with x/y coordinates in m
        beam_indices: Array of shape (N,) with the index of the beam of each point
            in the scan (unique)
        labels: Array of shape (N,) with the cluster label per point, -1 is noise
        num_beams: Number of beams of the scan
        angle_increment: Angle between two beams in rad
        lambda_angle: Smallest angle of incidence of a surface in rad
        sigma_r: Standard deviation of the range measurement in m
        max_beam_gap: Max. number of beams between two points of one cluster
        min_samples: Min. number of points of a segment
        returns: Array of shape (num_beams,), True for the beams with a return of
            the sensor (also those beyond the valid range of the points), None
            counts every beam

    Returns:
        Array of shape (N,) with the new cluster labels (0, 1, ... in the order of
        the old labels), -1 marks noise.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    beam_indices = np.asarray(beam_indices, dtype=np.int64)
    labels = np.asarray(labels)
    new_labels = np.full(len(labels), -1, dtype=int)

    # All clustered points, sorted by cluster and within each cluster by beam
    clustered = np.flatnonzero(labels != -1)
    order = clustered[np.lexsort((beam_indices[clustered], labels[clustered]))]
    if len(order) == 0:
        return new_labels
    sorted_labels = labels[order]
    starts = np.flatnonzero(np.r_[True, sorted_labels[1:] != sorted_labels[:-1]])
    ends = np.r_[starts[1:], len(order)]  # exclusive
    cluster = np.repeat(np.arange(len(starts)), ends - starts)

    # Pair of each point with the next one of its cluster around the circle, the
    # last point of a cluster is paired with the first one
    following = np.arange(1, len(order) + 1)
    following[ends - 1] = starts
    current, after = order, order[following]
    breaks, angle_gaps = pair_breaks(
        points,
        beam_indices,
        current,
        after,
        num_beams=num_beams,
        angle_increment=angle_increment,
        lambda_angle=lambda_angle,
        sigma_r=sigma_r,
        max_beam_gap=max_beam_gap,
        returns=returns,
        single=(ends - starts == 1)[cluster],
    )
    # The largest angle of each cluster lies outside of it, the walk starts there
    largest = np.maximum.reduceat(angle_gaps, starts)
    breaks |= angle_gaps == largest[cluster]

    # Segment of each point: number of breaks (and cluster ends) before it. The
    # run before the first break of a cluster continues its last run if the circle
    # is not broken between the last and the first point (cluster across the seam)
    boundaries = breaks.copy()
    boundaries[ends - 1] = True
    segments = np.r_[0, np.cumsum(boundaries)[:-1]]
    closed = ~breaks[ends - 1]
    join = (segments == segments[starts][cluster]) & closed[cluster]
    segments[join] = segments[ends - 1][cluster][join]

    sizes = np.bincount(segments)
    keep = sizes[segments] >= min_samples
    _, renumbered = np.unique(segments[keep], return_inverse=True)
    new_labels[order[keep]] = renumbered
    return new_labels


def segment_scan(
    points: np.ndarray,
    beam_indices: np.ndarray,
    *,
    num_beams: int,
    angle_increment: float,
    lambda_angle: float,
    sigma_r: float,
    max_beam_gap: int,
    min_samples: int,
    returns: np.ndarray | None = None,
) -> np.ndarray:
    """Segment all points of the scan at breakpoints (without DBSCAN).

    The points are walked in the order of their beams around the full circle and
    split between two consecutive points with the criteria of split_clusters: the
    distance is above breakpoint_distance or more than max_beam_gap beams lie
    between them. Here every valid point takes part, so the beams between two
    consecutive points never hit another valid point. They are beams without a
    return (inf, nan; bridged unless returns is None) or beams whose return lies
    outside of the valid range of the points (e.g. background beyond max_range).
    The second criterion therefore finds gaps to the far background, gaps to
    background within the valid range show up as two jumps in depth instead.

    The walk is closed across the seam of the scan (around ±pi): if the last and
    the first point are not separated, their segments are one. Without any
    breakpoint all points form one segment. Segments with fewer than min_samples
    points become noise.

    Args:
        points: Array of shape (N, 2) with x/y coordinates in m
        beam_indices: Array of shape (N,) with the index of the beam of each point
            in the scan (unique)
        num_beams: Number of beams of the scan
        angle_increment: Angle between two beams in rad
        lambda_angle: Smallest angle of incidence of a surface in rad
        sigma_r: Standard deviation of the range measurement in m
        max_beam_gap: Max. number of beams between two points of one segment
        min_samples: Min. number of points of a segment
        returns: Array of shape (num_beams,), True for the beams with a return of
            the sensor (also those beyond the valid range of the points), None
            counts every beam

    Returns:
        Array of shape (N,) with the segment label per point (0, 1, ... in the
        order of the beams, starting after the first breakpoint), -1 marks noise.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    beam_indices = np.asarray(beam_indices, dtype=np.int64)
    labels = np.full(len(points), -1, dtype=int)
    if len(points) == 0:
        return labels

    order = np.argsort(beam_indices, kind="stable")
    breaks, _ = pair_breaks(
        points,
        beam_indices,
        order,
        np.roll(order, -1),
        num_beams=num_beams,
        angle_increment=angle_increment,
        lambda_angle=lambda_angle,
        sigma_r=sigma_r,
        max_beam_gap=max_beam_gap,
        returns=returns,
        single=np.full(len(order), len(order) == 1),
    )

    # Segment of each point: number of breaks before it, the run before the first
    # break continues the last run if the circle is closed across the seam
    segments = np.r_[0, np.cumsum(breaks)[:-1]]
    if not breaks[-1]:
        segments[segments == 0] = segments[-1]

    sizes = np.bincount(segments)
    keep = sizes[segments] >= min_samples
    _, renumbered = np.unique(segments[keep], return_inverse=True)
    labels[order[keep]] = renumbered
    return labels
