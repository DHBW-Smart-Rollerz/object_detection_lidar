"""Classification of point clusters as round or angular (independent of ROS).

All obstacles of the CAuDri track (incl. the pedestrian figures) are angular
boxes. The 2D scan sees one side (a line) or a corner (two lines) of them. Round
contours come from objects next to or on the track that are no obstacles: legs
of people, bottles, chair legs. A cluster is round if a circle fits its visible
contour clearly better than a line or a corner and the circle is plausible (small
radius, bulges toward the sensor, noticeable arc).

Recall comes first: a cluster is only round if all criteria hold. Too few points,
a large radius or an ambiguous fit keep the object.
"""

import numpy as np

from object_detection_lidar.box_classification import fit_line, order_by_angle

# in m, smaller circles are no plausible objects (thinner than a chair leg),
# such fits come from a few noisy points
MIN_RADIUS = 0.01


def fit_circle(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Fit a circle through the points.

    Algebraic fit of Taubin (1991) in the SVD form of Chernov (2010): unlike the
    simpler fit of Kasa it is nearly unbiased for short and noisy arcs, so no
    iterative geometric refinement is needed (which would cost ~5 times more).

    Args:
        points: Array of shape (N, 2) with x/y coordinates in m, N >= 3

    Returns:
        Center, radius and the RMS of the distances of the points to the circle.
        The radius is inf for collinear points.
    """
    offset = points.mean(axis=0)
    p = points - offset  # better conditioned
    z = (p**2).sum(axis=1)
    z_mean = float(z.mean())
    if z_mean <= 0:
        return offset, np.inf, np.inf
    scale = 2 * np.sqrt(z_mean)
    # Circle a (x² + y²) + b x + c y + d = 0, d = -z_mean * a after centering
    _, _, vh = np.linalg.svd(
        np.column_stack(((z - z_mean) / scale, p)), full_matrices=False
    )
    a, b, c = vh[-1]
    a /= scale
    if abs(a) < np.finfo(float).eps:
        return offset, np.inf, np.inf
    center = np.array([-b, -c]) / (2 * a)
    radius = float(np.sqrt(b * b + c * c + 4 * a * a * z_mean) / (2 * abs(a)))
    rms = float(np.sqrt(np.mean((np.linalg.norm(p - center, axis=1) - radius) ** 2)))
    return center + offset, radius, rms


def best_corner(points: np.ndarray, min_side_points: int = 3) -> tuple[float, float]:
    """Find the best split of an ordered contour into two lines.

    Every split index is tried, the corner point belongs to both sides. The
    residual of a total least squares line is the smaller eigenvalue of the
    scatter matrix, which is computed for all splits at once from cumulative sums.

    Args:
        points: Array of shape (N, 2) with x/y coordinates in m, ordered along
            the contour
        min_side_points: Min. number of points of each side

    Returns:
        The RMS distance of all points to their line in m and the angle between
        both lines in deg, folded into [0°, 90°] (90° is a right angle). inf and
        0.0 if the contour is too short for two sides.
    """
    n = len(points)
    if n < 2 * min_side_points - 1:
        return np.inf, 0.0
    p = points - points.mean(axis=0)
    x, y = p[:, 0], p[:, 1]
    columns = np.column_stack((np.ones(n), x, y, x * x, y * y, x * y))
    prefix = np.cumsum(columns, axis=0)
    total = prefix[-1]

    def residual(sums: np.ndarray) -> np.ndarray:
        count, sx, sy, sxx, syy, sxy = sums.T
        cxx = sxx - sx * sx / count
        cyy = syy - sy * sy / count
        cxy = sxy - sx * sy / count
        # Smaller eigenvalue of [[cxx, cxy], [cxy, cyy]] = sum of squared residuals
        return np.maximum(
            (cxx + cyy) / 2 - np.sqrt(((cxx - cyy) / 2) ** 2 + cxy**2), 0.0
        )

    corners = np.arange(min_side_points - 1, n - min_side_points + 1)
    first = prefix[corners]  # points 0..corner
    second = total - prefix[corners] + columns[corners]  # points corner..n-1
    sse = residual(first) + residual(second)
    corner = int(corners[np.argmin(sse)])
    direction_first, _ = fit_line(points[: corner + 1])
    direction_second, _ = fit_line(points[corner:])
    cos_angle = abs(float(direction_first @ direction_second))
    angle = float(np.degrees(np.arccos(min(cos_angle, 1.0))))
    # The corner point is counted twice
    return float(np.sqrt(sse.min() / (n + 1))), angle


def arc_angle(points: np.ndarray, center: np.ndarray) -> float:
    """Return the angle of the arc covered by the points seen from the center.

    Args:
        points: Array of shape (N, 2) with x/y coordinates in m
        center: Center of the circle

    Returns:
        The angle between the outermost points in rad.
    """
    delta = points - center
    reference = delta.mean(axis=0)
    angles = np.arctan2(delta[:, 1], delta[:, 0]) - np.arctan2(
        reference[1], reference[0]
    )
    return float(np.ptp((angles + np.pi) % (2 * np.pi) - np.pi))


def is_round(
    cluster: np.ndarray,
    *,
    min_points: int,
    max_radius: float,
    min_arc_deg: float,
    noise: float,
    noise_per_m: float,
    max_circle_rms_factor: float,
    min_line_ratio: float,
    min_corner_ratio: float,
    right_angle_tolerance: float,
    min_right_corner_ratio: float,
) -> bool:
    """Check whether the visible contour of a cluster is round.

    All of the following must hold, otherwise the cluster is not round (recall
    first, an ambiguous object is kept):
    - at least min_points points (with few points every shape fits),
    - the radius lies in [MIN_RADIUS, max_radius],
    - the arc bulges toward the sensor (center behind the contour by at least
      half the radius) and covers at least min_arc_deg seen from its center (a
      straight side with noise fits a huge circle or a short arc),
    - the circle fits within the noise: circle RMS at most
      max_circle_rms_factor * (noise + noise_per_m * distance),
    - the circle fits clearly better than a line: line RMS above the noise and
      at least min_line_ratio times the circle RMS,
    - a corner does not fit clearly better than the circle: corner RMS at least
      min_corner_ratio times the circle RMS. Two lines approximate the visible
      arc of a cylinder well (two chords of ~60° each), so the corner RMS of
      cylinders is about as small as the circle RMS, while the corner RMS of real
      corners is well below the circle RMS,
    - the corner is no right angle that fits about as well as the circle: if the
      angle between both lines deviates at most right_angle_tolerance from 90°,
      the corner RMS must be at least min_right_corner_ratio times the circle
      RMS. With few points and the noise of far objects the corner of a small box
      fits a circle as well as two lines, but the two lines are at a right angle.
      The best two lines through the visible arc of a cylinder (usually ~110-130°
      wide, the grazing ends return nothing) are at ~60-75° and only near 90° for
      close cylinders with a wider visible arc, whose corner RMS is clearly above
      the circle RMS.

    Args:
        cluster: Array of shape (N, 2) with x/y coordinates in m (sensor at 0, 0)
        min_points: Min. number of points
        max_radius: Max. radius of the circle in m
        min_arc_deg: Min. angle of the visible arc seen from the center in deg
        noise: Range noise at 0 m in m
        noise_per_m: Increase of the range noise per m of distance
        max_circle_rms_factor: Max. circle RMS as multiple of the noise
        min_line_ratio: Min. ratio of line RMS to circle RMS
        min_corner_ratio: Min. ratio of corner RMS to circle RMS
        right_angle_tolerance: Max. deviation of a right corner from 90° in deg
        min_right_corner_ratio: Min. ratio of corner RMS to circle RMS for a
            right corner

    Returns:
        True if the cluster is round.
    """
    if len(cluster) < min_points:
        return False
    # Cheap checks first: straight clusters (most of them) need no circle fit
    distance = float(np.linalg.norm(cluster.mean(axis=0)))
    sigma = noise + noise_per_m * distance
    _, line_rms = fit_line(cluster)
    if line_rms <= sigma:
        return False
    points, _ = order_by_angle(cluster)
    center, radius, circle_rms = fit_circle(points)
    plausible = (
        MIN_RADIUS <= radius <= max_radius
        and np.linalg.norm(center) >= distance + 0.5 * radius
        and np.degrees(arc_angle(points, center)) >= min_arc_deg
        and circle_rms <= max_circle_rms_factor * sigma
        and line_rms >= min_line_ratio * circle_rms
    )
    if not plausible:
        return False
    corner_rms, corner_angle = best_corner(points)
    right = 90.0 - corner_angle <= right_angle_tolerance
    min_ratio = max(min_corner_ratio, min_right_corner_ratio if right else 0.0)
    return bool(corner_rms >= min_ratio * circle_rms)
