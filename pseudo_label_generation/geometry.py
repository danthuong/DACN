"""Geometry used by the GPro3D-style pseudo-label generator.

KITTI camera coordinates are X right, Y down, Z forward. KITTI labels place the
3D location at the bottom centre of an object. The existing MonoDETR KITTI
utility rotates local box points with::

    [[cos(ry), 0,  sin(ry)],
     [0,       1,  0      ],
     [-sin(ry),0,  cos(ry)]]

This is the paper matrix evaluated at ``theta_paper = -rotation_y``.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


TOUCH_POINT_NAMES = ("LF", "RF", "RR", "LR")


def kitti_y_rotation(rotation_y: float) -> np.ndarray:
    """Return KITTI's local-object to rect-camera Y rotation."""
    cosine = np.cos(rotation_y)
    sine = np.sin(rotation_y)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def touch_points_local(length: float, width: float, k_l: float, k_w: float) -> np.ndarray:
    """Return LF, RF, RR, LR candidates in the paper's local convention."""
    half_length = 0.5 * k_l * length
    half_width = 0.5 * k_w * width
    return np.asarray(
        [
            [half_length, 0.0, half_width],
            [half_length, 0.0, -half_width],
            [-half_length, 0.0, -half_width],
            [-half_length, 0.0, half_width],
        ],
        dtype=np.float64,
    )


def transform_touch_points(
    length: float,
    width: float,
    location: Sequence[float],
    rotation_y: float,
    k_l: float,
    k_w: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create local candidates and transform them to rect-camera coordinates."""
    local = touch_points_local(length, width, k_l, k_w)
    camera = local @ kitti_y_rotation(rotation_y).T + np.asarray(location, dtype=np.float64)
    return local, camera


def project_points(points_camera: np.ndarray, p2: np.ndarray, epsilon: float = 1e-6):
    """Project rect-camera points with the complete KITTI P2 matrix.

    Returns pixel coordinates (NaN for invalid points) and a validity mask. A
    point must be finite, in front of the rect camera, and have a positive
    projective denominator.
    """
    points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    projection = np.asarray(p2, dtype=np.float64).reshape(3, 4)
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    image_h = homogeneous @ projection.T
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(image_h).all(axis=1)
        & (points[:, 2] > epsilon)
        & (image_h[:, 2] > epsilon)
    )
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    pixels[valid] = image_h[valid, :2] / image_h[valid, 2:3]
    valid &= np.isfinite(pixels).all(axis=1)
    return pixels, valid


def backproject_pixels_to_plane(
    pixels: np.ndarray,
    p2: np.ndarray,
    plane_coefficients: Sequence[float],
    epsilon: float = 1e-8,
):
    """Intersect image rays with a plane in rect-camera coordinates.

    The complete ``P2 = [M | p4]`` matrix is used, including its translation
    column.  A ray starts at ``-M^-1 p4`` and passes through the supplied
    image pixel.  Invalid, parallel, or behind-camera intersections are
    returned as NaN and marked false in the validity mask.
    """
    image_points = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    projection = np.asarray(p2, dtype=np.float64).reshape(3, 4)
    plane = np.asarray(plane_coefficients, dtype=np.float64).reshape(4)
    intersections = np.full((len(image_points), 3), np.nan, dtype=np.float64)
    valid = np.isfinite(image_points).all(axis=1)
    if not np.isfinite(projection).all() or not np.isfinite(plane).all():
        return intersections, np.zeros(len(image_points), dtype=bool)

    matrix = projection[:, :3]
    try:
        camera_center = -np.linalg.solve(matrix, projection[:, 3])
        homogeneous_pixels = np.column_stack(
            (image_points, np.ones(len(image_points), dtype=np.float64))
        )
        ray_directions = np.linalg.solve(matrix, homogeneous_pixels.T).T
    except np.linalg.LinAlgError:
        return intersections, np.zeros(len(image_points), dtype=bool)

    normal, offset = plane[:3], float(plane[3])
    denominators = ray_directions @ normal
    valid &= np.isfinite(ray_directions).all(axis=1) & (np.abs(denominators) > epsilon)
    scales = np.full(len(image_points), np.nan, dtype=np.float64)
    scales[valid] = -(float(camera_center @ normal) + offset) / denominators[valid]
    candidate_points = camera_center + scales[:, None] * ray_directions
    valid &= (
        np.isfinite(candidate_points).all(axis=1)
        & (scales > epsilon)
        & (candidate_points[:, 2] > epsilon)
    )
    intersections[valid] = candidate_points[valid]
    return intersections, valid


def _largest_y_consensus(points: np.ndarray, max_y_gap: Optional[float]) -> np.ndarray:
    """Indices of the largest group whose camera-Y range fits ``max_y_gap``.

    KITTI camera Y points down, so this is a bottom-height consistency check,
    not an object-height check. Ties prefer the group with the smaller Y range.
    """
    if len(points) == 0:
        return np.asarray([], dtype=np.int64)
    if max_y_gap is None:
        return np.arange(len(points), dtype=np.int64)
    if not np.isfinite(max_y_gap) or max_y_gap <= 0:
        raise ValueError("max_y_gap must be positive and finite")

    order = np.argsort(points[:, 1], kind="stable")
    best = order[:1]
    left = 0
    for right in range(len(order)):
        while points[order[right], 1] - points[order[left], 1] > max_y_gap:
            left += 1
        candidate = order[left : right + 1]
        candidate_gap = float(np.ptp(points[candidate, 1])) if len(candidate) > 1 else 0.0
        best_gap = float(np.ptp(points[best, 1])) if len(best) > 1 else 0.0
        if len(candidate) > len(best) or (len(candidate) == len(best) and candidate_gap < best_gap):
            best = candidate
    return np.sort(best)


def fit_ground_plane(
    points: Iterable[Sequence[float]],
    epsilon: float = 1e-8,
    max_y_gap: Optional[float] = None,
    min_xz_spread_ratio: float = 0.0,
) -> Dict:
    """Fit ``aX+bY+cZ+d=0`` with centred SVD.

    At least three finite, non-collinear samples are required. The normal is
    normalised and oriented so ``b <= 0`` for stable serialised output.
    """
    candidates = np.asarray(list(points), dtype=np.float64).reshape(-1, 3)
    result = {
        "valid": False,
        "coefficients": None,
        "rms_error": None,
        "candidate_count": len(candidates),
        "sample_indices": [],
        "rejected_indices": [],
        "bottom_y_gap": None,
        "xz_spread_ratio": None,
    }
    if not np.isfinite(min_xz_spread_ratio) or min_xz_spread_ratio < 0 or min_xz_spread_ratio >= 1:
        raise ValueError("min_xz_spread_ratio must be finite and in [0, 1)")
    if not np.isfinite(candidates).all():
        result["reason"] = "non_finite_samples"
        return result

    selected_indices = _largest_y_consensus(candidates, max_y_gap)
    selected_set = set(selected_indices.tolist())
    result["sample_indices"] = selected_indices.tolist()
    result["rejected_indices"] = [index for index in range(len(candidates)) if index not in selected_set]
    array = candidates[selected_indices]
    if len(array):
        result["bottom_y_gap"] = float(np.ptp(array[:, 1]))
    if len(array) < 3:
        result["reason"] = "fewer_than_three_samples"
        return result

    centered_xz = array[:, [0, 2]] - array[:, [0, 2]].mean(axis=0)
    xz_singular_values = np.linalg.svd(centered_xz, compute_uv=False)
    if xz_singular_values[0] <= epsilon:
        result["xz_spread_ratio"] = 0.0
        result["reason"] = "insufficient_xz_spread"
        return result
    xz_spread_ratio = float(xz_singular_values[-1] / xz_singular_values[0])
    result["xz_spread_ratio"] = xz_spread_ratio
    if xz_spread_ratio < min_xz_spread_ratio:
        result["reason"] = "insufficient_xz_spread"
        return result

    centroid = array.mean(axis=0)
    _, singular_values, vh = np.linalg.svd(array - centroid, full_matrices=False)
    scale = max(float(singular_values[0]), 1.0)
    if len(singular_values) < 2 or singular_values[1] <= epsilon * scale:
        result["reason"] = "collinear_samples"
        return result

    normal = vh[-1]
    norm = np.linalg.norm(normal)
    if not np.isfinite(norm) or norm <= epsilon:
        result["reason"] = "degenerate_normal"
        return result
    normal = normal / norm
    if normal[1] > 0:
        normal = -normal
    d = -float(np.dot(normal, centroid))
    residuals = array @ normal + d
    result.update(
        {
            "valid": True,
            "coefficients": [float(normal[0]), float(normal[1]), float(normal[2]), d],
            "rms_error": float(np.sqrt(np.mean(residuals**2))),
        }
    )
    return result


def clip_line_to_image(
    slope: float, intercept: float, width: int, height: int, epsilon: float = 1e-9
) -> Optional[List[List[float]]]:
    """Clip ``v=slope*u+intercept`` to an image rectangle."""
    if width <= 0 or height <= 0 or not np.isfinite([slope, intercept]).all():
        return None
    xmax, ymax = float(width - 1), float(height - 1)
    candidates: List[np.ndarray] = []

    for u in (0.0, xmax):
        v = slope * u + intercept
        if -epsilon <= v <= ymax + epsilon:
            candidates.append(np.asarray([u, np.clip(v, 0.0, ymax)]))
    if abs(slope) > epsilon:
        for v in (0.0, ymax):
            u = (v - intercept) / slope
            if -epsilon <= u <= xmax + epsilon:
                candidates.append(np.asarray([np.clip(u, 0.0, xmax), v]))

    unique: List[np.ndarray] = []
    for point in candidates:
        if not any(np.linalg.norm(point - other) < 1e-6 for other in unique):
            unique.append(point)
    if len(unique) < 2:
        return None

    best_pair = max(
        ((left, right) for i, left in enumerate(unique) for right in unique[i + 1 :]),
        key=lambda pair: float(np.sum((pair[0] - pair[1]) ** 2)),
    )
    return [[float(value) for value in point] for point in best_pair]


def derive_horizon_line(
    plane_coefficients: Sequence[float], p2: np.ndarray, image_size: Sequence[int], epsilon: float = 1e-8
) -> Dict:
    """Derive GPro3D's ``v=k_h*u+b_h`` from a fitted ground plane."""
    a, b, c, d = [float(value) for value in plane_coefficients]
    result = {"valid": False, "slope": None, "intercept": None, "visible_segment": None}
    if abs(b) <= epsilon:
        result["reason"] = "plane_b_is_zero"
        return result

    projection = np.asarray(p2, dtype=np.float64).reshape(3, 4)
    fx, fy = float(projection[0, 0]), float(projection[1, 1])
    cx, cy = float(projection[0, 2]), float(projection[1, 2])
    if abs(fx) <= epsilon or abs(fy) <= epsilon:
        result["reason"] = "invalid_intrinsics"
        return result

    plane_a, plane_b, plane_c = -a / b, -c / b, -d / b
    slope = plane_a * fy / fx
    intercept = cy + plane_b * fy - slope * cx
    width, height = int(image_size[0]), int(image_size[1])
    result.update(
        {
            "valid": bool(np.isfinite([slope, intercept]).all()),
            "slope": float(slope),
            "intercept": float(intercept),
            "ground_y_form": [float(plane_a), float(plane_b), float(plane_c)],
            "raw_endpoints": [[0.0, float(intercept)], [float(width - 1), float(slope * (width - 1) + intercept)]],
            "visible_segment": clip_line_to_image(slope, intercept, width, height),
        }
    )
    if not result["valid"]:
        result["reason"] = "non_finite_horizon"
    elif result["visible_segment"] is None:
        result["reason"] = "line_outside_image"
    return result

