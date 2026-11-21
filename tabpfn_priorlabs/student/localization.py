"""T2 - 3D localisation in the base frame: T2.1 a detected cube, T2.2 the bin."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from pickplace.interfaces import BinInfo

from .perception import Detection
from .transforms import backproject_depth, transform_points


@dataclass
class CubeEstimate:
    color: str
    position: np.ndarray   # (3,) cube CENTRE in the base frame [m]
    yaw: float             # rotation about base z, reduced to (-pi/4, pi/4] (cube is 90deg-symmetric)
    n_points: int          # number of depth points used


@dataclass
class BinEstimate:
    position: np.ndarray   # (2,) bin CENTRE (x, y) in the base frame [m]
    yaw: float             # rotation of the bin about base z, reduced to (-pi/4, pi/4] (a square bin is 90deg-symmetric)
    n_points: int          # number of depth points used


def wrap_cube_yaw(yaw: float) -> float:
    """Reduce an angle to the cube's 90-degree symmetry range (-pi/4, pi/4]."""
    y = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return float(np.pi / 4 if np.isclose(y, -np.pi / 4) else y)


def estimate_cube_pose(det: Detection, depth: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray,
                       cube_size: float) -> CubeEstimate | None:
    """
    Estimate a table-supported cube from its RGB mask and optical-axis depth.

    Find the dominant horizontal top-face height near the known cube height,
    then fit its square footprint using robust projected bounds. Subtract half
    the cube size from the top height to obtain the centre, not the top surface.
    Return None for insufficient data or a footprint inconsistent with the size.
    Only supplied observations, calibration and cube dimensions are used.
    """
    depth = np.asarray(depth)
    mask = np.asarray(det.mask, dtype=bool)
    K = np.asarray(K, dtype=float)
    T_base_cam = np.asarray(T_base_cam, dtype=float)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("depth and detection mask must have the same (H, W) shape")
    if K.shape != (3, 3) or T_base_cam.shape != (4, 4):
        raise ValueError("K must be (3, 3) and T_base_cam must be (4, 4)")
    if not np.isfinite(cube_size) or cube_size <= 0:
        raise ValueError("cube_size must be positive and finite")
    if not np.isfinite(K).all() or not np.isfinite(T_base_cam).all():
        return None
    if abs(np.linalg.det(K)) < 1e-12:
        return None

    points_cam, _ = backproject_depth(depth, K, mask)
    points = transform_points(T_base_cam, points_cam)
    points = points[np.isfinite(points).all(axis=1)]
    # The exercise places cubes on the z=0 table. Reject background and low
    # side-face points before looking for the horizontal top surface.
    candidates = points[(points[:, 2] >= 0.5 * cube_size)
                        & (points[:, 2] <= 1.5 * cube_size)]
    if len(candidates) < 12:
        return None
    bins = max(8, int(np.ceil(cube_size / 0.002)))
    counts, edges = np.histogram(candidates[:, 2], bins=bins,
                                 range=(0.5 * cube_size, 1.5 * cube_size))
    peak = int(np.argmax(counts))
    height = 0.5 * (edges[peak] + edges[peak + 1])
    tolerance = max(0.0025, cube_size * 0.06)
    top = candidates[np.abs(candidates[:, 2] - height) <= tolerance]
    if len(top) < 12:
        return None
    top_height = float(np.median(top[:, 2]))
    if abs(top_height - cube_size) > 0.25 * cube_size:
        return None

    # Search one 90-degree symmetry period. Bounding-box area has a minimum
    # when its axes align with the square edges, unlike square-face PCA.
    angles = np.linspace(-np.pi / 4, np.pi / 4, 181, endpoint=False)
    c, s = np.cos(angles), np.sin(angles)
    origin = np.median(top[:, :2], axis=0)
    xy = top[:, :2] - origin
    u = xy[:, 0, None] * c + xy[:, 1, None] * s
    v = -xy[:, 0, None] * s + xy[:, 1, None] * c
    u_low, u_high = np.quantile(u, [0.01, 0.99], axis=0)
    v_low, v_high = np.quantile(v, [0.01, 0.99], axis=0)
    widths, heights = u_high - u_low, v_high - v_low
    best = int(np.argmin(widths * heights))
    spans = np.array([widths[best], heights[best]])
    if np.any(spans < 0.70 * cube_size) or np.any(spans > 1.25 * cube_size):
        return None
    centre_u = 0.5 * (u_low[best] + u_high[best])
    centre_v = 0.5 * (v_low[best] + v_high[best])
    centre_xy = origin + np.array([c[best] * centre_u - s[best] * centre_v,
                                   s[best] * centre_u + c[best] * centre_v])
    return CubeEstimate(
        color=det.color,
        position=np.r_[centre_xy, top_height - cube_size / 2],
        yaw=wrap_cube_yaw(float(angles[best])),
        n_points=len(top),
    )


@dataclass
class _SquareRingFit:
    centre: np.ndarray
    yaw: float
    inner: np.ndarray
    inliers: np.ndarray
    error: float


def _fit_square_ring(rim: np.ndarray, bin_info: BinInfo) -> _SquareRingFit | None:
    """Fit observed wall-top points to a known-size ring without any pose hint.

    Trim extreme projected coordinates, search a 90-degree orientation period,
    and require the resulting ring to explain at least 85% of the measurements.
    All four sides must have substantial support. The centre prior is a search
    region only; bin_info.xy and bin_info.yaw are never consulted.
    """
    rim = np.asarray(rim, dtype=float)
    if rim.ndim != 2 or rim.shape[1] != 3:
        raise ValueError("rim must have shape (N, 3)")
    if len(rim) < 60 or not np.isfinite(rim).all():
        return None
    inner = np.asarray(bin_info.inner, dtype=float)
    outer = np.asarray(bin_info.outer, dtype=float)
    size_tol = max(0.012, 0.05 * float(outer.max()))
    rim_tol = max(0.004, 0.4 * bin_info.wall_thickness)
    if np.any(np.abs(rim[:, 2] - bin_info.wall_height)
              > max(0.004, 0.05 * bin_info.wall_height)):
        return None
    angles = np.linspace(-np.pi / 4, np.pi / 4, 181, endpoint=False)
    cosines, sines = np.cos(angles), np.sin(angles)
    origin = np.median(rim[:, :2], axis=0)
    xy = rim[:, :2] - origin
    u = xy[:, 0, None] * cosines + xy[:, 1, None] * sines
    v = -xy[:, 0, None] * sines + xy[:, 1, None] * cosines
    low_u, high_u = np.quantile(u, [0.02, 0.98], axis=0)
    low_v, high_v = np.quantile(v, [0.02, 0.98], axis=0)
    widths, heights = high_u - low_u, high_v - low_v
    index = int(np.argmin(widths * heights))
    spans = np.array([widths[index], heights[index]])
    if np.linalg.norm(spans - outer[::-1]) < np.linalg.norm(spans - outer):
        inner, outer = inner[::-1], outer[::-1]
    if np.any(np.abs(spans - outer) > size_tol):
        return None
    c, s = cosines[index], sines[index]
    mid_u = (low_u[index] + high_u[index]) / 2
    mid_v = (low_v[index] + high_v[index]) / 2
    centre = origin + np.array([c * mid_u - s * mid_v, s * mid_u + c * mid_v])
    x_low, x_high = bin_info.search_x
    y_low, y_high = bin_info.search_y_abs
    if not (x_low - 0.005 <= centre[0] <= x_high + 0.005
            and y_low - 0.005 <= abs(centre[1]) <= y_high + 0.005):
        return None
    local = (rim[:, :2] - centre) @ np.array([[c, -s], [s, c]])
    half_inner, half_outer = inner / 2, outer / 2
    in_ring = (np.all(np.abs(local) <= half_outer + rim_tol, axis=1)
               & np.any(np.abs(local) >= half_inner - rim_tol, axis=1))
    if in_ring.mean() < 0.85:
        return None
    for axis in (0, 1):
        other = 1 - axis
        for sign in (-1, 1):
            side = local[in_ring
                         & (np.abs(local[:, axis] - sign * half_outer[axis])
                            <= bin_info.wall_thickness + rim_tol)
                         & (np.abs(local[:, other]) <= half_outer[other] + rim_tol)]
            if (len(side) < max(8, 0.03 * len(rim))
                    or np.ptp(side[:, other]) < 0.70 * inner[other]):
                return None
    error = float(np.linalg.norm(spans - outer)
                  + max(0.004, 0.05 * bin_info.wall_height) * (1.0 - in_ring.mean()))
    return _SquareRingFit(centre, wrap_cube_yaw(float(angles[index])), inner, in_ring, error)


def estimate_bin_pose(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray,
                      bin_info: BinInfo) -> BinEstimate | None:
    """
    Fit a raised rectangular rim and verify the recessed floor using RGB-D geometry.

    Search both allowed sides, using only dimensions and the approximate centre
    prior in bin_info. RGB colour is deliberately not required: the tan bin and
    grey table can look similar. Four supported rim sides, correct outer size,
    and visible floor at the expected height are required for a valid estimate.
    No configured bin pose, simulator state or scene files are accessed.
    """
    # STEP 1: Validate RGB/depth shapes, calibration and known bin dimensions.
    depth = np.asarray(depth)
    rgb = np.asarray(rgb)
    K = np.asarray(K, dtype=float)
    T_base_cam = np.asarray(T_base_cam, dtype=float)
    if depth.ndim != 2 or rgb.shape != (*depth.shape, 3):
        raise ValueError("rgb must have shape (H, W, 3) matching the depth image")
    if rgb.dtype != np.uint8:
        raise TypeError("rgb must have dtype uint8")
    if K.shape != (3, 3) or T_base_cam.shape != (4, 4):
        raise ValueError("K must be (3, 3) and T_base_cam must be (4, 4)")
    inner = np.asarray(bin_info.inner, dtype=float)
    outer = np.asarray(bin_info.outer, dtype=float)
    if (inner.shape != (2,) or not np.isfinite(inner).all() or np.any(inner <= 0)
            or not np.isfinite(bin_info.wall_thickness) or bin_info.wall_thickness <= 0
            or not 0 < bin_info.floor_thickness < bin_info.wall_height):
        raise ValueError("bin dimensions must describe positive walls above a recessed floor")
    if (not np.isfinite(K).all() or not np.isfinite(T_base_cam).all()
            or abs(np.linalg.det(K)) < 1e-12):
        return None

    # STEP 2: Denoise spatially, never across RGB channels. Contrast is optional
    # evidence used only to rank already-valid geometric candidates, not a gate.
    if depth.size == 0:
        return None
    denoised_rgb = ndimage.median_filter(rgb, size=(3, 3, 1), mode="nearest")
    grey = denoised_rgb.astype(float).mean(axis=-1) / 255.0
    edge_strength = np.hypot(ndimage.sobel(grey, axis=0, mode="nearest"),
                             ndimage.sobel(grey, axis=1, mode="nearest"))
    edge_scale = float(np.quantile(edge_strength, 0.99))
    if edge_scale > 1e-8:
        edge_strength = np.clip(edge_strength / edge_scale, 0, 1)

    # STEP 3: Backproject finite positive depths using camera intrinsics K.
    points_cam, pixels = backproject_depth(depth, K)
    # STEP 4: Transform camera points into the robot base frame.
    points = transform_points(T_base_cam, points_cam)
    finite = np.isfinite(points).all(axis=1)
    points, pixels = points[finite], pixels[finite]
    if len(points) < 60:
        return None
    height_tol = max(0.004, 0.05 * bin_info.wall_height)
    size_tol = max(0.012, 0.05 * float(outer.max()))
    rim_tol = max(0.004, 0.4 * bin_info.wall_thickness)
    # STEP 5: Search both permitted sides, allowing for the bin's full footprint.
    radius = float(np.linalg.norm(outer) / 2) + size_tol
    x_low, x_high = bin_info.search_x
    y_low, y_high = bin_info.search_y_abs
    in_search = ((points[:, 0] >= x_low - radius)
                 & (points[:, 0] <= x_high + radius)
                 & (np.abs(points[:, 1]) >= max(0, y_low - radius))
                 & (np.abs(points[:, 1]) <= y_high + radius))
    # STEP 6: Extract elevated rim points at the known wall-top height.
    on_rim = in_search & (np.abs(points[:, 2] - bin_info.wall_height) <= height_tol)
    rim_mask = np.zeros(depth.shape, dtype=bool)
    rim_mask[pixels[on_rim, 1], pixels[on_rim, 0]] = True
    # Join small pixel gaps in the thin rim only for candidate grouping. Fit
    # original measured points, never points fabricated by image morphology.
    grouped = ndimage.binary_dilation(rim_mask, iterations=1)
    labels, count = ndimage.label(grouped, structure=np.ones((3, 3), dtype=bool))
    point_labels = labels[pixels[:, 1], pixels[:, 0]]
    best_estimate, best_score = None, np.inf
    for label in range(1, count + 1):
        selected = on_rim & (point_labels == label)
        # STEPS 7-8: Robust ring fit returns centre and canonical yaw, using
        # inner/outer dimensions, wall height and both centre search bounds.
        fit = _fit_square_ring(points[selected], bin_info)
        if fit is None:
            continue
        centre, yaw = fit.centre, fit.yaw
        c, s = np.cos(yaw), np.sin(yaw)
        # STEP 9: Ring dimensions and wall support have passed; require a floor.
        delta = points[:, :2] - centre
        all_local = delta @ np.array([[c, -s], [s, c]])
        interior = np.all(np.abs(all_local) < fit.inner / 2 - rim_tol, axis=1)
        floor = interior & (np.abs(points[:, 2] - bin_info.floor_thickness) <= height_tol)
        if floor.sum() < 40 or floor.sum() < 0.50 * interior.sum():
            continue
        rim_pixels = pixels[selected][fit.inliers]
        edge_support = float(edge_strength[rim_pixels[:, 1], rim_pixels[:, 0]].mean())
        # At most 0.1 mm equivalent preference: RGB cannot rescue invalid geometry.
        score = fit.error + 0.0001 * (1.0 - edge_support)
        if score < best_score:
            best_score = score
            # STEP 10: Keep the best geometrically supported bin estimate.
            best_estimate = BinEstimate(
                position=centre, yaw=yaw,
                n_points=int(fit.inliers.sum() + floor.sum()),
            )
    # Returns None when no candidate passes the geometric checks.
    return best_estimate
