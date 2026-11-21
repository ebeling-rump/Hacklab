"""T3 - Coordinate transformations and forward kinematics.

Conventions: T_a_b is the 4x4 homogeneous pose of frame b expressed in frame a, so that p_a = T_a_b @ p_b.
"""
from __future__ import annotations

import numpy as np

D_FLANGE = 0.107          # flange offset along z7 [m]
HAND_YAW = -np.pi / 4     # hand frame is the flange rotated by -45 deg about z

# Panda's fixed modified-DH geometry (a [m], d [m], alpha [rad]).
# Joint angles are the measured encoder values; no scene information is needed.
_PANDA_MDH = (
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
)


def _intrinsics(K: np.ndarray) -> np.ndarray:
    """Validate a finite, invertible 3x3 intrinsic matrix."""
    K = np.asarray(K, dtype=float)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("K must be a finite (3, 3) matrix")
    if abs(np.linalg.det(K)) < 1e-12:
        raise ValueError("K must be invertible")
    return K


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a homogeneous transform [R | t; 0 0 0 | 1]."""
    R, t = np.asarray(R, dtype=float), np.asarray(t, dtype=float)
    if R.shape != (3, 3) or t.shape != (3,):
        raise ValueError("R must be (3, 3) and t must be (3,)")
    if not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError("R and t must be finite")
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    """Invert a rigid transform using R.T and -R.T @ t, not a matrix inverse."""
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("T must be a finite (4, 4) rigid transform")
    rotation = T[:3, :3].T
    return make_T(rotation, -rotation @ T[:3, 3])


def rot_z(angle: float) -> np.ndarray:
    """Right-handed 3x3 rotation about z by angle in radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def dh_transform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    """
    Modified DH: Tx(a) @ Rx(alpha) @ Rz(theta) @ Tz(d).
    Translations are in metres and rotation angles are in radians.
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st, 0.0, a],
        [ca * st, ca * ct, -sa, -d * sa],
        [sa * st, sa * ct, ca, d * ca],
        [0.0, 0.0, 0.0, 1.0],
    ])


def fk_hand(q: np.ndarray) -> np.ndarray:
    """
    T_base_hand from seven joint encoder angles q in radians.
    Chain the seven modified-DH links, then the fixed flange and hand mounting.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError("q must contain seven finite joint angles")
    T = np.eye(4)
    for (a, d, alpha), theta in zip(_PANDA_MDH, q):
        T = T @ dh_transform(a, d, alpha, theta)
    return T @ make_T(rot_z(HAND_YAW), np.array([0.0, 0.0, D_FLANGE]))


def camera_pose(q: np.ndarray, T_hand_cam: np.ndarray) -> np.ndarray:
    """T_base_cam = T_base_hand(q) @ T_hand_cam (computer-vision axes)."""
    T_hand_cam = np.asarray(T_hand_cam, dtype=float)
    if T_hand_cam.shape != (4, 4) or not np.isfinite(T_hand_cam).all():
        raise ValueError("T_hand_cam must be a finite (4, 4) transform")
    return fk_hand(q) @ T_hand_cam


def transform_points(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Apply the rigid transform T to an (N, 3) array of points."""
    T, P = np.asarray(T, dtype=float), np.asarray(P, dtype=float)
    if T.shape != (4, 4) or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("T must be (4, 4) and P must be (N, 3)")
    return P @ T[:3, :3].T + T[:3, 3]


def backproject(u: float, v: float, z: float, K: np.ndarray) -> np.ndarray:
    """Pixel (u, v) plus positive optical-axis depth z -> camera-frame point."""
    K = _intrinsics(K)
    if not np.isfinite([u, v, z]).all() or z <= 0:
        raise ValueError("pixel coordinates must be finite and depth must be positive")
    ray = np.linalg.solve(K, np.array([u, v, 1.0]))
    if abs(ray[2]) < 1e-12:
        raise ValueError("pixel ray must have a nonzero optical-axis component")
    return ray * (z / ray[2])


def backproject_depth(depth: np.ndarray, K: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Back-project finite positive depths selected by the optional mask.
    Returns (points_cam (N,3), pixels_uv (N,2)).
    """
    depth = np.asarray(depth)
    K = _intrinsics(K)
    if depth.ndim != 2:
        raise ValueError("depth must have shape (H, W)")
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != depth.shape:
            raise ValueError("mask must have the same shape as depth")
        valid &= mask
    v, u = np.nonzero(valid)
    pixels = np.column_stack((u, v))
    # Solve K * ray = [u, v, 1]; scale rays by optical-axis depth.
    rays = np.linalg.solve(K,
                           np.column_stack((u, v, np.ones(u.size))).T).T
    if np.any(np.abs(rays[:, 2]) < 1e-12):
        raise ValueError("pixel rays must have nonzero optical-axis components")
    points = rays * (depth[v, u] / rays[:, 2])[:, None]
    return points, pixels


def project(p_cam: np.ndarray, K: np.ndarray) -> tuple[float, float]:
    """Project a finite camera-frame point in front of the camera to (u, v)."""
    K = _intrinsics(K)
    p_cam = np.asarray(p_cam, dtype=float)
    if p_cam.shape != (3,) or not np.isfinite(p_cam).all() or p_cam[2] <= 0:
        raise ValueError("p_cam must be a finite (3,) point with positive z")
    pixel = K @ p_cam
    if abs(pixel[2]) < 1e-12:
        raise ValueError("projected homogeneous pixel must have nonzero scale")
    return float(pixel[0] / pixel[2]), float(pixel[1] / pixel[2])
