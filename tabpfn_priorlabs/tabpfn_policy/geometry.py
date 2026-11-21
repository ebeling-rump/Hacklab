"""Small geometry helpers shared by the expert, runner and feature code."""
from __future__ import annotations

import math

import numpy as np

HALF_PI = math.pi / 2


def wrap90(a: float) -> float:
    """Reduce an angle to (-pi/4, pi/4]: the 90-degree symmetry of the cube, the square bin and the gripper."""
    y = (a + math.pi / 4) % HALF_PI - math.pi / 4
    return math.pi / 4 if math.isclose(y, -math.pi / 4) else y


def tcp_yaw(R: np.ndarray) -> float:
    """Grasp yaw psi of a top-down hand: R = Rz(psi) @ R_DOWN, finger axis (hand y) = (cos psi, sin psi, 0)."""
    return math.atan2(R[1, 1], R[0, 1])


def quat_yaw(q_wxyz) -> float:
    w, x, y, z = q_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n <= max_norm or n == 0 else v * (max_norm / n)


def to_bin_frame(p_xy, bin_xy, bin_yaw: float) -> np.ndarray:
    c, s = math.cos(bin_yaw), math.sin(bin_yaw)
    d = np.asarray(p_xy, float)[:2] - np.asarray(bin_xy, float)[:2]
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
