"""T5 - Trajectory generation and joint-space control."""
from __future__ import annotations

import numpy as np

from pickplace.interfaces import A_MAX, V_MAX


class JointTrajectory:
    """Trapezoidal profile along a piecewise-linear joint-space path (rest at both ends).

    The path is parametrised by a scalar s measured in "seconds at full speed": segment i has length
    w_i = max_j |dq_j| / v_j, so ds/dt <= 1 keeps every joint within v_max, and the scalar
    acceleration limit is set by the tightest joint over all segments.
    """

    def __init__(self, waypoints, v_max=V_MAX, a_max=A_MAX):
        wp = [np.asarray(w, float) for w in waypoints]
        if not wp:
            raise ValueError("need at least one waypoint")
        v_max = np.broadcast_to(np.asarray(v_max, float), wp[0].shape)
        a_max = np.broadcast_to(np.asarray(a_max, float), wp[0].shape)
        pts = [wp[0]]
        for w in wp[1:]:
            if np.max(np.abs(w - pts[-1])) > 1e-9:
                pts.append(w)
        self.waypoints = np.array(pts)
        d = np.diff(self.waypoints, axis=0)
        self._w = np.max(np.abs(d) / v_max, axis=1) if len(d) else np.zeros(0)
        self._s_knots = np.r_[0.0, np.cumsum(self._w)]
        self._L = float(self._s_knots[-1])
        a_s = np.inf
        for dq, w in zip(d, self._w):
            u = np.abs(dq) / w
            m = u > 1e-12
            a_s = min(a_s, float(np.min(a_max[m] / u[m])))
        self._a = a_s if np.isfinite(a_s) else 1.0
        # trapezoid on s with peak speed 1 (or triangular if the path is short)
        if self._L >= 1.0 / self._a:
            self._vpk = 1.0
            self._ta = 1.0 / self._a
            self._tc = self._L - 1.0 / self._a
        else:
            self._vpk = float(np.sqrt(self._L * self._a))
            self._ta = self._vpk / self._a
            self._tc = 0.0
        self.duration = 2 * self._ta + self._tc if self._L > 0 else 0.0

    def _s(self, t: float) -> tuple[float, float]:
        a, ta, tc, vp = self._a, self._ta, self._tc, self._vpk
        if t <= 0:
            return 0.0, 0.0
        if t >= self.duration:
            return self._L, 0.0
        if t < ta:
            return 0.5 * a * t * t, a * t
        if t < ta + tc:
            return 0.5 * a * ta * ta + vp * (t - ta), vp
        td = self.duration - t
        return self._L - 0.5 * a * td * td, a * td

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (q, dq) at time t seconds (clamped to [0, duration])."""
        if len(self.waypoints) == 1 or self._L <= 0:
            return self.waypoints[0].copy(), np.zeros_like(self.waypoints[0])
        t = min(max(float(t), 0.0), self.duration)
        if t >= self.duration:
            return self.waypoints[-1].copy(), np.zeros_like(self.waypoints[-1])
        s, sd = self._s(t)
        i = int(np.clip(np.searchsorted(self._s_knots, s, side="right") - 1, 0, len(self._w) - 1))
        frac = (s - self._s_knots[i]) / self._w[i]
        seg = self.waypoints[i + 1] - self.waypoints[i]
        return self.waypoints[i] + np.clip(frac, 0.0, 1.0) * seg, seg / self._w[i] * sd


def execute_trajectory(robot, traj: JointTrajectory, monitor=None) -> bool:
    """Play the trajectory in simulated real time; abort (False) as soon as monitor() returns False."""
    dt = robot.control_dt
    n = int(np.ceil(traj.duration / dt))
    for k in range(1, n + 1):
        robot.set_joint_targets(traj.sample(k * dt)[0])
        robot.step()
        if monitor is not None and not monitor():
            return False
    robot.set_joint_targets(traj.sample(traj.duration)[0])
    return True


def settle(robot, q_goal, tol: float = 0.003, timeout: float = 2.0, dq_tol: float = 0.02) -> bool:
    """Hold q_goal until |q - q_goal| < tol and |dq| < dq_tol, or timeout (simulated seconds)."""
    q_goal = np.asarray(q_goal, float)
    robot.set_joint_targets(q_goal)
    t_end = robot.time + timeout
    while True:
        q, dq, _ = robot.get_joint_state()
        if np.max(np.abs(q - q_goal)) < tol and np.max(np.abs(dq)) < dq_tol:
            return True
        if robot.time >= t_end:
            return False
        robot.step()
