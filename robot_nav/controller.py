"""Robot control: observation tracking, obstacle predictors, local planner, safety filter.

Layers, from slowest to fastest:
  1. Predictor  - guesses where each obstacle will be over the next second.
                  Phase 3 plugs TabPFN in here; it may be slow or fail.
  2. Planner    - samples candidate velocities, rolls them out against the predictions,
                  picks the cheapest (progress to goal + clearance).
  3. Safety     - local and deterministic: uses only the *current* observed obstacle
                  positions and overrides the planner when an obstacle is too close.
                  It never waits on a predictor or a network call.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from obstacles import ARENA, R_OBS

R_ROBOT = 0.25
R_SUM = R_ROBOT + R_OBS   # center distance at which robot and obstacle touch


class ObstacleTracker:
    """What the robot can observe: obstacle positions sampled at the control rate.

    Velocity and acceleration are finite differences of observed positions, never the
    obstacles' internal (hidden) parameters.
    """

    def __init__(self, n_obstacles, dt, history=10):
        self.dt = dt
        self.history = deque(maxlen=history)
        self.n = n_obstacles

    def update(self, positions):
        self.history.append(np.asarray(positions, dtype=float).copy())

    @property
    def position(self):
        return self.history[-1]

    @property
    def velocity(self):
        if len(self.history) < 2:
            return np.zeros((self.n, 2))
        return (self.history[-1] - self.history[-2]) / self.dt


# ---- Predictors: (tracker, times[K]) -> predicted positions (n_obstacles, K, 2) or None ----

# Predictors may also receive robot_pos= and t= (sim time) as keyword arguments.

def predict_none(tracker, times, **_):
    """Ignore obstacles entirely (goal-seeking only). Used to show the task is not trivial."""
    return None


def predict_static(tracker, times, **_):
    """Obstacles stay where they are now (purely reactive avoidance)."""
    return np.repeat(tracker.position[:, None, :], len(times), axis=1)


def predict_constant_velocity(tracker, times, **_):
    """Extrapolate the last observed velocity."""
    return tracker.position[:, None, :] + tracker.velocity[:, None, :] * times[None, :, None]


PREDICTORS = {
    "none": predict_none,
    "static": predict_static,
    "constant_velocity": predict_constant_velocity,
}


# ---- Local planner ----

@dataclass
class PlannerConfig:
    v_max: float = 1.0          # m/s
    horizon: float = 1.0        # s, how far ahead candidates are rolled out
    dt: float = 0.1             # s, rollout resolution (matches the control rate)
    n_headings: int = 24
    speed_fracs: tuple = (0.15, 0.35, 0.7, 1.0)
    margin: float = 0.4         # desired clearance beyond touching (m)
    w_clear: float = 20.0       # weight of the clearance penalty
    w_room: float = 0.3         # weight of the extra-clearance bonus


class Planner:
    """Sampling-based local planner for a holonomic robot (a tiny dynamic-window approach)."""

    def __init__(self, cfg=PlannerConfig()):
        self.cfg = cfg
        self.times = np.arange(1, round(cfg.horizon / cfg.dt) + 1) * cfg.dt
        angles = np.linspace(0, 2 * np.pi, cfg.n_headings, endpoint=False)
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        self.candidates = np.vstack(
            [np.zeros((1, 2))] + [f * cfg.v_max * dirs for f in cfg.speed_fracs]
        )

    def plan(self, pos, goal, predictions, free_space=None, final=True):
        """Return the commanded velocity (2,) given predicted obstacle positions (n, K, 2).

        `goal` is the current target (a waypoint or the final goal; only the final goal is
        approached slowly). `free_space(points) -> bool` marks where the robot may be.
        """
        cfg = self.cfg
        to_goal = goal - pos
        dist = np.linalg.norm(to_goal)
        # Always include "straight to the target" (stopping on it if it is the final goal).
        speed = min(cfg.v_max, dist / cfg.horizon) if final else cfg.v_max
        direct = to_goal / max(dist, 1e-9) * speed
        V = np.vstack([self.candidates, direct])

        traj = pos + V[:, None, :] * self.times[None, :, None]          # (M, K, 2)
        cost = np.linalg.norm(traj[:, -1] - goal, axis=1)               # distance left after horizon

        # Stay inside the arena and (if given) the free space, e.g. warehouse aisles.
        out = np.abs(traj).max(axis=(1, 2)) > ARENA - R_ROBOT
        cost[out] += 100.0
        if free_space is not None:
            cost += 100.0 * (~free_space(traj)).mean(axis=1)

        if predictions is not None:
            d = np.linalg.norm(traj[:, None] - predictions[None], axis=-1)   # (M, n_obs, K)
            clearance = d.min(axis=(1, 2)) - R_SUM
            cost += cfg.w_clear * np.clip(cfg.margin - clearance, 0, None) ** 2
            # Small bonus for extra room (up to 1 m): when progress is blocked the robot
            # steps aside instead of standing in an AGV's path.
            cost -= cfg.w_room * np.minimum(clearance, 1.0)
            hit = clearance < 0                                         # predicted collision
            cost[hit] += 1000.0 - 1000.0 * clearance[hit]
        return V[np.argmin(cost)]


# ---- Safety filter (local, deterministic, no prediction) ----

def safety_filter(cmd, robot_pos, obstacle_pos, v_max=1.0, safety_dist=0.2, gain=4.0):
    """If an obstacle is within `safety_dist` of touching, cancel motion toward it and push away.

    Uses only current observed positions, so it works even if the predictor is late or wrong.
    """
    cmd = np.asarray(cmd, dtype=float).copy()
    for p in obstacle_pos:
        diff = robot_pos - p
        d = np.linalg.norm(diff)
        if d < R_SUM + safety_dist:
            n = diff / max(d, 1e-9)
            approach = cmd @ n
            if approach < 0:
                cmd -= approach * n
            cmd += gain * (R_SUM + safety_dist - d) * n
    speed = np.linalg.norm(cmd)
    return cmd if speed <= v_max else cmd * (v_max / speed)
