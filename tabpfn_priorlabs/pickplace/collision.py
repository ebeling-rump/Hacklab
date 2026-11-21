"""Collision checker for planning: robot + table + bin + obstacles (no cubes)."""
from __future__ import annotations

import mujoco
import numpy as np

from .scene import build_planning_model


class CollisionChecker:
    def __init__(self, task, margin: float = 0.015):
        """margin: obstacles and bin are inflated by this many metres (safety clearance)."""
        self.model = build_planning_model(task.bin, task.obstacles, margin)
        self.data = mujoco.MjData(self.model)

    def in_collision(self, q, gripper_width: float = 0.08) -> bool:
        """True if the arm (+ hand + fingers) penetrates anything or itself at configuration q."""
        d = self.data
        d.qpos[:7] = q
        d.qpos[7:9] = gripper_width / 2
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_collision(self.model, d)
        return d.ncon > 0

    def segment_free(self, qa, qb, resolution: float = 0.05, gripper_width: float = 0.08) -> bool:
        """Check the straight joint-space segment qa->qb at max `resolution` rad per sample."""
        qa, qb = np.asarray(qa, float), np.asarray(qb, float)
        n = max(2, int(np.ceil(np.max(np.abs(qb - qa)) / resolution)) + 1)
        return not any(self.in_collision(qa + (qb - qa) * t, gripper_width) for t in np.linspace(0, 1, n))
