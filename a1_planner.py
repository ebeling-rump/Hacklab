"""Waypoint planner: lidar occupancy map + A* path to the goal.

Every decision step (0.2 s) the planner
  1. writes the lidar scan into a 2D grid around the arena (10 cm cells):
     cells where a ray ends are marked occupied, cells a ray passed through are
     cleared again (so a moving box does not leave a permanent ghost wall);
  2. inflates obstacles by the robot's half-width: cells closer than R_HARD to
     an obstacle are (nearly) forbidden, cells closer than R_SOFT cost extra, so
     paths keep some clearance;
  3. runs A* from the robot to the goal on that cost map (unseen cells count
     as free: the plan is optimistic and gets corrected as the robot sees more);
  4. returns a waypoint LOOKAHEAD metres along the path.

The controllers then steer towards the waypoint instead of the far goal, which
turns detours (around a wall, through a slalom) into a sequence of nearby
targets. The planner uses the simulator's exact robot pose (perfect
localisation) - a real robot would need odometry/SLAM here.
"""

import heapq

import numpy as np
from scipy.ndimage import distance_transform_edt

from a1_lidar import GOAL, MAX_RANGE, ray_angles, yaw

RES = 0.1                      # m per cell
X_RANGE = (-2.0, 10.5)
Y_RANGE = (-4.5, 4.5)
R_HARD = 0.3                   # m: robot half-width incl. legs (as HALF_WIDTH in a1_lidar)
R_SOFT = 0.7                   # m: below this, cells get an extra cost
SOFT_WEIGHT = 3.0
BLOCKED_COST = 60.0            # inside R_HARD: allowed (so a path always exists) but very costly
LOOKAHEAD = 0.8                # m along the path
REPLAN_EVERY = 1               # decisions between re-plans


class Planner:
    def __init__(self, goal=GOAL):
        self.goal = np.asarray(goal, float)
        self.nx = int(round((X_RANGE[1] - X_RANGE[0]) / RES))
        self.ny = int(round((Y_RANGE[1] - Y_RANGE[0]) / RES))
        self.occupied = np.zeros((self.nx, self.ny), bool)
        self.path = np.empty((0, 2))
        self._angles = ray_angles()
        self._steps = 0

    # grid <-> world
    def cell(self, p):
        i = int(np.clip((p[0] - X_RANGE[0]) / RES, 0, self.nx - 1))
        j = int(np.clip((p[1] - Y_RANGE[0]) / RES, 0, self.ny - 1))
        return i, j

    def centre(self, ij):
        return np.array([X_RANGE[0] + (ij[0] + 0.5) * RES, Y_RANGE[0] + (ij[1] + 0.5) * RES])

    def _cells(self, pts):
        ij = np.floor((pts - [X_RANGE[0], Y_RANGE[0]]) / RES).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.nx) & (ij[:, 1] >= 0) & (ij[:, 1] < self.ny)
        return ij[ok]

    def update_map(self, data, ranges):
        pos, th = data.qpos[:2].copy(), yaw(data)
        dirs = np.stack([np.cos(self._angles + th), np.sin(self._angles + th)], 1)
        # Free space: points along each ray up to 1.5 cells before its end.
        steps = np.arange(RES / 2, MAX_RANGE, RES / 2)
        along = steps[None, :] < (ranges[:, None] - 1.5 * RES)
        free = (pos + dirs[:, None, :] * steps[None, :, None])[along]
        f = self._cells(free)
        self.occupied[f[:, 0], f[:, 1]] = False
        # Obstacles: ray end points that hit something.
        hit = ranges < MAX_RANGE - 1e-3
        h = self._cells(pos + dirs[hit] * ranges[hit, None])
        self.occupied[h[:, 0], h[:, 1]] = True

    def cost_map(self):
        dist = distance_transform_edt(~self.occupied) * RES
        cost = 1.0 + SOFT_WEIGHT * np.clip((R_SOFT - dist) / (R_SOFT - R_HARD), 0, 1)
        cost[dist < R_HARD] = BLOCKED_COST
        return cost

    def plan(self, start):
        cost = self.cost_map()
        s, g = self.cell(start), self.cell(self.goal)
        nbrs = [(di, dj, np.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
        best = {s: 0.0}
        parent = {}
        h = lambda c: np.hypot(c[0] - g[0], c[1] - g[1])
        heap = [(h(s), 0.0, s)]
        while heap:
            _, d, c = heapq.heappop(heap)
            if c == g:
                break
            if d > best.get(c, np.inf):
                continue
            for di, dj, step in nbrs:
                n = (c[0] + di, c[1] + dj)
                if not (0 <= n[0] < self.nx and 0 <= n[1] < self.ny):
                    continue
                nd = d + step * 0.5 * (cost[c] + cost[n])
                if nd < best.get(n, np.inf):
                    best[n], parent[n] = nd, c
                    heapq.heappush(heap, (nd + h(n), nd, n))
        cells = [g]
        while cells[-1] in parent:
            cells.append(parent[cells[-1]])
        return np.array([self.centre(c) for c in reversed(cells)])

    def waypoint(self, data, ranges):
        """Update the map, re-plan, and return the point to steer towards."""
        self.update_map(data, ranges)
        pos = data.qpos[:2]
        if self._steps % REPLAN_EVERY == 0 or len(self.path) == 0:
            self.path = self.plan(pos)
        self._steps += 1
        far = np.linalg.norm(self.path - pos, axis=1) >= LOOKAHEAD
        return self.path[np.argmax(far)] if far.any() else self.goal
