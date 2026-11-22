"""Model-agnostic push planner shared by all learned / analytical dynamics models.

1. Geometric layer: an SE(2) lattice (x, y, theta) over the workspace; a state is free if the
   object footprint (plus margin) does not intersect a wall. Backward Dijkstra from the goal set
   gives a cost-to-go field (a navigation function) that already "knows" the object must rotate
   to pass a doorway.
2. Dynamics layer: the model predicts the body-frame outcome (mean, std) of a library of candidate
   pushes. Because outcomes are body-frame (pose invariant in free space), the same predictions
   compose into multi-step sequences with no extra model calls.
3. Decision: minimise  ctg(predicted pose) + lambda_u * uncertainty  - beta * information
   with a depth-2 lookahead. beta > 0 gives active (dual-control) exploration.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .sim import compose, wrap, PUSHER_R, GAP

BIG = 10.0


def boundary_points(shape, spacing=0.008):
    ring = np.asarray(shape.polygon.exterior.coords)
    pts = []
    for a, b in zip(ring[:-1], ring[1:]):
        n = max(int(np.ceil(np.linalg.norm(b - a) / spacing)), 1)
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        pts.append(a + (b - a) * t)
    return np.concatenate(pts)


def in_walls(pts, walls, margin=0.0):
    """pts (..., 2) -> bool (...) inside any axis-aligned wall box (expanded by margin)."""
    out = np.zeros(pts.shape[:-1], bool)
    for cx, cy, hx, hy in walls:
        out |= (np.abs(pts[..., 0] - cx) < hx + margin) & (np.abs(pts[..., 1] - cy) < hy + margin)
    return out


class CostField:
    def __init__(self, shape, walls, bounds, goal, pos_tol=0.02, ang_tol=None, res=0.02, n_theta=24,
                 margin=0.012, w_theta=0.06):
        """goal = (x, y, theta); ang_tol=None -> orientation free at the goal."""
        self.shape, self.walls, self.bounds = shape, walls, bounds
        self.goal, self.pos_tol, self.ang_tol = np.asarray(goal, float), pos_tol, ang_tol
        self.res, self.nt, self.w_theta = res, n_theta, w_theta
        x0, x1, y0, y1 = bounds
        self.xs = np.arange(x0, x1 + 1e-9, res)
        self.ys = np.arange(y0, y1 + 1e-9, res)
        self.ths = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
        nx, ny, nt = len(self.xs), len(self.ys), n_theta
        bp = boundary_points(shape)
        X, Y = np.meshgrid(self.xs, self.ys, indexing="ij")
        free = np.zeros((nx, ny, nt), bool)
        for k, th in enumerate(self.ths):
            c, s = np.cos(th), np.sin(th)
            rb = bp @ np.array([[c, s], [-s, c]])  # rotate boundary points
            pts = np.stack([X[..., None] + rb[:, 0], Y[..., None] + rb[:, 1]], -1)
            hit = in_walls(pts, walls, margin).any(-1)
            # stay inside the workspace bounds too
            outb = (pts[..., 0] < x0 - 0.05) | (pts[..., 0] > x1 + 0.05) | (pts[..., 1] < y0 - 0.05) | (pts[..., 1] > y1 + 0.05)
            free[..., k] = ~(hit | outb.any(-1))
        self.free = free
        self._dijkstra()

    def _idx(self, i, j, k):
        return (i * len(self.ys) + j) * self.nt + k

    def _dijkstra(self):
        nx, ny, nt = self.free.shape
        I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nt), indexing="ij")
        src_all, dst_all, w_all = [], [], []
        moves = [(1, 0, 0), (0, 1, 0), (1, 1, 0), (1, -1, 0), (0, 0, 1)]
        dth = 2 * np.pi / nt
        for di, dj, dk in moves:
            i2, j2, k2 = I + di, J + dj, (K + dk) % nt
            valid = (i2 >= 0) & (i2 < nx) & (j2 >= 0) & (j2 < ny)
            valid &= self.free
            valid[valid] &= self.free[i2[valid], j2[valid], k2[valid]]
            w = self.res * np.hypot(di, dj) + self.w_theta * dth * abs(dk)
            a = self._idx(I[valid], J[valid], K[valid])
            b = self._idx(i2[valid], j2[valid], k2[valid])
            src_all += [a, b]; dst_all += [b, a]; w_all += [np.full(len(a), w)] * 2
        n = nx * ny * nt
        G = coo_matrix((np.concatenate(w_all), (np.concatenate(src_all), np.concatenate(dst_all))), shape=(n, n)).tocsr()
        X, Y = np.meshgrid(self.xs, self.ys, indexing="ij")
        dpos = np.hypot(X - self.goal[0], Y - self.goal[1])
        goal_mask = (dpos <= max(self.pos_tol, self.res * 0.75))[..., None] & self.free
        if self.ang_tol is not None:
            dth_goal = np.abs(wrap(self.ths - self.goal[2]))
            goal_mask &= (dth_goal <= max(self.ang_tol, np.pi / self.nt))[None, None, :]
        goals = np.flatnonzero(goal_mask.ravel())
        if len(goals) == 0:
            raise RuntimeError("goal set is empty / in collision")
        d = dijkstra(G, directed=False, indices=goals, min_only=True)
        ctg = d.reshape(nx, ny, nt)
        # Recovery gradient: cells in collision / unreachable get the cost of nearby free cells plus
        # a penalty growing with depth, so the planner can push an object back out of contact.
        from scipy.ndimage import minimum_filter
        filled = ctg.copy()
        pen = 0.0
        for it in range(12):
            bad = ~np.isfinite(filled)
            if not bad.any():
                break
            pen += 0.15
            nbr = minimum_filter(np.where(np.isfinite(filled), filled, 1e9), size=(3, 3, 3), mode="nearest")
            upd = bad & (nbr < 1e8)
            filled[upd] = nbr[upd] + self.res * 3 + (0.3 if it == 0 else 0.0)
        self.ctg = filled
        self.ctg_raw = ctg

    def __call__(self, poses):
        """Cost-to-go for poses (..., 3); BIG for collision / out of bounds."""
        poses = np.asarray(poses)
        x, y, th = poses[..., 0], poses[..., 1], poses[..., 2]
        fi = (x - self.xs[0]) / self.res
        fj = (y - self.ys[0]) / self.res
        i = np.clip(np.round(fi).astype(int), 0, len(self.xs) - 1)
        j = np.clip(np.round(fj).astype(int), 0, len(self.ys) - 1)
        k = np.round((wrap(th) + np.pi) / (2 * np.pi) * self.nt).astype(int) % self.nt
        out_b = (fi < -0.5) | (fi > len(self.xs) - 0.5) | (fj < -0.5) | (fj > len(self.ys) - 0.5)
        c = self.ctg[i, j, k]
        # continuous refinement inside the cell + exact goal distance near the goal
        dg = np.hypot(x - self.goal[0], y - self.goal[1])
        if self.ang_tol is not None:
            dg = dg + self.w_theta * np.maximum(np.abs(wrap(th - self.goal[2])) - 0.0, 0)
        cell = np.hypot(x - self.xs[i], y - self.ys[j])
        c = np.where(np.isfinite(c), np.maximum(c, 0) + 0.5 * cell, BIG)
        c = np.where(c < 3 * self.res, np.minimum(c, dg), c)
        return np.where(out_b, BIG, c)


class FreeSpaceCost:
    """Analytic cost for tasks without walls: position error + w_theta * angle error."""

    def __init__(self, goal, w_theta=0.06, use_theta=True):
        self.goal, self.w, self.use_theta = np.asarray(goal, float), w_theta, use_theta

    def __call__(self, poses):
        poses = np.asarray(poses)
        c = np.hypot(poses[..., 0] - self.goal[0], poses[..., 1] - self.goal[1])
        if self.use_theta:
            c = c + self.w * np.abs(wrap(poses[..., 2] - self.goal[2]))
        return c


def pusher_starts_world(Q, pose, offset=PUSHER_R + GAP):
    start_b = Q["p"] - Q["u"] * offset
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    return np.stack([x + c * start_b[:, 0] - s * start_b[:, 1], y + s * start_b[:, 0] + c * start_b[:, 1]], 1)


class Planner:
    def __init__(self, cost, walls=(), lam_u=0.5, w_theta=0.06, depth=2, beam=12, step_cost=0.004,
                 beta0=0.0, n_explore=5, start_offset=PUSHER_R + GAP, start_margin=PUSHER_R + 0.003):
        self.cost, self.walls = cost, list(walls)
        self.lam_u, self.w_theta, self.depth, self.beam = lam_u, w_theta, depth, beam
        self.step_cost = step_cost
        self.beta0, self.n_explore = beta0, n_explore
        self.start_offset, self.start_margin = start_offset, start_margin

    def choose(self, pose, Q, mean, std, n_data: int):
        pose = np.asarray(pose)
        unc = np.linalg.norm(std[:, :2], axis=1) + self.w_theta * std[:, 2]
        p1 = compose(pose, mean)
        c1 = self.cost(p1)
        # pusher must be placeable (not inside a wall)
        starts = pusher_starts_world(Q, pose, self.start_offset)
        blocked = in_walls(starts, self.walls, self.start_margin) if self.walls else np.zeros(len(mean), bool)
        score1 = c1 + self.lam_u * unc
        best_future = c1.copy()
        if self.depth >= 2:
            top = np.argsort(score1 + BIG * blocked)[: self.beam]
            p2 = compose(p1[top][:, None, :], mean[None, :, :])  # (beam, M, 3)
            c2 = self.cost(p2) + self.lam_u * unc[None, :] + self.step_cost
            best_future[top] = np.minimum(c1[top], c2.min(1))
        beta = self.beta0 * max(0.0, 1.0 - n_data / max(self.n_explore, 1))
        score = 0.5 * score1 + 0.5 * (best_future + self.lam_u * unc) - beta * unc + BIG * blocked
        i = int(np.argmin(score))
        return i, dict(score=score, pred=p1, c1=c1, unc=unc, blocked=blocked)
