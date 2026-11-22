"""Whole-body push primitives for the A1: walk around the crate, line up, push, back off.

Action (same parameterisation as the planar lab): contact perimeter parameter s, push angle phi
relative to the inward face normal, push distance d (m of walking after nominal contact).
"""
from __future__ import annotations

import heapq

import mujoco
import numpy as np
from shapely.geometry import Point, Polygon

from .env import crate_pose, crate_speed
from .robot import A1, NOSE, goto_command, wrap

ROBOT_R = 0.20        # clearance radius of the robot centre from walls
CRATE_CLEAR = 0.30    # clearance of the robot centre from the crate while walking around it
STANDOFF = 0.18       # start the push this far behind first contact


class Navigator:
    """Grid A* for the robot centre, avoiding crate (inflated) and walls."""

    def __init__(self, walls, bounds, res=0.05):
        self.walls, self.bounds, self.res = walls, bounds, res
        x0, x1, y0, y1 = bounds
        self.xs = np.arange(x0, x1 + 1e-9, res)
        self.ys = np.arange(y0, y1 + 1e-9, res)
        X, Y = np.meshgrid(self.xs, self.ys, indexing="ij")
        occ = np.zeros(X.shape, bool)
        for cx, cy, hx, hy in walls:
            occ |= (np.abs(X - cx) < hx + ROBOT_R) & (np.abs(Y - cy) < hy + ROBOT_R)
        self.static = occ
        self.X, self.Y = X, Y

    def plan(self, start, goal, crate_poly):
        from shapely import contains_xy
        infl = crate_poly.buffer(CRATE_CLEAR)
        occ = self.static | contains_xy(infl, self.X, self.Y)
        idx = lambda p: (int(np.clip(round((p[0] - self.xs[0]) / self.res), 0, len(self.xs) - 1)),
                         int(np.clip(round((p[1] - self.ys[0]) / self.res), 0, len(self.ys) - 1)))
        s, g = idx(start), idx(goal)
        occ = occ.copy()
        occ[s] = False
        occ[g] = False
        # Free a small disc around the start (robot may currently be touching the crate)
        for di in range(-6, 7):
            for dj in range(-6, 7):
                i, j = s[0] + di, s[1] + dj
                if 0 <= i < occ.shape[0] and 0 <= j < occ.shape[1] and di * di + dj * dj <= 36:
                    occ[i, j] = occ[i, j] and contains_xy(crate_poly.buffer(0.12), self.X[i, j], self.Y[i, j])
        nb = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        dist = {s: 0.0}
        prev = {}
        pq = [(0.0, s)]
        while pq:
            f, u = heapq.heappop(pq)
            if u == g:
                break
            for di, dj in nb:
                v = (u[0] + di, u[1] + dj)
                if not (0 <= v[0] < occ.shape[0] and 0 <= v[1] < occ.shape[1]) or occ[v]:
                    continue
                nd = dist[u] + np.hypot(di, dj)
                if nd < dist.get(v, 1e9):
                    dist[v] = nd
                    prev[v] = u
                    h = np.hypot(v[0] - g[0], v[1] - g[1])
                    heapq.heappush(pq, (nd + h, v))
        if g not in prev and g != s:
            return None
        path = [g]
        while path[-1] != s:
            path.append(prev[path[-1]])
        pts = [np.array([self.xs[i], self.ys[j]]) for i, j in path[::-1]]
        # shortcut: keep every 4th point
        step = max(1, int(round(0.2 / self.res)))
        return pts[::step] + [np.array(goal[:2])]


def crate_world_poly(shape, pose):
    return shape.world_polygon(pose)


def push_geometry(shape, crate_pose_, action):
    """World pre-push trunk pose and push direction."""
    s, phi, d = action[:3]
    p_b, n_b, _ = shape.perimeter_point(s)
    p_b, n_b = p_b[0], n_b[0]
    inward = -n_b
    c, sn = np.cos(phi), np.sin(phi)
    u_b = np.array([c * inward[0] - sn * inward[1], sn * inward[0] + c * inward[1]])
    x, y, th = crate_pose_
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    p_w = np.array([x, y]) + R @ p_b
    u_w = R @ u_b
    pre = p_w - u_w * (NOSE + STANDOFF)
    return pre, u_w, p_w


def run_until(model, data, robot, fn, t_max, recorder=None):
    """Step physics with control from fn() -> (fwd, turn, done) until done or timeout."""
    t_end = data.time + t_max
    while data.time < t_end:
        fwd, turn, done = fn()
        if done:
            break
        robot.command(fwd, turn)
        mujoco.mj_step(model, data)
        if recorder is not None:
            recorder(model, data)
        if not robot.upright():
            return False
    return True


def execute_push(model, data, robot: A1, shape, nav: Navigator, action, recorder=None, t_nav=25.0):
    """Navigate to the pre-push pose, push along u for d (+standoff), back off. Returns info."""
    cp0 = crate_pose(model, data)
    pre, u, p_w = push_geometry(shape, cp0, action)
    yaw_push = float(np.arctan2(u[1], u[0]))
    info = dict(ok=True, pre=pre, u=u, contact=p_w, phase_times={})
    # 1) navigate
    path = nav.plan(robot.pose, pre, crate_world_poly(shape, cp0))
    if path is None:
        info["ok"] = False
        info["reason"] = "no path"
        return info
    wp = [0]

    def nav_fn():
        pose = robot.pose
        while wp[0] < len(path) - 1 and np.hypot(*(path[wp[0]] - pose[:2])) < 0.12:
            wp[0] += 1
        last = wp[0] == len(path) - 1
        fwd, turn, done = goto_command(pose, path[wp[0]], pos_tol=0.05 if last else 0.12,
                                       final_yaw=yaw_push if last else None)
        return fwd, turn, done and last
    t0 = data.time
    if not run_until(model, data, robot, nav_fn, t_nav, recorder):
        info.update(ok=False, reason="fell")
        return info
    info["phase_times"]["nav"] = data.time - t0
    # 2) push along the line through the contact point
    start = robot.pose[:2].copy()
    L = STANDOFF + action[2]

    def push_fn():
        pose = robot.pose
        prog = (pose[:2] - start) @ u
        if prog >= L:
            return 0.0, 0.0, True
        lat = (pose[:2] - p_w) @ np.array([-u[1], u[0]])
        head = yaw_push - np.clip(1.5 * lat, -0.4, 0.4)
        return 0.55, float(np.clip(1.5 * wrap(head - pose[2]), -0.4, 0.4)), False
    t0 = data.time
    if not run_until(model, data, robot, push_fn, 4.0 + 3 * L, recorder):
        info.update(ok=False, reason="fell")
        return info
    info["phase_times"]["push"] = data.time - t0
    # 3) back off
    back0 = robot.pose[:2].copy()

    def back_fn():
        pose = robot.pose
        if (back0 - pose[:2]) @ u > 0.22:
            return 0.0, 0.0, True
        return -0.6, float(np.clip(1.2 * wrap(yaw_push - pose[2]), -0.4, 0.4)), False
    run_until(model, data, robot, back_fn, 3.0, recorder)
    # settle
    t_end = data.time + 1.5
    while data.time < t_end:
        robot.stand()
        mujoco.mj_step(model, data)
        if recorder is not None:
            recorder(model, data)
        if crate_speed(model, data) < 1e-3 and data.time > t_end - 1.0:
            break
    info["crate_after"] = crate_pose(model, data)
    info["crate_before"] = cp0
    return info
