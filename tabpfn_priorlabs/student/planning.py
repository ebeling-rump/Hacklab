"""T4 - Task planning, inverse kinematics, cartesian and joint-space motion planning."""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from pickplace.interfaces import Q_MAX, Q_MIN, TaskInfo

from .localization import CubeEstimate, wrap_cube_yaw
from .transforms import fk_hand

BIN_GRID = [(sx * 0.06, sy * 0.06) for sy in (-1, 0, 1) for sx in (-1, 0, 1)]  # candidate cells, bin frame [m]


@dataclass
class PickPlaceGoal:
    cube: CubeEstimate
    place_xy: np.ndarray   # (2,) where the cube centre should end up, BASE frame [m]
    place_yaw: float       # hand/cube yaw to use while placing, in (-pi/4, pi/4]


def plan_task(estimates: list[CubeEstimate], task: TaskInfo, occupied_bin_xy=()) -> list[PickPlaceGoal]:
    """
    Select the requested cubes that still lie outside the bin, give each a free destination cell
    (its colour slot if one is configured, else a free cell of BIN_GRID), and order the goals.
    Needs a known bin pose (`task.bin.pose_known`).
    """
    if not task.bin.pose_known:
        return []
    min_dist = task.cube_size + 0.015            # centre-to-centre clearance between placed cubes
    place_yaw = wrap_cube_yaw(task.bin.yaw)       # cube faces parallel to the bin walls
    wanted = set(task.pick_colors)

    # Cube positions that already block part of the bin floor: earlier placements and any cube seen inside.
    occupied = [np.asarray(p, float)[:2] for p in occupied_bin_xy]
    candidates = []
    for e in estimates:
        if e.color not in wanted:
            continue
        if task.in_bin_xy(e.position):
            occupied.append(np.asarray(e.position, float)[:2])   # already inside the bin
        else:
            candidates.append(e)

    slots = {c: np.asarray(task.bin_to_base([*xy, 0.0])[:2]) for c, xy in task.bin.slots.items() if c in wanted}
    reserved = list(slots.values())               # keep slot cells free for their own colour

    # Free grid cells, far side of the bin from the robot first so the arm never reaches over placed cubes.
    cells = [task.bin_to_base([cx, cy, 0.0])[:2] for cx, cy in BIN_GRID]
    cells.sort(key=lambda p: -np.linalg.norm(p))

    def free(p, ignore=None):
        for q in occupied:
            if np.linalg.norm(p - q) < min_dist:
                return False
        for q in reserved:
            if q is not ignore and np.linalg.norm(p - q) < min_dist:
                return False
        return True

    # Nearest to the bin first: short transports, and the cubes closest to the bin get cleared out of the way.
    bin_xy = np.asarray(task.bin.xy, float)
    candidates.sort(key=lambda e: np.linalg.norm(e.position[:2] - bin_xy))

    goals: list[PickPlaceGoal] = []
    for e in candidates:
        target = None
        slot = slots.pop(e.color, None)           # only the first cube of a colour gets the slot
        if slot is not None:
            if free(slot, ignore=slot):
                target = slot
            # slot blocked by something else: fall back to the grid
        if target is None:
            for p in cells:
                if free(p):
                    target = p
                    break
        if target is None:
            continue                              # bin is full
        if slot is not None and target is slot:
            reserved = [q for q in reserved if q is not slot]
        occupied.append(target)
        goals.append(PickPlaceGoal(cube=e, place_xy=np.asarray(target, float), place_yaw=place_yaw))
    return goals


def solve_ik(model: mujoco.MjModel, T_target: np.ndarray, q_init: np.ndarray, max_iter: int = 300,
             pos_tol: float = 1e-4, rot_tol: float = 1e-3) -> np.ndarray | None:
    """Damped-least-squares IK for body `hand` with joint-limit clipping. None if not converged."""
    data = mujoco.MjData(model)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    T_target = np.asarray(T_target, float)
    p_t, R_t = T_target[:3, 3], T_target[:3, :3]
    q = np.clip(np.asarray(q_init, float)[:7], Q_MIN, Q_MAX)
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    lam = 1e-2
    for _ in range(max_iter):
        data.qpos[:7] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        p = data.xpos[bid]
        R = data.xmat[bid].reshape(3, 3)
        e_p = p_t - p
        e_r = Rotation.from_matrix(R_t @ R.T).as_rotvec()
        if np.linalg.norm(e_p) < pos_tol and np.linalg.norm(e_r) < rot_tol:
            return q.copy()
        mujoco.mj_jacBody(model, data, jacp, jacr, bid)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])
        err = np.r_[e_p, e_r]
        dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), err)
        n = np.max(np.abs(dq))
        if n > 0.3:
            dq *= 0.3 / n
        q = np.clip(q + dq, Q_MIN, Q_MAX)
    return None


def cartesian_path(model: mujoco.MjModel, q_start: np.ndarray, T_target: np.ndarray, step: float = 0.01) -> list[np.ndarray] | None:
    """Straight-line hand motion (position lerp, orientation slerp) as sequential IK waypoints."""
    q_start = np.asarray(q_start, float)[:7]
    T0 = fk_hand(q_start)
    T_target = np.asarray(T_target, float)
    p0, p1 = T0[:3, 3], T_target[:3, 3]
    dist = float(np.linalg.norm(p1 - p0))
    rots = Rotation.from_matrix(np.stack([T0[:3, :3], T_target[:3, :3]]))
    ang = float(np.linalg.norm((rots[1] * rots[0].inv()).as_rotvec()))
    n = max(1, int(np.ceil(max(dist / step, ang / (step * 10)))))
    slerp = Slerp([0.0, 1.0], rots)
    path = [q_start.copy()]
    q = q_start.copy()
    for k in range(1, n + 1):
        s = k / n
        T = np.eye(4)
        T[:3, :3] = slerp([s]).as_matrix()[0]
        T[:3, 3] = p0 + s * (p1 - p0)
        q_new = solve_ik(model, T, q, max_iter=100)
        if q_new is None or np.max(np.abs(q_new - q)) > 0.3:
            return None
        path.append(q_new)
        q = q_new
    return path


def plan_path(q_start: np.ndarray, q_goal: np.ndarray, checker, rng: np.random.Generator | None = None,
              max_iter: int = 3000, step: float = 0.3, gripper_width: float = 0.08) -> list[np.ndarray] | None:
    """Collision-free joint path: direct segment, else RRT-Connect, then shortcutting."""
    rng = rng if rng is not None else np.random.default_rng(0)
    q_start = np.asarray(q_start, float)[:7]
    q_goal = np.asarray(q_goal, float)[:7]
    if np.any(q_goal < Q_MIN) or np.any(q_goal > Q_MAX):
        return None
    if checker.in_collision(q_goal, gripper_width) or checker.in_collision(q_start, gripper_width):
        return None
    free = lambda a, b: checker.segment_free(a, b, gripper_width=gripper_width)
    if free(q_start, q_goal):
        return [q_start, q_goal]

    def extend(nodes, parents, q_target):
        """Add one step from the nearest node toward q_target. Returns (status, index)."""
        arr = np.asarray(nodes)
        i = int(np.argmin(np.linalg.norm(arr - q_target, axis=1)))
        d = q_target - nodes[i]
        dist = np.linalg.norm(d)
        q_new = q_target.copy() if dist <= step else nodes[i] + d / dist * step
        if not free(nodes[i], q_new):
            return "trapped", None
        nodes.append(q_new)
        parents.append(i)
        return ("reached" if dist <= step else "advanced"), len(nodes) - 1

    def trace(nodes, parents, i):
        out = []
        while i != -1:
            out.append(nodes[i])
            i = parents[i]
        return out[::-1]

    ta, pa = [q_start], [-1]
    tb, pb = [q_goal], [-1]
    swapped = False
    path = None
    for _ in range(max_iter):
        q_rand = rng.uniform(Q_MIN, Q_MAX)
        st, ia = extend(ta, pa, q_rand)
        if st != "trapped":
            q_new = ta[ia]
            while True:
                st2, ib = extend(tb, pb, q_new)
                if st2 != "advanced":
                    break
            if st2 == "reached":
                first, second = trace(ta, pa, ia), trace(tb, pb, ib)[::-1]
                path = first + second[1:] if np.allclose(first[-1], second[0]) else first + second
                if swapped:
                    path = path[::-1]
                break
        ta, tb, pa, pb = tb, ta, pb, pa
        swapped = not swapped
    if path is None:
        return None
    # Shortcutting
    for _ in range(100):
        if len(path) <= 2:
            break
        i, j = sorted(rng.integers(0, len(path), size=2))
        if j - i < 2:
            continue
        if free(path[i], path[j]):
            path = path[:i + 1] + path[j:]
    return [np.asarray(q, float) for q in path]
