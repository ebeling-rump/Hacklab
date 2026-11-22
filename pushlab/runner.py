"""Closed-loop episodes: observe -> (learn) -> plan -> push -> record."""
from __future__ import annotations

import time

import numpy as np

from .actions import ActionSampler, query_from_actions, concat, to_action
from .physics_model import mean_radius
from .planner import CostField, FreeSpaceCost, Planner, in_walls, pusher_starts_world
from .sim import PushSim, body_delta, compose, wrap, PUSHER_R, GAP


class ReactiveController:
    """Adaptive engineering baseline (no dynamics model, no lookahead).

    Pushes along the direction to the (carrot) target, with a lever arm chosen to correct the
    orientation error. Online it fits a linear rotation law
        dtheta/d = a*arm + b*u_y + c*u_x        (b, c absorb the unknown centre of mass)
    and a translation gain g = moved / commanded, by recursive least squares.
    """
    name = "reactive"

    def reset(self, shape):
        c0 = mean_radius(shape)
        self.w = np.array([1.0 / c0 ** 2 * 0.6, 0.0, 0.0])
        self.P = np.diag([1.0 / c0 ** 4, 50.0, 50.0])
        self.g = 0.8
        self.timer = None
        return self

    def update(self, Q, y):
        u = Q["u"][0]
        arm = Q["p"][0, 0] * u[1] - Q["p"][0, 1] * u[0]
        x = np.array([arm, u[1], u[0]])
        d = Q["d"][0]
        target = y[2] / max(d, 1e-3)
        # RLS update
        Px = self.P @ x
        k = Px / (0.05 + x @ Px)
        self.w = self.w + k * (target - x @ self.w)
        self.P = self.P - np.outer(k, Px)
        moved = np.hypot(y[0], y[1])
        self.g = 0.7 * self.g + 0.3 * np.clip(moved / max(d + GAP, 1e-3), 0.1, 2.0)

    def choose(self, pose, target, Q, use_theta: bool, walls=()):
        pose = np.asarray(pose)
        dxy = np.asarray(target[:2]) - pose[:2]
        dist = np.linalg.norm(dxy)
        th = pose[2]
        c, s = np.cos(th), np.sin(th)
        u_w = np.stack([c * Q["u"][:, 0] - s * Q["u"][:, 1], s * Q["u"][:, 0] + c * Q["u"][:, 1]], 1)
        arm = Q["p"][:, 0] * Q["u"][:, 1] - Q["p"][:, 1] * Q["u"][:, 0]
        X = np.column_stack([arm, Q["u"][:, 1], Q["u"][:, 0]])
        dth_pred = (X @ self.w) * Q["d"]
        trans_pred = self.g * (Q["d"] + GAP)
        dth_err = float(wrap(target[2] - th)) if use_theta else 0.0
        want_rot = np.clip(dth_err, -0.6, 0.6)
        if dist > 1e-6:
            align = (u_w @ (dxy / dist))
        else:
            align = np.zeros(len(arm))
        dist_err = np.abs(trans_pred * np.clip(align, 0, 1) - dist)
        cost = 2.0 * (1 - align) + dist_err / 0.05 + np.abs(dth_pred - want_rot) / 0.25
        if dist < 0.03 and use_theta:  # mostly rotate in place
            cost = np.abs(dth_pred - want_rot) / 0.1 + trans_pred / 0.05 * (1 - np.abs(np.cos(np.arctan2(arm, 0.02))))
        if walls:
            starts = pusher_starts_world(Q, pose)
            cost = cost + 100 * in_walls(starts, walls, PUSHER_R + 0.003)
        return int(np.argmin(cost))


def carrot(field, pose, steps=6):
    """Greedy descent of the lattice cost-to-go field from `pose` (for the reactive controller)."""
    p = np.array(pose, float)
    res, dth = field.res, 2 * np.pi / field.nt
    moves = np.array([[dx, dy, dt] for dx in (-res, 0, res) for dy in (-res, 0, res) for dt in (-dth, 0, dth)])
    for _ in range(steps):
        cand = p[None] + moves
        cand[:, 2] = wrap(cand[:, 2])
        c = field(cand)
        p = cand[int(np.argmin(c))]
    return p


def run_episode(task, method: str, model=None, seed=0, n_cand=96, record=False, planner_kw=None,
                prior_data=None, sim_noise=True, log_predictions=False):
    """Run one episode. `model` implements reset/fit/predict (None for reactive)."""
    obj = task.obj
    sim = PushSim(obj, walls=task.walls, seed=seed, noise=sim_noise)
    sim.set_pose(task.start)
    sampler = ActionSampler(obj.shape, seed=seed)
    desc = obj.shape.descriptor()
    use_theta = task.ang_tol is not None
    t_plan0 = time.perf_counter()
    if task.walls:
        cost = CostField(obj.shape, task.walls, task.bounds, task.goal, task.pos_tol * 0.7,
                         None if task.ang_tol is None else task.ang_tol * 0.7)
    else:
        cost = FreeSpaceCost(task.goal, use_theta=use_theta)
    field_s = time.perf_counter() - t_plan0
    kw = dict(walls=task.walls)
    kw.update(planner_kw or {})
    planner = Planner(cost, **kw)
    if model is not None:
        model.reset(obj.shape, desc)
    reactive = ReactiveController().reset(obj.shape) if method == "reactive" else None

    Qs, Ys = [], []
    log = dict(poses=[sim.observe()], true=[sim.true_pose()], actions=[], pred=[], pred_std=[],
               walls=[], traj=[], cand_pred=[], plan_s=[], model_s=[])
    pusher_path = 0.0
    last_end = None
    success = False
    n_wall = 0
    stalls = 0
    for k in range(task.max_pushes):
        pose = log["poses"][-1]
        if _success(task, pose):
            success = True
            break
        A = sampler.sample(n_cand)
        Q = query_from_actions(obj.shape, A, desc)
        t0 = time.perf_counter()
        if method == "reactive":
            tgt = carrot(cost, pose) if task.walls else task.goal
            i = reactive.choose(pose, tgt, Q, use_theta or bool(task.walls), task.walls)
            mean = np.zeros((len(A), 3)); std = np.zeros((len(A), 3))
            t1 = t0
        else:
            mean, std = model.predict(Q)
            t1 = time.perf_counter()
            i, info = planner.choose(pose, Q, mean, std, len(Ys))
        t2 = time.perf_counter()
        log["model_s"].append(t1 - t0)
        log["plan_s"].append(t2 - t1)
        a = A[i]
        start = pusher_starts_world({k_: v[i:i + 1] for k_, v in Q.items() if isinstance(v, np.ndarray)}, pose)[0]
        if last_end is not None:
            pusher_path += np.linalg.norm(start - last_end)
        res = sim.push(to_action(a), observed_pose=pose, record=record)
        if not res.valid:
            stalls += 1
            # re-observe and try again next step (counts as a wasted action)
            log["poses"].append(sim.observe()); log["true"].append(sim.true_pose())
            log["actions"].append(a); log["pred"].append(compose(pose, mean[i])); log["pred_std"].append(std[i])
            log["walls"].append(False)
            continue
        u_w = start - pose[:2]
        pusher_path += GAP + a[2]
        last_end = start
        y = body_delta(res.pose_before, res.pose_after)
        qi = {k_: v[i:i + 1] for k_, v in Q.items() if isinstance(v, np.ndarray)}
        Qs.append(qi); Ys.append(y)
        n_wall += int(res.wall_contact)
        log["poses"].append(res.pose_after); log["true"].append(res.true_after)
        log["actions"].append(a); log["pred"].append(compose(pose, mean[i])); log["pred_std"].append(std[i])
        log["walls"].append(res.wall_contact)
        if record:
            log["traj"].append(res.traj)
        if log_predictions:
            log["cand_pred"].append(dict(A=A, mean=mean, std=std, chosen=i))
        if method == "reactive":
            reactive.update(qi, y)
        else:
            Qc = concat(Qs)
            model.fit(Qc, np.array(Ys))
    else:
        success = _success(task, log["poses"][-1])
    final = np.asarray(log["true"][-1])
    pos_err = float(np.hypot(*(final[:2] - np.asarray(task.goal[:2]))))
    ang_err = float(np.degrees(abs(wrap(final[2] - task.goal[2])))) if use_theta else np.nan
    n_push = len(log["actions"])
    out = dict(challenge=task.challenge, seed=task.seed, method=method, obj=obj.name, success=bool(success),
               pushes=n_push, pos_err_mm=1000 * pos_err, ang_err_deg=ang_err, wall_contacts=n_wall,
               invalid_pushes=stalls, pusher_path_m=pusher_path,
               model_s_mean=float(np.mean(log["model_s"])) if log["model_s"] else 0.0,
               plan_s_mean=float(np.mean(log["plan_s"])) if log["plan_s"] else 0.0, field_s=field_s,
               sim_time_s=sum((GAP + a[2]) / 0.15 + 0.15 for a in log["actions"]))
    return out, log


def _success(task, pose):
    if np.hypot(pose[0] - task.goal[0], pose[1] - task.goal[1]) > task.pos_tol:
        return False
    if task.ang_tol is not None and abs(wrap(pose[2] - task.goal[2])) > task.ang_tol:
        return False
    return True
