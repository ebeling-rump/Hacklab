"""Push action libraries and the observable feature representation shared by all models.

A query `Q` is a dict of arrays describing N pushes in the object's BODY frame (origin = the
observable geometric centroid). Everything in Q is observable: geometry of the contact and the
commanded push. Hidden parameters never appear here.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc
from shapely.geometry import Point, LineString

from .sim import PUSHER_R, GAP, Action, PUSH_SPEED

PHI_MAX = 0.6
D_RANGE = (0.02, 0.10)


def query_from_actions(shape, actions: np.ndarray, descriptor=None) -> dict:
    """actions: (N, 3) array of (s, phi, d)."""
    actions = np.atleast_2d(actions)
    if actions.shape[1] == 3:
        actions = np.column_stack([actions, np.full(len(actions), PUSH_SPEED)])
    s, phi, d, v = actions[:, 0], actions[:, 1], actions[:, 2], actions[:, 3]
    p, n_out, edge = shape.perimeter_point(s)
    n_in = -n_out
    c, sn = np.cos(phi), np.sin(phi)
    u = np.stack([c * n_in[:, 0] - sn * n_in[:, 1], sn * n_in[:, 0] + c * n_in[:, 1]], axis=1)
    Q = dict(s=s, phi=phi, d=d, v=v, p=p, n_in=n_in, u=u, edge=edge, actions=actions)
    if descriptor is not None:
        Q["desc"] = np.repeat(np.asarray(descriptor)[None], len(s), axis=0)
    return Q


def subset(Q: dict, idx) -> dict:
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(Q["s"]) else v)
            for k, v in Q.items()}


def concat(Qs: list[dict]) -> dict:
    keys = [k for k in Qs[0] if isinstance(Qs[0][k], np.ndarray)]
    return {k: np.concatenate([q[k] for q in Qs], axis=0) for k in keys}


def features(Q: dict, with_desc: bool = False) -> np.ndarray:
    """Observable feature matrix (body frame).

    [p_x, p_y, n_in_x, n_in_y, u_x, u_y, phi, d, torque_arm, v]
    torque_arm = p x u (signed lever arm of the push line about the geometric centroid) - an
    observable, physics-motivated feature that every learner gets.
    """
    p, u, n = Q["p"], Q["u"], Q["n_in"]
    arm = p[:, 0] * u[:, 1] - p[:, 1] * u[:, 0]
    X = np.column_stack([p[:, 0], p[:, 1], n[:, 0], n[:, 1], u[:, 0], u[:, 1], Q["phi"], Q["d"], arm, Q["v"]])
    if with_desc and "desc" in Q:
        X = np.column_stack([X, Q["desc"]])
    return X


def _reachable(shape, s, phi, margin=0.001):
    """Pusher start pose clear of the object and first contact near the nominal point."""
    Q = query_from_actions(shape, np.column_stack([s, phi, np.full_like(s, 0.05)]))
    ok = np.zeros(len(s), bool)
    poly = shape.polygon
    for i in range(len(s)):
        start = Q["p"][i] - Q["u"][i] * (PUSHER_R + GAP)
        if Point(*start).buffer(PUSHER_R + margin).intersects(poly):
            continue
        # The pusher centre line must not hit the object far before the nominal point.
        cphi = max(np.cos(phi[i]), 0.3)
        end = Q["p"][i] - Q["u"][i] * (PUSHER_R / cphi + 0.004)
        hit = LineString([start, end]).buffer(PUSHER_R * 0.95).intersection(poly)
        ok[i] = hit.area < 1e-6
    return ok


class ActionSampler:
    """Samples valid push actions for a shape (quasi-random, filtered for reachability)."""

    def __init__(self, shape, seed: int = 0):
        self.shape = shape
        self.rng = np.random.default_rng(seed)
        # Pre-compute a dense reachability table over (s, phi).
        self._s_grid = np.linspace(0, 1, 240, endpoint=False)
        self._phi_grid = np.linspace(-PHI_MAX, PHI_MAX, 9)
        S, P = np.meshgrid(self._s_grid, self._phi_grid, indexing="ij")
        self._ok = _reachable(shape, S.ravel(), P.ravel()).reshape(S.shape)

    def valid(self, s, phi):
        i = np.clip(np.round(np.asarray(s) % 1 * len(self._s_grid)).astype(int) % len(self._s_grid), 0, len(self._s_grid) - 1)
        j = np.clip(np.round((np.asarray(phi) + PHI_MAX) / (2 * PHI_MAX) * (len(self._phi_grid) - 1)).astype(int), 0,
                    len(self._phi_grid) - 1)
        return self._ok[i, j]

    def sample(self, n: int, d_range=D_RANGE, phi_max=PHI_MAX, v_range=(PUSH_SPEED, PUSH_SPEED)) -> np.ndarray:
        out = []
        for _ in range(50):
            if sum(len(o) for o in out) >= n:
                break
            sob = qmc.Sobol(4, scramble=True, seed=self.rng).random_base2(int(np.ceil(np.log2(max(64, 2 * n)))))
            s = sob[:, 0]
            phi = (sob[:, 1] * 2 - 1) * phi_max
            d = d_range[0] + sob[:, 2] * (d_range[1] - d_range[0])
            ok = self.valid(s, phi)
            v = v_range[0] + sob[:, 3] * (v_range[1] - v_range[0])
            out.append(np.column_stack([s, phi, d, v])[ok])
        a = np.concatenate(out)[:n]
        return a


def to_action(a) -> Action:
    return Action(float(a[0]), float(a[1]), float(a[2]), float(a[3]) if len(a) > 3 else PUSH_SPEED)
