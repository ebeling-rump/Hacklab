"""Analytical quasi-static pusher-slider model with online Bayesian system identification.

Model (Goyal et al. 1991; Lee & Cutkosky 1991; Lynch, Maekawa & Tanie 1992):
  * Ellipsoidal limit surface: body twist  V = (f_x, f_y, m / c^2)  for applied wrench (f, m).
  * Point pusher with Coulomb friction mu_p: sticking if the required contact force lies inside
    the friction cone, otherwise sliding along the cone edge (motion-cone model).
  * Integrated over the push with small substeps (contact point slides along the face).

Unknown parameters, estimated online from the robot's own pushes (MAP + Laplace posterior):
  com_x, com_y   centre of mass in body frame (pressure distribution centre)
  log c_scale    limit-surface radius relative to the uniform-pressure value
  logit mu_p     pusher-object friction
  log gain       actuator travel gain
The prior is the "nominal" model a roboticist would write down from geometry alone.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from shapely.geometry import Point

from .sim import PUSHER_R, GAP

N_SUB = 16


def mean_radius(shape, com=(0.0, 0.0), n=40):
    """Mean distance from `com` over the footprint (uniform pressure) -> c = m_max / f_max."""
    minx, miny, maxx, maxy = shape.polygon.bounds
    xs, ys = np.meshgrid(np.linspace(minx, maxx, n), np.linspace(miny, maxy, n))
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    from shapely import contains_xy
    inside = contains_xy(shape.polygon, pts[:, 0], pts[:, 1])
    pts = pts[inside]
    return float(np.mean(np.linalg.norm(pts - np.asarray(com), axis=1)))


def simulate_quasistatic(Q, com, c, mu_p, gain, n_sub=N_SUB):
    """Vectorised over N queries; params may be scalars or (N,) arrays. Returns (N, 3) body deltas."""
    p = Q["p"].copy()
    n_in = Q["n_in"]
    u0 = Q["u"]
    phi, d = Q["phi"], Q["d"]
    N = len(phi)
    com = np.broadcast_to(np.asarray(com, float), (N, 2)) if np.ndim(com) == 1 else np.asarray(com)
    c = np.broadcast_to(c, (N,))
    mu_p = np.broadcast_to(mu_p, (N,))
    gain = np.broadcast_to(gain, (N,))
    t_vec = np.stack([-n_in[:, 1], n_in[:, 0]], axis=1)  # tangent
    cosphi = np.cos(phi)
    tanphi = np.tan(phi)
    # first contact: pusher touches the face after travelling t_touch; contact shifted along tangent
    t_touch = PUSHER_R + GAP - PUSHER_R / np.maximum(cosphi, 0.2)
    # contact point = p - r_p (u / cos(phi) - n_in)   (flat-face geometry)
    p = p - PUSHER_R * (u0 / np.maximum(cosphi, 0.2)[:, None] - n_in)
    L = np.maximum(gain * (GAP + d) - t_touch, 0.0)
    ds = L / n_sub
    x = np.zeros(N); y = np.zeros(N); th = np.zeros(N)
    alpha = np.arctan(mu_p)
    ca, sa = np.cos(alpha), np.sin(alpha)
    fl = np.stack([ca * n_in[:, 0] - sa * n_in[:, 1], sa * n_in[:, 0] + ca * n_in[:, 1]], 1)
    fr = np.stack([ca * n_in[:, 0] + sa * n_in[:, 1], -sa * n_in[:, 0] + ca * n_in[:, 1]], 1)
    c2 = c ** 2
    for _ in range(n_sub):
        ct, st = np.cos(th), np.sin(th)
        vp = np.stack([ct * u0[:, 0] + st * u0[:, 1], -st * u0[:, 0] + ct * u0[:, 1]], 1)  # body frame
        r = p - com
        w = np.stack([-r[:, 1], r[:, 0]], 1)
        # A = I + w w^T / c^2 ; sticking force f = A^-1 vp  (Sherman-Morrison)
        ww = (w * w).sum(1)
        f_st = vp - w * ((w * vp).sum(1) / (c2 + ww))[:, None]

        def vc_of(f):
            return f + w * ((w * f).sum(1) / c2)[:, None]

        # inside friction cone?  angle(f, n_in) <= alpha
        fn = (f_st * n_in).sum(1)
        ft = (f_st * t_vec).sum(1)
        stick = (fn > 0) & (np.abs(ft) <= np.tan(alpha) * fn + 1e-12)
        f_edge = np.where((ft > 0)[:, None], fl, fr)
        # choose edge on the side of the sticking force tangential component
        # fl is rotated +alpha (towards +t), fr -alpha
        vce = vc_of(f_edge)
        kappa = (vp * n_in).sum(1) / np.maximum((vce * n_in).sum(1), 1e-9)
        f = np.where(stick[:, None], f_st, f_edge * kappa[:, None])
        f = np.where(((vp * n_in).sum(1) > 0)[:, None], f, 0.0)
        omega = (r[:, 0] * f[:, 1] - r[:, 1] * f[:, 0]) / c2
        v_com = f
        v_c = vc_of(f)
        # origin velocity (body frame): v_com + omega x (origin - com)
        v_o = v_com + omega[:, None] * np.stack([com[:, 1], -com[:, 0]], 1)
        # integrate (midpoint for heading)
        thm = th + 0.5 * omega * ds
        cm, sm = np.cos(thm), np.sin(thm)
        x += (cm * v_o[:, 0] - sm * v_o[:, 1]) * ds
        y += (sm * v_o[:, 0] + cm * v_o[:, 1]) * ds
        th += omega * ds
        slip = ((vp - v_c) * t_vec).sum(1)
        p = p + t_vec * (slip * ds)[:, None]
    return np.stack([x, y, th], 1)


class PhysicsModel:
    """Quasi-static model + MAP/Laplace online system identification."""
    name = "physics"

    def __init__(self, n_samples: int = 12, fit_params: bool = True, seed: int = 0):
        self.n_samples = n_samples
        self.fit_params = fit_params
        self.rng = np.random.default_rng(seed)

    def reset(self, shape, descriptor=None, prior_mean=None):
        self.shape = shape
        self.c0 = mean_radius(shape)
        minx, miny, maxx, maxy = shape.polygon.bounds
        hx, hy = (maxx - minx) / 2, (maxy - miny) / 2
        self.mu0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) if prior_mean is None else np.asarray(prior_mean)
        self.sd0 = np.array([0.15 * hx, 0.15 * hy, 0.3, 1.2, 0.12])
        self.theta = self.mu0.copy()
        self.cov = np.diag(self.sd0 ** 2)
        self.noise = np.array([0.006, 0.006, 0.06])
        self.n = 0
        return self

    def unpack(self, th):
        th = np.atleast_2d(th)
        com = th[:, :2]
        c = self.c0 * np.exp(th[:, 2])
        mu = 0.05 + 0.95 / (1 + np.exp(-(th[:, 3] + np.log(0.45 / 0.55))))
        gain = np.exp(th[:, 4])
        return com, c, mu, gain

    def _pred(self, Q, th):
        com, c, mu, gain = self.unpack(th)
        return simulate_quasistatic(Q, com[0], c[0], mu[0], gain[0])

    def fit(self, Q, Y):
        self.n = len(Y)
        if self.n == 0 or not self.fit_params:
            return self
        scale = np.array([0.01, 0.01, 0.1])

        def resid(th):
            r = (self._pred(Q, th) - Y) / scale
            return np.concatenate([r.ravel(), (th - self.mu0) / self.sd0])

        sol = least_squares(resid, self.theta, method="trf", x_scale=self.sd0, max_nfev=60)
        self.theta = sol.x
        J = sol.jac
        r = sol.fun[: 3 * self.n].reshape(-1, 3) * scale
        rms = np.sqrt(np.mean(r ** 2, axis=0)) if self.n >= 3 else self.noise
        self.noise = np.maximum(np.where(self.n >= 3, rms * np.sqrt(self.n / max(self.n - 2, 1)), self.noise),
                                [0.002, 0.002, 0.01])
        try:
            self.cov = np.linalg.inv(J.T @ J + 1e-9 * np.eye(5))
        except np.linalg.LinAlgError:
            self.cov = np.diag(self.sd0 ** 2)
        return self

    def predict(self, Q):
        mean = self._pred(Q, self.theta)
        if self.n_samples <= 0:
            return mean, np.broadcast_to(self.noise, mean.shape).copy()
        L = np.linalg.cholesky(self.cov + 1e-12 * np.eye(5))
        preds = []
        for _ in range(self.n_samples):
            th = self.theta + L @ self.rng.normal(size=5)
            preds.append(self._pred(Q, th))
        preds = np.array(preds)
        std = np.sqrt(preds.var(0) + self.noise ** 2)
        return mean, std

    def nominal(self, Q):
        return self._pred(Q, self.mu0)
