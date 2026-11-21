"""Phase 3: learned obstacle predictors (GP, TabPFN) for the planner.

Both models predict the AGV's deviation from constant velocity at +0.5 s and +1.0 s from
observable features (motion_data.FEATURES), trained on the Phase 2 train set. The planner
needs positions at 0.1 ... 1.0 s, so the deviation is interpolated piecewise-linearly
through 0 at t = 0, the +0.5 s value and the +1.0 s value.

Latency is modelled in *simulation time* (see AsyncPredictor): a prediction requested at
sim time t becomes usable at t + latency. Until then the planner uses the previous
prediction (shifted by its age) or constant velocity. The safety filter never waits for it.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from motion_data import FEATURES, MOTION, TARGETS, load

HISTORY_NEEDED = 5            # 0.4 s of 10 Hz history for velocity + acceleration
TARGET_TIMES = np.array([0.5, 1.0])


# ---------- features, identical to motion_data.build_rows ----------

def features_from_history(history, robot_pos, dt):
    """history: sequence of (n_obs, 2) observed positions, oldest first. Returns (n_obs, 9) or None."""
    if len(history) < HISTORY_NEEDED:
        return None
    h = np.asarray(history)[-HISTORY_NEEDED:]
    p = h[-1]
    v = (h[-1] - h[-2]) / dt
    v_old = (h[-4] - h[-5]) / dt
    acc = (v - v_old) / (3 * dt)
    rel = robot_pos - p
    dist = np.linalg.norm(rel, axis=1, keepdims=True)
    X = np.hstack([p, v, acc, rel, dist])
    assert X.shape[1] == len(FEATURES)
    return X


def path_from_residuals(p, v, res, times):
    """Positions (n, K, 2) at `times` from current p, v (n, 2) and residuals (n, 4): x05, y05, x10, y10."""
    r05, r10 = res[:, 0:2], res[:, 2:4]
    t = np.asarray(times)[None, :, None]
    r = np.where(t <= 0.5, r05[:, None] * (t / 0.5),
                 r05[:, None] + (r10 - r05)[:, None] * ((t - 0.5) / 0.5))
    return p[:, None] + v[:, None] * t + r


def training_rows(n, seed=0):
    """The same nested subsample of the Phase 2 train set as predict_motion.py."""
    train = load("train")
    order = np.random.default_rng(seed).permutation(len(train["x"]))
    idx = order[:n]
    X = np.column_stack([train[c] for c in FEATURES])[idx]
    Y = np.column_stack([train[c] for c in TARGETS])[idx]
    return X, Y


# ---------- models ----------

class GPModel:
    """Four Gaussian Processes (one per target), fitted once and cached with joblib."""

    def __init__(self, n=1000):
        import joblib
        path = MOTION / f"gp_{n}.joblib"
        if path.exists():
            self.scaler, self.gps = joblib.load(path)
        else:
            import warnings

            from sklearn.exceptions import ConvergenceWarning
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
            from sklearn.preprocessing import StandardScaler
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            X, Y = training_rows(n)
            self.scaler = StandardScaler().fit(X)
            self.gps = []
            for j in range(Y.shape[1]):
                kernel = ConstantKernel(0.1) * RBF(np.ones(X.shape[1])) + WhiteKernel(1e-3)
                gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=1, random_state=0)
                self.gps.append(gp.fit(self.scaler.transform(X), Y[:, j]))
            joblib.dump((self.scaler, self.gps), path)

    def predict(self, X):
        Xs = self.scaler.transform(X)
        return np.column_stack([gp.predict(Xs) for gp in self.gps])


class TabPFNModel:
    """TabPFN via the cloud API. combined=True: one model, target index as extra feature."""

    def __init__(self, n=1000, combined=True):
        try:
            import truststore
            truststore.inject_into_ssl()
        except ImportError:
            pass
        from tabpfn_client import TabPFNRegressor

        self.combined = combined
        X, Y = training_rows(n)
        if combined:
            self.model = TabPFNRegressor(random_state=0)
            self.model.fit(*self._stack(X, Y))
        else:
            self.models = [TabPFNRegressor(random_state=0).fit(X, Y[:, j]) for j in range(Y.shape[1])]
            self.pool = ThreadPoolExecutor(max_workers=len(self.models))

    @staticmethod
    def _stack(X, Y=None):
        """Rows (features, horizon, axis) for each of the 4 targets."""
        n = len(X)
        tags = np.array([[0.5, 0], [0.5, 1], [1.0, 0], [1.0, 1]])     # x05, y05, x10, y10
        Xs = np.vstack([np.hstack([X, np.tile(t, (n, 1))]) for t in tags])
        return (Xs, None if Y is None else Y.T.reshape(-1))

    def predict(self, X):
        if self.combined:
            pred = self.model.predict(self._stack(X)[0])
            return pred.reshape(4, len(X)).T
        futures = [self.pool.submit(m.predict, X) for m in self.models]
        return np.column_stack([f.result() for f in futures])


# ---------- predictor with latency, plugged into the planner ----------

class AsyncPredictor:
    """Learned predictor with inference latency modelled in simulation time.

    At most one request is in flight. A request made at sim time t (with the observations
    of time t) is computed immediately, its wall-clock duration L is measured, and the result
    becomes usable at sim time t + L (or t, if ideal=True: "the world waits for the model").
    Between results, the latest prediction is used shifted by its age; once it is older than
    `max_age`, or before the first one, constant velocity is used.
    """

    def __init__(self, model, dt, ideal=False, min_period=0.0, max_age=0.5, radius=4.0):
        self.model, self.dt, self.ideal = model, dt, ideal
        self.min_period, self.max_age, self.radius = min_period, max_age, radius
        self.latest = None            # (t_obs, path (n, K', 2) on a fine grid)
        self.pending = None           # (t_ready, t_obs, path)
        self.last_request = -np.inf
        self.latencies = []           # wall-clock seconds per model call
        self.ages = []                # age of the prediction used at each control step (s)
        self.fallbacks = 0            # control steps that used constant velocity
        self.errors = []              # failed model calls (message), e.g. API quota or network
        self.grid = np.round(np.arange(0.0, 2.01, dt), 6)   # predicted path from 0 to 2 s

    def __call__(self, tracker, times, robot_pos=None, t=0.0, **_):
        if self.pending is not None and t >= self.pending[0] - 1e-9:
            self.latest, self.pending = self.pending[1:], None
        if self.pending is None and t - self.last_request >= self.min_period - 1e-9:
            self._request(tracker, robot_pos, t)
            if self.ideal and self.pending is not None:
                self.latest, self.pending = self.pending[1:], None

        cv = tracker.position[:, None] + tracker.velocity[:, None] * np.asarray(times)[None, :, None]
        if self.latest is None or t - self.latest[0] > self.max_age + 1e-9:
            self.fallbacks += 1
            return cv
        age = t - self.latest[0]
        self.ages.append(age)
        idx = np.clip(np.round((age + np.asarray(times)) / self.dt).astype(int), 0, len(self.grid) - 1)
        return self.latest[1][:, idx]

    def _request(self, tracker, robot_pos, t):
        X = features_from_history(tracker.history, robot_pos, self.dt)
        if X is None:
            return
        self.last_request = t
        p, v = tracker.position, tracker.velocity
        near = np.linalg.norm(p - robot_pos, axis=1) < self.radius
        res = np.zeros((len(p), 4))
        if near.any():                                   # far AGVs: constant velocity, no model call
            t0 = time.perf_counter()
            try:
                res[near] = self.model.predict(X[near])
            except Exception as exc:                     # network, quota, server error: never crash the robot
                self.errors.append(f"{type(exc).__name__}: {str(exc)[:120]}")
                latency = time.perf_counter() - t0
                self.latencies.append(latency)
                return                                   # no new prediction; CV fallback kicks in when stale
            latency = time.perf_counter() - t0
            self.latencies.append(latency)
        else:
            latency = 0.0
        path = path_from_residuals(p, v, res, self.grid)
        delay = 0.0 if self.ideal else latency
        self.pending = (t + delay, t, path)

    def stats(self):
        lat = np.array(self.latencies) if self.latencies else np.array([np.nan])
        return {"model_calls": len(self.latencies), "latency_mean_ms": 1000 * float(np.mean(lat)),
                "latency_p95_ms": 1000 * float(np.percentile(lat, 95)),
                "prediction_age_ms": 1000 * float(np.mean(self.ages)) if self.ages else float("nan"),
                "cv_fallback_steps": self.fallbacks, "model_errors": len(self.errors)}
