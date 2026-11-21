"""Shared predictor interface for occupancy of unobserved grid cells.

Features X are cell-centre coordinates (n, 2) in metres (world x, y); labels y
are 0 (free) / 1 (occupied). Later TabPFN and Gaussian Process models should
subclass OccupancyPredictor so asm_experiment.py can evaluate them unchanged.
"""

from abc import ABC, abstractmethod

import numpy as np


class OccupancyPredictor(ABC):
    name = "base"

    @abstractmethod
    def fit(self, X_obs: np.ndarray, y_obs: np.ndarray) -> "OccupancyPredictor":
        """X_obs (n, 2) cell coordinates, y_obs (n,) in {0, 1}."""

    @abstractmethod
    def predict(self, X_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean_prob, uncertainty), each (m,). mean_prob = P(occupied)."""


class ConstantPrior(OccupancyPredictor):
    """P(occupied) = fraction of occupied observed cells, everywhere."""
    name = "constant_prior"

    def fit(self, X_obs, y_obs):
        self.p = float(np.mean(y_obs)) if len(y_obs) else 0.5
        return self

    def predict(self, X_query):
        n = len(X_query)
        # Bernoulli std as uncertainty.
        return np.full(n, self.p), np.full(n, np.sqrt(self.p * (1 - self.p)))


class NearestObserved(OccupancyPredictor):
    """Copy the label of the nearest observed cell. Uncertainty grows with distance."""
    name = "nearest_observed"

    def __init__(self, length_scale: float = 0.5):
        self.length_scale = length_scale

    def fit(self, X_obs, y_obs):
        self.X, self.y = np.asarray(X_obs, float), np.asarray(y_obs, float)
        return self

    def predict(self, X_query):
        X_query = np.asarray(X_query, float)
        mean = np.empty(len(X_query))
        dist = np.empty(len(X_query))
        for s in range(0, len(X_query), 1024):            # chunked brute force
            d2 = ((X_query[s:s + 1024, None, :] - self.X[None]) ** 2).sum(-1)
            j = d2.argmin(1)
            mean[s:s + 1024] = self.y[j]
            dist[s:s + 1024] = np.sqrt(d2[np.arange(len(j)), j])
        return mean, 1 - np.exp(-dist / self.length_scale)


def _bernoulli_std(p):
    return np.sqrt(np.clip(p, 0, 1) * (1 - np.clip(p, 0, 1)))


class GaussianProcess(OccupancyPredictor):
    """GP regression on 0/1 labels (RBF + white noise). Uncertainty = posterior std,
    i.e. epistemic: it grows away from observed cells."""
    name = "gp"

    def __init__(self, length_scale: float = 0.3, seed: int = 0):
        self.length_scale, self.seed = length_scale, seed

    def fit(self, X_obs, y_obs):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        kernel = (ConstantKernel(0.1, (1e-3, 10)) * RBF(self.length_scale, (0.05, 5))
                  + WhiteKernel(0.01, (1e-5, 1)))
        self.gp = GaussianProcessRegressor(kernel, normalize_y=True, random_state=self.seed)
        self.gp.fit(np.asarray(X_obs, float), np.asarray(y_obs, float))
        return self

    def predict(self, X_query):
        mean, std = self.gp.predict(np.asarray(X_query, float), return_std=True)
        return np.clip(mean, 0, 1), std


class TabPFN(OccupancyPredictor):
    """TabPFNClassifier via the cloud API (tabpfn_client, needs TABPFN_TOKEN).
    Uncertainty = Bernoulli std of the predicted probability."""
    name = "tabpfn"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X_obs, y_obs):
        try:
            import truststore
            truststore.inject_into_ssl()
        except ImportError:
            pass
        from tabpfn_client import TabPFNClassifier
        y = np.asarray(y_obs, int)
        self.single = None if len(np.unique(y)) > 1 else float(y[0])
        if self.single is None:
            self.clf = TabPFNClassifier(random_state=self.seed).fit(np.asarray(X_obs, float), y)
        return self

    def predict(self, X_query):
        if self.single is not None:
            p = np.full(len(X_query), self.single)
        else:
            proba = self.clf.predict_proba(np.asarray(X_query, float))
            p = proba[:, list(self.clf.classes_).index(1)]
        return p, _bernoulli_std(p)


# ---------- variants for the TabPFN follow-up test ----------

class SpatialFeatures:
    """Context features from the observed (training) cells only: x, y, distance to the
    nearest observed occupied / free cell and counts of each within `radius`. For training
    rows the cell itself is excluded (leave-one-out), so a training cell never sees its own
    label; the query cells are unobserved, so nothing to exclude there."""

    def __init__(self, radius: float = 0.35):
        self.radius = radius

    def fit(self, X_obs, y_obs):
        from scipy.spatial import cKDTree
        from sklearn.preprocessing import StandardScaler
        X_obs, y_obs = np.asarray(X_obs, float), np.asarray(y_obs, int)
        self.trees = {c: cKDTree(X_obs[y_obs == c]) if (y_obs == c).any() else None for c in (0, 1)}
        F = self._features(X_obs, y_obs)
        self.scaler = StandardScaler().fit(F)
        return self.scaler.transform(F)

    def transform(self, X_query):
        return self.scaler.transform(self._features(np.asarray(X_query, float), None))

    def _features(self, X, y):
        cols = [X[:, 0], X[:, 1]]
        for c in (1, 0):
            tree = self.trees[c]
            if tree is None:
                cols += [np.full(len(X), 10.0), np.zeros(len(X))]
                continue
            k = min(2, tree.n)
            d, _ = tree.query(X, k=k)
            d = d.reshape(len(X), k)
            n = np.array([len(i) for i in tree.query_ball_point(X, self.radius)], float)
            if y is not None:                       # leave-one-out for own-class training rows
                own = y == c
                d0 = np.where(own & (d[:, 0] == 0) & (k > 1), d[:, -1], d[:, 0])
                n = n - own
            else:
                d0 = d[:, 0]
            cols += [d0, n]
        return np.column_stack(cols)


class WithFeatures(OccupancyPredictor):
    """Wrap a predictor so it sees SpatialFeatures instead of raw (x, y)."""

    def __init__(self, base: OccupancyPredictor, name: str):
        self.base, self.name, self.feat = base, name, SpatialFeatures()

    def fit(self, X_obs, y_obs):
        self.base.fit(self.feat.fit(X_obs, y_obs), y_obs)
        return self

    def predict(self, X_query):
        return self.base.predict(self.feat.transform(X_query))


class Rebalanced(OccupancyPredictor):
    """Fit the base predictor on all occupied cells plus `ratio` x as many free cells
    (random, seeded). Raw probabilities, no prior correction."""

    def __init__(self, base: OccupancyPredictor, name: str, ratio: float = 3.0, seed: int = 0):
        self.base, self.name, self.ratio, self.seed = base, name, ratio, seed

    def fit(self, X_obs, y_obs):
        X_obs, y_obs = np.asarray(X_obs, float), np.asarray(y_obs, int)
        occ, free = np.flatnonzero(y_obs == 1), np.flatnonzero(y_obs == 0)
        n_free = min(len(free), int(self.ratio * max(len(occ), 1)))
        free = np.random.default_rng(self.seed).choice(free, n_free, replace=False)
        idx = np.sort(np.concatenate([occ, free]))
        self.base.fit(X_obs[idx], y_obs[idx])
        return self

    def predict(self, X_query):
        return self.base.predict(X_query)
