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
