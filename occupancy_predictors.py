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
