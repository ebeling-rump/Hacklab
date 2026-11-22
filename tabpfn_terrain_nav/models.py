"""Model zoo for per-site traversability classification.

Every model gets the same raw feature matrix (with NaNs). Models that cannot
handle missing values get median imputation plus missing-indicator columns.
All models use library defaults unless stated, except "XGBoost (tuned)" which
runs a small randomized search with cross-validation on the local data.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from terrain_sim import FEATURES

os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
warnings.filterwarnings("ignore")

IDX = {f: i for i, f in enumerate(FEATURES)}


class ConstantPrior:
    """Fallback when the local sample has a single class."""

    def fit(self, X, y):
        self.p = (y.sum() + 1) / (len(y) + 2)
        return self

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])


class GeometricRule:
    """Hand-tuned geometry thresholds, the classic costmap logic.

    Anything taller than 0.3 m, rough or steep is non-traversable; everything
    else is fine. Cells with no lidar returns are unknown (0.5).
    """

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        h, rough, slope = X[:, IDX["h_max"]], X[:, IDX["roughness"]], X[:, IDX["slope_deg"]]
        bad = (h > 0.3) | (np.nan_to_num(rough) > 0.06) | (np.nan_to_num(slope) > 15)
        p = np.where(bad, 0.95, 0.05)
        p = np.where(np.isnan(h), 0.5, p)
        return np.column_stack([1 - p, p])


class TabPFN:
    def __init__(self, n_estimators=8, seed=0):
        self.n_estimators, self.seed = n_estimators, seed

    def fit(self, X, y):
        import torch
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        torch.set_num_threads(2)
        self.m = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2, device="cpu", n_estimators=self.n_estimators, random_state=self.seed)
        self.m.fit(X, y)
        return self

    def predict_proba(self, X, chunk=4000):
        return np.vstack([self.m.predict_proba(X[i:i + chunk]) for i in range(0, len(X), chunk)])


def _imputed(est):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), est)


def _imputed_scaled(est):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), est)


class TunedXGB:
    """XGBoost with a 20-candidate randomized search, 3-fold CV on log loss."""

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, y):
        from xgboost import XGBClassifier

        k = int(min(3, np.bincount(y).min()))
        base = XGBClassifier(n_jobs=2, random_state=self.seed, verbosity=0)
        if k < 2:
            self.m = base.fit(X, y)
            return self
        space = {
            "n_estimators": [50, 100, 200, 400],
            "max_depth": [2, 3, 4, 6],
            "learning_rate": [0.02, 0.05, 0.1, 0.2],
            "min_child_weight": [1, 3, 5, 10],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.5, 0.8, 1.0],
            "reg_lambda": [0.1, 1, 10],
        }
        cv = StratifiedKFold(k, shuffle=True, random_state=self.seed)
        self.m = RandomizedSearchCV(base, space, n_iter=20, cv=cv, scoring="neg_log_loss",
                                    random_state=self.seed, n_jobs=1).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


def local_models(seed=0):
    """Models trained only on the local (this-site) calibration data."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "TabPFN v2": lambda: TabPFN(seed=seed),
        "CatBoost": lambda: CatBoostClassifier(verbose=0, random_seed=seed, thread_count=2, allow_writing_files=False),
        "XGBoost (tuned)": lambda: TunedXGB(seed=seed),
        "XGBoost": lambda: XGBClassifier(n_jobs=2, random_state=seed, verbosity=0),
        "LightGBM": lambda: LGBMClassifier(verbose=-1, n_jobs=2, random_state=seed),
        "Random forest": lambda: _imputed(RandomForestClassifier(500, n_jobs=2, random_state=seed)),
        "MLP": lambda: _imputed_scaled(MLPClassifier((64, 64), alpha=1e-3, max_iter=2000, random_state=seed)),
        "Logistic regression": lambda: _imputed_scaled(LogisticRegression(max_iter=2000)),
    }


def fit_predict(make, Xtr, ytr, Xte):
    if len(np.unique(ytr)) < 2:
        m = ConstantPrior().fit(Xtr, ytr)
    else:
        m = make().fit(Xtr, ytr)
    return np.clip(m.predict_proba(Xte)[:, 1], 1e-4, 1 - 1e-4)


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            out += sel.mean() * abs(y[sel].mean() - p[sel].mean())
    return out
