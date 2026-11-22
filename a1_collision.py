"""Shared pieces of the TabPFN collision safety filter for the Unitree A1.

Turns the samples written by a1_collision_dataset.py into a table:

    one row per sample =  lidar scan (min-pooled into sectors)
                        + the candidate command (forward, turn)
                        + the previous command (the gait smooths commands, so it
                          still matters at the start)
    label              =  did the robot touch a box while holding the candidate
                          command for the next 0.8 s?

and wraps TabPFN (local, v3.5-fast on CPU) as a collision-risk model.
"""

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "datasets"
N_SECTORS = 24          # 72 rays -> 24 sectors of 15 degrees (min range per sector)
CONTEXT_ROWS = 1500     # TabPFN training rows ("context"); CPU cost grows with this
N_ESTIMATORS = 4        # TabPFN ensemble size; 4 keeps one 15-candidate query ~0.3 s


def sectors(lidar: np.ndarray) -> np.ndarray:
    """(..., 72) ranges -> (..., N_SECTORS) closest range per sector.

    Ray 0 points backwards (angle -pi), so sector N_SECTORS/2 is straight ahead.
    """
    return lidar.reshape(*lidar.shape[:-1], N_SECTORS, -1).min(-1)


def feature_names() -> list[str]:
    deg = np.degrees(np.linspace(-np.pi, np.pi, N_SECTORS, endpoint=False)) + 7.5
    return ([f"lidar_{d:+.0f}deg" for d in deg]
            + ["forward", "turn", "prev_forward", "prev_turn"])


def features(lidar: np.ndarray, action: np.ndarray, prev_action: np.ndarray) -> np.ndarray:
    """Feature rows for (one or many) lidar scans + commands."""
    lidar, action, prev_action = (np.atleast_2d(a) for a in (lidar, action, prev_action))
    n = max(len(lidar), len(action), len(prev_action))
    rep = lambda a: np.repeat(a, n, 0) if len(a) == 1 else a
    return np.hstack([sectors(rep(lidar)), rep(action), rep(prev_action)]).astype(np.float32)


def load_table(env: str):
    """(X, y) for one layout from datasets/collision_<env>.npz (a1_collision_dataset.py).

    Label: collision within the next 0.8 s while holding the command.
    """
    d = np.load(DATA_DIR / f"collision_{env}.npz")
    X = features(d["lidar"], d["action"], d["prev_action"])
    return X, d["collision"].astype(int)


def context_subsample(X, y, n=CONTEXT_ROWS, pos_fraction=0.3, seed=0):
    """Pick the TabPFN context, oversampling collisions if they are rare."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_pos = min(len(pos), int(pos_fraction * n))
    idx = np.r_[rng.choice(pos, n_pos, replace=False),
                rng.choice(neg, min(len(neg), n - n_pos), replace=False)]
    return X[idx], y[idx]


class CollisionRisk:
    """P(collision during the next step) from TabPFN.

    The context is class-rebalanced, so raw TabPFN probabilities are too high;
    they are shifted back to the real collision rate (Bayes prior correction).
    """

    def __init__(self, n_estimators=N_ESTIMATORS, seed=0):
        import torch
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        torch.set_num_threads(4)
        self.model = TabPFNClassifier.create_default_for_version(
            ModelVersion.V3_5_FAST, device="cpu", n_estimators=n_estimators,
            fit_mode="fit_with_cache", random_state=seed)

    def fit(self, X, y, true_rate=None, **subsample_kw):
        true_rate = y.mean() if true_rate is None else true_rate
        Xc, yc = context_subsample(X, y, **subsample_kw)
        ctx_rate = yc.mean()
        # Odds multiplier that undoes the oversampling of collisions.
        self.odds_scale = (true_rate / (1 - true_rate)) / (ctx_rate / (1 - ctx_rate))
        self.model.fit(Xc, yc)
        return self

    def predict(self, X) -> np.ndarray:
        p = self.model.predict_proba(np.atleast_2d(X))[:, 1]
        odds = self.odds_scale * p / np.clip(1 - p, 1e-6, None)
        return odds / (1 + odds)
