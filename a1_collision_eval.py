"""Offline evaluation: can TabPFN predict a collision in the next 0.8 s?

Data: datasets/collision_env_*.npz from a1_collision_dataset.py.
Train on env_A (6 boxes), test on
  * held-out env_A samples    (same layout, new start poses)
  * all of env_B              (12 boxes in different places: unseen layout)

Compared against:
  * nearest-obstacle rule     (risk = -closest lidar range within +-30 deg of
                               the commanded turn direction; no training)
  * logistic regression, XGBoost (trained on ALL env_A train rows; TabPFN only
                               sees CONTEXT_ROWS of them)

    python a1_collision_eval.py
"""

import time
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from a1_collision import N_SECTORS, CollisionRisk, load_table

warnings.filterwarnings("ignore")
HOLDOUT_FRACTION = 0.2  # last 20% of env_A samples


def rule_score(X):
    """Higher = riskier: closest obstacle in the direction the command steers."""
    deg = np.degrees(np.linspace(-np.pi, np.pi, N_SECTORS, endpoint=False)) + 7.5
    heading = np.degrees(X[:, N_SECTORS + 1]) * 0.6   # turn command ~ heading change
    near = np.abs(deg[None] - heading[:, None]) <= 30
    return -np.where(near, X[:, :N_SECTORS], np.inf).min(1)


def action_spread(score, X, n=200, seed=0):
    """Mean (max - min) risk over the filter's 21 candidate commands per scan.

    A model that ignores the command (it only learned "a box is close") scores
    ~0 here even with a high ROC-AUC, and is useless for choosing commands.
    """
    from a1_safety_filter import CANDIDATES
    rows = X[np.random.default_rng(seed).choice(len(X), n, replace=False)]
    batch = np.repeat(rows, len(CANDIDATES), 0)
    batch[:, N_SECTORS:N_SECTORS + 2] = np.tile(CANDIDATES, (n, 1))
    s = np.asarray(score(batch)).reshape(n, len(CANDIDATES))
    return float((s.max(1) - s.min(1)).mean())


def recall_at_fpr(y, s, fpr_max=0.1):
    fpr, tpr, _ = roc_curve(y, s)
    return tpr[fpr <= fpr_max].max()


def main():
    XA, yA = load_table("env_A")
    XB, yB = load_table("env_B")
    tr = np.arange(len(yA)) < (1 - HOLDOUT_FRACTION) * len(yA)
    sets = {"env_A held-out": (XA[~tr], yA[~tr]), "env_B (unseen)": (XB, yB)}
    print(f"train rows {tr.sum()} (collision rate {yA[tr].mean():.1%}); "
          + ", ".join(f"{k}: {len(y)} rows, {y.mean():.1%}" for k, (_, y) in sets.items()))

    models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05),
    }
    scorers = {"nearest-obstacle rule": rule_score}
    for name, m in models.items():
        m.fit(XA[tr], yA[tr])
        scorers[name] = lambda X, m=m: m.predict_proba(X)[:, 1]

    t = time.time()
    risk = CollisionRisk().fit(XA[tr], yA[tr])
    print(f"TabPFN fit on {risk.model.n_features_in_} features: {time.time() - t:.0f}s")
    scorers["TabPFN v3.5-fast"] = risk.predict

    rows = []
    for name, score in scorers.items():
        for set_name, (X, y) in sets.items():
            t = time.time()
            s = score(X)
            ms = 1000 * (time.time() - t) / len(y)
            spread = action_spread(score, X) if name != "nearest-obstacle rule" else np.nan
            rows.append((name, set_name, roc_auc_score(y, s), average_precision_score(y, s),
                         recall_at_fpr(y, s), spread, ms))

    print(f"\n{'model':24s}{'test set':17s}{'ROC-AUC':>8s}{'PR-AUC':>8s}"
          f"{'recall@10%FPR':>15s}{'cmd spread':>12s}{'ms/row':>8s}")
    for r in rows:
        print(f"{r[0]:24s}{r[1]:17s}{r[2]:8.3f}{r[3]:8.3f}{r[4]:15.3f}{r[5]:12.3f}{r[6]:8.2f}")
    print("\nPR-AUC of a random guess = the collision rate of the test set.\n"
          "cmd spread = mean (max - min) predicted risk over the 21 candidate commands for\n"
          "the same scan; near 0 means the model ignores the command.")


if __name__ == "__main__":
    main()
