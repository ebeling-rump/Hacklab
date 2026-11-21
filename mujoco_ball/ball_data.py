"""Shared helpers for the ball-dynamics experiments: load the dataset and score predictions."""
import csv
from pathlib import Path

import numpy as np

DATA_PATH = Path(__file__).parent / "data" / "ball_runs.csv"
FEATURES = ["vx0", "vy0", "mu_slide", "mu_roll"]
TARGETS = ["x_final", "y_final"]


def load_split(path=DATA_PATH):
    """Return (X_train, Y_train, X_test, Y_test) as numpy arrays with FEATURES / TARGETS columns."""
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))

    def arrays(split):
        sub = [r for r in rows if r["split"] == split]
        X = np.array([[float(r[c]) for c in FEATURES] for r in sub])
        Y = np.array([[float(r[c]) for c in TARGETS] for r in sub])
        return X, Y

    return (*arrays("train"), *arrays("test"))


def position_errors(Y_true, Y_pred):
    """Per-run errors and summary metrics for predicted (x, y) positions."""
    diff = np.asarray(Y_pred) - np.asarray(Y_true)
    dist = np.hypot(diff[:, 0], diff[:, 1])
    return {
        "mean_dist": float(dist.mean()),         # mean Euclidean position error (m)
        "mae_x": float(np.abs(diff[:, 0]).mean()),
        "mae_y": float(np.abs(diff[:, 1]).mean()),
        "max_dist": float(dist.max()),
        "dist": dist,
    }


def print_errors(name, err):
    print(f"{name:>18}: mean position error {err['mean_dist']:.4f} m "
          f"(MAE x {err['mae_x']:.4f}, MAE y {err['mae_y']:.4f}, worst {err['max_dist']:.4f})")
