"""Fit TabPFN (via the cloud API, tabpfn-client) to predict the ball's final (x, y).

TabPFN regressors predict one target, so we fit one model for x_final and one for y_final.
Both models see the same 4 inputs: vx0, vy0, mu_slide, mu_roll.

Authentication (once): run `python -c "import tabpfn_client; tabpfn_client.interactive_login()"`
or set the TABPFN_TOKEN environment variable. This script never prints the token.

Usage:
    python tabpfn_model.py      # fits, evaluates, writes data/predictions.csv
"""
import csv
from pathlib import Path

import numpy as np

try:
    # Verify HTTPS against the OS certificate store instead of Python's bundled one.
    # Needed when antivirus or a corporate proxy re-signs TLS traffic (e.g. Avast Web Shield).
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from tabpfn_client import TabPFNRegressor

from ball_data import TARGETS, load_split, position_errors, print_errors
from baseline import predict_constant_velocity

PRED_PATH = Path(__file__).parent / "data" / "predictions.csv"


def fit_predict_tabpfn(X_train, Y_train, X_test):
    """Fit one TabPFN regressor per target column and return (n_test, n_targets) predictions."""
    preds = []
    for j, name in enumerate(TARGETS):
        model = TabPFNRegressor(random_state=0)
        model.fit(X_train, Y_train[:, j])
        preds.append(model.predict(X_test))
        print(f"  fitted + predicted {name}")
    return np.column_stack(preds)


def save_predictions(Y_test, predictions, path=PRED_PATH):
    """Write actual and predicted positions for each model, one row per test run."""
    names = list(predictions)
    with Path(path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_true", "y_true"] + [f"{n}_{a}" for n in names for a in ("x", "y")])
        for i in range(len(Y_test)):
            row = [*Y_test[i]] + [v for n in names for v in predictions[n][i]]
            writer.writerow([f"{v:.6f}" for v in row])


def main():
    X_train, Y_train, X_test, Y_test = load_split()
    print(f"Train runs: {len(X_train)}, test runs: {len(X_test)}")

    print("TabPFN (cloud):")
    pred_tabpfn = fit_predict_tabpfn(X_train, Y_train, X_test)
    pred_baseline = predict_constant_velocity(X_test)

    print_errors("constant velocity", position_errors(Y_test, pred_baseline))
    print_errors("TabPFN", position_errors(Y_test, pred_tabpfn))

    save_predictions(Y_test, {"baseline": pred_baseline, "tabpfn": pred_tabpfn})
    print(f"Saved predictions to {PRED_PATH}")


if __name__ == "__main__":
    main()
