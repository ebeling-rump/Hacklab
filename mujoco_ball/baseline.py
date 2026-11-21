"""Constant-velocity baseline: assume the ball keeps its initial velocity.

    position(t) = v0 * t        (the ball starts at the origin)

It ignores friction entirely, so it needs no training data.

Usage:
    python baseline.py
"""
import argparse

import numpy as np

from ball_data import load_split, position_errors, print_errors


def predict_constant_velocity(X, seconds=1.0):
    """X columns: vx0, vy0, ... -> predicted (x, y) after `seconds`."""
    return np.asarray(X)[:, :2] * seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=1.0, help="must match generate_data.py")
    args = parser.parse_args()

    _, _, X_test, Y_test = load_split()
    err = position_errors(Y_test, predict_constant_velocity(X_test, args.seconds))
    print(f"Test runs: {len(X_test)}")
    print_errors("constant velocity", err)


if __name__ == "__main__":
    main()
