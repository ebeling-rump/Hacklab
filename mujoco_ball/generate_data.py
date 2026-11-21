"""Generate a small dataset of ball runs for learning the ball's dynamics.

Each row is one independent simulation: the ball starts at the origin with a
random horizontal velocity and random friction, and we record where it is
after a fixed time. Runs are then split into train/test (one row = one run,
so no run appears in both sets).

Usage:
    python generate_data.py                       # 200 runs -> data/ball_runs.csv
    python generate_data.py --runs 500 --seconds 2 --out data/other.csv
"""
import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np

from simulate import load

# Sampling ranges for the randomized inputs.
V_RANGE = (-2.0, 2.0)          # vx, vy in m/s
MU_SLIDE_RANGE = (0.1, 1.0)    # sliding friction coefficient (dimensionless)
MU_ROLL_RANGE = (0.0005, 0.02) # rolling friction (MuJoCo: length, in m)

COLUMNS = ["run_id", "split", "vx0", "vy0", "mu_slide", "mu_roll", "x_final", "y_final"]


def run_once(vx, vy, mu_slide, mu_roll, seconds, model=None, data=None):
    """Simulate one run from the origin and return the ball's (x, y) after `seconds`."""
    if model is None:
        model, data = load()
    geom = model.geom("ball_geom").id
    model.geom_friction[geom] = [mu_slide, model.geom_friction[geom][1], mu_roll]

    mujoco.mj_resetDataKeyframe(model, data, model.key("push").id)
    data.qvel[:] = [vx, vy, 0, 0, 0, 0]  # no initial spin: the ball starts by sliding
    mujoco.mj_forward(model, data)

    n_steps = round(seconds / model.opt.timestep)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
    return float(data.qpos[0]), float(data.qpos[1])


def generate(n_runs, n_test, seconds, seed):
    rng = np.random.default_rng(seed)
    model, data = load()

    vx = rng.uniform(*V_RANGE, n_runs)
    vy = rng.uniform(*V_RANGE, n_runs)
    mu_slide = rng.uniform(*MU_SLIDE_RANGE, n_runs)
    # Log-uniform so both low and high rolling friction are well represented.
    mu_roll = np.exp(rng.uniform(*np.log(MU_ROLL_RANGE), n_runs))

    test_ids = set(rng.choice(n_runs, size=n_test, replace=False).tolist())

    rows = []
    for i in range(n_runs):
        x, y = run_once(vx[i], vy[i], mu_slide[i], mu_roll[i], seconds, model, data)
        split = "test" if i in test_ids else "train"
        rows.append([i, split, vx[i], vy[i], mu_slide[i], mu_roll[i], x, y])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--test", type=int, default=50, help="number of runs held out for testing")
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data" / "ball_runs.csv")
    args = parser.parse_args()

    rows = generate(args.runs, args.test, args.seconds, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for r in rows:
            writer.writerow([r[0], r[1], *(f"{v:.6f}" for v in r[2:])])
    n_train = sum(r[1] == "train" for r in rows)
    print(f"Saved {len(rows)} runs ({n_train} train, {len(rows) - n_train} test) to {args.out}")


if __name__ == "__main__":
    main()
