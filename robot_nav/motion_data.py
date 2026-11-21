"""Phase 2 data: AGV trajectories and observable features for motion prediction.

Trajectories come from warehouse episodes driven by the constant-velocity controller (so a
robot is moving among the AGVs, as it will in Phase 3). Each episode runs 30 s and is
logged at 10 Hz. Sets use disjoint seed ranges, so no trajectory is in two sets, and none
overlaps the navigation evaluation seeds (0-199):

    train      warehouse          seeds 1000-1039   standard fleet
    test       warehouse          seeds 2000-2019   same pattern, new trajectories
    flipped    warehouse_flipped  seeds 3000-3019   unseen: opposite routing at intersections
    fast       warehouse_fast     seeds 4000-4019   unseen: 1.2-1.5 m/s (train: 0.7-1.1)

Features (observable only, from the 10 Hz position history of the AGV and the robot):
    x, y        position
    vx, vy      velocity            (p[k] - p[k-1]) / 0.1 s
    ax, ay      recent acceleration (v[k] - v[k-3]) / 0.3 s
    rx, ry      robot position relative to the AGV
    r_dist      distance to the robot
Targets: the *residual* to constant-velocity extrapolation at +0.5 s and +1.0 s,
    p[k+h] - (p[k] + v[k] * h)
so every model predicts "how much does the AGV deviate from going straight on", and the
final prediction is CV + residual. The hidden AGV state is only kept as an analysis label.

Usage:
    python motion_data.py                 # generates data/motion/<set>.npz
    python motion_data.py --sets test     # just one set
"""
import argparse
import time

import numpy as np

from sim import CONTROL_DT, DATA, Episode

SETS = {
    "train": ("warehouse", range(1000, 1040)),
    "test": ("warehouse", range(2000, 2020)),
    "flipped": ("warehouse_flipped", range(3000, 3020)),
    "fast": ("warehouse_fast", range(4000, 4020)),
}
DURATION = 30.0
HORIZONS = {"05": 5, "10": 10}          # in 10 Hz steps: +0.5 s, +1.0 s
FEATURES = ["x", "y", "vx", "vy", "ax", "ay", "rx", "ry", "r_dist"]
TARGETS = ["res_x_05", "res_y_05", "res_x_10", "res_y_10"]
BEHAVIOURS = ["turn", "station_approach", "station_pause", "yield", "cruise"]   # priority order
MOTION = DATA / "motion"


def run_trajectories(scenario, seed):
    """Run one 30 s episode; return 10 Hz arrays of AGV positions, robot positions and AGV modes."""
    ep = Episode(seed, "constant_velocity", scenario=scenario)
    while ep.data.time < DURATION:       # keep going after the robot reaches its goal
        ep.step()
    a = ep.arrays()
    return a["obs_pos"], a["robot_pos"], a["obs_mode"]


def build_rows(obs, robot, modes):
    """Turn one episode into feature/target rows, one per (time step, AGV)."""
    dt = CONTROL_DT
    n, n_obs = obs.shape[:2]
    vel = np.zeros_like(obs)
    vel[1:] = (obs[1:] - obs[:-1]) / dt
    first, last = 4, n - max(HORIZONS.values())       # need 0.4 s of history and 1 s of future
    rows = []
    for k in range(first, last):
        for i in range(n_obs):
            p, v = obs[k, i], vel[k, i]
            acc = (v - vel[k - 3, i]) / (3 * dt)
            rel = robot[k] - p
            row = {"x": p[0], "y": p[1], "vx": v[0], "vy": v[1], "ax": acc[0], "ay": acc[1],
                   "rx": rel[0], "ry": rel[1], "r_dist": np.linalg.norm(rel)}
            for tag, h in HORIZONS.items():
                future = obs[k + h, i]
                res = future - (p + v * h * dt)
                row[f"res_x_{tag}"], row[f"res_y_{tag}"] = res
                row[f"fut_x_{tag}"], row[f"fut_y_{tag}"] = future
            window = set(modes[k:k + 11, i])            # analysis label: what happens in the next 1 s
            row["behaviour"] = next((b for b in BEHAVIOURS if b in window), "cruise")
            row["agv"] = i
            rows.append(row)
    return rows


def generate(set_name):
    scenario, seeds = SETS[set_name]
    rows = []
    t0 = time.perf_counter()
    for seed in seeds:
        obs, robot, modes = run_trajectories(scenario, seed)
        for r in build_rows(obs, robot, modes):
            r["episode"] = seed
            rows.append(r)
    cols = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    MOTION.mkdir(parents=True, exist_ok=True)
    np.savez(MOTION / f"{set_name}.npz", **cols)
    print(f"{set_name:<8} {scenario:<18} {len(seeds)} episodes -> {len(rows)} rows "
          f"({time.perf_counter() - t0:.0f} s)")


def load(set_name):
    """Return dict of column arrays for a generated set."""
    with np.load(MOTION / f"{set_name}.npz") as f:
        return {k: f[k] for k in f.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    args = parser.parse_args()
    for s in args.sets:
        generate(s)


if __name__ == "__main__":
    main()
