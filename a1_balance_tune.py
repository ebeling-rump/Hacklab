"""Tune the balance controller's gains on TRAINING floors only (slope 9-13 deg).

The test slopes of the danger-mode experiment (13-15 deg) are never used here.
Each reflex is tried alone with both signs, then the helpful ones together at
two strengths. Score: falls out of N paired episodes (same floors, commands and
pushes for every configuration), then mean distance.

    python a1_balance_tune.py        # writes results/balance_tune.json
"""

import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import mujoco
import numpy as np

from a1_balance import Balancer, Gains
from a1_slip_dataset import A1Sensors, build_model, run_episode

N = 40
SEED = 50_000
OUT = Path(__file__).parent / "results" / "balance_tune.json"
STEP = {"lean_pitch": 0.8, "lean_roll": 0.8, "level_pitch": 0.5, "level_roll": 0.5,
        "damp": 0.03, "vel_x": 0.3, "vel_y": 0.3}


def train_floor(e):
    rng = np.random.default_rng(SEED + e)
    return (float(np.exp(rng.uniform(np.log(0.1), 0.0))), float(rng.uniform(9, 13)),
            float(rng.uniform(0, 2 * np.pi)))


def score(sim, gains):
    model, data, sensors = sim
    bal = None if gains is None else Balancer(sensors, gains)
    falls, dist = 0, []
    for e in range(N):
        ep = run_episode(model, data, sensors, np.random.default_rng(SEED + 10_000 + e),
                         *train_floor(e), balancer=bal)
        falls += ep["fall_t"] is not None
        dist.append(float(np.linalg.norm(data.qpos[:2])))
    return falls, float(np.mean(dist))


def main():
    model = build_model()
    sim = (model, mujoco.MjData(model), A1Sensors(model))
    rows = []
    t0 = time.time()

    def run(name, gains):
        f, d = score(sim, gains)
        rows.append(dict(name=name, gains=None if gains is None else asdict(gains), falls=f, distance=d))
        print(f"{name:34s} falls {f:2d}/{N}  distance {d:.2f} m  ({time.time() - t0:.0f}s)", flush=True)
        return f

    base = run("blind gait", None)
    helpful = {}
    for k, step in STEP.items():
        best = None
        for sign in (+1, -1):
            g = replace(Gains(), **{k: sign * step})
            f = run(f"{k} = {sign * step:+.2f}", g)
            if f < base and (best is None or f < best[0]):
                best = (f, sign * step)
        if best:
            helpful[k] = best[1]
    print("helpful alone:", helpful)
    for scale in (0.5, 1.0):
        if helpful:
            run(f"combined x{scale}", Gains(**{k: v * scale for k, v in helpful.items()}))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(dict(helpful=helpful, rows=rows), indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
