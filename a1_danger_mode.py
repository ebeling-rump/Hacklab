"""Danger mode: brace early on risky floors, walk normally on safe ones.

Bracing all the time cuts falls on steep slopes (a1_recovery_demo.py), but it
also slows the robot on floors where nothing would happen. A danger mode tries
to get both: enter "brace" (wide, low, half speed) when a signal says the floor
is risky and stay there until the signal has been quiet for EXIT_HOLD s.

Floors (paired: every controller sees the same floor, commands and pushes):
  safe       slope 0-8 deg, friction 0.1-1.0  (icy flat floors slip but don't fall)
  dangerous  slope 13-15 deg, friction 0.1-1.0 (never seen by the fall model)

Controllers:
  never brace     the unchanged gait
  always brace    brace from the first second
  tilt rule       brace while the IMU's 2 s average body tilt is > 10 deg (no ML)
  TabPFN          brace while TabPFN fall risk (1 s ahead) is > THRESHOLD
                  (fixed in advance, not tuned on these floors)

    python a1_danger_mode.py --episodes 40     # writes results/danger_mode.json
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import mujoco
import numpy as np

from a1_collision import CollisionRisk
from a1_recovery_demo import MOVES
from a1_slip_dataset import HISTORY_NAMES, NORMAL, A1Sensors, build_model, run_episode
from a1_slip_eval import DATA, fall_splits

warnings.filterwarnings("ignore")
BRACE = MOVES["brace"]
EXIT_HOLD = 3.0      # s quiet before leaving danger mode
TILT_LIMIT = 10.0    # deg, tilt rule
THRESHOLD = 0.05     # TabPFN fall risk (base rate in training: 0.8%)
GRAV_Z = HISTORY_NAMES.index("grav_z_2s")
OUT = Path(__file__).parent / "results" / "danger_mode.json"


def floor(kind, seed):
    rng = np.random.default_rng(seed)
    mu = float(np.exp(rng.uniform(np.log(0.1), 0.0)))
    slope = float(rng.uniform(0, 8) if kind == "safe" else rng.uniform(13, 15))
    return mu, slope, float(rng.uniform(0, 2 * np.pi))


BASE_SEED = {"dangerous": 1000, "safe": 3000}


def episode_setup(kind, e):
    """Floor and command/push RNG of episode e - identical for every controller."""
    return floor(kind, BASE_SEED[kind] + e), np.random.default_rng(BASE_SEED[kind] + 10_000 + e)


class DangerMode:
    """Brace while value(f, h) > threshold, and for EXIT_HOLD s after."""

    def __init__(self, value, threshold):
        self.value_fn, self.threshold = value, threshold
        self.t, self.until, self.value = 0.0, -1.0, 0.0

    def __call__(self, f, h, target):
        self.t += 0.1
        self.value = self.value_fn(f, h)
        if self.value > self.threshold:
            self.until = self.t + EXIT_HOLD
        return BRACE if self.t <= self.until else NORMAL


def tilt_deg(h):
    return float(np.degrees(np.arccos(np.clip(-h[GRAV_Z], -1, 1))))


def fit_fall_model():
    d = dict(np.load(DATA))
    keep = [i for i, nm in enumerate(d["history_names"]) if not nm.startswith("grav_")]
    X = np.hstack([d["features"], d["history"][:, keep]])
    tr = fall_splits(d)["train"]
    t0 = time.time()
    risk = CollisionRisk().fit(X[tr], d["fall_next"][tr].astype(int), n=2000, pos_fraction=0.25)
    print(f"TabPFN fall model ready ({time.time() - t0:.0f}s)", flush=True)
    return risk, keep


def make_controllers(risk, keep):
    """name -> factory returning a fresh policy (None = unchanged gait)."""
    return {
        "never brace": lambda: None,
        "always brace": lambda: (lambda f, h, t: BRACE),
        "tilt rule": lambda: DangerMode(lambda f, h: tilt_deg(h), TILT_LIMIT),
        "TabPFN": lambda: DangerMode(lambda f, h: float(risk.predict(np.r_[f, h[keep]])[0]),
                                     THRESHOLD),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40, help="per floor kind")
    args = ap.parse_args()

    t0 = time.time()
    risk, keep = fit_fall_model()
    controllers = make_controllers(risk, keep)
    model = build_model()
    data = mujoco.MjData(model)
    sensors = A1Sensors(model)
    rows = []
    for kind in ("dangerous", "safe"):
        for e in range(args.episodes):
            for name, make in controllers.items():
                fl, rng = episode_setup(kind, e)
                ep = run_episode(model, data, sensors, rng, *fl, policy=make())
                rows.append(dict(kind=kind, episode=e, mu=fl[0], slope=fl[1], controller=name,
                                 fell=ep["fall_t"] is not None,
                                 distance=float(np.linalg.norm(data.qpos[:2])),
                                 braced_s=float(ep["reacting_time"]),
                                 duration=float(data.time)))
            print(f"{kind} {e + 1}/{args.episodes} done ({time.time() - t0:.0f}s)", flush=True)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1))
    print(f"\n{'controller':14s}{'falls (danger)':>16s}{'falls (safe)':>14s}"
          f"{'dist danger':>13s}{'dist safe':>11s}{'braced on safe':>16s}")
    for name in controllers:
        r = {k: [x for x in rows if x["controller"] == name and x["kind"] == k]
             for k in ("dangerous", "safe")}
        print(f"{name:14s}{sum(x['fell'] for x in r['dangerous']):>12d}/{len(r['dangerous'])}"
              f"{sum(x['fell'] for x in r['safe']):>11d}/{len(r['safe'])}"
              f"{np.mean([x['distance'] for x in r['dangerous']]):>11.2f} m"
              f"{np.mean([x['distance'] for x in r['safe']]):>9.2f} m"
              f"{np.mean([x['braced_s'] / x['duration'] for x in r['safe']]):>15.0%}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
