"""Online demo: TabPFN fall/slip warning makes the A1 slow down on floors it never saw.

Every 0.1 s TabPFN reads the robot's own sensors (IMU, joints, torques, foot
forces, command) and predicts P(fall within 1 s) - or P(slip within 0.2 s)
with --risk slip. When the risk is above --threshold the robot drops to
SLOW_SCALE of the commanded forward speed for HOLD seconds.

Each test episode runs twice with the same seed - same floor, same commands,
same pushes - once with the warning and once without, so the fall counts are a
paired comparison. Floors are drawn only from ranges excluded from that model's
training: 13-15 deg slopes for the fall model, mu < 0.2 and/or slope >= 10 deg
for the slip model.

    python a1_slip_demo.py --episodes 40              # fall warning
    python a1_slip_demo.py --episodes 40 --risk slip  # slip warning
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import mujoco
import numpy as np

from a1_collision import CollisionRisk
from a1_slip_dataset import NORMAL, A1Sensors, Reaction, SLOPE_MAX, build_model, run_episode
from a1_slip_eval import DATA, fall_splits, splits

warnings.filterwarnings("ignore")
SLOW_SCALE = 0.3
HOLD = 1.0          # s the robot stays slow after a warning


def unseen_floor(rng, risk="slip"):
    """Friction and/or slope outside the training ranges of that risk model."""
    if risk == "fall":   # fall model trained on slope < 13 deg, any friction
        return (float(np.exp(rng.uniform(np.log(0.1), 0.0))), float(rng.uniform(13, SLOPE_MAX)),
                float(rng.uniform(0, 2 * np.pi)))
    kind = rng.integers(3)
    mu = np.exp(rng.uniform(np.log(0.1), np.log(0.2))) if kind != 1 else rng.uniform(0.2, 1.0)
    slope = rng.uniform(10, SLOPE_MAX) if kind != 0 else rng.uniform(0, 10)
    return float(mu), float(slope), float(rng.uniform(0, 2 * np.pi))


class Warning:
    def __init__(self, risk, threshold):
        self.risk, self.threshold = risk, threshold
        self.slow_until, self.t, self.alarms = -1.0, 0.0, 0

    def __call__(self, features, history, target):
        self.t += 0.1
        if self.risk.predict(features)[0] > self.threshold:
            if self.t > self.slow_until:
                self.alarms += 1
            self.slow_until = self.t + HOLD
        return Reaction(scale=SLOW_SCALE) if self.t <= self.slow_until else NORMAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--risk", choices=["fall", "slip"], default="fall")
    ap.add_argument("--threshold", type=float, default=None,
                    help="warning level (default: 0.2 for fall, 0.8 for slip)")
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()
    threshold = args.threshold or (0.2 if args.risk == "fall" else 0.8)

    d = dict(np.load(DATA))
    tr = (fall_splits(d) if args.risk == "fall" else splits(d))["train"]
    y = d["fall_next" if args.risk == "fall" else "slip_next"].astype(int)
    t = time.time()
    risk = CollisionRisk().fit(d["features"][tr], y[tr], n=2000,
                               pos_fraction=max(y[tr].mean(), 0.25))
    print(f"TabPFN {args.risk} model ready ({time.time() - t:.0f}s), threshold {threshold}")

    model = build_model()
    data = mujoco.MjData(model)
    sensors = A1Sensors(model)
    results = []
    for e in range(args.episodes):
        floor = unseen_floor(np.random.default_rng(args.seed + e), args.risk)
        res = {"episode": e, "mu": floor[0], "slope": floor[1]}
        for mode in ("baseline", "warning"):
            rng = np.random.default_rng(args.seed + 10_000 + e)   # same commands + pushes
            policy = Warning(risk, threshold) if mode == "warning" else None
            ep = run_episode(model, data, sensors, rng, *floor, policy=policy)
            res[mode] = {"fell": ep["fall_t"] is not None,
                         "alarms": policy.alarms if policy else 0,
                         "distance": float(np.linalg.norm(data.qpos[:2]))}
        results.append(res)
        print(f"ep {e:2d} mu={floor[0]:.2f} slope={floor[1]:4.1f}: "
              f"baseline fell={res['baseline']['fell']!s:5} "
              f"warning fell={res['warning']['fell']!s:5} alarms={res['warning']['alarms']}",
              flush=True)

    b = sum(r["baseline"]["fell"] for r in results)
    w = sum(r["warning"]["fell"] for r in results)
    saved = sum(r["baseline"]["fell"] and not r["warning"]["fell"] for r in results)
    caused = sum(r["warning"]["fell"] and not r["baseline"]["fell"] for r in results)
    dist = [np.mean([r[m]["distance"] for r in results]) for m in ("baseline", "warning")]
    print(f"\nfalls: baseline {b}/{len(results)}, with {args.risk} warning {w}/{len(results)} "
          f"(prevented {saved}, new {caused}); mean distance walked "
          f"{dist[0]:.2f} m vs {dist[1]:.2f} m")
    out = Path(__file__).parent / "results" / f"slip_demo_{args.risk}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
