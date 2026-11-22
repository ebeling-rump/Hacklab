"""Turn TabPFN's fall warning into an action: recovery moves for the A1.

Recovery moves (a1_slip_dataset.Reaction, blended in over ~0.15 s):
  slow    keep trotting at 30% forward speed
  brace   trot at half speed with feet 0.15 rad wider and the body 0.12 rad lower
  plant   stop stepping: all four feet down, wide and low, no forward/turn
  stop    stop stepping, normal posture

Step 1 - feasibility (--feasibility): each move switched on ALL the time, paired
against the unchanged gait on the same floors/commands/pushes. If no move helps
even when always on, no warning can help.

Step 2 - the real test (default): the move is triggered by TabPFN's fall risk
(base + history features without the 2 s tilt averages, the best fall model)
and held for HOLD s. Control: the same move triggered at RANDOM moments with
the same alarm rate - if TabPFN does no better than random timing, its
prediction is not what helps.

All floors are unseen by the fall model: 13-15 deg slopes, friction 0.1-1.0.

    python a1_recovery_demo.py --feasibility --episodes 40
    python a1_recovery_demo.py --move plant --episodes 40
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import mujoco
import numpy as np

from a1_collision import CollisionRisk
from a1_slip_dataset import NORMAL, A1Sensors, Reaction, build_model, run_episode
from a1_slip_demo import unseen_floor
from a1_slip_eval import DATA, fall_splits

warnings.filterwarnings("ignore")
MOVES = {
    "slow": Reaction(scale=0.3),
    "brace": Reaction(scale=0.5, stance=0.15, crouch=0.12),
    "plant": Reaction(scale=0.0, turn_scale=0.0, stance=0.15, crouch=0.12, lift=0.0),
    "stop": Reaction(scale=0.0, turn_scale=0.0, lift=0.0),
}
HOLD = 1.0
RESULTS = Path(__file__).parent / "results"


class Trigger:
    """Hold `move` for HOLD s whenever `fire(features, history)` says so."""

    def __init__(self, move, fire):
        self.move, self.fire = move, fire
        self.t, self.until, self.alarms, self.samples = 0.0, -1.0, 0, 0

    def __call__(self, f, h, target):
        self.t += 0.1
        self.samples += 1
        if self.fire(f, h):
            if self.t > self.until:
                self.alarms += 1
            self.until = self.t + HOLD
        return self.move if self.t <= self.until else NORMAL


def episode(sim, e, seed, policy=None):
    model, data, sensors = sim
    floor = unseen_floor(np.random.default_rng(seed + e), "fall")
    rng = np.random.default_rng(seed + 10_000 + e)       # same commands + pushes every run
    ep = run_episode(model, data, sensors, rng, *floor, policy=policy)
    return {"fell": ep["fall_t"] is not None, "distance": float(np.linalg.norm(data.qpos[:2])),
            "reacting_s": float(ep["reacting_time"]), "mu": floor[0], "slope": floor[1]}


def summary(name, runs):
    falls = sum(r["fell"] for r in runs)
    dist = np.mean([r["distance"] for r in runs])
    react = np.mean([r["reacting_s"] for r in runs])
    print(f"  {name:28s} falls {falls:2d}/{len(runs)}   mean distance {dist:4.2f} m   "
          f"time in recovery move {react:4.1f} s/episode", flush=True)
    return dict(name=name, falls=falls, episodes=len(runs), distance=float(dist),
                reacting_s=float(react))


def feasibility(sim, n, seed):
    print("always-on moves vs the unchanged gait (paired episodes):")
    out = {"baseline": summary("unchanged gait", [episode(sim, e, seed) for e in range(n)])}
    for name, move in MOVES.items():
        runs = [episode(sim, e, seed, lambda f, h, t, m=move: m) for e in range(n)]
        out[name] = summary(f"{name} (always on)", runs)
    return out


def triggered(sim, n, seed, move_name, threshold):
    d = dict(np.load(DATA))
    keep = [i for i, nm in enumerate(d["history_names"]) if not nm.startswith("grav_")]
    X = np.hstack([d["features"], d["history"][:, keep]])
    tr = fall_splits(d)["train"]
    t = time.time()
    risk = CollisionRisk().fit(X[tr], d["fall_next"][tr].astype(int), n=2000, pos_fraction=0.25)
    print(f"TabPFN fall model ready ({time.time() - t:.0f}s); move '{move_name}', "
          f"threshold {threshold}, hold {HOLD}s", flush=True)
    move = MOVES[move_name]

    base = [episode(sim, e, seed) for e in range(n)]
    out = {"baseline": summary("unchanged gait", base)}
    trig = []
    alarms = samples = 0
    for e in range(n):
        p = Trigger(move, lambda f, h: risk.predict(np.r_[f, h[keep]])[0] > threshold)
        trig.append(episode(sim, e, seed, p))
        alarms, samples = alarms + p.alarms, samples + p.samples
        print(f"    ep {e:2d} mu={trig[-1]['mu']:.2f} slope={trig[-1]['slope']:.1f}: "
              f"unchanged fell={base[e]['fell']!s:5}  TabPFN-triggered fell={trig[-1]['fell']!s:5}  "
              f"alarms={p.alarms}", flush=True)
    out["tabpfn"] = summary(f"{move_name} when TabPFN warns", trig)

    # Random-timing control with the same alarm-onset rate per sample.
    rate = alarms / max(samples, 1)
    rnd = np.random.default_rng(seed + 99)
    ctrl = [episode(sim, e, seed, Trigger(move, lambda f, h: rnd.random() < rate)) for e in range(n)]
    out["random"] = summary(f"{move_name} at random times", ctrl)
    out["always"] = summary(f"{move_name} always on",
                            [episode(sim, e, seed, lambda f, h, t: move) for e in range(n)])
    prevented = sum(b["fell"] and not x["fell"] for b, x in zip(base, trig))
    caused = sum(x["fell"] and not b["fell"] for b, x in zip(base, trig))
    print(f"TabPFN-triggered vs unchanged: prevented {prevented}, new falls {caused}; "
          f"alarm onsets per sample {rate:.3f}")
    out["paired"] = dict(prevented=prevented, caused=caused, alarm_rate=rate)
    out["episodes"] = dict(baseline=base, tabpfn=trig, random=ctrl)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--feasibility", action="store_true")
    ap.add_argument("--move", choices=list(MOVES), default="plant")
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()
    model = build_model()
    sim = (model, mujoco.MjData(model), A1Sensors(model))
    if args.feasibility:
        out, name = feasibility(sim, args.episodes, args.seed), "recovery_feasibility.json"
    else:
        out = triggered(sim, args.episodes, args.seed, args.move, args.threshold)
        name = f"recovery_{args.move}.json"
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {RESULTS / name}")


if __name__ == "__main__":
    main()
