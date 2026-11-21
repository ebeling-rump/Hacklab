"""Where does prediction quality matter? Compare constant velocity against the oracle.

Reads data/metrics_<scenario>.csv and the episode logs written by `sim.py --save-logs`.

1. Prediction error of constant velocity, split by what the obstacle does in the next
   second (cruise / turn / slow for a station / stand at a station / yield). These labels
   come from hidden AGV state and are used for analysis only, never as features.
2. Per-seed comparison constant velocity vs oracle: seeds where better prediction would
   change the outcome (collision, near miss, unnecessary stop, or >= 1 s time to goal).
3. What the closest AGV was doing at the critical moment of those seeds.

Usage:
    python analyze.py                  # warehouse
"""
import argparse
import csv
from collections import Counter, defaultdict

import numpy as np

from sim import DATA

BEHAVIOURS = ["turn", "station_approach", "station_pause", "yield", "cruise"]
LABEL = {"turn": "turning at intersection", "station_approach": "slowing for station",
         "station_pause": "standing at station", "yield": "yielding / waiting", "cruise": "cruising straight"}


def load_rows(scenario):
    with (DATA / f"metrics_{scenario}.csv").open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("time_to_goal", "min_clearance", "unnecessary_stops", "stops", "collision_events"):
            r[k] = float(r[k]) if r[k] not in ("", "None") else float("nan")
        r["seed"] = int(r["seed"])
        r["near_miss"] = r["near_miss"] == "True"
        r["collided"] = r["collided"] == "True"
    return rows


def load_log(scenario, predictor, seed):
    return np.load(DATA / "episodes" / f"{scenario}_{predictor}_{seed}.npz", allow_pickle=False)


def upcoming_behaviour(modes, k, h=10):
    """Most informative behaviour of each obstacle in log steps k..k+h (priority order)."""
    window = modes[k:k + h + 1]                         # (h+1, n_obs) strings
    out = []
    for i in range(window.shape[1]):
        present = set(window[:, i])
        out.append(next((b for b in BEHAVIOURS if b in present), "cruise"))
    return out


def error_by_behaviour(scenario, seeds, predictor="constant_velocity", radius=3.0):
    errs = {"pred_05": defaultdict(list), "pred_10": defaultdict(list)}
    for s in seeds:
        log = load_log(scenario, predictor, s)
        obs, robot, modes = log["obs_pos"], log["robot_pos"], log["obs_mode"]
        n = len(obs)
        for key, h in (("pred_05", 5), ("pred_10", 10)):
            for k in range(n - h):
                near = np.linalg.norm(obs[k] - robot[k], axis=1) < radius
                beh = upcoming_behaviour(modes, k, h)
                e = np.linalg.norm(log[key][k] - obs[k + h], axis=1)
                for i in np.flatnonzero(near):
                    errs[key][beh[i]].append(e[i])
    return errs


def critical_behaviour(scenario, predictor, seed):
    """Behaviour of the closest AGV around the moment of smallest clearance."""
    log = load_log(scenario, predictor, seed)
    k, i = np.unravel_index(np.argmin(log["clearance"]), log["clearance"].shape)
    k0 = max(0, k - 10)                                # the second leading up to it
    return upcoming_behaviour(log["obs_mode"], k0, k - k0)[i], float(log["clearance"][k, i])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="warehouse")
    args = parser.parse_args()

    rows = load_rows(args.scenario)
    by = {(r["predictor"], r["seed"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows if r["predictor"] == "constant_velocity"})

    # 1. prediction error by upcoming behaviour
    errs = error_by_behaviour(args.scenario, seeds)
    print("Constant-velocity prediction error for AGVs within 3 m of the robot,")
    print("by what the AGV does during the prediction window:\n")
    print(f"{'upcoming behaviour':<26}{'share':>7}{'err @0.5 s':>12}{'err @1.0 s':>12}{'p90 @1.0 s':>12}")
    total = sum(len(v) for v in errs["pred_10"].values())
    for b in BEHAVIOURS:
        e5, e10 = np.array(errs["pred_05"][b]), np.array(errs["pred_10"][b])
        if len(e10):
            print(f"{LABEL[b]:<26}{len(e10) / total:>7.0%}{e5.mean():>12.3f}{e10.mean():>12.3f}"
                  f"{np.percentile(e10, 90):>12.3f}")

    # 2. per-seed: does better prediction change the outcome?
    print("\nPer seed, constant velocity (CV) vs oracle:")
    sensitive = []
    for s in seeds:
        cv, orc = by[("constant_velocity", s)], by.get(("oracle", s))
        if orc is None:
            continue
        reasons = []
        if cv["collided"] and not orc["collided"]:
            reasons.append("collision")
        if cv["near_miss"] and not orc["near_miss"] and not orc["collided"]:
            reasons.append("near miss")
        if cv["unnecessary_stops"] > orc["unnecessary_stops"]:
            reasons.append("unnecessary stop")
        if cv["time_to_goal"] - orc["time_to_goal"] >= 1.0:
            reasons.append(f"+{cv['time_to_goal'] - orc['time_to_goal']:.1f} s")
        if reasons:
            beh, gap = critical_behaviour(args.scenario, "constant_velocity", s)
            sensitive.append((s, reasons, beh, gap, cv, orc))
    worse = [s for s in seeds if ("oracle", s) in by and
             (by[("oracle", s)]["collided"] and not by[("constant_velocity", s)]["collided"])]
    print(f"  prediction-sensitive seeds: {len(sensitive)} of {len(seeds)}"
          f"  (oracle worse on collisions in {len(worse)})\n")
    print(f"{'seed':>5}  {'what better prediction fixes':<34}{'CV gap':>8}{'oracle gap':>11}"
          f"{'CV time':>9}{'oracle time':>12}  closest AGV was")
    for s, reasons, beh, gap, cv, orc in sensitive:
        print(f"{s:>5}  {', '.join(reasons):<34}{cv['min_clearance']:>8.2f}{orc['min_clearance']:>11.2f}"
              f"{cv['time_to_goal']:>9.1f}{orc['time_to_goal']:>12.1f}  {LABEL[beh]}")

    # 3. which behaviours cause the trouble
    print("\nCritical moments of prediction-sensitive seeds, by closest AGV's behaviour:")
    for beh, n in Counter(x[2] for x in sensitive).most_common():
        print(f"  {LABEL[beh]:<26}{n}")


if __name__ == "__main__":
    main()
