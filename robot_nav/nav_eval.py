"""Phase 3: navigation with learned obstacle prediction vs. without.

Predictors (all use the same planner and the same local safety filter):
    constant_velocity   no learning
    gp                  Gaussian Process (N = 1000), local, every control step, real latency
    tabpfn_ideal        TabPFN (cloud, N = 1000) every 0.2 s; the simulation waits for it
                        -> value of the predictions if latency were not an issue
    tabpfn_live         TabPFN with its measured latency in simulation time: a prediction
                        is usable only latency seconds after its observation; constant
                        velocity in between -> what the cloud model gives in real time
    oracle              true future (evaluation-only upper bound)

Inference latency (wall-clock per model call) is reported separately from simulation time.
Results are appended to data/nav_eval.csv one episode at a time; re-running skips episodes
already there (delete the file to start over).

Usage:
    python nav_eval.py --seeds 0 20 --predictors constant_velocity gp oracle
    python nav_eval.py --seeds 0 20 --predictors tabpfn_ideal tabpfn_live     # needs TABPFN_TOKEN
    python nav_eval.py --report
"""
import argparse
import csv
import time

import numpy as np

from learned import AsyncPredictor, GPModel, TabPFNModel
from sim import CONTROL_DT, DATA, Episode

OUT = DATA / "nav_eval.csv"
LOCAL = ["constant_velocity", "gp", "oracle"]
CLOUD = ["tabpfn_ideal", "tabpfn_live"]
_models = {}


def model(name):
    if name not in _models:
        t0 = time.perf_counter()
        _models[name] = GPModel(1000) if name == "gp" else TabPFNModel(1000, combined=True)
        print(f"  {name} model ready in {time.perf_counter() - t0:.1f} s")
    return _models[name]


def make_predictor(name):
    if name == "gp":
        return AsyncPredictor(model("gp"), CONTROL_DT, ideal=False, min_period=0.0, max_age=1.0)
    if name == "tabpfn_ideal":
        return AsyncPredictor(model("tabpfn"), CONTROL_DT, ideal=True, min_period=0.2, max_age=1.0)
    if name == "tabpfn_live":
        return AsyncPredictor(model("tabpfn"), CONTROL_DT, ideal=False, min_period=0.0, max_age=1.0)
    return None


def run(name, seed):
    pred = make_predictor(name)
    base = name if pred is None else "constant_velocity"
    ep = Episode(seed, base, predictor_obj=pred)
    ep.predictor_name = name
    m = ep.run()
    m["sim_wall_s"] = ep.wall_seconds - ep.predict_seconds    # simulation + planning, without inference
    m["predict_wall_s"] = ep.predict_seconds
    m.update(pred.stats() if pred is not None else
             {"model_calls": 0, "latency_mean_ms": float("nan"), "latency_p95_ms": float("nan"),
              "prediction_age_ms": 0.0, "cv_fallback_steps": 0})
    m["control_steps"] = ep.n_control
    return m


def done_keys():
    if not OUT.exists():
        return set()
    with OUT.open() as f:
        return {(r["predictor"], int(r["seed"])) for r in csv.DictReader(f)}


def append(row):
    """Append one episode; if the columns changed, rewrite the file with the union of columns."""
    rows, fields = [], list(row)
    if OUT.exists():
        with OUT.open() as f:
            reader = csv.DictReader(f)
            old_fields = reader.fieldnames or []
            if old_fields != fields:
                rows = list(reader)
                fields = old_fields + [k for k in fields if k not in old_fields]
    if rows or not OUT.exists():
        with OUT.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, restval="")
            w.writeheader()
            w.writerows(rows)
    with OUT.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=fields, restval="").writerow(row)


def report(seeds=None, only=None):
    with OUT.open() as f:
        rows = list(csv.DictReader(f))
    names = [n for n in dict.fromkeys(r["predictor"] for r in rows) if only is None or n in only]
    common = set.intersection(*[{int(r["seed"]) for r in rows if r["predictor"] == n} for n in names])
    if seeds is not None:
        common &= set(seeds)
    print(f"Paired comparison on {len(common)} seeds that every predictor completed\n")
    f = lambda r, k: float(r[k]) if r[k] not in ("", "None", "nan") else np.nan  # noqa: E731
    hdr = (f"{'predictor':<20}{'collide':>8}{'near':>6}{'reached':>8}{'time s':>8}{'min gap':>8}{'stops':>6}"
           f"{'err@.5':>8}{'err@1':>7}{'calls/ep':>9}{'lat ms':>8}{'p95 ms':>8}{'age ms':>8}{'CV fb':>7}"
           f"{'sim s/ep':>9}{'inf s/ep':>9}")
    print(hdr)
    summary = {}
    for n in names:
        sub = [r for r in rows if r["predictor"] == n and int(r["seed"]) in common]
        g = lambda k: np.nanmean([f(r, k) for r in sub])  # noqa: E731
        reached = [f(r, "time_to_goal") for r in sub if r["reached_goal"] == "True"]
        fb = np.nanmean([f(r, "cv_fallback_steps") / f(r, "control_steps") for r in sub])
        summary[n] = {r["seed"]: f(r, "time_to_goal") for r in sub}
        print(f"{n:<20}{np.mean([r['collided'] == 'True' for r in sub]):>8.0%}"
              f"{np.mean([r['near_miss'] == 'True' for r in sub]):>6.0%}{np.mean([r['reached_goal'] == 'True' for r in sub]):>8.0%}"
              f"{np.mean(reached):>8.2f}{g('min_clearance'):>8.3f}{g('stops'):>6.2f}"
              f"{g('pred_err_05'):>8.3f}{g('pred_err_10'):>7.3f}{g('model_calls'):>9.1f}"
              f"{g('latency_mean_ms'):>8.0f}{g('latency_p95_ms'):>8.0f}{g('prediction_age_ms'):>8.0f}{fb:>7.0%}"
              f"{g('sim_wall_s'):>9.2f}{g('predict_wall_s'):>9.2f}")

    if "constant_velocity" in summary:
        print("\nTime to goal vs constant velocity (paired, same seeds):")
        base = summary["constant_velocity"]
        for n in names:
            if n == "constant_velocity":
                continue
            d = np.array([base[s] - summary[n][s] for s in base if s in summary[n]])
            d = d[~np.isnan(d)]
            print(f"  {n:<18} faster by mean {d.mean():+.2f} s, median {np.median(d):+.2f} s; "
                  f">=1 s faster in {(d >= 1).sum()}, >=1 s slower in {(d <= -1).sum()} of {len(d)} seeds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs=2, type=int, default=[0, 20], metavar=("FIRST", "END"))
    parser.add_argument("--predictors", nargs="+", default=LOCAL + CLOUD, choices=LOCAL + CLOUD)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--only", nargs="+", help="report only these predictors")
    args = parser.parse_args()
    if not args.report:
        done = done_keys()
        for name in args.predictors:
            for seed in range(*args.seeds):
                if (name, seed) in done:
                    continue
                t0 = time.perf_counter()
                m = run(name, seed)
                append(m)
                print(f"{name:<18} seed {seed:>3}: reached={m['reached_goal']} t={m['time_to_goal']} "
                      f"gap={m['min_clearance']:.2f} calls={m['model_calls']} ({time.perf_counter() - t0:.0f} s wall)",
                      flush=True)
    report(only=args.only)


if __name__ == "__main__":
    main()
