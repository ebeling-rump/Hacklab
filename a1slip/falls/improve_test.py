"""Can TabPFN's danger mode be improved, and does it help a robot that balances itself?

TabPFN's examples always come from the original flat-floor data (data/slip.npz);
nothing is fitted on hills. Thresholds follow one rule fixed in advance: 5x the
label's base rate in the training examples (that rule gave the original 5%).

  part 2  improved warning on the unseen hills (a1_terrain, same 60 walks as
          a1slip/prediction/terrain_test.py), blind gait:
            early       label "fall within 2.5 s" instead of 1 s, hold brace 5 s
            early+plant same, and plant all four feet while risk > 3x threshold
  part 3  stress test with the balance reflex always on: hills up to 35 deg and
          pushes up to 120 N (calibrated on other walks: reflex alone fell 5/20),
          30 new walks:
            reflex | + always brace | + tilt rule | + TabPFN (1 s) | + TabPFN early
            | + TabPFN early+plant

    python -m a1slip.falls.improve_test part2     # -> results/improve_warning.json
    python -m a1slip.falls.improve_test part3     # -> results/stress_test.json
"""

import json
import sys
import time
import warnings

import mujoco
import numpy as np

from a1slip.paths import ROOT
from a1slip.sim import slip_dataset as D
from a1slip.sim import terrain as T
from a1slip.sim.balance import TUNED, Balancer
from a1slip.models.tabpfn_risk import CollisionRisk
from a1slip.falls.danger_mode import BRACE, THRESHOLD, TILT_LIMIT, DangerMode, tilt_deg
from a1slip.falls.recovery_demo import MOVES
from a1slip.prediction.slip_eval import DATA, fall_splits

warnings.filterwarnings("ignore")
RESULTS = ROOT / "results"
PLANT = MOVES["plant"]
EARLY_H = 2.5      # s, early-warning label horizon
EARLY_HOLD = 5.0   # s
PLANT_HOLD = 1.5   # s


def fit_models():
    d = dict(np.load(DATA))
    keep = [i for i, n in enumerate(d["history_names"]) if not n.startswith("grav_")]
    X = np.hstack([d["features"], d["history"][:, keep]])
    tr = fall_splits(d)["train"]
    # "Fall within 2.5 s": a fall ends its episode, so the last sample is <= 0.1 s before it.
    ep, t = d["episode"], d["t"]
    t_end = {e: t[ep == e].max() for e in np.unique(ep)}
    y_early = (d["episode_fell"] & (np.array([t_end[e] for e in ep]) - t <= EARLY_H)).astype(int)
    y_1s = d["fall_next"].astype(int)
    rate = float(y_early[tr].mean())
    thr = round(5 * rate, 3)
    t0 = time.time()
    m1 = CollisionRisk().fit(X[tr], y_1s[tr], n=2000, pos_fraction=0.25)
    me = CollisionRisk().fit(X[tr], y_early[tr], n=2000, pos_fraction=0.25)
    print(f"models ready ({time.time() - t0:.0f}s): 1 s model threshold {THRESHOLD:.0%} "
          f"(base rate {y_1s[tr].mean():.2%}); 2.5 s model threshold {thr:.1%} "
          f"(base rate {rate:.2%}), plant above {3 * thr:.1%}", flush=True)
    return m1, me, keep, thr


class Warning:
    """Brace while risk > thr (and for `hold` s after); plant while risk > plant_thr."""

    def __init__(self, risk_fn, thr, hold, plant_thr=None):
        self.risk_fn, self.thr, self.hold, self.plant_thr = risk_fn, thr, hold, plant_thr
        self.t, self.until, self.plant_until, self.value = 0.0, -1.0, -1.0, 0.0

    def __call__(self, f, h, target):
        self.t += 0.1
        self.value = p = self.risk_fn(f, h)
        if p > self.thr:
            self.until = self.t + self.hold
        if self.plant_thr is not None and p > self.plant_thr:
            self.plant_until = self.t + PLANT_HOLD
        if self.t <= self.plant_until:
            return PLANT
        return BRACE if self.t <= self.until else D.NORMAL


def run_set(model, sensors, episodes, base_seed, strategies, label):
    data = mujoco.MjData(model)
    rows, t0 = [], time.time()
    for e in range(episodes):
        for name, make in strategies.items():
            policy, bal = make()
            mu, start, rng = T.terrain_episode(e, base_seed=base_seed)
            ep = D.run_episode(model, data, sensors, rng, mu, 0, 0, start=start, policy=policy,
                               balancer=bal)
            rows.append(dict(set=label, episode=e, mu=mu, strategy=name, fell=ep["fall_t"] is not None,
                             distance=float(np.linalg.norm(data.qpos[:2] - start.xy)),
                             braced_s=float(ep["reacting_time"]), duration=float(data.time)))
        if (e + 1) % 10 == 0:
            print(f"[{label}] {e + 1}/{episodes} ({time.time() - t0:.0f}s)", flush=True)
    return rows


def summary(rows, title):
    print(f"\n{title}")
    for name in dict.fromkeys(r["strategy"] for r in rows):
        r = [x for x in rows if x["strategy"] == name]
        print(f"  {name:32s} falls {sum(x['fell'] for x in r):2d}/{len(r)}  distance "
              f"{np.mean([x['distance'] for x in r]):.2f} m  braced "
              f"{np.mean([x['braced_s'] / x['duration'] for x in r]):.0%}", flush=True)


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "part2"
    m1, me, keep, thr = fit_models()
    r1 = lambda f, h: float(m1.predict(np.r_[f, h[keep]])[0])
    re_ = lambda f, h: float(me.predict(np.r_[f, h[keep]])[0])

    if part == "part2":
        rows = []
        for seed, tname in ((1, "steep hills"), (2, "moderate hills")):
            model = T.build_terrain_model(seed)
            sensors = T.TerrainSensors(model)
            rows += run_set(model, sensors, 30, 7000 + 1000 * seed, {
                "TabPFN early (2.5 s), hold 5 s": lambda: (Warning(re_, thr, EARLY_HOLD), None),
                "TabPFN early + plant": lambda: (Warning(re_, thr, EARLY_HOLD, 3 * thr), None),
            }, tname)
        old = json.loads((RESULTS / "terrain_test.json").read_text())["behaviour"]
        for r in old:
            r["set"] = r.pop("terrain")
        rows = [r for r in old] + rows
        (RESULTS / "improve_warning.json").write_text(json.dumps(dict(threshold=thr, rows=rows), indent=1))
        summary(rows, "Unseen hills, blind gait, 60 walks (first five rows from a1slip/prediction/terrain_test.py):")
    else:
        T.MAX_SLOPE, D.PUSH_FORCE = 35.0, 120.0
        model = T.build_terrain_model(1)
        sensors = T.TerrainSensors(model)
        bal = lambda: Balancer(sensors, TUNED)
        rows = run_set(model, sensors, 30, 11_000, {
            "reflex": lambda: (None, bal()),
            "reflex + always brace": lambda: ((lambda f, h, t: BRACE), bal()),
            "reflex + tilt rule": lambda: (DangerMode(lambda f, h: tilt_deg(h), TILT_LIMIT), bal()),
            "reflex + TabPFN (1 s)": lambda: (Warning(r1, THRESHOLD, 3.0), bal()),
            "reflex + TabPFN early": lambda: (Warning(re_, thr, EARLY_HOLD), bal()),
            "reflex + TabPFN early + plant": lambda: (Warning(re_, thr, EARLY_HOLD, 3 * thr), bal()),
        }, "stress")
        (RESULTS / "stress_test.json").write_text(json.dumps(dict(
            threshold=thr, condition="hills up to 35 deg, pushes up to 120 N", rows=rows), indent=1))
        summary(rows, "Stress test (hills up to 35 deg, pushes up to 120 N), balance reflex on, 30 walks:")


if __name__ == "__main__":
    main()
