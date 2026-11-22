"""Unseen-terrain test: real rolling hills, every model frozen (nothing retrained).

All models were fitted on the original data (flat floor, slope from tilted
gravity, datasets/slip.npz). Here the robot walks on real 3D hills
(a1_terrain.py; terrain 1 = steep, terrain 2 = moderate) with random friction.

Part A - prediction quality. Record walks with the blind trot, label them as
before (slip within 0.2 s, fall within 1 s), and score the frozen models:
  falls: TabPFN, XGBoost, logistic regression (base + history without tilt),
         and the 2 s tilt average as a no-ML score
  slips: TabPFN and XGBoost (base + all history)
Part B - behaviour. The same walks per terrain for five strategies:
  blind trot, always brace, tilt rule, TabPFN danger mode, balance reflex

    python a1_terrain_test.py      # writes results/terrain_test.json
"""

import json
import time
import warnings
from pathlib import Path

import mujoco
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from a1_balance import TUNED, Balancer
from a1_collision import CollisionRisk
from a1_danger_mode import BRACE, THRESHOLD, TILT_LIMIT, DangerMode, tilt_deg
from a1_slip_dataset import labels, run_episode
from a1_slip_eval import DATA, fall_splits, splits
from a1_terrain import TerrainSensors, build_terrain_model, terrain_episode

warnings.filterwarnings("ignore")
TERRAINS = {1: "steep hills", 2: "moderate hills"}
EPISODES = 30
OUT = Path(__file__).parent / "results" / "terrain_test.json"


def xgb(X, y):
    return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8).fit(X, y)


def main():
    t0 = time.time()
    d = dict(np.load(DATA))
    names = list(d["history_names"])
    keep = [i for i, n in enumerate(names) if not n.startswith("grav_")]
    gz = names.index("grav_z_2s")
    Xf = np.hstack([d["features"], d["history"][:, keep]])
    Xs = np.hstack([d["features"], d["history"]])
    trf, trs = fall_splits(d)["train"], splits(d)["train"]
    yf, ys = d["fall_next"].astype(int), d["slip_next"].astype(int)
    fall_tab = CollisionRisk().fit(Xf[trf], yf[trf], n=2000, pos_fraction=0.25)
    fall_xgb = xgb(Xf[trf], yf[trf])
    fall_log = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xf[trf], yf[trf])
    slip_tab = CollisionRisk().fit(Xs[trs], ys[trs], n=2000, pos_fraction=max(ys[trs].mean(), 0.25))
    slip_xgb = xgb(Xs[trs], ys[trs])
    print(f"frozen models ready ({time.time() - t0:.0f}s)", flush=True)

    out = {"prediction": {}, "behaviour": []}
    for seed, tname in TERRAINS.items():
        model = build_terrain_model(seed)
        data = mujoco.MjData(model)
        sensors = TerrainSensors(model)

        # ---- Part A: record blind walks, score the frozen models -------------------
        F, H, Yf, Ys, falls = [], [], [], [], 0
        for e in range(EPISODES):
            mu, start, rng = terrain_episode(e, base_seed=7000 + 1000 * seed)
            ep = run_episode(model, data, sensors, rng, mu, 0, 0, start=start)
            _, nxt, fall = labels(ep)
            F.append(ep["features"]); H.append(ep["history"]); Ys.append(nxt); Yf.append(fall)
            falls += ep["fall_t"] is not None
        F, H = np.vstack(F), np.vstack(H)
        Yf, Ys = np.concatenate(Yf).astype(int), np.concatenate(Ys).astype(int)
        xf, xs = np.hstack([F, H[:, keep]]), np.hstack([F, H])
        tilt = np.array([tilt_deg(h) for h in H])
        pred = {"rows": int(len(Yf)), "fall_rate": float(Yf.mean()), "slip_rate": float(Ys.mean()),
                "blind_falls": f"{falls}/{EPISODES}"}
        if 0 < Yf.mean() < 1:
            pred["fall_auc"] = {
                "TabPFN": float(roc_auc_score(Yf, fall_tab.predict(xf))),
                "XGBoost": float(roc_auc_score(Yf, fall_xgb.predict_proba(xf)[:, 1])),
                "logistic": float(roc_auc_score(Yf, fall_log.predict_proba(xf)[:, 1])),
                "tilt 2 s (no ML)": float(roc_auc_score(Yf, tilt)),
            }
        pred["slip_auc"] = {
            "TabPFN": float(roc_auc_score(Ys, slip_tab.predict(xs))),
            "XGBoost": float(roc_auc_score(Ys, slip_xgb.predict_proba(xs)[:, 1])),
        }
        out["prediction"][tname] = pred
        print(f"[{tname}] prediction: {json.dumps(pred)}", flush=True)

        # ---- Part B: behaviour of five strategies on the same walks ---------------
        strategies = {
            "blind trot": lambda: (None, None),
            "always brace": lambda: ((lambda f, h, t: BRACE), None),
            "tilt rule": lambda: (DangerMode(lambda f, h: tilt_deg(h), TILT_LIMIT), None),
            "TabPFN danger mode": lambda: (DangerMode(
                lambda f, h: float(fall_tab.predict(np.r_[f, h[keep]])[0]), THRESHOLD), None),
            "balance reflex": lambda: (None, Balancer(sensors, TUNED)),
        }
        for e in range(EPISODES):
            for sname, make in strategies.items():
                policy, bal = make()
                mu, start, rng = terrain_episode(e, base_seed=7000 + 1000 * seed)
                ep = run_episode(model, data, sensors, rng, mu, 0, 0, start=start,
                                 policy=policy, balancer=bal)
                out["behaviour"].append(dict(
                    terrain=tname, episode=e, mu=mu, strategy=sname, fell=ep["fall_t"] is not None,
                    distance=float(np.linalg.norm(data.qpos[:2] - start.xy)),
                    braced_s=float(ep["reacting_time"]), duration=float(data.time)))
            if (e + 1) % 10 == 0:
                print(f"[{tname}] behaviour {e + 1}/{EPISODES} ({time.time() - t0:.0f}s)", flush=True)

    OUT.write_text(json.dumps(out, indent=1))
    print("\nbehaviour (falls out of episodes, share of time braced):")
    for tname in TERRAINS.values():
        for sname in ("blind trot", "always brace", "tilt rule", "TabPFN danger mode", "balance reflex"):
            r = [x for x in out["behaviour"] if x["terrain"] == tname and x["strategy"] == sname]
            print(f"  {tname:15s} {sname:20s} falls {sum(x['fell'] for x in r):2d}/{len(r)}  "
                  f"distance {np.mean([x['distance'] for x in r]):.2f} m  "
                  f"braced {np.mean([x['braced_s'] / x['duration'] for x in r]):.0%}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
