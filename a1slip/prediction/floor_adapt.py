""""The A1 learns a new floor in 10 seconds": few-shot adaptation of the slip model.

For each of several floors the slip model never saw (mu < 0.2 and/or slope >= 10):
  experience  the robot walks ~10 s on the new floor -> the first N_NEW = 100
              samples (one per 0.1 s), each labelled afterwards "did a foot slip
              in the next 0.2 s?"
  test        5 further episodes on the same floor (~350 samples)

and we compare
  general       model trained on normal floors only (no experience of this floor)
  general+10s   the same plus the 100 new rows
                  TabPFN: the 100 rows are simply added to its context (no training)
                  XGBoost: retrained on all rows with the new rows weighted x10
  300+10s       rebalanced: only 300 general rows + the 100 new ones, so the new
                floor is 25% of the data instead of 5%
  10s only      only the 100 new rows

    python -m a1slip.prediction.floor_adapt --features history   # + the robot's short-history features

Metrics on the test samples: ROC-AUC (ranking), log loss (are the probabilities
right?), and mean predicted risk vs the real slip rate on that floor.

    python -m a1slip.prediction.floor_adapt        # prints a table, writes results/floor_adapt.json
"""

import argparse
import json
import time
import warnings

import mujoco
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from xgboost import XGBClassifier

from a1slip.paths import ROOT
from a1slip.models.tabpfn_risk import CollisionRisk
from a1slip.sim.slip_dataset import A1Sensors, build_model, labels, run_episode
from a1slip.prediction.slip_eval import DATA, splits

warnings.filterwarnings("ignore")
FLOORS = [(0.12, 0.0), (0.15, 6.0), (0.12, 12.0), (0.30, 12.0), (0.60, 14.0), (0.18, 14.0)]
N_NEW = 100            # = 10 s of walking at one sample per 0.1 s
TEST_EPISODES = 5
GENERAL_CONTEXT = 2000
NEW_WEIGHT = 10.0      # XGBoost: up-weight the new floor's rows
SMALL_CONTEXT = 300    # rebalanced: 300 general + 100 new rows (new floor = 25% of context)
RESULTS = ROOT / "results"
USE_HISTORY = False    # set by --features


def collect(model, data, sensors, mu, slope, seeds):
    feats, ys = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        ep = run_episode(model, data, sensors, rng, mu, slope, rng.uniform(0, 2 * np.pi))
        feats.append(np.hstack([ep["features"], ep["history"]]) if USE_HISTORY else ep["features"])
        ys.append(labels(ep)[1])
    return np.vstack(feats), np.concatenate(ys).astype(int)


def tabpfn(X, y):
    m = CollisionRisk().model
    t = time.time()
    m.fit(X, y)
    return m, time.time() - t


def xgb(X, y, w=None):
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8).fit(X, y, sample_weight=w)


def main():
    global USE_HISTORY
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["base", "history"], default="base",
                    help="history = base + the robot's own short-history features")
    args = ap.parse_args()
    USE_HISTORY = args.features == "history"
    out = RESULTS / ("floor_adapt.json" if not USE_HISTORY else "floor_adapt_history.json")

    d = dict(np.load(DATA))
    tr = np.flatnonzero(splits(d)["train"])
    Xg = np.hstack([d["features"], d["history"]])[tr] if USE_HISTORY else d["features"][tr]
    yg = d["slip_next"][tr].astype(int)
    rng = np.random.default_rng(0)
    ctx = rng.choice(len(yg), GENERAL_CONTEXT, replace=False)
    small = ctx[:SMALL_CONTEXT]
    tab_general, t_gen = tabpfn(Xg[ctx], yg[ctx])
    xgb_general = xgb(Xg, yg)
    print(f"general models ready (TabPFN context {GENERAL_CONTEXT} rows, fit {t_gen:.0f}s; "
          f"XGBoost on {len(yg)} rows)", flush=True)

    model = build_model()
    data = mujoco.MjData(model)
    sensors = A1Sensors(model)
    rows = []
    for f, (mu, slope) in enumerate(FLOORS):
        Xn, yn = collect(model, data, sensors, mu, slope, [5000 + 10 * f, 5001 + 10 * f])
        Xn, yn = Xn[:N_NEW], yn[:N_NEW]
        Xt, yt = collect(model, data, sensors, mu, slope,
                         [6000 + 10 * f + k for k in range(TEST_EPISODES)])

        tab_adapt, t_adapt = tabpfn(np.vstack([Xg[ctx], Xn]), np.r_[yg[ctx], yn])
        tab_small, t_small = tabpfn(np.vstack([Xg[small], Xn]), np.r_[yg[small], yn])
        w = np.r_[np.ones(len(yg)), np.full(len(yn), NEW_WEIGHT)]
        cand = {("TabPFN", "general"): tab_general.predict_proba,
                ("TabPFN", "general+10s"): tab_adapt.predict_proba,
                ("TabPFN", "300+10s"): tab_small.predict_proba,
                ("XGBoost", "general"): xgb_general.predict_proba,
                ("XGBoost", "general+10s"): xgb(np.vstack([Xg, Xn]), np.r_[yg, yn], w).predict_proba,
                ("XGBoost", "300+10s"): xgb(np.vstack([Xg[small], Xn]),
                                            np.r_[yg[small], yn]).predict_proba}
        if 0 < yn.mean() < 1:
            tab_new, t_new = tabpfn(Xn, yn)
            cand[("TabPFN", "10s only")] = tab_new.predict_proba
            cand[("XGBoost", "10s only")] = xgb(Xn, yn).predict_proba
        for (name, cond), predict in cand.items():
            p = np.clip(predict(Xt)[:, 1], 1e-4, 1 - 1e-4)
            rows.append(dict(mu=mu, slope=slope, model=name, condition=cond,
                             roc_auc=float(roc_auc_score(yt, p)) if 0 < yt.mean() < 1 else None,
                             log_loss=float(log_loss(yt, p, labels=[0, 1])),
                             predicted_rate=float(p.mean()), actual_rate=float(yt.mean()),
                             new_rows_slip_rate=float(yn.mean()), test_rows=len(yt)))
        print(f"floor mu={mu} slope={slope}: actual slip rate {yt.mean():.2f} "
              f"(10 s sample: {yn.mean():.2f}); TabPFN refit: 2000+100 rows {t_adapt:.0f}s, "
              f"300+100 rows {t_small:.0f}s", flush=True)
        for r in rows[-len(cand):]:
            auc = f"{r['roc_auc']:.3f}" if r["roc_auc"] is not None else "  n/a"
            print(f"   {r['model']:8s}{r['condition']:13s} AUC {auc}  logloss {r['log_loss']:.3f}  "
                  f"predicted risk {r['predicted_rate']:.2f}", flush=True)

    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nmean over floors ({args.features} features):")
    for name in ("TabPFN", "XGBoost"):
        for cond in ("general", "general+10s", "300+10s", "10s only"):
            rs = [r for r in rows if r["model"] == name and r["condition"] == cond]
            if rs:
                aucs = [r["roc_auc"] for r in rs if r["roc_auc"] is not None]
                err = np.mean([abs(r["predicted_rate"] - r["actual_rate"]) for r in rs])
                print(f"  {name:8s}{cond:13s} AUC {np.mean(aucs):.3f}  "
                      f"logloss {np.mean([r['log_loss'] for r in rs]):.3f}  "
                      f"|predicted-actual risk| {err:.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
