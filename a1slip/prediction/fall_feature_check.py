"""Why did history features make TabPFN's fall warning worse on unseen steep slopes?

Hypothesis: the 2 s average tilt features (grav_x/y/z_2s) take values on 13-15 deg
slopes that never occur in training (< 13 deg), and TabPFN handles inputs outside
its context worse than trees do.

Fall task, same split as a1slip.prediction.slip_eval.fall_splits. Feature sets:
  base                 51 instantaneous sensor values
  +history             base + all 11 history features
  +history, no tilt    base + history without grav_*_2s (8 features)
  +tilt only           base + grav_*_2s (3 features)
TabPFN is run with two different 2000-row contexts (seeds 0 and 1) to show how
much of a difference is just which rows were picked. XGBoost uses all rows.

    python -m a1slip.prediction.fall_feature_check   # writes results/fall_feature_check.json
"""

import json
import warnings

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from a1slip.paths import ROOT
from a1slip.models.tabpfn_risk import CollisionRisk
from a1slip.prediction.slip_eval import DATA, fall_splits

warnings.filterwarnings("ignore")
OUT = ROOT / "results" / "fall_feature_check.json"


def main():
    d = dict(np.load(DATA))
    names = list(d["history_names"])
    tilt = [names.index(n) for n in ("grav_x_2s", "grav_y_2s", "grav_z_2s")]
    rest = [i for i in range(len(names)) if i not in tilt]
    base, hist = d["features"], d["history"]
    sets = {"base": base,
            "+history": np.hstack([base, hist]),
            "+history, no tilt": np.hstack([base, hist[:, rest]]),
            "+tilt only": np.hstack([base, hist[:, tilt]])}
    y = d["fall_next"].astype(int)
    sp = fall_splits(d)
    tr = np.flatnonzero(sp["train"])
    tests = {k: np.flatnonzero(v) for k, v in sp.items() if k != "train"}

    # How far outside the training range are the tilt averages on unseen slopes?
    lo, hi = hist[tr][:, tilt].min(0), hist[tr][:, tilt].max(0)
    for k, idx in tests.items():
        out = ((hist[idx][:, tilt] < lo) | (hist[idx][:, tilt] > hi)).any(1).mean()
        print(f"{k:18s}: rows with a 2 s tilt average outside the training range: {out:.0%}")

    rows = []
    for fs, X in sets.items():
        models = {"XGBoost (all rows)": XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8).fit(X[tr], y[tr]).predict_proba}
        for seed in (0, 1):
            r = CollisionRisk(seed=seed).fit(X[tr], y[tr], n=2000, pos_fraction=0.25, seed=seed)
            models[f"TabPFN seed {seed}"] = lambda Z, r=r: np.c_[1 - r.predict(Z), r.predict(Z)]
        for name, predict in models.items():
            for k, idx in tests.items():
                s = predict(X[idx])[:, 1]
                rows.append(dict(features=fs, model=name, split=k,
                                 roc_auc=float(roc_auc_score(y[idx], s)),
                                 pr_auc=float(average_precision_score(y[idx], s))))
                print(f"{fs:20s}{name:20s}{k:18s} ROC-AUC {rows[-1]['roc_auc']:.3f}  "
                      f"PR-AUC {rows[-1]['pr_auc']:.3f}", flush=True)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1))
    print("\nsteeper (unseen) ROC-AUC:")
    for fs in sets:
        v = {r["model"]: r["roc_auc"] for r in rows
             if r["features"] == fs and r["split"] == "steeper (unseen)"}
        print(f"  {fs:20s} TabPFN {v['TabPFN seed 0']:.3f} / {v['TabPFN seed 1']:.3f}   "
              f"XGBoost {v['XGBoost (all rows)']:.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
