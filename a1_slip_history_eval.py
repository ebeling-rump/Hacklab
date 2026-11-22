"""Do short-history features (incl. the legs-vs-IMU slip detector) help on new floors?

Same splits and labels as a1_slip_eval.py. Every model is run on two feature sets:
  base       the 51 instantaneous sensor values
  +history   base + 11 history features the robot computes itself
             (a1_slip_dataset.History: legs-vs-IMU slip gap over 0.1/0.5/2 s,
             foot disagreement, variability of joints/forces/gyro, mean tilt)
and, for slip only, a privileged reference:
  +true past slip   base + the TRUE fraction of the last 2 s with a slip
                    (what a perfect slip detector would give)

Models: TabPFN (2000-row context), XGBoost on the same 2000 rows, XGBoost on all rows.

    python a1_slip_history_eval.py   # prints tables, writes results/slip_history_eval.json
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from a1_collision import CollisionRisk, context_subsample
from a1_slip_eval import DATA, fall_splits, splits

warnings.filterwarnings("ignore")
CONTEXT = 2000
OUT = Path(__file__).parent / "results" / "slip_history_eval.json"


def xgb(X, y):
    return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8).fit(X, y)


def run(task, y, sets, sp):
    tr = np.flatnonzero(sp["train"])
    pos = max(y[tr].mean(), 0.25)
    rows = []
    print(f"\n=== {task} ===", flush=True)
    for fs_name, X in sets.items():
        Xc, yc = context_subsample(X[tr], y[tr], n=CONTEXT, pos_fraction=pos, seed=0)
        models = {"XGBoost (all rows)": xgb(X[tr], y[tr]).predict_proba,
                  "XGBoost (2000 rows)": xgb(Xc, yc).predict_proba}
        t = time.time()
        risk = CollisionRisk().fit(X[tr], y[tr], n=CONTEXT, pos_fraction=pos)
        models["TabPFN (2000 rows)"] = lambda Z, r=risk: np.c_[1 - r.predict(Z), r.predict(Z)]
        print(f"  [{fs_name}] TabPFN fit {time.time() - t:.0f}s", flush=True)
        for split, mask in sp.items():
            if split == "train":
                continue
            idx = np.flatnonzero(mask)
            if y[idx].min() == y[idx].max():
                continue
            for name, predict in models.items():
                s = predict(X[idx])[:, 1]
                rows.append(dict(task=task, features=fs_name, model=name, split=split,
                                 roc_auc=float(roc_auc_score(y[idx], s)),
                                 pr_auc=float(average_precision_score(y[idx], s))))
                print(f"  {fs_name:16s}{split:18s}{name:22s} ROC-AUC {rows[-1]['roc_auc']:.3f}  "
                      f"PR-AUC {rows[-1]['pr_auc']:.3f}", flush=True)
    return rows


def main():
    d = dict(np.load(DATA))
    base, hist = d["features"], d["history"]
    slip_sets = {"base": base, "+history": np.hstack([base, hist]),
                 "+true past slip": np.hstack([base, d["true_slip_2s"][:, None]])}
    rows = run("slip in next 0.2 s", d["slip_next"].astype(int), slip_sets, splits(d))
    rows += run("fall in next 1 s", d["fall_next"].astype(int),
                {"base": base, "+history": np.hstack([base, hist])}, fall_splits(d))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1))

    print("\nROC-AUC summary (base -> +history):")
    for task in ("slip in next 0.2 s", "fall in next 1 s"):
        for split in dict.fromkeys(r["split"] for r in rows if r["task"] == task):
            for model in dict.fromkeys(r["model"] for r in rows):
                v = {r["features"]: r["roc_auc"] for r in rows
                     if r["task"] == task and r["split"] == split and r["model"] == model}
                if "base" in v:
                    extra = f"   (true past slip {v['+true past slip']:.3f})" if "+true past slip" in v else ""
                    print(f"  {task[:4]:5s}{split:18s}{model:22s} {v['base']:.3f} -> "
                          f"{v['+history']:.3f}{extra}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
