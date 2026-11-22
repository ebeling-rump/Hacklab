"""Offline test: predict foot slip (next 0.2 s) and falls (next 1 s) from the A1's own sensors.

Data: data/slip.npz (a1slip/sim/slip_dataset.py). Split by episode.
Slip:
  train          mu >= 0.2 and slope < 10 deg   (80% of those episodes)
  test: same     the other 20% of those episodes
  test: icier    mu < 0.2,  slope < 10          (friction never seen in training)
  test: steeper  mu >= 0.2, slope >= 10         (slopes never seen in training)
  test: both     mu < 0.2,  slope >= 10
Fall (no falls happen below ~8 deg, so the slip split has none to learn from):
  train          slope < 13 deg, any friction   (80% of those episodes)
  test: same     the other 20%
  test: steeper  slope 13-15 deg                (never seen in training)

Models on the 51 robot-measurable features (IMU, joints, torques, foot forces, command):
  logistic regression, XGBoost (all train rows), TabPFN v3.5-fast (2000 context rows)
Reference rows that use information the robot does NOT have (marked "privileged"):
  slipping now  - ground-truth current slip, as a persistence guess (slip only)
  true floor    - XGBoost on the true friction + slope + command

    python -m a1slip.prediction.slip_eval          # prints tables, writes results/slip_eval.json
"""

import json
import time
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from a1slip.paths import ROOT
from a1slip.models.tabpfn_risk import RiskModel

warnings.filterwarnings("ignore")
DATA = ROOT / "data" / "slip.npz"
OUT = ROOT / "results" / "slip_eval.json"
TABPFN_CONTEXT = 2000


def splits(d):
    lo_mu, steep = d["mu"] < 0.2, d["slope_deg"] >= 10
    normal = ~lo_mu & ~steep
    held = d["episode"] % 5 == 0
    return {"train": normal & ~held, "same floors": normal & held,
            "icier (unseen)": lo_mu & ~steep, "steeper (unseen)": ~lo_mu & steep,
            "icier + steeper": lo_mu & steep}


FALL_SLOPE_SPLIT = 13.0


def fall_splits(d):
    """Falls never happen below ~8 deg (0/256 episodes), so the slip split leaves
    no falls to learn from. Train on all frictions with slope < 13 deg instead and
    test on held-out episodes plus the steepest slopes (13-15 deg, never seen)."""
    steep = d["slope_deg"] >= FALL_SLOPE_SPLIT
    held = d["episode"] % 5 == 0
    return {"train": ~steep & ~held, "same floors": ~steep & held,
            "steeper (unseen)": steep}


def recall_at_fpr(y, s, fpr_max=0.1):
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= fpr_max].max())


def evaluate(task, X, y, d, sp, extra_ref=None):
    tr = sp["train"]
    print(f"\n=== {task}: train {tr.sum()} rows, positive rate {y[tr].mean():.1%} ===")
    models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "xgboost": XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8),
    }
    scorers = {}
    for name, m in models.items():
        m.fit(X[tr], y[tr])
        scorers[name] = lambda idx, m=m: m.predict_proba(X[idx])[:, 1]

    t = time.time()
    pos = y[tr].mean()
    risk = RiskModel().fit(X[tr], y[tr], n=TABPFN_CONTEXT, pos_fraction=max(pos, 0.25))
    print(f"TabPFN fit {time.time() - t:.0f}s")
    scorers["TabPFN"] = lambda idx: risk.predict(X[idx])

    cond = np.c_[d["mu"], d["slope_deg"], X[:, -2:]]
    floor = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05).fit(cond[tr], y[tr])
    scorers["true floor (privileged)"] = lambda idx: floor.predict_proba(cond[idx])[:, 1]
    if extra_ref is not None:
        scorers.update(extra_ref)

    rows = []
    for split, mask in sp.items():
        if split == "train":
            continue
        idx = np.flatnonzero(mask)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        for name, score in scorers.items():
            t = time.time()
            s = score(idx)
            rows.append(dict(task=task, split=split, model=name, n=len(idx),
                             positives=int(yy.sum()), roc_auc=float(roc_auc_score(yy, s)),
                             pr_auc=float(average_precision_score(yy, s)),
                             recall_at_10fpr=recall_at_fpr(yy, s),
                             ms_per_row=1000 * (time.time() - t) / len(idx)))

    print(f"{'test set':18s}{'model':26s}{'rows':>6s}{'pos':>6s}{'ROC-AUC':>9s}{'PR-AUC':>8s}"
          f"{'rec@10%FPR':>11s}")
    for r in rows:
        print(f"{r['split']:18s}{r['model']:26s}{r['n']:6d}{r['positives']:6d}{r['roc_auc']:9.3f}"
              f"{r['pr_auc']:8.3f}{r['recall_at_10fpr']:11.3f}")
    return rows


def main():
    d = dict(np.load(DATA))
    X = d["features"]
    sp = splits(d)
    print({k: f"{v.sum()} rows / {len(np.unique(d['episode'][v]))} episodes" for k, v in sp.items()})

    slip_now = d["slip_now"].astype(float)
    rows = evaluate("slip in next 0.2 s", X, d["slip_next"].astype(int), d, sp,
                    {"slipping now (privileged)": lambda idx: slip_now[idx]})
    fsp = fall_splits(d)
    for k, v in fsp.items():
        eps = np.unique(d["episode"][v])
        print(f"fall split {k}: {v.sum()} rows, {len(eps)} episodes, "
              f"{d['episode_fell'][v][np.unique(d['episode'][v], return_index=True)[1]].sum()} falls")
    rows += evaluate("fall in next 1 s", X, d["fall_next"].astype(int), d, fsp)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
