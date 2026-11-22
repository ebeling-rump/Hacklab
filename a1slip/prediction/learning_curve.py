"""Learning curve: how many examples does each model need to predict a foot slip?

Same task as a1slip/prediction/slip_eval.py (slip within 0.2 s from the robot's own sensors),
but every model gets only N training rows, N = 20 ... 2000, drawn from the
normal-floor training episodes. Each N is repeated REPS times with different
random rows. TabPFN uses all N rows as its context (no subsampling, no tuning);
XGBoost and logistic regression use fixed, reasonable settings.

Test sets (fixed, TEST_ROWS each): held-out normal floors, unseen icier floors,
unseen steeper slopes.

    python -m a1slip.prediction.learning_curve      # prints a table, writes results/learning_curve.json
"""

import json
import time
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from a1slip.paths import ROOT
from a1slip.models.tabpfn_risk import RiskModel
from a1slip.prediction.slip_eval import DATA, splits

warnings.filterwarnings("ignore")
SIZES = [20, 50, 100, 200, 500, 1000, 2000]
REPS = 3
TEST_ROWS = 600
TEST_SETS = ["same floors", "icier (unseen)", "steeper (unseen)"]
OUT = ROOT / "results" / "learning_curve.json"


def tabpfn_scorer(X, y):
    m = RiskModel().model          # TabPFN v3.5-fast, CPU, 4 estimators
    m.fit(X, y)
    return lambda Z: m.predict_proba(Z)[:, 1]


def xgb_scorer(X, y):
    m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8).fit(X, y)
    return lambda Z: m.predict_proba(Z)[:, 1]


def logistic_scorer(X, y):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)).fit(X, y)
    return lambda Z: m.predict_proba(Z)[:, 1]


MODELS = {"TabPFN": tabpfn_scorer, "XGBoost": xgb_scorer, "logistic": logistic_scorer}


def main():
    d = dict(np.load(DATA))
    X, y = d["features"], d["slip_next"].astype(int)
    sp = splits(d)
    rng = np.random.default_rng(0)
    train = np.flatnonzero(sp["train"])
    tests = {k: rng.choice(np.flatnonzero(sp[k]), TEST_ROWS, replace=False) for k in TEST_SETS}

    rows = []
    for n in SIZES:
        for rep in range(REPS):
            idx = rng.choice(train, n, replace=False)
            while y[idx].min() == y[idx].max():          # need both classes
                idx = rng.choice(train, n, replace=False)
            for name, make in MODELS.items():
                t = time.time()
                score = make(X[idx], y[idx])
                for test, tidx in tests.items():
                    rows.append(dict(model=name, n=n, rep=rep, test=test,
                                     roc_auc=float(roc_auc_score(y[tidx], score(X[tidx])))))
                rows[-1]["seconds"] = time.time() - t
        summary = "  ".join(
            f"{m} " + "/".join(f"{np.mean([r['roc_auc'] for r in rows if r['model'] == m and r['n'] == n and r['test'] == t]):.3f}"
                               for t in TEST_SETS) for m in MODELS)
        print(f"N={n:5d}: {summary}   (same/icier/steeper)", flush=True)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
