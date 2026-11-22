"""Learning-curve benchmark: how good is each model after n self-labelled cells?

For each test site the robot drives a calibration loop in the left half of the
map and labels n cells by measured slip. Models are evaluated on 1,000 cells
from the right half of the same site. Two cross-site baselines use 20,000 rows
pooled from 20 other sites: one zero-shot, one combined with the local rows.

Output: results/learning_curves.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

from models import GeometricRule, ece, fit_predict, local_models
from terrain_sim import make_dataset, make_site

POOL_SEEDS = range(20)
SIZES = [32, 64, 128, 256, 512, 1024]
N_TEST = 1000


def pick_test_sites(k, start=1000):
    """Sites with something to learn: 10 to 55 percent high-slip cells in the test half."""
    out, s = [], start
    while len(out) < k:
        site = make_site(s)
        _, y, _, _ = make_dataset(site, N_TEST, "test", np.random.default_rng(s))
        if 0.10 <= y.mean() <= 0.55:
            out.append(s)
        s += 1
    return out


def pooled_data():
    Xs, ys = [], []
    for s in POOL_SEEDS:
        X, y, _, _ = make_dataset(make_site(s), 1000, "calib", np.random.default_rng(s))
        Xs.append(X)
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def pooled_xgb():
    return XGBClassifier(n_jobs=2, verbosity=0, max_depth=4, n_estimators=300, learning_rate=0.05, random_state=0)


def metrics(y, p):
    return dict(auc=roc_auc_score(y, p), logloss=log_loss(y, p), brier=brier_score_loss(y, p), ece=ece(y, p))


def main(n_sites=10, reps=1, out="results/learning_curves.csv"):
    Path(out).parent.mkdir(exist_ok=True)
    Xp, yp = pooled_data()
    zero_shot = pooled_xgb().fit(Xp, yp)
    sites = pick_test_sites(n_sites)
    print("test sites:", sites, flush=True)
    rows = pd.read_csv(out).to_dict("records") if Path(out).exists() else []
    done = {(r["site"], r["rep"], r["n"]) for r in rows if r["model"] == "Cross-site + local XGBoost"}
    have_refs = {r["site"] for r in rows if r["model"] == "Geometric rule"}
    for sid in sites:
        site = make_site(sid)
        Xte, yte, _, _ = make_dataset(site, N_TEST, "test", np.random.default_rng(sid))
        if sid not in have_refs:
            p = GeometricRule().predict_proba(Xte)[:, 1]
            rows.append(dict(site=sid, rep=0, n=0, model="Geometric rule", fit_s=0, **metrics(yte, p)))
            p = np.clip(zero_shot.predict_proba(Xte)[:, 1], 1e-4, 1 - 1e-4)
            rows.append(dict(site=sid, rep=0, n=0, model="Cross-site XGBoost", fit_s=0, **metrics(yte, p)))
        for rep in range(reps):
            for n in SIZES:
                if (sid, rep, n) in done:
                    continue
                rng = np.random.default_rng(10_000 * rep + 100 * n + sid)
                Xtr, ytr, _, _ = make_dataset(site, n, "calib", rng)
                for name, make in local_models(seed=rep).items():
                    t = time.time()
                    p = fit_predict(make, Xtr, ytr, Xte)
                    rows.append(dict(site=sid, rep=rep, n=n, model=name, fit_s=time.time() - t, **metrics(yte, p)))
                # Cross-site + local: pooled rows plus local rows, local weight balances the pool.
                t = time.time()
                w = np.concatenate([np.ones(len(yp)), np.full(n, len(yp) / n)])
                m = pooled_xgb().fit(np.vstack([Xp, Xtr]), np.concatenate([yp, ytr]), sample_weight=w)
                p = np.clip(m.predict_proba(Xte)[:, 1], 1e-4, 1 - 1e-4)
                rows.append(dict(site=sid, rep=rep, n=n, model="Cross-site + local XGBoost",
                                 fit_s=time.time() - t, **metrics(yte, p)))
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"site {sid} rep {rep} n {n} done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=10)
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()
    main(a.sites, a.reps)
