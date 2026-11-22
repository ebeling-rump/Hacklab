"""Follow-up: give TabPFN the same archive of other sites that the pooled XGBoost uses.

TabPFN's context = the local calibration rows + 512 rows sampled from the
20-site archive, with an extra column is_local (1 for this site, 0 for the
archive; test rows are 1), so the model can learn in context how this site
differs from the others. Same test sites, sizes and test cells as benchmark.py.

Output: results/learning_curves_archive.csv
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from benchmark import N_TEST, metrics, pick_test_sites, pooled_data
from models import TabPFN
from terrain_sim import make_dataset, make_site

SIZES = [32, 64, 128, 256, 512]
N_ARCHIVE = 512


def main(out="results/learning_curves_archive.csv"):
    Xp, yp = pooled_data()
    rows = []
    for sid in pick_test_sites(10):
        site = make_site(sid)
        Xte, yte, _, _ = make_dataset(site, N_TEST, "test", np.random.default_rng(sid))
        for n in SIZES:
            rng = np.random.default_rng(100 * n + sid)  # same local draw as benchmark.py, rep 0
            Xtr, ytr, _, _ = make_dataset(site, n, "calib", rng)
            pick = np.random.default_rng(sid + n).choice(len(yp), N_ARCHIVE, replace=False)
            X = np.vstack([np.column_stack([Xp[pick], np.zeros(N_ARCHIVE)]), np.column_stack([Xtr, np.ones(n)])])
            y = np.concatenate([yp[pick], ytr])
            t = time.time()
            p = TabPFN(seed=0).fit(X, y).predict_proba(np.column_stack([Xte, np.ones(len(Xte))]))[:, 1]
            p = np.clip(p, 1e-4, 1 - 1e-4)
            rows.append(dict(site=sid, rep=0, n=n, model="TabPFN v2 + archive", fit_s=time.time() - t, **metrics(yte, p)))
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"site {sid} n {n} done", flush=True)


if __name__ == "__main__":
    main()
