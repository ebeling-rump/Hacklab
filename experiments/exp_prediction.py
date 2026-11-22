"""Challenge 2 (offline): few-shot prediction learning curves on unknown objects.

For every test object: models see the first n of its pushes (random actions) and predict the
outcome of 40 held-out pushes. Metrics per (model, n): position error, orientation error,
Gaussian NLL, 1-sigma coverage, fit/predict latency.

    python experiments/exp_prediction.py --data test --models physics_nominal physics_sysid gp gp_res tabpfn tabpfn_res
"""
import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pushlab.actions import query_from_actions, subset  # noqa: E402
from pushlab.learners import make_model  # noqa: E402

NS = [0, 1, 2, 3, 5, 10, 20, 40]
N_TEST = 40


def metrics(mean, std, Y):
    pos = np.linalg.norm(mean[:, :2] - Y[:, :2], axis=1)
    ang = np.abs(mean[:, 2] - Y[:, 2])
    z = (mean - Y) / np.maximum(std, 1e-6)
    nll = 0.5 * (z ** 2 + np.log(2 * np.pi * np.maximum(std, 1e-6) ** 2)).sum(1)
    cov = (np.abs(z) < 1).mean()
    return dict(pos_mm=1000 * pos.mean(), ang_deg=np.degrees(ang).mean(), nll=nll.mean(), cov1=cov)


def eval_object(args):
    rec, models, ns, source, seed = args
    import torch
    torch.set_num_threads(1)
    obj = rec["obj"]
    ok = rec["ok"]
    A, Y = rec["actions"][ok], rec["Y"][ok]
    Yt = rec["Y_true"][ok]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Y))
    test, pool_idx = perm[:N_TEST], perm[N_TEST:]
    desc = obj.shape.descriptor()
    Qall = query_from_actions(obj.shape, A, desc)
    Qte = subset(Qall, test)
    rows = []
    for name in models:
        m = make_model(name, source=[s for s in (source or []) if s[0] is not obj.shape])
        for n in ns:
            if n > len(pool_idx):
                continue
            tr = pool_idx[:n]
            m.reset(obj.shape, desc)
            t0 = time.perf_counter()
            m.fit(subset(Qall, tr), Y[tr])
            t1 = time.perf_counter()
            mean, std = m.predict(Qte)
            t2 = time.perf_counter()
            r = metrics(mean, std, Yt[test])
            rows.append(dict(model=name, n=n, obj=obj.name, family=obj.name.split("_")[0], seed=seed,
                             fit_s=t1 - t0, pred_s=t2 - t1, **r))
    return rows


def load(name):
    with open(ROOT / "data" / f"pushes_{name}.pkl", "rb") as f:
        return pickle.load(f)


def source_from(recs, n_per=60):
    out = []
    for r in recs:
        ok = r["ok"]
        A, Y = r["actions"][ok][:n_per], r["Y"][ok][:n_per]
        desc = r["obj"].shape.descriptor()
        out.append((r["obj"].shape, desc, query_from_actions(r["obj"].shape, A), Y))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="test")
    ap.add_argument("--source", default=None, help="dataset used as cross-object experience")
    ap.add_argument("--models", nargs="+", default=["physics_nominal", "physics_sysid", "gp", "gp_res",
                                                    "tabpfn", "tabpfn_res"])
    ap.add_argument("--ns", nargs="+", type=int, default=NS)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    recs = load(args.data)
    source = source_from(load(args.source)) if args.source else None
    jobs = [(r, args.models, args.ns, source, rep * 100 + i) for rep in range(args.reps) for i, r in enumerate(recs)]
    t = time.time()
    with Pool(args.workers) as pool:
        rows = [x for res in pool.imap_unordered(eval_object, jobs) for x in res]
    df = pd.DataFrame(rows)
    out = Path(args.out or ROOT / "results" / f"prediction_{args.data}.csv")
    out.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(df.groupby(["model", "n"])[["pos_mm", "ang_deg", "nll", "cov1", "pred_s"]].mean().unstack(0).round(2).to_string())
    print(f"{time.time() - t:.0f}s -> {out}")


if __name__ == "__main__":
    main()
