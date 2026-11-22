"""3D few-shot prediction for whole-body A1 pushes (leave-one-crate-out).

For each crate: models see n of its pushes (n = 0..12), predict the rest. Transfer models also
see the pushes of the OTHER crates (prior experience, different sizes/masses/frictions).

    python -m a1push.eval_prior
"""
from __future__ import annotations

import os
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
NS = [0, 1, 2, 3, 5, 8, 12]
MODELS = ["physics_nominal", "physics_sysid", "gp_hybrid", "tabpfn_hybrid", "gp_transfer_hybrid",
          "tabpfn_transfer_hybrid"]


def job(args):
    data, i, rep = args
    import torch
    torch.set_num_threads(1)
    from pushlab.actions import query_from_actions, subset
    from pushlab.learners import make_model
    from experiments.exp_prediction import metrics
    tgt = data[i]
    source = [(o["shape"], o["shape"].descriptor(), query_from_actions(o["shape"], o["actions"]), o["Y"])
              for j, o in enumerate(data) if j != i]
    shape = tgt["shape"]
    desc = shape.descriptor()
    Q = query_from_actions(shape, tgt["actions"], desc)
    Y = tgt["Y"]
    rng = np.random.default_rng(rep * 31 + i)
    perm = rng.permutation(len(Y))
    test, pool = perm[:6], perm[6:]
    rows = []
    for name in MODELS:
        m = make_model(name, source=source)
        for n in NS:
            if n > len(pool):
                continue
            m.reset(shape, desc)
            m.fit(subset(Q, pool[:n]), Y[pool[:n]])
            mean, std = m.predict(subset(Q, test))
            rows.append(dict(model=name, n=n, obj=f"crate{i}", seed=rep, **metrics(mean, std, Y[test])))
    return rows


def main():
    with open(ROOT / "data" / "a1_prior.pkl", "rb") as f:
        data = pickle.load(f)
    jobs = [(data, i, rep) for rep in range(4) for i in range(len(data))]
    with Pool(4) as pool:
        rows = [r for res in pool.imap_unordered(job, jobs) for r in res]
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "a1_prediction.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    print(df.groupby(["model", "n"])[["pos_mm", "ang_deg", "nll"]].mean().unstack(0).round(1).to_string())


if __name__ == "__main__":
    main()
