"""Active experimentation (offline, pool-based): which push should the robot try next?

For each test object, a model sequentially picks its next training push from a pool of 60
candidate pushes, either at random or by maximum predictive uncertainty (uncertainty sampling,
a standard approximation of Bayesian experimental design / information gain). Prediction error
on 40 held-out pushes is tracked after every new observation.

    python experiments/exp_active.py --data htest --models physics_sysid gp_hybrid tabpfn_hybrid
"""
import argparse
import os
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
from experiments.exp_prediction import load, metrics  # noqa: E402

N_POOL, N_TEST, N_STEPS = 60, 40, 12
YS = np.array([0.03, 0.03, 0.3])


def run(args):
    rec, models, strategies, seed = args
    import torch
    torch.set_num_threads(1)
    obj = rec["obj"]
    ok = rec["ok"]
    A, Y, Yt = rec["actions"][ok], rec["Y"][ok], rec["Y_true"][ok]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Y))
    test, pool = perm[:N_TEST], perm[N_TEST:N_TEST + N_POOL]
    desc = obj.shape.descriptor()
    Q = query_from_actions(obj.shape, A, desc)
    Qte = subset(Q, test)
    rows = []
    for name in models:
        for strat in strategies:
            m = make_model(name)
            m.reset(obj.shape, desc)
            chosen = []
            r2 = np.random.default_rng(seed + 7)
            for step in range(N_STEPS + 1):
                if chosen:
                    m.fit(subset(Q, np.array(chosen)), Y[chosen])
                mean, std = m.predict(Qte)
                rows.append(dict(model=name, strategy=strat, n=len(chosen), obj=obj.name, seed=seed,
                                 **metrics(mean, std, Yt[test])))
                if step == N_STEPS:
                    break
                remaining = np.array([i for i in pool if i not in chosen])
                if strat == "random" or len(chosen) == 0:
                    nxt = r2.choice(remaining)
                else:
                    _, s = m.predict(subset(Q, remaining))
                    score = (s / YS).sum(1) + 1e-6 * r2.random(len(remaining))
                    nxt = remaining[int(np.argmax(score))]
                chosen.append(int(nxt))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="htest")
    ap.add_argument("--models", nargs="+", default=["physics_sysid", "gp_hybrid", "tabpfn_hybrid"])
    ap.add_argument("--strategies", nargs="+", default=["random", "uncertainty"])
    ap.add_argument("--max-objects", type=int, default=24)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    recs = load(args.data)[: args.max_objects]
    out = ROOT / "results" / f"active_{args.data}.csv"
    rows, t = [], time.time()
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run, [(r, args.models, args.strategies, 500 + j)
                                                         for j, r in enumerate(recs)])):
            rows.extend(r)
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"[{i + 1}/{len(recs)}] {time.time() - t:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.groupby(["n", "model", "strategy"])["pos_mm"].mean().unstack([1, 2]).round(1).to_string())


if __name__ == "__main__":
    main()
