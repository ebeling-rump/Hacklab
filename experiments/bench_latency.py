"""Uncontended latency of one planning step: fit on n pushes + predict a 96-push candidate library.

    python experiments/bench_latency.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pushlab.actions import query_from_actions, subset  # noqa: E402
from pushlab.learners import make_model  # noqa: E402
from experiments.exp_prediction import load, source_from  # noqa: E402


def main():
    import torch
    torch.set_num_threads(4)
    rec = load("htest")[0]
    source = source_from(load("htrain"))
    obj = rec["obj"]
    Q = query_from_actions(obj.shape, rec["actions"], obj.shape.descriptor())
    Y = rec["Y"]
    rows = []
    for name in ["physics_sysid", "gp_hybrid", "tabpfn_hybrid", "tabpfn_transfer_hybrid"]:
        for n in [5, 20]:
            m = make_model(name, source=source)
            ts = []
            for rep in range(4):
                t = time.perf_counter()
                m.reset(obj.shape, obj.shape.descriptor())
                m.fit(subset(Q, np.arange(n)), Y[:n])
                m.predict(subset(Q, np.arange(96) % len(Y)))
                ts.append(time.perf_counter() - t)
            rows.append(dict(model=name, n_train=n, n_query=96, step_s_median=np.median(ts[1:])))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).round(3).to_csv(ROOT / "results" / "latency.csv", index=False)


if __name__ == "__main__":
    main()
