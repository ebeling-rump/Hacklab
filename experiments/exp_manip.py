"""Closed-loop manipulation benchmark across the challenge suite.

Identical tasks (seeded), identical action library size, identical planner for every model.

    python experiments/exp_manip.py --challenges C1 C3 C4 C5 --seeds 12 \
        --methods reactive physics_sysid gp_res tabpfn_res tabpfn_hybrid
"""
import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pushlab.tasks import CHALLENGES  # noqa: E402
from pushlab.runner import run_episode  # noqa: E402
from pushlab.learners import make_model  # noqa: E402

_SOURCE = None


def _init(source_name):
    global _SOURCE
    import torch
    torch.set_num_threads(1)
    if source_name:
        from experiments.exp_prediction import load, source_from
        _SOURCE = source_from(load(source_name))


def job(args):
    ch, seed, method, kw = args
    task = CHALLENGES[ch](seed)
    base = method.replace("_active", "")
    model = None if method == "reactive" else make_model(base, source=_SOURCE)
    pk = dict(kw.get("planner", {}))
    if method.endswith("_active"):
        pk.update(beta0=kw.get("beta", 1.0), n_explore=kw.get("n_explore", 4))
    t = time.time()
    try:
        out, log = run_episode(task, method, model, seed=seed, planner_kw=pk, n_cand=kw.get("n_cand", 96))
    except Exception as e:  # keep the benchmark running; record the failure explicitly
        return dict(challenge=task.challenge, seed=seed, method=method, error=repr(e), success=False), None
    out["wall_s"] = time.time() - t
    if kw.get("save_logs"):
        d = ROOT / "data" / "episodes"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{ch}_{seed}_{method}.pkl", "wb") as f:
            pickle.dump(dict(out=out, log=log), f)
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenges", nargs="+", default=["C1", "C3", "C4", "C5"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=["reactive", "physics_sysid", "gp_res", "tabpfn_res"])
    ap.add_argument("--source", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-cand", type=int, default=96)
    ap.add_argument("--save-logs", action="store_true")
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()
    kw = dict(n_cand=args.n_cand, save_logs=args.save_logs)
    jobs = [(c, s, m, kw) for c in args.challenges for s in range(args.seed0, args.seed0 + args.seeds)
            for m in args.methods]
    # heavy (TabPFN) jobs first for better load balancing
    jobs.sort(key=lambda j: 0 if "tabpfn" in j[2] else 1)
    t = time.time()
    rows = []
    out_path = ROOT / "results" / f"manip_{args.tag}.csv"
    out_path.parent.mkdir(exist_ok=True)
    with Pool(args.workers, initializer=_init, initargs=(args.source,)) as pool:
        for i, (r, _) in enumerate(pool.imap_unordered(job, jobs)):
            rows.append(r)
            if i % 10 == 0 or i == len(jobs) - 1:
                pd.DataFrame(rows).to_csv(out_path, index=False)
                print(f"[{i + 1}/{len(jobs)}] {time.time() - t:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    pd.set_option("display.width", 200)
    g = df.groupby(["challenge", "method"])
    print(g.agg(success=("success", "mean"), pushes=("pushes", "mean"), pos_mm=("pos_err_mm", "median"),
                ang=("ang_err_deg", "median"), walls=("wall_contacts", "mean"),
                model_s=("model_s_mean", "mean")).round(3).to_string())
    print(f"{time.time() - t:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
