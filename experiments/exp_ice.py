"""Challenge 8 (ice) - few-shot prediction on a floor with visible ice patches of unknown slipperiness.

Per object (C8 task seed): 100 random pushes from random poses on that object's floor (start pose
anywhere, so the object is on grip, on ice or half on ice). 40 held-out test pushes, the rest is
the pool the model learns from. Models get the usual action features; "+ice" models additionally
get the 3 visible-ice features (pushlab/ground.py), "_blind" models do not.

    python experiments/exp_ice.py --objects 16 --workers 3
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

from pushlab.actions import ActionSampler, query_from_actions, subset, to_action  # noqa: E402
from pushlab.ground import ground_extra  # noqa: E402
from pushlab.learners import make_model  # noqa: E402
from pushlab.sim import PushSim, body_delta  # noqa: E402
from pushlab.tasks import CHALLENGES  # noqa: E402
from experiments.exp_prediction import metrics  # noqa: E402

NS = [0, 3, 5, 10, 20, 40, 60]
N_PUSH, N_TEST = 100, 40


def collect(seed):
    t = CHALLENGES["C8"](seed)
    obj = t.obj
    rng = np.random.default_rng(seed)
    sim = PushSim(obj, ice=t.ice, mu_ice=t.mu_ice, seed=seed)
    sampler = ActionSampler(obj.shape, seed=seed)
    Qs, Ys, Yt = [], [], []
    while len(Ys) < N_PUSH:
        if rng.random() < 0.6:   # near the big patch (often half on ice)
            cx, cy, hx, hy = t.ice[0]
            pose = (cx + rng.uniform(-hx, hx), cy + rng.uniform(-hy - 0.08, hy + 0.08), rng.uniform(-np.pi, np.pi))
        else:
            pose = (rng.uniform(-0.45, 0.45), rng.uniform(-0.35, 0.35), rng.uniform(-np.pi, np.pi))
        sim.set_pose(pose)
        obs = sim.observe()
        a = sampler.sample(1, v_range=t.v_range)[0]
        res = sim.push(to_action(a), observed_pose=obs)
        if not res.valid:
            continue
        q = query_from_actions(obj.shape, a[None], obj.shape.descriptor())
        q["extra"] = ground_extra(obj.shape, obs, t.ice, 1)
        Qs.append(q)
        Ys.append(body_delta(res.pose_before, res.pose_after))
        Yt.append(body_delta(res.true_before, res.true_after) if hasattr(res, "true_before") else Ys[-1])
    Q = {k: np.concatenate([q[k] for q in Qs]) for k in Qs[0] if isinstance(Qs[0][k], np.ndarray)}
    return t, Q, np.array(Ys), np.array(Yt)


def ice_source(n_objects=20, per=60):
    """Experience: pushes on other objects on OTHER icy floors (C8 seeds 1000+), cached."""
    import pickle
    path = ROOT / "data" / f"ice_source_{n_objects}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    src = []
    for sd in range(1000, 1000 + n_objects):
        t, Q, Y, _ = collect(sd)
        Q = {k: v[:per] for k, v in Q.items()}
        src.append((t.obj.shape, t.obj.shape.descriptor(), Q, Y[:per]))
        print("source object", sd, flush=True)
    path.parent.mkdir(exist_ok=True)
    pickle.dump(src, open(path, "wb"))
    return src


def eval_object(args):
    seed, models, ns, source = args
    import torch
    torch.set_num_threads(1)
    t, Q, Y, Yt = collect(seed)
    obj = t.obj
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Y))
    test, pool = perm[:N_TEST], perm[N_TEST:]
    Qte = subset(Q, test)
    frac_te = Q["extra"][test, 0]
    rows = []
    for name in models:
        m = make_model(name, source=source)
        for n in ns:
            tr = pool[:n]
            m.reset(obj.shape, obj.shape.descriptor())
            t0 = time.perf_counter()
            m.fit(subset(Q, tr), Y[tr])
            mean, std = m.predict(Qte)
            dt = time.perf_counter() - t0
            r = metrics(mean, std, Yt[test])
            on = frac_te > 0.3
            pos = 1000 * np.linalg.norm(mean[:, :2] - Yt[test][:, :2], axis=1)
            rows.append(dict(model=name, n=n, obj=obj.name, seed=seed, step_s=dt,
                             pos_mm_on_ice=pos[on].mean() if on.any() else np.nan,
                             pos_mm_grip=pos[~on].mean() if (~on).any() else np.nan, **r))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", type=int, default=16)
    ap.add_argument("--models", nargs="+", default=["physics_nominal", "physics_sysid", "gp_hybrid_blind", "gp_hybrid",
                                                    "tabpfn_hybrid_blind", "tabpfn_hybrid", "tabpfn_api_hybrid"])
    ap.add_argument("--ns", nargs="+", type=int, default=NS)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "results" / "pred_ice.csv"))
    ap.add_argument("--source", type=int, default=0, help="number of other icy-floor objects as experience")
    args = ap.parse_args()
    rows = []
    if Path(args.out).exists():  # resume: keep finished objects
        prev = pd.read_csv(args.out)
        done = set(prev.seed.unique())
        rows = prev.to_dict("records")
    else:
        done = set()
    source = ice_source(args.source) if args.source else None
    jobs = [(s, args.models, args.ns, source) for s in range(args.objects) if s not in done]
    t = time.time()
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(eval_object, jobs)):
            rows.extend(res)
            pd.DataFrame(rows).to_csv(args.out, index=False)
            print(f"[{i + 1}/{len(jobs)}] {time.time() - t:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.groupby(["model", "n"])[["pos_mm", "pos_mm_on_ice", "ang_deg", "cov1"]].mean().unstack(0).round(1).to_string())


if __name__ == "__main__":
    main()
