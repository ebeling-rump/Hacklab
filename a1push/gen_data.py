"""Prior experience for the A1: random whole-body pushes on several different crates (open floor).

    python -m a1push.gen_data --crates 6 --pushes 20
"""
from __future__ import annotations

import argparse
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def crate_spec(seed):
    from pushlab.shapes import rect, l_shape
    rng = np.random.default_rng(900 + seed)
    if seed % 3 == 2:
        shape = l_shape(rng.uniform(0.55, 0.75), rng.uniform(0.35, 0.45), rng.uniform(0.2, 0.25))
    else:
        shape = rect(rng.uniform(0.5, 0.8), rng.uniform(0.25, 0.4))
    minx, miny, maxx, maxy = shape.polygon.bounds
    com = (float(rng.uniform(-0.3, 0.3) * maxx), float(rng.uniform(-0.25, 0.25) * maxy))
    return shape, dict(mass=float(rng.uniform(2.5, 6.5)), mu=float(rng.uniform(0.25, 0.5)), com=com)


def run_crate(args):
    seed, n = args
    import torch
    torch.set_num_threads(1)
    from pushlab.actions import ActionSampler
    from pushlab.sim import body_delta
    from .env import Crate, Scene, build
    from .pusher import Navigator, execute_push
    from .robot import A1
    shape, hid = crate_spec(seed)
    rng = np.random.default_rng(seed)
    sc = Scene(Crate([tuple(b) for b in shape.boxes], **hid), crate_start=(1.0, 0.0, rng.uniform(-np.pi, np.pi)))
    m, d = build(sc)
    robot = A1(m, d)
    nav = Navigator([], (-3.0, 5.0, -4.0, 4.0))
    sampler = ActionSampler(shape, seed=seed)
    acts = sampler.sample(n, d_range=(0.08, 0.4))
    A, Y, falls = [], [], 0
    for a in acts:
        info = execute_push(m, d, robot, shape, nav, a)
        if not info["ok"]:
            if info.get("reason") == "fell":
                falls += 1
                robot.recover()
            continue
        A.append(a)
        Y.append(body_delta(info["crate_before"], info["crate_after"]))
    return dict(seed=seed, shape=shape, hidden=hid, actions=np.array(A), Y=np.array(Y), falls=falls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crates", type=int, default=6)
    ap.add_argument("--pushes", type=int, default=20)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    t = time.time()
    with Pool(args.workers) as pool:
        out = pool.map(run_crate, [(s, args.pushes) for s in range(args.crates)])
    (ROOT / "data").mkdir(exist_ok=True)
    with open(ROOT / "data" / "a1_prior.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"{sum(len(o['Y']) for o in out)} pushes on {len(out)} crates, falls {sum(o['falls'] for o in out)}, "
          f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
