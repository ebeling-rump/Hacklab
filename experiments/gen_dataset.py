"""Generate push datasets: for each object, K random pushes from rest (MuJoCo).

Outputs data/pushes_<name>.pkl : list of dicts with object (incl. hidden params, for evaluation
only), actions (K,3), Y (K,3) observed body-frame deltas, Y_true, wall/valid flags.

    python experiments/gen_dataset.py --name train --seed 1 --n-objects 40 --families train
    python experiments/gen_dataset.py --name test  --seed 2 --n-objects 24 --families all
"""
import argparse
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pushlab.objects import object_set, TRAIN_FAMILIES, HELDOUT_FAMILIES  # noqa: E402
from pushlab.shapes import FAMILIES  # noqa: E402
from pushlab.sim import PushSim, body_delta  # noqa: E402
from pushlab.actions import ActionSampler, to_action  # noqa: E402

DATA = ROOT / "data"


def run_object(args):
    obj, k, seed, v_range = args
    sim = PushSim(obj, seed=seed)
    sampler = ActionSampler(obj.shape, seed=seed)
    acts = sampler.sample(k, v_range=v_range)
    rng = np.random.default_rng(seed)
    Y, Yt, ok = [], [], []
    for a in acts:
        th = rng.uniform(-np.pi, np.pi)
        sim.set_pose((0.0, 0.0, th))
        r = sim.push(to_action(a))
        Y.append(body_delta(r.pose_before, r.pose_after))
        Yt.append(body_delta(r.true_before, r.true_after))
        ok.append(r.valid and sim.tilt() < 5)
    return dict(obj=obj, actions=acts, Y=np.array(Y), Y_true=np.array(Yt), ok=np.array(ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-objects", type=int, default=24)
    ap.add_argument("--pushes", type=int, default=150)
    ap.add_argument("--families", default="all", choices=["all", "train", "heldout"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--regime", default="standard", choices=["standard", "hetero"],
                    help="hetero: variable push speed 0.05-0.45 m/s and 50%% caster-wheeled objects")
    args = ap.parse_args()
    fams = dict(all=FAMILIES, train=TRAIN_FAMILIES, heldout=HELDOUT_FAMILIES)[args.families]
    hetero = args.regime == "hetero"
    objs = object_set(args.seed, args.n_objects, families=fams, p_casters=0.5 if hetero else 0.0)
    v_range = (0.05, 0.45) if hetero else (0.15, 0.15)
    t = time.time()
    with Pool(args.workers) as pool:
        out = pool.map(run_object, [(o, args.pushes, args.seed * 1000 + i, v_range) for i, o in enumerate(objs)])
    DATA.mkdir(exist_ok=True)
    with open(DATA / f"pushes_{args.name}.pkl", "wb") as f:
        pickle.dump(out, f)
    n_ok = sum(int(o["ok"].sum()) for o in out)
    print(f"{len(out)} objects, {n_ok}/{len(out) * args.pushes} valid pushes, {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
