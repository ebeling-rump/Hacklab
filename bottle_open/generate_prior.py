"""Prior experience: random exploratory strokes on the 150 *training* bottles.

Each stroke becomes one row: observable bottle state before the stroke + action -> outcome
(progress, slip, damage). Saved to data/prior.npz. Test bottles are never touched here.

    python generate_prior.py
"""
from pathlib import Path

import numpy as np

from controllers import RandomExplorer
from features import FEATURES, TARGETS, history_rows
from sim import TRAIN_SEEDS, run_episode, sample_bottle

DATA = Path(__file__).parent / "data"


def main(max_strokes=25):
    DATA.mkdir(exist_ok=True)
    Xs, Ys, ids = [], [], []
    for seed in TRAIN_SEEDS:
        ep = run_episode(sample_bottle(seed), RandomExplorer(seed), seed=seed, max_strokes=max_strokes)
        X, Y = history_rows(ep["history"])
        Xs.append(X); Ys.append(Y); ids += [seed] * len(X)
    X, Y = np.vstack(Xs), np.vstack(Ys)
    np.savez(DATA / "prior.npz", X=X, Y=Y, bottle=np.array(ids), features=FEATURES, targets=TARGETS)
    print(f"{len(X)} rows from {len(TRAIN_SEEDS)} training bottles")
    print(f"  strokes with progress: {np.mean(Y[:, 0] > 3):.0%}, slip: {np.mean(Y[:, 1] > 5):.0%}, "
          f"damage: {np.mean(Y[:, 2]):.0%}")


if __name__ == "__main__":
    main()
