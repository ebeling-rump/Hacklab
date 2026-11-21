"""Minimal Active Spatial Mapping experiment: lidar -> partial occupancy grid.

Reuses the Unitree A1 + lidar scene from a1_dataset.py / a1_lidar.py. The robot
is placed at a few random collision-free poses (or walks with random commands,
--walk); at each pose the rangefinder ring is read and every ray is cast into a
2D grid (unknown / free / occupied). The ground-truth grid comes from the box
geoms of the compiled MuJoCo model. We report map coverage and accuracy, then
fit an OccupancyPredictor baseline on observed cells and score it on the
unknown cells.

    python asm_experiment.py                    # env_A, 5 poses
    python asm_experiment.py --env env_B --poses 10 --png asm_results/grid.png
    python asm_experiment.py --walk --steps 20  # random walk instead of teleporting
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from a1_dataset import ARENA, ENVIRONMENTS, STEP_TIME, build_model, reset_robot
from a1_lidar import MAX_RANGE, MAX_TURN, N_RAYS, SMOOTHING, gait
from occupancy_predictors import ConstantPrior, GaussianProcess, NearestObserved, TabPFN

PREDICTORS = {"constant_prior": ConstantPrior, "nearest_observed": NearestObserved,
              "gp": GaussianProcess, "tabpfn": TabPFN}

UNKNOWN, FREE, OCC = -1, 0, 1
RES = 0.1                 # cell size (m)
EXTENT = ARENA + 1.0      # grid covers [-EXTENT, EXTENT]^2


class Grid:
    def __init__(self, res=RES, extent=EXTENT):
        self.res, self.extent = res, extent
        self.n = int(round(2 * extent / res))
        self.cells = np.full((self.n, self.n), UNKNOWN, np.int8)   # [ix, iy]
        c = -extent + res * (np.arange(self.n) + 0.5)
        self.cx, self.cy = np.meshgrid(c, c, indexing="ij")

    def idx(self, xy):
        i = np.floor((np.asarray(xy) + self.extent) / self.res).astype(int)
        ok = ((i >= 0) & (i < self.n)).all(-1)
        return i, ok

    def coords(self):
        return np.stack([self.cx.ravel(), self.cy.ravel()], 1)


def lidar_scan(model, data):
    """World-frame ray origins (N,2), unit directions (N,2), ranges (N,), hit mask."""
    sids = [model.site(f"lidar{i}").id for i in range(N_RAYS)]
    origin = data.site_xpos[sids, :2]
    direction = data.site_xmat[sids].reshape(-1, 3, 3)[:, :2, 2]   # site +z axis
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    r = data.sensordata[:N_RAYS].copy()
    hit = (r >= 0) & (r < MAX_RANGE)
    r[~hit] = MAX_RANGE
    return origin, direction, r, hit


def integrate_scan(grid: Grid, origin, direction, ranges, hit):
    """Ray casting: cells along each ray -> free, end cell of a hit -> occupied."""
    step = grid.res / 3
    for o, d, r, h in zip(origin, direction, ranges, hit):
        t = np.arange(0.0, r, step)
        i, ok = grid.idx(o + t[:, None] * d)
        i = i[ok]
        free = grid.cells[i[:, 0], i[:, 1]] != OCC     # never erase an obstacle
        grid.cells[i[free, 0], i[free, 1]] = FREE
        if h:
            # Push the endpoint half a cell into the surface so it lands in the box cell.
            (j,), (okj,) = grid.idx((o + (r + grid.res / 2) * d)[None])
            if okj:
                grid.cells[j[0], j[1]] = OCC


def ground_truth(model, data, grid: Grid):
    """Cells whose centre lies inside any box geom footprint (axis-aligned boxes)."""
    gt = np.zeros((grid.n, grid.n), np.int8)
    for g in range(model.ngeom):
        name = model.geom(g).name
        if not name.startswith("box"):
            continue
        (x, y), (hx, hy) = data.geom_xpos[g, :2], model.geom_size[g, :2]
        gt[(np.abs(grid.cx - x) <= hx) & (np.abs(grid.cy - y) <= hy)] = OCC
    return gt


def binary_metrics(pred, truth):
    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    acc = float((pred == truth).mean()) if len(truth) else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return dict(accuracy=acc, precision=prec, recall=rec, tp=tp, fp=fp, fn=fn)


def evaluate_map(grid: Grid, gt):
    obs = grid.cells != UNKNOWN
    m = binary_metrics(grid.cells[obs], gt[obs])
    return dict(cells=grid.cells.size, observed=int(obs.sum()), unknown=int((~obs).sum()),
                coverage=float(obs.mean()), **m)


def training_set(grid: Grid, max_train: int, rng):
    """Observed cells used to fit every predictor (identical for all). If there are
    more than max_train, keep all occupied cells and a random subset of free ones."""
    obs = (grid.cells != UNKNOWN).ravel()
    idx = np.flatnonzero(obs)
    y = grid.cells.ravel()[idx]
    if max_train and len(idx) > max_train:
        occ, free = idx[y == OCC], idx[y == FREE]
        free = rng.choice(free, max(max_train - len(occ), 0), replace=False)
        idx = np.sort(np.concatenate([occ, free]))
    return idx, obs


def evaluate_predictor(predictor, grid: Grid, gt, train_idx, obs):
    """Fit on the shared training cells, score only on genuinely unobserved cells."""
    X, y_obs, y_gt = grid.coords(), grid.cells.ravel(), gt.ravel()
    t0 = time.perf_counter()
    predictor.fit(X[train_idx], y_obs[train_idx])
    t1 = time.perf_counter()
    p, u = predictor.predict(X[~obs])
    t2 = time.perf_counter()
    m = binary_metrics((p >= 0.5).astype(int), y_gt[~obs])
    m["brier"] = float(np.mean((p - y_gt[~obs]) ** 2))
    m["mean_uncertainty"] = float(np.mean(u))
    m.update(fit_s=t1 - t0, predict_s=t2 - t1, runtime_s=t2 - t0,
             n_train=int(len(train_idx)), n_query=int((~obs).sum()))
    full = grid.cells.ravel().astype(float).copy()
    full[~obs] = p
    return m, full.reshape(grid.n, grid.n)


def collect(model, data, grid, rng, boxes, poses, walk, steps):
    if not walk:
        for _ in range(poses):
            reset_robot(model, data, boxes, rng)
            integrate_scan(grid, *lidar_scan(model, data))
        return poses
    reset_robot(model, data, boxes, rng)
    integrate_scan(grid, *lidar_scan(model, data))
    substeps = int(round(STEP_TIME / model.opt.timestep))
    alpha, cmd = model.opt.timestep / SMOOTHING, np.zeros(2)
    for _ in range(steps):
        action = np.array([rng.uniform(0.3, 1.0), rng.uniform(-MAX_TURN, MAX_TURN)])
        for _ in range(substeps):
            cmd += alpha * (action - cmd)
            data.ctrl[:] = gait(data.time, *cmd)
            mujoco.mj_step(model, data)
        integrate_scan(grid, *lidar_scan(model, data))
    return steps + 1


def save_png(path, grid, gt, maps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = [-grid.extent, grid.extent, -grid.extent, grid.extent]
    panels = [(gt, "ground truth", dict(cmap="gray_r", vmin=0, vmax=1)),
              (grid.cells, "observed (-1 unknown)", dict(cmap="viridis", vmin=-1, vmax=1))]
    panels += [(img, f"{name}: P(occupied)", dict(cmap="gray_r", vmin=0, vmax=1))
               for name, img in maps.items()]
    fig, ax = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4.5))
    for a, (img, t, kw) in zip(ax, panels):
        im = a.imshow(img.T, origin="lower", extent=ext, **kw)
        a.set_title(t)
        fig.colorbar(im, ax=a, fraction=0.046)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="env_A", choices=list(ENVIRONMENTS))
    ap.add_argument("--poses", type=int, default=5)
    ap.add_argument("--walk", action="store_true", help="random walk instead of teleport")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--png", default="asm_results/occupancy.png", help="'' to skip")
    ap.add_argument("--out", default="asm_results/comparison.json", help="'' to skip")
    ap.add_argument("--predictors", nargs="+", default=["nearest_observed", "gp", "tabpfn"],
                    choices=list(PREDICTORS))
    ap.add_argument("--max-train", type=int, default=2000,
                    help="cap on shared training cells (all occupied + random free); 0 = all")
    args = ap.parse_args()

    boxes = ENVIRONMENTS[args.env]
    model = build_model(boxes)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    grid, rng = Grid(), np.random.default_rng(args.seed)
    gt = ground_truth(model, data, grid)

    n_scans = collect(model, data, grid, rng, boxes, args.poses, args.walk, args.steps)
    print(f"{args.env}: {len(boxes)} boxes, {n_scans} scans x {N_RAYS} rays, "
          f"grid {grid.n}x{grid.n} @ {grid.res} m, GT occupied cells {int(gt.sum())}")

    m = evaluate_map(grid, gt)
    print(f"observed map : coverage {m['coverage']:.3f} ({m['observed']}/{m['cells']}), "
          f"unknown {m['unknown']}, accuracy {m['accuracy']:.3f}, "
          f"precision {m['precision']:.3f}, recall {m['recall']:.3f} "
          f"(tp {m['tp']}, fp {m['fp']}, fn {m['fn']})")

    train_idx, obs = training_set(grid, args.max_train, rng)
    n_occ = int((grid.cells.ravel()[train_idx] == OCC).sum())
    print(f"training cells (shared by all predictors): {len(train_idx)} "
          f"({n_occ} occupied); query = {int((~obs).sum())} unobserved cells")

    results, maps = {}, {}
    for name in args.predictors:
        pm, full = evaluate_predictor(PREDICTORS[name](), grid, gt, train_idx, obs)
        results[name], maps[name] = pm, full
        print(f"{name:16s} unobserved cells: brier {pm['brier']:.3f}, "
              f"precision {pm['precision']:.3f}, recall {pm['recall']:.3f}, "
              f"accuracy {pm['accuracy']:.3f}, runtime {pm['runtime_s']:.2f} s "
              f"(fit {pm['fit_s']:.2f}, predict {pm['predict_s']:.2f})")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(dict(
            env=args.env, poses=args.poses, walk=args.walk, seed=args.seed,
            map=m, n_train=len(train_idx), predictors=results), indent=2, default=float))
        print(f"saved {args.out}")
    if args.png:
        save_png(args.png, grid, gt, maps)


if __name__ == "__main__":
    main()
