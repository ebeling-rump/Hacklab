"""Seed/environment sweep: TabPFN vs GP vs spatial baselines on unobserved cells.

For every (env, seed) the LiDAR observations and the shared training cells are built
once and reused by every predictor. Metrics are computed only on cells the LiDAR never
observed. Ground truth is used for scoring only; leakage checks are asserted per run.

    python asm_sweep.py                              # env_A, env_B x seeds 0-4
    python asm_sweep.py --seeds 0 1 --predictors constant_prior nearest_observed gp
"""

import argparse
import csv
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from sklearn.metrics import roc_auc_score

from a1_dataset import ENVIRONMENTS, build_model
from asm_experiment import (OCC, PREDICTORS, UNKNOWN, Grid, collect, evaluate_map,
                            evaluate_predictor, ground_truth, save_png, training_set)

METRICS = ["brier", "bss", "precision", "recall", "f1", "accuracy", "balanced_acc",
           "auroc", "runtime_s"]


def extra_metrics(p, y, prior_brier):
    pred = (p >= 0.5).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0          # 0 if it never predicts occupied
    rec = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    brier = float(np.mean((p - y) ** 2))
    return dict(
        precision=prec, recall=rec,
        f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        balanced_acc=(rec + spec) / 2,
        auroc=float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) and np.ptp(p) > 0 else 0.5,
        brier=brier, bss=1 - brier / prior_brier)


def run(env, seed, predictors, max_train, poses, maps_dir):
    boxes = ENVIRONMENTS[env]
    model = build_model(boxes)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    grid, rng = Grid(), np.random.default_rng(seed)
    gt = ground_truth(model, data, grid)
    collect(model, data, grid, rng, boxes, poses, False, 0)
    mapm = evaluate_map(grid, gt)
    train_idx, obs = training_set(grid, max_train, rng)

    # --- leakage checks ---
    assert not np.any(~obs[train_idx]), "training cell is unobserved"
    assert set(np.unique(grid.cells.ravel()[train_idx])) <= {0, 1}, "training label not from LiDAR"
    y_q = gt.ravel()[~obs]
    occ_frac = float(y_q.mean())
    # Brier of the "true" base rate on the query cells (reference for BSS, not a predictor).
    prior_brier = float(np.mean((occ_frac - y_q) ** 2))

    rows, maps = [], {}
    for name in predictors:
        try:
            m, full = evaluate_predictor(PREDICTORS[name](), grid, gt, train_idx, obs)
        except Exception as e:                       # e.g. TabPFN API failure
            print(f"  {env} seed {seed} {name}: FAILED {type(e).__name__}: {e}")
            continue
        p = full.ravel()[~obs]
        m.update(extra_metrics(p, y_q, prior_brier))
        rows.append(dict(env=env, seed=seed, predictor=name, coverage=mapm["coverage"],
                         n_train=len(train_idx), n_train_occ=int((grid.cells.ravel()[train_idx] == OCC).sum()),
                         n_query=int((~obs).sum()), query_occ_frac=occ_frac,
                         **{k: m[k] for k in METRICS + ["fit_s", "predict_s"]}))
        maps[name] = full
        print(f"  {env} seed {seed} {name:16s} brier {m['brier']:.3f} prec {m['precision']:.2f} "
              f"rec {m['recall']:.2f} auroc {m['auroc']:.3f} {m['runtime_s']:.1f}s")
    if maps_dir and seed == 0:
        save_png(Path(maps_dir) / f"maps_{env}_seed{seed}.png", grid, gt, maps)
    return rows


def summarise(rows, predictors):
    out = []
    for env in sorted({r["env"] for r in rows}) + ["all"]:
        for name in predictors:
            sel = [r for r in rows if r["predictor"] == name and env in (r["env"], "all")]
            if not sel:
                continue
            s = dict(env=env, predictor=name, runs=len(sel),
                     query_occ_frac=float(np.mean([r["query_occ_frac"] for r in sel])))
            for k in METRICS:
                v = np.array([r[k] for r in sel], float)
                s[k], s[k + "_sd"] = float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0
            out.append(s)
    return out


def paired_wins(rows, a, b, key="brier"):
    """How often a beats b (lower brier) on identical (env, seed) runs."""
    ra = {(r["env"], r["seed"]): r[key] for r in rows if r["predictor"] == a}
    rb = {(r["env"], r["seed"]): r[key] for r in rows if r["predictor"] == b}
    common = sorted(set(ra) & set(rb))
    d = np.array([ra[k] - rb[k] for k in common])
    return int((d < 0).sum()), len(common), float(d.mean()) if len(d) else float("nan")


def markdown(summary, rows):
    cols = [("brier", "Brier ↓", 3), ("bss", "BSS ↑", 2), ("precision", "Precision", 2),
            ("recall", "Recall", 2), ("f1", "F1", 2), ("accuracy", "Accuracy", 3),
            ("balanced_acc", "Bal. acc", 2), ("auroc", "AUROC", 3), ("runtime_s", "Runtime s", 1)]
    lines = ["| Env | Predictor | Runs | Occ. frac (query) | " + " | ".join(c[1] for c in cols) + " |",
             "|" + "---|" * (4 + len(cols))]
    for s in summary:
        cells = [f"{s[k]:.{d}f} ± {s[k + '_sd']:.{d}f}" for k, _, d in cols]
        lines.append(f"| {s['env']} | {s['predictor']} | {s['runs']} | {s['query_occ_frac']:.3f} | "
                     + " | ".join(cells) + " |")
    lines += ["", "Paired Brier comparison on identical runs (wins / runs, mean Δ Brier):", ""]
    for a, b in (("tabpfn", "gp"), ("tabpfn", "nearest_observed"), ("gp", "nearest_observed")):
        w, n, d = paired_wins(rows, a, b)
        if n:
            lines.append(f"- {a} vs {b}: {a} lower Brier in {w}/{n} runs, mean Δ {d:+.4f}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=["env_A", "env_B"], choices=list(ENVIRONMENTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--predictors", nargs="+", choices=list(PREDICTORS),
                    default=["constant_prior", "nearest_observed", "gp", "tabpfn"])
    ap.add_argument("--poses", type=int, default=5)
    ap.add_argument("--max-train", type=int, default=2000)
    ap.add_argument("--out", default="asm_results/sweep")
    args = ap.parse_args()
    import sys
    sys.stdout.reconfigure(encoding="utf-8")        # Windows console: ±, ↓, Δ

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0, rows = time.perf_counter(), []
    for env in args.envs:
        for seed in args.seeds:
            rows += run(env, seed, args.predictors, args.max_train, args.poses, out)
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = summarise(rows, args.predictors)
    (out / "summary.json").write_text(json.dumps(dict(args=vars(args), summary=summary), indent=2))
    md = markdown(summary, rows)
    (out / "summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"total {time.perf_counter() - t0:.0f} s, saved to {out}")


if __name__ == "__main__":
    main()
