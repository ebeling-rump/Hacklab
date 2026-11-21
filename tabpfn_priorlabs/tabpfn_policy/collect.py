"""Collect teacher demonstrations: run the expert closed loop and log every control step.

    python -m tabpfn_policy.collect --episodes 50 --seed-start 0 --noise 0.3 [--workers 8] [--cam-every 1]
                                    [--out data/teacher.parquet]

One row per control step (wide format; the long "one row per output_id" table is built at training time):
  meta      episode, seed, step, t, noise, success (episode outcome)
  features  FEATURES from features.py, built from the wrist camera + encoders + known bin pose (no ground truth)
  phase     expert phase computed on the TRUE state (label) and phase_perc = same phase machine on the perceived state
  labels    a_dx, a_dy, a_dz [mm], a_dyaw [rad], a_grip [mm]  = the expert's CLEAN action
  executed  x_dx, x_dy, x_dz, x_dyaw = the action actually sent (expert + noise)
  gt_*      ground-truth cube-relative features, for diagnosing perception error only (never a policy input)

Use seeds < 10000 for training; keep 10000+ for held-out evaluation.
"""
from __future__ import annotations

import os

# One BLAS thread per process: OpenBLAS otherwise commits ~0.5 GB per library (numpy, scipy) per process for
# its per-core buffers, i.e. ~1 GB per worker on a 16-core machine. Must be set before numpy is imported;
# spawned workers re-import this module, so it applies to them too.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .expert import Expert, Phase
from .features import FEATURES, CubeTracker, feature_vector, perceived_state
from .geometry import wrap90
from .runner import run_expert_episode

ROOT = Path(__file__).resolve().parents[1]


def collect_episode(episode: int, seed: int, noise: float, max_steps: int = 600, cam_every: int = 1):
    """cam_every: render + run perception every k-th control step (belief is held/aged in between)."""
    expert = Expert()
    rows = []
    ctx = {}

    def on_step(ep, step, phase, a, a_exec):
        if "tracker" not in ctx:
            ctx["tracker"] = CubeTracker(ep.robot, ep.cfg.cubes[0].color)
            ctx["phase_perc"] = Phase.APPROACH
        p, _ = ep.tcp()
        if step % cam_every == 0:
            belief = ctx["tracker"].update(ep.robot.get_observation(), p)
        else:
            belief = ctx["tracker"].hold(ep.width(), p)
        s_perc = perceived_state(ep, belief)
        ctx["phase_perc"] = expert.next_phase(s_perc, ctx["phase_perc"])
        s_true = ep.true_state()
        row = dict(episode=episode, seed=seed, step=step, t=step * ep.dt, noise=noise, cam_every=cam_every)
        row.update(feature_vector(s_perc, belief))
        row.update(phase=int(phase), phase_perc=int(ctx["phase_perc"]))
        row.update(a_dx=a[0] * 1000, a_dy=a[1] * 1000, a_dz=a[2] * 1000, a_dyaw=a[3], a_grip=a[4] * 1000)
        row.update(x_dx=a_exec[0] * 1000, x_dy=a_exec[1] * 1000, x_dz=a_exec[2] * 1000, x_dyaw=a_exec[3])
        g = (s_true.cube_pos - s_true.tcp_pos) * 1000
        row.update(gt_cube_dx=g[0], gt_cube_dy=g[1], gt_cube_dz=g[2],
                   gt_cube_dyaw=wrap90(s_true.cube_yaw - s_true.tcp_yaw))
        rows.append(row)

    res, stats = run_expert_episode(seed, expert=expert, noise=noise, max_steps=max_steps, on_step=on_step)
    for r in rows:
        r["success"] = bool(stats["success"])
    return rows, stats


def summarize(df: pd.DataFrame, ep_stats: list[dict]) -> None:
    n = len(ep_stats)
    ok = sum(s["success"] for s in ep_stats)
    print(f"\n=== {n} episodes, success {ok}/{n} = {ok / max(n, 1):.0%}, {len(df)} rows "
          f"({len(df) / max(n, 1):.0f} steps/episode avg) ===")
    steps = np.array([s["steps"] for s in ep_stats])
    print(f"steps/episode: min {steps.min()} median {int(np.median(steps))} max {steps.max()}; "
          f"IK failures total {sum(s['ik_failures'] for s in ep_stats)}; "
          f"collision steps total {sum(s['collisions'] for s in ep_stats)}")
    retries = sum(s["phases"].count(int(Phase.APPROACH)) > 1 for s in ep_stats)
    print(f"episodes with a re-approach (drop / abort recovery): {retries}")
    print("\nrows per phase, perception visibility and perceived-vs-true phase agreement:")
    g = df.groupby("phase")
    tab = pd.DataFrame({
        "rows": g.size(),
        "cube_seen": g["cube_vis"].mean().round(2),
        "in_hand_belief": g["cube_src"].apply(lambda s: (s == 1).mean()).round(2),
        "phase_perc_agree": (df["phase"] == df["phase_perc"]).groupby(df["phase"]).mean().round(2),
    })
    tab.index = [Phase(i).name for i in tab.index]
    print(tab.to_string())
    seen = df[df.cube_src == 0]
    err = np.hypot(seen.cube_dx - seen.gt_cube_dx, seen.cube_dy - seen.gt_cube_dy)
    print(f"\nperception xy error when seen [mm]: median {err.median():.1f}, p95 {err.quantile(0.95):.1f}, "
          f"max {err.max():.1f};  z error median {(seen.cube_dz - seen.gt_cube_dz).abs().median():.1f} mm")
    print("\nfeature / label ranges:")
    cols = list(FEATURES) + ["a_dx", "a_dy", "a_dz", "a_dyaw", "a_grip"]
    print(df[cols].describe().T[["mean", "std", "min", "max"]].round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0, help="action noise std as a fraction of the step clip")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--cam-every", type=int, default=1, help="run the camera/perception every k control steps")
    ap.add_argument("--workers", type=int, default=8, help="parallel episodes (separate processes)")
    ap.add_argument("--out", default=str(ROOT / "data" / "teacher.parquet"))
    a = ap.parse_args()
    out = Path(a.out)
    # Every finished episode is saved right away as a part file; a rerun with the same arguments skips
    # episodes whose part already exists, so a crash never loses finished work.
    parts = out.parent / (out.stem + "_parts")
    parts.mkdir(parents=True, exist_ok=True)

    def part_path(seed: int) -> Path:
        return parts / f"seed{seed:05d}_noise{a.noise:g}_cam{a.cam_every}.parquet"

    todo = [e for e in range(a.episodes) if not part_path(a.seed_start + e).exists()]
    print(f"{a.episodes - len(todo)} episodes already collected in {parts}, {len(todo)} to run", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as pool:
        futs = {pool.submit(collect_episode, e, a.seed_start + e, a.noise, a.max_steps, a.cam_every): e for e in todo}
        for f in as_completed(futs):
            rows, st = f.result()
            p = part_path(st["seed"])
            pd.DataFrame(rows).to_parquet(p.with_suffix(".tmp"), index=False)
            p.with_suffix(".tmp").replace(p)
            p.with_suffix(".json").write_text(json.dumps(st))
            print(f"ep {futs[f]:4d} seed {st['seed']:5d} {'OK  ' if st['success'] else 'FAIL'} steps {st['steps']:4d}  "
                  f"[{time.time() - t0:6.1f}s]", flush=True)
    seeds = [a.seed_start + e for e in range(a.episodes)]
    df = pd.concat([pd.read_parquet(part_path(s)) for s in seeds], ignore_index=True)
    ep_stats = [json.loads(part_path(s).with_suffix(".json").read_text()) for s in seeds]
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    summarize(df, ep_stats)


if __name__ == "__main__":
    main()
