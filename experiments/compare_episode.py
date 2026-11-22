"""Run the same task with several methods and render a side-by-side top-down animation.

    python experiments/compare_episode.py --challenge C5 --seed 3 --methods physics_sysid tabpfn_hybrid
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon as MplPoly, Circle, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "2")

from pushlab.tasks import CHALLENGES  # noqa: E402
from pushlab.runner import run_episode  # noqa: E402
from pushlab.learners import make_model  # noqa: E402
from pushlab.sim import PUSHER_R  # noqa: E402
from experiments.make_figures import COLORS, LABELS  # noqa: E402


def poly_xy(shape, pose):
    return np.asarray(shape.world_polygon(pose).exterior.coords)


def frames_from_log(log):
    """Flatten per-push trajectories into frames: (obj_pose, pusher_xy, push_index)."""
    fr = []
    for k, traj in enumerate(log["traj"]):
        for (_, pose, pxy) in traj[::2]:
            fr.append((np.asarray(pose), np.asarray(pxy), k))
    return fr


def draw_static(ax, task):
    for cx, cy, hx, hy in task.walls:
        ax.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, color="#4a5160"))
    g = task.goal
    if task.ang_tol is not None:
        ax.add_patch(MplPoly(poly_xy(task.obj.shape, g), closed=True, fill=False, ec="#2ca02c", lw=1.5, ls="--"))
    else:
        ax.add_patch(Circle(g[:2], task.pos_tol, fill=False, ec="#2ca02c", lw=1.5, ls="--"))
    ax.plot(g[0], g[1], "*", color="#2ca02c", ms=12)
    x0, x1, y0, y1 = task.bounds
    ax.set_xlim(x0 - 0.05, x1 + 0.05)
    ax.set_ylim(y0 - 0.05, y1 + 0.05)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def render(task, results, out_mp4, out_png, fps=25):
    import imageio.v2 as imageio
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.6))
    axes = np.atleast_1d(axes)
    seqs = [frames_from_log(log) for _, _, log in results]
    T = max(len(s) for s in seqs)
    shape = task.obj.shape
    com = np.asarray(task.obj.hidden.com)
    writer = imageio.get_writer(str(out_mp4), fps=fps, codec="libx264", quality=7, macro_block_size=8)
    step = max(1, T // 900)
    for t in list(range(0, T, step)) + [T - 1] * fps:
        for ax, (name, out, log), seq in zip(axes, results, seqs):
            ax.clear()
            draw_static(ax, task)
            if not seq:
                continue
            pose, pxy, k = seq[min(t, len(seq) - 1)]
            done = t >= len(seq) - 1
            # path so far
            poses = np.array([s[0] for s in seq[: min(t, len(seq) - 1) + 1]])
            ax.plot(poses[:, 0], poses[:, 1], color=COLORS.get(name, "k"), lw=1.2, alpha=0.7)
            # predicted outcome of the current push (+ 1-sigma position circle)
            if k < len(log["pred"]) and not done:
                pp = log["pred"][k]
                sd = log["pred_std"][k]
                ax.add_patch(MplPoly(poly_xy(shape, pp), closed=True, fill=True, fc="#1f77b4", alpha=0.15,
                                     ec="#1f77b4", ls="--"))
                ax.add_patch(Circle(pp[:2], max(np.linalg.norm(sd[:2]), 0.002), fill=False, ec="#1f77b4", lw=0.8))
            ax.add_patch(MplPoly(poly_xy(shape, pose), closed=True, fc="#f28e2b", ec="k", lw=1))
            c, s = np.cos(pose[2]), np.sin(pose[2])
            cw = pose[:2] + np.array([c * com[0] - s * com[1], s * com[0] + c * com[1]])
            ax.plot(*cw, "o", color="k", ms=3)
            if np.all(np.abs(pxy) < 5):
                ax.add_patch(Circle(pxy, PUSHER_R, color="#1f4fb4"))
            status = ("SUCCESS" if out["success"] else "FAILED") if done else f"push {k + 1}"
            ax.set_title(f"{LABELS.get(name, name)}\n{status} | pushes {min(k + 1, out['pushes'])} | "
                         f"wall contacts {sum(log['walls'][:k + 1])}", fontsize=10)
        fig.suptitle(f"{task.challenge} seed {task.seed}: same unknown object ({task.obj.name}), same start, "
                     f"same planner — different dynamics model\n(orange = actual, blue = predicted outcome "
                     f"of the current push, black dot = hidden centre of mass)", fontsize=10)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(img)
    writer.close()
    fig.savefig(out_png)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenge", default="C5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=["physics_sysid", "tabpfn_hybrid"])
    ap.add_argument("--source", default=None)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(2)
    source = None
    if args.source:
        from experiments.exp_prediction import load, source_from
        source = source_from(load(args.source))
    task = CHALLENGES[args.challenge](args.seed)
    results = []
    for m in args.methods:
        base = m.replace("_active", "")
        model = None if m == "reactive" else make_model(base, source=source)
        pk = dict(beta0=1.0, n_explore=4) if m.endswith("_active") else None
        out, log = run_episode(task, m, model, seed=args.seed, record=True, planner_kw=pk)
        print(m, {k: out[k] for k in ["success", "pushes", "pos_err_mm", "ang_err_deg", "wall_contacts"]})
        results.append((m, out, log))
    d = ROOT / "results" / "videos"
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{args.challenge}_s{args.seed}_" + "_vs_".join(args.methods)
    render(task, results, d / f"{stem}.mp4", d / f"{stem}.png")
    print("->", d / f"{stem}.mp4")


if __name__ == "__main__":
    main()
