"""Side-by-side animation of one seed under two predictors, with the predictions drawn in.

Dots in front of each AGV: where the predictor expects it over the next 1 s.
Faint line: where the AGV actually went in that second. The gap between them is the
prediction error the planner had to live with. Local predictors only (no API calls).

Usage:
    python compare_episodes.py --seed 3                         # constant_velocity vs gp
    python compare_episodes.py --seed 20 --right oracle
Outputs data/compare_<left>_vs_<right>_<seed>.gif and a key-frame .png.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from controller import R_ROBOT  # noqa: E402
from nav_eval import make_predictor  # noqa: E402
from obstacles import ARENA  # noqa: E402
from sim import DATA, Episode  # noqa: E402
from warehouse import R_AGV, STATIONS  # noqa: E402

ROBOT_C = "#2a78d6"
OBS_C = ["#eb6834", "#6250d6", "#1baf7a", "#e87ba4"]
SHELF_C, STATION_C = "#d9cbb8", "#eda100"
INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
TITLES = {"constant_velocity": "constant velocity", "gp": "GP prediction (local)", "oracle": "oracle (perfect)",
          "static": "static obstacles"}


def record(seed, name):
    pred = make_predictor(name)
    ep = Episode(seed, name if pred is None else "constant_velocity", predictor_obj=pred)
    frames = []
    while not ep.done and ep.data.time < ep.sc.max_time:
        for _ in range(ep.control_every):
            ep.step()
            if ep.done:
                break
        p = ep.predictions
        frames.append({"t": ep.data.time, "robot": ep.robot_pos, "speed": float(np.linalg.norm(ep.data.qvel[:2])),
                       "obs": ep.observed_obstacles(), "pred": None if p is None else p.copy(), "done": ep.done})
    return ep, frames


def draw_panel(ax, ep, frames, k, title):
    ax.clear()
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-ARENA, ARENA)
    ax.set_ylim(-ARENA, ARENA)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for x0, x1, y0, y1 in ep.sc.shelves:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=SHELF_C, zorder=0))
    ax.scatter(STATIONS[:, 0], STATIONS[:, 1], marker="s", s=70, color=STATION_C, alpha=0.5, zorder=1)
    ax.plot(*ep.goal, "*", color="#0ca30c", markersize=18, zorder=2)

    k = min(k, len(frames) - 1)
    f = frames[k]
    trail = np.array([fr["robot"] for fr in frames[:k + 1]])
    ax.plot(trail[:, 0], trail[:, 1], color=ROBOT_C, linewidth=2, alpha=0.6, zorder=3)
    for i, pos in enumerate(f["obs"]):
        c = OBS_C[i % len(OBS_C)]
        future = np.array([fr["obs"][i] for fr in frames[k:k + 11]])
        ax.plot(future[:, 0], future[:, 1], color=c, linewidth=5, alpha=0.25, zorder=2, solid_capstyle="round")
        if f["pred"] is not None and np.linalg.norm(pos - f["robot"]) < 4.0:
            ax.scatter(f["pred"][i, 1::2, 0], f["pred"][i, 1::2, 1], s=16, color=c, edgecolors=INK,
                       linewidths=0.5, zorder=4)
        ax.add_patch(Circle(pos, R_AGV, color=c, zorder=3))
    ax.add_patch(Circle(f["robot"], R_ROBOT, color=ROBOT_C, zorder=5))
    status = "goal reached" if f["done"] else ("waiting" if f["speed"] < 0.05 else f"{f['speed']:.1f} m/s")
    ax.set_title(f"{title}\nt = {f['t']:5.1f} s   {status}", loc="left", color=INK, fontsize=11)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--left", default="constant_velocity")
    parser.add_argument("--right", default="gp")
    parser.add_argument("--every", type=int, default=2, help="control steps per animation frame")
    args = parser.parse_args()

    left, right = record(args.seed, args.left), record(args.seed, args.right)
    n = max(len(left[1]), len(right[1]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.1), facecolor=SURFACE)
    fig.text(0.01, 0.01, "dots = predicted AGV positions for the next 1 s (AGVs within 4 m)   "
             "faint band = where the AGV actually went", color=INK_2, fontsize=9)

    images = []
    from PIL import Image
    for k in list(range(0, n, args.every)) + [n - 1] * 10:          # hold the last frame
        draw_panel(axes[0], *left, k, TITLES.get(args.left, args.left))
        draw_panel(axes[1], *right, k, TITLES.get(args.right, args.right))
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
    stem = DATA / f"compare_{args.left}_vs_{args.right}_{args.seed}"
    images = [im.quantize(colors=128) for im in images]
    images[0].save(f"{stem}.gif", save_all=True, append_images=images[1:], duration=100 * args.every, loop=0)

    # key frame: shortly after the two robots' paths first diverge (> 0.3 m apart)
    m = min(len(left[1]), len(right[1]))
    apart = [i for i in range(m) if np.linalg.norm(left[1][i]["robot"] - right[1][i]["robot"]) > 0.3]
    k = min(apart[0] + 3, m - 1) if apart else n // 2
    draw_panel(axes[0], *left, k, TITLES.get(args.left, args.left))
    draw_panel(axes[1], *right, k, TITLES.get(args.right, args.right))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(f"{stem}.png", dpi=110, facecolor=SURFACE)
    print(f"Saved {stem}.gif ({len(images)} frames) and {stem}.png (t = {left[1][k]['t']:.1f} s)")


if __name__ == "__main__":
    main()
