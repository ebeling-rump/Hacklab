"""Record one episode and save a top-down trajectory plot and (optionally) an animated GIF.

Usage:
    python plot_episode.py --seed 3                         # data/episode_<pred>_<seed>.png
    python plot_episode.py --seed 3 --gif                   # + .gif rendered by MuJoCo (top camera)
    python plot_episode.py --seed 3 --predictor none --no-safety
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from controller import PREDICTORS, R_ROBOT  # noqa: E402
from obstacles import ARENA, R_OBS  # noqa: E402
from sim import DATA, Episode  # noqa: E402

ROBOT_C = "#2a78d6"
OBS_C = ["#eb6834", "#6250d6"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
PATTERN_LABEL = {"constant_velocity": "constant velocity", "random_turns": "abrupt turns"}


def record(seed, predictor, safety, gif):
    ep = Episode(seed, predictor, safety=safety)
    frames = []
    renderer = mujoco.Renderer(ep.model, 480, 480) if gif else None
    while not ep.done:
        for _ in range(ep.control_every):
            ep.step()
            if ep.done:
                break
        if renderer is not None:
            renderer.update_scene(ep.data, camera="top")
            frames.append(renderer.render().copy())
    if renderer is not None:
        renderer.close()
    return ep, frames


def plot(ep, out):
    log = {k: np.array(v) for k, v in ep.log.items()}
    m = ep.metrics()
    fig, ax = plt.subplots(figsize=(7.4, 7.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-ARENA, ARENA)
    ax.set_ylim(-ARENA, ARENA)
    ax.set_aspect("equal")
    ax.grid(color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_2, labelsize=9)
    for s in ax.spines.values():
        s.set_color(INK_2)

    # obstacle paths, with a marker every second
    for i, o in enumerate(ep.obstacles):
        p = log["obs_pos"][:, i]
        ax.plot(p[:, 0], p[:, 1], color=OBS_C[i], linewidth=1.6,
                label=f"obstacle {i}: {PATTERN_LABEL.get(o.pattern, o.pattern)}")
        ax.plot(p[::10, 0], p[::10, 1], "o", color=OBS_C[i], markersize=3)
        ax.add_patch(Circle(p[-1], R_OBS, color=OBS_C[i], alpha=0.35))
    # robot path
    r = log["robot_pos"]
    ax.plot(r[:, 0], r[:, 1], color=ROBOT_C, linewidth=2.2, label="robot")
    ax.plot(r[::10, 0], r[::10, 1], "o", color=ROBOT_C, markersize=3)
    ax.add_patch(Circle(r[-1], R_ROBOT, color=ROBOT_C, alpha=0.35))
    ax.plot(*r[0], "s", color=INK, markersize=7)
    ax.annotate("start", r[0], xytext=(6, 6), textcoords="offset points", color=INK_2, fontsize=9)
    ax.plot(*ep.goal, "*", color="#0ca30c", markersize=16, label="goal")

    # closest approach
    k = np.unravel_index(np.argmin(log["clearance"]), log["clearance"].shape)
    a, b = r[k[0]], log["obs_pos"][k[0], k[1]]
    ax.plot([a[0], b[0]], [a[1], b[1]], color=INK, linewidth=1, linestyle=":")
    ax.annotate(f"closest gap {log['clearance'][k]:.2f} m\nat t={log['t'][k[0]]:.1f} s", (a + b) / 2,
                xytext=(10, -24), textcoords="offset points", color=INK, fontsize=9)

    result = (f"reached goal in {m['time_to_goal']:.1f} s" if m["reached_goal"] else "did not reach goal")
    ax.set_title(f"seed {ep.seed}, predictor: {ep.predictor_name}"
                 f"{'' if ep.safety else ', safety off'}\n{result}, collisions: {m['collision_events']}, "
                 f"min gap {m['min_clearance']:.2f} m (dots every 1 s)",
                 color=INK, fontsize=10, loc="left")
    ax.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def save_gif(frames, out, fps=10):
    from PIL import Image

    images = [Image.fromarray(f) for f in frames]
    images[0].save(out, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--predictor", default="constant_velocity", choices=list(PREDICTORS))
    parser.add_argument("--no-safety", action="store_true")
    parser.add_argument("--gif", action="store_true", help="also render an animated GIF (top camera)")
    args = parser.parse_args()

    ep, frames = record(args.seed, args.predictor, not args.no_safety, args.gif)
    DATA.mkdir(exist_ok=True)
    stem = f"episode_{args.predictor}{'_nosafety' if args.no_safety else ''}_{args.seed}"
    plot(ep, DATA / f"{stem}.png")
    print(f"Saved {DATA / (stem + '.png')}")
    if frames:
        save_gif(frames, DATA / f"{stem}.gif")
        print(f"Saved {DATA / (stem + '.gif')} ({len(frames)} frames)")
    ep.save_log(DATA / f"{stem}.npz")
    print(f"Saved trajectory log {DATA / (stem + '.npz')}")


if __name__ == "__main__":
    main()
