"""Record one episode and save a top-down trajectory plot and (optionally) an animated GIF.

Usage:
    python plot_episode.py --seed 3                          # warehouse, constant velocity
    python plot_episode.py --seed 3 --gif                    # + GIF rendered by MuJoCo (top camera)
    python plot_episode.py --seed 3 --predictor oracle
    python plot_episode.py --scenario open --seed 9 --predictor none --no-safety
Outputs go to data/episode_<scenario>_<predictor>[_nosafety]_<seed>.{png,gif,npz}.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from controller import R_ROBOT  # noqa: E402
from obstacles import ARENA  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402
from sim import ALL_PREDICTORS, DATA, Episode  # noqa: E402

ROBOT_C = "#2a78d6"
OBS_C = ["#eb6834", "#6250d6", "#1baf7a", "#e87ba4", "#e34948"]
SHELF_C, STATION_C = "#d9cbb8", "#eda100"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
PATTERN_LABEL = {"constant_velocity": "constant velocity", "random_turns": "abrupt turns",
                 "warehouse_agv": "AGV"}


def record(seed, predictor, safety, scenario, gif):
    ep = Episode(seed, predictor, safety=safety, scenario=scenario)
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
    a = ep.arrays()
    m = ep.metrics()
    fig, ax = plt.subplots(figsize=(7.6, 7.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-ARENA, ARENA)
    ax.set_ylim(-ARENA, ARENA)
    ax.set_aspect("equal")
    ax.tick_params(colors=INK_2, labelsize=9)
    for s in ax.spines.values():
        s.set_color(INK_2)

    if ep.sc.shelves:
        from warehouse import STATIONS
        for x0, x1, y0, y1 in ep.sc.shelves:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=SHELF_C, zorder=0))
        ax.scatter(STATIONS[:, 0], STATIONS[:, 1], marker="s", s=90, color=STATION_C, alpha=0.5,
                   zorder=1, label="pick station")
    else:
        ax.grid(color=GRID, linewidth=0.8)

    for i, o in enumerate(ep.obstacles):
        p = a["obs_pos"][:, i]
        c = OBS_C[i % len(OBS_C)]
        ax.plot(p[:, 0], p[:, 1], color=c, linewidth=1.5, zorder=2,
                label=f"obstacle {i}: {PATTERN_LABEL.get(o.pattern, o.pattern)}")
        ax.plot(p[::10, 0], p[::10, 1], "o", color=c, markersize=3, zorder=2)
        ax.add_patch(Circle(p[-1], o.radius, color=c, alpha=0.4, zorder=2))

    r = a["robot_pos"]
    ax.plot(r[:, 0], r[:, 1], color=ROBOT_C, linewidth=2.4, label="robot", zorder=3)
    ax.plot(r[::10, 0], r[::10, 1], "o", color=ROBOT_C, markersize=3, zorder=3)
    ax.add_patch(Circle(r[-1], R_ROBOT, color=ROBOT_C, alpha=0.4, zorder=3))
    ax.plot(*r[0], "s", color=INK, markersize=7, zorder=4)
    ax.annotate("start", r[0], xytext=(6, 6), textcoords="offset points", color=INK, fontsize=9)
    ax.plot(*ep.goal, "*", color="#0ca30c", markersize=16, label="goal", zorder=4)

    # stops: filled = needed, open = unnecessary
    for k, dur, unnecessary in ep.stops():
        ax.plot(*r[k], "o", markersize=9, zorder=5, markeredgecolor=INK, markeredgewidth=1.4,
                markerfacecolor="none" if unnecessary else INK)
        ax.annotate(f"{'unnecessary ' if unnecessary else ''}stop {dur:.1f} s", r[k], xytext=(8, -12),
                    textcoords="offset points", color=INK, fontsize=8)

    k = np.unravel_index(np.argmin(a["clearance"]), a["clearance"].shape)
    p1, p2 = r[k[0]], a["obs_pos"][k[0], k[1]]
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=INK, linewidth=1, linestyle=":", zorder=5)
    ax.annotate(f"closest gap {a['clearance'][k]:.2f} m, t={a['t'][k[0]]:.1f} s", (p1 + p2) / 2,
                xytext=(10, 10), textcoords="offset points", color=INK, fontsize=8)

    result = f"reached goal in {m['time_to_goal']:.1f} s" if m["reached_goal"] else "did not reach goal"
    ax.set_title(f"{ep.scenario}, seed {ep.seed}, predictor: {ep.predictor_name}"
                 f"{'' if ep.safety else ', safety off'}\n{result} | collisions {m['collision_events']} | "
                 f"min gap {m['min_clearance']:.2f} m | stops {m['stops']} ({m['unnecessary_stops']} unnecessary)",
                 color=INK, fontsize=9.5, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=8, frameon=False,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def save_gif(frames, out, fps=10):
    from PIL import Image

    images = [Image.fromarray(f) for f in frames]
    images[0].save(out, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="warehouse", choices=list(SCENARIOS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--predictor", default="constant_velocity", choices=ALL_PREDICTORS)
    parser.add_argument("--no-safety", action="store_true")
    parser.add_argument("--gif", action="store_true", help="also render an animated GIF (top camera)")
    args = parser.parse_args()

    ep, frames = record(args.seed, args.predictor, not args.no_safety, args.scenario, args.gif)
    DATA.mkdir(exist_ok=True)
    stem = f"episode_{args.scenario}_{args.predictor}{'_nosafety' if args.no_safety else ''}_{args.seed}"
    plot(ep, DATA / f"{stem}.png")
    print(f"Saved {DATA / (stem + '.png')}")
    if frames:
        save_gif(frames, DATA / f"{stem}.gif")
        print(f"Saved {DATA / (stem + '.gif')} ({len(frames)} frames)")
    ep.save_log(DATA / f"{stem}.npz")


if __name__ == "__main__":
    main()
