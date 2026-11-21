"""Plot actual vs predicted ball positions for the baseline and TabPFN.

Reads data/predictions.csv (written by tabpfn_model.py), so it makes no API calls.

Usage:
    python plot_results.py            # writes data/predictions.png
    python plot_results.py --show     # also opens a window
"""
import argparse
import csv
from pathlib import Path

import matplotlib

import numpy as np

from tabpfn_model import PRED_PATH

# Validated categorical pair (blue / orange) plus text and surface tokens.
MODELS = {  # key in predictions.csv -> (label, color, marker)
    "baseline": ("Constant velocity", "#eb6834", "^"),
    "tabpfn": ("TabPFN", "#2a78d6", "o"),
}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load_predictions(path=PRED_PATH):
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def mean_error(p, model):
    return np.hypot(p[f"{model}_x"] - p["x_true"], p[f"{model}_y"] - p["y_true"]).mean()


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.set_aspect("equal")


def plot(p, out):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), facecolor=SURFACE)
    lim = 1.1 * max(np.abs(v).max() for v in p.values())

    # Panels 1-2: predicted vs actual for each coordinate. Perfect = on the diagonal.
    for ax, axis in zip(axes[:2], ("x", "y")):
        style(ax, f"Final {axis} position: predicted vs actual")
        ax.plot([-lim, lim], [-lim, lim], color=INK_2, linewidth=1, linestyle="--", zorder=1)
        ax.text(lim * 0.95, lim * 0.83, "perfect", color=INK_2, fontsize=9, ha="right", rotation=45)
        for key, (label, color, marker) in MODELS.items():
            ax.scatter(p[f"{axis}_true"], p[f"{key}_{axis}"], s=36, c=color, marker=marker,
                       edgecolors=SURFACE, linewidths=1, label=label, zorder=2)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"actual {axis} (m)", color=INK_2, fontsize=9)
        ax.set_ylabel(f"predicted {axis} (m)", color=INK_2, fontsize=9)

    # Panel 3: top-down floor view. Each test run: actual end point, lines to each prediction.
    ax = axes[2]
    style(ax, "Floor view: where each ball ended up")
    for key, (label, color, marker) in MODELS.items():
        for i in range(len(p["x_true"])):
            ax.plot([p["x_true"][i], p[f"{key}_x"][i]], [p["y_true"][i], p[f"{key}_y"][i]],
                    color=color, linewidth=1, alpha=0.6, zorder=1)
        ax.scatter(p[f"{key}_x"], p[f"{key}_y"], s=30, c=color, marker=marker,
                   edgecolors=SURFACE, linewidths=1, label=f"{label} prediction", zorder=2)
    ax.scatter(p["x_true"], p["y_true"], s=30, facecolors=SURFACE, edgecolors=INK,
               linewidths=1.2, label="Actual (MuJoCo)", zorder=3)
    ax.scatter([0], [0], s=60, marker="+", c=INK, zorder=3)
    ax.text(0.05, -0.12, "start", color=INK_2, fontsize=9)
    ax.set_xlim(-lim * 1.45, lim * 1.45)
    ax.set_ylim(-lim * 1.45, lim * 1.45)
    ax.set_xlabel("x (m)", color=INK_2, fontsize=9)
    ax.set_ylabel("y (m)", color=INK_2, fontsize=9)
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK)

    errs = ", ".join(f"{label} {mean_error(p, k):.3f} m" for k, (label, _, _) in MODELS.items())
    axes[0].legend(loc="upper left", fontsize=9, frameon=False, labelcolor=INK)
    fig.suptitle(f"Ball position after 1 s, 50 test runs. Mean position error: {errs}",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=PRED_PATH.with_suffix(".png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")

    plot(load_predictions(), args.out)
    print(f"Saved plot to {args.out}")
    if args.show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
