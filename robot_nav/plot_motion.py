"""Plot Phase 2 results from cached predictions (no API calls).

    python plot_motion.py      # data/motion/prediction_errors.png + per-set behaviour table
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from motion_data import BEHAVIOURS, MOTION  # noqa: E402
from predict_motion import EVAL_SETS, LABEL, errors, eval_data  # noqa: E402

COLORS = {"cv": "#52514e", "gp": "#eb6834", "tabpfn": "#2a78d6"}
NAMES = {"cv": "constant velocity", "gp": "Gaussian Process", "tabpfn": "TabPFN"}
SET_NAMES = {"test": "test\n(same pattern)", "flipped": "flipped routing\n(unseen)", "fast": "fast fleet\n(unseen)"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NS = [100, 300, 1000]


def load_pred(model, n):
    with np.load(MOTION / f"pred_{model}_{n}.npz") as f:
        return f["pred"]


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def main():
    E = eval_data(np.random.default_rng(0))       # same rows as predict_motion.py
    preds = {("cv", 0): load_pred("cv", 0)} | {(m, n): load_pred(m, n) for m in ("gp", "tabpfn") for n in NS}
    e10 = {k: errors(E, p)[1] for k, p in preds.items()}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1, 1.2, 1.4]})

    # 1. learning curve on the test set
    ax = axes[0]
    style(ax, "Error at +1 s vs training samples (test set)")
    test = E["set"] == "test"
    ax.axhline(e10[("cv", 0)][test].mean(), color=COLORS["cv"], linestyle="--", linewidth=1.5)
    ax.text(NS[0], e10[("cv", 0)][test].mean() + 0.004, NAMES["cv"], color=INK_2, fontsize=9)
    for m in ("gp", "tabpfn"):
        ys = [e10[(m, n)][test].mean() for n in NS]
        ax.plot(NS, ys, "o-", color=COLORS[m], linewidth=2, markersize=6, label=NAMES[m])
        ax.annotate(f"{ys[-1]:.3f} m", (NS[-1], ys[-1]), xytext=(6, -3), textcoords="offset points",
                    color=INK, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(NS, [str(n) for n in NS])
    ax.set_ylim(0, None)
    ax.set_xlabel("training samples N", color=INK_2, fontsize=9)
    ax.set_ylabel("mean position error at +1 s (m)", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)

    # 2. generalization to unseen patterns (N = 1000)
    ax = axes[1]
    style(ax, "Held-out and unseen patterns (N = 1000)")
    models = [("cv", 0), ("gp", 1000), ("tabpfn", 1000)]
    x = np.arange(len(EVAL_SETS))
    w = 0.26
    for j, key in enumerate(models):
        vals = [e10[key][E["set"] == s].mean() for s in EVAL_SETS]
        bars = ax.bar(x + (j - 1) * w, vals, w - 0.02, color=COLORS[key[0]], label=NAMES[key[0]])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.2f}", ha="center", color=INK, fontsize=8)
    ax.set_xticks(x, [SET_NAMES[s] for s in EVAL_SETS])
    ax.set_ylabel("mean position error at +1 s (m)", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)

    # 3. by upcoming behaviour (test set, N = 1000)
    ax = axes[2]
    style(ax, "By what the AGV does next (test set, N = 1000)")
    x = np.arange(len(BEHAVIOURS))
    for j, key in enumerate(models):
        vals = [e10[key][test & (E["behaviour"] == b)].mean() for b in BEHAVIOURS]
        ax.bar(x + (j - 1) * w, vals, w - 0.02, color=COLORS[key[0]], label=NAMES[key[0]])
    ax.set_xticks(x, [LABEL[b].replace(" at ", "\nat ").replace(" for ", "\nfor ").replace(" / ", " /\n")
                      .replace("cruising ", "cruising\n") for b in BEHAVIOURS])
    ax.set_ylabel("mean position error at +1 s (m)", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)

    fig.tight_layout()
    out = MOTION / "prediction_errors.png"
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    print(f"Saved {out}")

    print("\nError at +1 s by behaviour and set (N = 1000): cv / gp / tabpfn")
    for s in EVAL_SETS:
        print(f"  {s}")
        for b in BEHAVIOURS:
            mask = (E["set"] == s) & (E["behaviour"] == b)
            if mask.sum() >= 10:
                print(f"    {LABEL[b]:<26} n={mask.sum():>4}   " +
                      " / ".join(f"{e10[k][mask].mean():.3f}" for k in models))


if __name__ == "__main__":
    main()
