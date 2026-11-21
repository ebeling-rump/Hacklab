"""Phase 3 summary chart from data/nav_eval.csv (no simulation, no API calls).

    python plot_phase3.py      # data/phase3_summary.png
"""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sim import DATA  # noqa: E402

ORDER = ["gp", "tabpfn_ideal", "tabpfn_live", "oracle"]
NAMES = {"constant_velocity": "constant\nvelocity", "gp": "GP\n(local)", "tabpfn_ideal": "TabPFN\nideal",
         "tabpfn_live": "TabPFN\nlive (cloud)", "oracle": "oracle\n(perfect)"}
COLORS = {"constant_velocity": "#52514e", "gp": "#eb6834", "tabpfn_ideal": "#2a78d6",
          "tabpfn_live": "#86b6ef", "oracle": "#1baf7a"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


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
    with (DATA / "nav_eval.csv").open() as f:
        rows = list(csv.DictReader(f))
    by = {}
    for r in rows:
        by.setdefault(r["predictor"], {})[int(r["seed"])] = r
    cv = by["constant_velocity"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), facecolor=SURFACE)

    # 1. time saved vs constant velocity on the same seeds, every seed as a dot
    ax = axes[0]
    style(ax, "Time to goal saved vs constant velocity\n(same seeds; dot = one seed, bar = mean)")
    rng = np.random.default_rng(0)
    for i, p in enumerate(ORDER):
        seeds = sorted(by[p])
        d = np.array([num(cv[s]["time_to_goal"]) - num(by[p][s]["time_to_goal"]) for s in seeds])
        d = d[~np.isnan(d)]
        ax.bar(i, d.mean(), 0.6, color=COLORS[p], alpha=0.35)
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(d)), d, s=14, color=COLORS[p], zorder=3)
        ax.text(i, -5.2, f"mean {d.mean():+.1f} s\nn={len(d)} seeds", ha="center", va="top",
                color=INK, fontsize=9)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_ylim(-7.0, 15.5)
    ax.set_xticks(range(len(ORDER)), [NAMES[p] for p in ORDER])
    ax.set_ylabel("seconds faster than constant velocity", color=INK_2, fontsize=9)

    # 2. online prediction error at +1 s
    ax = axes[1]
    style(ax, "Prediction error at +1 s, inside the control loop\n(AGVs within 3 m of the robot)")
    preds = ["constant_velocity", "tabpfn_live", "gp", "tabpfn_ideal", "oracle"]
    for i, p in enumerate(preds):
        v = np.nanmean([num(r["pred_err_10"]) for r in by[p].values()])
        ax.bar(i, v, 0.6, color=COLORS[p])
        ax.text(i, v + 0.003, f"{100 * v:.1f} cm", ha="center", color=INK, fontsize=9)
    ax.set_xticks(range(len(preds)), [NAMES[p] for p in preds])
    ax.set_ylabel("mean position error (m)", color=INK_2, fontsize=9)

    # 3. latency per prediction vs the time budget
    ax = axes[2]
    style(ax, "Latency per prediction vs time budget\n(log scale; bar = mean, tick = p95)")
    lat = ["gp", "tabpfn_ideal", "tabpfn_live"]
    for i, p in enumerate(lat):
        mean = np.nanmean([num(r["latency_mean_ms"]) for r in by[p].values()])
        p95 = np.nanmean([num(r["latency_p95_ms"]) for r in by[p].values()])
        ax.bar(i, mean, 0.55, color=COLORS[p])
        ax.plot([i - 0.2, i + 0.2], [p95, p95], color=INK, linewidth=2)
        ax.text(i, p95 * 1.25, f"{mean:,.0f} ms" if mean >= 10 else f"{mean:.1f} ms", ha="center",
                color=INK, fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(0.3, 20000)
    for y, label in ((100, "control period 100 ms"), (1000, "prediction horizon 1 s")):
        ax.axhline(y, color=INK_2, linestyle="--", linewidth=1)
        ax.text(-0.45, y * 1.12, label, ha="left", color=INK_2, fontsize=8)
    ax.set_xticks(range(len(lat)), [NAMES[p] for p in lat])
    ax.set_ylabel("milliseconds", color=INK_2, fontsize=9)

    fig.text(0.01, 0.005, "All predictors: 0% collisions, 100% goal reached (same local safety filter). "
             "TabPFN ideal: simulation waits for each cloud call; live: prediction usable only after its latency.",
             color=INK_2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = DATA / "phase3_summary.png"
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
