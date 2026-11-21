"""Figures from the cached evaluation results (data/results/*.json).

    python plot_results.py            -> data/summary.png, data/per_bottle.png, data/episode_<seed>.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim import OOD_SEEDS, OPEN_TURN, TEST_SEEDS

DATA = Path(__file__).parent / "data"
RES = DATA / "results"
FIG = Path(__file__).parent / "results"     # tracked figures
ORDER = ["classical", "tabpfn_prior_only", "tabpfn", "oracle"]
NAMES = {"classical": "classical\nadaptive", "tabpfn_prior_only": "TabPFN\nprior only",
         "tabpfn": "TabPFN\nclosed loop", "oracle": "oracle\n(knows bottle)"}
COLORS = {"classical": "#52514e", "tabpfn_prior_only": "#86b6ef", "tabpfn": "#2a78d6", "oracle": "#1baf7a"}
INK, MUTED, GRID = "#262624", "#6b6a64", "#e4e3dd"


def load(name, seeds=TEST_SEEDS):
    return {s: json.loads((RES / f"{name}_{s}.json").read_text()) for s in seeds
            if (RES / f"{name}_{s}.json").exists()}


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def summary(results, tag="test"):
    names = [n for n in ORDER if results.get(n)]
    panels = [
        ("strokes to open", lambda rs: np.mean([r["strokes"] for r in rs if r["opened"]]), "{:.1f}"),
        ("simulated time to open [s]", lambda rs: np.mean([r["time"] for r in rs if r["opened"]]), "{:.1f}"),
        ("exploratory strokes\n(no cap progress)", lambda rs: np.mean([r["exploratory"] for r in rs]), "{:.1f}"),
        ("bottles damaged\n(excessive grip force)", lambda rs: 100 * np.mean([r["damage"] > 0 for r in rs]), "{:.0f}%"),
        ("peak grip / needed grip\n(median)", lambda rs: np.median([r["force_ratio"] for r in rs]), "{:.2f}"),
        ("opened [%]", lambda rs: 100 * np.mean([r["opened"] for r in rs]), "{:.0f}%"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    for ax, (title, fn, fmt) in zip(axes.ravel(), panels):
        vals = [fn(list(results[n].values())) for n in names]
        x = np.arange(len(names))
        ax.bar(x, vals, width=0.62, color=[COLORS[n] for n in names], edgecolor="white", linewidth=2)
        for xi, v in zip(x, vals):
            ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(x, [NAMES[n] for n in names], fontsize=7.5, color=INK)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        style(ax)
    n = len(next(iter(results.values())))
    label = "out-of-distribution (slippery caps)" if tag == "ood" else "in-distribution"
    fig.suptitle(f"Opening {n} unseen bottles, {label}: same physics, same safety layer, different decision-maker",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / f"summary_{tag}.png", dpi=150)
    print("wrote", FIG / f"summary_{tag}.png")


def per_bottle(results, tag="test"):
    if not (results.get("classical") and results.get("tabpfn")):
        return
    seeds = sorted(set(results["classical"]) & set(results["tabpfn"]))
    c = np.array([results["classical"][s]["strokes"] for s in seeds])
    t = np.array([results["tabpfn"][s]["strokes"] for s in seeds])
    fig, ax = plt.subplots(figsize=(5, 4.6))
    lim = max(c.max(), t.max()) + 1
    ax.plot([0, lim], [0, lim], color=GRID, lw=1.5, zorder=0)
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, (2, len(seeds)))
    dmg = np.array([results["tabpfn"][s]["damage"] > 0 or results["classical"][s]["damage"] > 0 for s in seeds])
    ax.scatter(c + jitter[0], t + jitter[1], s=36, color=COLORS["tabpfn"], edgecolor="white", linewidth=1.5,
               label="bottle")
    ax.scatter((c + jitter[0])[dmg], (t + jitter[1])[dmg], s=80, facecolor="none", edgecolor="#e34948",
               linewidth=1.2, label="damaged by either controller")
    ax.set_xlabel("strokes, classical adaptive", fontsize=9, color=INK)
    ax.set_ylabel("strokes, TabPFN closed loop", fontsize=9, color=INK)
    better = np.mean(t < c); worse = np.mean(t > c)
    ax.set_title(f"Per bottle: TabPFN fewer strokes on {better:.0%}, more on {worse:.0%}",
                 fontsize=9.5, color=INK, loc="left")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / f"per_bottle_{tag}.png", dpi=150)
    print("wrote", FIG / f"per_bottle_{tag}.png")


def episode(results, seed):
    names = [n for n in ("classical", "tabpfn", "oracle") if seed in results.get(n, {})]
    b = results[names[0]][seed]["bottle"]
    f_need = b["t_break"] / (2 * b["mu"] * 0.016)
    f_crush = b["k_wall"] * b["d_lim"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for n in names:
        h = results[n][seed]["history"]
        k = np.arange(1, len(h) + 1)
        kw = dict(color=COLORS[n], lw=2, marker="o", ms=4, label=NAMES[n].replace("\n", " "))
        axes[0].plot(k, np.cumsum([o["progress"] for o in h]), **kw)
        axes[1].plot(k, [o["fn"] for o in h], **kw)
        axes[2].plot(k, [o["tau"] for o in h], **kw)
        for o, ki in zip(h, k):
            if o["slip"] > 5:
                axes[1].plot(ki, o["fn"], "x", color="#e34948", ms=7)
    axes[0].axhline(np.rad2deg(OPEN_TURN), color=MUTED, ls="--", lw=1)
    axes[0].text(1, np.rad2deg(OPEN_TURN) + 8, "cap off", fontsize=8, color=MUTED)
    axes[1].axhline(f_crush, color="#e34948", ls="--", lw=1)
    axes[1].text(1, f_crush, "crush limit (hidden)", fontsize=8, color="#e34948", va="top")
    axes[1].axhline(f_need, color=MUTED, ls=":", lw=1)
    axes[1].text(1, f_need, "grip needed at breakaway (hidden)", fontsize=8, color=MUTED, va="top")
    axes[2].axhline(b["t_break"], color=MUTED, ls=":", lw=1)
    axes[2].text(1, b["t_break"] + 0.03, "breakaway torque (hidden)", fontsize=8, color=MUTED)
    axes[2].axhline(b["t_run"], color=MUTED, ls=":", lw=1)
    axes[2].text(1, b["t_run"] + 0.03, "running torque (hidden)", fontsize=8, color=MUTED)
    for ax, t in zip(axes, ["cap rotation [deg]", "grip force [N]  (x = slipped)", "wrist torque [Nm]"]):
        ax.set_title(t, fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("stroke", fontsize=8.5, color=MUTED)
        style(ax)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Test bottle {seed}: mu={b['mu']:.2f}, breakaway {b['t_break']:.2f} Nm, "
                 f"wall {b['k_wall']:.0f} N/mm, torsion {b['k_tors']:.1f} Nm/rad",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / f"episode_{seed}.png", dpi=150)
    print("wrote", FIG / f"episode_{seed}.png")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    every = {}
    for tag, seeds in (("test", TEST_SEEDS), ("ood", OOD_SEEDS)):
        results = {n: load(n, seeds) for n in ORDER}
        results = {n: r for n, r in results.items() if r}
        if results:
            summary(results, tag)
            per_bottle(results, tag)
        for n, r in results.items():
            every.setdefault(n, {}).update(r)
    for s in [int(a) for a in sys.argv[1:]] or [list(TEST_SEEDS)[0]]:
        episode(every, s)
