"""Summaries and figures from results/*.csv and results/nav_paths.npz."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from terrain_sim import N, TERRAIN, make_site

OUT = Path("results")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Reference categorical palette, fixed slot order.
SERIES = {
    "TabPFN v2": "#2a78d6",
    "CatBoost": "#eb6834",
    "XGBoost (tuned)": "#1baf7a",
    "Random forest": "#eda100",
    "Cross-site + local XGBoost": "#e87ba4",
    "TabPFN v2 + archive": "#008300",
}
LOCAL_ONLY = ["TabPFN v2", "CatBoost", "XGBoost (tuned)", "XGBoost", "LightGBM", "Random forest", "MLP",
              "Logistic regression"]
REFS = {"Geometric rule": "#52514e", "Cross-site XGBoost": "#8d8b85"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e2dc"
TERRAIN_CM = ListedColormap(["#c8a27a", "#9bc26b", "#5f8f3a", "#2f5a2a", "#6b4a2e", "#e6d39a", "#8c8c8c", "#111111"])

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 11, "axes.titlecolor": INK,
})


def learning_curves():
    df = pd.read_csv(OUT / "learning_curves.csv")
    if (OUT / "learning_curves_archive.csv").exists():
        df = pd.concat([df, pd.read_csv(OUT / "learning_curves_archive.csv")])
    df = df[df.rep == 0]  # one calibration draw per site (two sites also have a second draw; unused)
    per_site = df.groupby(["model", "n", "site"], as_index=False)[["auc", "logloss", "brier", "ece", "fit_s"]].mean()
    g = per_site.groupby(["model", "n"])
    summ = g.mean(numeric_only=True).drop(columns="site")
    se = g.sem(numeric_only=True).drop(columns="site").add_suffix("_se")
    # fit times: site 1001 partly ran while navigate.py competed for the 2 CPU cores, so its timings are inflated
    clean = per_site[per_site.site != 1001].groupby(["model", "n"])["fit_s"].mean()
    summ["fit_s"] = clean
    summ = summ.join(se.drop(columns="fit_s_se")).reset_index()
    summ.to_csv(OUT / "lc_summary.csv", index=False)

    # rank of each local-only model per (site, n) by log loss
    local = df[df.model.isin(LOCAL_ONLY)].copy()
    local["rank"] = local.groupby(["site", "rep", "n"])["logloss"].rank()
    wins = local.assign(win=local["rank"] == 1).groupby(["model", "n"])["win"].mean().unstack("n")
    wins.to_csv(OUT / "lc_win_rate.csv")
    ranks = local.groupby(["model", "n"])["rank"].mean().unstack("n")
    ranks.to_csv(OUT / "lc_mean_rank.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric, label in [(axes[0], "logloss", "Log loss on held-out cells (lower is better)"),
                              (axes[1], "auc", "AUROC on held-out cells (higher is better)")]:
        for name, col in REFS.items():
            v = summ[(summ.model == name)][metric].iloc[0]
            ax.axhline(v, color=col, lw=1.2, ls="--", zorder=1)
            ax.text(1024, v, f"{name} ", color=INK2, va="bottom", ha="right", fontsize=8)
        for name, col in SERIES.items():
            s = summ[summ.model == name]
            if s.empty:
                continue
            ax.plot(s.n, s[metric], color=col, lw=2, marker="o", ms=4, label=name, zorder=3)
            ax.fill_between(s.n, s[metric] - s[metric + "_se"], s[metric] + s[metric + "_se"], color=col, alpha=0.12, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks([32, 64, 128, 256, 512, 1024], ["32", "64", "128", "256", "512", "1024"])
        ax.set_xlabel("Self-labelled cells from this site's calibration loop")
        ax.set_title(label, loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="lower center", ncol=6)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "learning_curves.png", dpi=150)
    plt.close(fig)
    return summ, wins, ranks


def navigation():
    df = pd.read_csv(OUT / "navigation.csv")
    order = ["No terrain model", "Geometric rule", "Cross-site XGBoost", "Cross-site + local XGBoost",
             "CatBoost", "XGBoost (tuned)", "TabPFN v2", "Oracle"]
    rows = []
    for m in order:
        d = df[df.method == m]
        ok = d[d.outcome == "success"]
        rows.append(dict(method=m, missions=len(d), success=(d.outcome == "success").mean(),
                         stuck=(d.outcome == "stuck").mean(), no_path=(d.outcome == "no path").mean(),
                         path_m_success=ok.path_m.mean(), detour_success=(ok.path_m / ok.straight_m).mean()))
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "nav_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    cols = ["#2a78d6" if m == "TabPFN v2" else "#b9b7b0" for m in order]
    y = np.arange(len(order))[::-1]
    ax.barh(y, summ.success * 100, color=cols, height=0.6)
    for yi, v in zip(y, summ.success):
        ax.text(v * 100 + 1, yi, f"{v:.0%}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, order)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Missions completed without getting stuck (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Navigation on unseen terrain, {int(summ.missions.iloc[0])} missions per method", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "navigation.png", dpi=150)
    plt.close(fig)
    return summ


def example_maps(site_id=None, episode=None):
    z = np.load(OUT / "nav_paths.npz")
    nav = pd.read_csv(OUT / "navigation.csv")
    if site_id is None:
        # pick a mission where TabPFN succeeds and the geometric rule or cross-site model does not
        piv = nav.pivot_table(index=["site", "episode"], columns="method", values="outcome", aggfunc="first")
        lens = nav.pivot_table(index=["site", "episode"], columns="method", values="path_m", aggfunc="first")
        far = (lens[["Geometric rule", "Cross-site XGBoost", "XGBoost (tuned)"]] > 12).all(axis=1)
        t = (piv["TabPFN v2"] == "success") & far
        others = [piv[m] != "success" for m in ["Geometric rule", "Cross-site XGBoost", "XGBoost (tuned)"]]
        for cond in [t & others[0] & others[1] & others[2], t & others[0] & others[1], t & (others[0] | others[1]), t]:
            if cond.any():
                site_id, episode = piv[cond].index[0]
                break
    site = make_site(int(site_id))
    show = ["Geometric rule", "Cross-site XGBoost", "XGBoost (tuned)", "TabPFN v2"]
    fig, axes = plt.subplots(1, 6, figsize=(18, 4.4), layout="constrained")
    sl = np.s_[:, N // 2:]
    terr = site.terrain[sl]
    axes[0].imshow(terr, cmap=TERRAIN_CM, vmin=0, vmax=7, origin="lower")
    axes[0].set_title("Terrain (truth)", loc="left")
    oracle = z[f"{site_id}|risk|Oracle"][sl]
    axes[1].imshow(oracle, cmap="magma", vmin=0, vmax=1, origin="lower")
    axes[1].set_title("True high-slip probability", loc="left")
    ends = z[f"{site_id}|{episode}|endpoints"]
    for ax, m in zip(axes[2:], show):
        im = ax.imshow(z[f"{site_id}|risk|{m}"][sl], cmap="magma", vmin=0, vmax=1, origin="lower")
        p = z[f"{site_id}|{episode}|{m}"]
        out = nav[(nav.site == site_id) & (nav.episode == episode) & (nav.method == m)].outcome.iloc[0]
        ax.plot(p[:, 1] - N // 2, p[:, 0], color="#1baf7a", lw=2)
        if out == "stuck":
            ax.plot(p[-1, 1] - N // 2, p[-1, 0], "x", color="#ffffff", ms=11, mew=3)
        ax.set_title(f"{m}: {out}", loc="left")
    for ax in axes:
        ax.plot(ends[:, 1] - N // 2, ends[:, 0], "o", color="#ffffff", ms=6, mec=INK)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01, label="P(high slip)")
    fig.savefig(FIG / "example_maps.png", dpi=130)
    plt.close(fig)
    return int(site_id), int(episode)


if __name__ == "__main__":
    s, w, r = learning_curves()
    print(s.pivot(index="model", columns="n", values="logloss").round(3))
    print(w.round(2))
    if (OUT / "navigation.csv").exists():
        print(navigation().round(3))
        print("example:", example_maps())
