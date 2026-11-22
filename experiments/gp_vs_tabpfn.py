"""Head-to-head: TabPFN vs Gaussian process with identical inputs, paired per object.

Reads the prediction CSVs already in results/ and writes
results/gp_vs_tabpfn.csv and results/figures/gp_vs_tabpfn.png.

    .venv\\Scripts\\python experiments/gp_vs_tabpfn.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
NS = [1, 3, 5, 10, 20]
PAIRS = [  # (regime, label, gp model, tabpfn model, csv files)
    ("standard", "raw learner", "gp", "tabpfn", ["pred_std_fast", "pred_std_tabpfn"]),
    ("standard", "hybrid (physics + residual)", "gp_hybrid", "tabpfn_hybrid", ["pred_std_fast", "pred_std_tabpfn"]),
    ("heterogeneous", "raw learner", "gp", "tabpfn", ["pred_hetero"]),
    ("heterogeneous", "hybrid (physics + residual)", "gp_hybrid", "tabpfn_hybrid", ["pred_hetero"]),
    ("heterogeneous", "transfer-hybrid (+40 other objects)", "gp_transfer_hybrid", "tabpfn_transfer_hybrid",
     ["pred_transfer_het"]),
]


def boot(x, rng, k=4000):
    b = rng.choice(x, (k, len(x))).mean(1)
    return np.percentile(b, [2.5, 97.5])


def main():
    rng = np.random.default_rng(0)
    rows = []
    for regime, label, g, t, files in PAIRS:
        d = pd.concat([pd.read_csv(R / f"{f}.csv") for f in files])
        for n in NS:
            a = d[(d.model == g) & (d.n == n)].set_index("obj")
            b = d[(d.model == t) & (d.n == n)].set_index("obj")
            common = a.index.intersection(b.index)
            if len(common) == 0:
                continue
            dp = (b.loc[common, "pos_mm"] - a.loc[common, "pos_mm"]).values
            lo, hi = boot(dp, rng)
            rows.append(dict(regime=regime, model=label, n=n, objects=len(common),
                             gp_pos_mm=a.loc[common, "pos_mm"].mean(), tabpfn_pos_mm=b.loc[common, "pos_mm"].mean(),
                             diff_mm=dp.mean(), ci_lo=lo, ci_hi=hi,
                             tabpfn_wins=(dp < 0).mean() if np.any(dp != 0) else np.nan,  # all-tie: both fall back to physics
                             gp_nll_median=a.loc[common, "nll"].median(), tabpfn_nll_median=b.loc[common, "nll"].median(),
                             gp_cov1=a.loc[common, "cov1"].mean(), tabpfn_cov1=b.loc[common, "cov1"].mean()))
    df = pd.DataFrame(rows)
    df.round(3).to_csv(R / "gp_vs_tabpfn.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.round(2).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    sty = {"raw learner": ("#999999", "o"), "hybrid (physics + residual)": ("#1b7f3b", "s"),
           "transfer-hybrid (+40 other objects)": ("#00a6a6", "D")}
    for regime, ls in [("standard", "--"), ("heterogeneous", "-")]:
        for label, (c, m) in sty.items():
            s = df[(df.regime == regime) & (df.model == label)]
            if s.empty:
                continue
            x = np.array([NS.index(n) for n in s.n]) + (0.06 if regime == "heterogeneous" else -0.06)
            axes[0].errorbar(x, s.diff_mm, yerr=[s.diff_mm - s.ci_lo, s.ci_hi - s.diff_mm], color=c, marker=m,
                             ls=ls, capsize=4, label=f"{label}, {regime}")
            axes[1].plot(x, 100 * s.tabpfn_wins, color=c, marker=m, ls=ls)
            axes[2].plot(x, s.gp_cov1, color=c, marker=m, ls=":", alpha=0.6)
            axes[2].plot(x, s.tabpfn_cov1, color=c, marker=m, ls=ls)
    axes[0].axhline(0, color="k", lw=1)
    axes[0].set_title("Position error: TabPFN minus GP [mm]\n(paired per object, 95% CI; below 0 = TabPFN better)")
    axes[1].axhline(50, color="k", lw=1)
    axes[1].set_title("Objects where TabPFN is more accurate [%]")
    axes[2].axhline(0.683, color="k", lw=1)
    axes[2].set_title("Calibration: share of outcomes inside ±1σ\n(ideal 68%; solid TabPFN, dotted GP)")
    for ax in axes:
        ax.set_xticks(range(len(NS)), [str(n) for n in NS])
        ax.set_xlabel("own pushes on the new object")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("TabPFN vs Gaussian process with identical inputs (24 test objects per regime)", fontsize=13)
    fig.tight_layout()
    fig.savefig(R / "figures" / "gp_vs_tabpfn.png", dpi=130)
    print("->", R / "figures" / "gp_vs_tabpfn.png")


if __name__ == "__main__":
    main()
