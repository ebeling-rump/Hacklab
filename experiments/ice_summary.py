"""Challenge 8 (ice): paired summary and figure from results/pred_ice*.csv.

    python experiments/ice_summary.py  -> results/ice_paired.csv, results/figures/ice_prediction.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
NS = [3, 5, 10, 20, 40]
SHOW = [("physics_sysid", "Physics + online sysID", "#46536A", "--"),
        ("tabpfn_hybrid", "TabPFN hybrid, own pushes only", "#9BB7D4", ":"),
        ("gp_transfer_hybrid", "GP + experience (20 other icy floors)", "#A0522D", "-"),
        ("tabpfn_transfer_hybrid_blind", "TabPFN + experience, ice inputs hidden", "#7FC8E8", "-."),
        ("tabpfn_transfer_hybrid", "TabPFN v2 + experience + visible ice", "#3B8ED8", "-"),
        ("tabpfn_api_transfer_hybrid", "TabPFN v3.5 API + experience + visible ice", "#F08A24", "-")]
PAIRS = [("tabpfn_api_transfer_hybrid", "physics_sysid"), ("tabpfn_transfer_hybrid", "physics_sysid"),
         ("tabpfn_transfer_hybrid", "gp_transfer_hybrid"), ("tabpfn_transfer_hybrid", "tabpfn_transfer_hybrid_blind"),
         ("tabpfn_api_transfer_hybrid", "tabpfn_transfer_hybrid")]


def main():
    t = pd.read_csv(R / "pred_ice_transfer.csv")
    p = pd.read_csv(R / "pred_ice.csv")
    d = pd.concat([t, p[p.obj.isin(t.obj.unique())]])
    n_obj = t.obj.nunique()
    rng = np.random.default_rng(0)
    rows = []
    for a, b in PAIRS:
        for n in NS:
            x = d[(d.model == a) & (d.n == n)].set_index("obj").pos_mm
            y = d[(d.model == b) & (d.n == n)].set_index("obj").pos_mm
            c = x.index.intersection(y.index)
            diff = (x[c] - y[c]).values
            bs = rng.choice(diff, (4000, len(diff))).mean(1)
            rows.append(dict(a=a, b=b, n=n, objects=len(c), a_mm=x[c].mean(), b_mm=y[c].mean(), diff_mm=diff.mean(),
                             ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), a_better=(diff < 0).mean()))
    df = pd.DataFrame(rows)
    df.round(3).to_csv(R / "ice_paired.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(2).to_string())

    fig, ax = plt.subplots(figsize=(11, 6))
    for m, lab, c, ls in SHOW:
        s = d[(d.model == m) & d.n.isin(NS)].groupby("n").pos_mm.mean().reindex(NS)
        ax.plot(range(len(NS)), s.values, ls, color=c, lw=3, marker="o", label=lab)
    ax.set_xticks(range(len(NS)), [str(n) for n in NS])
    ax.set_xlabel("own pushes on the new object (on a new icy floor)")
    ax.set_ylabel("position prediction error [mm]")
    ax.set_title(f"Challenge 8, visible ice of unknown slipperiness: {n_obj} unseen objects and floors")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(R / "figures" / "ice_prediction.png", dpi=130)
    print("objects:", n_obj, "->", R / "figures" / "ice_prediction.png")


if __name__ == "__main__":
    main()
