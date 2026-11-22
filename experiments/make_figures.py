"""Make all static figures from results/*.csv.

    python experiments/make_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

COLORS = {
    "physics_nominal": "#9aa0a6", "physics_sysid": "#3b6fb6", "reactive": "#7f7f7f",
    "gp": "#f2b134", "gp_res": "#e8a33d", "gp_physfeat": "#d9822b", "gp_hybrid": "#c0602a",
    "gp_transfer": "#8c4a1f",
    "tabpfn": "#9ad1a5", "tabpfn_res": "#5cb874", "tabpfn_physfeat": "#2e9e57", "tabpfn_hybrid": "#127a3d",
    "tabpfn_transfer": "#0b4f28", "tabpfn_transfer_raw": "#6a3d9a",
    "tabpfn_hybrid_active": "#b5179e", "tabpfn_transfer_hybrid": "#00a6a6", "gp_transfer_hybrid": "#a0522d", "physics_sysid_active": "#7209b7", "gp_hybrid_active": "#f72585",
}
LABELS = {
    "physics_nominal": "Physics (nominal, no learning)", "physics_sysid": "Physics + online sysID (A)",
    "reactive": "Adaptive reactive (B)", "gp": "GP (C)", "gp_res": "GP residual", "gp_physfeat": "GP + physics feats",
    "gp_hybrid": "GP hybrid (sysID + GP residual)", "gp_transfer": "GP transfer",
    "tabpfn": "TabPFN (D)", "tabpfn_res": "TabPFN residual", "tabpfn_physfeat": "TabPFN + physics feats",
    "tabpfn_hybrid": "TabPFN hybrid (sysID + TabPFN residual)", "tabpfn_transfer": "TabPFN transfer (F)",
    "tabpfn_transfer_raw": "TabPFN transfer (no physics)",
    "tabpfn_transfer_hybrid": "TabPFN transfer-hybrid (F)", "gp_transfer_hybrid": "GP transfer-hybrid", "tabpfn_hybrid_active": "TabPFN hybrid + active (E)",
}

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


def ci(x, n_boot=1000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = rng.choice(x, (n_boot, len(x))).mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def learning_curves(csvs, title, out, models=None):
    df = pd.concat([pd.read_csv(c) for c in csvs if Path(c).exists()], ignore_index=True)
    if df.empty:
        return
    df = df.drop_duplicates(["model", "n", "obj", "seed"])
    models = [m for m in (models or COLORS) if m in df.model.unique()]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for m in models:
        d = df[df.model == m]
        for ax, col in zip(axes, ["pos_mm", "ang_deg", "nll"]):
            g = d.groupby("n")[col]
            mu = g.median() if col == "nll" else g.mean()
            lo_hi = np.array([ci(v) for _, v in g])
            x = mu.index.values
            xs = np.where(x == 0, 0.5, x)
            ax.plot(xs, mu.values, "-o", ms=3, color=COLORS.get(m), label=LABELS.get(m, m),
                    lw=2.2 if "tabpfn" in m or m == "physics_sysid" else 1.3)
            ax.fill_between(xs, lo_hi[:, 0], lo_hi[:, 1], color=COLORS.get(m), alpha=0.12)
    for ax, lab in zip(axes, ["position error [mm]", "orientation error [deg]", "median Gaussian NLL (lower=better)"]):
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 3, 5, 10, 20, 40])
        ax.set_xticklabels(["0", "1", "2", "3", "5", "10", "20", "40"])
        ax.set_xlabel("pushes observed on the new object")
        ax.set_ylabel(lab)
    axes[2].set_ylim(-11, 0)
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    # table
    cols = [c for c in ["pos_mm", "ang_deg", "nll", "cov1", "pred_s"] if c in df]
    tab = df[df.model.isin(models)].groupby(["model", "n"])[cols].mean()
    tab.round(3).to_csv(str(out).replace(".png", ".csv"))


def auc_table(csvs, out):
    df = pd.concat([pd.read_csv(c) for c in csvs if Path(c).exists()], ignore_index=True)
    if df.empty:
        return None
    df = df.drop_duplicates(["model", "n", "obj", "seed"])
    ns = [1, 2, 3, 5, 10, 20]
    d = df[df.n.isin(ns)]
    # normalised error: pos/10mm + ang/5deg, averaged over the n-grid (area under learning curve)
    d = d.assign(score=d.pos_mm / 10 + d.ang_deg / 5)
    per_obj = d.groupby(["model", "obj", "seed"])["score"].mean().reset_index()
    rows = []
    ref = per_obj[per_obj.model == "physics_sysid"].set_index(["obj", "seed"])["score"]
    for m, g in per_obj.groupby("model"):
        g = g.set_index(["obj", "seed"])["score"]
        common = g.index.intersection(ref.index)
        diff = (g[common] - ref[common]).values
        lo, hi = ci(diff)
        rows.append(dict(model=m, auc_score=g.mean(), vs_physics_sysid=diff.mean(), ci_lo=lo, ci_hi=hi,
                         wins=(diff < 0).mean(), n_obj=len(common)))
    tab = pd.DataFrame(rows).sort_values("auc_score")
    tab.round(3).to_csv(out, index=False)
    return tab


def manip_figure(csv, out):
    if not Path(csv).exists():
        return
    df = pd.read_csv(csv)
    if "error" in df:
        df = df[df["error"].isna()] if df["error"].notna().any() else df
    chs = sorted(df.challenge.unique())
    methods = [m for m in COLORS if m in df.method.unique()]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    w = 0.8 / max(len(methods), 1)
    for j, m in enumerate(methods):
        sr, lo, hi, pushes, perr = [], [], [], [], []
        for c in chs:
            d = df[(df.challenge == c) & (df.method == m)]
            s = d.success.astype(float).values
            sr.append(s.mean() if len(s) else np.nan)
            l, h = ci(s) if len(s) > 1 else (np.nan, np.nan)
            lo.append(l); hi.append(h)
            pushes.append(d[d.success].pushes.mean() if d.success.any() else np.nan)
            perr.append(d.pos_err_mm.median() if len(d) else np.nan)
        x = np.arange(len(chs)) + j * w
        axes[0].bar(x, sr, w, color=COLORS[m], label=LABELS.get(m, m))
        axes[0].errorbar(x, sr, yerr=[np.array(sr) - np.array(lo), np.array(hi) - np.array(sr)], fmt="none",
                         ecolor="k", lw=0.7, capsize=2)
        axes[1].bar(x, pushes, w, color=COLORS[m])
        axes[2].bar(x, perr, w, color=COLORS[m])
    for ax, lab in zip(axes, ["success rate (95% bootstrap CI)", "pushes to success (successful runs)",
                              "median final position error [mm]"]):
        ax.set_xticks(np.arange(len(chs)) + 0.4 - w / 2)
        ax.set_xticklabels(chs)
        ax.set_ylabel(lab)
    axes[2].set_yscale("log")
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def active_figure(csv, out):
    if not Path(csv).exists():
        return
    df = pd.read_csv(csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (m, s), d in df.groupby(["model", "strategy"]):
        for ax, col in zip(axes, ["pos_mm", "ang_deg"]):
            g = d.groupby("n")[col].mean()
            ax.plot(g.index, g.values, "-" if s == "uncertainty" else "--", color=COLORS.get(m), lw=2,
                    label=f"{LABELS.get(m, m)} - {s}")
    axes[0].set_ylabel("position error [mm]")
    axes[1].set_ylabel("orientation error [deg]")
    for ax in axes:
        ax.set_xlabel("pushes chosen so far")
    axes[0].legend(fontsize=7)
    fig.suptitle("Active experimentation: uncertainty-driven vs random push selection")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    learning_curves([RES / "pred_std_fast.csv", RES / "pred_std_tabpfn.csv"],
                    "Challenge 2 - standard regime (quasi-static pushes): few-shot prediction on unseen objects",
                    FIG / "learning_curves_standard.png")
    learning_curves([RES / "pred_hetero.csv"],
                    "Heterogeneous regime (variable push speed, caster-wheeled objects): few-shot prediction",
                    FIG / "learning_curves_hetero.png")
    head = ["physics_nominal", "physics_sysid", "gp_hybrid", "tabpfn_hybrid"]
    learning_curves([RES / "pred_std_fast.csv", RES / "pred_std_tabpfn.csv"],
                    "Standard regime: headline comparison", FIG / "headline_standard.png", models=head)
    learning_curves([RES / "pred_hetero.csv"], "Heterogeneous regime: headline comparison",
                    FIG / "headline_hetero.png", models=head)
    learning_curves([RES / "a1_prediction.csv"], "Unitree A1 whole-body pushes (3D, leave-one-crate-out)",
                    FIG / "a1_prediction.png")
    learning_curves([RES / "pred_transfer_std.csv", RES / "pred_std_fast.csv", RES / "pred_std_tabpfn.csv"],
                    "Challenge 7 - transfer to unseen objects (standard regime)", FIG / "transfer_standard.png",
                    models=["physics_nominal", "physics_sysid", "gp_hybrid", "tabpfn_hybrid", "gp_transfer",
                            "tabpfn_transfer", "tabpfn_transfer_raw"])
    learning_curves([RES / "pred_transfer_het.csv", RES / "pred_hetero.csv"],
                    "Challenge 7 - transfer to unseen objects (heterogeneous regime)", FIG / "transfer_hetero.png",
                    models=["physics_nominal", "physics_sysid", "gp_hybrid", "tabpfn_hybrid", "gp_transfer",
                            "tabpfn_transfer", "tabpfn_transfer_raw", "gp_transfer_hybrid", "tabpfn_transfer_hybrid"])
    for name, csvs in [("standard", [RES / "pred_std_fast.csv", RES / "pred_std_tabpfn.csv", RES / "pred_transfer_std.csv"]),
                       ("hetero", [RES / "pred_hetero.csv", RES / "pred_transfer_het.csv"])]:
        t = auc_table(csvs, RES / f"auc_{name}.csv")
        if t is not None:
            print(name)
            print(t.to_string(index=False))
    for tag in ["v1", "v2", "hetero"]:
        manip_figure(RES / f"manip_{tag}.csv", FIG / f"manipulation_{tag}.png")
    active_figure(RES / "active_htest.csv", FIG / "active_hetero.png")
    active_figure(RES / "active_test.csv", FIG / "active_standard.png")
    print("figures ->", FIG)


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


def manip_stats(csvs, out, ref="physics_sysid"):
    """Paired (same task seed) comparisons of every method against `ref`."""
    from scipy.stats import wilcoxon
    df = pd.concat([pd.read_csv(c) for c in csvs if Path(c).exists()], ignore_index=True)
    rows = []
    for scope, d in [("pooled", df)] + list(df.groupby("challenge")):
        for m in d.method.unique():
            if m == ref:
                continue
            a = d[d.method == m].set_index(["challenge", "seed"])
            b = d[d.method == ref].set_index(["challenge", "seed"])
            idx = a.index.intersection(b.index)
            if len(idx) < 3:
                continue
            a, b = a.loc[idx], b.loc[idx]
            r = dict(scope=scope, method=m, ref=ref, n=len(idx), success=a.success.mean(), ref_success=b.success.mean())
            for col in ["pushes", "wall_contacts"]:
                diff = (a[col] - b[col]).values.astype(float)
                r[f"d_{col}"] = diff.mean()
                r[f"p_{col}"] = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            rows.append(r)
    t = pd.DataFrame(rows)
    t.round(4).to_csv(out, index=False)
    return t


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "stats":
    pd.set_option("display.width", 220)
    print(manip_stats([RES / "manip_v2.csv"], RES / "manip_stats_v2.csv").round(3).to_string())
    print(manip_stats([RES / "manip_hetero.csv", RES / "manip_hetero_transfer.csv"], RES / "manip_stats_hetero.csv").round(3).to_string())


def paired_figure(csvs, out, title, models, ref="physics_sysid", ns=(0, 1, 2, 3, 5, 10, 20, 40)):
    df = pd.concat([pd.read_csv(c) for c in csvs if Path(c).exists()], ignore_index=True)
    df = df.drop_duplicates(["model", "n", "obj", "seed"])
    df["score"] = df.pos_mm / 10 + df.ang_deg / 5
    piv = df.pivot_table(index=["obj", "seed", "n"], columns="model", values="score")
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for k, m in enumerate(models):
        xs, mu, lo, hi = [], [], [], []
        for n in ns:
            try:
                p = piv.xs(n, level="n")
            except KeyError:
                continue
            if m not in p or ref not in p:
                continue
            d = (p[m] - p[ref]).dropna().values
            if len(d) < 3:
                continue
            l, h = ci(d, 2000)
            xs.append(n); mu.append(d.mean()); lo.append(l); hi.append(h)
        xs = np.array(xs, float)
        xp = np.where(xs == 0, 0.5, xs) * (1 + 0.04 * (k - len(models) / 2))
        ax.errorbar(xp, mu, yerr=[np.array(mu) - lo, np.array(hi) - mu], fmt="-o", ms=4, capsize=3,
                    color=COLORS.get(m), label=LABELS.get(m, m), lw=2)
    ax.axhline(0, color=COLORS[ref], lw=2, ls="--", label=LABELS[ref] + " (reference)")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 3, 5, 10, 20, 40]); ax.set_xticklabels(["0", "1", "2", "3", "5", "10", "20", "40"])
    ax.set_xlabel("pushes observed on the new object")
    ax.set_ylabel("error score difference vs reference\n(pos/10mm + ang/5deg; below 0 = better)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "paired":
    paired_figure([RES / "pred_hetero.csv", RES / "pred_transfer_het.csv"], FIG / "paired_hetero.png",
                  "Heterogeneous regime, 24 unseen objects: paired difference to physics + online sysID (95% CI)",
                  ["gp_hybrid", "tabpfn_hybrid", "gp_transfer_hybrid", "tabpfn_transfer_hybrid"])
    paired_figure([RES / "pred_std_fast.csv", RES / "pred_std_tabpfn.csv"], FIG / "paired_standard.png",
                  "Standard regime, 24 unseen objects: paired difference to physics + online sysID (95% CI)",
                  ["gp_hybrid", "tabpfn_hybrid"])
