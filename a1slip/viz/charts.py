"""Charts of how well TabPFN predicts slips and falls (PNGs for the README and slides).

Recomputes predictions on the floors the models never saw and draws:
  fall_roc.png         falls within 1 s on unseen 13-15 deg slopes: ROC curves of
                       TabPFN, XGBoost, logistic regression, the 2 s tilt average
                       (a no-ML rule) and a model told the true floor
  fall_lead_time.png   TabPFN fall risk in the seconds before each fall vs walks
                       that never fell ("does it see it coming?")
  slip_roc.png         slips within 0.2 s on unseen icy + steep floors, with and
                       without the slip-detector history features
  learning_curve.png   slip ROC-AUC vs number of training rows (results/learning_curve.json)
  tradeoff.png         danger-mode trade-off incl. the balance reflex
                       (results/danger_mode.json, results/balancer_baseline.json)

    python -m a1slip.viz.charts      # -> results/figures/*.png, results/charts_data.json
"""

import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from a1slip.paths import ROOT
from a1slip.models.tabpfn_risk import CollisionRisk, context_subsample
from a1slip.prediction.slip_eval import DATA, fall_splits, splits

warnings.filterwarnings("ignore")
FIG = ROOT / "results" / "figures"
# Validated categorical palette (fixed order) + muted ink; every series also has a
# dash style and a direct label, so identity never relies on colour alone.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, SOFT, MUTED, GRID, SURF = "#1E2A3B", "#4A5568", "#898781", "#e1e0d9", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": SOFT, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 15, "axes.titlesize": 19, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False,
    "lines.linewidth": 2.5, "lines.solid_capstyle": "round",
})


def xgb(X, y):
    return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8).fit(X, y)


def roc_plot(ax, y, scores, title):
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.5, ls=(0, (2, 3)))
    ax.text(0.62, 0.55, "guessing", color=MUTED, fontsize=13, rotation=33)
    out = {}
    for name, (s, colour, ls) in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        out[name] = float(auc)
        ax.plot(fpr, tpr, color=colour, ls=ls, label=f"{name}  {auc:.3f}",
                lw=3.2 if name.startswith("TabPFN") else 2.2)
    ax.set_xlabel("False alarms (share of safe moments flagged)")
    ax.set_ylabel("Caught (share of real events flagged)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=14, title="ROC-AUC", title_fontsize=13)
    return out


def plot_lead_time(lt):
    c, med, q1, q3 = (np.array(lt[k]) for k in ("seconds_before", "median", "q1", "q3"))
    fig, ax = plt.subplots(figsize=(11, 6.4), dpi=120)
    ax.fill_between(c, q1, q3, color=BLUE, alpha=0.15, lw=0)
    ax.plot(c, med, color=BLUE, lw=3.2, marker="o", ms=7, mec=SURF, mew=2)
    ax.axhline(0.05, color=ORANGE, lw=1.8, ls=(0, (2, 3)), label="danger-mode threshold: 5%")
    ax.axhline(lt["never_fell_median"], color=MUTED, lw=2, ls=(0, (6, 3)),
               label=f"walks that never fell: median {lt['never_fell_median']:.1%}")
    ax.legend(loc="upper left", fontsize=13)
    for x, v in zip(c[:3], med[:3]):
        ax.text(x - 0.04, v, f"{v:.0%}", color=BLUE, fontsize=14, va="center", ha="left",
                fontweight="bold")
    ax.set_xlim(3.0, 0)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Seconds before the fall")
    ax.set_ylabel("TabPFN fall risk (median; band = middle 50%)")
    ax.set_title(f"Fall risk climbs in the last second before a fall ({lt['falls']} falls)")
    fig.tight_layout()
    fig.savefig(FIG / "fall_lead_time.png")
    plt.close(fig)


def spread_labels(values, min_gap):
    """Nudge label y positions apart (keeps order) so text never overlaps."""
    order = np.argsort(values)
    ys = np.array(values, float)
    for a, b in zip(order[:-1], order[1:]):
        if ys[b] - ys[a] < min_gap:
            ys[b] = ys[a] + min_gap
    return ys


def plot_learning_curve():
    lc = json.loads((ROOT / "results" / "learning_curve.json").read_text())
    tests = ["same floors", "icier (unseen)", "steeper (unseen)"]
    sizes = sorted({r["n"] for r in lc})
    models = (("TabPFN", BLUE, "-"), ("XGBoost", ORANGE, (0, (6, 3))), ("logistic", AQUA, (0, (1, 2))))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), dpi=120, sharey=True)
    for ax, test in zip(axes, tests):
        ends = []
        for model, colour, ls in models:
            v = np.array([[r["roc_auc"] for r in lc if r["model"] == model and r["n"] == n
                           and r["test"] == test] for n in sizes])
            ax.fill_between(sizes, v.min(1), v.max(1), color=colour, alpha=0.12, lw=0)
            ax.plot(sizes, v.mean(1), color=colour, ls=ls, marker="o", ms=5, mec=SURF, mew=1.5,
                    lw=3 if model == "TabPFN" else 2.2)
            ends.append(v.mean(1)[-1])
        for (model, colour, _), y_end, y_lab in zip(models, ends, spread_labels(ends, 0.022)):
            ax.text(sizes[-1] * 1.12, y_lab, f"{model} {y_end:.2f}", color=colour, fontsize=12,
                    va="center", fontweight="bold")
        ax.axhline(0.5, color=MUTED, lw=1.2, ls=(0, (2, 3)))
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{n:,}" if n in (20, 100, 500, 2000) else "" for n in sizes],
                           fontsize=12)
        ax.minorticks_off()
        ax.set_xlim(15, 9000)
        ax.set_title(test.capitalize(), fontsize=16)
        ax.set_xlabel("Training rows")
    axes[0].set_ylabel("Slip ROC-AUC (band = range of 3 draws)")
    fig.suptitle("Learning curve: slip prediction with limited data", x=0.01, ha="left",
                 fontsize=19, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "learning_curve.png")
    plt.close(fig)


def plot_terrain():
    """Unseen real terrain (results/terrain_test.json): fall-prediction ROC-AUC per model
    across the three unseen conditions, and the falls/distance trade-off on hills."""
    tt = json.loads((ROOT / "results" / "terrain_test.json").read_text())
    cd = json.loads((ROOT / "results" / "charts_data.json").read_text())["fall_roc_auc"]
    conds = ["13–15° slopes\n(tilted gravity)", "Steep real hills", "Moderate real hills"]
    vals = {
        "TabPFN, 2,000 rows": [cd["TabPFN, 2,000 rows"], tt["prediction"]["steep hills"]["fall_auc"]["TabPFN"],
                               tt["prediction"]["moderate hills"]["fall_auc"]["TabPFN"]],
        "XGBoost, 18,067 rows": [cd["XGBoost, 18,067 rows"], tt["prediction"]["steep hills"]["fall_auc"]["XGBoost"],
                                 tt["prediction"]["moderate hills"]["fall_auc"]["XGBoost"]],
        "Logistic regression": [cd["Logistic regression"], tt["prediction"]["steep hills"]["fall_auc"]["logistic"],
                                tt["prediction"]["moderate hills"]["fall_auc"]["logistic"]],
        "Tilt rule (no ML)": [cd["Tilt rule, 2 s average (no ML)"],
                              tt["prediction"]["steep hills"]["fall_auc"]["tilt 2 s (no ML)"],
                              tt["prediction"]["moderate hills"]["fall_auc"]["tilt 2 s (no ML)"]],
    }
    colours = [BLUE, ORANGE, AQUA, YELLOW]
    hatches = ["", "//", "..", "xx"]
    fig, ax = plt.subplots(figsize=(13, 6.4), dpi=120)
    w = 0.2
    x = np.arange(len(conds))
    for i, ((name, v), c, hch) in enumerate(zip(vals.items(), colours, hatches)):
        bars = ax.bar(x + (i - 1.5) * w, np.array(v) - 0.5, w * 0.9, bottom=0.5, color=c, label=name,
                      hatch=hch, edgecolor=SURF, linewidth=0)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.006, f"{val:.2f}", ha="center",
                    fontsize=12, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(conds, fontsize=14)
    ax.set_ylim(0.5, 1.09)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Fall within 1 s, ROC-AUC (0.5 = guessing)")
    ax.set_title("Frozen models on conditions never seen in training")
    ax.legend(loc="upper left", ncols=4, fontsize=13, bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / "terrain_auc.png")
    plt.close(fig)

    rows = tt["behaviour"]
    pts = [("Blind trot", MUTED, "o"), ("Always brace", MUTED, "s"), ("Tilt rule", YELLOW, "D"),
           ("TabPFN danger mode", BLUE, "o"), ("Balance reflex", AQUA, "^")]
    keys = {"Blind trot": "blind trot", "Always brace": "always brace", "Tilt rule": "tilt rule",
            "TabPFN danger mode": "TabPFN danger mode", "Balance reflex": "balance reflex"}
    # Label offsets chosen per point so the three close points don't overlap.
    place = {"Blind trot": ((-14, 12), "right", "bottom"), "Always brace": ((-16, -4), "right", "top"),
             "Tilt rule": ((4, -22), "left", "top"), "TabPFN danger mode": ((16, 10), "left", "bottom"),
             "Balance reflex": ((-14, 12), "right", "bottom")}
    fig, ax = plt.subplots(figsize=(11, 6.4), dpi=120)
    for name, c, mk in pts:
        r = [x for x in rows if x["strategy"] == keys[name]]
        f, dist = sum(x["fell"] for x in r), np.mean([x["distance"] for x in r])
        ax.scatter(dist, f, s=260 if "TabPFN" in name else 200, color=c, marker=mk,
                   edgecolor=SURF, linewidth=2, zorder=3)
        off, ha, va = place[name]
        ax.annotate(f"{name}\n{f}/{len(r)} falls, {dist:.2f} m", (dist, f), textcoords="offset points",
                    xytext=off, ha=ha, va=va, fontsize=13, color=INK)
    ax.set_xlim(0.5, 1.6)
    ax.set_ylim(-3, 33)
    ax.set_xlabel("Average distance walked (further = less slowed down)")
    ax.set_ylabel("Falls in 60 walks on real hills")
    ax.set_title("On real hills: same pattern as on flat slopes")
    fig.tight_layout()
    fig.savefig(FIG / "terrain_tradeoff.png")
    plt.close(fig)


def plot_ice():
    """Ice patches (results/ice_governor.json, results/ice_patrol.json)."""
    gv = json.loads((ROOT / "results" / "ice_governor.json").read_text())

    def mean(name):
        r = [x for x in gv if x["strategy"] == name and x["reached"]]
        return np.mean([x["time"] for x in r]), np.mean([x["slip_s"] for x in r]), len(r)

    fig, ax = plt.subplots(figsize=(11, 6.6), dpi=120)
    fixed = [(s, *mean(f"fixed {s}")) for s in (0.3, 0.55, 0.8, 1.0)]
    for s, t, sl, n in fixed:
        ax.scatter(t, sl, s=150, color=MUTED, marker="s", edgecolor=SURF, linewidth=2, zorder=3)
        ax.annotate(f"fixed speed {s}" + ("" if n == 5 else f" (reached {n}/5)"), (t, sl),
                    textcoords="offset points", xytext=(10, 6), fontsize=12, color=SOFT)
    pts = [("Detector-only governor", "detector governor (no TabPFN)", AQUA, "^", (-12, -30)),
           ("XGBoost governor", "XGBoost governor (risk < 0.65)", ORANGE, "D", (10, 8)),
           ("TabPFN governor", "TabPFN governor (risk < 0.65)", BLUE, "o", (-14, -26))]
    for label, key, c, mk, off in pts:
        t, sl, n = mean(key)
        ax.scatter(t, sl, s=260 if "TabPFN" in label else 200, color=c, marker=mk, edgecolor=SURF,
                   linewidth=2, zorder=4)
        ax.annotate(f"{label}\n{t:.0f} s, {sl:.1f} s slipping", (t, sl), textcoords="offset points",
                    xytext=off, fontsize=13, color=INK, ha="left" if off[0] > 0 else "right",
                    fontweight="bold" if "TabPFN" in label else "normal")
    ax.set_xlabel("Time from A to B (faster = left)")
    ax.set_ylabel("Seconds with a foot slipping (fewer = lower)")
    ax.set_xlim(26, 54)
    ax.set_ylim(0, 30)
    ax.set_title("Ice patches: TabPFN governor slips like crawling, near full speed")
    fig.tight_layout()
    fig.savefig(FIG / "ice_governor.png")
    plt.close(fig)

    pt = json.loads((ROOT / "results" / "ice_patrol.json").read_text())
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=120)
    for name, c, ls, mk in (("no map", MUTED, (0, (6, 3)), "s"), ("detector map", AQUA, (0, (1, 2)), "^"),
                            ("TabPFN map", BLUE, "-", "o")):
        legs = sorted({x["leg"] for x in pt})
        med = [np.median([x["slip_s"] for x in pt if x["strategy"] == name and x["leg"] == l]) for l in legs]
        ax.plot(np.array(legs) + 1, med, color=c, ls=ls, marker=mk, ms=8, mec=SURF, mew=2,
                lw=3.2 if "TabPFN" in name else 2.2, label=name)
        ax.text(6.15, med[-1], f"{name} {med[-1]:.1f} s", color=c, fontsize=13, va="center",
                fontweight="bold")
    ax.set_xticks(range(1, 7))
    ax.set_xlim(0.8, 7.4)
    ax.set_ylim(0, None)
    ax.set_xlabel("Leg of the patrol (A→B, B→A, …)")
    ax.set_ylabel("Seconds slipping per leg (median of 5 layouts)")
    ax.set_title("Patrol: the slip map learns to route around the ice")
    fig.tight_layout()
    fig.savefig(FIG / "ice_patrol.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    d = dict(np.load(DATA))
    base, hist = d["features"], d["history"]
    names = list(d["history_names"])
    keep = [i for i, n in enumerate(names) if not n.startswith("grav_")]
    gz = names.index("grav_z_2s")
    data_out = {}

    # ---- Falls on unseen steep slopes ------------------------------------------------
    X = np.hstack([base, hist[:, keep]])
    y = d["fall_next"].astype(int)
    sp = fall_splits(d)
    tr, te = sp["train"], sp["steeper (unseen)"]
    risk = CollisionRisk().fit(X[tr], y[tr], n=2000, pos_fraction=0.25)
    p_tab = risk.predict(X[te])
    p_xgb = xgb(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    p_log = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(
        X[tr], y[tr]).predict_proba(X[te])[:, 1]
    tilt = np.degrees(np.arccos(np.clip(-hist[:, gz], -1, 1)))
    cond = np.c_[d["mu"], d["slope_deg"], base[:, -2:]]
    p_floor = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05).fit(
        cond[tr], y[tr]).predict_proba(cond[te])[:, 1]
    fig, ax = plt.subplots(figsize=(11, 7.2), dpi=120)
    data_out["fall_roc_auc"] = roc_plot(ax, y[te], {
        "TabPFN, 2,000 rows": (p_tab, BLUE, "-"),
        "XGBoost, 18,067 rows": (p_xgb, ORANGE, (0, (6, 3))),
        "Logistic regression": (p_log, AQUA, (0, (1, 2))),
        "Tilt rule, 2 s average (no ML)": (tilt[te], YELLOW, (0, (8, 2, 2, 2))),
        "Told the true floor": (p_floor, MAGENTA, (0, (3, 2))),
    }, "Fall within 1 s, on 13–15° slopes never seen in training")
    fig.tight_layout()
    fig.savefig(FIG / "fall_roc.png")
    plt.close(fig)

    # ---- Lead time: TabPFN risk before each fall --------------------------------------
    ep, t = d["episode"][te], d["t"][te]
    fell = d["episode_fell"][te]
    t_end = {e: t[ep == e].max() for e in np.unique(ep)}
    before = np.array([t_end[e] - ti for e, ti in zip(ep, t)])
    bins = np.arange(0, 3.01, 0.2)
    centres, med, q1, q3 = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = fell & (before >= lo) & (before < hi)
        if m.sum() >= 5:
            centres.append((lo + hi) / 2)
            med.append(np.median(p_tab[m]))
            q1.append(np.percentile(p_tab[m], 25))
            q3.append(np.percentile(p_tab[m], 75))
    safe_med = float(np.median(p_tab[~fell]))
    safe_q3 = float(np.percentile(p_tab[~fell], 75))
    n_falls = len({e for e, f in zip(ep, fell) if f})
    data_out["lead_time"] = dict(seconds_before=[float(c) for c in centres],
                                 median=[float(v) for v in med], q1=[float(v) for v in q1],
                                 q3=[float(v) for v in q3], never_fell_median=safe_med,
                                 never_fell_q3=safe_q3, falls=n_falls)
    plot_lead_time(data_out["lead_time"])

    # ---- Slips on unseen icy + steep floors ------------------------------------------
    ys = d["slip_next"].astype(int)
    ssp = splits(d)
    tr, te = ssp["train"], ssp["icier + steeper"]
    pos = max(ys[tr].mean(), 0.25)
    Xh = np.hstack([base, hist])
    tab_h = CollisionRisk().fit(Xh[tr], ys[tr], n=2000, pos_fraction=pos).predict(Xh[te])
    tab_b = CollisionRisk().fit(base[tr], ys[tr], n=2000, pos_fraction=pos).predict(base[te])
    Xc, yc = context_subsample(Xh[tr], ys[tr], n=2000, pos_fraction=pos, seed=0)
    xgb2k = xgb(Xc, yc).predict_proba(Xh[te])[:, 1]
    xgbal = xgb(Xh[tr], ys[tr]).predict_proba(Xh[te])[:, 1]
    fig, ax = plt.subplots(figsize=(11, 7.2), dpi=120)
    data_out["slip_roc_auc"] = roc_plot(ax, ys[te], {
        "TabPFN + slip detector, 2,000 rows": (tab_h, BLUE, "-"),
        "TabPFN, sensors only": (tab_b, BLUE, (0, (1, 2))),
        "XGBoost + slip detector, 2,000 rows": (xgb2k, ORANGE, (0, (6, 3))),
        "XGBoost + slip detector, 10,290 rows": (xgbal, ORANGE, (0, (1, 2))),
    }, "Slip within 0.2 s, on icy and steep floors never seen in training")
    fig.tight_layout()
    fig.savefig(FIG / "slip_roc.png")
    plt.close(fig)

    plot_learning_curve()

    # ---- Danger-mode trade-off incl. the balance reflex -------------------------------
    dm = json.loads((ROOT / "results" / "danger_mode.json").read_text())
    bb = json.loads((ROOT / "results" / "balancer_baseline.json").read_text())

    def stat(rows, ctrl):
        falls = sum(r["fell"] for r in rows if r["controller"] == ctrl and r["kind"] == "dangerous")
        dist = np.mean([r["distance"] for r in rows if r["controller"] == ctrl and r["kind"] == "safe"])
        return falls, dist

    pts = [("Blind trot", *stat(dm, "never brace"), MUTED, "o"),
           ("Always brace", *stat(dm, "always brace"), MUTED, "s"),
           ("Tilt rule (no ML)", *stat(dm, "tilt rule"), YELLOW, "D"),
           ("Blind trot + TabPFN", *stat(dm, "TabPFN"), BLUE, "o"),
           ("Balance reflex (no TabPFN)", *stat(bb, "balancer only"), AQUA, "^")]
    fig, ax = plt.subplots(figsize=(11, 6.6), dpi=120)
    for name, f, dist, colour, mk in pts:
        ax.scatter(dist, f, s=260 if "TabPFN" in name else 200, color=colour, marker=mk,
                   edgecolor=SURF, linewidth=2, zorder=3)
        ax.annotate(f"{name}\n{f}/40 falls, {dist:.2f} m", (dist, f), textcoords="offset points",
                    xytext=(-14, 12), ha="right", fontsize=13, color=INK)
    ax.set_xlim(0.9, 2.3)
    ax.set_ylim(-3, 32)
    ax.set_xlabel("Distance walked on 40 safe floors (further = less slowed down)")
    ax.set_ylabel("Falls on 40 unseen steep slopes")
    ax.set_title("Danger mode helps a blind robot; a balance reflex does better")
    fig.tight_layout()
    fig.savefig(FIG / "tradeoff.png")
    plt.close(fig)
    data_out["tradeoff"] = [dict(name=n, falls=int(f), safe_distance=float(dd)) for n, f, dd, _, _ in pts]

    (ROOT / "results" / "charts_data.json").write_text(json.dumps(data_out, indent=1))
    print(json.dumps({k: v for k, v in data_out.items() if k.endswith("auc")}, indent=1))
    print("lead time:", {k: v for k, v in data_out["lead_time"].items() if k != "q1"})
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
