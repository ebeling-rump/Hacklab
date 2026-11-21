"""Offline evaluation of a retrieval-context TabPFN policy on teacher data.

    python -m tabpfn_policy.offline_eval --queries 500 --ks 64 256 1024 [--model-version v2] [--devices cpu cuda]

Protocol
  * Split data/teacher.parquet by EPISODE (never by row): --test-frac of episodes are held out.
  * Inputs = perception features (features.FEATURES) + `phase_perc` (phase machine run on the PERCEIVED state).
    The true-state `phase` is only used to group the report; gt_* / x_* columns are never inputs.
  * Movement outputs (dx, dy, dz, dyaw) -> ONE TabPFNRegressor on the "one row per output" table:
    each state becomes 4 rows [features..., output_id] -> standardized target of that output.
  * Gripper -> classification (closed if a_grip < 40 mm), TabPFNClassifier on the same retrieved states.
  * Context retrieval: per query, the k nearest TRAIN states in a KD-tree on standardized inputs.
  * Baselines on the same split: XGBoost (one model on the whole long train table + a classifier), k-NN mean / vote.
  * Errors are reported in mm (dx, dy, dz), mrad (dyaw), accuracy (grip), on a random sample of held-out queries.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .expert import Phase
from .features import FEATURES

ROOT = Path(__file__).resolve().parents[1]
INPUTS = list(FEATURES) + ["phase_perc"]
MOVE_OUT = ["dx", "dy", "dz", "dyaw"]
LABEL_COL = {"dx": "a_dx", "dy": "a_dy", "dz": "a_dz", "dyaw": "a_dyaw"}
REPORT_SCALE = {"dx": 1.0, "dy": 1.0, "dz": 1.0, "dyaw": 1000.0}     # -> mm, mm, mm, mrad
GRIP_THRESHOLD_MM = 40.0


# ---------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------
def split_by_episode(df: pd.DataFrame, test_frac: float, seed: int):
    eps = np.array(sorted(df.episode.unique()))
    rng = np.random.default_rng(seed)
    test_eps = set(rng.choice(eps, size=max(1, int(round(test_frac * len(eps)))), replace=False).tolist())
    te = df.episode.isin(test_eps)
    return df[~te].reset_index(drop=True), df[te].reset_index(drop=True), sorted(test_eps)


class Standardizer:
    def __init__(self, X: np.ndarray):
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd < 1e-9] = 1.0

    def __call__(self, X):
        return (X - self.mu) / self.sd


def long_table(X: np.ndarray, Y: np.ndarray | None):
    """(n, d) states + (n, m) targets -> (n*m, d+1) rows with output_id appended, (n*m,) targets."""
    n, m = X.shape[0], len(MOVE_OUT)
    Xl = np.concatenate([np.repeat(X, m, axis=0), np.tile(np.arange(m), n)[:, None]], axis=1)
    return Xl, (None if Y is None else Y.reshape(-1))


# ---------------------------------------------------------------------------------------------------------
# TabPFN
# ---------------------------------------------------------------------------------------------------------
def make_tabpfn(kind: str, version: str, device: str, n_estimators: int | None):
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion
    cls = TabPFNRegressor if kind == "reg" else TabPFNClassifier
    mv = {v.value: v for v in ModelVersion}[version]
    m = cls.create_default_for_version(mv)
    params = {"device": device}
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    m.set_params(**{k: v for k, v in params.items() if k in m.get_params()})
    return m


class RetrievalTabPFN:
    """Per query: k nearest train states -> fit TabPFN on their long rows -> predict the query's 4 output rows."""

    def __init__(self, Xtr_s, Ytr_s, Gtr, tree, version, device, n_estimators):
        self.Xtr_s, self.Ytr_s, self.Gtr, self.tree = Xtr_s, Ytr_s, Gtr, tree
        self.reg = make_tabpfn("reg", version, device, n_estimators)
        self.clf = make_tabpfn("clf", version, device, n_estimators)

    def predict_one(self, xq_s: np.ndarray, k: int):
        _, idx = self.tree.query(xq_s, k=k)
        idx = np.atleast_1d(idx)
        t0 = time.perf_counter()
        Xc, yc = long_table(self.Xtr_s[idx], self.Ytr_s[idx])
        Xq, _ = long_table(xq_s[None], None)
        self.reg.fit(Xc, yc)
        y = self.reg.predict(Xq)
        t_reg = time.perf_counter() - t0
        g = self.Gtr[idx]
        t1 = time.perf_counter()
        if g.min() == g.max():                      # all neighbours agree: nothing to classify
            grip = int(g[0])
        else:
            self.clf.fit(self.Xtr_s[idx], g)
            grip = int(self.clf.predict(xq_s[None])[0])
        t_clf = time.perf_counter() - t1
        return y, grip, t_reg, t_clf


# ---------------------------------------------------------------------------------------------------------
# metrics / report
# ---------------------------------------------------------------------------------------------------------
def unscale(Y_s, y_mu, y_sd):
    return Y_s * y_sd + y_mu


def report_rows(name, Y_pred, G_pred, Y_true, G_true, phase):
    rows = []
    err = np.abs(Y_pred - Y_true) * np.array([REPORT_SCALE[o] for o in MOVE_OUT])
    for ph in [None] + sorted(set(phase.tolist())):
        m = np.ones(len(phase), bool) if ph is None else phase == ph
        r = dict(method=name, phase="ALL" if ph is None else Phase(ph).name, n=int(m.sum()))
        for j, o in enumerate(MOVE_OUT):
            r[f"mae_{o}"] = float(err[m, j].mean())
        r["grip_acc"] = float((G_pred[m] == G_true[m]).mean())
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "teacher.parquet"))
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--keep-failed", action="store_true", help="also train on episodes the teacher failed")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--queries", type=int, default=500, help="random held-out states to evaluate")
    ap.add_argument("--ks", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--model-version", default="v2", help="v2 | v2.5 | v2.6 | v3 | v3.5 | v3.5-fast")
    ap.add_argument("--n-estimators", type=int, default=1,
                    help="TabPFN ensemble size (1 = fastest; latency grows ~linearly with it)")
    ap.add_argument("--budget-min", type=float, default=5.0,
                    help="minutes of TabPFN time per k; the number of queries per k is chosen to fit")
    ap.add_argument("--min-queries", type=int, default=30, help="evaluate at least this many queries per k")
    ap.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    ap.add_argument("--phase-weight", type=float, default=1.0, help="extra weight of phase in the KD-tree metric")
    ap.add_argument("--max-minutes", type=float, default=40.0, help="abort if the estimated total is longer")
    ap.add_argument("--yes", action="store_true", help="run even if the estimate exceeds --max-minutes")
    ap.add_argument("--allow-browser", action="store_true", help="allow TabPFN's browser login (licence) flow")
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()
    if not a.allow_browser:
        os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    # Contexts of 4k rows exceed TabPFN's default CPU guard (1000 rows for v2, 5000 for v3+); we want to measure that.
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

    import torch
    devices = [d for d in a.devices if d == "cpu" or (d == "cuda" and torch.cuda.is_available())]
    skipped = sorted(set(a.devices) - set(devices))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- data, split, scaling ---------------------------------------------------------------------------
    df = pd.read_parquet(a.data)
    tr, te, test_eps = split_by_episode(df, a.test_frac, a.split_seed)
    if not a.keep_failed:
        n0 = tr.episode.nunique()
        tr = tr[tr.success].reset_index(drop=True)
        if tr.episode.nunique() < n0:
            print(f"dropped {n0 - tr.episode.nunique()} failed teacher episodes from the train split")
    print(f"data {len(df)} rows / {df.episode.nunique()} episodes -> train {len(tr)} rows "
          f"({tr.episode.nunique()} ep), test {len(te)} rows ({len(test_eps)} ep, held out)")
    Xtr, Xte = tr[INPUTS].to_numpy(float), te[INPUTS].to_numpy(float)
    std = Standardizer(Xtr)
    w = np.ones(len(INPUTS))
    w[INPUTS.index("phase_perc")] = a.phase_weight
    Xtr_s, Xte_s = std(Xtr) * w, std(Xte) * w
    Ytr = tr[[LABEL_COL[o] for o in MOVE_OUT]].to_numpy(float)
    Yte = te[[LABEL_COL[o] for o in MOVE_OUT]].to_numpy(float)
    y_mu, y_sd = Ytr.mean(0), Ytr.std(0)
    y_sd[y_sd < 1e-9] = 1.0
    Ytr_s = (Ytr - y_mu) / y_sd
    Gtr = (tr.a_grip.to_numpy() < GRIP_THRESHOLD_MM).astype(int)   # 1 = close
    Gte = (te.a_grip.to_numpy() < GRIP_THRESHOLD_MM).astype(int)
    tree = cKDTree(Xtr_s)

    rng = np.random.default_rng(a.split_seed + 1)
    q = rng.choice(len(te), size=min(a.queries, len(te)), replace=False)   # random order: prefixes are random too
    Xq_s, Yq, Gq, Pq = Xte_s[q], Yte[q], Gte[q], te.phase.to_numpy()[q]
    print(f"query sample: {len(q)} random held-out states; phases: "
          f"{ {k: int(v) for k, v in pd.Series([Phase(p).name for p in Pq]).value_counts().items()} }")

    results, latency = [], []

    # ---- baselines (predicted for the whole sample; scored on every subset TabPFN uses) ---------------
    from xgboost import XGBClassifier, XGBRegressor
    t0 = time.perf_counter()
    Xl, yl = long_table(Xtr_s, Ytr_s)
    xgb = XGBRegressor(n_estimators=400, max_depth=8, learning_rate=0.05, subsample=0.8, n_jobs=-1)
    xgb.fit(Xl, yl)
    xgbc = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=-1).fit(Xtr_s, Gtr)
    t_fit = time.perf_counter() - t0
    Xql, _ = long_table(Xq_s, None)
    t0 = time.perf_counter()
    base = {"xgboost(all train rows)": (unscale(xgb.predict(Xql).reshape(-1, 4), y_mu, y_sd), xgbc.predict(Xq_s))}
    t_pred = (time.perf_counter() - t0) / len(q)
    print(f"xgboost: fit {t_fit:.1f}s on {len(Xl)} long rows, predict {t_pred * 1e3:.2f} ms/query")
    for kn in (1, 10):
        _, idx = tree.query(Xq_s, k=kn)
        idx = idx.reshape(len(q), kn)
        base[f"{kn}-NN"] = (unscale(Ytr_s[idx].mean(1), y_mu, y_sd), (Gtr[idx].mean(1) >= 0.5).astype(int))

    def score_baselines(n: int, label: str):
        for name, (Yp, Gp) in base.items():
            results.extend(dict(r, eval_set=label) for r in report_rows(name, Yp[:n], Gp[:n], Yq[:n], Gq[:n], Pq[:n]))

    score_baselines(len(q), f"all {len(q)}")

    # ---- TabPFN with retrieval -------------------------------------------------------------------------
    for dev in devices:
        try:
            model = RetrievalTabPFN(Xtr_s, Ytr_s, Gtr, tree, a.model_version, dev, a.n_estimators)
            model.predict_one(Xq_s[0], min(a.ks))   # load weights / warm up
        except Exception as e:  # licence / login / download problems end up here
            print(f"\nTabPFN ({a.model_version}, {dev}) could not be loaded: {type(e).__name__}: {e}")
            print("For v2.5+ weights accept the licence at https://ux.priorlabs.ai and set TABPFN_TOKEN, "
                  "or run with --allow-browser once.")
            continue
        # Per k: time 2 warm queries, then evaluate as many of the sample as fit into --budget-min.
        plan = {}
        for k in a.ks:
            if k > len(Xtr_s):
                print(f"skip k={k} > {len(Xtr_s)} train states")
                continue
            sec = float(np.mean([sum(model.predict_one(Xq_s[i], k)[2:]) for i in range(1, 3)]))
            n = int(np.clip(a.budget_min * 60 / max(sec, 1e-3), min(a.min_queries, len(q)), len(q)))
            plan[k] = (n, sec)
        total = sum(n * sec for n, sec in plan.values()) / 60
        print(f"\nTabPFN {a.model_version} on {dev}, n_estimators={a.n_estimators}: plan "
              + ", ".join(f"k={k}: {n} queries x {sec:.1f}s" for k, (n, sec) in plan.items())
              + f"  -> estimated {total:.1f} min", flush=True)
        if total > a.max_minutes and not a.yes:
            print(f"estimate exceeds --max-minutes {a.max_minutes}; lower --budget-min / --queries or pass --yes")
            continue
        for k, (n, _) in plan.items():
            Yp_s, Gp, tr_, tc_ = np.zeros((n, 4)), np.zeros(n, int), [], []
            t0 = time.time()
            for i in range(n):
                y, g, t_reg, t_clf = model.predict_one(Xq_s[i], k)
                Yp_s[i], Gp[i] = y, g
                tr_.append(t_reg)
                tc_.append(t_clf)
                if (i + 1) % 25 == 0:
                    print(f"  k={k} {i + 1}/{n} [{time.time() - t0:.0f}s]", flush=True)
            label = f"first {n}"
            results.extend(dict(r, eval_set=label) for r in
                           report_rows(f"tabpfn k={k} {dev}", unscale(Yp_s, y_mu, y_sd), Gp, Yq[:n], Gq[:n], Pq[:n]))
            score_baselines(n, label)
            tot = np.array(tr_) + np.array(tc_)
            latency.append(dict(device=dev, k=k, context_rows=4 * k, queries=n, reg_mean_ms=1e3 * np.mean(tr_),
                                reg_p95_ms=1e3 * np.percentile(tr_, 95), total_mean_ms=1e3 * tot.mean(),
                                total_p95_ms=1e3 * np.percentile(tot, 95)))
            print(f"  k={k}: latency per query mean {1e3 * tot.mean():.0f} ms, p95 {1e3 * np.percentile(tot, 95):.0f} ms",
                  flush=True)

    # ---- report ----------------------------------------------------------------------------------------
    res = pd.DataFrame(results).drop_duplicates(subset=["method", "phase", "eval_set"])
    lat = pd.DataFrame(latency)
    pd.set_option("display.width", 220)
    print("\n=== overall: MAE in mm (dx, dy, dz), mrad (dyaw); grip accuracy. "
          "Each TabPFN row is compared with the baselines on the SAME query subset ===")
    allr = res[res.phase == "ALL"]
    for es, g in allr.groupby("eval_set", sort=False):
        print(f"\n-- eval set: {es} queries")
        print(g.drop(columns=["phase", "eval_set"]).round(3).to_string(index=False))
    print("\n=== per phase (largest eval set per method) ===")
    big = res.sort_values("n", ascending=False).drop_duplicates(subset=["method", "phase"])
    for ph in [Phase(i).name for i in range(len(Phase))]:
        g = big[big.phase == ph]
        if len(g):
            print(f"\n-- {ph}")
            print(g.drop(columns=["phase"]).round(3).to_string(index=False))
    if len(lat):
        print("\n=== TabPFN latency per query (fit + predict, regressor + gripper classifier) ===")
        print(lat.round(1).to_string(index=False))
    if skipped:
        print(f"\n(devices skipped, not available: {skipped})")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    res.to_csv(out / f"offline_eval_{stamp}.csv", index=False)
    lat.to_csv(out / f"offline_latency_{stamp}.csv", index=False)
    meta = dict(vars(a), n_train_rows=len(tr), n_test_rows=len(te), test_episodes=test_eps, n_queries=len(q))
    (out / f"offline_eval_{stamp}.json").write_text(json.dumps(meta, indent=2, default=str))
    plot(res, lat, out / f"offline_error_vs_k_{stamp}.png", a.model_version)
    print(f"\nsaved {out}{os.sep}offline_*_{stamp}.*")


def plot(res: pd.DataFrame, lat: pd.DataFrame, path: Path, version: str):
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    allr = res[res.phase == "ALL"]
    tab = allr[allr.method.str.startswith("tabpfn")].copy()
    if tab.empty:
        return
    tab["k"] = tab.method.str.extract(r"k=(\d+)").astype(int)
    tab["dev"] = tab.method.str.split().str[-1]
    metrics = [f"mae_{o}" for o in MOVE_OUT] + ["grip_acc"]
    units = ["mm", "mm", "mm", "mrad", "accuracy"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.6))
    base = allr[~allr.method.str.startswith("tabpfn") & allr.eval_set.str.startswith("all")]
    for ax, m, u in zip(axes, metrics, units):
        for dev, g in tab.groupby("dev"):
            g = g.sort_values("k")
            ax.plot(g.k, g[m], "o-", label=f"TabPFN {version} ({dev})")
            for _, r in g.iterrows():
                ax.annotate(f"n={r.n}", (r.k, r[m]), textcoords="offset points", xytext=(4, 4), fontsize=7)
        for (_, r), ls in zip(base.iterrows(), ["--", ":", "-."]):
            ax.axhline(r[m], ls=ls, color="grey", lw=1, label=r.method)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("k retrieved states  (context rows = 4k)")
        ax.set_title(f"{m} [{u}]")
        sec = ax.secondary_xaxis("top", functions=(lambda k: 4 * k, lambda r: r / 4))
        sec.set_xlabel("context rows")
        if len(tab):
            ticks = sorted(4 * tab.k.unique())
            sec.set_xticks(ticks, [str(t) for t in ticks])
            sec.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xticks(sorted(tab.k.unique()), [str(k) for k in sorted(tab.k.unique())])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
