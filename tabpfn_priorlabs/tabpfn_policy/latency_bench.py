"""How fast can TabPFN act as a 20 Hz policy on this CPU? Speed vs. accuracy of context strategies.

    python -m tabpfn_policy.latency_bench --model-version v3.5-fast --queries 300

Variants (movement outputs dx, dy, dz, dyaw; same long "one row per output" table as offline_eval):
  refit-retrieval k   per query: k nearest train states -> fit -> predict   (offline_eval's method, reference)
  global-cached M     ONE context of M train states (stratified over phases), fit_mode="fit_with_cache" once;
                      every query only runs predict()
  phase-cached M      one cached context of M states per perceived phase (8 models); query routed by phase_perc
  phase-preproc M     same contexts, fit_mode="fit_preprocessors" (only preprocessing cached, no KV cache)
Reports one-time fit cost, single-query predict latency (mean / p95: what a control loop pays per step),
batched throughput, and MAE vs. XGBoost on the same held-out queries.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .expert import Phase
from .offline_eval import (INPUTS, LABEL_COL, MOVE_OUT, REPORT_SCALE, Standardizer, long_table, make_tabpfn,
                           split_by_episode)

ROOT = Path(__file__).resolve().parents[1]
SCALE = np.array([REPORT_SCALE[o] for o in MOVE_OUT])


def stratified(phase: np.ndarray, m: int, rng) -> np.ndarray:
    """m indices spread evenly over the phases present (fewer if a phase is small)."""
    ph = np.unique(phase)
    per = max(1, m // len(ph))
    idx = [rng.choice(np.flatnonzero(phase == p), size=min(per, int((phase == p).sum())), replace=False) for p in ph]
    return np.concatenate(idx)


def mae(Yp, Yt):
    return (np.abs(Yp - Yt) * SCALE).mean(0)


def timed_single(model, Xq_s, n):
    """Predict one query at a time (as a controller would). Returns predictions (n, 4) and per-call seconds."""
    out, ts = np.zeros((n, 4)), []
    for i in range(n):
        Xl, _ = long_table(Xq_s[i:i + 1], None)
        t = time.perf_counter()
        out[i] = model.predict(Xl)
        ts.append(time.perf_counter() - t)
    return out, np.array(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "teacher.parquet"))
    ap.add_argument("--model-version", default="v3.5-fast")
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--refit-queries", type=int, default=40, help="queries for the slow refit reference (0 = skip)")
    ap.add_argument("--refit-k", type=int, default=256)
    ap.add_argument("--global-m", type=int, nargs="*", default=[256, 1024], help="(none = skip)")
    ap.add_argument("--phase-m", type=int, nargs="+", default=[256])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4,
                    help="torch CPU threads (4 was fastest here; 12-16 threads add overhead on tiny batches)")
    ap.add_argument("--preproc-m", type=int, default=None,
                    help="per-phase M for the fit_preprocessors variant (default: first --phase-m; -1 = skip)")
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
    import torch
    torch.set_num_threads(a.threads)

    df = pd.read_parquet(a.data)
    tr, te, _ = split_by_episode(df, 0.2, a.seed)
    tr = tr[tr.success].reset_index(drop=True)
    std = Standardizer(tr[INPUTS].to_numpy(float))
    Xtr_s, Xte_s = std(tr[INPUTS].to_numpy(float)), std(te[INPUTS].to_numpy(float))
    Ytr = tr[[LABEL_COL[o] for o in MOVE_OUT]].to_numpy(float)
    Yte = te[[LABEL_COL[o] for o in MOVE_OUT]].to_numpy(float)
    y_mu, y_sd = Ytr.mean(0), Ytr.std(0)
    y_sd[y_sd < 1e-9] = 1.0
    Ytr_s = (Ytr - y_mu) / y_sd
    Ptr = tr.phase_perc.to_numpy()
    rng = np.random.default_rng(a.seed + 1)
    q = rng.choice(len(te), size=min(a.queries, len(te)), replace=False)   # same sampler as offline_eval
    Xq_s, Yq, Pq_perc = Xte_s[q], Yte[q], te.phase_perc.to_numpy()[q]
    n = len(q)
    un = lambda Ys: Ys * y_sd + y_mu
    rows = []
    print(f"train {len(tr)} states, {n} held-out queries, model {a.model_version}, CPU, {a.threads} torch threads",
          flush=True)

    # ---- XGBoost reference on the same queries --------------------------------------------------------
    from xgboost import XGBRegressor
    Xl, yl = long_table(Xtr_s, Ytr_s)
    xgb = XGBRegressor(n_estimators=400, max_depth=8, learning_rate=0.05, subsample=0.8, n_jobs=-1).fit(Xl, yl)
    Yx = un(xgb.predict(long_table(Xq_s, None)[0]).reshape(-1, 4))
    _, tx = timed_single(xgb, Xq_s, min(n, 100))
    rows.append(dict(variant="xgboost (all train rows)", context_states=len(tr), fit_s=np.nan,
                     pred_mean_ms=1e3 * tx.mean(), pred_p95_ms=1e3 * np.percentile(tx, 95),
                     **{f"mae_{o}": v for o, v in zip(MOVE_OUT, mae(Yx, Yq))}, n=n))

    def report(name, m_states, fit_s, Yp_s, ts, nq, batch_ms=np.nan):
        r = dict(variant=name, context_states=m_states, fit_s=fit_s, pred_mean_ms=1e3 * ts.mean(),
                 pred_p95_ms=1e3 * np.percentile(ts, 95), batch_ms_per_query=batch_ms,
                 **{f"mae_{o}": v for o, v in zip(MOVE_OUT, mae(un(Yp_s), Yq[:nq]))}, n=nq)
        # XGBoost on exactly the same queries, for a fair comparison of subsets
        r["xgb_same_q"] = " / ".join(f"{v:.2f}" for v in mae(Yx[:nq], Yq[:nq]))
        rows.append(r)
        print(f"  {name:28s} fit {fit_s:6.1f}s  predict mean {r['pred_mean_ms']:7.1f} ms  p95 {r['pred_p95_ms']:7.1f} ms  "
              f"MAE dx/dy/dz/dyaw {r['mae_dx']:.2f}/{r['mae_dy']:.2f}/{r['mae_dz']:.2f}/{r['mae_dyaw']:.2f}", flush=True)

    # ---- reference: refit on retrieved neighbours per query ------------------------------------------
    tree = cKDTree(Xtr_s)
    reg = make_tabpfn("reg", a.model_version, "cpu", 1)
    nr = min(a.refit_queries, n)
    Yp, ts = np.zeros((max(nr, 1), 4)), []
    for i in range(nr):
        _, idx = tree.query(Xq_s[i], k=a.refit_k)
        Xc, yc = long_table(Xtr_s[idx], Ytr_s[idx])
        t = time.perf_counter()
        reg.fit(Xc, yc)
        Yp[i] = reg.predict(long_table(Xq_s[i:i + 1], None)[0])
        ts.append(time.perf_counter() - t)
    if nr > 0:
        report(f"refit-retrieval k={a.refit_k}", a.refit_k, np.nan, Yp, np.array(ts), nr)

    # ---- one cached global context -------------------------------------------------------------------
    for m in a.global_m:
        idx = stratified(Ptr, m, rng)
        reg = make_tabpfn("reg", a.model_version, "cpu", 1)
        reg.set_params(fit_mode="fit_with_cache")
        t = time.perf_counter()
        reg.fit(*long_table(Xtr_s[idx], Ytr_s[idx]))
        fit_s = time.perf_counter() - t
        Yp, ts = timed_single(reg, Xq_s, n)
        t = time.perf_counter()
        reg.predict(long_table(Xq_s, None)[0])
        batch = 1e3 * (time.perf_counter() - t) / n
        report(f"global-cached M={len(idx)}", len(idx), fit_s, Yp, ts, n, batch)

    # ---- one cached context per perceived phase ------------------------------------------------------
    for m in a.phase_m:
        modes = ("fit_with_cache", "fit_preprocessors") if m == (a.preproc_m or a.phase_m[0]) else ("fit_with_cache",)
        for mode in modes:
            models, fit_s, used = {}, 0.0, 0
            for p in np.unique(Ptr):
                pool = np.flatnonzero(Ptr == p)
                idx = rng.choice(pool, size=min(m, len(pool)), replace=False)
                used += len(idx)
                reg = make_tabpfn("reg", a.model_version, "cpu", 1)
                reg.set_params(fit_mode=mode)
                t = time.perf_counter()
                reg.fit(*long_table(Xtr_s[idx], Ytr_s[idx]))
                fit_s += time.perf_counter() - t
                models[int(p)] = reg
            Yp, ts = np.zeros((n, 4)), []
            for i in range(n):
                reg = models[int(Pq_perc[i])] if int(Pq_perc[i]) in models else next(iter(models.values()))
                t = time.perf_counter()
                Yp[i] = reg.predict(long_table(Xq_s[i:i + 1], None)[0])
                ts.append(time.perf_counter() - t)
            tag = "phase-cached" if mode == "fit_with_cache" else "phase-preproc"
            report(f"{tag} M={m}/phase", used, fit_s, Yp, np.array(ts), n)

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("\n=== speed vs accuracy (MAE: mm, mm, mm, mrad; control budget at 20 Hz = 50 ms/step) ===")
    print(res.round(2).to_string(index=False))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    res.to_csv(out / f"latency_bench_{a.model_version}_{stamp}.csv", index=False)
    print(f"\nsaved {out / f'latency_bench_{a.model_version}_{stamp}.csv'}")


if __name__ == "__main__":
    main()
