"""Phase 2: predict AGV positions 0.5 s and 1.0 s ahead from few observations.

Models (all predict the residual to constant velocity, see motion_data.py):
    cv        constant velocity (residual = 0)
    gp        Gaussian Process (scikit-learn, ARD RBF + noise), trained locally
    tabpfn    TabPFN regressor via the cloud API (tabpfn-client), one model per target

Each learned model is trained on N randomly chosen rows of the train set (N = 100, 300, 1000:
"very few observations") and evaluated on held-out trajectories:
    test     same motion pattern, different episodes
    flipped  unseen routing at intersections
    fast     unseen speed range
Predictions are cached in data/motion/pred_<model>_<N>.npz, so reports and plots never
re-call the API. TabPFN needs TABPFN_TOKEN (never printed).

Usage:
    python predict_motion.py                    # all models, N = 100 300 1000
    python predict_motion.py --models cv gp     # local only, no API calls
    python predict_motion.py --report-only      # recompute tables from cached predictions
"""
import argparse
import time

import numpy as np

from motion_data import BEHAVIOURS, FEATURES, MOTION, TARGETS, load

EVAL_SETS = ["test", "flipped", "fast"]
EVAL_ROWS = 1000          # rows sampled per evaluation set (keeps API payloads small)
NEAR = 3.0                # m: "near the robot"
LABEL = {"turn": "turning at intersection", "station_approach": "slowing for station",
         "station_pause": "standing at station", "yield": "yielding / waiting", "cruise": "cruising straight"}


def matrix(d, cols):
    return np.column_stack([d[c] for c in cols]).astype(float)


def eval_data(rng):
    """Fixed random subsample of each evaluation set, concatenated with a set label."""
    parts = []
    for s in EVAL_SETS:
        d = load(s)
        idx = rng.choice(len(d["x"]), size=min(EVAL_ROWS, len(d["x"])), replace=False)
        parts.append({k: v[idx] for k, v in d.items()} | {"set": np.full(len(idx), s)})
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# ---------- models: fit on (X, Y) with 4 target columns, return (pred (n,4), timings) ----------

def fit_predict_cv(X, Y, Xe):
    return np.zeros((len(Xe), Y.shape[1])), {"fit_s": 0.0, "predict_s": 0.0}


def fit_predict_gp(X, Y, Xe):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.preprocessing import StandardScaler

    import warnings

    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)   # length scales hitting bounds = irrelevant feature

    scaler = StandardScaler().fit(X)
    Xs, Xes = scaler.transform(X), scaler.transform(Xe)
    preds, fit_s, pred_s = [], 0.0, 0.0
    for j in range(Y.shape[1]):
        kernel = ConstantKernel(0.1) * RBF(length_scale=np.ones(X.shape[1])) + WhiteKernel(1e-3)
        gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=1, random_state=0)
        t0 = time.perf_counter()
        gp.fit(Xs, Y[:, j])
        t1 = time.perf_counter()
        preds.append(gp.predict(Xes))
        t2 = time.perf_counter()
        fit_s, pred_s = fit_s + t1 - t0, pred_s + t2 - t1
    return np.column_stack(preds), {"fit_s": fit_s, "predict_s": pred_s}


def fit_predict_tabpfn(X, Y, Xe):
    try:
        import truststore
        truststore.inject_into_ssl()     # TLS-scanning antivirus / proxies
    except ImportError:
        pass
    from tabpfn_client import TabPFNRegressor

    preds, fit_s, pred_s = [], 0.0, 0.0
    for j in range(Y.shape[1]):
        model = TabPFNRegressor(random_state=0)
        t0 = time.perf_counter()
        model.fit(X, Y[:, j])
        t1 = time.perf_counter()
        preds.append(model.predict(Xe))
        t2 = time.perf_counter()
        fit_s, pred_s = fit_s + t1 - t0, pred_s + t2 - t1
        print(f"    tabpfn target {TARGETS[j]}: fit {t1 - t0:.1f} s, predict {t2 - t1:.1f} s")
    return np.column_stack(preds), {"fit_s": fit_s, "predict_s": pred_s}


MODELS = {"cv": fit_predict_cv, "gp": fit_predict_gp, "tabpfn": fit_predict_tabpfn}


# ---------- evaluation ----------

def errors(E, pred):
    """Position error (m) at +0.5 s and +1.0 s for each row."""
    true = matrix(E, TARGETS)
    diff = pred - true
    return np.hypot(diff[:, 0], diff[:, 1]), np.hypot(diff[:, 2], diff[:, 3])


def report(results, E):
    print("\nMean position error (m) at +0.5 s / +1.0 s, all AGVs:")
    print(f"{'model':<14}{'N':>6}" + "".join(f"{s:>18}" for s in EVAL_SETS) + f"{'near robot (test)':>20}")
    for (name, n), r in results.items():
        e05, e10 = errors(E, r["pred"])
        cells = "".join(f"{e05[E['set'] == s].mean():>8.3f} / {e10[E['set'] == s].mean():.3f}" for s in EVAL_SETS)
        near = (E["set"] == "test") & (E["r_dist"] < NEAR)
        print(f"{name:<14}{n:>6}{cells}{e05[near].mean():>11.3f} / {e10[near].mean():.3f}")

    print("\nError at +1.0 s on the test set by upcoming AGV behaviour (largest N per model):")
    best = {}
    for (name, n) in results:
        best[name] = max(n, best.get(name, 0))
    test = E["set"] == "test"
    print(f"{'behaviour':<26}{'share':>7}" + "".join(f"{m:>10}" for m in best))
    for b in BEHAVIOURS:
        mask = test & (E["behaviour"] == b)
        if mask.any():
            cells = "".join(f"{errors(E, results[(m, best[m])]['pred'])[1][mask].mean():>10.3f}" for m in best)
            print(f"{LABEL[b]:<26}{mask.mean() / test.mean():>7.0%}{cells}")

    print("\nTiming (wall-clock, 4 targets): fit / predict for "
          f"{len(E['x'])} rows, per row")
    for (name, n), r in results.items():
        t = r["timing"]
        print(f"{name:<14}{n:>6}  fit {t['fit_s']:7.2f} s   predict {t['predict_s']:7.2f} s"
              f"   ({1000 * t['predict_s'] / len(E['x']):.3f} ms/row)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--n-train", nargs="+", type=int, default=[100, 300, 1000])
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    train = load("train")
    E = eval_data(rng)
    Xe = matrix(E, FEATURES)
    order = rng.permutation(len(train["x"]))       # nested subsets: the 100 are inside the 300 etc.

    results = {}
    for name in args.models:
        for n in ([0] if name == "cv" else args.n_train):
            path = MOTION / f"pred_{name}_{n}.npz"
            if args.report_only:
                if not path.exists():
                    continue
                with np.load(path) as f:
                    results[(name, n)] = {"pred": f["pred"], "timing": {"fit_s": float(f["fit_s"]),
                                                                        "predict_s": float(f["predict_s"])}}
                continue
            idx = order[:n] if n else order[:1]
            X, Y = matrix(train, FEATURES)[idx], matrix(train, TARGETS)[idx]
            print(f"{name} N={n}")
            pred, timing = MODELS[name](X, Y, Xe)
            np.savez(path, pred=pred, **timing)
            results[(name, n)] = {"pred": pred, "timing": timing}
    report(results, E)


if __name__ == "__main__":
    main()
