"""Closed-loop navigation on predicted costmaps.

Planner: A* (8-connected, geometric distances) on a costmap built from each
method's predicted high-slip probability p:  step cost = distance * (1 + ALPHA * p).
Hard obstacles seen by the lidar are inflated by the robot radius and are
impassable for every method. The robot works in the right half of the map,
which none of the local calibration data came from.

(A pure gap-based local planner was tried first and timed out on most episodes
even with perfect terrain knowledge: concave high-slip patches trap it the same
way dead ends do, so a global planner is needed to compare risk maps fairly.)

Each method only changes the risk map the planner sees:
    No terrain model     risk = 0 everywhere (pure geometric gap rule, obstacles only)
    Geometric rule       hand-tuned height / roughness / slope thresholds
    Cross-site XGBoost   trained on 20 other sites, no local data
    Cross-site + local   pooled data plus the local calibration rows
    CatBoost, XGBoost (tuned), TabPFN v2   trained on N_LOCAL local rows only
    Oracle               the true probability of a high-slip label

Execution: every time the robot enters a cell along the planned route it gets stuck with the true
p_stuck of that cell. Random draws are shared across methods (common random
numbers), so all methods face the same luck.

Output: results/navigation.csv, results/nav_paths.npz
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from scipy.stats import norm
from skimage.graph import route_through_array
from xgboost import XGBClassifier

from benchmark import pooled_data, pooled_xgb
from models import GeometricRule, fit_predict, local_models
from terrain_sim import CELL, N, OBSTACLE, SLIP_THRESHOLD, make_dataset, make_site, observe

N_LOCAL = 256
ALPHA = 30.0
EPISODES_PER_SITE = 8
COL_LO = N // 2                  # robot operates in the right (unseen) half

METHODS = ["No terrain model", "Geometric rule", "Cross-site XGBoost", "Cross-site + local XGBoost",
           "CatBoost", "XGBoost (tuned)", "TabPFN v2", "Oracle"]


def risk_maps(site, Xp, yp, zero_shot, rng):
    """Predicted high-slip probability per cell of the right half, for every method."""
    rr, cc = np.nonzero(site.terrain[:, COL_LO:] != OBSTACLE)
    cc = cc + COL_LO
    X = observe(site, rr, cc, rng.uniform(3, 12, len(rr)), rng)
    Xtr, ytr, _, _ = make_dataset(site, N_LOCAL, "calib", rng)
    zoo = local_models(seed=0)
    preds = {
        "No terrain model": np.zeros(len(rr)),
        "Geometric rule": GeometricRule().predict_proba(X)[:, 1],
        "Cross-site XGBoost": zero_shot.predict_proba(X)[:, 1],
        "CatBoost": fit_predict(zoo["CatBoost"], Xtr, ytr, X),
        "XGBoost (tuned)": fit_predict(zoo["XGBoost (tuned)"], Xtr, ytr, X),
        "TabPFN v2": fit_predict(zoo["TabPFN v2"], Xtr, ytr, X),
        "Oracle": oracle_map(site)[rr, cc],
    }
    w = np.concatenate([np.ones(len(yp)), np.full(N_LOCAL, len(yp) / N_LOCAL)])
    m = pooled_xgb().fit(np.vstack([Xp, Xtr]), np.concatenate([yp, ytr]), sample_weight=w)
    preds["Cross-site + local XGBoost"] = m.predict_proba(X)[:, 1]
    maps = {}
    for k, v in preds.items():
        R = np.full((N, N), np.nan)
        R[rr, cc] = v
        maps[k] = R
    return maps


def plan(site, risk, start, goal):
    """A* route on the method's costmap; returns (K, 2) cell path or None."""
    blocked = binary_dilation(site.terrain == OBSTACLE, iterations=1)
    cost = 1.0 + ALPHA * np.nan_to_num(risk, nan=1.0)
    cost[blocked] = np.inf
    cost[:, :COL_LO] = np.inf
    s, g = tuple(int(v) for v in start), tuple(int(v) for v in goal)
    if not (np.isfinite(cost[s]) and np.isfinite(cost[g])):
        return None
    path, total = route_through_array(cost, s, g, fully_connected=True, geometric=True)
    return None if not np.isfinite(total) else np.array(path)


def run_episode(site, risk, start, goal, u):
    path = plan(site, risk, start, goal)
    if path is None:
        return "no path", np.array([start])
    for i, (r, c) in enumerate(path[1:], 1):
        if u[r, c] < site.p_stuck[r, c]:
            return "stuck", path[: i + 1]
    return "success", path


def pick_endpoints(site, rng, oracle, tries=200):
    """Start near the bottom, goal near the top of the right half.

    A mission counts only if it is feasible: a route exists whose every cell has
    true slip below 0.6 (found by A* on the oracle map). Otherwise resample.
    """
    ok = (site.terrain != OBSTACLE) & (site.slip < 0.3)

    def pick(r_lo, r_hi):
        rr, cc = np.nonzero(ok[r_lo:r_hi, COL_LO + 5: N - 5])
        i = rng.integers(len(rr))
        return (rr[i] + r_lo + 0.5, cc[i] + COL_LO + 5 + 0.5)

    for _ in range(tries):
        start, goal = pick(3, 12), pick(N - 12, N - 3)
        path = plan(site, oracle, start, goal)
        if path is not None and site.slip[path[:, 0], path[:, 1]].max() < 0.6:
            return start, goal
    return None


def oracle_map(site):
    rr, cc = np.nonzero(site.terrain != OBSTACLE)
    R = np.full((N, N), np.nan)
    R[rr, cc] = norm.cdf((site.slip[rr, cc] - SLIP_THRESHOLD) / 0.05)
    return R


def nav_sites(k, start=2000):
    """Sites with 15 to 50 percent high-slip cells and feasible missions."""
    s = start
    while k > 0:
        site = make_site(s)
        right = site.terrain[:, COL_LO:] != OBSTACLE
        frac = (site.slip[:, COL_LO:][right] > SLIP_THRESHOLD).mean()
        if 0.15 <= frac <= 0.5 and pick_endpoints(site, np.random.default_rng(0), oracle_map(site)) is not None:
            k -= 1
            yield s
        s += 1


def main(n_sites=8, out="results/navigation.csv"):
    Path(out).parent.mkdir(exist_ok=True)
    Xp, yp = pooled_data()
    zero_shot = pooled_xgb().fit(Xp, yp)
    rows, keep = [], {}
    for sid in nav_sites(n_sites):
        site = make_site(sid)
        rng = np.random.default_rng(sid)
        maps = risk_maps(site, Xp, yp, zero_shot, rng)
        for ep in range(EPISODES_PER_SITE):
            ends = pick_endpoints(site, rng, maps["Oracle"])
            if ends is None:
                continue
            start, goal = ends
            u = rng.random((N, N))
            for meth in METHODS:
                outcome, path = run_episode(site, maps[meth], start, goal, u)
                seg = np.diff(path, axis=0)
                length = np.linalg.norm(seg, axis=1).sum() * CELL
                cells = np.unique(path.astype(int), axis=0)
                hi = (site.slip[cells[:, 0], cells[:, 1]] > SLIP_THRESHOLD).mean()
                rows.append(dict(site=sid, episode=ep, method=meth, outcome=outcome,
                                 path_m=length, straight_m=np.linalg.norm(np.subtract(goal, start)) * CELL,
                                 frac_high_slip_cells=hi))
                keep[f"{sid}|{ep}|{meth}"] = path
            keep[f"{sid}|{ep}|endpoints"] = np.array([start, goal])
        for meth in METHODS:
            keep[f"{sid}|risk|{meth}"] = maps[meth]
        print(f"nav site {sid} done", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
    np.savez_compressed("results/nav_paths.npz", **keep)


if __name__ == "__main__":
    main()
