"""Closed-loop TabPFN student: one cached TabPFN context per (perceived) phase.

Inputs  = offline_eval.INPUTS = features.FEATURES (perception + encoders + known bin pose) + phase_perc.
Outputs = movement [dx, dy, dz, dyaw] from ONE TabPFNRegressor per phase on the "one row per output" table,
          gripper (close / open) from a TabPFNClassifier per phase on the same states (constant if the phase's
          context holds a single class).
Context = up to `m` random teacher states of that phase (successful episodes only), fit once with
          fit_mode="fit_with_cache"; acting is then predict() only (~55 ms on this CPU with 4 torch threads).
Learning = add rows to a phase's context and refit that phase (no gradient steps).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .expert import ExpertConfig
from .offline_eval import GRIP_THRESHOLD_MM, INPUTS, LABEL_COL, MOVE_OUT, Standardizer, long_table, make_tabpfn


class TabPFNStudent:
    def __init__(self, train: pd.DataFrame, m: int = 256, version: str = "v3.5-fast", seed: int = 0,
                 expert_cfg: ExpertConfig | None = None):
        self.cfg = expert_cfg or ExpertConfig()
        train = train[train.success].reset_index(drop=True)
        self.std = Standardizer(train[INPUTS].to_numpy(float))   # scaling from all training states
        Y = train[[LABEL_COL[o] for o in MOVE_OUT]].to_numpy(float)
        self.y_mu, self.y_sd = Y.mean(0), Y.std(0)
        self.y_sd[self.y_sd < 1e-9] = 1.0
        rng = np.random.default_rng(seed)
        self.reg, self.grip_model, self.grip_const, self.context_size = {}, {}, {}, {}
        self.fit_seconds = 0.0
        for p in sorted(train.phase_perc.unique()):
            pool = np.flatnonzero(train.phase_perc.to_numpy() == p)
            idx = rng.choice(pool, size=min(m, len(pool)), replace=False)
            d = train.iloc[idx]
            Xs = self.std(d[INPUTS].to_numpy(float))
            Ys = (d[[LABEL_COL[o] for o in MOVE_OUT]].to_numpy(float) - self.y_mu) / self.y_sd
            G = (d.a_grip.to_numpy() < GRIP_THRESHOLD_MM).astype(int)          # 1 = close
            t = time.perf_counter()
            reg = make_tabpfn("reg", version, "cpu", 1)
            reg.set_params(fit_mode="fit_with_cache")
            reg.fit(*long_table(Xs, Ys))
            self.reg[int(p)] = reg
            if G.min() == G.max():
                self.grip_const[int(p)] = int(G[0])
            else:
                clf = make_tabpfn("clf", version, "cpu", 1)
                clf.set_params(fit_mode="fit_with_cache")
                clf.fit(Xs, G)
                self.grip_model[int(p)] = clf
            self.fit_seconds += time.perf_counter() - t
            self.context_size[int(p)] = len(idx)

    def act(self, features: dict, phase_perc: int) -> tuple[np.ndarray, float]:
        """features: features.feature_vector(...) of the perceived state. Returns (action [m, m, m, rad, m], ms)."""
        t = time.perf_counter()
        x = np.array([[features[c] for c in INPUTS[:-1]] + [phase_perc]], float)
        xs = self.std(x)
        p = phase_perc if phase_perc in self.reg else min(self.reg, key=lambda k: abs(k - phase_perc))
        y = self.reg[p].predict(long_table(xs, None)[0]) * self.y_sd + self.y_mu    # mm, mm, mm, rad
        if p in self.grip_model:
            close = int(self.grip_model[p].predict(xs)[0])
        else:
            close = self.grip_const[p]
        c = self.cfg
        d = y[:3] / 1000.0
        n = float(np.linalg.norm(d))
        if n > c.max_step:                                       # same action limits as the teacher
            d *= c.max_step / n
        dyaw = float(np.clip(y[3], -c.max_dyaw, c.max_dyaw))
        grip = c.close_w if close else c.open_w
        return np.r_[d, dyaw, grip], 1e3 * (time.perf_counter() - t)
