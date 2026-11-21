"""High-level controllers. Each maps the stroke history of the current bottle to the next action
(grip force F [N], wrist torque tau [Nm]). All of them run through the same low-level safety
controller in sim.py.

  RandomExplorer      broad random strokes; only used to collect prior data on training bottles
  ClassicalAdaptive   rule-based adaptive controller (no learning across bottles): escalate the
                      torque until the cap moves, raise the grip after a slip using the friction
                      estimate, back off after high deflection
  TabPFNController    Act -> Observe -> Predict -> Optimize: TabPFN predicts progress, slip and
                      damage for 144 candidate actions from prior strokes on training bottles plus
                      this bottle's own strokes; a separate optimizer picks the best safe one
  Oracle              knows the hidden bottle parameters: evaluation-only upper bound
"""
import re
import time

import numpy as np

from features import FEATURES, PROGRESS_MIN, SLIP_MIN, TWO_R, bottle_state, history_rows, rows
from sim import F_MAX, TAU_MAX


class RandomExplorer:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, history):
        r = self.rng
        # log-uniform force, uniform torque: covers slip, hold, break and crush regimes
        return float(np.exp(r.uniform(np.log(6), np.log(F_MAX)))), float(r.uniform(0.1, TAU_MAX))


class ClassicalAdaptive:
    """Hand-tuned adaptive escalation controller (what an engineer would write without data)."""
    MU_ASSUMED = 0.5         # until a friction bound is observed
    DEFL_ALLOWED = 1.5       # mm, conservative generic deflection budget for the grip force
    GRIP_MARGIN = 1.3

    def reset(self):
        self.F, self.tau = 20.0, 0.3

    def _grip_for(self, tau, s):
        mu = s["mu_lo"] if np.isfinite(s["mu_lo"]) else self.MU_ASSUMED
        if np.isfinite(s["mu_hi"]):
            mu = min(mu, s["mu_hi"]) if np.isfinite(s["mu_lo"]) else 0.8 * s["mu_hi"]
        return self.GRIP_MARGIN * tau / (TWO_R * mu)

    def act(self, history):
        if not history:
            return self.F, self.tau
        s, o = bottle_state(history), history[-1]
        f_cap = s["k_est"] * self.DEFL_ALLOWED if np.isfinite(s["k_est"]) else F_MAX
        if o["damage"]:
            f_cap = min(f_cap, 0.8 * o["F"])
        moved, slipped = o["progress"] > PROGRESS_MIN, o["slip"] > SLIP_MIN
        if slipped:
            self.F = max(self.F * 1.4, self._grip_for(self.tau, s))
            if self.F > f_cap:           # cannot grip harder: twist more gently
                self.tau *= 0.85
        elif not moved:
            self.tau = min(self.tau * 1.3 + 0.05, TAU_MAX)
            self.F = max(self.F, self._grip_for(self.tau, s))
        elif o["progress"] < 30:         # moving but slowly: a bit more torque
            self.tau = min(self.tau * 1.15, TAU_MAX)
            self.F = max(self.F, self._grip_for(self.tau, s))
        self.F = float(np.clip(self.F, 5.0, min(f_cap, F_MAX)))
        return self.F, float(self.tau)


class Oracle:
    """Knows the hidden parameters (evaluation-only)."""

    def __init__(self, bottle):
        self.b = bottle

    def reset(self):
        pass

    def act(self, history):
        b = self.b
        broken = any(o["progress"] > PROGRESS_MIN for o in history)
        tau = min(TAU_MAX, 1.25 * b.t_break + 0.05) if not broken else min(TAU_MAX, b.t_run + 0.8)
        F = min(1.3 * tau / (TWO_R * b.mu), 0.8 * b.f_crush, F_MAX)
        return float(F), float(tau)


# ---------------------------------------------------------------- TabPFN


class TabPFNPredictor:
    """One TabPFN regressor for all three targets (target index as an extra feature), via the
    cloud API. Targets are scaled to similar ranges."""
    SCALE = np.array([60.0, 60.0, 1.0])

    def __init__(self):
        try:
            import truststore
            truststore.inject_into_ssl()
        except ImportError:
            pass
        from tabpfn_client import TabPFNRegressor
        self.cls = TabPFNRegressor

    def fit_predict(self, X, Y, Xq):
        nt = Y.shape[1]
        tag = lambda A, j: np.hstack([A, np.full((len(A), 1), j)])
        Xs = np.vstack([tag(X, j) for j in range(nt)])
        ys = np.concatenate([Y[:, j] / self.SCALE[j] for j in range(nt)])
        Xqs = np.vstack([tag(Xq, j) for j in range(nt)])
        model = self.cls(random_state=0)
        model.fit(Xs, ys)
        pred = np.asarray(model.predict(Xqs))
        return pred.reshape(nt, len(Xq)).T * self.SCALE


class TabPFNController:
    F_GRID = np.linspace(8, F_MAX, 12)
    TAU_GRID = np.linspace(0.15, TAU_MAX, 12)
    DAMAGE_MAX = 0.05            # chance constraint on predicted damage
    DAMAGE_MAX_BLIND = 0.01      # ... before any stiffness measurement
    W_SLIP, W_FORCE, W_TAU, W_DAMAGE = 0.5, 0.04, 2.0, 50.0

    def __init__(self, prior_X, prior_Y, predictor=None, n_prior=1200, seed=0, use_own=True):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(prior_X))[:n_prior]
        self.pX, self.pY = prior_X[idx], prior_Y[idx]
        self.predictor = predictor or TabPFNPredictor()
        self.use_own = use_own
        Fg, Tg = np.meshgrid(self.F_GRID, self.TAU_GRID)
        self.cF, self.cT = Fg.ravel(), Tg.ravel()
        self.log = []

    def reset(self):
        self.log = []

    def act(self, history):
        s = bottle_state(history)
        X, Y = self.pX, self.pY
        if self.use_own and history:
            hX, hY = history_rows(history)
            X, Y = np.vstack([X, hX]), np.vstack([Y, hY])
        Xq = rows(s, self.cF, self.cT)
        for attempt in range(4):
            t0 = time.perf_counter()
            try:
                pred = self.predictor.fit_predict(X, Y, Xq)
                break
            except RuntimeError as e:
                # cloud rate limit (HTTP 429, 1000 fits/hour): wait it out, then retry.
                # The wait is not counted as model latency.
                if "429" not in str(e) or attempt == 3:
                    raise
                wait = re.search(r"Retry in (\d+)s", str(e))
                wait = min(int(wait.group(1)) + 5, 900) if wait else 300
                print(f"  TabPFN rate limit, waiting {wait} s", flush=True)
                time.sleep(wait)
        latency = time.perf_counter() - t0
        prog, slip, dmg = pred[:, 0], np.clip(pred[:, 1], 0, None), np.clip(pred[:, 2], 0, 1)
        score = prog - self.W_SLIP * slip - self.W_FORCE * self.cF - self.W_TAU * self.cT - self.W_DAMAGE * dmg
        # risk-averse until the wall stiffness has been measured (the first grasp is blind)
        ok = dmg <= (self.DAMAGE_MAX if np.isfinite(s["k_est"]) else self.DAMAGE_MAX_BLIND)
        if np.isfinite(s["k_est"]):
            ok &= self.cF / s["k_est"] < 2.4         # generic deflection limit with margin
        if not ok.any():
            ok = dmg <= dmg.min() + 1e-9
        i = int(np.argmax(np.where(ok, score, -np.inf)))
        self.log.append({"latency": latency, "pred": pred[i].tolist(), "F": float(self.cF[i]),
                         "tau": float(self.cT[i])})
        return float(self.cF[i]), float(self.cT[i])
