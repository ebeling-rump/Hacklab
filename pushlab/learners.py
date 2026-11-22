"""Data-driven dynamics models sharing one interface.

    model.reset(shape, descriptor)       new (unknown) object
    model.fit(Q, Y)                      all pushes observed so far on this object
    model.predict(Q) -> mean, std        (N, 3) each: body-frame (dx, dy, dtheta)

Learners: GP (Bauza & Rodriguez-style, ARD-RBF), TabPFN (in-context, no gradient updates).
Wrappers: residual-over-physics, physics-informed features, cross-object transfer.
"""
from __future__ import annotations

import time
import warnings

import numpy as np

from .actions import features, concat
from .physics_model import PhysicsModel

PRIOR_STD = np.array([0.03, 0.03, 0.25])
FEAT_SCALE = np.array([0.1, 0.1, 1, 1, 1, 1, 0.5, 0.05, 0.1, 0.2])
DESC_SCALE = np.array([0.1, 0.1, 0.01, 0.4, 0.05, 0.3])
Y_SCALE = np.array([0.03, 0.03, 0.3])


class Timer:
    def __init__(self):
        self.fit_s, self.pred_s, self.n_fit, self.n_pred = 0.0, 0.0, 0, 0


class Learner:
    name = "learner"
    min_n = 2

    def __init__(self, phys_feats=False, with_desc=False):
        self.phys_feats = phys_feats
        self.with_desc = with_desc
        self.timer = Timer()
        self.phys = None

    def reset(self, shape, descriptor=None):
        self.shape = shape
        self.descriptor = descriptor
        self.n = 0
        if self.phys_feats:
            self.phys = PhysicsModel(fit_params=False).reset(shape)
        return self

    def X(self, Q):
        X = features(Q, with_desc=self.with_desc) / np.concatenate(
            [FEAT_SCALE, DESC_SCALE if (self.with_desc and "desc" in Q) else []])
        if self.phys_feats:
            X = np.column_stack([X, self.phys.nominal(Q) / Y_SCALE])
        if "extra" in Q and not getattr(self, "blind", False):
            X = np.column_stack([X, Q["extra"]])
        return X

    def fit(self, Q, Y):
        self.n = len(Y)
        if self.n < self.min_n:
            return self
        t = time.perf_counter()
        self._fit(self.X(Q), np.asarray(Y) / Y_SCALE)
        self.timer.fit_s += time.perf_counter() - t
        self.timer.n_fit += 1
        return self

    def predict(self, Q):
        N = len(Q["phi"])
        if self.n < self.min_n:
            return np.zeros((N, 3)), np.tile(PRIOR_STD, (N, 1))
        t = time.perf_counter()
        m, s = self._predict(self.X(Q))
        self.timer.pred_s += time.perf_counter() - t
        self.timer.n_pred += 1
        return m * Y_SCALE, s * Y_SCALE


class GPLearner(Learner):
    name = "gp"

    def __init__(self, restarts=2, **kw):
        super().__init__(**kw)
        self.restarts = restarts

    def _fit(self, X, Y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
        self.gps = []
        F = X.shape[1]
        for j in range(Y.shape[1]):
            k = C(1.0, (1e-2, 1e2)) * RBF(np.ones(F), (0.05, 50.0)) + WhiteKernel(0.01, (1e-5, 1.0))
            gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=self.restarts,
                                          random_state=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(X, Y[:, j])
            self.gps.append(gp)

    def _predict(self, X):
        ms, ss = [], []
        for gp in self.gps:
            m, s = gp.predict(X, return_std=True)
            ms.append(m); ss.append(s)
        return np.column_stack(ms), np.column_stack(ss)


_TABPFN_CACHE = {}


class TabPFNLearner(Learner):
    name = "tabpfn"

    def __init__(self, n_estimators=2, version="v2", **kw):
        super().__init__(**kw)
        self.n_estimators = n_estimators
        self.version = version

    def _make(self):
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        return TabPFNRegressor.create_default_for_version(
            ModelVersion(self.version), n_estimators=self.n_estimators, device="cpu", random_state=0,
            ignore_pretraining_limits=True)

    def _fit(self, X, Y):
        self.models = []
        for j in range(Y.shape[1]):
            m = self._make()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y = Y[:, j]
                if np.ptp(y) < 1e-9:  # TabPFN dislikes constant targets
                    y = y + np.random.default_rng(j).normal(0, 1e-4, len(y))
                m.fit(X, y)
            self.models.append(m)

    def _predict(self, X):
        ms, ss = [], []
        for m in self.models:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = m.predict(X, output_type="main")
            q = np.asarray(out["quantiles"])  # default quantiles 0.1 ... 0.9
            ms.append(np.asarray(out["mean"]))
            ss.append(np.maximum((q[-1] - q[0]) / 2.563, 1e-4))
        return np.column_stack(ms), np.column_stack(ss)


class TabPFNAPILearner(TabPFNLearner):
    """TabPFN through Prior Labs' cloud API (tabpfn-client; server default model, v3.5 at time of
    writing). Reads the TABPFN_TOKEN environment variable; never prints it. Each fitted target
    costs one metered predict call per planning step."""
    name = "tabpfn_api"
    calls = 0  # class-wide counter of predict requests (for quota bookkeeping)

    def __init__(self, **kw):
        super().__init__(**kw)

    def _make(self):
        try:
            import truststore  # Avast Web Shield re-signs TLS; use the OS store
            truststore.inject_into_ssl()
        except ImportError:
            pass
        from tabpfn_client import TabPFNRegressor
        return TabPFNRegressor(random_state=0)

    @staticmethod
    def _retry(fn, *a, **kw):
        """The API allows 60 fits per minute; back off and retry on HTTP 429."""
        import time
        for k in range(8):
            try:
                return fn(*a, **kw)
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__} {e}"
                transient = any(t in msg for t in ("429", "Protocol", "Connect", "Timeout", "timed out",
                                                   "502", "503", "504", "Remote"))
                if not transient or k == 7:
                    raise
                time.sleep(15)

    def _fit(self, X, Y):
        self.models = []
        for j in range(Y.shape[1]):
            m = self._make()
            y = Y[:, j]
            if np.ptp(y) < 1e-9:
                y = y + np.random.default_rng(j).normal(0, 1e-4, len(y))
            self._retry(m.fit, X, y)
            self.models.append(m)

    def _predict(self, X):
        ms, ss = [], []
        for m in self.models:
            out = self._retry(m.predict, X, output_type="main")
            TabPFNAPILearner.calls += 1
            q = np.asarray(out["quantiles"])
            ms.append(np.asarray(out["mean"]))
            ss.append(np.maximum((q[-1] - q[0]) / 2.563, 1e-4))
        return np.column_stack(ms), np.column_stack(ss)


class Residual:
    """Learner on the residual of the (nominal or fitted) physics model."""

    def __init__(self, learner: Learner, refit_physics=False, name=None):
        self.learner = learner
        self.refit = refit_physics
        self.name = name or f"{learner.name}_res"
        self.timer = learner.timer

    def reset(self, shape, descriptor=None):
        self.phys = PhysicsModel(fit_params=self.refit, n_samples=0).reset(shape)
        self.phys_unc = PhysicsModel(fit_params=self.refit, n_samples=8).reset(shape)
        self.learner.reset(shape, descriptor)
        return self

    def fit(self, Q, Y):
        if self.refit:
            self.phys.fit(Q, Y)
            self.phys_unc.theta, self.phys_unc.cov, self.phys_unc.noise, self.phys_unc.n =                 self.phys.theta, self.phys.cov, self.phys.noise, self.phys.n
        base = self.phys.predict(Q)[0]
        self.learner.fit(Q, np.asarray(Y) - base)
        self.n = len(Y)
        return self

    def predict(self, Q):
        base = self.phys.predict(Q)[0]
        m, s = self.learner.predict(Q)
        if self.learner.n < self.learner.min_n:
            # not enough data for the learner yet: use the physics posterior (Laplace) uncertainty
            s = self.phys_unc.predict(Q)[1]
        return base + m, s


class Transfer:
    """Cross-object transfer: pooled context of previous objects + the new object's pushes.

    Rows carry the observable object descriptor and an indicator `is_target`, so the learner can
    use other objects' experience as a prior and the target rows to specialise.
    Optionally a residual over nominal physics.
    """

    def __init__(self, learner: Learner, source: list, residual=True, name=None, max_source=400, seed=0,
                 use_ground=True):
        self.use_ground = use_ground    # keep observable ground features (Q["extra"]) if present
        self.learner = learner
        self.learner.with_desc = True
        self.source = source            # list of (shape, descriptor, Q, Y)
        self.residual = residual
        self.max_source = max_source
        self.rng = np.random.default_rng(seed)
        self.name = name or f"{learner.name}_transfer"
        self.timer = learner.timer
        self._src_cache = None

    def _base(self, shape, Q, fit_data=None):
        if not self.residual:
            return np.zeros((len(Q["phi"]), 3))
        if self.residual == "sysid" and fit_data is not None:
            cache = getattr(self, "_pm_cache", None)
            if cache is not None and cache[0] is fit_data:
                pm = cache[1]
            else:
                pm = PhysicsModel(fit_params=True, n_samples=0).reset(shape)
                pm.fit(*fit_data)
                self._pm_cache = (fit_data, pm)
            return pm._pred(Q, pm.theta)
        return PhysicsModel(fit_params=False, n_samples=0).reset(shape).nominal(Q)

    def _source_rows(self):
        if self._src_cache is None:
            Qs, Ys = [], []
            per = max(1, self.max_source // max(len(self.source), 1))
            for shape, desc, Q, Y in self.source:
                idx = self.rng.permutation(len(Y))[:per]
                q = {k: v[idx] for k, v in Q.items() if isinstance(v, np.ndarray) and len(v) == len(Y)}
                q["desc"] = np.repeat(np.asarray(desc)[None], len(idx), 0)
                flag = np.zeros((len(idx), 1))
                q["extra"] = np.column_stack([Q["extra"][idx], flag]) if (self.use_ground and "extra" in Q) else flag
                Qs.append(q)
                Ys.append(Y[idx] - self._base(shape, q, (q, Y[idx]) if self.residual == "sysid" else None))
            self._src_cache = (concat(Qs), np.concatenate(Ys))
        return self._src_cache

    def reset(self, shape, descriptor):
        self._fit_data = None
        self.shape, self.descriptor = shape, np.asarray(descriptor)
        self.learner.reset(shape, descriptor)
        self.learner.min_n = 2
        Qs, Ys = self._source_rows()
        self.learner.fit(Qs, Ys)
        self.n = 0
        return self

    def _tag(self, Q):
        q = dict(Q)
        q["desc"] = np.repeat(self.descriptor[None], len(Q["phi"]), 0)
        flag = np.ones((len(Q["phi"]), 1))
        q["extra"] = np.column_stack([Q["extra"], flag]) if (self.use_ground and "extra" in Q) else flag
        return q

    def fit(self, Q, Y):
        Qs, Ys = self._source_rows()
        self.n = len(Y)
        self._fit_data = (Q, np.asarray(Y)) if self.n else None
        if self.n:
            qt = self._tag(Q)
            q = concat([Qs, {k: qt[k] for k in Qs}])
            y = np.concatenate([Ys, np.asarray(Y) - self._base(self.shape, Q, self._fit_data)])
        else:
            q, y = Qs, Ys
        self.learner.fit(q, y)
        return self

    def predict(self, Q):
        m, s = self.learner.predict(self._tag(Q))
        return self._base(self.shape, Q, getattr(self, "_fit_data", None)) + m, s


class PhysicsWrapper:
    """Adapter so PhysicsModel has the same reset signature/timer as learners."""

    def __init__(self, fit_params=True, name=None, prior_mean=None):
        self.m = PhysicsModel(fit_params=fit_params)
        self.name = name or ("physics_sysid" if fit_params else "physics_nominal")
        self.timer = Timer()
        self.prior_mean = prior_mean

    def reset(self, shape, descriptor=None):
        self.m.reset(shape, prior_mean=self.prior_mean)
        return self

    def fit(self, Q, Y):
        t = time.perf_counter()
        self.m.fit(Q, Y)
        self.timer.fit_s += time.perf_counter() - t
        self.timer.n_fit += 1
        self.n = len(Y)
        return self

    def predict(self, Q):
        t = time.perf_counter()
        out = self.m.predict(Q)
        self.timer.pred_s += time.perf_counter() - t
        self.timer.n_pred += 1
        return out


def make_model(name: str, source=None, n_estimators=2):
    if name == "physics_nominal":
        return PhysicsWrapper(fit_params=False)
    if name == "physics_sysid":
        return PhysicsWrapper(fit_params=True)
    if name == "gp":
        return GPLearner()
    if name == "gp_res":
        return Residual(GPLearner(), name="gp_res")
    if name == "tabpfn":
        return TabPFNLearner(n_estimators=n_estimators)
    if name == "tabpfn_res":
        return Residual(TabPFNLearner(n_estimators=n_estimators), name="tabpfn_res")
    if name == "tabpfn_physfeat":
        m = TabPFNLearner(n_estimators=n_estimators, phys_feats=True)
        m.name = "tabpfn_physfeat"
        return m
    if name == "gp_physfeat":
        m = GPLearner(phys_feats=True)
        m.name = "gp_physfeat"
        return m
    if name == "tabpfn_hybrid":
        return Residual(TabPFNLearner(n_estimators=n_estimators), refit_physics=True, name=name)
    if name == "gp_hybrid":
        return Residual(GPLearner(), refit_physics=True, name=name)
    if name in ("gp_transfer_hybrid_blind", "tabpfn_transfer_hybrid_blind"):  # transfer without ground features
        m = make_model(name.replace("_blind", ""), source, n_estimators)
        m.use_ground = False
        m.name = name
        return m
    if name in ("gp_hybrid_blind", "tabpfn_hybrid_blind"):  # same model, ignores ground features
        m = make_model(name.replace("_blind", ""), source, n_estimators)
        m.learner.blind = True
        m.name = name
        return m
    if name == "tabpfn_api_hybrid":
        return Residual(TabPFNAPILearner(), refit_physics=True, name=name)
    if name == "tabpfn_api_transfer_hybrid":
        return Transfer(TabPFNAPILearner(), source, residual="sysid", name=name, max_source=300)
    if name == "tabpfn_transfer_raw":
        return Transfer(TabPFNLearner(n_estimators=n_estimators), source, residual=False, name=name)
    if name == "tabpfn_transfer_hybrid":
        return Transfer(TabPFNLearner(n_estimators=n_estimators), source, residual="sysid", name=name, max_source=300)
    if name == "gp_transfer_hybrid":
        return Transfer(GPLearner(restarts=0), source, residual="sysid", name=name, max_source=150)
    if name == "tabpfn_transfer":
        return Transfer(TabPFNLearner(n_estimators=n_estimators), source, residual=True, name=name)
    if name == "gp_transfer":
        return Transfer(GPLearner(restarts=0), source, residual=True, name=name, max_source=150)
    raise ValueError(name)
