"""Observable per-bottle state ("what the robot has learned about this bottle so far") and the
feature rows used by TabPFN. Everything here is computed from stroke observations only; the
bottle's hidden parameters are never read.
"""
import numpy as np

from sim import CAP_R

NAN = float("nan")
TWO_R = 2 * CAP_R
PROGRESS_MIN = 3.0   # deg, a stroke "moved the cap"
SLIP_MIN = 5.0       # deg, a stroke "slipped"

STATE = ["n", "seal_broken", "k_est", "tau_hold_max", "mu_lo", "mu_hi", "ktors_est",
         "tau_prog_min", "last_progress", "last_slip", "damage_seen"]
ACTION = ["F", "tau", "mu_req", "defl_pred"]
FEATURES = STATE + ACTION
TARGETS = ["progress", "slip", "damage"]


def bottle_state(history):
    """Running estimates from the strokes so far (NaN = not yet observable)."""
    s = dict(n=len(history), seal_broken=0, k_est=NAN, tau_hold_max=NAN, mu_lo=NAN, mu_hi=NAN,
             ktors_est=NAN, tau_prog_min=NAN, last_progress=NAN, last_slip=NAN, damage_seen=0)
    if not history:
        return s
    ks, mu_lo, mu_hi, kt, hold, prog = [], [], [], [], [], []
    for o in history:
        fn = max(o["fn"], 1.0)
        if o["deflection"] > 0.05:
            ks.append(o["fn"] / o["deflection"])
        moved = o["progress"] > PROGRESS_MIN
        slipped = o["slip"] > SLIP_MIN
        if moved:
            s["seal_broken"] = 1
            prog.append(o["tau"])
        if slipped:
            mu_hi.append(o["tau"] / (TWO_R * fn))           # friction could not carry tau
        elif not moved:                                       # the cap held the full torque
            hold.append(o["tau"])
            mu_lo.append(o["tau"] / (TWO_R * fn))
            if o["twist"] > 0.5:
                kt.append(o["tau"] / np.deg2rad(o["twist"]))
        s["damage_seen"] += o["damage"]
    s["k_est"] = float(np.median(ks)) if ks else NAN
    s["mu_lo"] = max(mu_lo) if mu_lo else NAN
    s["mu_hi"] = min(mu_hi) if mu_hi else NAN
    s["ktors_est"] = float(np.median(kt)) if kt else NAN
    s["tau_hold_max"] = max(hold) if hold and not s["seal_broken"] else NAN
    s["tau_prog_min"] = min(prog) if prog else NAN
    s["last_progress"] = history[-1]["progress"]
    s["last_slip"] = history[-1]["slip"]
    return s


def action_features(state, F, tau):
    F = np.asarray(F, float)
    tau = np.asarray(tau, float)
    k = state["k_est"]
    return {"F": F, "tau": tau, "mu_req": tau / (TWO_R * F),
            "defl_pred": F / k if np.isfinite(k) else np.full_like(F, NAN)}


def rows(state, F, tau):
    """Feature matrix (n, len(FEATURES)) for candidate actions F, tau (arrays)."""
    F = np.atleast_1d(F).astype(float)
    tau = np.atleast_1d(tau).astype(float)
    a = action_features(state, F, tau)
    cols = [np.full(len(F), float(state[c])) for c in STATE] + [a[c] for c in ACTION]
    return np.column_stack(cols)


def history_rows(history):
    """(X, Y) for every stroke of an episode: state *before* the stroke, action, outcome."""
    X, Y = [], []
    for i, o in enumerate(history):
        X.append(rows(bottle_state(history[:i]), o["F"], o["tau"])[0])
        Y.append([o["progress"], o["slip"], o["damage"]])
    return np.array(X), np.array(Y)
