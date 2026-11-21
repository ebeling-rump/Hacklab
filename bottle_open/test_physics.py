"""Physics and pipeline sanity tests (pytest-compatible; also runnable as `python test_physics.py`).

They check that the simplified physics produces the regimes the controllers must learn:
hold below breakaway, break above it, slip with too little grip, damage above the crush limit,
torsional wind-up, thread lift, and that features never read hidden parameters.
"""
import numpy as np

from features import FEATURES, bottle_state, history_rows
from sim import OPEN_TURN, Bottle, BottleSim, SafetyController, run_episode, sample_bottle, TEST_SEEDS, TRAIN_SEEDS

B = Bottle(mu=0.7, t_break=0.8, t_run=0.25, k_wall=40, d_lim=2.0, k_tors=6)   # f_need 35.7 N, f_crush 80 N


def test_seal_holds_below_breakaway():
    s = BottleSim(B)
    o = s.stroke(50, 0.6)
    assert not s.seal_broken and abs(o["progress"]) < 3 and o["slip"] < 3
    assert abs(o["twist"] - np.rad2deg(0.6 / B.k_tors)) < 1.5      # bottle winds up, tau / k_tors


def test_seal_breaks_above_breakaway_and_cap_turns():
    s = BottleSim(B)
    o = s.stroke(50, 1.0)
    assert s.seal_broken and o["progress"] > 20 and o["slip"] < 3


def test_slip_when_grip_too_weak():
    s = BottleSim(B)
    o = s.stroke(20, 1.0)            # grip capacity 0.7 * 20 * 0.032 = 0.45 Nm < 0.8 Nm
    assert o["slip"] > 5 and not s.seal_broken and "slip_stop" in o["safety"]   # watchdog cuts the torque


def test_running_torque_after_break():
    s = BottleSim(B)
    s.stroke(50, 1.0)
    o = s.stroke(50, 0.4)            # 0.4 Nm > t_run 0.25 keeps it turning
    assert o["progress"] > 5
    o = s.stroke(50, 0.15)           # below t_run: it stays
    assert o["progress"] < 3


def test_damage_above_crush_force_only():
    assert BottleSim(B).stroke(70, 0.3)["damage"] == 0
    assert BottleSim(B).stroke(90, 0.3)["damage"] == 1


def test_thread_lifts_cap():
    s = BottleSim(B)
    for _ in range(3):
        s.stroke(50, 1.0)
    turn = s.cap_rel()
    lift = s.d.qpos[s.q_lift]
    assert turn > 1.0 and abs(lift - 0.003 * turn / (2 * np.pi)) < 2e-4


def test_safety_layer_clamps_and_cuts():
    sc = SafetyController()
    F, tau = sc.filter(500, 9, slip_speed=0, twist=0, deflection=0.1, dt=5e-4)
    assert F == 100 and tau == 2.0
    for _ in range(50):
        F, tau = sc.filter(50, 1, slip_speed=10, twist=0, deflection=0.1, dt=5e-4)
    assert tau == 0 and "slip_stop" in sc.events
    F, tau = SafetyController().filter(50, 1, slip_speed=0, twist=0, deflection=3.0, dt=5e-4)
    assert F == 0 and tau == 0


def test_bottles_openable_and_splits_disjoint():
    assert not set(TRAIN_SEEDS) & set(TEST_SEEDS)
    for seed in list(TEST_SEEDS)[:10]:
        b = sample_bottle(seed)
        assert b.f_need * 1.2 < 0.85 * b.f_crush


def test_features_use_observations_only():
    class Fixed:
        def reset(self): pass
        def act(self, h): return 50.0, 1.0
    ep = run_episode(B, Fixed(), max_strokes=3)
    X, Y = history_rows(ep["history"])
    assert X.shape == (3, len(FEATURES)) and Y.shape == (3, 3)
    s = bottle_state(ep["history"])
    assert s["seal_broken"] == 1 and abs(s["k_est"] - B.k_wall) < 3     # stiffness is observable


def test_oracle_opens():
    from controllers import Oracle
    ep = run_episode(B, Oracle(B))
    assert ep["opened"] and len(ep["history"]) <= 10


if __name__ == "__main__":
    import sys
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("PASS", name)
            except AssertionError as e:
                fails += 1; print("FAIL", name, e)
    sys.exit(fails)
