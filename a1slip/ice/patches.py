"""Where a reflex can't help: choosing speed and route on ground with ice patches.

Scene: flat grippy ground (friction 1.0, green) with ice patches (friction 0.1,
pale blue). One large patch always lies across the straight line from A (0, 0)
to B (8, 0), plus a few small ones. Contact friction is the larger of foot and
ground, so the feet are set to 0.1 and the ground decides. The robot cannot see
the ice; it can only feel it.

TabPFN's slip model is used frozen: its 2,000 examples come from the original
flat-floor data (whole floors of one friction, never patches).

Experiment 1 - speed governor (A -> B, straight line):
  fixed speeds 0.3 / 0.55 / 0.8 / 1.0 vs a TabPFN governor that every 0.2 s
  picks the fastest of those speeds whose predicted slip risk is below a limit.
  Score: slip-seconds (time with a loaded foot sliding) and time to B.
Experiment 2 - patrol with a slip map (A -> B -> A ... 6 legs):
  the robot writes its slip risk into a map where it walks; an A* planner
  routes around high-risk cells on later legs.
    no map          straight line every leg
    detector map    risk from the slip-detector features alone (legs-vs-IMU gap over
                    0.1/0.5/2 s + foot disagreement, logistic regression) - the
                    no-TabPFN baseline
    TabPFN map      risk from TabPFN's slip model (62 features)
  Both maps are scaled the same way: 0 = typical grippy risk, 1 = typical icy risk,
  measured on the training examples.

    python -m a1slip.ice.patches governor           # -> results/ice_governor.json
    python -m a1slip.ice.patches governor_baseline  # adds the no-TabPFN detector governor
    python -m a1slip.ice.patches patrol             # -> results/ice_patrol.json
"""

import heapq
import json
import sys
import time
import warnings

import mujoco
import numpy as np
from sklearn.linear_model import LogisticRegression

from a1slip.paths import ROOT
from a1slip.models.tabpfn_risk import CollisionRisk
from a1slip.sim.robot import MAX_TURN, SMOOTHING, ensure_model, gait, wrap, yaw
from a1slip.sim.slip_dataset import CHECK_EVERY, SUSTAIN, A1Sensors, History, feature_names
from a1slip.prediction.slip_eval import DATA, splits

warnings.filterwarnings("ignore")
RESULTS = ROOT / "results"
A, B = np.array([0.0, 0.0]), np.array([8.0, 0.0])
ICE_MU, GROUND_MU = 0.1, 1.0
DECIDE = 0.2           # s between decisions
SPEEDS = [0.3, 0.55, 0.8, 1.0]
GOAL_RADIUS = 0.4
CELL = 0.2             # m, slip-map grid
X_RANGE, Y_RANGE = (-2.0, 10.0), (-4.0, 4.0)
SPREAD = 0.5           # m, a risk reading updates the map within this radius
MAP_WEIGHT = 10.0      # A* cost of a cell = 1 + MAP_WEIGHT * scaled risk
LOOKAHEAD = 0.8


def patches(seed):
    rng = np.random.default_rng(seed)
    big = (rng.uniform(3.0, 5.0), rng.uniform(-0.3, 0.3), rng.uniform(0.8, 1.4), rng.uniform(0.8, 1.3))
    small = [(rng.uniform(1.0, 7.0), rng.uniform(-3.0, 3.0), *rng.uniform(0.3, 0.6, 2)) for _ in range(3)]
    return [big] + small


def build_patch_model(seed):
    spec = mujoco.MjSpec.from_file(str(ensure_model()))
    spec.body("trunk").add_site(name="imu", pos=[0, 0, 0])
    spec.add_sensor(name="gyro", type=mujoco.mjtSensor.mjSENS_GYRO,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    spec.add_sensor(name="acc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    for i, (x, y, hx, hy) in enumerate(patches(seed)):
        spec.worldbody.add_geom(name=f"ice{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, 0.004],
                                pos=[x, y, 0.004], friction=[ICE_MU, 0.005, 0.0001],
                                rgba=[0.82, 0.92, 0.98, 1])
    for p, col in ((A, [0.95, 0.6, 0.2, 1]), (B, [0.2, 0.5, 0.95, 1])):
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.25, 0.002, 0],
                                pos=[*p, 0.009], rgba=col, contype=0, conaffinity=0)
    m = spec.compile()
    m.mat_rgba[m.material("groundplane").id] = [0.45, 0.70, 0.40, 1]   # grippy ground: green
    return m


class PatchSensors(A1Sensors):
    """Feet at friction 0.1 and no contact priority, so the ground's friction decides."""

    def __init__(self, model):
        super().__init__(model)
        model.geom_priority[self.feet] = 0
        model.geom_friction[self.feet, 0] = ICE_MU
        model.opt.gravity[:] = [0, 0, -9.81]


def on_ice(model, x, y):
    for g in range(model.ngeom):
        if model.geom(g).name.startswith("ice"):
            px, py = model.geom_pos[g][:2]
            hx, hy = model.geom_size[g][:2]
            if abs(x - px) <= hx and abs(y - py) <= hy:
                return True
    return False


# ---------------------------------------------------------------- slip map + A*
class SlipMap:
    def __init__(self):
        self.nx = int(round((X_RANGE[1] - X_RANGE[0]) / CELL))
        self.ny = int(round((Y_RANGE[1] - Y_RANGE[0]) / CELL))
        self.sum = np.zeros((self.nx, self.ny))
        self.n = np.zeros((self.nx, self.ny))
        xs = X_RANGE[0] + (np.arange(self.nx) + 0.5) * CELL
        ys = Y_RANGE[0] + (np.arange(self.ny) + 0.5) * CELL
        self.cx, self.cy = np.meshgrid(xs, ys, indexing="ij")

    def add(self, x, y, risk01):
        m = np.hypot(self.cx - x, self.cy - y) <= SPREAD
        self.sum[m] += risk01
        self.n[m] += 1

    def risk(self):
        return np.where(self.n > 0, self.sum / np.maximum(self.n, 1), 0.0)

    def cell(self, p):
        return (int(np.clip((p[0] - X_RANGE[0]) / CELL, 0, self.nx - 1)),
                int(np.clip((p[1] - Y_RANGE[0]) / CELL, 0, self.ny - 1)))

    def plan(self, start, goal):
        cost = 1.0 + MAP_WEIGHT * self.risk()
        s, g = self.cell(start), self.cell(goal)
        best, parent, heap = {s: 0.0}, {}, [(0.0, 0.0, s)]
        nb = [(i, j, np.hypot(i, j)) for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j]
        while heap:
            _, d, c = heapq.heappop(heap)
            if c == g:
                break
            if d > best.get(c, np.inf):
                continue
            for i, j, st in nb:
                n = (c[0] + i, c[1] + j)
                if 0 <= n[0] < self.nx and 0 <= n[1] < self.ny:
                    nd = d + st * 0.5 * (cost[c] + cost[n])
                    if nd < best.get(n, np.inf):
                        best[n], parent[n] = nd, c
                        heapq.heappush(heap, (nd + np.hypot(n[0] - g[0], n[1] - g[1]), nd, n))
        cells = [g]
        while cells[-1] in parent:
            cells.append(parent[cells[-1]])
        return np.array([[self.cx[c], self.cy[c]] for c in reversed(cells)])


# ---------------------------------------------------------------- simulation
def walk(model, sensors, goals, choose, max_time=60.0, on_frame=None):
    """Walk through `goals` in order. choose(features, history, data, target) -> (forward, turn)
    is called every DECIDE s; target is the current goal. Returns per-leg stats."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    dt = model.opt.timestep
    alpha = dt / SMOOTHING
    hist = History(sensors, CHECK_EVERY * dt)
    cmd, target = np.zeros(2), np.zeros(2)
    streak = np.zeros(4, int)
    sliding = np.zeros(4, bool)
    leg, leg_start, legs = 0, 0.0, []
    slip_s = ice_s = path = 0.0
    last_xy = data.qpos[:2].copy()
    k, fell = 0, False
    while data.time < max_time and leg < len(goals):
        if k % int(round(DECIDE / dt)) == 0:
            f = sensors.features(data, cmd)
            h, _ = hist.features()
            target = np.asarray(choose(f, h, data, goals[leg]), float)
        cmd += alpha * (target - cmd)
        data.ctrl[:] = gait(data.time, *cmd)
        mujoco.mj_step(model, data)
        k += 1
        if k % CHECK_EVERY == 0:
            forces = sensors.foot_forces(data)
            raw = sensors.sliding_feet(data, forces)
            streak = np.where(raw, streak + 1, 0)
            sliding = streak >= SUSTAIN
            hist.update(data, forces, sliding.any())
            xy = data.qpos[:2].copy()
            path += float(np.linalg.norm(xy - last_xy))
            last_xy = xy
            slip_s += CHECK_EVERY * dt * sliding.any()
            ice_s += CHECK_EVERY * dt * on_ice(model, *xy)
            if np.linalg.norm(xy - goals[leg]) < GOAL_RADIUS:
                legs.append(dict(leg=leg, time=data.time - leg_start, slip_s=slip_s, ice_s=ice_s,
                                 path=path))
                leg, leg_start, slip_s, ice_s, path = leg + 1, data.time, 0.0, 0.0, 0.0
            if data.qpos[2] < 0.15:
                fell = True
                break
        if on_frame is not None and on_frame(data, sliding, leg) is False:
            break
    return dict(legs=legs, completed=len(legs), fell=fell, time=float(data.time))


def steer(data, goal, forward):
    d = goal - data.qpos[:2]
    err = wrap(np.arctan2(d[1], d[0]) - yaw(data))
    return forward * max(0.3, np.cos(err)), float(np.clip(1.5 * err, -MAX_TURN, MAX_TURN))


def models():
    """Frozen TabPFN slip model + detector-only logistic baseline, and their grippy/icy
    reference risks on the training examples (for scaling the maps)."""
    d = dict(np.load(DATA))
    X = np.hstack([d["features"], d["history"]])
    y = d["slip_next"].astype(int)
    tr = splits(d)["train"]
    tab = CollisionRisk().fit(X[tr], y[tr], n=2000, pos_fraction=max(y[tr].mean(), 0.25))
    names = list(d["history_names"])
    gap = [names.index(n) for n in ("leg_imu_gap_0.1s", "leg_imu_gap_0.5s", "leg_imu_gap_2s",
                                    "feet_disagree_0.5s")]
    det = LogisticRegression(max_iter=2000).fit(d["history"][tr][:, gap], y[tr])
    rng = np.random.default_rng(0)
    ref = {}
    for name, fn in (("TabPFN", lambda Z: tab.predict(Z)),
                     ("detector", lambda Z: det.predict_proba(Z[:, [51 + i for i in gap]])[:, 1])):
        grip = rng.choice(np.flatnonzero(d["mu"] >= 0.5), 300, replace=False)
        ice = rng.choice(np.flatnonzero(d["mu"] < 0.2), 300, replace=False)
        ref[name] = (float(np.median(fn(X[grip]))), float(np.median(fn(X[ice]))))
    print("reference risks (grippy, icy):", ref, flush=True)
    return tab, det, gap, ref


def governor_experiment(seeds=range(5)):
    tab, _, _, _ = models()
    fi = feature_names().index("cmd_forward")
    rows = []
    for seed in seeds:
        model = build_patch_model(seed)
        sensors = PatchSensors(model)
        strategies = {f"fixed {s}": (lambda s: lambda f, h, d, g: steer(d, g, s))(s) for s in SPEEDS}
        for limit in (0.5, 0.65):
            def gov(f, h, d, g, limit=limit):
                cands = np.repeat(np.r_[f, h][None], len(SPEEDS), 0)
                cands[:, fi] = SPEEDS
                p = tab.predict(cands)
                ok = [s for s, pi in zip(SPEEDS, p) if pi < limit]
                return steer(d, g, max(ok) if ok else SPEEDS[int(np.argmin(p))])
            strategies[f"TabPFN governor (risk < {limit})"] = gov
        for name, choose in strategies.items():
            out = walk(model, sensors, [B], choose, max_time=60)
            leg = out["legs"][0] if out["legs"] else None
            rows.append(dict(seed=seed, strategy=name, reached=bool(leg), fell=out["fell"],
                             time=leg["time"] if leg else None, slip_s=leg["slip_s"] if leg else None,
                             ice_s=leg["ice_s"] if leg else None))
            print(f"seed {seed} {name:30s} " + (f"time {leg['time']:5.1f} s  slip {leg['slip_s']:4.1f} s  "
                  f"on ice {leg['ice_s']:4.1f} s" if leg else "did not reach B"), flush=True)
    (RESULTS / "ice_governor.json").write_text(json.dumps(rows, indent=1))
    print("\nmean over seeds:")
    for name in dict.fromkeys(r["strategy"] for r in rows):
        r = [x for x in rows if x["strategy"] == name and x["reached"]]
        print(f"  {name:30s} reached {len(r)}/{len(seeds)}  time {np.mean([x['time'] for x in r]):5.1f} s  "
              f"slip {np.mean([x['slip_s'] for x in r]):4.1f} s")


def governor_baseline(seeds=range(5)):
    """No-TabPFN baseline for the governor: crawl (0.3) while the slip-detector-only
    logistic model says 'icy' (scaled risk > 0.5), else full speed (1.0)."""
    _, det, gap, ref = models()
    lo, hi = ref["detector"]
    path = RESULTS / "ice_governor.json"
    rows = [r for r in json.loads(path.read_text()) if not r["strategy"].startswith("detector")]
    for seed in seeds:
        model = build_patch_model(seed)
        sensors = PatchSensors(model)

        def choose(f, h, d, g):
            p = float(det.predict_proba([h[gap]])[0, 1])
            return steer(d, g, 0.3 if (p - lo) / (hi - lo) > 0.5 else 1.0)
        out = walk(model, sensors, [B], choose, max_time=60)
        leg = out["legs"][0] if out["legs"] else None
        rows.append(dict(seed=seed, strategy="detector governor (no TabPFN)", reached=bool(leg),
                         fell=out["fell"], time=leg["time"] if leg else None,
                         slip_s=leg["slip_s"] if leg else None, ice_s=leg["ice_s"] if leg else None))
        print(f"seed {seed} detector governor " + (f"time {leg['time']:5.1f} s  slip {leg['slip_s']:4.1f} s"
              if leg else "did not reach B"), flush=True)
    path.write_text(json.dumps(rows, indent=1))


def governor_xgb(seeds=range(5)):
    """Same governor with XGBoost (all 10,290 training rows, same 62 features) instead of TabPFN."""
    from xgboost import XGBClassifier
    d = dict(np.load(DATA))
    X = np.hstack([d["features"], d["history"]])
    y = d["slip_next"].astype(int)
    tr = splits(d)["train"]
    xgb = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8).fit(X[tr], y[tr])
    fi = feature_names().index("cmd_forward")
    path = RESULTS / "ice_governor.json"
    rows = [r for r in json.loads(path.read_text()) if not r["strategy"].startswith("XGBoost")]
    for seed in seeds:
        model = build_patch_model(seed)
        sensors = PatchSensors(model)
        for limit in (0.5, 0.65):
            def gov(f, h, dd, g, limit=limit):
                cands = np.repeat(np.r_[f, h][None], len(SPEEDS), 0)
                cands[:, fi] = SPEEDS
                p = xgb.predict_proba(cands)[:, 1]
                ok = [s for s, pi in zip(SPEEDS, p) if pi < limit]
                return steer(dd, g, max(ok) if ok else SPEEDS[int(np.argmin(p))])
            out = walk(model, sensors, [B], gov, max_time=60)
            leg = out["legs"][0] if out["legs"] else None
            rows.append(dict(seed=seed, strategy=f"XGBoost governor (risk < {limit})", reached=bool(leg),
                             fell=out["fell"], time=leg["time"] if leg else None,
                             slip_s=leg["slip_s"] if leg else None, ice_s=leg["ice_s"] if leg else None))
            print(f"seed {seed} XGBoost governor < {limit} " + (f"time {leg['time']:5.1f} s  slip "
                  f"{leg['slip_s']:4.1f} s" if leg else "did not reach B"), flush=True)
    path.write_text(json.dumps(rows, indent=1))


def patrol_controller(kind, tab, det, gap, ref):
    """kind 'no' | 'detector' | 'TabPFN' -> (choose, slip map or None)."""
    if kind == "no":
        return (lambda f, h, d, g: steer(d, g, 1.0)), None
    smap, state = SlipMap(), {"path": None, "goal": None, "t": -1.0}
    lo, hi = ref[kind]

    def choose(f, h, d, g):
        p = float(tab.predict(np.r_[f, h])[0]) if kind == "TabPFN" else \
            float(det.predict_proba([h[gap]])[0, 1])
        smap.add(*d.qpos[:2], np.clip((p - lo) / (hi - lo), 0, 1))
        if state["goal"] is not g or d.time - state["t"] >= 0.5:
            state["path"], state["goal"], state["t"] = smap.plan(d.qpos[:2], g), g, d.time
        path = state["path"]
        far = np.linalg.norm(path - d.qpos[:2], axis=1) >= LOOKAHEAD
        wp = path[np.argmax(far)] if far.any() else g
        return steer(d, wp, 1.0)
    choose.smap, choose.state = smap, state
    return choose, smap


def patrol_experiment(seeds=range(5), n_legs=6):
    tab, det, gap, ref = models()
    goals = [B if i % 2 == 0 else A for i in range(n_legs)]
    rows = []
    for seed in seeds:
        model = build_patch_model(seed)
        sensors = PatchSensors(model)

        for kind in ("no map", "detector map", "TabPFN map"):
            t = time.time()
            choose, _ = patrol_controller(kind.replace(" map", ""), tab, det, gap, ref)
            out = walk(model, sensors, goals, choose, max_time=240)
            for leg in out["legs"]:
                rows.append(dict(seed=seed, strategy=kind, **leg))
            print(f"seed {seed} {kind:13s} legs {out['completed']}/{n_legs}  slip per leg "
                  f"{[round(l['slip_s'], 1) for l in out['legs']]}  ice per leg "
                  f"{[round(l['ice_s'], 1) for l in out['legs']]}  ({time.time() - t:.0f}s)", flush=True)
    (RESULTS / "ice_patrol.json").write_text(json.dumps(rows, indent=1))
    print("\nmean slip-seconds and time per leg (legs 1-2 = learning, 3-6 = after):")
    for kind in ("no map", "detector map", "TabPFN map"):
        r = [x for x in rows if x["strategy"] == kind]
        for name, sel in (("legs 1-2", lambda x: x["leg"] < 2), ("legs 3-6", lambda x: x["leg"] >= 2)):
            s = [x for x in r if sel(x)]
            print(f"  {kind:13s} {name}: slip {np.mean([x['slip_s'] for x in s]):4.1f} s  on ice "
                  f"{np.mean([x['ice_s'] for x in s]):4.1f} s  time {np.mean([x['time'] for x in s]):5.1f} s")


if __name__ == "__main__":
    {"governor": governor_experiment, "governor_baseline": governor_baseline,
     "governor_xgb": governor_xgb, "patrol": patrol_experiment}[sys.argv[1]]()
