"""Outdoor course: grass, wet grass, sand, gravel, mud, water, a dirt path, trees and rocks.

A 3D MuJoCo world built from 0.5 m terrain tiles. Each terrain type has its own
friction (and sand/mud/water are soft), gravel has small bumps, trees and rocks
are solid obstacles. The robot still cannot see friction: it only feels slips.

Same approach as the ice patches, plus one new step:
  1. FELT   - every 0.2 s the robot writes how much its feet slipped into a map
              where it walks (in simulation from the contacts; on a real robot from
              the legs-vs-IMU slip detector). Note: the frozen body-sensor TabPFN slip
              model from the flat floors does NOT transfer to these terrains (it rates
              water lowest, because the robot is stuck there), so it is not used here.
  2. SEEN   - a second, tiny TabPFN learns *on this site, in context* what
              slippery ground looks like: its examples are (what the tile looks
              like to a camera: colour + roughness, what the robot felt there).
              It then predicts the risk of every tile the robot has not walked on.
              No training, no data from other sites.
A* routes around high-risk cells and around trees and rocks (obstacles are
assumed known, e.g. from lidar). Patrol A -> B -> A -> B.

Strategies (same layouts, same start):
    obstacles only          A* around trees/rocks, ignores slipperiness
    felt map                + the robot's own TabPFN slip map (ice-patch method)
    felt + seen (TabPFN)    + TabPFN predicting unvisited tiles from appearance
    felt + seen (XGBoost)   same, XGBoost on the same few rows

    python -m a1slip.ice.outdoor calibrate    # true slip rate of each terrain type -> results/outdoor_types.json
    python -m a1slip.ice.outdoor experiment   # all strategies x layouts (parallel) -> results/outdoor/*.json
    python -m a1slip.ice.outdoor summary      # table
    python -m a1slip.ice.outdoor figure       # map figure -> results/figures/outdoor_maps.png
"""

import heapq
import json
import os
import subprocess
import sys
import time
import warnings

import mujoco
import numpy as np

from a1slip.paths import ROOT
from a1slip.ice import patches as P
from a1slip.models.tabpfn_risk import CollisionRisk
from a1slip.sim.robot import ensure_model

warnings.filterwarnings("ignore")
RESULTS = ROOT / "results"
OUTDIR = RESULTS / "outdoor"

A, B = np.array([0.0, 0.0]), np.array([10.0, 0.0])
X0, X1, Y0, Y1 = -1.0, 11.0, -3.5, 3.5
TILE = 0.5
NX, NY = int((X1 - X0) / TILE), int((Y1 - Y0) / TILE)          # 24 x 14 tiles
CELL = 0.25                                                    # planning grid
CX, CY = int((X1 - X0) / CELL), int((Y1 - Y0) / CELL)          # 48 x 28 cells
INFLATE = 0.35          # m added around obstacles for the planner
MAP_WEIGHT = 10.0
FELT_SPREAD = 0.35
LOOKAHEAD = 0.8
LEGS = 4

# name: friction, colour, contact softness (solref time constant), bumps
TYPES = {
    "path":      (0.90, [0.74, 0.60, 0.42], 0.02, False),
    "grass":     (0.75, [0.40, 0.66, 0.30], 0.02, False),
    "wet grass": (0.35, [0.24, 0.46, 0.26], 0.02, False),
    "gravel":    (0.60, [0.56, 0.56, 0.54], 0.02, True),
    "sand":      (0.45, [0.90, 0.82, 0.58], 0.05, False),
    "mud":       (0.20, [0.40, 0.28, 0.18], 0.06, False),
    "water":     (0.10, [0.28, 0.52, 0.82], 0.03, False),
}
NAMES = list(TYPES)
HAZARD = ("mud", "water")


# ---------------------------------------------------------------- layout
def blob_field(rng, n_blobs, sigma, extra=()):
    xs = X0 + (np.arange(NX) + 0.5) * TILE
    ys = Y0 + (np.arange(NY) + 0.5) * TILE
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    f = np.zeros((NX, NY))
    for _ in range(n_blobs):
        cx, cy = rng.uniform(X0, X1), rng.uniform(Y0, Y1)
        f += rng.choice([-1, 1]) * np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sigma ** 2))
    f = (f - f.mean()) / (f.std() + 1e-9)
    for cx, cy, amp, s in extra:
        f += amp * np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * s ** 2))
    return f, gx, gy


def path_y(x, side):
    return side * 2.3 * np.sin(np.pi * np.clip(x, 0, 10) / 10)


def layout(seed):
    """Terrain type per tile, trees and rocks."""
    rng = np.random.default_rng(100 + seed)
    side = 1 if seed % 2 == 0 else -1
    # A pond with a mud rim always sits across the straight line A-B.
    wet, gx, gy = blob_field(rng, 10, 1.1, extra=[(rng.uniform(4.0, 6.0), rng.uniform(-0.4, 0.4), 3.2, 1.1)])
    soil, _, _ = blob_field(rng, 10, 1.0)
    grid = np.full((NX, NY), NAMES.index("grass"))
    grid[soil > 0.8] = NAMES.index("sand")
    grid[soil < -0.9] = NAMES.index("gravel")
    grid[wet > 0.7] = NAMES.index("wet grass")
    grid[wet > 1.3] = NAMES.index("mud")
    grid[wet > 2.0] = NAMES.index("water")
    on_path = np.abs(gy - path_y(gx, side)) < 0.45
    on_path &= (gx > -0.6) & (gx < 10.6)
    grid[on_path] = NAMES.index("path")
    trees, rocks = [], []
    tries = 0
    while len(trees) < 16 and tries < 2000:
        tries += 1
        x, y = rng.uniform(X0 + 0.4, X1 - 0.4), rng.uniform(Y0 + 0.4, Y1 - 0.4)
        if min(np.hypot(x - A[0], y - A[1]), np.hypot(x - B[0], y - B[1])) < 1.2:
            continue
        if abs(y - path_y(x, side)) < 0.9 or NAMES[grid[tile(x, y)]] == "water":
            continue
        if any(np.hypot(x - a, y - b) < 1.0 for a, b, _ in trees + rocks):
            continue
        trees.append((x, y, rng.uniform(0.10, 0.16)))
    while len(rocks) < 6 and tries < 4000:
        tries += 1
        x, y = rng.uniform(X0 + 0.4, X1 - 0.4), rng.uniform(Y0 + 0.4, Y1 - 0.4)
        if min(np.hypot(x - A[0], y - A[1]), np.hypot(x - B[0], y - B[1])) < 1.2:
            continue
        if abs(y - path_y(x, side)) < 0.9 or any(np.hypot(x - a, y - b) < 1.0 for a, b, _ in trees + rocks):
            continue
        rocks.append((x, y, rng.uniform(0.15, 0.25)))
    return dict(grid=grid, trees=trees, rocks=rocks, side=side)


def tile(x, y):
    return (int(np.clip((x - X0) / TILE, 0, NX - 1)), int(np.clip((y - Y0) / TILE, 0, NY - 1)))


def build_outdoor_model(lay, gravel_seed=0):
    spec = mujoco.MjSpec.from_file(str(ensure_model()))
    spec.body("trunk").add_site(name="imu", pos=[0, 0, 0])
    spec.add_sensor(name="gyro", type=mujoco.mjtSensor.mjSENS_GYRO,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    spec.add_sensor(name="acc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    rng = np.random.default_rng(gravel_seed)
    wb = spec.worldbody
    for i in range(NX):
        for j in range(NY):
            name = NAMES[lay["grid"][i, j]]
            mu, col, tc, bumps = TYPES[name]
            shade = 1.0 + 0.05 * ((i + j) % 2)
            x, y = X0 + (i + 0.5) * TILE, Y0 + (j + 0.5) * TILE
            wb.add_geom(name=f"t_{i}_{j}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[TILE / 2, TILE / 2, 0.004],
                        pos=[x, y, 0.004], friction=[mu, 0.005, 0.0001], solref=[tc, 1.0],
                        rgba=[*(np.clip(np.array(col) * shade, 0, 1)), 1])
            if bumps:
                for _ in range(10):
                    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.013, 0, 0],
                                pos=[x + rng.uniform(-0.23, 0.23), y + rng.uniform(-0.23, 0.23), 0.009],
                                friction=[mu, 0.005, 0.0001], rgba=[0.45, 0.45, 0.44, 1])
            if name == "water":     # visual water surface the feet dip into
                wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[TILE / 2, TILE / 2, 0.001],
                            pos=[x, y, 0.035], rgba=[0.35, 0.62, 0.95, 0.45], contype=0, conaffinity=0)
    for k, (x, y, r) in enumerate(lay["trees"]):
        wb.add_geom(name=f"tree{k}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, 0.6, 0],
                    pos=[x, y, 0.6], rgba=[0.36, 0.25, 0.16, 1])
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.45, 0, 0], pos=[x, y, 1.35],
                    rgba=[0.16, 0.38, 0.16, 1], contype=0, conaffinity=0)
    for k, (x, y, s) in enumerate(lay["rocks"]):
        wb.add_geom(name=f"rock{k}", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[s, s * 0.8, s * 0.6],
                    pos=[x, y, s * 0.3], rgba=[0.48, 0.47, 0.45, 1])
    for p, col in ((A, [0.95, 0.6, 0.2, 1]), (B, [0.2, 0.5, 0.95, 1])):
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.25, 0.002, 0],
                    pos=[*p, 0.011], rgba=col, contype=0, conaffinity=0)
    m = spec.compile()
    m.mat_rgba[m.material("groundplane").id] = [0.40, 0.62, 0.34, 1]
    return m


def uniform_layout(name):
    return dict(grid=np.full((NX, NY), NAMES.index(name)), trees=[], rocks=[], side=1)


# ---------------------------------------------------------------- what a camera would see
def appearance(lay, seed=0):
    """Per tile: R, G, B, roughness - noisy, as a camera + height estimate would give."""
    rng = np.random.default_rng(500 + seed)
    f = np.zeros((NX, NY, 4), np.float32)
    for i in range(NX):
        for j in range(NY):
            name = NAMES[lay["grid"][i, j]]
            col = np.array(TYPES[name][1])
            f[i, j, :3] = col + rng.normal(0, 0.05, 3)
            f[i, j, 3] = (1.0 if TYPES[name][3] else 0.0) + rng.normal(0, 0.2)
    return f


# ---------------------------------------------------------------- planning map
class OutdoorMap:
    def __init__(self, lay):
        xs = X0 + (np.arange(CX) + 0.5) * CELL
        ys = Y0 + (np.arange(CY) + 0.5) * CELL
        self.cx, self.cy = np.meshgrid(xs, ys, indexing="ij")
        self.blocked = np.zeros((CX, CY), bool)
        for x, y, r in lay["trees"]:
            self.blocked |= np.hypot(self.cx - x, self.cy - y) <= r + INFLATE
        for x, y, s in lay["rocks"]:
            self.blocked |= np.hypot(self.cx - x, self.cy - y) <= s + INFLATE
        self.sum = np.zeros((CX, CY))
        self.n = np.zeros((CX, CY))
        self.pred = None            # per-cell predicted risk (seen model), or None
        ti = np.clip(((self.cx - X0) / TILE).astype(int), 0, NX - 1)
        tj = np.clip(((self.cy - Y0) / TILE).astype(int), 0, NY - 1)
        self.cell_tile = (ti, tj)

    def add(self, x, y, risk01):
        m = np.hypot(self.cx - x, self.cy - y) <= FELT_SPREAD
        self.sum[m] += risk01
        self.n[m] += 1

    def risk(self):
        felt = np.where(self.n > 0, self.sum / np.maximum(self.n, 1), 0.0)
        if self.pred is None:
            return felt
        return np.where(self.n > 0, felt, self.pred)

    def set_tile_pred(self, p_tile):
        self.pred = p_tile[self.cell_tile]

    def cell(self, p):
        return (int(np.clip((p[0] - X0) / CELL, 0, CX - 1)), int(np.clip((p[1] - Y0) / CELL, 0, CY - 1)))

    def plan(self, start, goal):
        cost = 1.0 + MAP_WEIGHT * self.risk()
        s, g = self.cell(start), self.cell(goal)
        blocked = self.blocked.copy()
        blocked[s] = blocked[g] = False
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
                if 0 <= n[0] < CX and 0 <= n[1] < CY and not blocked[n]:
                    nd = d + st * 0.5 * (cost[c] + cost[n])
                    if nd < best.get(n, np.inf):
                        best[n], parent[n] = nd, c
                        heapq.heappush(heap, (nd + np.hypot(n[0] - g[0], n[1] - g[1]), nd, n))
        cells = [g]
        while cells[-1] in parent:
            cells.append(parent[cells[-1]])
        return np.array([[self.cx[c], self.cy[c]] for c in reversed(cells)])


# ---------------------------------------------------------------- controllers
class SeenModel:
    """Learns on site, in context: tile appearance -> 'felt slippery here?'"""

    def __init__(self, kind):
        self.kind = kind
        self.X, self.y = [], []
        self.model = None

    def add(self, feat, label):
        self.X.append(feat)
        self.y.append(int(label))

    def ready(self):
        y = np.array(self.y)
        return len(y) >= 10 and y.sum() >= 3 and (len(y) - y.sum()) >= 3

    def predict_tiles(self, feats):
        X, y = np.array(self.X), np.array(self.y)
        flat = feats.reshape(-1, feats.shape[-1])
        if self.kind == "TabPFN":
            m = CollisionRisk(n_estimators=4).fit(X, y, n=1500, pos_fraction=0.5)
            p = m.predict(flat)
        else:
            from xgboost import XGBClassifier
            m = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, verbosity=0).fit(X, y)
            p = m.predict_proba(flat)[:, 1]
        return p.reshape(feats.shape[:2])


SLIPPERY = 0.35        # a 0.2 s window counts as 'felt slippery' if a loaded foot slid > 35% of it


def make_controller(kind, lay, feats, slipbuf):
    """kind: 'obstacles only' | 'felt map' | 'felt + seen (TabPFN)' | 'felt + seen (XGBoost)'.

    slipbuf counts, since the last decision, simulation steps and steps with a loaded foot
    sliding (the robot's measured slip; on a real robot: the legs-vs-IMU slip detector)."""
    omap = OutdoorMap(lay)
    seen = SeenModel("TabPFN" if "TabPFN" in kind else "XGBoost") if "seen" in kind else None
    st = {"path": None, "goal": None, "t": -1.0, "fit_t": -9.0, "n_fit": 0, "trace": []}

    def choose(f, h, d, g):
        xy = d.qpos[:2].copy()
        st["trace"].append([round(float(xy[0]), 2), round(float(xy[1]), 2)])
        frac = slipbuf["slip"] / max(slipbuf["all"], 1)
        slipbuf["slip"] = slipbuf["all"] = 0
        if kind != "obstacles only" and d.time > 1.0:
            omap.add(*xy, frac)
            if seen is not None:
                seen.add(feats[tile(*xy)], frac > SLIPPERY)
                if seen.ready() and d.time - st["fit_t"] >= 2.0:
                    omap.set_tile_pred(seen.predict_tiles(feats))
                    st["fit_t"], st["n_fit"] = d.time, st["n_fit"] + 1
        if st["goal"] is not g or d.time - st["t"] >= 0.5:
            st["path"], st["goal"], st["t"] = omap.plan(xy, g), g, d.time
        path = st["path"]
        far = np.linalg.norm(path - xy, axis=1) >= LOOKAHEAD
        wp = path[np.argmax(far)] if far.any() else g
        return P.steer(d, wp, 1.0)

    choose.omap, choose.state, choose.seen = omap, st, seen
    return choose


# ---------------------------------------------------------------- experiment
KINDS = ["obstacles only", "felt map", "felt + seen (TabPFN)", "felt + seen (XGBoost)"]
SEEDS = range(3)


def install_hazard_lookup(lay):
    """walk() counts 'ice seconds' with P.on_ice; here that means time on mud or water."""
    haz = np.isin(lay["grid"], [NAMES.index(n) for n in HAZARD])
    P.on_ice = lambda model, x, y: bool(haz[tile(x, y)])


def run_one(kind, seed):
    import torch
    torch.set_num_threads(2)
    lay = layout(seed)
    install_hazard_lookup(lay)
    model = build_outdoor_model(lay)
    sensors = P.PatchSensors(model)
    feats = appearance(lay, seed)
    slipbuf = {"slip": 0, "all": 0}
    choose = make_controller(kind, lay, feats, slipbuf)
    goals = [B if i % 2 == 0 else A for i in range(LEGS)]
    snaps = {}

    def on_frame(data, sliding, leg):
        slipbuf["all"] += 1
        slipbuf["slip"] += bool(sliding.any())
        if leg not in snaps:
            snaps[leg] = choose.omap.risk().round(3).tolist()
        return True

    t = time.time()
    out = P.walk(model, sensors, goals, choose, max_time=60 * LEGS, on_frame=on_frame)
    res = dict(kind=kind, seed=seed, legs=out["legs"], completed=out["completed"], fell=out["fell"],
               sim_time=out["time"], wall=time.time() - t, trace=choose.state["trace"],
               risk_end=choose.omap.risk().round(3).tolist(), risk_leg_start=snaps,
               n_examples=len(choose.seen.y) if choose.seen else 0, n_fits=choose.state["n_fit"])
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / f"{kind.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus')}_{seed}.json"
     ).write_text(json.dumps(res))
    print(f"{kind} seed {seed}: legs {out['completed']}/{LEGS} fell {out['fell']} slip per leg "
          f"{[round(l['slip_s'], 1) for l in out['legs']]} ({res['wall']:.0f}s)", flush=True)


def experiment():
    jobs = [(k, s) for s in SEEDS for k in KINDS]
    procs = []
    env = dict(os.environ, OMP_NUM_THREADS="2")
    for k, s in jobs:
        procs.append(subprocess.Popen([sys.executable, "-m", "a1slip.ice.outdoor", "run", k, str(s)], env=env, cwd=ROOT))
        while sum(p.poll() is None for p in procs) >= 6:
            time.sleep(2)
    for p in procs:
        p.wait()
    summary()


def calibrate():
    """True slip rate of each terrain type: walk straight 10 m at full speed on a floor of only that type."""
    out = {}
    for name in NAMES:
        lay = uniform_layout(name)
        install_hazard_lookup(lay)
        model = build_outdoor_model(lay)
        sensors = P.PatchSensors(model)
        cnt = {"slip": 0, "all": 0}

        def on_frame(data, sliding, leg):
            cnt["all"] += 1
            cnt["slip"] += bool(sliding.any())
            return True
        r = P.walk(model, sensors, [B], lambda f, h, d, g: P.steer(d, g, 1.0), max_time=25, on_frame=on_frame)
        leg = r["legs"][0] if r["legs"] else None
        t = leg["time"] if leg else r["time"]
        frac = cnt["slip"] / max(cnt["all"], 1)
        out[name] = dict(mu=TYPES[name][0], reached=bool(leg), fell=r["fell"], time=t, slip_fraction=frac)
        print(f"{name:10s} mu {TYPES[name][0]:.2f}  slipping {100 * frac:4.0f}% of the time  "
              + (f"reached B in {t:5.1f} s" if leg else f"did not reach B in 25 s (fell={r['fell']})"), flush=True)
    (RESULTS / "outdoor_types.json").write_text(json.dumps(out, indent=1))


def load_runs():
    return [json.loads(p.read_text()) for p in sorted(OUTDIR.glob("*.json"))]


def summary():
    runs = load_runs()
    print(f"\n{'strategy':24s} {'legs done':>9s} {'falls':>5s} {'slip/leg, trip 1':>17s} {'slip/leg, trips 2-4':>20s} "
          f"{'hazard s/leg':>12s} {'time/leg':>9s}")
    for k in KINDS:
        rs = [r for r in runs if r["kind"] == k]
        if not rs:
            continue
        l1 = [l for r in rs for l in r["legs"] if l["leg"] == 0]
        l2 = [l for r in rs for l in r["legs"] if l["leg"] >= 1]
        allg = l1 + l2
        print(f"{k:24s} {sum(r['completed'] for r in rs):>5d}/{LEGS * len(rs):<3d} {sum(r['fell'] for r in rs):>5d} "
              f"{np.mean([l['slip_s'] for l in l1]) if l1 else float('nan'):>17.1f} "
              f"{np.mean([l['slip_s'] for l in l2]) if l2 else float('nan'):>20.1f} "
              f"{np.mean([l['ice_s'] for l in allg]) if allg else float('nan'):>12.1f} "
              f"{np.mean([l['time'] for l in allg]) if allg else float('nan'):>9.1f}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run_one(sys.argv[2], int(sys.argv[3]))
    else:
        {"calibrate": calibrate, "experiment": experiment, "summary": summary,
         "figure": lambda: __import__("a1_outdoor_figure").main()}[cmd]()
