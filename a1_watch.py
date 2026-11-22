"""Watch the headline experiments live in the MuJoCo viewer (same code and layouts as the results).

    python a1_watch.py governor                       # ice patches, TabPFN speed governor, layout 0
    python a1_watch.py governor --strategy full       # same course at fixed full speed (compare)
    python a1_watch.py governor --strategy crawl      # fixed crawl 0.3
    python a1_watch.py patrol                         # ice patches, TabPFN slip map, 6 trips A <-> B
    python a1_watch.py outdoor                        # outdoor course, TabPFN felt + seen map, 4 trips
    python a1_watch.py outdoor --strategy obstacles   # same course without a slip map
    options: --layout N (0-4 ice, 0-2 outdoor), --speed 2 (faster than real time)

In the viewer: red spheres = sliding feet; patrol/outdoor also show the slip map on the ground
(yellow -> red = slippery, outdoor: dark = safe, bright = slippery) and the planned route (blue).
TabPFN runs on CPU, so the simulation can run slower than real time while it thinks.
"""

import argparse
import time
import warnings

import matplotlib
import mujoco
import mujoco.viewer
import numpy as np

import a1_ice_patches as P
from a1_record_videos import add_geom
from a1_slip_dataset import feature_names

warnings.filterwarnings("ignore")
CMAP = matplotlib.colormaps["magma"]


def governor_choose(tab, strategy):
    fi = feature_names().index("cmd_forward")
    if strategy == "full":
        return lambda f, h, d, g: P.steer(d, g, 1.0)
    if strategy == "crawl":
        return lambda f, h, d, g: P.steer(d, g, 0.3)

    def gov(f, h, d, g):
        cands = np.repeat(np.r_[f, h][None], len(P.SPEEDS), 0)
        cands[:, fi] = P.SPEEDS
        p = tab.predict(cands)
        ok = [s for s, pi in zip(P.SPEEDS, p) if pi < 0.65]
        speed = max(ok) if ok else P.SPEEDS[int(np.argmin(p))]
        gov.speed = speed
        return P.steer(d, g, speed)
    gov.speed = 1.0
    return gov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", choices=["governor", "patrol", "outdoor"])
    ap.add_argument("--strategy", default="tabpfn",
                    help="governor: tabpfn | full | crawl;  outdoor: tabpfn | xgboost | felt | obstacles")
    ap.add_argument("--layout", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed vs real time")
    ap.add_argument("--headless", action="store_true", help="no window (testing)")
    ap.add_argument("--max-time", type=float, default=None)
    a = ap.parse_args()

    slipbuf, draw = None, None
    if a.scene in ("governor", "patrol"):
        model = P.build_patch_model(a.layout)
        sensors = P.PatchSensors(model)
        tab, det, gap, ref = P.models() if (a.scene == "patrol" or a.strategy == "tabpfn") else (None,) * 4
        if a.scene == "governor":
            choose, goals, max_time = governor_choose(tab, a.strategy), [P.B], 60
        else:
            choose, smap = P.patrol_controller("TabPFN", tab, det, gap, ref)
            goals, max_time = [P.B if i % 2 == 0 else P.A for i in range(6)], 240

            def draw(scn, data):
                risk = smap.risk()
                idx = np.argwhere((smap.n > 0) & (risk > 0.15))
                for i, j in idx[:700]:
                    c = [1.0, 0.85 - 0.6 * risk[i, j], 0.1, 0.7]
                    add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [P.CELL / 2, P.CELL / 2, 0.002],
                             [smap.cx[i, j], smap.cy[i, j], 0.02], c)
                path = choose.state["path"]
                if path is not None:
                    for p in path[::2]:
                        add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.05, 0, 0], [p[0], p[1], 0.08], [0.2, 0.6, 1, 1])
        center, dist = [4.0, 0.0, 0.0], 9.0
    else:
        import a1_outdoor as O
        kind = {"tabpfn": "felt + seen (TabPFN)", "xgboost": "felt + seen (XGBoost)", "felt": "felt map",
                "obstacles": "obstacles only"}[a.strategy]
        lay = O.layout(a.layout)
        O.install_hazard_lookup(lay)
        model = O.build_outdoor_model(lay)
        sensors = P.PatchSensors(model)
        slipbuf = {"slip": 0, "all": 0}
        choose = O.make_controller(kind, lay, O.appearance(lay, a.layout), slipbuf)
        goals, max_time = [O.B if i % 2 == 0 else O.A for i in range(O.LEGS)], 60 * O.LEGS
        tiles = [(O.X0 + (i + 0.5) * O.TILE, O.Y0 + (j + 0.5) * O.TILE) for i in range(O.NX) for j in range(O.NY)]

        def draw(scn, data):
            if kind != "obstacles only":
                om = choose.omap
                risk = om.risk()
                for (x, y) in tiles:
                    c = om.cell((x, y))
                    if om.n[c] > 0 or om.pred is not None:
                        add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [O.TILE / 2, O.TILE / 2, 0.002], [x, y, 0.05],
                                 list(CMAP(float(risk[c]))[:3]) + [0.55])
            path = choose.state["path"]
            if path is not None:
                for p in path[::2]:
                    add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.05, 0, 0], [p[0], p[1], 0.1], [0.2, 0.6, 1, 1])
        center, dist = [5.0, 0.0, 0.0], 11.0

    st = {"v": None, "wall0": None, "step": 0}

    def on_frame(data, sliding, leg):
        if slipbuf is not None:
            slipbuf["all"] += 1
            slipbuf["slip"] += bool(sliding.any())
        st["step"] += 1
        if a.headless or st["step"] % 16:
            return True
        if st["v"] is None:
            st["v"] = mujoco.viewer.launch_passive(model, data)
            st["v"].cam.lookat[:] = center
            st["v"].cam.distance, st["v"].cam.elevation, st["v"].cam.azimuth = dist, -55, 90
            st["wall0"], st["sim0"] = time.time(), data.time
        v = st["v"]
        if not v.is_running():
            return False
        with v.lock():
            scn = v.user_scn
            scn.ngeom = 0
            if draw is not None:
                draw(scn, data)
            for i, g in enumerate(sensors.feet):
                if sliding[i]:
                    add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.05, 0, 0], data.geom_xpos[g], [1, 0.1, 0.1, 1])
        v.sync()
        ahead = (data.time - st["sim0"]) / a.speed - (time.time() - st["wall0"])
        if ahead > 0:
            time.sleep(ahead)
        return True

    out = P.walk(model, sensors, goals, choose, max_time=a.max_time or max_time, on_frame=on_frame)
    print(f"{a.scene} ({a.strategy}, layout {a.layout}): {out['completed']}/{len(goals)} trips, "
          f"slipping per trip {[round(l['slip_s'], 1) for l in out['legs']]}, fell {out['fell']}", flush=True)
    if st["v"] is not None and st["v"].is_running():
        print("Done - close the viewer window to exit.", flush=True)
        while st["v"].is_running():
            time.sleep(0.1)


if __name__ == "__main__":
    main()
