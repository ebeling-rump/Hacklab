"""Videos for the ice-patch experiments (offscreen rendering, exact replays).

  ice_governor_comparison.mp4  layout 1, A -> B: fixed full speed | crawl 0.3 | TabPFN governor
  ice_patrol_tabpfn.mp4        layout 2, 6-leg patrol, top-down, 2x speed: TabPFN's slip map
                               drawn on the ground (orange = slippery), route bending around ice

    python a1_ice_videos.py            # both
    python a1_ice_videos.py patrol     # patrol video only
"""

import time
import warnings
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

import a1_ice_patches as P
from a1_record_videos import H, W, add_geom, composite, label
from a1_slip_dataset import feature_names

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "videos"
STEPS_PER_FRAME = 17


def record_walk(name, title, model, sensors, goals, choose, info, max_time, cam_setup, every=1,
                draw_map=None):
    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    cam_setup(cam)
    fps = 1.0 / (STEPS_PER_FRAME * model.opt.timestep)
    writer = imageio.get_writer(OUT / f"{name}.mp4", fps=fps, codec="libx264", quality=8,
                                macro_block_size=16)
    step, slip = [0], [0.0]
    last = [None]

    def on_frame(data, sliding, leg):
        step[0] += 1
        slip[0] += model.opt.timestep * sliding.any()
        if step[0] % (STEPS_PER_FRAME * every):
            return True
        renderer.update_scene(data, cam)
        scn = renderer.scene
        if draw_map is not None:
            draw_map(scn)
        for i, g in enumerate(sensors.feet):
            if sliding[i]:
                add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.06 if every > 1 else 0.045, 0, 0],
                         data.geom_xpos[g], [1, 0.1, 0.1, 0.95])
        lines = [title, info(data, leg), f"t = {data.time:5.1f} s    slipping so far {slip[0]:4.1f} s"
                 + ("    ON ICE" if P.on_ice(model, *data.qpos[:2]) else "")]
        frame = label(renderer.render(), lines)
        last[0] = (frame, lines)
        writer.append_data(frame)
        return True

    t = time.time()
    out = P.walk(model, sensors, goals, choose, max_time=max_time, on_frame=on_frame)
    done = out["completed"] == len(goals)
    banner = (f"Reached B in {out['legs'][0]['time']:.0f} s" if len(goals) == 1 and done else
              f"{out['completed']} legs, {slip[0]:.1f} s slipping")
    end = label(last[0][0], last[0][1], banner, banner_good=True)
    for _ in range(int(1.5 * fps)):
        writer.append_data(end)
    writer.close()
    renderer.close()
    print(f"{name}: {banner} ({time.time() - t:.0f}s)", flush=True)
    return fps


def main():
    import sys
    OUT.mkdir(exist_ok=True)
    tab, det, gap, ref = P.models()
    fi = feature_names().index("cmd_forward")
    only_patrol = "patrol" in sys.argv[1:]

    def track(cam):
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = 1
        cam.distance, cam.elevation, cam.azimuth = 3.0, -30, 120

    # ---- governor comparison (layout 1) ----------------------------------------------
    seed = 1
    model = P.build_patch_model(seed)
    sensors = P.PatchSensors(model)
    speed = {"v": 0.0}

    def fixed(s):
        def choose(f, h, d, g):
            speed["v"] = s
            return P.steer(d, g, s)
        return choose

    def gov(f, h, d, g, limit=0.65):
        cands = np.repeat(np.r_[f, h][None], len(P.SPEEDS), 0)
        cands[:, fi] = P.SPEEDS
        p = tab.predict(cands)
        ok = [s for s, pi in zip(P.SPEEDS, p) if pi < limit]
        speed["v"] = max(ok) if ok else P.SPEEDS[int(np.argmin(p))]
        return P.steer(d, g, speed["v"])

    info = lambda d, leg: f"speed command {speed['v']:.2f}"
    fps = None
    if not only_patrol:
        for name, title, choose in (("ice_gov_full", "Fixed full speed", fixed(1.0)),
                                    ("ice_gov_crawl", "Fixed crawl (0.3)", fixed(0.3)),
                                    ("ice_gov_tabpfn", "TabPFN speed governor", gov)):
            fps = record_walk(name, title, model, sensors, [P.B], choose, info, 60, track)
        composite("ice_governor_comparison", ["ice_gov_full", "ice_gov_crawl", "ice_gov_tabpfn"], fps)

    # ---- patrol with TabPFN slip map (layout 2), top-down at 2x ------------------------
    seed = 2
    model = P.build_patch_model(seed)
    sensors = P.PatchSensors(model)
    choose, smap = P.patrol_controller("TabPFN", tab, det, gap, ref)
    goals = [P.B if i % 2 == 0 else P.A for i in range(6)]

    def top(cam):
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [4.0, 0.0, 0.0]
        cam.distance, cam.elevation, cam.azimuth = 10.5, -72, 90

    def draw_map(scn):
        r = smap.risk()
        for i, j in zip(*np.nonzero(r > 0.55)):          # only clearly slippery cells
            add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [P.CELL / 2, P.CELL / 2, 0.002],
                     [smap.cx[i, j], smap.cy[i, j], 0.012],
                     [0.95, 0.45, 0.1, 0.2 + 0.6 * (r[i, j] - 0.55) / 0.45])
        path = choose.state["path"]
        if path is not None:
            for x, y in path[::2]:
                add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.05, 0, 0], [x, y, 0.03],
                         [0.2, 0.45, 0.95, 0.9])

    record_walk("ice_patrol_tabpfn", "Patrol A <-> B with TabPFN slip map (2x speed)", model, sensors,
                goals, choose, lambda d, leg: f"leg {min(leg + 1, 6)} of 6    orange = mapped as slippery,"
                " blue = planned route", 240, top, every=2, draw_map=draw_map)


if __name__ == "__main__":
    main()
