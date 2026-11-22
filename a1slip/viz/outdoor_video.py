"""Video of the outdoor course: TabPFN 'felt + seen' patrol, layout 1 (offscreen MuJoCo rendering).

Left: 3D chase view of the terrain. Right: top-down view with the robot's slip map drawn on
the ground (dark = safe, bright = slippery; tiles it has not walked on are TabPFN's
predictions from appearance) and the planned route (blue dots). 2x speed.

    python -m a1slip.viz.outdoor_video              # TabPFN felt + seen -> videos/outdoor_tabpfn.mp4
    python -m a1slip.viz.outdoor_video obstacles    # obstacles only     -> videos/outdoor_obstacles_only.mp4
"""

import sys
import time
import warnings

import imageio.v2 as imageio
import matplotlib
import mujoco
import numpy as np

from a1slip.paths import ROOT
from a1slip.ice import patches as P
from a1slip.ice import outdoor as O
from a1slip.viz.record_videos import H, W, add_geom, label

warnings.filterwarnings("ignore")
OUT = ROOT / "videos"
STEPS_PER_FRAME = 17
EVERY = 2                    # keep every 2nd frame -> 2x speed
CMAP = matplotlib.colormaps["magma"]


def main():
    obstacles = "obstacles" in sys.argv[1:]
    kind = "obstacles only" if obstacles else "felt + seen (TabPFN)"
    name = "outdoor_obstacles_only" if obstacles else "outdoor_tabpfn"
    seed = 0
    lay = O.layout(seed)
    O.install_hazard_lookup(lay)
    model = O.build_outdoor_model(lay)
    model.vis.global_.offwidth, model.vis.global_.offheight = max(W, 640), max(H, 480)
    sensors = P.PatchSensors(model)
    feats = O.appearance(lay, seed)
    slipbuf = {"slip": 0, "all": 0}
    choose = O.make_controller(kind, lay, feats, slipbuf)
    goals = [O.B if i % 2 == 0 else O.A for i in range(O.LEGS)]

    r3d, rtop = mujoco.Renderer(model, H, W), mujoco.Renderer(model, H, W)
    chase, top = mujoco.MjvCamera(), mujoco.MjvCamera()
    chase.type = top.type = mujoco.mjtCamera.mjCAMERA_FREE
    chase.distance, chase.elevation = 3.2, -28
    top.lookat[:] = [5.0, 0.0, 0.0]
    top.distance, top.elevation, top.azimuth = 13.5, -89.9, 90
    fps = 1.0 / (STEPS_PER_FRAME * model.opt.timestep)
    OUT.mkdir(exist_ok=True)
    writer = imageio.get_writer(OUT / f"{name}.mp4", fps=fps, codec="libx264", quality=8, macro_block_size=16)
    st = {"step": 0, "slip": 0.0, "last": None}
    tiles_c = [(O.X0 + (i + 0.5) * O.TILE, O.Y0 + (j + 0.5) * O.TILE) for i in range(O.NX) for j in range(O.NY)]

    def on_frame(data, sliding, leg):
        slipbuf["all"] += 1
        slipbuf["slip"] += bool(sliding.any())
        st["step"] += 1
        st["slip"] += model.opt.timestep * sliding.any()
        if st["step"] % (STEPS_PER_FRAME * EVERY):
            return True
        xy = data.qpos[:2]
        chase.lookat[:] = [xy[0], xy[1], 0.25]
        yawd = np.degrees(P.yaw(data)) if hasattr(P, "yaw") else 0.0
        chase.azimuth = yawd + 180 - 25
        r3d.update_scene(data, chase)
        for i, g in enumerate(sensors.feet):
            if sliding[i]:
                add_geom(r3d.scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.045, 0, 0], data.geom_xpos[g], [1, 0.1, 0.1, 0.95])
        rtop.update_scene(data, top)
        scn = rtop.scene
        if kind != "obstacles only":
            risk = choose.omap.risk()
            known = (choose.omap.n > 0) | (choose.omap.pred is not None)
            for (x, y) in tiles_c:
                c = choose.omap.cell((x, y))
                if not known[c]:
                    continue
                rgba = list(CMAP(float(risk[c]))[:3]) + [0.75]
                add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [O.TILE / 2, O.TILE / 2, 0.002], [x, y, 0.06], rgba)
        path = choose.state["path"]
        if path is not None:
            for p in path[::2]:
                add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.06, 0, 0], [p[0], p[1], 0.1], [0.2, 0.6, 1.0, 1])
        add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.18, 0, 0], [xy[0], xy[1], 0.5], [1.0, 0.3, 0.2, 1])
        on_haz = P.on_ice(model, *xy)
        here = O.NAMES[lay["grid"][O.tile(*xy)]]
        title = "Outdoor course, TabPFN felt + seen map (2x)" if kind != "obstacles only" else \
            "Outdoor course, obstacles only (2x)"
        lines_l = [title, f"trip {min(leg, O.LEGS - 1) + 1} of {O.LEGS}    on: {here}",
                   f"t = {data.time:5.1f} s    slipping so far {st['slip']:4.1f} s" + ("    IN MUD/WATER" if on_haz else "")]
        n_ex = len(choose.seen.y) if choose.seen else 0
        lines_r = ["Slip map: bright = slippery, dark = safe",
                   "walked tiles = felt; others = TabPFN prediction from appearance" if kind != "obstacles only"
                   else "no slip map: A* around trees and rocks only",
                   f"TabPFN examples (felt on this site): {n_ex}" if kind != "obstacles only" else ""]
        frame = np.concatenate([label(r3d.render(), lines_l), np.full((H, 8, 3), 20, np.uint8),
                                label(rtop.render(), lines_r)], axis=1)
        st["last"] = frame
        writer.append_data(frame)
        return True

    t = time.time()
    out = P.walk(model, sensors, goals, choose, max_time=60 * O.LEGS, on_frame=on_frame)
    for _ in range(int(1.5 * fps / EVERY)):
        writer.append_data(st["last"])
    writer.close()
    print(f"{name}: {out['completed']}/{O.LEGS} trips, slip per trip "
          f"{[round(l['slip_s'], 1) for l in out['legs']]} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
