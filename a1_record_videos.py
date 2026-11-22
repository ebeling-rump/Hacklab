"""Record MP4 videos of the A1's behaviours (offscreen MuJoCo rendering, no window).

Each clip replays one logged episode exactly (same floor, commands and pushes;
the simulation is deterministic) and draws the same markers as the viewer:
  ball above the body  blue = braced, grey = normal gait
  red spheres          sliding feet
  white arrow          downhill
plus a text label: strategy, time, state, TabPFN fall risk or tilt, and the result.

Clips (videos/<name>.mp4) and side-by-side comparisons:
  blind_falls, tabpfn_braces, balance_reflex, always_brace   steep episode 9
  tilt_safe_floor, tabpfn_safe_floor                         safe episode 30
  episode9_comparison.mp4    blind | TabPFN | balance reflex
  safe_floor_comparison.mp4  tilt rule | TabPFN

    python a1_record_videos.py               # all clips (~5 min, TabPFN loads once)
    python a1_record_videos.py blind_falls   # one clip
"""

import sys
import time
import warnings
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from a1_balance import TUNED, Balancer
from a1_danger_mode import THRESHOLD, TILT_LIMIT, episode_setup, fit_fall_model, make_controllers
from a1_slip_dataset import NORMAL, A1Sensors, build_model, run_episode
from a1_terrain import TerrainSensors, build_terrain_model, terrain_episode

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "videos"
W, H = 640, 368            # multiples of 16 for H.264
STEPS_PER_FRAME = 17       # 17 x 2 ms -> 29.4 fps of simulated time
HOLD = 1.5                 # s the last frame stays on screen
CLIPS = {
    "blind_falls": ("dangerous", 9, "never brace", False, "Blind trot"),
    "tabpfn_braces": ("dangerous", 9, "TabPFN", False, "Blind trot + TabPFN danger mode"),
    "balance_reflex": ("dangerous", 9, "never brace", True, "Balance reflex (no TabPFN)"),
    "always_brace": ("dangerous", 9, "always brace", False, "Always brace"),
    "tilt_safe_floor": ("safe", 30, "tilt rule", False, "Tilt rule (no ML)"),
    "tabpfn_safe_floor": ("safe", 30, "TabPFN", False, "TabPFN danger mode"),
    # Real hills (a1_terrain.py, terrain 1 = steep), frozen models: an unseen condition.
    "hills_ice_blind": ("hills1", 12, "never brace", False, "Blind trot"),
    "hills_ice_tabpfn": ("hills1", 12, "TabPFN", False, "Blind trot + TabPFN danger mode"),
    "hills_ice_balance": ("hills1", 12, "never brace", True, "Balance reflex (no TabPFN)"),
    "hills_blind": ("hills1", 24, "never brace", False, "Blind trot"),
    "hills_tabpfn": ("hills1", 24, "TabPFN", False, "Blind trot + TabPFN danger mode"),
    "hills_balance": ("hills1", 24, "never brace", True, "Balance reflex (no TabPFN)"),
}
COMPOSITES = {
    "episode9_comparison": ["blind_falls", "tabpfn_braces", "balance_reflex"],
    "safe_floor_comparison": ["tilt_safe_floor", "tabpfn_safe_floor"],
    "hills_ice_comparison": ["hills_ice_blind", "hills_ice_tabpfn", "hills_ice_balance"],
    "hills_comparison": ["hills_blind", "hills_tabpfn", "hills_balance"],
}


def font(size):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


F_BIG, F_SMALL, F_BANNER = font(22), font(17), font(40)


def label(frame, lines, banner=None, banner_good=True):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, W, 34 + 22 * (len(lines) - 1)], fill=(20, 33, 61, 190))
    d.text((12, 6), lines[0], font=F_BIG, fill=(246, 245, 240))
    for i, line in enumerate(lines[1:]):
        d.text((12, 36 + 22 * i), line, font=F_SMALL, fill=(195, 210, 232))
    if banner:
        col = (31, 140, 90, 220) if banner_good else (200, 70, 40, 220)
        d.rectangle([0, H - 64, W, H], fill=col)
        d.text((W // 2, H - 32), banner, font=F_BANNER, fill=(255, 255, 255), anchor="mm")
    return np.asarray(img)


def add_geom(scn, kind, size, pos, rgba):
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, kind, np.asarray(size, float), np.asarray(pos, float),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    scn.ngeom += 1
    return g


def record(name, risk_keep, sim):
    kind, e, ctrl_name, balanced, title = CLIPS[name]
    risk, keep = risk_keep
    start = None
    if kind.startswith("hills"):            # real 3D terrain (a1_terrain.py)
        seed = int(kind[-1])
        model = build_terrain_model(seed)
        data = mujoco.MjData(model)
        sensors = TerrainSensors(model)
        mu, start, rng = terrain_episode(e, base_seed=7000 + 1000 * seed)
        fl = (mu, 0.0, 0.0)
    else:
        model, data, sensors = sim
        fl, rng = episode_setup(kind, e)
    policy = make_controllers(risk, keep)[ctrl_name]()
    balancer = Balancer(sensors, TUNED) if balanced else None
    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = sensors.trunk
    # On hills the camera looks down more steeply so crests never hide the robot.
    cam.distance, cam.elevation, cam.azimuth = (3.2, -38, 120) if start else (2.3, -18, 120)
    fps = 1.0 / (STEPS_PER_FRAME * model.opt.timestep)
    path = OUT / f"{name}.mp4"
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=16)
    step = [0]
    last = [None]
    floor_txt = (f"real hills ({'steep' if kind == 'hills1' else 'moderate'}), "
                 f"{'icy, ' if fl[0] < 0.2 else ''}friction {fl[0]:.2f}" if start else
                 f"{'icy ' if fl[0] < 0.2 else ''}{fl[1]:.1f} deg slope, friction {fl[0]:.2f}")

    def on_step(d, reaction, sliding):
        step[0] += 1
        if step[0] % STEPS_PER_FRAME:
            return True
        renderer.update_scene(d, cam)
        scn = renderer.scene
        trunk = d.xpos[sensors.trunk].copy()
        braced = reaction != NORMAL
        add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.06, 0, 0], trunk + [0, 0, 0.3],
                 [0.2, 0.45, 0.9, 0.95] if braced else [0.6, 0.6, 0.6, 0.9])
        for i, gid in enumerate(sensors.feet):
            if sliding[i]:
                add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.045, 0, 0], d.geom_xpos[gid],
                         [1, 0.1, 0.1, 0.95])
        gdir = model.opt.gravity[:2]
        if np.linalg.norm(gdir) > 1e-6:
            g = add_geom(scn, mujoco.mjtGeom.mjGEOM_ARROW, [0, 0, 0], [0, 0, 0], [1, 1, 1, 0.85])
            if g is not None:
                start = trunk + [0, 0, 0.45]
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.015, start,
                                     start + np.r_[0.4 * gdir / np.linalg.norm(gdir), 0])
        info = f"t = {d.time:4.1f} s    {'BRACED' if braced else 'normal gait'}"
        if ctrl_name == "TabPFN":
            info += f"    fall risk {policy.value:.0%} (brace > {THRESHOLD:.0%})"
        elif ctrl_name == "tilt rule":
            info += f"    tilt {policy.value:.1f} deg (brace > {TILT_LIMIT:.0f})"
        frame = label(renderer.render(), [title, floor_txt, info])
        last[0] = (frame, [title, floor_txt, info])
        writer.append_data(frame)
        return True

    t = time.time()
    ep = run_episode(model, data, sensors, rng, *fl, policy=policy, on_step=on_step,
                     balancer=balancer, start=start)
    fell = ep["fall_t"] is not None
    dist = float(np.linalg.norm(data.qpos[:2] - (start.xy if start else 0)))
    renderer.update_scene(data, cam)
    end_lines = last[0][1] if last[0] else [title, floor_txt, ""]
    banner = f"FELL after {dist:.2f} m" if fell else f"Stayed up, {dist:.2f} m"
    end = label(renderer.render(), end_lines, banner, banner_good=not fell)
    for _ in range(int(HOLD * fps)):
        writer.append_data(end)
    writer.close()
    renderer.close()
    print(f"{name}: {banner} -> {path} ({time.time() - t:.0f}s)", flush=True)
    return fps


def composite(name, parts, fps):
    """Side by side, streamed frame by frame; a clip that ends early holds its last frame."""
    readers = [iter(imageio.get_reader(OUT / f"{p}.mp4")) for p in parts]
    last = [None] * len(parts)
    writer = imageio.get_writer(OUT / f"{name}.mp4", fps=fps, codec="libx264", quality=8,
                                macro_block_size=16)
    gap = np.full((H, 8, 3), 20, np.uint8)
    while True:
        alive = False
        for i, r in enumerate(readers):
            try:
                last[i] = next(r)
                alive = True
            except StopIteration:
                pass
        if not alive:
            break
        row = []
        for f in last:
            row += [f, gap]
        writer.append_data(np.hstack(row[:-1]))
    writer.close()
    print(f"{name}: {' | '.join(parts)} -> {OUT / (name + '.mp4')}", flush=True)


def main():
    names = sys.argv[1:] or list(CLIPS)
    OUT.mkdir(exist_ok=True)
    need_tabpfn = any(CLIPS[n][2] == "TabPFN" for n in names)
    risk_keep = fit_fall_model() if need_tabpfn else (None, None)
    model = build_model()
    sim = (model, mujoco.MjData(model), A1Sensors(model))
    fps = None
    for n in names:
        fps = record(n, risk_keep, sim)
    for cname, parts in COMPOSITES.items():
        if all((OUT / f"{p}.mp4").exists() for p in parts):
            composite(cname, parts, fps or 1.0 / (STEPS_PER_FRAME * 0.002))


if __name__ == "__main__":
    main()
