"""Builds the MuJoCo model (Franka Panda + wrist camera + cubes + bin + obstacles)."""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from .config import COLORS, CUBE_HALF, SceneConfig

ROOT = Path(__file__).resolve().parents[2]
PANDA_XML = ROOT / "robots" / "franka_emika_panda" / "panda.xml"
CAM_NAME = "wrist_camera"
ROBOT_BODIES = ["link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "left_finger", "right_finger"]


def yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _robot_spec() -> mujoco.MjSpec:
    if not PANDA_XML.exists():
        raise FileNotFoundError(f"{PANDA_XML} not found. Run scripts/fetch_assets.sh first.")
    return mujoco.MjSpec.from_file(str(PANDA_XML))


def _finger_friction(spec: mujoco.MjSpec, mu: float) -> None:
    for name in ("left_finger", "right_finger"):
        for g in spec.body(name).geoms:
            if g.contype or g.conaffinity or g.group == 3:  # collision geoms only
                g.friction = [mu, 0.005, 0.0001]
                g.condim = 4


def add_static_scene(spec: mujoco.MjSpec, cfg_bin, obstacles, margin: float = 0.0) -> None:
    """Floor/table, bin and obstacles. Used by both the full simulation and the planner's collision model
    (`margin` > 0 inflates bin walls/floor and obstacles = safety clearance for planning)."""
    wb = spec.worldbody
    m = margin
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.5, 1.5, 0.05],
                rgba=[0.55, 0.55, 0.55, 1], friction=[1.0, 0.005, 0.0001])
    if cfg_bin.xy is not None and cfg_bin.yaw is not None:   # a hidden bin is not part of the planner's model
        b = wb.add_body(name="bin", pos=[cfg_bin.xy[0], cfg_bin.xy[1], 0.0], quat=yaw_quat(cfg_bin.yaw))
        ix, iy, t, wh, ft = cfg_bin.inner[0], cfg_bin.inner[1], cfg_bin.wall_thickness, cfg_bin.wall_height, cfg_bin.floor_thickness
        col = [0.80, 0.78, 0.70, 1]
        kw = dict(type=mujoco.mjtGeom.mjGEOM_BOX, rgba=col, friction=[1.0, 0.005, 0.0001])
        b.add_geom(name="bin_floor", size=[ix / 2 + t + m, iy / 2 + t + m, ft / 2 + m], pos=[0, 0, ft / 2], **kw)
        b.add_geom(name="bin_wall_px", size=[t / 2 + m, iy / 2 + t + m, wh / 2 + m], pos=[ix / 2 + t / 2, 0, wh / 2], **kw)
        b.add_geom(name="bin_wall_nx", size=[t / 2 + m, iy / 2 + t + m, wh / 2 + m], pos=[-ix / 2 - t / 2, 0, wh / 2], **kw)
        b.add_geom(name="bin_wall_py", size=[ix / 2 + m, t / 2 + m, wh / 2 + m], pos=[0, iy / 2 + t / 2, wh / 2], **kw)
        b.add_geom(name="bin_wall_ny", size=[ix / 2 + m, t / 2 + m, wh / 2 + m], pos=[0, -iy / 2 - t / 2, wh / 2], **kw)
    for i, o in enumerate(obstacles):
        ob = wb.add_body(name=f"obstacle{i}", pos=[o.xy[0], o.xy[1], 0.0], quat=yaw_quat(o.yaw))
        ob.add_geom(name=f"obstacle_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[h + m for h in o.half], pos=[0, 0, o.half[2]],
                    rgba=[0.30, 0.30, 0.32, 1], friction=[1.0, 0.005, 0.0001])


def build_robot_only(gravcomp: bool = True) -> mujoco.MjModel:
    """Robot without any scene: the agent-visible kinematic model."""
    spec = _robot_spec()
    return spec.compile()


def build_planning_model(task_bin, obstacles, margin: float = 0.0) -> mujoco.MjModel:
    """Robot + table + bin + obstacles (inflated by `margin`), NO cubes. Used by CollisionChecker."""
    spec = _robot_spec()
    add_static_scene(spec, task_bin, obstacles, margin)
    return spec.compile()


def build_scene(cfg: SceneConfig) -> mujoco.MjModel:
    spec = _robot_spec()
    ph = cfg.physics
    spec.option.timestep = ph.timestep
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.noslip_iterations = 3
    spec.visual.map.znear = 0.02
    spec.visual.map.zfar = 3.0
    spec.visual.headlight.ambient = [0.20, 0.20, 0.20]
    spec.visual.headlight.diffuse = [0.20, 0.20, 0.20]
    spec.visual.headlight.specular = [0, 0, 0]

    if ph.gravcomp:
        for name in ROBOT_BODIES:
            spec.body(name).gravcomp = 1.0
    _finger_friction(spec, ph.friction)

    spec.worldbody.add_light(name="sun", pos=[0.4, 0, 2.0], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             diffuse=[0.3, 0.3, 0.3], specular=[0, 0, 0])
    for light in spec.lights:  # no cast shadows
        light.castshadow = 0
        light.diffuse = [0.25, 0.25, 0.25]  # keep faces below saturation so hues stay faithful
    add_static_scene(spec, cfg.bin, cfg.obstacles)

    for i, c in enumerate(cfg.cubes):
        body = spec.worldbody.add_body(name=f"cube{i}", pos=[c.xy[0], c.xy[1], CUBE_HALF], quat=yaw_quat(c.yaw))
        body.add_freejoint(name=f"cube{i}_free")
        r, g, b = COLORS[c.color]
        body.add_geom(name=f"cube{i}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3,
                      rgba=[r, g, b, 1], density=ph.cube_density, friction=[ph.friction, 0.005, 0.0001], condim=4)
    return spec.compile()
