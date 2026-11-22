"""MuJoCo scene: Unitree A1 + a heavy crate (hidden mass / friction / CoM) + optional walls."""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .robot import ensure_model, yaw_of


@dataclass
class Crate:
    boxes: list                 # (cx, cy, hx, hy) in body frame (centroid origin)
    height: float = 0.22
    mass: float = 4.0
    mu: float = 0.35
    com: tuple = (0.0, 0.0)
    rgba: tuple = (0.72, 0.5, 0.28, 1)


@dataclass
class Scene:
    crate: Crate
    walls: list = field(default_factory=list)   # (cx, cy, hx, hy)
    wall_h: float = 0.45
    goal: tuple | None = None                   # (x, y) marker
    robot_start: tuple = (0.0, 0.0, 0.0)
    crate_start: tuple = (1.0, 0.0, 0.0)


def build(scene: Scene):
    spec = mujoco.MjSpec.from_file(str(ensure_model()))
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.visual.quality.shadowsize = 4096
    wb = spec.worldbody
    floor = spec.geom("floor") if hasattr(spec, "geom") else None
    c = scene.crate
    body = wb.add_body(name="crate", pos=[*scene.crate_start[:2], 0.0])
    body.add_freejoint(name="crate_free")
    for i, (cx, cy, hx, hy) in enumerate(c.boxes):
        g = body.add_geom(name=f"crate{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, c.height / 2],
                          pos=[cx, cy, c.height / 2], rgba=list(c.rgba))
        g.friction = [c.mu, 0.005, 0.0001]
        g.priority = 3     # crate friction governs crate-floor contacts
        g.condim = 3
    # visual slats so rotation is visible
    for i, (cx, cy, hx, hy) in enumerate(c.boxes):
        s = body.add_geom(name=f"slat{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx * 0.98, 0.012, 0.004],
                          pos=[cx, cy, c.height + 0.002], rgba=[0.45, 0.3, 0.15, 1])
        s.contype = s.conaffinity = 0
    m = body.add_geom(name="crate_com", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0],
                      pos=[c.com[0], c.com[1], c.height + 0.02], rgba=[0.9, 0.1, 0.1, 1])
    m.contype = m.conaffinity = 0
    m.group = 3
    # inertia of a uniform box about the CoM (approx.)
    xs = [b[0] for b in c.boxes]
    hxs = max(abs(b[0]) + b[2] for b in c.boxes)
    hys = max(abs(b[1]) + b[3] for b in c.boxes)
    body.explicitinertial = True
    body.mass = c.mass
    body.ipos = [c.com[0], c.com[1], c.height / 2]
    Iz = c.mass * ((2 * hxs) ** 2 + (2 * hys) ** 2) / 12
    body.inertia = [c.mass * ((2 * hys) ** 2 + c.height ** 2) / 12, c.mass * ((2 * hxs) ** 2 + c.height ** 2) / 12, Iz]
    for i, (cx, cy, hx, hy) in enumerate(scene.walls):
        g = wb.add_geom(name=f"wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, scene.wall_h / 2],
                        pos=[cx, cy, scene.wall_h / 2], rgba=[0.55, 0.58, 0.65, 1])
        g.friction = [0.3, 0.005, 0.0001]
    if scene.goal is not None:
        g = wb.add_geom(name="goal", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.12, 0.002, 0],
                        pos=[scene.goal[0], scene.goal[1], 0.002], rgba=[0.2, 0.85, 0.3, 0.6])
        g.contype = g.conaffinity = 0
    # translucent "ghost" of the crate showing the model's predicted outcome of the next push
    gh = wb.add_body(name="ghost", mocap=True, pos=[0, 0, -5])
    for i, (cx, cy, hx, hy) in enumerate(c.boxes):
        g = gh.add_geom(name=f"ghost{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, c.height / 2],
                        pos=[cx, cy, c.height / 2], rgba=[0.2, 0.6, 1.0, 0.28])
        g.contype = g.conaffinity = 0
    cam = wb.add_camera(name="overhead", pos=[0.8, 0.0, 4.2], quat=[1, 0, 0, 0])
    cam.fovy = 55
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    x, y, th = scene.robot_start
    data.qpos[0:2] = [x, y]
    data.qpos[3:7] = [np.cos(th / 2), 0, 0, np.sin(th / 2)]
    set_crate(model, data, scene.crate_start)
    mujoco.mj_forward(model, data)
    return model, data


def crate_adr(model):
    j = model.joint("crate_free")
    return int(j.qposadr[0]), int(j.dofadr[0])


def set_crate(model, data, pose):
    qa, va = crate_adr(model)
    x, y, th = pose
    data.qpos[qa:qa + 3] = [x, y, 0.001]
    data.qpos[qa + 3:qa + 7] = [np.cos(th / 2), 0, 0, np.sin(th / 2)]
    data.qvel[va:va + 6] = 0


def crate_pose(model, data):
    qa, _ = crate_adr(model)
    q = data.qpos
    return np.array([q[qa], q[qa + 1], yaw_of(q[qa + 3:qa + 7])])


def crate_speed(model, data):
    _, va = crate_adr(model)
    return float(np.linalg.norm(data.qvel[va:va + 3]))
