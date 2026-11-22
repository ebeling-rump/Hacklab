"""Challenge suite: progressively harder pushing tasks (planar, MuJoCo physics)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .objects import sample_object, PushObject, sample_hidden
from .shapes import rect, l_shape, t_shape


@dataclass
class Task:
    challenge: str
    seed: int
    obj: PushObject
    start: tuple
    goal: tuple
    pos_tol: float = 0.02
    ang_tol: float | None = None        # None = orientation free
    walls: list = field(default_factory=list)
    bounds: tuple = (-0.6, 0.6, -0.6, 0.6)
    max_pushes: int = 20

    @property
    def needs_field(self):
        return bool(self.walls)


def _rand_goal(rng, dist=(0.25, 0.35)):
    a = rng.uniform(-np.pi, np.pi)
    r = rng.uniform(*dist)
    return r * np.cos(a), r * np.sin(a)


def c1_basic(seed):
    """Push a simple object to a target position (unknown mass/friction/actuator gain)."""
    rng = np.random.default_rng(10_000 + seed)
    obj = sample_object(rng, str(rng.choice(["rect", "square", "disc"])), com_frac=0.15)
    gx, gy = _rand_goal(rng)
    return Task("C1_basic", seed, obj, (0, 0, rng.uniform(-np.pi, np.pi)), (gx, gy, 0.0), 0.02, None,
                max_pushes=15)


def c2_unknown(seed):
    """Position target, any object family with large hidden-parameter ranges."""
    rng = np.random.default_rng(20_000 + seed)
    obj = sample_object(rng, None, com_frac=0.35)
    gx, gy = _rand_goal(rng)
    return Task("C2_unknown", seed, obj, (0, 0, rng.uniform(-np.pi, np.pi)), (gx, gy, 0.0), 0.02, None,
                max_pushes=15)


def c3_precision(seed):
    """Full pose target (x, y, theta)."""
    rng = np.random.default_rng(30_000 + seed)
    obj = sample_object(rng, str(rng.choice(["rect", "square"])), com_frac=0.25)
    gx, gy = _rand_goal(rng)
    th0 = rng.uniform(-np.pi, np.pi)
    dth = rng.choice([-1, 1]) * rng.uniform(np.radians(45), np.radians(120))
    return Task("C3_precision", seed, obj, (0, 0, th0), (gx, gy, float(np.angle(np.exp(1j * (th0 + dth))))),
                0.02, np.radians(8), max_pushes=20)


def c4_asymmetric(seed):
    """Pose target with L/T/U/Z/plus objects and strongly off-centre mass."""
    rng = np.random.default_rng(40_000 + seed)
    obj = sample_object(rng, str(rng.choice(["L", "T", "U", "Z", "plus"])), com_frac=0.5)
    gx, gy = _rand_goal(rng)
    th0 = rng.uniform(-np.pi, np.pi)
    dth = rng.choice([-1, 1]) * rng.uniform(np.radians(45), np.radians(120))
    return Task("C4_asymmetric", seed, obj, (0, 0, th0), (gx, gy, float(np.angle(np.exp(1j * (th0 + dth))))),
                0.02, np.radians(8), max_pushes=20)


def c5_doorway(seed):
    """An elongated / L-shaped object must be rotated to pass a doorway narrower than its length."""
    rng = np.random.default_rng(50_000 + seed)
    kind = ["rect", "L", "T"][seed % 3]
    if kind == "rect":
        shape = rect(rng.uniform(0.22, 0.26), rng.uniform(0.07, 0.09))
    elif kind == "L":
        shape = l_shape(rng.uniform(0.22, 0.26), rng.uniform(0.09, 0.11), 0.06)
    else:
        shape = t_shape(rng.uniform(0.20, 0.24), rng.uniform(0.10, 0.12), 0.06)
    obj = PushObject(f"door_{kind}_{seed}", shape, sample_hidden(rng, shape, com_frac=0.4))
    w, h = shape.extents
    short = min(w, h)
    # minimal width over orientations ~ the "short" side for these shapes
    door = short + rng.uniform(0.035, 0.05)
    t = 0.03
    L = 0.7
    walls = [(0.0, (door / 2 + L) / 2 + door / 4, t / 2, (L - door / 2) / 2),
             (0.0, -((door / 2 + L) / 2 + door / 4), t / 2, (L - door / 2) / 2)]
    # recompute exactly: wall segments from |y| = door/2 to |y| = L
    walls = [(0.0, (door / 2 + L) / 2, t / 2, (L - door / 2) / 2),
             (0.0, -(door / 2 + L) / 2, t / 2, (L - door / 2) / 2)]
    long_axis_is_x = w >= h
    th0 = (np.pi / 2 if long_axis_is_x else 0.0) + rng.uniform(-0.25, 0.25)  # broadside to the door
    start = (-0.32, rng.uniform(-0.12, 0.12), th0)
    goal = (0.32, rng.uniform(-0.1, 0.1), 0.0)
    return Task("C5_doorway", seed, obj, start, goal, 0.03, None, walls, (-0.55, 0.55, -0.5, 0.5),
                max_pushes=40)


def c6_cluttered(seed):
    """Navigate an object through a cluttered room with an L-shaped corridor and pillars."""
    rng = np.random.default_rng(60_000 + seed)
    shape = rect(rng.uniform(0.16, 0.2), rng.uniform(0.07, 0.09)) if seed % 2 == 0 else \
        l_shape(rng.uniform(0.16, 0.19), rng.uniform(0.09, 0.11), 0.055)
    obj = PushObject(f"clutter_{seed}", shape, sample_hidden(rng, shape, com_frac=0.35))
    t = 0.03
    walls = [  # outer box
        (0.0, 0.62, 0.72, t / 2), (0.0, -0.62, 0.72, t / 2), (0.72, 0.0, t / 2, 0.62), (-0.72, 0.0, t / 2, 0.62),
        # inner wall forcing an L-route, with a gap
        (-0.1, 0.25, t / 2, 0.37),
        (0.2, -0.12, 0.3, t / 2),
    ]
    # pillars
    for _ in range(2):
        px, py = rng.uniform(-0.45, -0.3), rng.uniform(-0.35, 0.0)
        walls.append((px, py, 0.03, 0.03))
    start = (-0.45, 0.4, rng.uniform(-np.pi, np.pi))
    goal = (0.45, 0.35, 0.0)
    return Task("C6_cluttered", seed, obj, start, goal, 0.03, None, walls, (-0.7, 0.7, -0.6, 0.6),
                max_pushes=60)


CHALLENGES = {"C1": c1_basic, "C2": c2_unknown, "C3": c3_precision, "C4": c4_asymmetric,
              "C5": c5_doorway, "C6": c6_cluttered}
