"""Test scenes for the TabPFN safety filter.

The robot always starts at (0, 0) facing +x and walks to a1_lidar.GOAL (8, 0).
The collision model was trained only on static, randomly placed boxes with
half-sizes 0.15-0.5 m (datasets/collision_env_*.npz), so every scene below is
new to it, and the dynamic ones are out of its training distribution: a single
lidar scan carries no velocity information.

Static
  random    7 scattered boxes (the scene used by a1_lidar.py)
  dense     14 smaller boxes packed more tightly
  large     4 big boxes (half-size 0.5-0.8 m, larger than anything in training),
            one of them always on the straight line to the goal
  wall      a wall across the path with one 1.2 m gap, 1.0-1.8 m off the centre line
  corridor  two side walls 2 m apart with a slalom of boxes inside

Dynamic (blue boxes, kinematic: they push the robot, never the other way)
  crossing  3 boxes sliding sideways across the path at 0.25-0.45 m/s
  oncoming  1 box walking straight at the robot from behind the goal + 2 static boxes
  mixed     5 static boxes + 2 crossing boxes
"""

from dataclasses import dataclass, field
from typing import Callable

import mujoco
import numpy as np

from a1_lidar import build_model, random_boxes

STATIC = ["random", "dense", "large", "wall", "corridor"]
DYNAMIC = ["crossing", "oncoming", "mixed"]
SCENARIOS = STATIC + DYNAMIC


@dataclass
class Mover:
    size: tuple[float, float]                  # half-sizes x, y
    path: Callable[[float], tuple[float, float]]  # time -> (x, y)


@dataclass
class Scene:
    name: str
    boxes: list
    movers: list[Mover] = field(default_factory=list)

    def __post_init__(self):
        self.model = build_model(self.boxes, movers=[m.size for m in self.movers])
        self._mocap = [self.model.body_mocapid[self.model.body(f"mover{i}").id]
                       for i in range(len(self.movers))]

    def update(self, data):
        """Move the dynamic boxes to their position at data.time."""
        for k, m in zip(self._mocap, self.movers):
            data.mocap_pos[k, :2] = m.path(data.time)

    def obstacle_geoms(self) -> set[int]:
        """Static boxes/walls (world body) and moving boxes."""
        m = self.model
        mover_bodies = {m.body(f"mover{i}").id for i in range(len(self.movers))}
        return {g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
                and (m.geom_bodyid[g] == 0 or m.geom_bodyid[g] in mover_bodies)}


def scatter(rng, n, size=(0.15, 0.4), x=(1.5, 6.5), y=(-1.8, 1.8), spacing=1.2,
            tries=5000, boxes=()):
    """Up to n boxes with centres at least `spacing` apart (including `boxes`)."""
    boxes = list(boxes)
    n += len(boxes)
    for _ in range(tries):
        if len(boxes) == n:
            break
        bx, by = rng.uniform(*x), rng.uniform(*y)
        if all(np.hypot(bx - b[0], by - b[1]) > spacing for b in boxes):
            boxes.append((bx, by, *rng.uniform(*size, size=2)))
    return boxes


def crossing(x, amplitude, speed, phase):
    """Slides along y between -amplitude and +amplitude; peak speed = speed."""
    w = speed / amplitude
    return lambda t: (x, amplitude * np.sin(w * t + phase))


def straight(start, velocity):
    start, velocity = np.asarray(start, float), np.asarray(velocity, float)
    return lambda t: tuple(start + velocity * t)


def make_scene(name: str, seed: int) -> Scene:
    rng = np.random.default_rng(seed)
    if name == "random":   # same layouts as a1_lidar.py --seed N
        return Scene(name, random_boxes(rng))
    if name == "dense":
        return Scene(name, scatter(rng, 14, size=(0.12, 0.3), spacing=0.95))
    if name == "large":   # one big box always on the straight line to the goal
        first = (rng.uniform(3.0, 5.0), rng.uniform(-0.3, 0.3), *rng.uniform(0.5, 0.8, size=2))
        return Scene(name, scatter(rng, 3, size=(0.5, 0.8), x=(2.0, 6.5), spacing=2.4,
                                   boxes=[first]))
    if name == "wall":    # the gap is never on the straight line
        gap_y, half_gap, x = rng.choice([-1, 1]) * rng.uniform(1.0, 1.8), 0.6, 4.0
        lo, hi = gap_y - half_gap, gap_y + half_gap      # wall spans y in [-3, 3]
        return Scene(name, [(x, (-3 + lo) / 2, 0.1, (lo + 3) / 2),
                            (x, (hi + 3) / 2, 0.1, (3 - hi) / 2)])
    if name == "corridor":   # slalom: each box blocks the centre line, ~0.95 m free beside it
        walls = [(4.0, 1.1, 3.5, 0.1), (4.0, -1.1, 3.5, 0.1)]
        side = rng.choice([-1, 1])
        inner = [(x, side * (-1) ** i * 0.25, 0.2, 0.3) for i, x in enumerate((2.5, 4.5, 6.5))]
        return Scene(name, walls + inner)
    if name == "crossing":
        movers = [Mover((0.25, 0.25), crossing(x, 1.5, rng.uniform(0.25, 0.45),
                                               rng.uniform(0, 2 * np.pi)))
                  for x in (2.5, 4.5, 6.5)]
        return Scene(name, [], movers)
    if name == "oncoming":
        movers = [Mover((0.25, 0.25), straight((9.0, rng.uniform(-0.2, 0.2)), (-0.3, 0.0)))]
        return Scene(name, scatter(rng, 2, x=(2.0, 6.0), spacing=2.0), movers)
    if name == "mixed":
        movers = [Mover((0.25, 0.25), crossing(x, 1.5, rng.uniform(0.25, 0.45),
                                               rng.uniform(0, 2 * np.pi)))
                  for x in (3.0, 5.5)]
        return Scene(name, scatter(rng, 5, x=(1.5, 7.0), y=(-2.0, 2.0)), movers)
    raise ValueError(f"unknown scenario {name!r}; choose from {SCENARIOS}")
