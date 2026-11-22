"""Physical objects = observable shape + HIDDEN physical parameters.

The hidden parameters (mass, frictions, centre of mass, inertia, actuator gain) are used
only by the simulator and by evaluation code. Models only ever see `Shape.descriptor()`
and the outcomes of their own pushes.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from shapely.geometry import Point

from .shapes import Shape, random_shape, FAMILIES


@dataclass
class Hidden:
    mass: float          # kg
    mu_floor: float      # object-floor friction
    mu_pusher: float     # object-pusher friction
    com: tuple           # centre of mass in body frame (m)
    inertia_scale: float # I_z relative to uniform density
    act_gain: float      # actual pusher travel / commanded travel (actuator miscalibration)
    casters: bool = False     # rides on 4 wheels -> strongly anisotropic friction
    caster_axis: float = 0.0  # wheel axle direction in body frame (rad)
    roll_damping: float = 0.002

    def as_dict(self):
        d = asdict(self)
        d["com_x"], d["com_y"] = d.pop("com")
        return d


@dataclass
class PushObject:
    name: str
    shape: Shape
    hidden: Hidden
    height: float = 0.05

    @property
    def descriptor(self):
        return self.shape.descriptor()


def sample_hidden(rng: np.random.Generator, shape: Shape, com_frac: float = 0.35,
                  ranges: dict | None = None, p_casters: float = 0.0) -> Hidden:
    r = dict(mass=(0.3, 2.5), mu_floor=(0.2, 0.7), mu_pusher=(0.1, 0.8),
             inertia=(0.7, 1.4), gain=(0.8, 1.15))
    if ranges:
        r.update(ranges)
    minx, miny, maxx, maxy = shape.polygon.bounds
    # CoM: uniformly inside the (shrunk) polygon, so it is physically plausible.
    inner = shape.polygon.buffer(-0.01)
    for _ in range(1000):
        c = np.array([rng.uniform(minx, maxx), rng.uniform(miny, maxy)]) * com_frac
        if inner.contains(Point(*c)):
            break
    else:
        c = np.zeros(2)
    return Hidden(mass=float(rng.uniform(*r["mass"])),
                  mu_floor=float(rng.uniform(*r["mu_floor"])),
                  mu_pusher=float(rng.uniform(*r["mu_pusher"])),
                  com=(float(c[0]), float(c[1])),
                  inertia_scale=float(rng.uniform(*r["inertia"])),
                  act_gain=float(rng.uniform(*r["gain"])),
                  casters=bool(rng.random() < p_casters),
                  caster_axis=float(rng.choice([0.0, np.pi / 2]) + rng.normal(0, 0.1)),
                  roll_damping=float(10 ** rng.uniform(-5, -3.5)))


def sample_object(rng: np.random.Generator, family: str | None = None, scale: float = 1.0,
                  com_frac: float = 0.35, ranges: dict | None = None, name: str | None = None,
                  p_casters: float = 0.0) -> PushObject:
    family = family or str(rng.choice(FAMILIES))
    shape = random_shape(rng, family, scale)
    hidden = sample_hidden(rng, shape, com_frac, ranges, p_casters)
    return PushObject(name or f"{family}_{rng.integers(1e6):06d}", shape, hidden)


# Families used for training/meta-experience vs. held-out families for transfer tests.
TRAIN_FAMILIES = ["rect", "square", "L", "disc", "U"]
HELDOUT_FAMILIES = ["T", "plus", "Z"]


def object_set(seed: int, n: int, families=None, **kw) -> list[PushObject]:
    rng = np.random.default_rng(seed)
    fams = families or FAMILIES
    return [sample_object(rng, fams[i % len(fams)], **kw) for i in range(n)]
