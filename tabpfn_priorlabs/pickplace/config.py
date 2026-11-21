"""Scene configuration: dataclasses, YAML I/O and a seeded random scene generator.

All lengths in metres, angles in radians. All poses are expressed in the robot
BASE frame, which coincides with the world frame.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# RGB (0..1) of the cube materials.
COLORS: dict[str, tuple[float, float, float]] = {
    "red": (0.85, 0.10, 0.10),
    "green": (0.10, 0.70, 0.15),
    "blue": (0.10, 0.25, 0.90),
    "yellow": (0.95, 0.85, 0.10),
    "orange": (0.95, 0.50, 0.05),
    "purple": (0.55, 0.15, 0.70),
}
CUBE_HALF = 0.02  # cube edge = 0.04 m

# Region from which randomize_bin() samples the bin centre (x, |y|; both sides of the robot) and yaw.
BIN_REGION_X = (0.30, 0.50)
BIN_REGION_Y_ABS = (0.36, 0.46)

Q_HOME = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]  # Menagerie "home" keyframe


@dataclass
class CubeCfg:
    color: str
    xy: list[float]
    yaw: float = 0.0


@dataclass
class BinCfg:
    xy: list[float] = field(default_factory=lambda: [0.40, 0.40])
    yaw: float = 0.0
    inner: list[float] = field(default_factory=lambda: [0.24, 0.24])  # inner edge lengths (x, y) in bin frame
    wall_height: float = 0.08  # wall top above table
    wall_thickness: float = 0.01
    floor_thickness: float = 0.01
    slots: dict[str, list[float]] = field(default_factory=dict) # Optional target slot per colour, (x, y) in the BIN frame (origin = bin centre).
    known: bool = True # known=False hides xy/yaw from the agent: the bin must be localised from the camera image.
    # Rough prior (what an operator would know) for where the bin centre can be: x in search_x,
    # |y| in search_y_abs (either side of the robot). Only used/exposed when known=False.
    search_x: list[float] = field(default_factory=lambda: [0.28, 0.52])
    search_y_abs: list[float] = field(default_factory=lambda: [0.32, 0.50])


@dataclass
class ObstacleCfg:
    xy: list[float]
    half: list[float] = field(default_factory=lambda: [0.03, 0.03, 0.10])  # half extents (x, y, z)
    yaw: float = 0.0


@dataclass
class SensorCfg:
    width: int = 320
    height: int = 240
    fovy_deg: float = 60.0
    depth_noise_std: float = 0.001  # m, Gaussian, applied to valid pixels
    rgb_noise_std: float = 1.0  # 8-bit levels


@dataclass
class PhysicsCfg:
    timestep: float = 0.002
    control_dt: float = 0.01  # one RobotInterface.step() = control_dt
    gravcomp: bool = True  # gravity compensation on all arm links
    cube_density: float = 500.0  # kg/m^3 -> 0.032 kg
    friction: float = 1.2  # cube/finger sliding friction


@dataclass
class FaultCfg:
    """Event-triggered failure injection.

    nudge_after_obs: right after the `obs_index`-th camera frame that shows ALL cubes of the scene (every cube
        centre inside the image), cube `cube` is displaced by `delta` (xy). Stale estimates become wrong. Frames that
        do not show the whole workspace (e.g. the bin-search views of a hidden-bin scene) do not count.
    drop_on_lift: the first time cube `cube` is lifted more than 8 cm above the table, the
        gripper is forced open for `duration` s (simulates a slipped grasp).
    """

    type: str
    cube: int = 0
    delta: list[float] = field(default_factory=lambda: [0.03, 0.02])
    obs_index: int = 1
    duration: float = 0.4


@dataclass
class SceneConfig:
    name: str = "default"
    seed: int = 0
    cubes: list[CubeCfg] = field(default_factory=list)
    bin: BinCfg = field(default_factory=BinCfg)
    obstacles: list[ObstacleCfg] = field(default_factory=list)
    pick_colors: Optional[list[str]] = None  # None -> all cube colours must be moved
    sensor: SensorCfg = field(default_factory=SensorCfg)
    physics: PhysicsCfg = field(default_factory=PhysicsCfg)
    faults: list[FaultCfg] = field(default_factory=list)
    time_limit_s: float = 180.0  # simulated seconds

    def target_colors(self) -> list[str]:
        return list(self.pick_colors) if self.pick_colors is not None else sorted({c.color for c in self.cubes})

    def to_dict(self) -> dict:
        return asdict(self)

    def digest(self) -> str:
        """Short content hash."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def _build(cls, d: dict | None):
    return cls(**(d or {}))


def from_dict(d: dict) -> SceneConfig:
    d = copy.deepcopy(d)
    return SceneConfig(
        name=d.get("name", "default"),
        seed=int(d.get("seed", 0)),
        cubes=[_build(CubeCfg, c) for c in d.get("cubes", [])],
        bin=_build(BinCfg, d.get("bin")),
        obstacles=[_build(ObstacleCfg, o) for o in d.get("obstacles", [])],
        pick_colors=d.get("pick_colors"),
        sensor=_build(SensorCfg, d.get("sensor")),
        physics=_build(PhysicsCfg, d.get("physics")),
        faults=[_build(FaultCfg, f) for f in d.get("faults", [])],
        time_limit_s=float(d.get("time_limit_s", 180.0)),
    )


def min_clearances(cfg: SceneConfig) -> dict[str, float]:
    """Smallest cube-cube centre distance, cube-obstacle and cube-bin distances (m) in a scene."""
    bh = [cfg.bin.inner[0] / 2 + cfg.bin.wall_thickness, cfg.bin.inner[1] / 2 + cfg.bin.wall_thickness]
    cc = min([math.dist(a.xy, b.xy) for i, a in enumerate(cfg.cubes) for b in cfg.cubes[i + 1:]] or [9.0])
    co = min([_rect_dist(c.xy, o.xy, o.half[:2], o.yaw) for c in cfg.cubes for o in cfg.obstacles] or [9.0])
    cb = min([_rect_dist(c.xy, cfg.bin.xy, bh, cfg.bin.yaw) for c in cfg.cubes] or [9.0])
    ob = min([_rect_dist(o.xy, cfg.bin.xy, bh, cfg.bin.yaw) - 1.5 * max(o.half[:2]) for o in cfg.obstacles] or [9.0])
    return {"cube_cube": cc, "cube_obstacle": co, "cube_bin": cb, "obstacle_bin": ob}


# Recommended minimum clearances (the hand plate is +/-10 cm wide). Scenes below these are legal but physically hard.
MIN_CLEARANCE = {"cube_cube": 0.08, "cube_obstacle": 0.14, "cube_bin": 0.13, "obstacle_bin": 0.03}


def load_config(path: str | Path) -> SceneConfig:
    with open(path) as f:
        cfg = from_dict(yaml.safe_load(f))
    validate(cfg)
    for k, v in min_clearances(cfg).items():
        if v < MIN_CLEARANCE[k]:
            import warnings
            warnings.warn(f"{cfg.name}: {k} clearance {v:.3f} m < recommended {MIN_CLEARANCE[k]} m")
    return cfg


def validate(cfg: SceneConfig) -> None:
    for c in cfg.cubes:
        if c.color not in COLORS:
            raise ValueError(f"unknown colour {c.color!r}; choose from {sorted(COLORS)}")
    for col in cfg.bin.slots:
        if col not in COLORS:
            raise ValueError(f"slot for unknown colour {col!r}")
    if cfg.pick_colors is not None:
        missing = set(cfg.pick_colors) - {c.color for c in cfg.cubes}
        if missing:
            raise ValueError(f"pick_colors {sorted(missing)} have no cube in the scene")
    if cfg.bin.slots:
        for col in cfg.bin.slots:
            if sum(c.color == col for c in cfg.cubes) > 1:
                raise ValueError(f"slot for {col!r} is ambiguous: more than one cube of that colour")
        hx, hy = cfg.bin.inner[0] / 2, cfg.bin.inner[1] / 2
        for col, (sx, sy) in cfg.bin.slots.items():
            if abs(sx) > hx - 0.06 or abs(sy) > hy - 0.06:
                raise ValueError(f"slot {col!r} too close to the bin wall for the gripper (>= 6 cm needed)")
    if not cfg.bin.known:
        b = cfg.bin
        if abs(b.inner[0] - b.inner[1]) > 1e-9:
            raise ValueError("a hidden bin (known: false) must be square: its yaw is only observable modulo 90 deg")
        if b.slots:
            raise ValueError("slots need a known bin pose (a square bin looks identical under 90 deg turns)")
        if not (b.search_x[0] <= b.xy[0] <= b.search_x[1] and b.search_y_abs[0] <= abs(b.xy[1]) <= b.search_y_abs[1]):
            raise ValueError("the hidden bin centre lies outside its own search prior (search_x / search_y_abs)")
    if cfg.sensor.width < 64 or cfg.sensor.height < 48:
        raise ValueError("sensor resolution too small")


# Seeded random scenes
_BIN_SITES = {"left": ([0.40, 0.42], 0.0), "right": ([0.40, -0.42], 0.0)}


def _rect_dist(p, centre, half_xy, yaw):
    """Distance from point p to an (oriented) rectangle; 0 inside."""
    c, s = math.cos(-yaw), math.sin(-yaw)
    dx, dy = p[0] - centre[0], p[1] - centre[1]
    lx, ly = c * dx - s * dy, s * dx + c * dy
    ex, ey = max(abs(lx) - half_xy[0], 0.0), max(abs(ly) - half_xy[1], 0.0)
    return math.hypot(ex, ey)


def sample_bin_pose(rng: np.random.Generator) -> tuple[list[float], float]:
    """Random bin centre (either side of the robot) and yaw, from BIN_REGION_X / BIN_REGION_Y_ABS."""
    side = 1.0 if rng.random() < 0.5 else -1.0
    xy = [float(rng.uniform(*BIN_REGION_X)), float(side * rng.uniform(*BIN_REGION_Y_ABS))]
    return xy, float(rng.uniform(-math.pi / 2, math.pi / 2))


def randomize_bin(cfg: SceneConfig, seed: int | None = None, hide: bool | None = None) -> SceneConfig:
    """Return a copy of `cfg` whose bin is moved to a random pose (reproducible: same cfg + seed => same bin).

    The new pose keeps the cube/obstacle clearances (MIN_CLEARANCE). Slots are kept only if the bin stays
    known. `hide=True` additionally sets bin.known=False (agent must localise it)."""
    out = copy.deepcopy(cfg)
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    bh = [out.bin.inner[0] / 2 + out.bin.wall_thickness, out.bin.inner[1] / 2 + out.bin.wall_thickness]
    for _ in range(2000):
        xy, yaw = sample_bin_pose(rng)
        ok = all(_rect_dist(c.xy, xy, bh, yaw) >= MIN_CLEARANCE["cube_bin"] for c in out.cubes)
        ok &= all(_rect_dist(o.xy, xy, bh, yaw) - 1.5 * max(o.half[:2]) >= 0.08 for o in out.obstacles)
        if ok:
            out.bin.xy, out.bin.yaw = xy, yaw
            break
    else:
        raise RuntimeError("No free bin pose found for this scene.")
    if hide is not None:
        out.bin.known = not hide
    validate(out)
    return out


def random_scene(
    seed: int,
    n_cubes: int = 3,
    colors: list[str] | None = None,
    n_obstacles: int = 0,
    bin_side: str = "left",
    bin_yaw: float | None = None,
    n_distractors: int = 0,
    with_slots: bool = False,
    randomize_bin_pose: bool = False,
    bin_known: bool = True,
    bin_xy: list[float] | None = None,
    obstacle_spacing: float = 0.20,
    strict_obstacles: bool = False,
    faults: list[FaultCfg] | None = None,
    name: str | None = None,
    **overrides,
) -> SceneConfig:
    """Reproducible random scene. Same arguments + seed => identical SceneConfig.

    randomize_bin_pose=True: bin centre/yaw are sampled from BIN_REGION_* (either side) instead of the fixed
    left/right site; bin_known=False hides the pose from the agent (it must localise it from the image;
    incompatible with with_slots). `bin_xy=[x, y]` fixes the bin centre explicitly.
    `obstacle_spacing`: minimum distance between obstacle centres (default 0.20 m; three obstacles need a smaller value).
    By default a scene may end up with fewer obstacles than requested if it do not fit (kept for reproducibility of
    existing seeds); strict_obstacles=True raises ValueError instead.

    Cubes are sampled in x in [0.30, 0.60], y in [-0.20, 0.20] with >= 8 cm centre spacing,
    >= 14 cm clearance to obstacles, >= 13 cm to the bin (the hand plate is +-10 cm wide). Colours are distinct (max 6 cubes).
    """
    rng = np.random.default_rng(seed)
    palette = list(colors) if colors else list(rng.permutation(sorted(COLORS)))
    n_total = n_cubes + n_distractors
    if n_total > len(COLORS):
        raise ValueError("at most 6 cubes (distinct colours)")
    cols = palette[:n_total] if len(palette) >= n_total else list(rng.permutation(sorted(COLORS)))[:n_total]
    cols = [str(c) for c in cols]

    bxy, _ = _BIN_SITES[bin_side]
    if bin_xy is not None:
        bxy = [float(bin_xy[0]), float(bin_xy[1])]
    byaw = float(rng.uniform(-math.pi / 2, math.pi / 2)) if bin_yaw is None else bin_yaw
    if randomize_bin_pose:
        bxy, byaw = sample_bin_pose(rng) if bin_yaw is None else (sample_bin_pose(rng)[0], bin_yaw)
    cfg_bin = BinCfg(xy=list(bxy), yaw=byaw, known=bin_known)
    bin_half_outer = [cfg_bin.inner[0] / 2 + cfg_bin.wall_thickness, cfg_bin.inner[1] / 2 + cfg_bin.wall_thickness]

    for _attempt in range(100):
        obstacles: list[ObstacleCfg] = []
        for _ in range(n_obstacles):
            for _try in range(200):
                xy = [float(rng.uniform(0.35, 0.55)), float(rng.uniform(-0.15, 0.15))]
                half = [float(rng.uniform(0.02, 0.05)), float(rng.uniform(0.02, 0.05)), float(rng.uniform(0.05, 0.075))]
                if all(math.dist(xy, o.xy) > obstacle_spacing for o in obstacles) and (
                        not randomize_bin_pose
                        or _rect_dist(xy, bxy, bin_half_outer, byaw) - 1.5 * max(half[:2]) >= 0.08):
                    obstacles.append(ObstacleCfg(xy=xy, half=half, yaw=float(rng.uniform(0, math.pi / 2))))
                    break
        cubes: list[CubeCfg] = []
        for col in cols:
            for _try in range(300):
                xy = [float(rng.uniform(0.30, 0.60)), float(rng.uniform(-0.20, 0.20))]
                ok = all(math.dist(xy, c.xy) >= 0.08 for c in cubes)
                ok &= _rect_dist(xy, bxy, bin_half_outer, byaw) >= 0.13
                ok &= all(_rect_dist(xy, o.xy, o.half[:2], o.yaw) >= 0.14 for o in obstacles)
                if ok:
                    cubes.append(CubeCfg(color=col, xy=xy, yaw=float(rng.uniform(-math.pi / 2, math.pi / 2))))
                    break
            else:
                break
        if len(cubes) == len(cols):
            break
    else:
        raise RuntimeError("Could not place cubes; relax the scene.")

    if strict_obstacles and len(obstacles) < n_obstacles:
        raise ValueError(f"only {len(obstacles)} of {n_obstacles} obstacles fit; lower obstacle_spacing")
    pick = cols[:n_cubes] if n_distractors else None
    if with_slots:
        grid = [[-0.045, -0.045], [0.045, -0.045], [-0.045, 0.045], [0.045, 0.045]]
        cfg_bin.slots = {col: grid[i] for i, col in enumerate(cols[:min(n_cubes, 4)])}

    cfg = SceneConfig(
        name=name or f"random_s{seed}_n{n_cubes}",
        seed=seed,
        cubes=cubes,
        bin=cfg_bin,
        obstacles=obstacles,
        pick_colors=pick,
        faults=list(faults or []),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    validate(cfg)
    return cfg
