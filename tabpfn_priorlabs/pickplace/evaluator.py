"""Evaluation on ground-truth simulator state. Should not be imported by student code."""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field

import numpy as np

from .config import CUBE_HALF, SceneConfig
from .env import PickPlaceEnv, SimTimeLimit

# objective success criteria
IN_BIN_MARGIN = 0.01        # cube centre must be >= 1 cm inside the inner bin edge (x and y)
Z_TOL = 0.006               # cube centre must be at/above floor_top + half - 6 mm (i.e. not through the floor)
REST_SPEED = 0.02           # m/s, cube must be at rest after the 1 s settle phase
SLOT_TOL = 0.04             # m, distance to the requested slot (bin frame xy)
DISTRACTOR_MOVE_TOL = 0.03  # m, non-target cubes must not be moved further than this
W_PLACED, W_COLLISION, W_CLEAN = 0.70, 0.15, 0.15


@dataclass
class EpisodeResult:
    scenario: str
    digest: str
    n_targets: int
    n_placed: int                 # targets in the bin and (if a slot is requested) within SLOT_TOL of it
    n_in_bin: int
    distractors_disturbed: int
    collision_steps: int          # control steps with robot-obstacle / robot-bin contact
    collision_geoms: list
    sim_time: float
    timed_out: bool
    error: str | None
    per_cube: list = field(default_factory=list)

    @property
    def all_placed(self) -> bool:
        return self.n_placed == self.n_targets

    @property
    def score(self) -> float:
        """Rubric score in [0, 1]: 70% placement fraction, 15% collision-free, 15% clean run."""
        if self.n_targets == 0 or self.n_placed == 0:
            return 0.0
        clean = self.distractors_disturbed == 0 and not self.timed_out and self.error is None
        return (W_PLACED * self.n_placed / self.n_targets
                + W_COLLISION * (self.collision_steps == 0) + W_CLEAN * clean)

    def summary(self) -> str:
        return (f"[{self.scenario} #{self.digest}] placed {self.n_placed}/{self.n_targets} "
                f"collisions={self.collision_steps} distractors_moved={self.distractors_disturbed} "
                f"t={self.sim_time:.1f}s timeout={self.timed_out} error={self.error} score={self.score:.2f}")


class Evaluator:
    def __init__(self, env: PickPlaceEnv):
        self.env = env
        self.initial = [env.gt_cube_pos(i) for i in range(len(env.cfg.cubes))]
        self.collision_steps = 0
        self.collision_geoms: set[str] = set()
        env.add_step_callback(self._on_step)

    def _on_step(self, env: PickPlaceEnv) -> None:
        hits = [n for n in env.gt_robot_contacts() if n.startswith("obstacle") or n.startswith("bin_")]
        if hits:
            self.collision_steps += 1
            self.collision_geoms.update(hits)

    def finalize(self, timed_out: bool = False, error: str | None = None, settle_s: float = 1.0) -> EpisodeResult:
        env, cfg = self.env, self.env.cfg
        env.enforce_time_limit = False
        for _ in range(int(settle_s / cfg.physics.control_dt)):
            env.step(1)
        targets = set(cfg.target_colors())
        b = cfg.bin
        c, s = np.cos(b.yaw), np.sin(b.yaw)
        n_in = n_placed = n_dist = 0
        per = []
        for i, cube in enumerate(cfg.cubes):
            p = env.gt_cube_pos(i)
            d = p - np.array([b.xy[0], b.xy[1], 0.0])
            lx, ly = c * d[0] + s * d[1], -s * d[0] + c * d[1]
            in_bin = (abs(lx) <= b.inner[0] / 2 - IN_BIN_MARGIN and abs(ly) <= b.inner[1] / 2 - IN_BIN_MARGIN
                      and p[2] >= b.floor_thickness + CUBE_HALF - Z_TOL and p[2] <= b.wall_height + CUBE_HALF)
            rest = env.gt_cube_speed(i) <= REST_SPEED
            if cube.color in targets:
                slot = b.slots.get(cube.color)
                slot_ok = True if slot is None else bool(np.hypot(lx - slot[0], ly - slot[1]) <= SLOT_TOL)
                ok = in_bin and rest and slot_ok
                n_in += int(in_bin and rest)
                n_placed += int(ok)
                per.append(dict(color=cube.color, in_bin=bool(in_bin), at_rest=bool(rest), slot_ok=slot_ok, bin_xy=(float(lx), float(ly))))
            else:
                moved = np.linalg.norm(p[:2] - self.initial[i][:2]) > DISTRACTOR_MOVE_TOL or in_bin
                n_dist += int(moved)
                per.append(dict(color=cube.color, distractor=True, moved=bool(moved)))
        return EpisodeResult(cfg.name, cfg.digest(), len(targets), n_placed, n_in, n_dist, self.collision_steps,
                             sorted(self.collision_geoms), env.time, timed_out or env.time >= cfg.time_limit_s - 1e-9,
                             error, per)


def run_episode(cfg: SceneConfig, agent_factory) -> tuple[EpisodeResult, PickPlaceEnv]:
    """Build env + evaluator, run `agent_factory(robot_interface).run()`, return the result."""
    env = PickPlaceEnv(cfg)
    ev = Evaluator(env)
    robot = env.robot_interface()
    timed_out, error = False, None
    try:
        agent_factory(robot).run()
    except SimTimeLimit:
        timed_out = True
    except Exception as e:  # crash: still grade what happened
        error = f"{type(e).__name__}: {e}"
        error += " @ " + traceback.format_exc().strip().splitlines()[-3].strip()
    result = ev.finalize(timed_out, error)
    env.hold_viewer()
    env.close()
    return result, env
