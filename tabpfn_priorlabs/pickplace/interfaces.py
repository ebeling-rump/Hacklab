"""Interfaces to sense and act."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from .calibration import T_HAND_CAM, T_HAND_TCP, intrinsics
from .collision import CollisionChecker
from .config import CUBE_HALF, Q_HOME, ObstacleCfg
from .scene import build_robot_only

if TYPE_CHECKING:  # pragma: no cover
    from .env import PickPlaceEnv

N_JOINTS = 7
# Planning limits (conservative; the simulated actuators are position servos).
V_MAX = np.full(N_JOINTS, 1.5)   # rad/s
A_MAX = np.full(N_JOINTS, 3.0)   # rad/s^2
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


@dataclass
class Observation:
    rgb: np.ndarray        # (H, W, 3) uint8
    depth: np.ndarray      # (H, W) float32, metres along the optical axis (z_cam); 0.0 = invalid
    q: np.ndarray          # (7,) joint positions [rad]
    dq: np.ndarray         # (7,) joint velocities [rad/s]
    gripper_width: float   # finger gap [m], 0 = closed, 0.08 = open
    time: float            # simulated seconds since reset


class BinPoseUnknown(RuntimeError):
    """The bin pose is hidden (bin.known: false). Localise the bin and use TaskInfo.with_bin_pose()."""


@dataclass(frozen=True)
class BinInfo:
    """What the agent knows about the bin. Dimensions are always known; xy/yaw are None while hidden."""

    inner: tuple                 # inner edge lengths (x, y) [m]; a hidden bin is always square
    wall_height: float           # wall top above the table [m]
    wall_thickness: float
    floor_thickness: float
    slots: dict = field(default_factory=dict)   # colour -> (x, y) in the bin frame (only with a known pose)
    xy: tuple | None = None      # bin centre in the base frame, None = unknown
    yaw: float | None = None     # rotation about base z [rad], None = unknown
    search_x: tuple = (0.28, 0.52)       # prior for the centre: x range ...
    search_y_abs: tuple = (0.32, 0.50)   # ... and |y| range (the bin may be on either side)

    @property
    def pose_known(self) -> bool:
        return self.xy is not None and self.yaw is not None

    @property
    def outer(self) -> tuple:
        """Outer edge lengths (x, y), i.e. inner + 2 * wall thickness."""
        return (self.inner[0] + 2 * self.wall_thickness, self.inner[1] + 2 * self.wall_thickness)


@dataclass(frozen=True)
class TaskInfo:
    """The *known map*: what a real cell operator would configure. Contains no cube poses; the bin pose is
    only present if the scene says so (bin.known) - otherwise estimate it and call with_bin_pose()."""

    pick_colors: tuple[str, ...]
    cube_size: float
    bin: BinInfo
    obstacles: tuple[ObstacleCfg, ...]
    time_limit_s: float

    @property
    def bin_floor_top(self) -> float:
        return self.bin.floor_thickness

    def with_bin_pose(self, xy, yaw: float) -> "TaskInfo":
        """Copy of this TaskInfo with the (estimated) bin pose filled in. xy: bin centre in the base frame."""
        return dataclasses.replace(self, bin=dataclasses.replace(self.bin, xy=(float(xy[0]), float(xy[1])), yaw=float(yaw)))

    def _need_pose(self) -> None:
        if not self.bin.pose_known:
            raise BinPoseUnknown("The bin pose is not known: localise the bin from the camera and use task.with_bin_pose(xy, yaw)")

    def T_base_bin(self) -> np.ndarray:
        self._need_pose()
        c, s = math.cos(self.bin.yaw), math.sin(self.bin.yaw)
        T = np.eye(4)
        T[:2, :2] = [[c, -s], [s, c]]
        T[:2, 3] = self.bin.xy
        return T

    def bin_to_base(self, p_bin) -> np.ndarray:
        p = np.asarray(p_bin, float)
        out = self.T_base_bin() @ np.array([p[0], p[1], p[2] if p.size > 2 else 0.0, 1.0])
        return out[:3]

    def base_to_bin(self, p_base) -> np.ndarray:
        p = np.asarray(p_base, float)
        out = np.linalg.inv(self.T_base_bin()) @ np.array([p[0], p[1], p[2] if p.size > 2 else 0.0, 1.0])
        return out[:3]

    def in_bin_xy(self, p_base, margin: float = 0.0) -> bool:
        """True if the point lies above the inner bin floor (shrunk by `margin`). Needs a known bin pose."""
        b = self.base_to_bin(p_base)
        return abs(b[0]) <= self.bin.inner[0] / 2 - margin and abs(b[1]) <= self.bin.inner[1] / 2 - margin


class RobotInterface:
    """What the agent sees: sensors, actuators, a private robot model, a collision checker."""

    def __init__(self, env: "PickPlaceEnv"):
        self._env = env
        s = env.cfg.sensor
        self.K = intrinsics(s.width, s.height, s.fovy_deg)       # camera matrix (pixels)
        self.image_size = (s.width, s.height)
        self.T_hand_cam = T_HAND_CAM.copy()                      # hand-eye calibration result
        self.T_hand_tcp = T_HAND_TCP.copy()
        self.control_dt = env.cfg.physics.control_dt
        self.q_home = np.array(Q_HOME)
        self.task = TaskInfo(
            pick_colors=tuple(env.cfg.target_colors()),
            cube_size=2 * CUBE_HALF,
            bin=BinInfo(
                inner=tuple(env.cfg.bin.inner), wall_height=env.cfg.bin.wall_height,
                wall_thickness=env.cfg.bin.wall_thickness, floor_thickness=env.cfg.bin.floor_thickness,
                slots={c: tuple(v) for c, v in env.cfg.bin.slots.items()},
                xy=tuple(env.cfg.bin.xy) if env.cfg.bin.known else None,
                yaw=float(env.cfg.bin.yaw) if env.cfg.bin.known else None,
                search_x=tuple(env.cfg.bin.search_x), search_y_abs=tuple(env.cfg.bin.search_y_abs)),
            obstacles=tuple(env.cfg.obstacles),
            time_limit_s=env.cfg.time_limit_s,
        )

    @property
    def time(self) -> float:
        return self._env.time

    def get_observation(self) -> Observation:
        return self._env.observe()

    def get_joint_state(self) -> tuple[np.ndarray, np.ndarray, float]:
        """(q, dq, gripper_width) without rendering. Cheap; use it inside control loops."""
        return self._env.joint_state()

    def set_joint_targets(self, q) -> None:
        """Position-servo set-points for joints 1-7 (rad). Clipped to the actuator range."""
        self._env.set_arm_ctrl(np.asarray(q, float))

    def set_gripper_width(self, width: float) -> None:
        """Commanded finger gap in metres, clipped to [0, 0.08]. Closing on an object squeezes (<= 100 N)."""
        self._env.set_gripper_ctrl(float(width))

    def step(self, n: int = 1) -> None:
        """Advance simulation by n control periods (control_dt each)."""
        self._env.step(n)

    def load_robot_model(self) -> mujoco.MjModel:
        """A *private* kinematic copy of the Panda (no scene, no cubes). Create your own MjData.
        Use for Jacobians / IK: body name 'hand', joints joint1..joint7."""
        return build_robot_only()

    def collision_checker(self, margin: float = 0.015, task: TaskInfo | None = None):
        """Collision checker for the arm vs table, obstacles and (if its pose is known) the bin.
        With a hidden bin the checker initially contains no bin; after localising it, pass
        `task=task.with_bin_pose(...)` to get one that includes the bin walls."""
        return CollisionChecker(task or self.task, margin)
