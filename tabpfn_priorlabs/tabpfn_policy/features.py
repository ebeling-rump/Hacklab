"""Student-side state features, built ONLY from robot-side signals (camera, encoders, known task info).

CubeTracker keeps a belief of the target cube pose from the wrist RGB-D camera:
  src 0 = seen this frame (detect_cubes + estimate_cube_pose)
  src 1 = not seen, but the finger gap says something cube-sized is held -> cube is at the TCP
  src 2 = not seen -> last estimate, aged
Feature units: mm, mm/s, rad (see FEATURES).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from student.localization import estimate_cube_pose
from student.perception import detect_cubes
from student.transforms import camera_pose

from .expert import ExpertState
from .geometry import wrap90

FEATURES = (
    "cube_dx", "cube_dy", "cube_dz",     # cube belief - TCP [mm]
    "cube_dyaw",                         # wrap90(cube yaw - TCP yaw) [rad]
    "bin_dx", "bin_dy",                  # bin centre - TCP [mm]
    "bin_dyaw",                          # wrap90(bin yaw - TCP yaw) [rad]
    "tcp_z",                             # TCP height above the table [mm]
    "tcp_yaw",                           # absolute gripper yaw [rad] (encoders; wrist-range awareness)
    "width", "width_vel",                # finger gap [mm], its rate [mm/s]
    "vx", "vy", "vz",                    # TCP velocity [mm/s]
    "cube_vis", "cube_src", "cube_age",  # perception bookkeeping
)


@dataclass
class CubeBelief:
    pos: np.ndarray | None = None
    yaw: float = 0.0
    src: int = 2
    age: int = 0
    n_points: int = 0


class CubeTracker:
    def __init__(self, robot, color: str, held_window=(0.032, 0.060)):
        self.robot, self.color = robot, color
        self.held_window = held_window
        self.b = CubeBelief()

    def update(self, obs, tcp_pos: np.ndarray) -> CubeBelief:
        T_base_cam = camera_pose(obs.q, self.robot.T_hand_cam)
        best = None
        for det in detect_cubes(obs.rgb, [self.color]):
            est = estimate_cube_pose(det, obs.depth, self.robot.K, T_base_cam, self.robot.task.cube_size)
            if est is not None and (best is None or est.n_points > best.n_points):
                best = est
        if best is not None:
            b = self.b
            b.pos, b.yaw, b.src, b.age, b.n_points = best.position.copy(), float(best.yaw), 0, 0, best.n_points
            return b
        return self.hold(obs.gripper_width, tcp_pos)

    def hold(self, gripper_width: float, tcp_pos: np.ndarray) -> CubeBelief:
        """No (usable) camera frame this step: in-hand belief if the fingers hold something, else age the estimate."""
        b = self.b
        if self.held_window[0] <= gripper_width <= self.held_window[1] and b.pos is not None:
            b.pos, b.src, b.age, b.n_points = np.asarray(tcp_pos, float).copy(), 1, 0, 0
        else:
            b.src, b.age, b.n_points = 2, b.age + 1, 0
        return b


def perceived_state(ep, belief: CubeBelief) -> ExpertState:
    """ExpertState built from the belief instead of ground truth (bin pose from the known TaskInfo)."""
    p, yaw = ep.tcp()
    t = ep.robot.task
    cube = belief.pos if belief.pos is not None else np.r_[p[:2], t.cube_size / 2]
    return ExpertState(tcp_pos=p, tcp_yaw=yaw, width=ep.width(), width_vel=ep.width_velocity(),
                       tcp_vel=ep.tcp_velocity(), cube_pos=np.asarray(cube, float), cube_yaw=belief.yaw,
                       bin_xy=np.asarray(t.bin.xy, float), bin_yaw=wrap90(t.bin.yaw), bin_inner=tuple(t.bin.inner),
                       floor_top=t.bin_floor_top, cube_size=t.cube_size)


def feature_vector(s: ExpertState, belief: CubeBelief) -> dict:
    d_cube = (s.cube_pos - s.tcp_pos) * 1000
    d_bin = (s.bin_xy - s.tcp_pos[:2]) * 1000
    v = (np.zeros(3) if s.tcp_vel is None else np.asarray(s.tcp_vel)) * 1000
    return dict(
        cube_dx=d_cube[0], cube_dy=d_cube[1], cube_dz=d_cube[2],
        cube_dyaw=wrap90(s.cube_yaw - s.tcp_yaw),
        bin_dx=d_bin[0], bin_dy=d_bin[1], bin_dyaw=wrap90(s.bin_yaw - s.tcp_yaw),
        tcp_z=s.tcp_pos[2] * 1000, tcp_yaw=s.tcp_yaw,
        width=s.width * 1000, width_vel=s.width_vel * 1000,
        vx=v[0], vy=v[1], vz=v[2],
        cube_vis=int(belief.src == 0), cube_src=belief.src, cube_age=belief.age,
    )
