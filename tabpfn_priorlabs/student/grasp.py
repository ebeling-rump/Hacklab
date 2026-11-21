"""T6 - Grasp and place pose computation, gripper actions.

Use only top-down grasps with the canonical hand orientation. Adjust the hand's rotation according
to the cube's yaw so that the gripper's closing axis aligns with the cube's face normal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .localization import CubeEstimate
from .transforms import invert_T, make_T, rot_z

R_DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
TCP_OFFSET = 0.1034  # hand -> TCP along hand z


@dataclass
class GraspPlan:
    T_pregrasp: np.ndarray  # hand poses (base frame)
    T_grasp: np.ndarray
    T_lift: np.ndarray


def hand_pose_for_tcp(p_tcp: np.ndarray, yaw: float) -> np.ndarray:
    """Top-down hand pose (R = Rz(yaw) @ R_DOWN) whose TCP is at p_tcp."""
    R = rot_z(yaw) @ R_DOWN
    return make_T(R, np.asarray(p_tcp, float) - R[:, 2] * TCP_OFFSET)


def plan_grasp(cube: CubeEstimate, cube_size: float, approach_height: float = 0.12, lift_z: float = 0.40) -> GraspPlan:
    """Pre-grasp (above), grasp (TCP at the cube centre height) and lift poses, finger axis along a cube face normal."""
    x, y = float(cube.position[0]), float(cube.position[1])
    z = cube_size / 2
    return GraspPlan(
        T_pregrasp=hand_pose_for_tcp([x, y, z + approach_height], cube.yaw),
        T_grasp=hand_pose_for_tcp([x, y, z], cube.yaw),
        T_lift=hand_pose_for_tcp([x, y, lift_z], cube.yaw),
    )


def plan_place(place_xy: np.ndarray, place_yaw: float, floor_top: float, cube_size: float, drop_clearance: float = 0.025,
               transit_z: float = 0.40) -> tuple[np.ndarray, np.ndarray]:
    """(approach pose above the cell, release pose with the cube bottom `drop_clearance` above the bin floor)."""
    x, y = float(place_xy[0]), float(place_xy[1])
    z_release = floor_top + drop_clearance + cube_size / 2   # cube is held at its centre
    return hand_pose_for_tcp([x, y, transit_z], place_yaw), hand_pose_for_tcp([x, y, z_release], place_yaw)


def close_gripper(robot, timeout: float = 1.5) -> float:
    """Close the gripper until the fingers stop moving (or timeout); returns the final finger gap [m]."""
    robot.set_gripper_width(0.0)
    n_max = int(timeout / robot.control_dt)
    prev, still = robot.get_joint_state()[2], 0
    for k in range(n_max):
        robot.step()
        w = robot.get_joint_state()[2]
        still = still + 1 if abs(w - prev) < 2e-5 else 0
        prev = w
        if still >= 8 and k > 10:
            break
    return float(robot.get_joint_state()[2])


def open_gripper(robot, width: float = 0.08, duration: float = 0.4) -> None:
    """Command `width` and step for `duration` simulated seconds."""
    robot.set_gripper_width(width)
    robot.step(max(1, int(round(duration / robot.control_dt))))
