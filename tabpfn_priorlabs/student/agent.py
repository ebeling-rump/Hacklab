"""Agent: The PROVIDED orchestration plus two methods that you must write.

Provided (read it, you may refine it, but do not rename it): `__init__`, `ik_candidates`,
`goto_pose`, `line_to`, `scan`, `bin_scan_poses` and the main loop `run`.

YOU: T8 `locate_bin` (find and localise a hidden bin)
     T9 `pick_and_place` (grasp one cube, carry it, place it in the bin, recover from failures)

Note for T8/T9: you can use the helper functions by the names imported below (`plan_grasp`,
`plan_place`, `open_gripper`, etc.) and the agent's own `self.goto_pose` / `self.line_to` /
`self.policy` fields.
"""
from __future__ import annotations

import numpy as np

from pickplace.interfaces import RobotInterface

from .control import JointTrajectory, execute_trajectory, settle
from .grasp import close_gripper, open_gripper, plan_grasp, plan_place
from .localization import estimate_bin_pose, estimate_cube_pose
from .perception import detect_cubes
from .planning import cartesian_path, plan_path, plan_task, solve_ik
from .recovery import Action, GraspStatus, RecoveryPolicy, classify_grasp, still_holding
from .transforms import camera_pose

_R_DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])   # top-down hand orientation (see grasp.py)


def _top_down_hand_pose(p_tcp, yaw: float = 0.0) -> np.ndarray:
    """
    PROVIDED: hand pose whose TCP is at p_tcp, pointing straight down. Used for the fixed
    observation poses so that scanning/locating the bin does not depend on your grasp code.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ _R_DOWN
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(p_tcp, float) - R[:, 2] * 0.1034
    return T


SCAN_TCP = np.array([0.47, 0.09, 0.55])   # TCP position of the observation pose (base frame)
RELEASE_WIDTH = 0.06
# Where to look for a hidden bin: the camera looks ~9 cm towards -y of the hand, so a hand at y = +/-y_mid + 0.09
# centres the view on a bin at y = +-y_mid. With these poses a bin anywhere in the default prior
# (x 0.28-0.52, |y| 0.32-0.50, any yaw) is completely inside the image.
BIN_SCAN_X, BIN_SCAN_Z, CAM_Y_OFFSET = 0.43, 0.55, 0.09


class PickPlaceAgent:
    def __init__(self, robot: RobotInterface, seed: int = 0):
        self.robot = robot
        self.task = robot.task
        self.model = robot.load_robot_model()
        self.checker = robot.collision_checker()
        self.rng = np.random.default_rng(seed)
        self.policy = RecoveryPolicy(max_attempts=3)
        self.placed_bin_xy: list[np.ndarray] = []
        top = max([o.half[2] * 2 for o in self.task.obstacles] + [self.task.bin.wall_height, 0.0])
        self.transit_z = max(0.35, top + 0.20)
        self.log: list[str] = []

    # Motion helpers
    def _q(self) -> np.ndarray:
        return self.robot.get_joint_state()[0]

    def _play(self, waypoints, monitor=None) -> bool:
        traj = JointTrajectory(waypoints)
        ok = execute_trajectory(self.robot, traj, monitor)
        if ok:
            settle(self.robot, np.asarray(waypoints[-1]), tol=0.004, timeout=1.5)
        return ok

    def ik_candidates(self, T_hand) -> list:
        """Distinct IK solutions for a hand pose, best posture first.

        The Panda is redundant (7 joints for a 6D pose): an IK started from the current posture
        keeps its drifted elbow angle and can end up in a branch from which straight Cartesian moves
        are impossible. We therefore rank solutions by distance to the nominal ready posture (joints
        2-6; joint 1 follows the target azimuth, joint 7 the grasp yaw).
        """
        q0 = self._q()
        az = float(np.arctan2(T_hand[1, 3], T_hand[0, 3]))
        seeds = [np.r_[az, self.robot.q_home[1:]], self.robot.q_home, np.r_[az, q0[1:]], q0]
        sols: list[np.ndarray] = []
        for seed in seeds:
            q = solve_ik(self.model, T_hand, seed)
            if q is not None and not any(np.allclose(q, o, atol=1e-3) for o in sols):
                sols.append(q)
        sols.sort(key=lambda q: float(np.linalg.norm((q - self.robot.q_home)[1:6])))
        return sols

    def goto_pose(self, T_hand, gripper_width: float = 0.08, monitor=None, require=None) -> bool:
        """
        Free-space move (collision-checked joint-space path) to a hand pose, using the best IK branch.
        `require(q)` may veto a candidate, e.g. when the following straight-line move is impossible
        from it.
        """
        q0 = self._q()
        for q_goal in self.ik_candidates(T_hand):
            if require is not None and not require(q_goal):
                continue
            path = plan_path(q0, q_goal, self.checker, self.rng, gripper_width=gripper_width)
            if path is not None:
                return self._play(path, monitor)
        self.log.append("goto_pose: no IK/path")
        return False

    def line_to(self, T_hand, monitor=None) -> bool:
        """Straight Cartesian move."""
        path = cartesian_path(self.model, self._q(), T_hand)
        if path is None:
            self.log.append("line_to: IK failed")
            return False
        return self._play(path, monitor)

    def scan(self):
        """Observes the scene and returns cube positions."""
        T_scan = _top_down_hand_pose(SCAN_TCP, 0.0)
        self.goto_pose(T_scan)
        for _ in range(10):  # let the arm stop shaking before taking the picture
            self.robot.step()
        obs = self.robot.get_observation()
        T_base_cam = camera_pose(obs.q, self.robot.T_hand_cam)
        colors = [c for c in self.task.pick_colors if c not in self.policy.given_up]
        ests = []
        for det in detect_cubes(obs.rgb, colors):
            est = estimate_cube_pose(det, obs.depth, self.robot.K, T_base_cam, self.task.cube_size)
            if est is not None:
                ests.append(est)
        return ests

    def bin_scan_poses(self):
        ym = 0.5 * (self.task.bin.search_y_abs[0] + self.task.bin.search_y_abs[1])
        return [_top_down_hand_pose([BIN_SCAN_X, ym + CAM_Y_OFFSET, BIN_SCAN_Z], 0.0),     # bin on the +y side
                _top_down_hand_pose([BIN_SCAN_X, -ym + CAM_Y_OFFSET, BIN_SCAN_Z], 0.0)]    # bin on the -y side

    def locate_bin(self) -> bool:
        """Find the hidden bin from the two observation poses, update task and collision checker."""
        if self.task.bin.pose_known:
            self.log.append("locate_bin: pose already known")
            return True
        for T_scan in self.bin_scan_poses():
            if not self.goto_pose(T_scan):
                continue
            for _ in range(10):
                self.robot.step()
            obs = self.robot.get_observation()
            T_base_cam = camera_pose(obs.q, self.robot.T_hand_cam)
            est = estimate_bin_pose(obs.rgb, obs.depth, self.robot.K, T_base_cam, self.task.bin)
            if est is not None:
                self.task = self.task.with_bin_pose(est.position, est.yaw)
                self.checker = self.robot.collision_checker(task=self.task)
                self.log.append(f"locate_bin: found at {np.round(est.position, 3)} yaw {est.yaw:.3f}")
                return True
        self.log.append("locate_bin: bin not found")
        return False

    def _abort_grasp(self, color: str, status: GraspStatus, T_up) -> bool:
        """Open the gripper, retreat upwards and record the failure."""
        open_gripper(self.robot, 0.08, duration=0.3)
        self.policy.next_action(color, status)
        self.line_to(T_up)
        return False

    def pick_and_place(self, goal) -> bool:
        cube, size = goal.cube, self.task.cube_size
        color = cube.color
        gp = plan_grasp(cube, size, lift_z=self.transit_z)
        if not self.goto_pose(gp.T_pregrasp, 0.08):
            self.log.append(f"pick {color}: cannot reach pregrasp")
            self.policy.next_action(color, GraspStatus.EMPTY)
            return False
        if not self.line_to(gp.T_grasp):
            self.log.append(f"pick {color}: descent failed")
            return self._abort_grasp(color, GraspStatus.EMPTY, gp.T_pregrasp)
        width = close_gripper(self.robot)
        status = classify_grasp(width, size)
        if status != GraspStatus.HELD:
            self.log.append(f"pick {color}: grasp {status.value} (w={width:.4f})")
            return self._abort_grasp(color, status, gp.T_pregrasp)
        holding = lambda: still_holding(self.robot.get_joint_state()[2], size)
        if not self.line_to(gp.T_lift, monitor=holding):
            self.log.append(f"pick {color}: lost/failed lift")
            return self._abort_grasp(color, GraspStatus.EMPTY, gp.T_pregrasp)
        self.policy.next_action(color, GraspStatus.HELD)
        T_above, T_release = plan_place(goal.place_xy, goal.place_yaw, self.task.bin_floor_top, size, transit_z=self.transit_z)
        if not (self.goto_pose(T_above, width, monitor=holding) and self.line_to(T_release, monitor=holding)):
            self.log.append(f"place {color}: transport failed")
            open_gripper(self.robot, 0.08, duration=0.3)
            self.policy.next_action(color, GraspStatus.EMPTY)
            self.line_to(T_above)
            return False
        open_gripper(self.robot, RELEASE_WIDTH, duration=0.4)
        self.placed_bin_xy.append(np.asarray(goal.place_xy, float))
        self.line_to(T_above)
        self.log.append(f"placed {color}")
        return True

    def run(self) -> None:
        if not self.locate_bin():
            return
        max_rounds = 4 * len(self.task.pick_colors) + 4
        for _ in range(max_rounds):
            goals = plan_task(self.scan(), self.task, self.placed_bin_xy)
            goals = [g for g in goals if g.cube.color not in self.policy.given_up]
            if not goals:
                break
            self.pick_and_place(goals[0])
        # Park at the observation pose with an open gripper
        open_gripper(self.robot, 0.08, duration=0.2)
        self.goto_pose(_top_down_hand_pose(SCAN_TCP, 0.0))
