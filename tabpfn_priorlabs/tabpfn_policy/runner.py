"""Closed-loop execution of a task-space policy: TCP move -> IK -> joint targets -> position servo.

One `SimEpisode` wraps a PickPlaceEnv with a single target cube. Each control period (`ctrl_steps`
sim control steps, default 5 x 10 ms = 20 Hz):
    state  = episode.true_state()                 # or perceived features, for a student
    action = policy(state)                        # [dx, dy, dz, dyaw, grip]
    episode.apply(action)                         # IK on the actual TCP + delta, servo, step
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pickplace.config import SceneConfig, random_scene
from pickplace.env import PickPlaceEnv
from pickplace.evaluator import Evaluator
from student.grasp import hand_pose_for_tcp
from student.planning import solve_ik
from student.transforms import fk_hand

from .expert import Expert, ExpertState, Phase
from .geometry import quat_yaw, tcp_yaw, wrap90


def make_scene(seed: int, randomize_bin: bool = True) -> SceneConfig:
    """One cube, bin pose known to the agent; bin pose randomised (either side, any yaw) by default."""
    return random_scene(seed, n_cubes=1, randomize_bin_pose=randomize_bin, bin_known=True,
                        name=f"tabpfn_s{seed}")


@dataclass
class StepInfo:
    ik_ok: bool
    tcp_pos: np.ndarray
    tcp_yaw: float


class SimEpisode:
    def __init__(self, cfg: SceneConfig, ctrl_steps: int = 5, lead: float = 2.5, viewer=None):
        """lead: the position servo covers only ~40 % of a commanded step per 50 ms, so the IK target is
        placed `lead` x the commanded delta ahead of the actual TCP (a feed-forward gain; the commanded
        delta is still what the policy outputs and what gets recorded)."""
        self.cfg = cfg
        self.env = PickPlaceEnv(cfg, viewer=viewer)
        self.evaluator = Evaluator(self.env)
        self.robot = self.env.robot_interface()
        self.model = self.robot.load_robot_model()
        self.ctrl_steps = ctrl_steps
        self.lead = lead
        self.ik_failures = 0
        self._prev_tcp = self.tcp()[0]
        self._tcp_vel = np.zeros(3)
        self._prev_w = self.width()
        self._w_vel = 0.0

    # ---- robot-side quantities (no privileged access) ---------------------------------------------
    @property
    def dt(self) -> float:
        return self.ctrl_steps * self.robot.control_dt

    def tcp(self) -> tuple[np.ndarray, float]:
        q = self.robot.get_joint_state()[0]
        T = fk_hand(q) @ self.robot.T_hand_tcp
        return T[:3, 3].copy(), tcp_yaw(T[:3, :3])

    def tcp_velocity(self) -> np.ndarray:
        """Finite difference of the TCP over the last control period [m/s]."""
        return self._tcp_vel.copy()

    def width_velocity(self) -> float:
        return self._w_vel

    def width(self) -> float:
        return float(self.robot.get_joint_state()[2])

    # ---- privileged ground truth (LABELING ONLY) ---------------------------------------------------
    def true_cube(self, i: int = 0) -> tuple[np.ndarray, float]:
        a = self.env._cube_qadr[i]
        qpos = self.env.data.qpos
        return qpos[a:a + 3].copy(), wrap90(quat_yaw(qpos[a + 3:a + 7]))

    def true_state(self) -> ExpertState:
        p, yaw = self.tcp()
        cp, cyaw = self.true_cube()
        b = self.cfg.bin
        return ExpertState(tcp_pos=p, tcp_yaw=yaw, width=self.width(), width_vel=self._w_vel,
                           tcp_vel=self._tcp_vel.copy(), cube_pos=cp, cube_yaw=cyaw,
                           bin_xy=np.asarray(b.xy, float), bin_yaw=wrap90(b.yaw), bin_inner=tuple(b.inner),
                           floor_top=b.floor_thickness, cube_size=self.robot.task.cube_size)

    # ---- acting ------------------------------------------------------------------------------------
    def apply(self, action) -> StepInfo:
        a = np.asarray(action, float)
        q = self.robot.get_joint_state()[0]
        p, yaw = self.tcp()
        p_tgt = p + self.lead * a[:3]
        p_tgt[2] = max(p_tgt[2], 0.005)                     # never command the TCP into the table
        T_hand = hand_pose_for_tcp(p_tgt, yaw + self.lead * a[3])
        q_tgt = solve_ik(self.model, T_hand, q, max_iter=60, pos_tol=2e-4, rot_tol=2e-3)
        ok = q_tgt is not None
        if ok:
            self.robot.set_joint_targets(q_tgt)
        else:
            self.ik_failures += 1                            # keep the previous joint targets
        self.robot.set_gripper_width(float(a[4]))
        self.robot.step(self.ctrl_steps)
        p_new, yaw_new = self.tcp()
        self._tcp_vel = (p_new - self._prev_tcp) / self.dt
        self._prev_tcp = p_new
        w = self.width()
        self._w_vel = (w - self._prev_w) / self.dt
        self._prev_w = w
        return StepInfo(ok, p_new, yaw_new)

    def finish(self):
        """Grade with the official evaluator (1 s settle), close the env. Returns EpisodeResult."""
        res = self.evaluator.finalize()
        self.env.hold_viewer()          # no-op unless a viewer with hold > 0 is open
        self.env.close()
        return res


HELD_PHASES = (Phase.GRASP, Phase.LIFT, Phase.CARRY, Phase.LOWER)


def run_expert_episode(seed: int, expert: Expert | None = None, max_steps: int = 600, noise: float = 0.0,
                       rng: np.random.Generator | None = None, on_step=None, randomize_bin: bool = True,
                       held_noise_scale: float = 0.3, viewer=None):
    """Run the expert closed loop on `make_scene(seed)`.

    noise: std of Gaussian action noise as a fraction of the step clip (DART-style), scaled by held_noise_scale
    while the cube is being grasped / held (noise there mostly just shakes the cube out of the weak grip); the
    EXECUTED action is noisy, the expert's clean action is what `on_step` receives as the label.
    on_step(episode, step, phase, clean_action, executed_action) is called BEFORE the action is applied.
    Returns (EpisodeResult, dict of episode stats).
    """
    expert = expert or Expert()
    rng = rng or np.random.default_rng(seed + 10_000)
    ep = SimEpisode(make_scene(seed, randomize_bin), viewer=viewer)
    phase = Phase.APPROACH
    phases_seen, steps = [], 0
    try:
        for steps in range(max_steps):
            a, phase = expert.act(ep.true_state(), phase)
            if not phases_seen or phases_seen[-1] != int(phase):
                phases_seen.append(int(phase))
            if phase == Phase.DONE:
                break
            a_exec = a.copy()
            if noise > 0:
                c = expert.cfg
                sc = noise * (held_noise_scale if phase in HELD_PHASES else 1.0)
                a_exec[:3] += rng.normal(0, sc * c.max_step, 3)
                a_exec[3] += rng.normal(0, sc * c.max_dyaw)
            if on_step is not None:
                on_step(ep, steps, phase, a, a_exec)
            ep.apply(a_exec)
    finally:
        res = ep.finish()
    stats = dict(seed=seed, steps=steps, sim_time=steps * ep.dt, final_phase=int(phase), phases=phases_seen,
                 ik_failures=ep.ik_failures, success=res.n_placed == res.n_targets, collisions=res.collision_steps)
    return res, stats
