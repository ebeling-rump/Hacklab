"""Privileged simulation environment. Owns MuJoCo model/data."""
from __future__ import annotations

from contextlib import nullcontext

import mujoco
import numpy as np

from .calibration import GRIPPER_MAX_WIDTH, intrinsics, T_HAND_TCP
from .config import CUBE_HALF, Q_HOME, SceneConfig, validate
from .interfaces import Observation, RobotInterface
from .scene import CAM_NAME, ROBOT_BODIES, build_scene, yaw_quat
from .viewer import ViewerOptions, open_viewer


class SimTimeLimit(RuntimeError):
    """Raised by step() when the simulated time limit of the scenario is exceeded."""


class PickPlaceEnv:
    def __init__(self, cfg: SceneConfig, viewer: ViewerOptions | None | str = "auto"):
        """viewer: ViewerOptions to open a live MuJoCo window, None for headless, "auto" (default) = enabled
        only if the environment variable PICKPLACE_VIEWER is set (see pickplace/viewer.py)."""
        validate(cfg)
        self.cfg = cfg
        self.model = build_scene(cfg)
        self.data = mujoco.MjData(self.model)
        self._substeps = max(1, round(cfg.physics.control_dt / cfg.physics.timestep))
        self._renderer: mujoco.Renderer | None = None
        self._callbacks = []
        self.enforce_time_limit = True
        self._viewer = None
        m = self.model
        self._cube_qadr = [m.jnt_qposadr[m.body(f"cube{i}").jntadr[0]] for i in range(len(cfg.cubes))]
        self._cube_vadr = [m.jnt_dofadr[m.body(f"cube{i}").jntadr[0]] for i in range(len(cfg.cubes))]
        self._robot_body_ids = {m.body(n).id for n in ROBOT_BODIES}
        self._cam_id = m.camera(CAM_NAME).id
        self.reset()
        opts = ViewerOptions.from_env() if viewer == "auto" else viewer
        self._viewer = open_viewer(self.model, self.data, opts, self._cam_id)
        self._viewer_sync()

    def reset(self) -> RobotInterface:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:7] = Q_HOME
        d.qpos[7:9] = GRIPPER_MAX_WIDTH / 2
        for i, c in enumerate(self.cfg.cubes):
            a = self._cube_qadr[i]
            d.qpos[a:a + 7] = [c.xy[0], c.xy[1], CUBE_HALF, *yaw_quat(c.yaw)]
        d.ctrl[:7] = Q_HOME
        self._grip_cmd = GRIPPER_MAX_WIDTH
        self._grip_override_until = -1.0
        self._obs_count = 0          # camera frames taken
        self._workspace_obs = 0      # frames that show every cube
        self._fired: set[int] = set()
        self._rng = np.random.default_rng(self.cfg.seed)
        mujoco.mj_forward(m, d)
        self.enforce_time_limit = False
        for _ in range(50):  # 0.5 s settle
            self._substep_block()
        d.time = 0.0
        self.enforce_time_limit = True
        if self._viewer is not None:
            self._viewer.reset_clock(0.0)
            self._viewer_sync(pace=False)
        return self.robot_interface()

    def close(self) -> None:
        """Free the OpenGL context and close the viewer window. Call when done (the autograder does)."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def hold_viewer(self) -> None:
        """If a viewer is open and a hold time was requested, keep it open for inspection (no-op otherwise)."""
        if self._viewer is not None and self._viewer.opts.hold > 0:
            self._viewer.hold()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def robot_interface(self) -> RobotInterface:
        return RobotInterface(self)

    @property
    def time(self) -> float:
        return float(self.data.time)

    def add_step_callback(self, fn) -> None:
        self._callbacks.append(fn)

    def _substep_block(self):
        d = self.data
        open_forced = d.time < self._grip_override_until
        d.ctrl[7] = 255.0 if open_forced else self._grip_cmd / GRIPPER_MAX_WIDTH * 255.0
        with (self._viewer.lock() if self._viewer is not None else nullcontext()):
            for _ in range(self._substeps):
                mujoco.mj_step(self.model, d)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            if self.enforce_time_limit and self.time >= self.cfg.time_limit_s:
                raise SimTimeLimit(f"simulated time limit {self.cfg.time_limit_s}s exceeded")
            self._substep_block()
            self._check_drop_faults()
            for cb in self._callbacks:
                cb(self)
            self._viewer_sync()

    def _viewer_sync(self, pace: bool = True) -> None:
        if self._viewer is None:
            return
        if pace:
            self._viewer.sync(self.time)
        else:
            self._viewer.refresh()

    def set_arm_ctrl(self, q: np.ndarray) -> None:
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("joint targets must be 7 finite numbers")
        lo, hi = self.model.actuator_ctrlrange[:7].T
        self.data.ctrl[:7] = np.clip(q, lo, hi)

    def set_gripper_ctrl(self, width: float) -> None:
        self._grip_cmd = float(np.clip(width, 0.0, GRIPPER_MAX_WIDTH))

    def joint_state(self):
        d = self.data
        return d.qpos[:7].copy(), d.qvel[:7].copy(), float(d.qpos[7] + d.qpos[8])

    def observe(self) -> Observation:
        rgb, depth = self.render_rgbd()
        s = self.cfg.sensor
        if s.rgb_noise_std > 0:
            rgb = np.clip(rgb + self._rng.normal(0, s.rgb_noise_std, rgb.shape), 0, 255).astype(np.uint8)
        valid = depth > 0
        if s.depth_noise_std > 0:
            depth = np.where(valid, depth + self._rng.normal(0, s.depth_noise_std, depth.shape), 0.0).astype(np.float32)
        q, dq, w = self.joint_state()
        obs = Observation(rgb=rgb, depth=depth, q=q, dq=dq, gripper_width=w, time=self.time)
        self._obs_count += 1
        self._check_obs_faults()
        return obs

    def render_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        s = self.cfg.sensor
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, s.height, s.width)
        r = self._renderer
        r.update_scene(self.data, camera=CAM_NAME)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(self.data, camera=CAM_NAME)
        depth = r.render().copy().astype(np.float32)
        r.disable_depth_rendering()
        depth[depth >= self.model.vis.map.zfar * self.model.stat.extent * 0.999] = 0.0
        depth[depth >= 2.99] = 0.0
        return rgb, depth

    def _shows_all_cubes(self, margin: float = 10.0) -> bool:
        """True if the centre of every cube projects inside the current camera image (>= margin px from the border).
        Used so that faults tied to "the first look at the workspace" are not triggered by other views, e.g. the
        bin-search observations of a hidden-bin scene."""
        s = self.cfg.sensor
        K = intrinsics(s.width, s.height, s.fovy_deg)
        Ti = np.linalg.inv(self.gt_camera_T())
        for i in range(len(self.cfg.cubes)):
            p = Ti @ np.append(self.gt_cube_pos(i), 1.0)
            if p[2] <= 0.05:
                return False
            u, v = K[0, 0] * p[0] / p[2] + K[0, 2], K[1, 1] * p[1] / p[2] + K[1, 2]
            if not (margin <= u <= s.width - 1 - margin and margin <= v <= s.height - 1 - margin):
                return False
        return True

    def _check_obs_faults(self):
        if any(f.type == "nudge_after_obs" for f in self.cfg.faults) and self._shows_all_cubes():
            self._workspace_obs += 1
        for k, f in enumerate(self.cfg.faults):
            if f.type == "nudge_after_obs" and k not in self._fired and self._workspace_obs == f.obs_index:
                self._fired.add(k)
                a = self._cube_qadr[f.cube]
                self.data.qpos[a] += f.delta[0]
                self.data.qpos[a + 1] += f.delta[1]
                mujoco.mj_forward(self.model, self.data)

    def _check_drop_faults(self) -> None:
        for k, f in enumerate(self.cfg.faults):
            if f.type == "drop_on_lift" and k not in self._fired:
                p = self.gt_cube_pos(f.cube)
                if p[2] > CUBE_HALF + 0.08 and np.linalg.norm(p - self.gt_tcp_pos()) < 0.05:
                    self._fired.add(k)
                    self._grip_override_until = self.time + f.duration

    # PRIVILEGED ground truth (for grading only)
    def gt_cube_pos(self, i: int) -> np.ndarray:
        a = self._cube_qadr[i]
        return self.data.qpos[a:a + 3].copy()

    def gt_cube_speed(self, i: int) -> float:
        return float(np.linalg.norm(self.data.qvel[self._cube_vadr[i]:self._cube_vadr[i] + 3]))

    def gt_hand_T(self) -> np.ndarray:
        b = self.data.body("hand")
        T = np.eye(4)
        T[:3, :3] = b.xmat.reshape(3, 3)
        T[:3, 3] = b.xpos
        return T

    def gt_tcp_pos(self) -> np.ndarray:
        return (self.gt_hand_T() @ T_HAND_TCP)[:3, 3]

    def gt_camera_T(self) -> np.ndarray:
        """T_base_cam in the OpenCV convention (x right, y down, z forward)."""
        T = np.eye(4)
        T[:3, :3] = self.data.cam_xmat[self._cam_id].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        T[:3, 3] = self.data.cam_xpos[self._cam_id]
        return T

    def gt_robot_contacts(self) -> list:
        """Names of non-robot geoms touched by robot geoms right now (obstacles, bin, cubes, table)."""
        m, d = self.model, self.data
        hits = []
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            r1, r2 = b1 in self._robot_body_ids, b2 in self._robot_body_ids
            if r1 != r2:
                hits.append(m.geom(c.geom2 if r1 else c.geom1).name)
        return hits
