"""MuJoCo planar pushing simulator.

A cylindrical pusher (point-like finger, radius 1.5 cm) on two velocity-servoed slide joints
pushes a flat rigid object lying on the floor (free joint: the object can translate, rotate
and - in principle - tip; pressure distribution therefore follows the hidden centre of mass).

The hidden parameters of the object (mass, frictions, CoM, inertia, actuator gain) are written
into the MuJoCo model. Models never read them: they only see observed poses (with sensor noise).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from shapely.geometry import Point

from .objects import PushObject

PUSHER_R = 0.015
PUSH_SPEED = 0.15      # m/s commanded
GAP = 0.02             # pusher starts this far (along the push line) before the contact point
OBS_NOISE = (0.002, 0.002, np.deg2rad(0.5))
EXEC_NOISE = 0.002     # contact-point placement noise (m)
SPEED_NOISE = 0.05     # relative pusher speed noise
PARK = np.array([30.0, 30.0])
WHEEL_R = 0.01


def caster_points(shape):
    """Four wheel positions inside the footprint (near its extremes)."""
    from shapely.geometry import Point as P
    inner = shape.polygon.buffer(-0.015)
    if inner.is_empty:
        inner = shape.polygon
    ring = np.asarray(inner.exterior.coords)[:-1]
    pts = []
    for d in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        i = int(np.argmax(ring @ np.array(d)))
        pts.append(ring[i] * 0.9)
    return pts


def yaw_from_quat(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Action:
    s: float      # perimeter parameter of nominal contact point in [0, 1)
    phi: float    # push direction relative to the inward normal (rad)
    d: float      # commanded push travel after nominal contact (m)
    v: float = PUSH_SPEED  # commanded pusher speed (m/s)

    def as_array(self):
        return np.array([self.s, self.phi, self.d, self.v])


@dataclass
class PushResult:
    pose_before: np.ndarray      # observed
    pose_after: np.ndarray       # observed
    true_before: np.ndarray
    true_after: np.ndarray
    valid: bool = True
    wall_contact: bool = False
    touched: bool = True
    traj: list = field(default_factory=list)   # optional (t, obj_pose, pusher_xy)


def action_geometry(obj: PushObject, pose, action: Action, gap=GAP):
    """World-frame pusher start point and unit push direction for an action at `pose`."""
    p_b, n_b, _ = obj.shape.perimeter_point(action.s)
    p_b, n_b = p_b[0], n_b[0]
    c, s = np.cos(action.phi), np.sin(action.phi)
    inward = -n_b
    u_b = np.array([c * inward[0] - s * inward[1], s * inward[0] + c * inward[1]])
    start_b = p_b - u_b * (PUSHER_R + gap)
    x, y, th = pose
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return np.array([x, y]) + R @ start_b, R @ u_b, p_b, u_b


class PushSim:
    def __init__(self, obj: PushObject, walls=(), seed: int = 0, timestep=0.002,
                 noise: bool = True, wall_height=0.12):
        self.obj = obj
        self.walls = list(walls)          # (cx, cy, hx, hy)
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.model = self._build(timestep, wall_height)
        self.data = mujoco.MjData(self.model)
        self.obj_geoms = set(i for i in range(self.model.ngeom)
                             if self.model.geom_bodyid[i] == self.model.body("object").id)
        self.pq = int(self.model.joint("px").qposadr[0])
        self.pv = int(self.model.joint("px").dofadr[0])
        self.wheel_dofs = [int(self.model.joint(f"wj{i}").dofadr[0]) for i in range(4)] if obj.hidden.casters else []
        self.wall_geoms = set(self.model.geom(f"wall{i}").id for i in range(len(self.walls)))
        self.set_pose((0.0, 0.0, 0.0))

    # ----------------------------------------------------------------- building
    def _build(self, timestep, wall_height):
        o, h = self.obj, self.obj.hidden
        spec = mujoco.MjSpec()
        spec.option.timestep = timestep
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10
        wb = spec.worldbody
        spec.visual.global_.offwidth = 1920
        spec.visual.global_.offheight = 1080

        floor = wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1],
                            rgba=[0.85, 0.85, 0.82, 1])
        floor.friction = [h.mu_floor, 0.005, 0.0001]
        floor.priority = 2
        floor.contype, floor.conaffinity = 1, 1
        for i, (cx, cy, hx, hy) in enumerate(self.walls):
            g = wb.add_geom(name=f"wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=[hx, hy, wall_height / 2], pos=[cx, cy, wall_height / 2],
                            rgba=[0.35, 0.38, 0.45, 1])
            g.friction = [0.3, 0.005, 0.0001]
            g.priority = 2
            g.contype, g.conaffinity = 1, 1

        body = wb.add_body(name="object", pos=[0, 0, 0])
        body.add_freejoint(name="obj_free")
        hz = o.height / 2
        lift = 0.0
        if h.casters:
            lift = 2 * WHEEL_R + 0.002
            for i, (wx, wy) in enumerate(caster_points(o.shape)):
                wb_ = body.add_body(name=f"wheel{i}", pos=[wx, wy, WHEEL_R])
                ax = [np.cos(h.caster_axis), np.sin(h.caster_axis), 0]  # wheel axle direction
                jw = wb_.add_joint(name=f"wj{i}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=ax)
                jw.damping = [h.roll_damping, 0.0, 0.0]
                # cylinder axis = local z -> rotate z onto the axle direction
                ang = h.caster_axis
                q1 = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4) * -np.sin(ang), np.sin(np.pi / 4) * np.cos(ang), 0.0])
                wg = wb_.add_geom(name=f"wheel{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[WHEEL_R, 0.005, 0],
                                  quat=q1, rgba=[0.1, 0.1, 0.1, 1])
                wg.mass = 0.02
                wg.contype, wg.conaffinity = 1, 1
        for i, (cx, cy, hx, hy) in enumerate(o.shape.boxes):
            g = body.add_geom(name=f"obj_box{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                              size=[hx, hy, hz], pos=[cx, cy, hz + lift], rgba=[0.95, 0.55, 0.15, 1])
            g.friction = [h.mu_pusher, 0.005, 0.0001]
            g.contype, g.conaffinity = 1, 3
        for i, (cx, cy, r) in enumerate(o.shape.discs):
            g = body.add_geom(name=f"obj_disc{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                              size=[r, hz, 0], pos=[cx, cy, hz + lift], rgba=[0.95, 0.55, 0.15, 1])
            g.friction = [h.mu_pusher, 0.005, 0.0001]
            g.contype, g.conaffinity = 1, 3
        # CoM marker (visual only, for videos)
        m = body.add_geom(name="com_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.006, 0, 0],
                          pos=[h.com[0], h.com[1], o.height + 0.002], rgba=[0.1, 0.1, 0.1, 1])
        m.contype, m.conaffinity = 0, 0
        m.group = 3
        cg = o.shape.uniform_inertia_c()
        com2 = h.com[0] ** 2 + h.com[1] ** 2
        Iz = h.mass * max(cg ** 2 * h.inertia_scale - com2, 0.2 * cg ** 2)
        Ixy = max(h.mass * (cg ** 2 / 2 + o.height ** 2 / 12), 0.51 * Iz)
        body.explicitinertial = True
        body.mass = h.mass
        body.ipos = [h.com[0], h.com[1], hz + lift]
        body.inertia = [Ixy, Ixy, Iz]

        pusher = wb.add_body(name="pusher", pos=[0, 0, 0])
        jx = pusher.add_joint(name="px", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0])
        jy = pusher.add_joint(name="py", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0])
        for j in (jx, jy):
            j.armature = 0.1
            j.damping = [1.0, 0.0, 0.0]
        pg = pusher.add_geom(name="pusher", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                             size=[PUSHER_R, 0.012, 0], pos=[0, 0, 0.027], rgba=[0.15, 0.45, 0.9, 1])
        pg.mass = 1.0
        pg.friction = [h.mu_pusher, 0.005, 0.0001]
        pg.priority = 2
        pg.contype, pg.conaffinity = 2, 0
        for j in ("px", "py"):
            a = spec.add_actuator(name=f"v{j}", target=j, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.set_to_velocity(kv=300)
            a.forcelimited = True
            a.forcerange = [-60, 60]
        return spec.compile()

    # ----------------------------------------------------------------- state
    def set_pose(self, pose):
        x, y, th = pose
        d = self.data
        d.qpos[:] = 0
        d.qvel[:] = 0
        d.qpos[0:3] = [x, y, 0.0005]
        d.qpos[3:7] = [np.cos(th / 2), 0, 0, np.sin(th / 2)]
        d.qpos[self.pq:self.pq + 2] = PARK
        mujoco.mj_forward(self.model, self.data)
        self._settle(0.15)

    def true_pose(self):
        q = self.data.qpos
        return np.array([q[0], q[1], yaw_from_quat(q[3:7])])

    def observe(self):
        p = self.true_pose()
        if self.noise:
            p = p + self.rng.normal(0, OBS_NOISE)
        p[2] = wrap(p[2])
        return p

    def tilt(self):
        w, x, y, z = self.data.qpos[3:7]
        return float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))

    def _settle(self, t_max=0.6):
        self.data.ctrl[:] = 0
        n = int(t_max / self.model.opt.timestep)
        for _ in range(0, n, 25):
            mujoco.mj_step(self.model, self.data, 25)
            if np.linalg.norm(self.data.qvel[:6]) < 1e-3:
                break

    def _wall_contact(self):
        d = self.data
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            if (g1 in self.obj_geoms and g2 in self.wall_geoms) or (g2 in self.obj_geoms and g1 in self.wall_geoms):
                return True
        return False

    def pusher_free(self, start_xy, pose=None, margin=0.002):
        """Can the pusher be placed at start_xy without overlapping the object or walls?"""
        pose = self.true_pose() if pose is None else pose
        c = Point(*start_xy).buffer(PUSHER_R + margin)
        if c.intersects(self.obj.shape.world_polygon(pose)):
            return False
        for cx, cy, hx, hy in self.walls:
            if abs(start_xy[0] - cx) < hx + PUSHER_R and abs(start_xy[1] - cy) < hy + PUSHER_R:
                return False
        return True

    # ----------------------------------------------------------------- pushing
    def push(self, action: Action, observed_pose=None, record: bool = False, record_dt=0.02) -> PushResult:
        """Execute one push defined relative to the (observed) object pose."""
        h = self.obj.hidden
        true_before = self.true_pose()
        obs_before = self.observe() if observed_pose is None else np.asarray(observed_pose)
        start, u, _, _ = action_geometry(self.obj, obs_before, action)
        if self.noise:
            start = start + self.rng.normal(0, EXEC_NOISE, 2)
        res = PushResult(obs_before, obs_before, true_before, true_before)
        if not self.pusher_free(start, true_before, margin=0.0):
            res.valid = False
            return res
        d = self.data
        d.qpos[self.pq:self.pq + 2] = start
        d.qvel[self.pv:self.pv + 2] = 0
        mujoco.mj_forward(self.model, d)
        speed = action.v * h.act_gain * (1 + (self.rng.normal(0, SPEED_NOISE) if self.noise else 0))
        T = (GAP + action.d) / action.v
        dt = self.model.opt.timestep
        chunk = 5
        n = int(np.ceil(T / dt / chunk))
        rec_every = max(1, int(round(record_dt / dt / chunk)))
        wall = False
        moved_before = true_before.copy()
        for k in range(n):
            d.ctrl[:] = speed * u
            mujoco.mj_step(self.model, d, chunk)
            if not wall and self._wall_contact():
                wall = True
            if record and k % rec_every == 0:
                res.traj.append((d.time, self.true_pose(), d.qpos[self.pq:self.pq + 2].copy()))
        # stop the pusher, let the object coast, then retract
        d.ctrl[:] = 0
        for k in range(int(0.15 / dt / chunk)):
            mujoco.mj_step(self.model, d, chunk)
            if record and k % rec_every == 0:
                res.traj.append((d.time, self.true_pose(), d.qpos[self.pq:self.pq + 2].copy()))
        d.qpos[self.pq:self.pq + 2] = PARK
        d.qvel[self.pv:self.pv + 2] = 0
        self._settle(0.5)
        if record:
            res.traj.append((d.time, self.true_pose(), d.qpos[self.pq:self.pq + 2].copy()))
        res.true_after = self.true_pose()
        res.pose_after = self.observe()
        res.wall_contact = wall or self._wall_contact()
        res.touched = np.linalg.norm(res.true_after[:2] - moved_before[:2]) > 1e-4 or \
            abs(wrap(res.true_after[2] - moved_before[2])) > 1e-3
        return res


def body_delta(pose_before, pose_after):
    """Displacement expressed in the body frame of pose_before: (dx_b, dy_b, dtheta)."""
    x0, y0, t0 = pose_before
    dx, dy = pose_after[0] - x0, pose_after[1] - y0
    c, s = np.cos(t0), np.sin(t0)
    return np.array([c * dx + s * dy, -s * dx + c * dy, float(wrap(pose_after[2] - t0))])


def compose(pose, delta):
    """Apply a body-frame displacement (vectorised over leading dims of delta)."""
    pose = np.asarray(pose, dtype=float)
    delta = np.asarray(delta, dtype=float)
    x, y, t = pose[..., 0], pose[..., 1], pose[..., 2]
    c, s = np.cos(t), np.sin(t)
    return np.stack([x + c * delta[..., 0] - s * delta[..., 1],
                     y + s * delta[..., 0] + c * delta[..., 1],
                     wrap(t + delta[..., 2])], axis=-1)
