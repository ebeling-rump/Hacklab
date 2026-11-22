"""Foot-slip and fall dataset for the Unitree A1 on varied floors, slopes and pushes.

Question: can a model predict an upcoming foot slip or fall from the sensors a
real A1 has (IMU, joint encoders, motor torques, foot contact force), without
being told the floor friction or slope?

Every episode randomises
  * floor friction  mu ~ log-uniform [0.1, 1.0]   (ice ... rubber)
  * slope           0-15 deg, random direction    (tilted gravity = tilted floor)
  * walking command new random (forward, turn) every second, smoothed
  * pushes          random horizontal shoves on the trunk (0-60 N for 0.1 s)

and records, every SAMPLE_EVERY seconds after WARMUP:
  features  (what the robot itself can measure; see feature_names())
  history   short-history features incl. a leg-kinematics slip detector (see History)
  slip_now  any stance foot sliding in the last 0.1 s   (ground truth, privileged)
  true_slip_2s fraction of the last 2 s with a slip     (ground truth, privileged)
  slip_next any stance foot sliding in the next 0.2 s   (label)
  fall_next the robot falls within the next 1.0 s       (label)
  meta      episode, t, mu, slope_deg

Ground truth comes from MuJoCo: a foot "slips" when it presses on the floor
with > CONTACT_FORCE N while its horizontal speed is > SLIP_SPEED, for at least
SUSTAIN consecutive checks. With this definition slips never happen on grippy
floors (mu >= 0.5, flat or 12 deg) and in 6-8% of checks on slippery floors or
slopes, measured with a steady 0.6 forward command. The robot
never sees foot speeds, friction or slope - only the label does.

    python -m a1slip.sim.slip_dataset              # blind gait     -> data/slip.npz
    python -m a1slip.sim.slip_dataset --balanced   # balance ctrl   -> data/slip_balanced.npz
"""

import time
from dataclasses import dataclass

import mujoco
import numpy as np

from a1slip.paths import ROOT
from a1slip.sim.robot import MAX_TURN, SMOOTHING, ensure_model, gait

FEET = ("FR", "FL", "RR", "RL")
N_EPISODES = 400
EPISODE = 8.0           # s of simulation per episode
WARMUP = 1.0            # s before the first sample
SAMPLE_EVERY = 0.1      # s between samples
CHECK_EVERY = 5         # physics steps between slip checks (100 Hz at dt=0.002)
SLIP_HORIZON = 0.2      # s: label window for slip_next
FALL_HORIZON = 1.0      # s: label window for fall_next
SLIP_SPEED = 0.15       # m/s horizontal foot speed while loaded
CONTACT_FORCE = 30.0    # N: a foot carrying real load (stance share of the 12 kg robot is ~60 N).
                        # Lower thresholds count touchdown/lift-off scuffing, which happens
                        # on every floor and is not slipping.
SUSTAIN = 2             # consecutive slip checks (20 ms) for it to count
FALL_HEIGHT = 0.15      # m trunk height
FALL_TILT = 0.5         # fall if body-z component of gravity > -0.5 (tilt > 60 deg)
MU_RANGE = (0.1, 1.0)
SLOPE_MAX = 15.0        # deg
PUSH_RATE = 0.5         # pushes per second
PUSH_FORCE = 60.0       # N, max
PUSH_TIME = 0.1         # s
OUT = ROOT / "data" / "slip.npz"


def feature_names() -> list[str]:
    j = [f"{leg}_{part}" for leg in FEET for part in ("hip", "thigh", "calf")]
    return ([f"grav_{a}" for a in "xyz"] + [f"gyro_{a}" for a in "xyz"]
            + [f"acc_{a}" for a in "xyz"] + [f"q_{n}" for n in j] + [f"dq_{n}" for n in j]
            + [f"tau_{n}" for n in j] + [f"force_{f}" for f in FEET] + ["cmd_forward", "cmd_turn"])


def build_model() -> mujoco.MjModel:
    """A1 on a flat floor with an IMU (gyro + accelerometer) on the trunk."""
    spec = mujoco.MjSpec.from_file(str(ensure_model()))
    spec.body("trunk").add_site(name="imu", pos=[0, 0, 0])
    spec.add_sensor(name="gyro", type=mujoco.mjtSensor.mjSENS_GYRO,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    spec.add_sensor(name="acc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    return spec.compile()


class A1Sensors:
    """Feature extraction + ground-truth slip/fall checks for one model."""

    def __init__(self, model):
        self.m = model
        self.trunk = model.body("trunk").id
        self.floor = model.geom("floor").id
        # Foot geoms: the sphere in each calf body.
        self.feet = [next(g for g in range(model.ngeom)
                          if model.geom_bodyid[g] == model.body(f"{leg}_calf").id
                          and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE)
                     for leg in FEET]
        self._foot_index = {g: i for i, g in enumerate(self.feet)}
        self._gyro = model.sensor_adr[model.sensor("gyro").id]
        self._acc = model.sensor_adr[model.sensor("acc").id]
        self._f6 = np.zeros(6)
        self._v6 = np.zeros(6)

    def set_floor(self, mu: float, slope_deg: float, azimuth: float):
        """Foot friction (feet have contact priority) and a tilted gravity vector."""
        self.m.geom_friction[self.feet, 0] = mu
        s = np.radians(slope_deg)
        self.m.opt.gravity[:] = 9.81 * np.array([np.sin(s) * np.cos(azimuth),
                                                 np.sin(s) * np.sin(azimuth), -np.cos(s)])

    def gravity_in_body(self, data) -> np.ndarray:
        """Unit gravity direction in the trunk frame (what an IMU's tilt estimate gives)."""
        g = self.m.opt.gravity / np.linalg.norm(self.m.opt.gravity)
        return data.xmat[self.trunk].reshape(3, 3).T @ g

    def foot_forces(self, data) -> np.ndarray:
        f = np.zeros(4)
        for k, c in enumerate(data.contact[:data.ncon]):
            i = self._foot_index.get(c.geom1, self._foot_index.get(c.geom2))
            if i is not None:
                mujoco.mj_contactForce(self.m, data, k, self._f6)
                f[i] += self._f6[0]
        return f

    def sliding_feet(self, data, forces=None) -> np.ndarray:
        """Per foot: loaded and moving horizontally right now (one check). Ground truth."""
        forces = self.foot_forces(data) if forces is None else forces
        out = np.zeros(4, bool)
        for i, g in enumerate(self.feet):
            if forces[i] > CONTACT_FORCE:
                mujoco.mj_objectVelocity(self.m, data, mujoco.mjtObj.mjOBJ_GEOM, g, self._v6, 0)
                out[i] = np.hypot(self._v6[3], self._v6[4]) > SLIP_SPEED
        return out

    def fallen(self, data) -> bool:
        return data.xpos[self.trunk, 2] < FALL_HEIGHT or self.gravity_in_body(data)[2] > -FALL_TILT

    def features(self, data, cmd) -> np.ndarray:
        sd = data.sensordata
        return np.concatenate([self.gravity_in_body(data), sd[self._gyro:self._gyro + 3],
                               sd[self._acc:self._acc + 3], data.qpos[7:19], data.qvel[6:18],
                               data.actuator_force[:12], self.foot_forces(data), cmd]
                              ).astype(np.float32)


HISTORY_NAMES = ["leg_imu_gap_0.1s", "leg_imu_gap_0.5s", "leg_imu_gap_2s",
                 "feet_disagree_0.5s", "dq_var_0.5s", "tau_thigh_0.5s", "force_var_0.5s",
                 "gyro_rms_0.5s", "grav_x_2s", "grav_y_2s", "grav_z_2s"]


class History:
    """Short-history features the robot can compute itself, updated at 100 Hz.

    Slip detector (legs vs IMU, as in legged-robot state estimators): each loaded
    foot is assumed to stand still, so it implies a trunk velocity
        u_i = -(d/dt p_i + omega x p_i)        (trunk frame)
    with p_i from the joint angles (forward kinematics) and omega from the gyro.
    Their mean is the leg-odometry velocity. Separately, the accelerometer is
    integrated (complementary filter, TAU) into an inertial velocity estimate.
    When the feet slide, leg odometry and the IMU disagree: "leg_imu_gap"
    (m/s) is that gap, averaged over 0.1 s, 0.5 s and 2 s - the 2 s value is
    the robot's own estimate of "how slippery has it been lately".
    "feet_disagree" is the spread between the loaded feet's u_i (catches one
    foot sliding while the other holds; misses both sliding together).

    Also: variability of joint speeds, thigh effort, foot-force variability and
    gyro activity over 0.5 s, and the mean gravity direction over 2 s (a slope
    estimate). `true_slip_2s` (ground truth) is kept separately as a privileged
    reference, never as a robot feature.
    """

    LONG, SHORT, NOW = 200, 50, 10          # checks (0.01 s each)
    TAU = 0.3                               # s: legs-vs-IMU complementary filter

    def __init__(self, sensors, check_dt):
        self.s, self.dt = sensors, check_dt
        self.prev = None
        self.buf = []
        self._thigh = [1, 4, 7, 10]

    def update(self, data, forces, slipped):
        s = self.s
        R = data.xmat[s.trunk].reshape(3, 3)
        p = (data.geom_xpos[s.feet] - data.xpos[s.trunk]) @ R     # feet in trunk frame
        w = data.sensordata[s._gyro:s._gyro + 3]
        dis = res = 0.0
        st = np.flatnonzero(forces > CONTACT_FORCE)
        # Inertial prediction of the trunk velocity (world frame), from the accelerometer.
        acc = data.sensordata[s._acc:s._acc + 3]
        a_world = R @ acc + s.m.opt.gravity
        self.v_hat = getattr(self, "v_hat", np.zeros(3)) + a_world * self.dt
        if self.prev is not None and len(st) >= 1:
            u = -((p - self.prev) / self.dt + np.cross(w, p))           # trunk frame
            v_leg = R @ u[st].mean(0)                                    # leg odometry, world
            res = float(np.linalg.norm((v_leg - self.v_hat)[:2]))        # legs vs IMU
            self.v_hat += (v_leg - self.v_hat) * min(1.0, self.dt / self.TAU)
            if len(st) >= 2:
                uh = u[st, :2]
                dis = max(np.linalg.norm(uh[i] - uh[j])
                          for i in range(len(st)) for j in range(i + 1, len(st)))
        self.prev = p
        self.buf.append((res, dis, np.abs(data.qvel[6:18]).mean(),
                         np.abs(data.actuator_force[self._thigh]).mean(), forces.sum(),
                         np.linalg.norm(w), *s.gravity_in_body(data), float(slipped)))
        if len(self.buf) > self.LONG:
            self.buf.pop(0)

    def features(self):
        if not self.buf:
            return np.zeros(len(HISTORY_NAMES), np.float32), 0.0
        b = np.array(self.buf)
        sh = b[-self.SHORT:]
        f = [b[-self.NOW:, 0].mean(), sh[:, 0].mean(), b[:, 0].mean(), sh[:, 1].mean(),
             sh[:, 2].std(), sh[:, 3].mean(), sh[:, 4].std(), np.sqrt((sh[:, 5] ** 2).mean()),
             *b[:, 6:9].mean(0)]
        return np.array(f, np.float32), float(b[:, 9].mean())


@dataclass(frozen=True)
class Reaction:
    """How the gait is modified (see a1slip.sim.robot.gait). NORMAL = the unchanged gait."""
    scale: float = 1.0       # forward command multiplier
    turn_scale: float = 1.0  # turn command multiplier
    stance: float = 0.0      # wider feet (rad of hip abduction)
    crouch: float = 0.0      # lower body (rad)
    lift: float = 1.0        # 0 = stop stepping, all feet stay down


NORMAL = Reaction()


def run_episode(model, data, sensors, rng, mu, slope, azimuth, policy=None, on_step=None,
                balancer=None, start=None):
    """Simulate one episode. policy(features, history, cmd_target) -> Reaction is
    called at every sample (0.1 s) and may change the gait; None = never.
    Posture changes are blended in with the same 0.15 s smoothing as commands.
    on_step(data, reaction, sliding_feet) is called after every physics step
    (for a viewer); returning False stops the episode. It does not change the
    simulation, so a replay with a viewer reproduces a headless run exactly.
    balancer(t, data, cmd, posture) -> joint targets replaces the blind gait
    (a1slip.sim.balance.Balancer); None = the original open-loop trot.
    start(model, data) places the robot after the reset (a1_terrain); None = origin.

    Returns dict with per-sample features/labels and the fall time (or None).
    """
    sensors.set_floor(mu, slope, azimuth)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    if start is not None:
        start(model, data)
    dt = model.opt.timestep
    alpha = dt / SMOOTHING
    n_steps = int(round(EPISODE / dt))
    sample_every = int(round(SAMPLE_EVERY / dt))
    trunk = sensors.trunk

    cmd, target, reaction = np.zeros(2), np.zeros(2), NORMAL
    posture = np.array([0.0, 0.0, 1.0])   # stance, crouch, lift (blended)
    reacting_steps = 0
    streak = np.zeros(4, int)   # consecutive sliding checks per foot
    slide_now = np.zeros(4, bool)
    history = History(sensors, CHECK_EVERY * dt)
    hist_feats, past_slip = [], []
    push_until, push = -1.0, np.zeros(3)
    slip_t, sample_t, feats, fall_t = [], [], [], None
    for k in range(n_steps):
        t = data.time
        if k % int(round(1.0 / dt)) == 0:              # new command every second
            target = np.array([rng.uniform(0, 1), rng.uniform(-MAX_TURN, MAX_TURN)])
        if t > push_until and rng.random() < PUSH_RATE * dt:
            a = rng.uniform(0, 2 * np.pi)
            push = rng.uniform(0, PUSH_FORCE) * np.array([np.cos(a), np.sin(a), 0])
            push_until = t + PUSH_TIME
        data.xfrc_applied[trunk, :3] = push if t < push_until else 0
        if k >= WARMUP / dt and k % sample_every == 0:
            f = sensors.features(data, cmd)
            h, ps = history.features()
            if policy is not None:
                reaction = policy(f, h, target)
            feats.append(f)
            hist_feats.append(h)
            past_slip.append(ps)
            sample_t.append(t)
        cmd += alpha * (target * [reaction.scale, reaction.turn_scale] - cmd)
        posture += alpha * (np.array([reaction.stance, reaction.crouch, reaction.lift]) - posture)
        reacting_steps += reaction != NORMAL
        data.ctrl[:] = (gait(t, *cmd, *posture) if balancer is None
                        else balancer(t, data, cmd, posture))
        mujoco.mj_step(model, data)
        fell_now = False
        if k % CHECK_EVERY == 0:
            forces = sensors.foot_forces(data)
            sliding = sensors.sliding_feet(data, forces)
            streak = np.where(sliding, streak + 1, 0)
            slide_now = streak >= SUSTAIN
            slipped = slide_now.any()
            if slipped:
                slip_t.append(data.time)
            history.update(data, forces, slipped)
            if sensors.fallen(data):
                fall_t = data.time
                fell_now = True
        if on_step is not None and on_step(data, reaction, slide_now) is False:
            break
        if fell_now:
            break
    return dict(features=np.array(feats).reshape(-1, len(feature_names())),
                history=np.array(hist_feats).reshape(-1, len(HISTORY_NAMES)),
                true_slip_2s=np.array(past_slip),
                t=np.array(sample_t), slip_t=np.array(slip_t), fall_t=fall_t,
                reacting_time=reacting_steps * dt)


def labels(ep):
    """slip_now / slip_next / fall_next for every sample of an episode."""
    t, s = ep["t"], ep["slip_t"]
    slip_now = np.array([((s > ti - 0.1) & (s <= ti)).any() for ti in t])
    slip_next = np.array([((s > ti) & (s <= ti + SLIP_HORIZON)).any() for ti in t])
    fall_next = (np.zeros(len(t), bool) if ep["fall_t"] is None
                 else (ep["fall_t"] - t) <= FALL_HORIZON)
    return slip_now, slip_next, fall_next


def random_floor(rng):
    mu = float(np.exp(rng.uniform(*np.log(MU_RANGE))))
    return mu, float(rng.uniform(0, SLOPE_MAX)), float(rng.uniform(0, 2 * np.pi))


def main(n_episodes=N_EPISODES, seed=0, out=OUT, balanced=False):
    model = build_model()
    data = mujoco.MjData(model)
    sensors = A1Sensors(model)
    rng = np.random.default_rng(seed)
    balancer = None
    if balanced:
        from a1slip.sim.balance import TUNED, Balancer
        balancer = Balancer(sensors, TUNED)
    cols = {k: [] for k in ("features", "history", "true_slip_2s", "slip_now", "slip_next",
                            "fall_next", "episode", "t", "mu", "slope_deg", "episode_fell")}
    t0, falls = time.time(), 0
    for e in range(n_episodes):
        mu, slope, az = random_floor(rng)
        ep = run_episode(model, data, sensors, rng, mu, slope, az, balancer=balancer)
        now, nxt, fall = labels(ep)
        n = len(ep["t"])
        falls += ep["fall_t"] is not None
        for k, v in (("features", ep["features"]), ("history", ep["history"]),
                     ("true_slip_2s", ep["true_slip_2s"]), ("slip_now", now), ("slip_next", nxt),
                     ("fall_next", fall), ("episode", np.full(n, e)), ("t", ep["t"]),
                     ("mu", np.full(n, mu)), ("slope_deg", np.full(n, slope)),
                     ("episode_fell", np.full(n, ep["fall_t"] is not None))):
            cols[k].append(v)
        if (e + 1) % 50 == 0:
            print(f"{e + 1}/{n_episodes} episodes, {falls} falls ({time.time() - t0:.0f}s)",
                  flush=True)
    ds = {k: np.concatenate(v) for k, v in cols.items()}
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, **ds, feature_names=np.array(feature_names()),
                        history_names=np.array(HISTORY_NAMES))
    print(f"{len(ds['t'])} samples -> {out}\n"
          f"  slip_next {ds['slip_next'].mean():.1%}, fall_next {ds['fall_next'].mean():.1%}, "
          f"episodes with a fall {falls}/{n_episodes}")


BALANCED_OUT = ROOT / "data" / "slip_balanced.npz"

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--balanced", action="store_true",
                    help="walk with the feedback balance controller (a1slip.sim.balance.TUNED) "
                         "-> data/slip_balanced.npz")
    a = ap.parse_args()
    main(out=BALANCED_OUT if a.balanced else OUT, balanced=a.balanced)
