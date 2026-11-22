"""Rolling-hill terrain for the A1: real 3D slopes instead of tilted gravity.

Every other experiment makes a "slope" by tilting gravity on a flat floor,
which looks identical on screen. Here the ground itself is a heightfield of
smooth random hills (slopes up to MAX_SLOPE in every direction, changing as the
robot walks), gravity is normal, and the ground is coloured by friction
(green = grippy ... pale blue-white = icy). None of the models are retrained:
this is a new, unseen condition for all of them.

    from a1_terrain import build_terrain_model, TerrainSensors, terrain_episode
"""

import mujoco
import numpy as np

from a1_lidar import ensure_model

HALF = 7.0              # m, terrain spans [-HALF, HALF]^2
N = 141                 # grid points per side (0.1 m)
MAX_SLOPE = 28.0        # deg, steepest point of the terrain (typical slopes ~10-15 deg)
START_RANGE = 4.0       # m, robot starts inside [-4, 4]^2
HOME_Z = 0.27           # trunk height of the standing keyframe above the ground
FALL_HEIGHT = 0.15
FALL_TILT = 0.5
WORLD_ONLY = np.array([1, 0, 0, 0, 0, 0], np.uint8)    # ray hits group 0 (terrain) only


def hills(seed):
    """Smooth random hills: a few long-wavelength waves, scaled so the steepest
    point is MAX_SLOPE degrees."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-HALF, HALF, N)
    X, Y = np.meshgrid(x, x)                     # rows = y, columns = x
    z = np.zeros_like(X)
    for _ in range(5):
        lam = rng.uniform(3.0, 7.0)
        a = rng.uniform(0, np.pi)
        k = 2 * np.pi / lam
        z += rng.uniform(0.5, 1.0) * np.sin(k * (np.cos(a) * X + np.sin(a) * Y) + rng.uniform(0, 2 * np.pi))
    gy, gx = np.gradient(z, x, x)
    z *= np.tan(np.radians(MAX_SLOPE)) / np.hypot(gx, gy).max()
    return z - z.min()


def build_terrain_model(seed=0):
    """A1 + IMU (as a1_slip_dataset.build_model) on a hill heightfield."""
    z = hills(seed)
    spec = mujoco.MjSpec.from_file(str(ensure_model()))
    spec.body("trunk").add_site(name="imu", pos=[0, 0, 0])
    spec.add_sensor(name="gyro", type=mujoco.mjtSensor.mjSENS_GYRO,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    spec.add_sensor(name="acc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    top = float(z.max())
    spec.add_hfield(name="hills", size=[HALF, HALF, top, 0.2], nrow=N, ncol=N,
                    userdata=(z / top).flatten().tolist())
    # Checker texture so the slopes are readable on screen; the material colour
    # (set per episode from the friction) tints it.
    spec.add_texture(name="terrain_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=512, height=512,
                     rgb1=[0.95, 0.95, 0.95], rgb2=[0.78, 0.78, 0.78])
    mat = spec.add_material(name="terrain_mat", texrepeat=[14, 14])
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "terrain_tex"
    spec.worldbody.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="hills",
                            material="terrain_mat")
    model = spec.compile()
    return model


def slope_stats(seed=0):
    z = hills(seed)
    x = np.linspace(-HALF, HALF, N)
    gy, gx = np.gradient(z, x, x)
    s = np.degrees(np.arctan(np.hypot(gx, gy)))
    inner = np.abs(x) <= START_RANGE + 2
    s_in = s[np.ix_(inner, inner)]
    return dict(height_range=float(z.max()), mean_slope=float(s_in.mean()),
                p90_slope=float(np.percentile(s_in, 90)), max_slope=float(s_in.max()))


def ground_height(model, data, x, y):
    gid = np.zeros(1, np.int32)
    dist = mujoco.mj_ray(model, data, np.array([x, y, 10.0]), np.array([0, 0, -1.0]),
                         WORLD_ONLY, 1, -1, gid)
    return 10.0 - dist if dist >= 0 else 0.0


def friction_colour(mu):
    """Pale icy blue-white at mu 0.1 ... grass green at mu 1.0 (log scale)."""
    w = np.clip(np.log(mu / 0.1) / np.log(10.0), 0, 1)
    ice, grass = np.array([0.80, 0.90, 0.97]), np.array([0.35, 0.55, 0.30])
    return np.r_[ice + w * (grass - ice), 1.0]


class TerrainSensors:
    """Wraps a1_slip_dataset.A1Sensors for the hill scene:
    - set_floor() sets the feet's friction and the ground colour, keeps gravity normal
    - fallen() measures the body's height above the ground under it, not above z = 0
    Everything the robot senses (features, history) is computed exactly as before."""

    def __init__(self, model):
        from a1_slip_dataset import A1Sensors
        self._s = A1Sensors(model)
        self.m = model
        self.terrain = model.geom("terrain").id
        self.mat = model.material("terrain_mat").id

    def __getattr__(self, name):          # delegate everything else to A1Sensors
        return getattr(self._s, name)

    def set_floor(self, mu, slope=0.0, azimuth=0.0):
        self.m.geom_friction[self._s.feet, 0] = mu
        self.m.opt.gravity[:] = [0, 0, -9.81]
        self.m.mat_rgba[self.mat] = friction_colour(mu)

    def fallen(self, data):
        p = data.xpos[self._s.trunk]
        h = p[2] - ground_height(self.m, data, p[0], p[1])
        return h < FALL_HEIGHT or self._s.gravity_in_body(data)[2] > -FALL_TILT


def terrain_episode(e, base_seed=7000):
    """(mu, start function, command/push rng) for terrain episode e - identical for
    every strategy, like a1_danger_mode.episode_setup."""
    rng = np.random.default_rng(base_seed + e)
    mu = float(np.exp(rng.uniform(np.log(0.1), 0.0)))
    x0, y0 = rng.uniform(-START_RANGE, START_RANGE, size=2)
    heading = float(rng.uniform(-np.pi, np.pi))

    def start(model, data):
        data.qpos[0:2] = x0, y0
        data.qpos[2] = ground_height(model, data, x0, y0) + HOME_Z + 0.03
        data.qpos[3:7] = [np.cos(heading / 2), 0, 0, np.sin(heading / 2)]
        mujoco.mj_forward(model, data)

    start.xy = np.array([x0, y0])
    return mu, start, np.random.default_rng(base_seed + 10_000 + e)
