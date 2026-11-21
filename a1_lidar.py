"""Unitree A1 with a 2D lidar that senses boxes and walks around them to a goal.

The robot only knows where the goal is. Obstacles are detected exclusively with
a ring of MuJoCo rangefinder sensors (a simple 360-degree lidar) mounted on top
of the trunk. Navigation is gap-based: among the headings whose corridor (as
wide as the robot) is free of lidar hits, steer towards the one closest to the
goal.

    pip install mujoco
    python a1_lidar.py               # viewer (lidar rays drawn in the scene)
    python a1_lidar.py --headless    # prints progress only
    python a1_lidar.py --seed 3      # different random obstacle layout
"""

import argparse
import subprocess
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

MENAGERIE_DIR = Path(__file__).parent / "mujoco_menagerie"
SCENE_XML = MENAGERIE_DIR / "unitree_a1" / "scene.xml"

GOAL = np.array([8.0, 0.0])

# Lidar.
N_RAYS = 72            # one ray every 5 degrees
LIDAR_HEIGHT = 0.08    # above the trunk centre
MAX_RANGE = 3.0        # readings beyond this (or no hit) count as free space

# Avoidance (gap finding).
CLEARANCE = 1.0        # a heading is "free" if its corridor is clear this far (m)
HALF_WIDTH = 0.3       # half-width of that corridor: robot body + legs + margin
SIDE_DIST = 0.45       # obstacles closer than this get a sideways nudge away
SLOW_DIST = 0.5        # obstacle closer than this in front -> slow down
STICKINESS = 1.0       # preference for keeping the previously chosen heading
MAX_TURN = 0.7         # turn command limit (full turns on the spot are unstable)
SMOOTHING = 0.15       # time constant of the command low-pass filter (s)

# Gait.
FREQ = 2.5
STRIDE = 0.25
LIFT = 0.5
STAND = np.array([0.0, 0.9, -1.8])   # hip, thigh, knee
PHASE = np.array([0.0, np.pi, np.pi, 0.0])   # FR, FL, RR, RL: diagonal trot
IS_LEFT = np.array([False, True, False, True])


def ensure_model() -> Path:
    if not SCENE_XML.exists():
        print("Cloning MuJoCo Menagerie (one-time download)...")
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/google-deepmind/mujoco_menagerie.git",
                        str(MENAGERIE_DIR)], check=True)
    return SCENE_XML


def random_boxes(rng: np.random.Generator, n: int = 7):
    """Boxes scattered between start and goal, keeping both areas clear."""
    boxes = []
    while len(boxes) < n:
        x, y = rng.uniform(1.5, 6.5), rng.uniform(-1.8, 1.8)
        sx, sy = rng.uniform(0.15, 0.4, size=2)
        if all(np.hypot(x - bx, y - by) > 1.2 for bx, by, *_ in boxes):
            boxes.append((x, y, sx, sy))
    return boxes


def build_model(boxes) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(ensure_model()))
    world, trunk = spec.worldbody, spec.body("trunk")

    for x, y, sx, sy in boxes:
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, 0.25],
                       pos=[x, y, 0.25], rgba=[0.8, 0.4, 0.2, 1])
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.2, 0.002, 0],
                   pos=[*GOAL, 0.002], rgba=[0.2, 0.9, 0.3, 1],
                   contype=0, conaffinity=0)

    # Lidar housing (visual only) and one rangefinder per ray. A rangefinder
    # measures along its site's +z axis and ignores the body it is attached to,
    # so sites on the trunk never see the robot itself.
    trunk.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.04, 0.03, 0],
                   pos=[0, 0, LIDAR_HEIGHT - 0.03], rgba=[0.1, 0.1, 0.1, 1],
                   contype=0, conaffinity=0, mass=0)
    q_y90 = np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0])  # z -> x
    for i, a in enumerate(ray_angles()):
        q_z = np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)])
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, q_z, q_y90)             # z -> (cos a, sin a, 0)
        trunk.add_site(name=f"lidar{i}", pos=[0, 0, LIDAR_HEIGHT], quat=quat,
                       size=[0.005] * 3, rgba=[0, 0, 0, 0])
        sensor = spec.add_sensor(name=f"lidar{i}", type=mujoco.mjtSensor.mjSENS_RANGEFINDER,
                                 objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"lidar{i}")
        if hasattr(mujoco, "mjtRayDataField"):  # MuJoCo >= 3.4: choose output = distance
            sensor.intprm[0] = 1 << int(mujoco.mjtRayDataField.mjRAYDATA_DIST)
    return spec.compile()


def ray_angles() -> np.ndarray:
    """Ray directions in the robot frame (0 = straight ahead, + = left)."""
    return np.linspace(-np.pi, np.pi, N_RAYS, endpoint=False)


def read_lidar(data) -> np.ndarray:
    r = data.sensordata[:N_RAYS].copy()
    r[r < 0] = MAX_RANGE                     # -1 means "no hit"
    return np.minimum(r, MAX_RANGE)


def gait(t: float, forward: float, turn: float) -> np.ndarray:
    """Open-loop trot. forward in [0, 1] scales speed, turn in [-1, 1] (+ = left)."""
    ctrl = np.zeros(12)
    for leg in range(4):
        phase = 2 * np.pi * FREQ * t + PHASE[leg]
        side = -1 if IS_LEFT[leg] else 1
        stride = STRIDE * np.clip(forward + side * turn, -1, 2)
        hip, thigh, knee = STAND
        thigh -= stride * np.sin(phase)
        knee -= LIFT * max(0.0, np.cos(phase))
        ctrl[3 * leg:3 * leg + 3] = hip, thigh, knee
    return ctrl


def yaw(data) -> float:
    w, x, y, z = data.qpos[3:7]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Navigator:
    """Gap-based obstacle avoidance (a minimal Vector Field Histogram).

    1. A heading is *free* if no lidar hit lies inside the corridor the robot
       would sweep walking straight that way: CLEARANCE long, 2*HALF_WIDTH wide.
    2. Among free directions pick the one closest to the goal direction, with a
       small bonus for staying close to the previous choice (avoids dithering).
    3. Nothing free -> turn in place towards the most open direction.
    """

    def __init__(self, timestep: float):
        self.prev = None   # previously chosen heading, world frame
        self.cmd = np.zeros(2)                           # smoothed (forward, turn)
        self.alpha = timestep / SMOOTHING                # low-pass filter gain

    def __call__(self, data, ranges: np.ndarray) -> tuple[float, float]:
        th = yaw(data)
        to_goal = GOAL - data.qpos[:2]
        goal_dir = wrap(np.arctan2(to_goal[1], to_goal[0]) - th)   # robot frame
        clearance = min(CLEARANCE, np.linalg.norm(to_goal))

        angles = ray_angles()
        hits = ranges < MAX_RANGE
        pts = ranges[hits, None] * np.stack([np.cos(angles[hits]), np.sin(angles[hits])], 1)
        # For every candidate heading: distance of each hit along / across it.
        along = pts @ np.stack([np.cos(angles), np.sin(angles)])       # (hits, rays)
        across = pts @ np.stack([-np.sin(angles), np.cos(angles)])
        in_corridor = (along > 0) & (along < clearance) & (np.abs(across) < HALF_WIDTH)
        free = ~in_corridor.any(axis=0)

        if free.any():
            cost = np.abs(wrap(angles - goal_dir))
            if self.prev is not None:
                cost += STICKINESS * np.abs(wrap(angles - (self.prev - th)))
            cost[~free] = np.inf
            target = angles[np.argmin(cost)]
        else:
            target = angles[np.argmax(np.convolve(np.r_[ranges[-2:], ranges, ranges[:2]],
                                                  np.ones(5) / 5, "valid"))]
        self.prev = wrap(target + th)

        # Nudge away from anything very close on the sides (protects the legs).
        close = ranges < SIDE_DIST
        nudge = -np.sum(np.sin(angles[close]) * (SIDE_DIST - ranges[close]))
        err = wrap(target + 2.0 * nudge)

        front = np.abs(angles) < np.radians(30)
        forward = 0.4 if ranges[front].min() < SLOW_DIST else 1.0
        forward *= max(0.0, np.cos(err)) ** 2       # turn first if badly aligned
        turn = np.clip(1.5 * err, -MAX_TURN, MAX_TURN)
        # Smooth the commands: abrupt changes mid-stride destabilise the trot.
        # Braking is the exception: slowing down happens immediately.
        self.cmd += self.alpha * (np.array([forward, turn]) - self.cmd)
        self.cmd[0] = min(self.cmd[0], forward)
        return float(self.cmd[0]), float(self.cmd[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = build_model(random_boxes(np.random.default_rng(args.seed)))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    navigate = Navigator(model.opt.timestep)
    state = {"done": False, "min_clear": MAX_RANGE}

    def step():
        ranges = read_lidar(data)
        state["min_clear"] = min(state["min_clear"], ranges.min())
        if np.linalg.norm(GOAL - data.qpos[:2]) < 0.3:
            if not state["done"]:
                print(f"t={data.time:5.1f}s goal reached!")
                state["done"] = True
            data.ctrl[:] = np.tile(STAND, 4)
        else:
            data.ctrl[:] = gait(data.time, *navigate(data, ranges))
        mujoco.mj_step(model, data)

    if args.headless:
        while data.time < 90 and not state["done"]:
            step()
            if int(round(data.time / model.opt.timestep)) % 2500 == 0:
                print(f"t={data.time:5.1f}s pos=({data.qpos[0]:.2f}, {data.qpos[1]:.2f}) "
                      f"closest obstacle={read_lidar(data).min():.2f} m")
        print(f"{'Success' if state['done'] else 'Did not reach the goal'}; "
              f"closest lidar reading on the way: {state['min_clear']:.2f} m")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = model.body("trunk").id
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -35
        while viewer.is_running():
            start = time.time()
            step()
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))


if __name__ == "__main__":
    main()
