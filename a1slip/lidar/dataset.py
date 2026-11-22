"""Generate lidar + pose + collision datasets with the Unitree A1 in MuJoCo.

Two environments with different box layouts (and number of boxes). For each one
we run 300 episodes of 20 steps. Every episode starts the robot at a random
collision-free pose and drives it with random walking commands. One step =
STEP_TIME seconds of simulation with the command held constant.

Saved per environment (data/<env>.npz), with E episodes, T=20 steps:
    lidar          (E, T+1, N_RAYS)  range readings in metres (MAX_RANGE = no hit)
    position       (E, T+1, 3)       trunk x, y, z in the world frame
    quaternion     (E, T+1, 4)       trunk orientation (w, x, y, z)
    yaw            (E, T+1)          heading angle (rad)
    action         (E, T, 2)         command (forward in [0, 1], turn in [-1, 1])
    collision      (E, T)            True if the robot touched any box during the step
    collision_box  (E, T)            index of the box touched (-1 = none)
    fallen         (E, T)            True if the trunk dropped below FALL_HEIGHT
    boxes          (n_boxes, 4)      box x, y, half-size x, half-size y
    ray_angles     (N_RAYS,)         lidar ray directions in the robot frame
Index t of lidar/position is the observation *before* action t; index t+1 the
observation after it.

    pip install mujoco
    python -m a1slip.lidar.dataset
"""


import mujoco
import numpy as np

from a1slip.paths import ROOT
from a1slip.sim.robot import (LIDAR_HEIGHT, MAX_TURN, N_RAYS, SMOOTHING,
                      ensure_model, gait, ray_angles, read_lidar, yaw)

N_EPISODES = 300
N_STEPS = 20
STEP_TIME = 0.2        # seconds of simulation per step
BOX_HEIGHT = 0.5
ARENA = 3.0            # robot starts inside [-ARENA, ARENA]^2
START_CLEARANCE = 0.5  # min distance between the start position and any box
FALL_HEIGHT = 0.15
OUT_DIR = ROOT / "data"


def make_boxes(n: int, seed: int) -> np.ndarray:
    """n non-overlapping boxes (x, y, half_x, half_y) inside the arena."""
    rng = np.random.default_rng(seed)
    boxes = []
    while len(boxes) < n:
        x, y = rng.uniform(-ARENA, ARENA, size=2)
        hx, hy = rng.uniform(0.15, 0.5, size=2)
        if all(np.hypot(x - b[0], y - b[1]) > 1.0 + max(hx, hy) for b in boxes):
            boxes.append((x, y, hx, hy))
    return np.array(boxes)


# Two environments: different box positions and number of boxes.
ENVIRONMENTS = {
    "env_A": make_boxes(6, seed=1),
    "env_B": make_boxes(12, seed=2),
}


def build_model(boxes: np.ndarray) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(ensure_model()))
    for i, (x, y, hx, hy) in enumerate(boxes):
        spec.worldbody.add_geom(name=f"box{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=[hx, hy, BOX_HEIGHT / 2], pos=[x, y, BOX_HEIGHT / 2],
                                rgba=[0.8, 0.4, 0.2, 1])
    # Lidar: one horizontal rangefinder per ray, mounted on the trunk.
    trunk = spec.body("trunk")
    q_y90 = np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0])  # site z -> x
    for i, a in enumerate(ray_angles()):
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)]), q_y90)
        trunk.add_site(name=f"lidar{i}", pos=[0, 0, LIDAR_HEIGHT], quat=quat)
        s = spec.add_sensor(name=f"lidar{i}", type=mujoco.mjtSensor.mjSENS_RANGEFINDER,
                            objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"lidar{i}")
        if hasattr(mujoco, "mjtRayDataField"):  # MuJoCo >= 3.4
            s.intprm[0] = 1 << int(mujoco.mjtRayDataField.mjRAYDATA_DIST)
    return spec.compile()


def dist_to_box(p: np.ndarray, box: np.ndarray) -> float:
    """Distance from a 2D point to an axis-aligned box footprint."""
    d = np.abs(p - box[:2]) - box[2:]
    return float(np.linalg.norm(np.maximum(d, 0)))


def reset_robot(model, data, boxes, rng):
    """Standing pose at a random position/heading away from all boxes."""
    while True:
        xy = rng.uniform(-ARENA, ARENA, size=2)
        if all(dist_to_box(xy, b) > START_CLEARANCE for b in boxes):
            break
    heading = rng.uniform(-np.pi, np.pi)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    data.qpos[:2] = xy
    data.qpos[3:7] = [np.cos(heading / 2), 0, 0, np.sin(heading / 2)]
    mujoco.mj_forward(model, data)


def box_contact(model, data, box_ids: dict) -> int:
    """Index of a box the robot touches right now, or -1."""
    for c in data.contact[:data.ncon]:
        for g_box, g_other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            if g_box in box_ids and model.geom_bodyid[g_other] != 0:  # robot geom
                return box_ids[g_box]
    return -1


def observe(data):
    return read_lidar(data), data.qpos[:3].copy(), data.qpos[3:7].copy(), yaw(data)


def generate(name: str, boxes: np.ndarray, seed: int) -> dict:
    model = build_model(boxes)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    box_ids = {model.geom(f"box{i}").id: i for i in range(len(boxes))}
    substeps = int(round(STEP_TIME / model.opt.timestep))
    alpha = model.opt.timestep / SMOOTHING

    E, T = N_EPISODES, N_STEPS
    out = dict(
        lidar=np.zeros((E, T + 1, N_RAYS), np.float32),
        position=np.zeros((E, T + 1, 3), np.float32),
        quaternion=np.zeros((E, T + 1, 4), np.float32),
        yaw=np.zeros((E, T + 1), np.float32),
        action=np.zeros((E, T, 2), np.float32),
        collision=np.zeros((E, T), bool),
        collision_box=np.full((E, T), -1, np.int16),
        fallen=np.zeros((E, T), bool),
    )

    for ep in range(E):
        reset_robot(model, data, boxes, rng)
        cmd = np.zeros(2)  # smoothed command actually sent to the gait
        out["lidar"][ep, 0], out["position"][ep, 0], out["quaternion"][ep, 0], \
            out["yaw"][ep, 0] = observe(data)

        for t in range(T):
            action = np.array([rng.uniform(0.3, 1.0), rng.uniform(-MAX_TURN, MAX_TURN)])
            hit = -1
            for _ in range(substeps):
                cmd += alpha * (action - cmd)
                data.ctrl[:] = gait(data.time, *cmd)
                mujoco.mj_step(model, data)
                if hit < 0:
                    hit = box_contact(model, data, box_ids)

            out["action"][ep, t] = action
            out["collision"][ep, t] = hit >= 0
            out["collision_box"][ep, t] = hit
            out["fallen"][ep, t] = data.qpos[2] < FALL_HEIGHT
            out["lidar"][ep, t + 1], out["position"][ep, t + 1], \
                out["quaternion"][ep, t + 1], out["yaw"][ep, t + 1] = observe(data)

        if (ep + 1) % 50 == 0:
            print(f"{name}: {ep + 1}/{E} episodes")

    out.update(boxes=boxes, ray_angles=ray_angles(), step_time=STEP_TIME)
    return out


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for i, (name, boxes) in enumerate(ENVIRONMENTS.items()):
        ds = generate(name, boxes, seed=100 + i)
        path = OUT_DIR / f"{name}.npz"
        np.savez_compressed(path, **ds)
        col = ds["collision"]
        print(f"{name}: {len(boxes)} boxes -> {path}\n"
              f"  steps with collision: {col.sum()}/{col.size} ({100 * col.mean():.1f}%), "
              f"episodes with >=1 collision: {col.any(1).sum()}/{len(col)}, "
              f"episodes with a fall: {ds['fallen'].any(1).sum()}")


if __name__ == "__main__":
    main()
