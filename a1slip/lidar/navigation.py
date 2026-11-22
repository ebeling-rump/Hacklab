"""Unitree A1 quadruped walks through waypoints around obstacles in MuJoCo.

Simplest possible setup: an open-loop trot gait (sine waves on the joints) plus
steering by making the strides on one side longer than on the other.

    pip install mujoco
    python -m a1slip.lidar.navigation             # viewer
    python -m a1slip.lidar.navigation --headless  # prints progress only
"""

import argparse
import subprocess
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from a1slip.paths import ROOT

MENAGERIE_DIR = ROOT / "mujoco_menagerie"
SCENE_XML = MENAGERIE_DIR / "unitree_a1" / "scene.xml"

# Route around the obstacles (x, y in metres). Robot starts at the origin facing +x.
WAYPOINTS = [(1.5, 0.0), (2.5, 1.5), (4.0, 1.5), (5.0, 0.0), (6.5, 0.0)]
OBSTACLES = [(2.5, 0.0), (4.0, 0.3), (4.0, 2.7)]  # box centres

# Gait parameters.
FREQ = 2.5      # steps per second (Hz)
STRIDE = 0.25   # thigh swing amplitude (rad)
LIFT = 0.5      # extra knee bend while a foot is in the air (rad)
STAND = np.array([0.0, 0.9, -1.8])  # hip, thigh, knee of the "home" pose
# Leg order in the actuator list: FR, FL, RR, RL. Diagonal pairs move together.
PHASE = np.array([0.0, np.pi, np.pi, 0.0])
IS_LEFT = np.array([False, True, False, True])


def ensure_model() -> Path:
    if not SCENE_XML.exists():
        print("Cloning MuJoCo Menagerie (one-time download)...")
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/google-deepmind/mujoco_menagerie.git",
                        str(MENAGERIE_DIR)], check=True)
    return SCENE_XML


def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(ensure_model()))
    for x, y in OBSTACLES:
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.3, 0.6, 0.25],
                                pos=[x, y, 0.25], rgba=[0.8, 0.4, 0.2, 1])
    for x, y in WAYPOINTS:  # flat markers on the floor, no collision
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.1, 0.002, 0],
                                pos=[x, y, 0.002], rgba=[0.2, 0.9, 0.3, 1],
                                contype=0, conaffinity=0)
    return spec.compile()


def gait(t: float, turn: float) -> np.ndarray:
    """Joint targets for all 12 actuators. turn in [-1, 1]: + turns left."""
    ctrl = np.zeros(12)
    for leg in range(4):
        phase = 2 * np.pi * FREQ * t + PHASE[leg]
        # Shorter strides on the inside of the turn.
        side = -1 if IS_LEFT[leg] else 1
        stride = STRIDE * np.clip(1 + side * turn, 0, 2)
        hip, thigh, knee = STAND
        thigh -= stride * np.sin(phase)                 # swing leg fore/aft
        knee -= LIFT * max(0.0, np.cos(phase))          # lift foot during swing
        ctrl[3 * leg:3 * leg + 3] = hip, thigh, knee
    return ctrl


def yaw(data) -> float:
    w, x, y, z = data.qpos[3:7]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    model = build_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

    target = 0

    def step():
        nonlocal target
        pos = data.qpos[:2]
        if target < len(WAYPOINTS):
            goal = np.array(WAYPOINTS[target])
            if np.linalg.norm(goal - pos) < 0.3:
                print(f"t={data.time:5.1f}s reached waypoint {target + 1}/{len(WAYPOINTS)}")
                target += 1
                return
            heading = np.arctan2(*(goal - pos)[::-1])
            err = np.angle(np.exp(1j * (heading - yaw(data))))  # wrap to [-pi, pi]
            data.ctrl[:] = gait(data.time, np.clip(1.5 * err, -1, 1))
        else:
            data.ctrl[:] = np.tile(STAND, 4)  # done: stand still
        mujoco.mj_step(model, data)

    if args.headless:
        while data.time < 60 and target < len(WAYPOINTS):
            step()
            if int(data.time * 500) % 2500 == 0:
                print(f"t={data.time:5.1f}s pos=({data.qpos[0]:.2f}, {data.qpos[1]:.2f}) "
                      f"height={data.qpos[2]:.2f} yaw={np.degrees(yaw(data)):.0f}°")
        print("Done" if target == len(WAYPOINTS) else "Did not finish")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.trackbodyid = model.body("trunk").id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.distance = 2.0
        while viewer.is_running():
            start = time.time()
            step()
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))


if __name__ == "__main__":
    main()
