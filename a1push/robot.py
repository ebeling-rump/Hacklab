"""Unitree A1 (MuJoCo Menagerie) with a simple trot gait, used as a whole-body pusher.

The gait extends the repository's open-loop trot (a1_navigation.py): per-side stride lengths
may be negative, so the robot can walk forward/backward and turn in place. A pose controller
drives the trunk to a target (x, y, yaw); a line-following controller pushes along a direction.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MENAGERIE = ROOT / "mujoco_menagerie"
A1_SCENE = MENAGERIE / "unitree_a1" / "scene.xml"

FREQ = 2.5
STRIDE = 0.25
LIFT = 0.55
STAND = np.array([0.0, 0.9, -1.8])
PHASE = np.array([0.0, np.pi, np.pi, 0.0])     # FR, FL, RR, RL (diagonal pairs)
IS_LEFT = np.array([False, True, False, True])
NOSE = 0.30   # distance from trunk centre to the front bumper contact (m)


def ensure_model() -> Path:
    if not A1_SCENE.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                        "https://github.com/google-deepmind/mujoco_menagerie.git", str(MENAGERIE)], check=True)
        subprocess.run(["git", "sparse-checkout", "set", "unitree_a1"], cwd=MENAGERIE, check=True)
    return A1_SCENE


def gait(t, fwd, turn):
    """Joint targets. fwd in [-1, 1] (+ forward), turn in [-1, 1] (+ = counter-clockwise)."""
    ctrl = np.zeros(12)
    for leg in range(4):
        ph = 2 * np.pi * FREQ * t + PHASE[leg]
        side = -1.0 if IS_LEFT[leg] else 1.0
        stride = STRIDE * np.clip(fwd + side * turn, -1.6, 1.6)
        hip, thigh, knee = STAND
        thigh = thigh - stride * np.sin(ph)
        knee = knee - LIFT * max(0.0, np.cos(ph)) * (1.0 if (abs(fwd) + abs(turn)) > 0.05 else 0.0)
        ctrl[3 * leg:3 * leg + 3] = hip, thigh, knee
    return ctrl


def yaw_of(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class A1:
    """Thin wrapper around model/data for the trunk state and gait commands."""

    def __init__(self, model, data):
        self.m, self.d = model, data
        self.trunk = model.body("trunk").id

    @property
    def pose(self):
        q = self.d.qpos
        return np.array([q[0], q[1], yaw_of(q[3:7])])

    @property
    def height(self):
        return float(self.d.qpos[2])

    def upright(self):
        w, x, y, z = self.d.qpos[3:7]
        return 1 - 2 * (x * x + y * y) > 0.7 and self.height > 0.15

    def recover(self):
        """Stand-up recovery: put the robot back on its feet at its current x, y, yaw."""
        x, y, th = self.pose
        key = self.m.key("home")
        self.d.qpos[7:19] = key.qpos[7:19]
        self.d.qpos[0:3] = [x, y, 0.30]
        self.d.qpos[3:7] = [np.cos(th / 2), 0, 0, np.sin(th / 2)]
        self.d.qvel[0:18] = 0
        self.stand()
        mujoco.mj_forward(self.m, self.d)
        for _ in range(250):
            self.stand()
            mujoco.mj_step(self.m, self.d)

    def command(self, fwd, turn):
        self.d.ctrl[:12] = gait(self.d.time, fwd, turn)

    def stand(self):
        self.d.ctrl[:12] = np.tile(STAND, 4)


def goto_command(pose, target, pos_tol=0.05, final_yaw=None):
    """Simple unicycle controller -> (fwd, turn, done)."""
    dx, dy = target[0] - pose[0], target[1] - pose[1]
    dist = np.hypot(dx, dy)
    if dist > pos_tol:
        head = np.arctan2(dy, dx)
        err = wrap(head - pose[2])
        # walk backwards if the target is behind and close
        if abs(err) > 2.2 and dist < 0.4:
            err_b = wrap(head + np.pi - pose[2])
            return -0.6, float(np.clip(1.6 * err_b, -0.6, 0.6)), False
        fwd = float(np.clip(1.5 * dist, 0.25, 1.0)) * max(0.0, np.cos(err)) ** 2
        return fwd, float(np.clip(1.6 * err, -0.6, 0.6)), False
    if final_yaw is not None:
        err = wrap(final_yaw - pose[2])
        if abs(err) > 0.06:
            return 0.0, float(np.clip(1.5 * err, -0.5, 0.5)) + 0.1 * np.sign(err), False
    return 0.0, 0.0, True
