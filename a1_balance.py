"""A feedback balance controller for the A1 trot (a stronger baseline than the blind gait).

The original gait (a1_lidar.gait) is open loop: it never reads a sensor. Real
quadrupeds balance actively. This module adds the three classic reflexes of
model-based legged control, on top of the same trot and position actuators:

  lean        keep the legs aligned with gravity: offset all thighs by the
              body's pitch and all hips by its roll relative to gravity, so the
              body's weight stays over the feet on a slope
  level       per-leg length changes (crouch per leg) that rotate the body back
              toward upright, plus damping from the gyro
  foot place  Raibert-style stepping: shift the feet in the direction the body
              is moving faster than commanded (catches pushes); sideways too

Inputs are what a real robot has: IMU tilt (gravity direction in the body
frame) and gyro, plus the body velocity. The velocity is read from the
simulator here - on a real robot it comes from a state estimator (the
legs-vs-IMU filter in a1_slip_dataset.History is a simple one). That is the
one simplification.

Every gain's sign and size is chosen by a1_balance_tune.py on training floors
(slope < 13 deg) only; the test slopes (13-15 deg) are never used for tuning.
"""

from dataclasses import dataclass

import numpy as np

from a1_lidar import FREQ, IS_LEFT, LIFT, PHASE, STANCE_SIGN, STAND, STRIDE

LEG_X = np.array([0.183, 0.183, -0.183, -0.183])    # FR, FL, RR, RL hip positions
LEG_Y = np.array([-0.13, 0.13, -0.13, 0.13])
SIDE = np.where(IS_LEFT, -1.0, 1.0)                  # as in a1_lidar.gait
MAX_THIGH = 0.35     # rad, clip on thigh offsets
MAX_HIP = 0.25       # rad, clip on hip offsets
MAX_LEN = 0.15       # rad of per-leg crouch (keeps the body above the fall height)
V_PER_CMD = 0.45     # m/s of forward speed per unit forward command


@dataclass
class Gains:
    lean_pitch: float = 0.0   # thigh offset per rad of pitch vs gravity
    lean_roll: float = 0.0    # hip offset per rad of roll vs gravity
    level_pitch: float = 0.0  # per-leg crouch per rad of pitch (front vs rear)
    level_roll: float = 0.0   # per-leg crouch per rad of roll (left vs right)
    damp: float = 0.0         # same, per rad/s from the gyro
    vel_x: float = 0.0        # thigh offset per m/s of forward speed error
    vel_y: float = 0.0        # hip offset per m/s of sideways speed


# Chosen by a1_balance_tune.py on training slopes 9-13 deg (40 paired episodes each):
#   blind gait 8/40 falls; lean_roll +0.8 -> 2, level_roll -0.5 -> 2, damp -0.03 -> 4;
#   lean_pitch, level_pitch, vel_x: no help in either sign; vel_y +0.3 -> 6 (alone).
#   The three roll reflexes together: x1 -> 0/40 falls but 1.62 m; x0.5 -> 0/40, 2.32 m.
TUNED = Gains(lean_roll=0.4, level_roll=-0.25, damp=-0.015)


class Balancer:
    """Callable used by run_episode instead of the blind gait:
    balancer(t, data, cmd, posture) -> 12 joint targets."""

    def __init__(self, sensors, gains: Gains):
        self.s, self.g = sensors, gains

    def __call__(self, t, data, cmd, posture):
        s, g = self.s, self.g
        gb = s.gravity_in_body(data)
        pitch = np.arctan2(gb[0], -gb[2])
        roll = np.arctan2(gb[1], -gb[2])
        w = data.sensordata[s._gyro:s._gyro + 3]
        R = data.xmat[s.trunk].reshape(3, 3)
        v = R.T @ data.qvel[:3]                        # body-frame velocity (estimator stand-in)
        forward, turn = cmd
        stance, crouch, lift = posture

        d_thigh = np.clip(g.lean_pitch * pitch + g.vel_x * (v[0] - V_PER_CMD * forward),
                          -MAX_THIGH, MAX_THIGH)
        d_hip = np.clip(g.lean_roll * roll + g.vel_y * v[1], -MAX_HIP, MAX_HIP)
        d_len = np.clip(g.level_pitch * pitch * np.sign(LEG_X) + g.level_roll * roll * np.sign(LEG_Y)
                        + g.damp * (w[1] * np.sign(LEG_X) - w[0] * np.sign(LEG_Y)),
                        -MAX_LEN, MAX_LEN)

        ctrl = np.zeros(12)
        for leg in range(4):
            phase = 2 * np.pi * FREQ * t + PHASE[leg]
            side = SIDE[leg]
            stride = STRIDE * np.clip(forward + side * turn, -1, 2)
            c = crouch + d_len[leg]
            hip = -side * stance * STANCE_SIGN + d_hip
            thigh = STAND[1] + c + d_thigh - stride * np.sin(phase)
            knee = STAND[2] - 2 * c - lift * LIFT * max(0.0, np.cos(phase))
            ctrl[3 * leg:3 * leg + 3] = hip, thigh, knee
        return ctrl
