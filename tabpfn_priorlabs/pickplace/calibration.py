"""T1.1 - Camera / gripper calibration constants.

Frames (all right-handed):
  base   : robot base frame. z is up, table plane z = 0.
  hand   : MuJoCo body "hand" of the Franka Emika Panda. z points from the flange towards the
           fingertips, the fingers open/close along hand-y.
  tcp    : tool centre point between the finger pads: hand frame translated by 0.1034 m along z.
  cam    : computer-vision camera frame: x right, y down, z forward into the scene.

T_HAND_TCP maps TCP coordinates into hand coordinates; T_HAND_CAM maps camera
coordinates into hand coordinates. For homogeneous points: p_hand = T_HAND_CAM @ p_cam.
"""
from __future__ import annotations

import math

import numpy as np

TCP_OFFSET = 0.1034  # m, hand -> tcp along hand z (from Franka's published TCP)

#Gripper Point
#NO ROATION BETWEEN hAND AND TCP ONLY ALONG Z BECAUSE OF THE OFFSET UP AND DOWN 
#
T_HAND_TCP = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, TCP_OFFSET],
    [0.0, 0.0, 0.0, 1.0],
])

GRIPPER_MAX_WIDTH = 0.08  # m, fully open finger gap (2 x 0.04 m slide travel)

# Camera mounted beside the hand (+x), with a 15-degree tilt towards -hand-x.
CAM_POS_IN_HAND = np.array([0.065, 0.0, 0.040])
CAM_TILT = math.radians(15.0)

_s, _c = math.sin(CAM_TILT), math.cos(CAM_TILT)
#Camera rotation relative to the hand
_x_cv = np.array([0.0, 1.0, 0.0])
_y_cv = np.array([-_c, 0.0, -_s])
_z_cv = np.array([-_s, 0.0, _c])
R_HAND_CAM = np.stack([_x_cv, _y_cv, _z_cv], axis=1)  # camera's X, Y and Z axes are oriented relative to the hand frame.


#TRansformation matrix from camera to hand frame (T_HAND_CAM (rotation and translation) + rotation matrix R_HAND_CAM)
T_HAND_CAM = np.block([
    [R_HAND_CAM, CAM_POS_IN_HAND[:, None]],
    [np.zeros((1, 3)), np.ones((1, 1))],
])


def intrinsics(width: int, height: int, fovy_deg: float) -> np.ndarray:
    """Pinhole K (no distortion). fx = fy, principal point at the image centre."""
    fy = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    return np.array([[fy, 0.0, (width - 1) / 2.0], [0.0, fy, (height - 1) / 2.0], [0.0, 0.0, 1.0]])
