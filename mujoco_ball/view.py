"""Open the ball simulation in MuJoCo's interactive viewer (needs a display).

Usage:
    python view.py          # Linux / Windows
    mjpython view.py        # macOS (the viewer must run on the main thread)
"""
from pathlib import Path

import mujoco
import mujoco.viewer

MODEL_PATH = Path(__file__).parent / "ball.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("push").id)

# Blocking viewer: runs the physics loop itself and returns when the window closes.
mujoco.viewer.launch(model, data)
