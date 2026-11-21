"""TabPFN closed-loop policy for the Panda pick-and-place sim (hackathon project).

Self-contained: imports from `pickplace/` and `student/` but never modifies them.
"""
import os
import sys

# MuJoCo picks its GL backend at import time; mirror scripts/run_episode.py.
if "MUJOCO_GL" not in os.environ and sys.platform in ("darwin", "win32"):
    os.environ["MUJOCO_GL"] = "glfw"
