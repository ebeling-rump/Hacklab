"""Repository locations used by every script, so nothing depends on the working directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
VIDEOS = ROOT / "videos"
MENAGERIE = ROOT / "mujoco_menagerie"
