"""Optional live MuJoCo viewer (passive viewer in its own window).

Enable without touching code:   PICKPLACE_VIEWER=1  (CLI: scripts/run_episode.py --viewer, pytest --viewer)
  PICKPLACE_VIEWER_SPEED   simulated seconds per wall-clock second (default 1 = real time, 0 = as fast as possible)
  PICKPLACE_VIEWER_CAMERA  `free` (orbit camera, default) or `wrist` (what the robot camera sees)
  PICKPLACE_VIEWER_CONTACTS 1 = draw contact points and forces
  PICKPLACE_VIEWER_HOLD    keep the window open this many seconds after an episode ends (default 0)
(Redraws are limited to 60 per wall-clock second; the simulation itself is never skipped.)

Requires a display and MUJOCO_GL=glfw (the default on a desktop Linux session; osmesa/egl are refused with a
warning because they crash the viewer). If the window cannot be opened (no display, SSH without X forwarding, ...) a warning is printed and the run
continues headless, so enabling the viewer can never break an evaluation. Closing the window also just
continues headless. On macOS MuJoCo's passive viewer needs the `mjpython` launcher (see the MuJoCo docs).
The viewer is read-only for the simulation: it never changes physics, sensors or grading.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass

import mujoco


def _x_server_reachable(display: str) -> bool:
    """For a local X display (":99", ":0.0") check that its socket accepts connections. A stale DISPLAY (ended SSH session,
    dead Xvfb) would otherwise make MuJoCo/GLFW abort the whole process. Remote displays ("host:10.0") are assumed reachable."""
    if not display.startswith(":"):
        return True
    import socket
    s = socket.socket(socket.AF_UNIX)
    try:
        s.settimeout(1.0)
        s.connect(f"/tmp/.X11-unix/X{display[1:].split('.')[0]}")
        return True
    except OSError:
        return False
    finally:
        s.close()


def has_display() -> bool:
    """True if a window can plausibly be opened: always on macOS/Windows, on Linux a reachable X display or a Wayland session."""
    if sys.platform in ("darwin", "win32"):
        return True
    if os.environ.get("WAYLAND_DISPLAY"):
        return True
    d = os.environ.get("DISPLAY", "")
    return bool(d) and _x_server_reachable(d)


@dataclass
class ViewerOptions:
    speed: float = 1.0        # simulated s per wall s; 0 = unpaced
    max_fps: float = 60.0     # redraw rate limit (0 = redraw after every control step)
    hold: float = 0.0         # keep the window open this many wall seconds after an episode (manual inspection)
    camera: str = "free"      # "free" | "wrist"
    contacts: bool = False

    def __post_init__(self):
        if self.camera not in ("free", "wrist"):
            raise ValueError("viewer camera must be 'free' or 'wrist'")
        if self.speed < 0:
            raise ValueError("viewer speed must be >= 0")

    @classmethod
    def from_env(cls) -> "ViewerOptions | None":
        if os.environ.get("PICKPLACE_VIEWER", "").strip().lower() in ("", "0", "false", "no", "off"):
            return None
        return cls(speed=float(os.environ.get("PICKPLACE_VIEWER_SPEED", "1")),
                   camera=os.environ.get("PICKPLACE_VIEWER_CAMERA", "free").strip().lower(),
                   contacts=os.environ.get("PICKPLACE_VIEWER_CONTACTS", "0").strip().lower() in ("1", "true", "yes"),
                   hold=float(os.environ.get("PICKPLACE_VIEWER_HOLD", "0")))


class LiveViewer:
    """Thin wrapper around mujoco.viewer.launch_passive: locking, real-time pacing, graceful degradation."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, opts: ViewerOptions, wrist_cam_id: int):
        import mujoco.viewer  # imported lazily: headless runs never touch GLFW
        self.opts = opts
        self._h = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
        cam = self._h.cam
        if opts.camera == "wrist":
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = wrist_cam_id
        else:
            cam.lookat[:] = [0.30, 0.20, 0.25]
            cam.distance, cam.azimuth, cam.elevation = 1.7, 145.0, -25.0
        if opts.contacts:
            self._h.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            self._h.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        self._wall0 = time.perf_counter()
        self._sim0 = 0.0
        self._closed_notice = False
        self._last_draw = 0.0

    @property
    def alive(self) -> bool:
        return self._h is not None and self._h.is_running()

    def lock(self):
        """Hold while mutating MjData from the simulation thread (the viewer renders in another thread)."""
        return self._h.lock() if self.alive else nullcontext()

    def reset_clock(self, sim_time: float) -> None:
        self._wall0, self._sim0 = time.perf_counter(), sim_time

    def refresh(self) -> None:
        """Redraw without pacing (used after reset)."""
        if self.alive:
            self._h.sync()

    def sync(self, sim_time: float) -> None:
        if not self.alive:
            if not self._closed_notice:
                self._closed_notice = True
                print("[LiveViewer] viewer window closed; continuing headless")
            return
        now = time.perf_counter()
        if self.opts.max_fps <= 0 or now - self._last_draw >= 1.0 / self.opts.max_fps:
            self._h.sync()
            self._last_draw = now
        if self.opts.speed > 0:
            target = self._wall0 + (sim_time - self._sim0) / self.opts.speed
            now = time.perf_counter()
            if target > now:
                time.sleep(target - now)
            elif now - target > 0.25:   # we were busy (planning, rendering): do not "catch up" faster than real time
                self.reset_clock(sim_time)

    def hold(self) -> None:
        """Keep drawing for `opts.hold` wall seconds (or until the window is closed) so the final state can be inspected."""
        t_end = time.perf_counter() + self.opts.hold
        while self.alive and time.perf_counter() < t_end:
            self._h.sync()
            time.sleep(0.03)

    def close(self) -> None:
        if self._h is not None:
            try:
                self._h.close()
            finally:
                self._h = None


def open_viewer(model, data, opts: ViewerOptions | None, wrist_cam_id: int) -> LiveViewer | None:
    """Return a LiveViewer or None (disabled / unavailable). Never raises for display problems."""
    if opts is None:
        return None
    if not has_display():
        warnings.warn("Live viewer needs a display that is reachable (DISPLAY unset, or no X server behind it); continuing headless", RuntimeWarning)
        return None
    backend = os.environ.get("MUJOCO_GL", "").strip().lower()
    if backend in ("osmesa", "egl"):
        # With these backends MuJoCo's viewer aborts the whole process natively ("gladLoadGL error") - not catchable.
        warnings.warn(f"Live viewer needs MUJOCO_GL=glfw (currently {backend!r}); continuing headless", RuntimeWarning)
        return None
    try:
        return LiveViewer(model, data, opts, wrist_cam_id)
    except Exception as e:  # no DISPLAY, GLFW init failure, mjpython required on macOS, ...
        warnings.warn(f"Live viewer unavailable ({type(e).__name__}: {e}); continuing headless", RuntimeWarning)
        return None
