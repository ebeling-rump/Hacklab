"""Run the agent on one scenario/episode."""
import argparse
import ctypes
import os
import sys
import threading
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def select_gl_backend(viewer: bool) -> None:
    """Pick MUJOCO_GL. Must run before anything imports mujoco."""
    if "MUJOCO_GL" in os.environ:                       # an explicit user choice always wins
        backend = os.environ["MUJOCO_GL"]
    elif sys.platform in ("darwin", "win32"):           # osmesa/egl are not available here
        backend = "glfw"
    elif viewer and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        backend = "glfw"                                # Linux with a display, live viewer
    else:
        backend = "osmesa"                              # Linux headless (software rendering)
    os.environ["MUJOCO_GL"] = backend
    # PyOpenGL only understands 'egl' / 'osmesa' here; 'glfw' is not a valid PYOPENGL_PLATFORM.
    if backend in ("egl", "osmesa"):
        os.environ.setdefault("PYOPENGL_PLATFORM", backend)

select_gl_backend("--viewer" in sys.argv)


class WallClockTimeout(TimeoutError):
    def __init__(self, *args):
        super().__init__(*(args or ("wall-clock timeout",)))


def start_wall_clock_guard(seconds: float, grace: float = 15.0):
    """
    After `seconds`, raises WallClockTimeout in the thread that called this function. Like a
    Python-level signal handler, it is delivered the next time that thread executes Python
    bytecode, so a long call inside native code (e.g. mj_step) finishes first. If the thread has not
    exited `grace` seconds later, the process is killed outright. Returns a cancel() function.
    seconds <= 0 disables the guard.
    """
    if seconds <= 0:
        return lambda: None
    target_id = threading.get_ident()   # the thread running the episode (not necessarily the main thread)
    timers = []

    def hard_kill():
        sys.stderr.write(f"wall-clock timeout: episode did not stop within {grace:.0f}s, killing process\n")
        sys.stderr.flush()
        os._exit(124)

    def fire():
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(target_id), ctypes.py_object(WallClockTimeout))
        backstop = threading.Timer(grace, hard_kill)
        backstop.daemon = True
        timers.append(backstop)
        backstop.start()

    first = threading.Timer(seconds, fire)
    first.daemon = True
    timers.append(first)
    first.start()

    def cancel():
        for t in timers:
            t.cancel()

    return cancel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--random-seed", type=int)
    ap.add_argument("--cubes", type=int, default=3)
    ap.add_argument("--obstacles", type=int, default=0)
    ap.add_argument("--randomize-bin", action="store_true",
                    help="place the bin at a random pose (either side of the robot, random yaw), keeping all clearances")
    ap.add_argument("--bin-seed", type=int, default=None, help="seed of the random bin pose (default: the scene seed)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--hide-bin", action="store_true", help="hide the bin pose from the agent: it must localise the bin")
    g.add_argument("--show-bin", action="store_true", help="give the bin pose to the agent (overrides `known: false`)")
    ap.add_argument("--wall-timeout", type=int, default=600)
    ap.add_argument("--viewer", action="store_true", help="open the live MuJoCo viewer (needs a display)")
    ap.add_argument("--speed", type=float, default=1.0, help="viewer speed: sim seconds per wall second, 0 = unpaced")
    ap.add_argument("--camera", choices=["free", "wrist"], default="free", help="viewer camera")
    ap.add_argument("--contacts", action="store_true", help="draw contact points/forces in the viewer")
    a = ap.parse_args()
    if a.viewer:
        os.environ.update(PICKPLACE_VIEWER="1", PICKPLACE_VIEWER_SPEED=str(a.speed),
                          PICKPLACE_VIEWER_CAMERA=a.camera, PICKPLACE_VIEWER_CONTACTS="1" if a.contacts else "0")
        a.wall_timeout = 0 if a.speed != 0 else a.wall_timeout   # real-time pacing must not trip the wall-clock guard
        if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            print("Warning: --viewer requested but no DISPLAY/WAYLAND_DISPLAY found; the viewer will likely fail",
                  file=sys.stderr)
    sys.path.insert(0, str(ROOT))
    from pickplace.config import load_config, random_scene, randomize_bin, validate
    from pickplace.evaluator import run_episode
    from student.agent import PickPlaceAgent

    if a.config:
        cfg = load_config(a.config)
        if a.randomize_bin:
            cfg = randomize_bin(cfg, seed=a.bin_seed, hide=True if a.hide_bin else (False if a.show_bin else None))
        elif a.hide_bin or a.show_bin:
            cfg.bin.known = a.show_bin
    else:
        cfg = random_scene(a.random_seed or 0, a.cubes, n_obstacles=a.obstacles, randomize_bin_pose=a.randomize_bin, bin_known=not a.hide_bin)
        if a.show_bin:
            cfg.bin.known = True

    validate(cfg)
    print(f"scene {cfg.name} #{cfg.digest()}: bin at ({cfg.bin.xy[0]:.3f}, {cfg.bin.xy[1]:.3f}) yaw {cfg.bin.yaw:.2f} rad, "
          f"is {'known to' if cfg.bin.known else 'hidden from'} the agent")

    cancel_guard = start_wall_clock_guard(a.wall_timeout)
    res = None
    try:
        res, _ = run_episode(cfg, PickPlaceAgent)
        print(res.summary())
        for c in res.per_cube:
            print("   ", c)
    finally:
        cancel_guard()


if __name__ == "__main__":
    main()
