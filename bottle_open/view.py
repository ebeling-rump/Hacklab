"""Replay one opening episode in the MuJoCo viewer, or save a snapshot strip offscreen.

    python view.py --controller tabpfn --seed 20002 --cached       # replay a cached evaluation episode
    python view.py --controller classical --seed 10000             # run the controller live, then replay
    python view.py --controller oracle --seed 10000 --snapshot     # results/snapshot.png, no window

--cached re-simulates the exact actions stored in data/results/<controller>_<seed>.json
(the physics and sensor noise are seeded, so it reproduces the evaluated episode without any
TabPFN calls). Without --cached, `tabpfn` needs TABPFN_TOKEN and calls the cloud API per stroke.
Viewer options: --slow 2 plays at half speed, --loop repeats until the window is closed.
Regrasps (wrist back to 0 deg with open fingers) are instantaneous jumps in the replay.
"""
import argparse
import json
import time

import mujoco
import numpy as np

from evaluate import DATA, RES, make
from sim import BottleSim, run_episode, sample_bottle


class Replay:
    """Controller that repeats recorded actions."""

    def __init__(self, actions):
        self.actions = actions

    def reset(self):
        self.i = 0

    def act(self, history):
        F, tau = self.actions[self.i]
        self.i += 1
        return F, tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="classical")
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--cached", action="store_true", help="replay the actions of a cached evaluation episode")
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--slow", type=float, default=1.0, help="slow-motion factor for the viewer")
    ap.add_argument("--loop", action="store_true")
    a = ap.parse_args()
    b = sample_bottle(a.seed)
    if a.cached:
        cached = json.loads((RES / f"{a.controller}_{a.seed}.json").read_text())
        ctrl = Replay([(o["F"], o["tau"]) for o in cached["history"]])
        n = len(cached["history"])
    else:
        ctrl, n = make(a.controller, b), None
    ep = run_episode(b, ctrl, seed=a.seed, record=True, verbose=True, **({"max_strokes": n} if n else {}))
    print("opened" if ep["opened"] else "not opened", f"after {len(ep['history'])} strokes")
    sim = BottleSim(b)
    m, d = sim.m, sim.d
    frames = ep["frames"]
    if a.snapshot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        r = mujoco.Renderer(m, 480, 400)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0, 0, 0.22]; cam.distance = 0.32; cam.elevation = -20; cam.azimuth = 120
        picks = np.linspace(0, len(frames) - 1, 4).astype(int)
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
        for ax, i in zip(axes, picks):
            d.qpos[:] = frames[i]
            mujoco.mj_forward(m, d)
            r.update_scene(d, cam)
            ax.imshow(r.render()); ax.axis("off")
            ax.set_title(f"cap {np.rad2deg(frames[i][sim.q_cap]):.0f} deg", fontsize=9)
        fig.tight_layout()
        (DATA.parent / "results").mkdir(exist_ok=True)
        fig.savefig(DATA.parent / "results" / "snapshot.png", dpi=120)
        print("wrote", DATA.parent / "results" / "snapshot.png")
        return
    from mujoco import viewer as mj_viewer
    with mj_viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = [0, 0, 0.22]; v.cam.distance = 0.35; v.cam.elevation = -20; v.cam.azimuth = 120
        while v.is_running():
            d.qpos[:] = frames[0]; mujoco.mj_forward(m, d); v.sync()
            time.sleep(1.5)                       # give the window time to appear
            for q in frames:
                if not v.is_running():
                    break
                d.qpos[:] = q
                mujoco.mj_forward(m, d)
                v.sync()
                time.sleep(0.01 * a.slow)         # frames are 10 ms of sim time apart
            if not a.loop:
                break
        while v.is_running():
            time.sleep(0.1)


if __name__ == "__main__":
    main()
