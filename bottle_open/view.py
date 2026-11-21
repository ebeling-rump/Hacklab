"""Replay one opening episode in the MuJoCo viewer, or save a snapshot strip offscreen.

    python view.py --controller classical --seed 10000             # interactive viewer
    python view.py --controller oracle --seed 10000 --snapshot     # data/snapshot.png, no window

`tabpfn` needs TABPFN_TOKEN and calls the cloud API for every stroke.
"""
import argparse
import time

import mujoco
import numpy as np

from evaluate import DATA, make
from sim import BottleSim, run_episode, sample_bottle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="classical")
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--snapshot", action="store_true")
    a = ap.parse_args()
    b = sample_bottle(a.seed)
    ep = run_episode(b, make(a.controller, b), seed=a.seed, record=True, verbose=True)
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
        for q in frames:
            if not v.is_running():
                break
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            v.sync()
            time.sleep(0.01)           # frames are 10 ms of sim time apart
        while v.is_running():
            time.sleep(0.1)


if __name__ == "__main__":
    main()
