"""Watch episodes live in the MuJoCo viewer (real time, needs a display).

Small dots show what the controller is thinking:
  - obstacle-coloured dots: the predictor's guess of each obstacle over the next 1 s
  - blue dots: where the robot would go along the chosen command

Usage:
    python view.py                                    # warehouse, seed 0, constant velocity
    python view.py --seed 7 --predictor oracle
    python view.py --scenario open --predictor none --no-safety   # Phase 1 arena, watch it crash
Close the viewer before batch runs: it keeps the CPU busy and skews latency numbers.
When an episode ends, the next seed starts. Close the window to quit.
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from scenarios import SCENARIOS
from sim import ALL_PREDICTORS, Episode

OBS_RGBA = [np.array(c + [0.7]) for c in ([0.92, 0.41, 0.2], [0.38, 0.31, 0.84], [0.11, 0.69, 0.48],
                                          [0.91, 0.48, 0.64], [0.89, 0.29, 0.28])]
ROBOT_RGBA = np.array([0.16, 0.47, 0.84, 0.7])


def draw_markers(viewer, ep):
    scn = viewer.user_scn
    scn.ngeom = 0

    def dot(pos, rgba, size=0.06):
        if scn.ngeom >= scn.maxgeom:
            return
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [size, 0, 0],
                            [pos[0], pos[1], 0.45], np.eye(3).flatten(), rgba)
        scn.ngeom += 1

    if ep.predictions is not None:
        for i, path in enumerate(ep.predictions):
            for p in path[1::2]:
                dot(p, OBS_RGBA[i % len(OBS_RGBA)])
    for t in ep.planner.times[1::2]:
        dot(ep.robot_pos + ep.cmd * t, ROBOT_RGBA, 0.045)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="warehouse", choices=list(SCENARIOS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--predictor", default="constant_velocity", choices=ALL_PREDICTORS)
    parser.add_argument("--no-safety", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed factor")
    args = parser.parse_args()

    seed = args.seed
    ep = Episode(seed, args.predictor, safety=not args.no_safety, scenario=args.scenario)
    with mujoco.viewer.launch_passive(ep.model, ep.data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0]
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 13, -60, 90
        while viewer.is_running():
            wall0, sim0 = time.perf_counter(), ep.data.time
            for _ in range(ep.control_every):       # one control period (0.1 s)
                ep.step()
            draw_markers(viewer, ep)
            viewer.sync()
            if ep.done:
                m = ep.metrics()
                print(f"seed {seed}: reached={m['reached_goal']} t={m['time_to_goal']} "
                      f"collisions={m['collision_events']} min gap={m['min_clearance']:.3f} m")
                time.sleep(1.5)
                seed += 1
                ep = Episode(seed, args.predictor, safety=not args.no_safety, scenario=args.scenario,
                             model=ep.model, data=ep.data)
                continue
            lag = (ep.data.time - sim0) / args.speed - (time.perf_counter() - wall0)
            if lag > 0:
                time.sleep(lag)


if __name__ == "__main__":
    main()
