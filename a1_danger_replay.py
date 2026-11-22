"""Find and replay the danger-mode episodes where TabPFN beats the other strategies.

    python a1_danger_replay.py --list
        reads results/danger_mode.json (from a1_danger_mode.py) and prints
          - steep floors where TabPFN stayed up but the unchanged gait fell
          - steep floors where TabPFN stayed up but the tilt rule fell
          - steep floors where the tilt rule stayed up but TabPFN fell (fairness)
          - safe floors where the tilt rule braced for nothing and TabPFN walked further

    python a1_danger_replay.py --kind dangerous --episode 7 --controller TabPFN
        replays that exact episode in the MuJoCo viewer. Same floor, commands
        and pushes as the logged run; the simulation is deterministic, so the
        outcome should match the log (checked and printed at the end).

Open two at once (two terminals) to compare controllers side by side on the
same episode, e.g. --controller TabPFN and --controller "never brace".

In the viewer:
  ball above the body  blue = braced (wide, low, half speed), grey = normal gait
  red spheres          sliding feet
  white arrow          downhill
  top-left text        controller, state, TabPFN fall risk or tilt, time
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from a1_danger_mode import (OUT as LOG, THRESHOLD, TILT_LIMIT, episode_setup, fit_fall_model,
                            make_controllers)
from a1_slip_dataset import NORMAL, A1Sensors, build_model, run_episode

warnings.filterwarnings("ignore")
CONTROLLERS = ["never brace", "always brace", "tilt rule", "TabPFN"]


def load_log():
    rows = json.loads(Path(LOG).read_text())
    by = {}
    for r in rows:
        by.setdefault((r["kind"], r["episode"]), {})[r["controller"]] = r
    return by


def list_wins():
    by = load_log()
    steep = {k: v for k, v in by.items() if k[0] == "dangerous"}
    safe = {k: v for k, v in by.items() if k[0] == "safe"}

    def show(title, items):
        print(f"\n{title}: {len(items)}")
        for (kind, e), v in items:
            c = {n: ("FELL" if v[n]["fell"] else f"{v[n]['distance']:.2f} m") for n in CONTROLLERS}
            print(f"  --kind {kind} --episode {e:2d}   mu {v['TabPFN']['mu']:.2f} "
                  f"slope {v['TabPFN']['slope']:4.1f}   " + "  ".join(f"{n}: {c[n]}" for n in CONTROLLERS))

    show("Steep floors where TabPFN stayed up and the unchanged gait fell",
         sorted(((k, v) for k, v in steep.items()
                 if not v["TabPFN"]["fell"] and v["never brace"]["fell"]),
                key=lambda kv: -kv[1]["TabPFN"]["distance"]))
    show("Steep floors where TabPFN stayed up and the tilt rule fell",
         [(k, v) for k, v in steep.items() if not v["TabPFN"]["fell"] and v["tilt rule"]["fell"]])
    show("Steep floors where the tilt rule stayed up and TabPFN fell (TabPFN's losses)",
         [(k, v) for k, v in steep.items() if v["TabPFN"]["fell"] and not v["tilt rule"]["fell"]])

    gaps = sorted(((v["TabPFN"]["distance"] - v["tilt rule"]["distance"], k, v)
                   for k, v in safe.items()), key=lambda g: -g[0])[:5]
    print("\nSafe floors where the tilt rule braced for nothing (TabPFN walked further), top 5:")
    for gap, (kind, e), v in gaps:
        tr = v["tilt rule"]
        print(f"  --kind {kind} --episode {e:2d}   mu {tr['mu']:.2f} slope {tr['slope']:4.1f}   "
              f"TabPFN {v['TabPFN']['distance']:.2f} m vs tilt rule {tr['distance']:.2f} m "
              f"(+{gap:.2f} m; tilt rule braced {tr['braced_s'] / tr['duration']:.0%} of the time)")


def replay(kind, e, name, speed):
    risk = keep = None
    if name == "TabPFN":
        risk, keep = fit_fall_model()
    make = make_controllers(risk, keep)[name]
    policy = make()
    model = build_model()
    data = mujoco.MjData(model)
    sensors = A1Sensors(model)
    fl, rng = episode_setup(kind, e)
    dt = model.opt.timestep
    blue = np.array([0.2, 0.45, 0.9, 0.9], np.float32)
    grey = np.array([0.6, 0.6, 0.6, 0.9], np.float32)
    red = np.array([1, 0.1, 0.1, 0.9], np.float32)

    def text(v, left, right):
        try:
            v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                         left, right))
        except Exception:
            pass

    with mujoco.viewer.launch_passive(model, data) as v:
        with v.lock():
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = sensors.trunk
            v.cam.distance, v.cam.elevation = 2.2, -20
        last = [time.time(), 0]

        def on_step(data, reaction, sliding):
            if not v.is_running():
                return False
            braced = reaction != NORMAL
            with v.lock():
                scn, n = v.user_scn, 0
                trunk = data.xpos[sensors.trunk]
                mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.06, 0, 0]),
                                    trunk + [0, 0, 0.3], np.eye(3).flatten(), blue if braced else grey)
                n += 1
                for i, g in enumerate(sensors.feet):
                    if sliding[i]:
                        mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_SPHERE,
                                            np.array([0.045, 0, 0]), data.geom_xpos[g].copy(),
                                            np.eye(3).flatten(), red)
                        n += 1
                gdir = model.opt.gravity[:2]
                if np.linalg.norm(gdir) > 1e-6:
                    start = trunk + [0, 0, 0.45]
                    end = start + np.r_[0.4 * gdir / np.linalg.norm(gdir), 0]
                    mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                                        np.zeros(3), np.eye(3).flatten(),
                                        np.array([1, 1, 1, 0.8], np.float32))
                    mujoco.mjv_connector(scn.geoms[n], mujoco.mjtGeom.mjGEOM_ARROW, 0.015, start, end)
                    n += 1
                scn.ngeom = n
            last[1] += 1
            if last[1] % 50 == 0:   # refresh text 10x per simulated second
                sig = ""
                if name == "TabPFN":
                    sig = f"fall risk {policy.value:.0%} (brace > {THRESHOLD:.0%})"
                elif name == "tilt rule":
                    sig = f"tilt {policy.value:.1f} deg (brace > {TILT_LIMIT:.0f})"
                text(v, "controller\nstate\nsignal\nfloor\ntime",
                     f"{name}\n{'BRACED' if braced else 'normal gait'}\n{sig}\n"
                     f"mu {fl[0]:.2f}, slope {fl[1]:.1f} deg\n{data.time:.1f} s")
            v.sync()
            # Real-time pacing (scaled by --speed); TabPFN thinking time just adds pauses.
            target = last[0] + dt / speed
            now = time.time()
            if target > now:
                time.sleep(target - now)
            last[0] = max(target, time.time() - 0.05)
            return True

        ep = run_episode(model, data, sensors, rng, *fl, policy=policy, on_step=on_step)
        fell = ep["fall_t"] is not None
        dist = float(np.linalg.norm(data.qpos[:2]))
        logged = load_log().get((kind, e), {}).get(name)
        match = ("no log entry" if logged is None else
                 "yes" if logged["fell"] == fell and abs(logged["distance"] - dist) < 1e-6 else
                 f"NO (log: fell={logged['fell']}, {logged['distance']:.2f} m)")
        result = f"{'FELL' if fell else 'stayed up'}, {dist:.2f} m walked"
        print(f"{name} on {kind} episode {e}: {result}; matches logged run: {match}", flush=True)
        text(v, "controller\nresult\nmatches log", f"{name}\n{result}\n{match}")
        while v.is_running():   # keep the window open
            v.sync()
            time.sleep(0.03)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--kind", choices=["dangerous", "safe"], default="dangerous")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--controller", choices=CONTROLLERS, default="TabPFN")
    ap.add_argument("--speed", type=float, default=1.0, help="0.5 = slow motion")
    args = ap.parse_args()
    if args.list:
        list_wins()
    else:
        replay(args.kind, args.episode, args.controller, args.speed)


if __name__ == "__main__":
    main()
