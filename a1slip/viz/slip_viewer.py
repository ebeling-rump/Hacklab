"""Watch the A1 on a slippery slope with TabPFN's live slip and fall risk.

What you see in the MuJoCo viewer:
  red spheres     feet that are sliding right now (ground truth from MuJoCo)
  ball above body TabPFN fall risk: green = safe ... red = fall likely within 1 s
  white arrow     downhill direction (the slope is made by tilting gravity, so
                  the floor looks flat: watch the robot drift along the arrow)
  top-left text   slip risk (next 0.2 s), fall risk (next 1 s), command, pushes

The robot does what the data generator did: a new random walking command every
second and random shoves on the trunk. Nothing reacts to the risk - this shows
the prediction, not a controller. After a fall it resets.

Models: TabPFN slip (base + history features) and fall (base + history without
the 2 s tilt averages), trained on the normal floors of data/slip.npz.
Predictions run every 0.2 s and take ~0.3 s on CPU, so motion is slower than
real time.

    python -m a1slip.viz.slip_viewer                      # mu 0.15, 12 deg
    python -m a1slip.viz.slip_viewer --mu 0.6 --slope 14  # grippy but steep
"""

import argparse
import time
import warnings

import mujoco
import mujoco.viewer
import numpy as np

from a1slip.models.tabpfn_risk import RiskModel
from a1slip.sim.robot import MAX_TURN, SMOOTHING, gait
from a1slip.sim.slip_dataset import (CHECK_EVERY, CONTACT_FORCE, PUSH_FORCE, PUSH_RATE, PUSH_TIME,
                             SUSTAIN, A1Sensors, History, build_model)
from a1slip.prediction.slip_eval import DATA, fall_splits, splits

warnings.filterwarnings("ignore")
PREDICT_EVERY = 0.2   # s
WARMUP = 0.3          # s before the first prediction (history buffers need some data).
                      # The training data only starts at 1 s, so a fall in the first second
                      # (it happens on steep ice) is outside what the model has seen.
MAX_EPISODE = 30.0


def fit_models():
    d = dict(np.load(DATA))
    names = list(d["history_names"])
    keep = [i for i, n in enumerate(names) if not n.startswith("grav_")]
    Xs = np.hstack([d["features"], d["history"]])
    Xf = np.hstack([d["features"], d["history"][:, keep]])
    tr_s, tr_f = splits(d)["train"], fall_splits(d)["train"]
    ys, yf = d["slip_next"][tr_s].astype(int), d["fall_next"][tr_f].astype(int)
    t = time.time()
    slip = RiskModel().fit(Xs[tr_s], ys, n=2000, pos_fraction=max(ys.mean(), 0.25))
    fall = RiskModel().fit(Xf[tr_f], yf, n=2000, pos_fraction=0.25)
    print(f"TabPFN slip + fall models ready ({time.time() - t:.0f}s)", flush=True)
    return slip, fall, keep


def risk_colour(p):
    p = float(np.clip(p, 0, 1))
    return np.array([min(1, 2 * p), min(1, 2 * (1 - p)), 0.15, 0.9], np.float32)


def draw(v, model, data, sensors, sliding, p_fall):
    scn = v.user_scn
    n = 0
    trunk = data.xpos[sensors.trunk]
    # Fall-risk ball above the body.
    mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.06, 0, 0]),
                        trunk + [0, 0, 0.3], np.eye(3).flatten(), risk_colour(p_fall))
    n += 1
    # Sliding feet.
    for i, g in enumerate(sensors.feet):
        if sliding[i]:
            mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.045, 0, 0]),
                                data.geom_xpos[g].copy(), np.eye(3).flatten(),
                                np.array([1, 0.1, 0.1, 0.9], np.float32))
            n += 1
    # Downhill arrow.
    gdir = model.opt.gravity[:2]
    if np.linalg.norm(gdir) > 1e-6:
        start = trunk + [0, 0, 0.45]
        end = start + np.r_[0.4 * gdir / np.linalg.norm(gdir), 0]
        mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                            np.eye(3).flatten(), np.array([1, 1, 1, 0.8], np.float32))
        mujoco.mjv_connector(scn.geoms[n], mujoco.mjtGeom.mjGEOM_ARROW, 0.015, start, end)
        n += 1
    scn.ngeom = n


def set_text(v, left, right):
    try:
        v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                     left, right))
    except Exception:
        pass   # older MuJoCo: no overlay text; the terminal still prints the risks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, default=0.15, help="floor friction (0.1 ice ... 1 rubber)")
    ap.add_argument("--slope", type=float, default=12.0, help="slope in degrees")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    slip_model, fall_model, keep = fit_models()
    model = build_model()
    data = mujoco.MjData(model)
    sensors = A1Sensors(model)
    rng = np.random.default_rng(args.seed)
    sensors.set_floor(args.mu, args.slope, rng.uniform(0, 2 * np.pi))
    dt = model.opt.timestep
    alpha = dt / SMOOTHING

    with mujoco.viewer.launch_passive(model, data) as v:
        with v.lock():
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = sensors.trunk
            v.cam.distance, v.cam.elevation = 2.2, -20
        episode = 0
        while v.is_running():
            episode += 1
            mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
            mujoco.mj_forward(model, data)
            hist = History(sensors, CHECK_EVERY * dt)
            cmd, target = np.zeros(2), np.zeros(2)
            streak, sliding = np.zeros(4, int), np.zeros(4, bool)
            push_until, push = -1.0, np.zeros(3)
            p_slip = p_fall = 0.0
            next_predict, k, fell = WARMUP, 0, False
            set_text(v, "slip risk (0.2 s)\nfall risk (1 s)", "warming up\nwarming up")
            while v.is_running() and data.time < MAX_EPISODE:
                wall = time.time()
                t = data.time
                if k % int(round(1.0 / dt)) == 0:
                    target = np.array([rng.uniform(0, 1), rng.uniform(-MAX_TURN, MAX_TURN)])
                if t > push_until and rng.random() < PUSH_RATE * dt:
                    a = rng.uniform(0, 2 * np.pi)
                    push = rng.uniform(0, PUSH_FORCE) * np.array([np.cos(a), np.sin(a), 0])
                    push_until = t + PUSH_TIME
                data.xfrc_applied[sensors.trunk, :3] = push if t < push_until else 0
                if t >= next_predict:
                    f = sensors.features(data, cmd)
                    h, _ = hist.features()
                    p_slip = float(slip_model.predict(np.r_[f, h])[0])
                    p_fall = float(fall_model.predict(np.r_[f, h[keep]])[0])
                    next_predict = t + PREDICT_EVERY
                    push_txt = f"{np.linalg.norm(push):.0f} N" if t < push_until else "-"
                    set_text(v, "slip risk (0.2 s)\nfall risk (1 s)\ncommand\npush\nfloor",
                             f"{p_slip:.0%}\n{p_fall:.0%}\n"
                             f"fwd {target[0]:.2f}  turn {target[1]:+.2f}\n{push_txt}\n"
                             f"mu {args.mu}, {args.slope:.0f} deg")
                cmd += alpha * (target - cmd)
                data.ctrl[:] = gait(t, *cmd)
                mujoco.mj_step(model, data)
                k += 1
                if k % CHECK_EVERY == 0:
                    forces = sensors.foot_forces(data)
                    raw = sensors.sliding_feet(data, forces)
                    streak = np.where(raw, streak + 1, 0)
                    sliding = (streak >= SUSTAIN) & (forces > CONTACT_FORCE)
                    hist.update(data, forces, sliding.any())
                    if sensors.fallen(data):
                        fell = True
                with v.lock():
                    draw(v, model, data, sensors, sliding, p_fall)
                v.sync()
                if fell:
                    break
                time.sleep(max(0.0, dt - (time.time() - wall)))
            print(f"episode {episode}: {'fell' if fell else 'no fall'} at t={data.time:.1f}s "
                  f"(last fall risk {p_fall:.0%})", flush=True)
            if fell:
                set_text(v, "FELL", f"fall risk just before: {p_fall:.0%}")
                end = time.time() + 2.0
                while v.is_running() and time.time() < end:
                    v.sync()
                    time.sleep(0.02)


if __name__ == "__main__":
    main()
