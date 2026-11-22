"""Action-dependent collision data for the TabPFN safety filter.

The random-walk datasets of a1_dataset.py label a collision during one 0.2 s
step. Over such a short time the command barely changes the outcome (and most
collisions are "still touching from last step"), so a model trained on them
only learns "a box is close". To choose between commands, the label must
depend on the command. Every sample here is therefore:

  1. reset the robot at a random pose (biased to be near a box),
  2. walk WARMUP s with a random "previous" command (realistic gait phase and
     velocity; samples already touching a box afterwards are discarded),
  3. record the lidar scan,
  4. hold a random candidate command for HORIZON s,
  5. label = did the robot touch a box during those HORIZON s?

Saved to datasets/collision_<env>.npz with lidar (N, 72), prev_action (N, 2),
action (N, 2), collision (N,), fallen (N,).

    python a1_collision_dataset.py            # both layouts, N_SAMPLES each
"""

import time
from pathlib import Path

import mujoco
import numpy as np

from a1_dataset import ENVIRONMENTS, box_contact, build_model, dist_to_box
from a1_lidar import MAX_TURN, SMOOTHING, gait, read_lidar

N_SAMPLES = 3000
WARMUP = 0.6        # s of walking with the previous command
HORIZON = 0.8       # s the candidate command is held (the filter re-plans every 0.2 s)
NEAR_BOX = (0.25, 1.0)   # start distance range to the nearest box (most samples)
ARENA = 3.0
FALL_HEIGHT = 0.15
OUT_DIR = Path(__file__).parent / "datasets"


def random_command(rng):
    return np.array([rng.uniform(0.0, 1.0), rng.uniform(-MAX_TURN, MAX_TURN)])


def reset_near_box(model, data, boxes, rng):
    while True:
        xy = rng.uniform(-ARENA, ARENA, size=2)
        dists = [dist_to_box(xy, b) for b in boxes]
        d = min(dists)
        if NEAR_BOX[0] < d < NEAR_BOX[1] or (d > NEAR_BOX[0] and rng.random() < 0.05):
            break
    h = rng.uniform(-np.pi, np.pi)
    if rng.random() < 0.5:   # half the time: roughly facing the nearest box
        bx, by = boxes[int(np.argmin(dists))][:2]
        h = np.arctan2(by - xy[1], bx - xy[0]) + rng.uniform(-1.0, 1.0)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    data.qpos[:2] = xy
    data.qpos[3:7] = [np.cos(h / 2), 0, 0, np.sin(h / 2)]
    mujoco.mj_forward(model, data)


def simulate(model, data, cmd, target, seconds, alpha, box_ids):
    """Run the smoothed gait towards `target`; return (cmd, touched_box)."""
    hit = False
    for _ in range(int(round(seconds / model.opt.timestep))):
        cmd += alpha * (target - cmd)
        data.ctrl[:] = gait(data.time, *cmd)
        mujoco.mj_step(model, data)
        hit |= box_contact(model, data, box_ids) >= 0
    return cmd, hit


def generate(name, boxes, seed, n=N_SAMPLES):
    model = build_model(boxes)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    box_ids = {model.geom(f"box{i}").id: i for i in range(len(boxes))}
    alpha = model.opt.timestep / SMOOTHING
    out = {k: [] for k in ("lidar", "prev_action", "action", "collision", "fallen")}
    t0 = time.time()
    while len(out["collision"]) < n:
        reset_near_box(model, data, boxes, rng)
        prev = random_command(rng)
        cmd, hit = simulate(model, data, np.zeros(2), prev, WARMUP, alpha, box_ids)
        if hit or data.qpos[2] < FALL_HEIGHT or box_contact(model, data, box_ids) >= 0:
            continue
        lidar = read_lidar(data)
        action = random_command(rng)
        _, hit = simulate(model, data, cmd, action, HORIZON, alpha, box_ids)
        for k, v in zip(out, (lidar, prev, action, hit, data.qpos[2] < FALL_HEIGHT)):
            out[k].append(v)
        if len(out["collision"]) % 500 == 0:
            print(f"{name}: {len(out['collision'])}/{n} ({time.time() - t0:.0f}s)")
    return {k: np.array(v) for k, v in out.items()}


def main():
    OUT_DIR.mkdir(exist_ok=True)
    for i, (name, boxes) in enumerate(ENVIRONMENTS.items()):
        ds = generate(name, boxes, seed=200 + i)
        np.savez_compressed(OUT_DIR / f"collision_{name}.npz", **ds, boxes=boxes,
                            horizon=HORIZON, warmup=WARMUP)
        print(f"{name}: collision rate {ds['collision'].mean():.1%}, "
              f"falls {ds['fallen'].mean():.1%}")


if __name__ == "__main__":
    main()
