"""Run the ball simulation headless and print its trajectory.

Usage:
    python simulate.py              # 3 seconds, print every 0.25 s
    python simulate.py --seconds 5 --csv trajectory.csv
"""
import argparse
import csv
from pathlib import Path

import mujoco

MODEL_PATH = Path(__file__).parent / "ball.xml"


def load():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("push").id)
    return model, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--print-every", type=float, default=0.25)
    parser.add_argument("--csv", type=Path, help="optional path to save t,x,y,z,vx,vy,vz")
    args = parser.parse_args()

    model, data = load()
    print_steps = max(1, round(args.print_every / model.opt.timestep))
    rows = []

    print(f"{'t':>6} {'x':>8} {'y':>8} {'z':>8} {'speed':>8}")
    step = 0
    while data.time < args.seconds:
        if step % print_steps == 0:
            x, y, z = data.qpos[:3]
            vx, vy, vz = data.qvel[:3]
            speed = (vx**2 + vy**2 + vz**2) ** 0.5
            print(f"{data.time:6.2f} {x:8.3f} {y:8.3f} {z:8.3f} {speed:8.3f}")
        rows.append((data.time, *data.qpos[:3], *data.qvel[:3]))
        mujoco.mj_step(model, data)
        step += 1

    if args.csv:
        with args.csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "x", "y", "z", "vx", "vy", "vz"])
            writer.writerows(rows)
        print(f"Saved {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
