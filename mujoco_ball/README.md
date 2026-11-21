# MuJoCo ball on a plane

A minimal MuJoCo sandbox: a 10 cm, 0.5 kg ball is pushed across a flat floor,
slides briefly, starts rolling, and is slowed by rolling friction.

| File | What it does |
|---|---|
| `ball.xml` | The scene (MJCF): floor, ball, friction, and a `push` keyframe with the start velocity |
| `simulate.py` | Runs headless and prints position/speed; `--csv` saves the trajectory |
| `view.py` | Opens the interactive 3D viewer |

## Setup (once)

```bash
cd mujoco_ball
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run headless

```bash
python simulate.py
python simulate.py --seconds 5 --csv trajectory.csv
```

## Run the interactive viewer (needs your own screen)

```bash
python view.py      # Linux / Windows
mjpython view.py    # macOS (mjpython is installed with mujoco)
```

In the viewer:

- **Space**: pause / resume. **Right arrow**: step one frame while paused.
- **Left-drag**: rotate camera. **Right-drag**: pan. **Scroll**: zoom.
- **Double-click the ball** to select it, then **Ctrl + right-drag** to push it
  (Ctrl + left-drag applies a torque / spin).
- **Backspace**: reset. Left panel > *Simulation* > *Key* > load `push` to restart with the initial push.
- **Tab** / **Shift+Tab**: hide the side panels.

## Things to try

- Change the start velocity in `ball.xml` (`qvel="2 0.5 0 ..."`).
- Change friction (`friction="sliding torsional rolling"`) and watch how fast it stops.
- Change `mass` or `size` of the ball.
