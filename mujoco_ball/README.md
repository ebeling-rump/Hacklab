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

## Experiment: can TabPFN learn the ball's dynamics?

Predict where the ball is after 1 s from its start velocity and friction, using 150
simulated runs, and compare against a constant-velocity baseline.

| File | What it does |
|---|---|
| `generate_data.py` | 200 runs with random `vx0, vy0, mu_slide, mu_roll` → `data/ball_runs.csv` (150 train / 50 test, by run) |
| `ball_data.py` | Shared helpers: load the train/test split, compute position errors |
| `baseline.py` | Constant-velocity predictor: `position = v0 * t` |
| `tabpfn_model.py` | TabPFN via the cloud API (`tabpfn-client`), one regressor each for x and y → `data/predictions.csv` |
| `plot_results.py` | Actual vs predicted scatter plots → `data/predictions.png` |

```bash
python generate_data.py     # ~5 s, deterministic (seed 0)
python baseline.py
python tabpfn_model.py      # needs a TabPFN token, see below
python plot_results.py
```

**TabPFN authentication:** set the `TABPFN_TOKEN` environment variable (create a key at
https://platform.priorlabs.ai/account/api-keys), or run
`python -c "import tabpfn_client; tabpfn_client.interactive_login()"` once.
Never commit the token. The training and test rows are sent to Prior Labs' servers.

If you get `CERTIFICATE_VERIFY_FAILED` (e.g. antivirus HTTPS scanning such as Avast Web Shield,
or a corporate proxy), `truststore` from `requirements.txt` makes Python use the OS certificate
store; `tabpfn_model.py` enables it automatically.

**First results** (seed 0, 50 test runs, mean position error after 1 s):

| Model | Error |
|---|---|
| Constant velocity | 0.541 m |
| One fitted factor, `0.665 * v0` | 0.098 m |
| Linear regression on all 4 inputs | 0.099 m |
| `v0 * (a + b*mu_roll + c*mu_slide)` | 0.048 m |
| **TabPFN** | **0.035 m** |

TabPFN is most accurate at low rolling friction and worst for balls that nearly stop
within the second (~0.12 m error), which are rare in the training data. It interpolates
within the training range; it has not learned the laws of motion.
