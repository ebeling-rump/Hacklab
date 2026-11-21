# Predictive Robot Navigation

Research question: can TabPFN learn to predict the movement of dynamic obstacles from
very few observations, and can these predictions improve robot navigation?

**So far (no TabPFN yet):**
- Phase 1: open arena with two simple obstacles, controller, metrics, visualization (`--scenario open`).
- Phase 1b: warehouse intersection scenario with rule-following AGVs, the baseline for the
  rest of the project (`--scenario warehouse`, default).

## Warehouse scenario (`warehouse.py`)

- 3 x 3 grid of 2 m wide aisles at x, y in {-3, 0, 3}; shelves in between (solid for the robot).
- Robot starts in a west dead-end stub and must reach an east stub, changing rows through a
  vertical aisle when start and goal rows differ (waypoints at the intersections).
- **4 AGVs** (radius 0.3 m, cruise 0.7-1.1 m/s, accel-limited) with deterministic rules that
  depend only on observable things (position, heading, nearby vehicles):
  - one-way aisles (perimeter counter-clockwise, middle aisles west/north),
  - location-based routing at intersections (turn where i + j is odd, else straight if allowed),
  - slow to 0.35 m/s before turning, stop 1.5 s at five pick stations (yellow squares),
  - stop-line and priority-to-the-right at intersections, keep distance to AGVs ahead,
  - **right of way over the robot:** an AGV only emergency-brakes when the robot is < 0.7 m
    directly in its path. The robot has to anticipate.
- Obstacles react to where the robot is, so after the first interaction two controllers can
  see different AGV traffic for the same seed.
- The hidden AGV state (cruise speed, routing table, current behaviour `mode`) is logged for
  analysis only and must never be a model feature.

## The open arena (Phase 1)

- 10 m x 10 m arena, physics at 500 Hz, obstacles at 100 Hz, controller at 10 Hz.
- **Robot:** holonomic disc (radius 0.25 m) on two slide joints with velocity servos, max 1 m/s.
  Starts at x = -4, goal at x = +4 (y randomized per seed).
- **Obstacles** (radius 0.3 m, kinematic, bounce off the walls), see `obstacles.py`:
  - `constant_velocity` (orange): crosses the robot's path in a straight line.
  - `random_turns` (purple): constant speed, abrupt new heading every 0.5-1.5 s.
- Contacts are disabled; a **collision** is counted when the gap between robot and obstacle
  drops below 0 m, a **near miss** when the smallest gap is between 0 and 0.3 m.
- One seed fixes start, goal and obstacle motion. Obstacles never react to the robot, so
  every controller faces exactly the same obstacles for a given seed.

## The controller (`controller.py`)

1. **Predictor:** guesses obstacle positions for the next 1 s from *observed* positions only
   (finite differences; the obstacles' hidden parameters are never read).
   `none` (ignore obstacles), `static` (they stay put), `constant_velocity`. TabPFN plugs in here in Phase 3.
   `oracle` (in `sim.py`) simulates a copy of the true obstacles: an **evaluation-only upper
   bound** for "perfect prediction", never a model.
2. **Planner:** samples 98 candidate velocities, rolls each out for 1 s against the predictions,
   picks the one that gets closest to the current waypoint while keeping clearance and staying
   in the aisles; a small bonus for extra room makes a blocked robot step aside.
3. **Safety filter:** local and deterministic, identical for every predictor. Uses only the
   current obstacle positions; when one is within 0.2 m of touching, it cancels motion toward it
   and pushes away. It never waits for a predictor or a network call.

## Run

```bash
cd robot_nav
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python sim.py --save-logs         # warehouse, 50 seeds x 4 predictors -> data/metrics_warehouse.csv
python sim.py --no-safety --predictors none constant_velocity oracle
python analyze.py                 # where does prediction quality matter? (needs --save-logs)
python plot_episode.py --seed 3 --gif                     # top-down plot + GIF + log in data/
python plot_episode.py --seed 3 --predictor oracle
python view.py --seed 3           # live viewer (dots = predictions and planned robot path)
python sim.py --scenario open     # Phase 1 arena
```

Close the viewer before batch runs: it keeps the CPU busy and distorts latency numbers.
`--save-logs` writes per-episode trajectories (`data/episodes/*.npz`, 10 Hz: time, robot
position/velocity/command/target, obstacle positions, hidden obstacle mode, gaps, predictions
at +0.5 s and +1.0 s). `data/` is not committed.

**Metrics:** collision = gap < 0 m; near miss = smallest gap in [0, 0.3) m; stop = robot slower
than 0.05 m/s for >= 0.3 s; unnecessary stop = driving straight on toward the target for 1 s
would have kept >= 0.2 m from the obstacles' actual positions; prediction error = distance
between predicted and actual position at +0.5 s / +1.0 s for obstacles within 3 m of the robot.

## Warehouse baseline (seeds 0-49)

| Predictor | Safety | Collisions | Near misses | Reached | Time to goal | Min gap | Stops / ep. | Pred. error @1 s |
|---|---|---|---|---|---|---|---|---|
| none | off | 64% | 10% | 100% | 12.1 s | 0.01 m | 0.00 | - |
| none | on | 16% | 58% | 94% | 16.5 s | 0.31 m | 0.16 | - |
| static | on | 0% | 62% | 94% | 14.2 s | 0.43 m | 0.34 | 0.48 m |
| **constant velocity** | on | **0%** | **62%** | **100%** | **15.2 s** | **0.45 m** | **0.42** | **0.16 m** |
| constant velocity | off | 0% | 62% | 100% | 15.1 s | 0.44 m | 0.40 | 0.16 m |
| oracle (upper bound) | on | 0% | 56% | 100% | 14.5 s | 0.47 m | 0.26 | 0.006 m |

Constant-velocity error at +1 s by what the AGV does next: turning 0.24 m (31% of samples),
slowing for a station 0.16 m, standing at a station 0.13 m, yielding 0.16 m, cruising 0.09 m.

Prediction-sensitive seeds (oracle fixes a near miss or saves >= 1 s): 10 of 50, mostly at
AGVs turning at intersections (4) and at pick stations (3). Paired over all seeds the oracle
saves 0.63 s on average (median 0.0 s; >= 1 s faster in 7 seeds, >= 1 s slower in 3).

## Phase 2: motion prediction (`motion_data.py`, `predict_motion.py`, `plot_motion.py`)

```bash
python motion_data.py             # trajectories -> data/motion/{train,test,flipped,fast}.npz (~3 min)
python predict_motion.py --models cv gp           # local models
python predict_motion.py --models tabpfn          # needs TABPFN_TOKEN; sends simulated rows to Prior Labs
python predict_motion.py --report-only            # tables from cached predictions
python plot_motion.py                             # data/motion/prediction_errors.png
```

- **Data:** 30 s warehouse episodes with the constant-velocity robot driving, logged at 10 Hz.
  Disjoint seed ranges: train 1000-1039 (45,920 rows), test 2000-2019, unseen *flipped routing*
  3000-3019, unseen *fast fleet* (1.2-1.5 m/s) 4000-4019; 1000 rows sampled per evaluation set.
- **Features (observable only):** position, velocity, acceleration over the last 0.3 s, robot
  position relative to the AGV and its distance. **Targets:** deviation from constant velocity
  at +0.5 s and +1.0 s (x, y); prediction = constant velocity + predicted deviation.
- **Models:** constant velocity; Gaussian Process (ARD RBF + noise, scikit-learn); TabPFN
  (cloud, one regressor per target). Learned models see only N = 100 / 300 / 1000 training rows.

Mean position error at +1.0 s (m):

| Model | N | Test (same pattern) | Flipped routing (unseen) | Fast fleet (unseen) |
|---|---|---|---|---|
| Constant velocity | - | 0.172 | 0.149 | 0.294 |
| Gaussian Process | 100 / 300 / 1000 | 0.124 / 0.092 / 0.073 | 0.173 / 0.139 / 0.111 | 0.214 / 0.190 / 0.170 |
| **TabPFN** | 100 / 300 / 1000 | **0.107 / 0.072 / 0.051** | **0.163 / 0.096 / 0.065** | **0.182 / 0.145 / 0.115** |

By upcoming behaviour (test, N = 1000, CV / GP / TabPFN): turning 0.243 / 0.065 / 0.035,
slowing for a station 0.159 / 0.056 / 0.038, standing at a station 0.138 / 0.130 / 0.127,
yielding 0.131 / 0.201 / 0.171, cruising 0.077 / 0.048 / 0.032.
On flipped routing, cruising AGVs get worse with learning (CV 0.054, GP 0.135, TabPFN 0.064):
the models expect turns where the training fleet turned.

**Latency:** TabPFN via the cloud takes ~2 s per fit and ~1.5-2.5 s per predict call (per target,
3000 rows batched); GP predicts 3000 rows in 0.4 s locally but needs 5 min to fit on 1000 rows.

## Phase 1 results, open arena (seeds 0-49)

| Controller | Safety filter | Collision rate | Near-miss rate | Reached goal | Mean time to goal | Mean smallest gap |
|---|---|---|---|---|---|---|
| none (goal only) | off | 54% | 26% | 100% | 8.5 s | 0.04 m |
| static | off | 14% | 58% | 100% | 9.1 s | 0.26 m |
| constant velocity | off | 0% | 28% | 100% | 9.0 s | 0.42 m |
| none (goal only) | on | 6% | 74% | 100% | 9.1 s | 0.22 m |
| static | on | 0% | 72% | 100% | 9.3 s | 0.28 m |
| constant velocity | on | 0% | 28% | 100% | 9.0 s | 0.42 m |

Planner + safety take about 0.2-0.6 ms per control step.
