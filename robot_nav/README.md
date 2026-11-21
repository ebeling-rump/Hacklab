# Predictive Robot Navigation

Research question: can TabPFN learn to predict the movement of dynamic obstacles from
very few observations, and can these predictions improve robot navigation?

**Phase 1 (this folder so far):** a dynamic MuJoCo environment, a navigation controller
without TabPFN, metrics, and visualization.

## The environment

- 10 m x 10 m arena, physics at 500 Hz, controller at 10 Hz.
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
2. **Planner:** samples 74 candidate velocities, rolls each out for 1 s against the predictions,
   picks the one that gets closest to the goal while keeping clearance.
3. **Safety filter:** local and deterministic. Uses only the current obstacle positions; when one
   is within 0.2 m of touching, it cancels motion toward it and pushes away. It never waits for a
   predictor or a network call.

## Run

```bash
cd robot_nav
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python sim.py                     # 20 seeds x 3 predictors, summary table -> data/metrics.csv
python sim.py --no-safety         # planner only
python plot_episode.py --seed 9 --gif        # top-down plot + GIF + trajectory log in data/
python view.py --seed 9           # live viewer (dots = predictions and planned robot path)
```

`sim.py --save-logs` writes per-episode trajectories (`data/episodes/*.npz`, 10 Hz: time, robot
position/velocity/command, obstacle positions, gaps). `data/` is not committed.

## Phase 1 results (seeds 0-49)

| Controller | Safety filter | Collision rate | Near-miss rate | Reached goal | Mean time to goal | Mean smallest gap |
|---|---|---|---|---|---|---|
| none (goal only) | off | 54% | 24% | 100% | 8.5 s | 0.04 m |
| static | off | 14% | 60% | 100% | 9.2 s | 0.24 m |
| constant velocity | off | 0% | 30% | 100% | 9.0 s | 0.39 m |
| none (goal only) | on | 6% | 72% | 100% | 9.1 s | 0.23 m |
| static | on | 2% | 72% | 100% | 9.4 s | 0.27 m |
| constant velocity | on | 0% | 30% | 100% | 9.0 s | 0.39 m |

Planner + safety take about 0.2-0.4 ms per control step; a 20 s episode simulates in ~0.4 s
on an idle machine.
