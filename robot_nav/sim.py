"""Run navigation episodes: robot drives to a goal while two obstacles move.

One episode = one seed. The seed fixes the start, goal and obstacle motion, so the same
seed replays the same obstacles regardless of which controller drives the robot.

Usage:
    python sim.py                                   # 20 episodes, all controllers, summary table
    python sim.py --episodes 50 --predictors static constant_velocity
    python sim.py --no-safety                       # planner only, safety filter off
    python sim.py --save-logs                       # also write data/episodes/<predictor>_<seed>.npz
"""
import argparse
import csv
import time
from pathlib import Path

import mujoco
import numpy as np

from controller import PREDICTORS, R_SUM, ObstacleTracker, Planner, PlannerConfig, safety_filter
from obstacles import make_obstacles

HERE = Path(__file__).parent
SCENE = HERE / "scene.xml"
DATA = HERE / "data"

CONTROL_DT = 0.1     # s, planner + safety rate (10 Hz); physics runs at 500 Hz
NEAR_MISS = 0.3      # clearance (gap between surfaces, m) below this counts as a near miss
GOAL_TOL = 0.2       # m
MAX_TIME = 20.0      # s


def make_scenario(seed):
    rng = np.random.default_rng(seed)
    start = np.array([-4.0, rng.uniform(-1, 1)])
    goal = np.array([4.0, rng.uniform(-1, 1)])
    return start, goal, make_obstacles(rng)


class Episode:
    """Steps physics, obstacles and controller together; records a 10 Hz log and metrics."""

    def __init__(self, seed, predictor="constant_velocity", safety=True, max_time=MAX_TIME,
                 model=None, data=None):
        """Pass `model`/`data` to reuse them (e.g. a live viewer); they are reset here."""
        self.model = model if model is not None else mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = data if data is not None else mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        self.seed, self.predictor_name, self.safety, self.max_time = seed, predictor, safety, max_time

        start, self.goal, self.obstacles = make_scenario(seed)
        self.data.qpos[:2] = start
        self.model.site_pos[self.model.site("goal").id][:2] = self.goal
        self.mocap = [self.model.body_mocapid[self.model.body(f"obs_{i}").id]
                      for i in range(len(self.obstacles))]
        self._place_obstacles()
        mujoco.mj_forward(self.model, self.data)

        self.dt = self.model.opt.timestep
        self.control_every = round(CONTROL_DT / self.dt)
        self.tracker = ObstacleTracker(len(self.obstacles), CONTROL_DT)
        self.planner = Planner(PlannerConfig())
        self.predict = PREDICTORS[predictor]
        self.cmd = np.zeros(2)
        self.predictions = None
        self.n_steps = 0

        # metrics
        self.min_clearance = np.inf
        self.collision_events = 0
        self._in_collision = np.zeros(len(self.obstacles), dtype=bool)
        self.time_to_goal = None
        self.predict_seconds = 0.0   # wall-clock time spent in the predictor
        self.plan_seconds = 0.0      # wall-clock time in planner + safety filter
        self.n_control = 0
        self.log = {k: [] for k in ("t", "robot_pos", "robot_vel", "cmd", "obs_pos", "clearance")}

    # -- helpers --
    @property
    def robot_pos(self):
        return self.data.qpos[:2].copy()

    @property
    def done(self):
        return self.time_to_goal is not None or self.data.time >= self.max_time

    def observed_obstacles(self):
        return np.array([o.pos for o in self.obstacles])

    def _place_obstacles(self):
        for o, m in zip(self.obstacles, self.mocap):
            self.data.mocap_pos[m] = [o.pos[0], o.pos[1], 0.2]

    # -- control (10 Hz) --
    def _control(self):
        obs = self.observed_obstacles()
        self.tracker.update(obs)
        robot = self.robot_pos

        t0 = time.perf_counter()
        self.predictions = self.predict(self.tracker, self.planner.times)
        t1 = time.perf_counter()
        cmd = self.planner.plan(robot, self.goal, self.predictions)
        if self.safety:
            cmd = safety_filter(cmd, robot, obs, v_max=self.planner.cfg.v_max)
        t2 = time.perf_counter()

        self.predict_seconds += t1 - t0
        self.plan_seconds += t2 - t1
        self.n_control += 1
        self.cmd = cmd

        clearance = np.linalg.norm(obs - robot, axis=1) - R_SUM
        for key, value in (("t", self.data.time), ("robot_pos", robot), ("robot_vel", self.data.qvel[:2].copy()),
                           ("cmd", cmd), ("obs_pos", obs), ("clearance", clearance)):
            self.log[key].append(value)

    # -- one physics step (2 ms) --
    def step(self):
        if self.n_steps % self.control_every == 0:
            self._control()
        for o in self.obstacles:
            o.step(self.dt)
        self._place_obstacles()
        self.data.ctrl[:] = self.cmd
        mujoco.mj_step(self.model, self.data)
        self.n_steps += 1

        clearance = np.linalg.norm(self.observed_obstacles() - self.robot_pos, axis=1) - R_SUM
        self.min_clearance = min(self.min_clearance, clearance.min())
        colliding = clearance < 0
        self.collision_events += int((colliding & ~self._in_collision).sum())
        self._in_collision = colliding
        if np.linalg.norm(self.goal - self.robot_pos) < GOAL_TOL:
            self.time_to_goal = self.data.time

    def run(self):
        t0 = time.perf_counter()
        while not self.done:
            self.step()
        self.wall_seconds = time.perf_counter() - t0
        return self.metrics()

    def metrics(self):
        return {
            "seed": self.seed,
            "predictor": self.predictor_name,
            "safety": self.safety,
            "reached_goal": self.time_to_goal is not None,
            "time_to_goal": self.time_to_goal,
            "collided": self.collision_events > 0,
            "collision_events": self.collision_events,
            "near_miss": bool(0 <= self.min_clearance < NEAR_MISS),
            "min_clearance": float(self.min_clearance),
            "predict_ms": 1000 * self.predict_seconds / max(self.n_control, 1),
            "plan_ms": 1000 * self.plan_seconds / max(self.n_control, 1),
            "sim_seconds": self.data.time,
        }

    def save_log(self, path):
        arrays = {k: np.array(v) for k, v in self.log.items()}
        np.savez(path, goal=self.goal, seed=self.seed, predictor=self.predictor_name,
                 patterns=np.array([o.pattern for o in self.obstacles]), **arrays)


def summarize(rows):
    """Aggregate per-episode metrics into one line per (predictor, safety)."""
    out = []
    for key in dict.fromkeys((r["predictor"], r["safety"]) for r in rows):
        sub = [r for r in rows if (r["predictor"], r["safety"]) == key]
        reached = [r["time_to_goal"] for r in sub if r["reached_goal"]]
        out.append({
            "controller": f"{key[0]}{'' if key[1] else ' (no safety)'}",
            "episodes": len(sub),
            "collision_rate": np.mean([r["collided"] for r in sub]),
            "near_miss_rate": np.mean([r["near_miss"] for r in sub]),
            "success_rate": np.mean([r["reached_goal"] for r in sub]),
            "mean_time_to_goal": np.mean(reached) if reached else float("nan"),
            "mean_min_clearance": np.mean([r["min_clearance"] for r in sub]),
            "predict_ms": np.mean([r["predict_ms"] for r in sub]),
            "plan_ms": np.mean([r["plan_ms"] for r in sub]),
        })
    return out


def print_summary(summary):
    print(f"{'controller':<30}{'n':>4}{'collide':>9}{'near miss':>11}{'reached':>9}"
          f"{'time (s)':>10}{'min gap (m)':>13}{'predict ms':>12}{'plan ms':>9}")
    for s in summary:
        print(f"{s['controller']:<30}{s['episodes']:>4}{s['collision_rate']:>9.0%}{s['near_miss_rate']:>11.0%}"
              f"{s['success_rate']:>9.0%}{s['mean_time_to_goal']:>10.2f}{s['mean_min_clearance']:>13.3f}"
              f"{s['predict_ms']:>12.3f}{s['plan_ms']:>9.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--predictors", nargs="+", default=list(PREDICTORS), choices=list(PREDICTORS))
    parser.add_argument("--no-safety", action="store_true", help="disable the safety filter")
    parser.add_argument("--save-logs", action="store_true")
    args = parser.parse_args()

    rows = []
    t0 = time.perf_counter()
    for predictor in args.predictors:
        for seed in range(args.first_seed, args.first_seed + args.episodes):
            ep = Episode(seed, predictor, safety=not args.no_safety)
            rows.append(ep.run())
            if args.save_logs:
                (DATA / "episodes").mkdir(parents=True, exist_ok=True)
                ep.save_log(DATA / "episodes" / f"{predictor}_{seed}.npz")
    print(f"{len(rows)} episodes in {time.perf_counter() - t0:.1f} s wall-clock\n")
    print_summary(summarize(rows))

    DATA.mkdir(exist_ok=True)
    out = DATA / f"metrics{'_nosafety' if args.no_safety else ''}.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-episode metrics: {out}")


if __name__ == "__main__":
    main()
