"""Run navigation episodes and measure them.

One episode = (scenario, seed, predictor). The seed fixes start, goal and obstacle
behaviour; obstacles only react to *where the robot is*, so every predictor faces the
same situation at the start of a given seed.

Predictors (see controller.py): none, static, constant_velocity, and `oracle`.
`oracle` is an evaluation-only upper bound: it simulates a copy of the true obstacles
1 s ahead (with the robot frozen). It reads hidden state and must never be used as a
model or feature source.

Usage:
    python sim.py                                     # warehouse, 50 seeds, all predictors
    python sim.py --scenario open --episodes 20       # Phase 1 arena
    python sim.py --predictors constant_velocity --episodes 50
    python sim.py --no-safety
"""
import argparse
import copy
import csv
import time
from pathlib import Path

import mujoco
import numpy as np

from controller import PREDICTORS, R_SUM, ObstacleTracker, Planner, PlannerConfig, safety_filter
from scenarios import SCENARIOS

HERE = Path(__file__).parent
DATA = HERE / "data"

CONTROL_DT = 0.1      # s, planner + safety rate (10 Hz); physics runs at 500 Hz
OBSTACLE_EVERY = 5    # physics steps per obstacle update (100 Hz)
NEAR_MISS = 0.3       # m: smallest gap between surfaces below this (but >= 0) is a near miss
GOAL_TOL = 0.2        # m
WAYPOINT_TOL = 0.5    # m
STOP_SPEED = 0.05     # m/s: robot slower than this...
STOP_MIN = 0.3        # s: ...for at least this long counts as a stop
SAFE_GAP = 0.2        # m: a stop is "unnecessary" if driving on would have kept at least this gap
ALL_PREDICTORS = list(PREDICTORS) + ["oracle"]


class Episode:
    """Steps physics, obstacles and controller together; records a 10 Hz log and metrics."""

    def __init__(self, seed, predictor="constant_velocity", safety=True, scenario="warehouse",
                 model=None, data=None, predictor_obj=None):
        """Pass `model`/`data` to reuse them (e.g. a live viewer of the same scenario).

        `predictor_obj`: a callable (tracker, times, robot_pos=, t=) -> (n, K, 2), e.g. a
        learned.AsyncPredictor; `predictor` is then only used as its name.
        """
        self.sc = SCENARIOS[scenario](seed)
        self.model = model if model is not None else mujoco.MjModel.from_xml_string(self.sc.xml)
        self.data = data if data is not None else mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        self.seed, self.predictor_name, self.safety, self.scenario = seed, predictor, safety, scenario

        self.goal, self.obstacles = self.sc.goal, self.sc.obstacles
        self.waypoints, self.wp = self.sc.waypoints, 0
        self.data.qpos[:2] = self.sc.start
        self.model.site_pos[self.model.site("goal").id][:2] = self.goal
        self.mocap = [self.model.body_mocapid[self.model.body(f"obs_{i}").id]
                      for i in range(len(self.obstacles))]
        self._place_obstacles()
        mujoco.mj_forward(self.model, self.data)

        self.dt = self.model.opt.timestep
        self.control_every = round(CONTROL_DT / self.dt)
        self.tracker = ObstacleTracker(len(self.obstacles), CONTROL_DT)
        self.planner = Planner(PlannerConfig())
        if predictor_obj is not None:
            self.predict = predictor_obj
        else:
            self.predict = self._predict_oracle if predictor == "oracle" else PREDICTORS[predictor]
        self.cmd = np.zeros(2)
        self.predictions = None
        self.n_steps = 0

        self.min_clearance = np.inf
        self.collision_events = 0
        self._in_collision = np.zeros(len(self.obstacles), dtype=bool)
        self.time_to_goal = None
        self.predict_seconds = 0.0   # wall-clock in the predictor
        self.plan_seconds = 0.0      # wall-clock in planner + safety filter
        self.n_control = 0
        self.log = {k: [] for k in ("t", "robot_pos", "robot_vel", "cmd", "target", "obs_pos",
                                    "obs_mode", "clearance", "pred_05", "pred_10")}

    # -- helpers --
    @property
    def robot_pos(self):
        return self.data.qpos[:2].copy()

    @property
    def target(self):
        return self.waypoints[self.wp]

    @property
    def done(self):
        return self.time_to_goal is not None or self.data.time >= self.sc.max_time

    def observed_obstacles(self):
        return np.array([o.pos for o in self.obstacles])

    def _place_obstacles(self):
        for o, m in zip(self.obstacles, self.mocap):
            self.data.mocap_pos[m] = [o.pos[0], o.pos[1], 0.2]

    def _step_obstacles(self, obstacles, dt, robot_pos):
        for o in obstacles:
            o.step(dt, robot_pos=robot_pos, agvs=obstacles)

    def _predict_oracle(self, tracker, times, sub_dt=0.02, **_):
        """Evaluation-only upper bound: roll a copy of the true obstacles forward."""
        clones = copy.deepcopy(self.obstacles)
        robot = self.robot_pos
        out, t = [], 0.0
        for target_t in times:
            while t < target_t - 1e-9:
                self._step_obstacles(clones, sub_dt, robot)
                t += sub_dt
            out.append([c.pos.copy() for c in clones])
        return np.array(out).transpose(1, 0, 2)

    # -- control (10 Hz) --
    def _control(self):
        robot = self.robot_pos
        if self.wp < len(self.waypoints) - 1 and np.linalg.norm(self.target - robot) < WAYPOINT_TOL:
            self.wp += 1
        obs = self.observed_obstacles()
        self.tracker.update(obs)

        t0 = time.perf_counter()
        self.predictions = self.predict(self.tracker, self.planner.times, robot_pos=robot, t=self.data.time)
        t1 = time.perf_counter()
        cmd = self.planner.plan(robot, self.target, self.predictions, self.sc.free_space,
                                final=self.wp == len(self.waypoints) - 1)
        if self.safety:
            cmd = safety_filter(cmd, robot, obs, v_max=self.planner.cfg.v_max)
        t2 = time.perf_counter()

        self.predict_seconds += t1 - t0
        self.plan_seconds += t2 - t1
        self.n_control += 1
        self.cmd = cmd

        nan = np.full((len(obs), 2), np.nan)
        p = self.predictions
        entries = {
            "t": self.data.time, "robot_pos": robot, "robot_vel": self.data.qvel[:2].copy(), "cmd": cmd,
            "target": self.target.copy(), "obs_pos": obs, "obs_mode": [o.mode for o in self.obstacles],
            "clearance": np.linalg.norm(obs - robot, axis=1) - R_SUM,
            "pred_05": nan if p is None else p[:, 4], "pred_10": nan if p is None else p[:, 9],
        }
        for k, v in entries.items():
            self.log[k].append(v)

    # -- one physics step (2 ms) --
    def step(self):
        if self.n_steps % self.control_every == 0:
            self._control()
        if self.n_steps % OBSTACLE_EVERY == 0:          # obstacles update at 100 Hz
            self._step_obstacles(self.obstacles, self.dt * OBSTACLE_EVERY, self.robot_pos)
            self._place_obstacles()
        self.data.ctrl[:] = self.cmd
        mujoco.mj_step(self.model, self.data)
        self.n_steps += 1

        clearance = np.linalg.norm(self.observed_obstacles() - self.robot_pos, axis=1) - R_SUM
        self.min_clearance = min(self.min_clearance, clearance.min())
        colliding = clearance < 0
        self.collision_events += int((colliding & ~self._in_collision).sum())
        self._in_collision = colliding
        if self.wp == len(self.waypoints) - 1 and np.linalg.norm(self.goal - self.robot_pos) < GOAL_TOL:
            self.time_to_goal = self.data.time

    def run(self):
        t0 = time.perf_counter()
        while not self.done:
            self.step()
        self.wall_seconds = time.perf_counter() - t0
        return self.metrics()

    # -- analysis of the log --
    def arrays(self):
        return {k: np.array(v) for k, v in self.log.items()}

    def stops(self):
        """Stop events: (start index, duration s, unnecessary?) from the 10 Hz log.

        A stop is unnecessary if driving straight on toward the current target at full
        speed for 1 s would have kept >= SAFE_GAP from the obstacles' actual positions
        and stayed in free space. (Approximation: AGVs that yielded to the stopped robot
        might have behaved differently if it had kept moving.)
        """
        a = self.arrays()
        slow = np.linalg.norm(a["robot_vel"], axis=1) < STOP_SPEED
        slow[0] = False                                   # standing at the start is not a stop
        events, k, n = [], 0, len(slow)
        v_max = self.planner.cfg.v_max
        while k < n:
            if not slow[k]:
                k += 1
                continue
            j = k
            while j < n and slow[j]:
                j += 1
            dur = (j - k) * CONTROL_DT
            if dur >= STOP_MIN - 1e-9:
                pos, tgt = a["robot_pos"][k], a["target"][k]
                d = tgt - pos
                reach = min(1.0, np.linalg.norm(d) / v_max)
                steps = np.arange(1, 11)
                ghost = pos + d / max(np.linalg.norm(d), 1e-9) * v_max * np.minimum(steps * CONTROL_DT, reach)[:, None]
                idx = np.minimum(k + steps, n - 1)
                gap = (np.linalg.norm(a["obs_pos"][idx] - ghost[:, None], axis=-1) - R_SUM).min()
                free = True if self.sc.free_space is None else bool(self.sc.free_space(ghost).all())
                events.append((k, dur, bool(gap >= SAFE_GAP and free)))
            k = j
        return events

    def prediction_errors(self, radius=3.0):
        """Mean prediction error (m) at 0.5 s and 1.0 s for obstacles within `radius` of the robot."""
        a = self.arrays()
        out = {}
        for key, h in (("pred_05", 5), ("pred_10", 10)):
            n = len(a["t"]) - h
            if n <= 0 or np.isnan(a[key]).all():
                out[key] = float("nan")
                continue
            err = np.linalg.norm(a[key][:n] - a["obs_pos"][h:h + n], axis=-1)            # (n, n_obs)
            near = np.linalg.norm(a["obs_pos"][:n] - a["robot_pos"][:n, None], axis=-1) < radius
            out[key] = float(err[near].mean()) if near.any() else float("nan")
        return out

    def metrics(self):
        stops = self.stops()
        perr = self.prediction_errors()
        return {
            "scenario": self.scenario,
            "seed": self.seed,
            "predictor": self.predictor_name,
            "safety": self.safety,
            "reached_goal": self.time_to_goal is not None,
            "time_to_goal": self.time_to_goal,
            "collided": self.collision_events > 0,
            "collision_events": self.collision_events,
            "near_miss": bool(0 <= self.min_clearance < NEAR_MISS),
            "min_clearance": float(self.min_clearance),
            "stops": len(stops),
            "unnecessary_stops": sum(s[2] for s in stops),
            "stop_seconds": sum(s[1] for s in stops),
            "pred_err_05": perr["pred_05"],
            "pred_err_10": perr["pred_10"],
            "predict_ms": 1000 * self.predict_seconds / max(self.n_control, 1),
            "plan_ms": 1000 * self.plan_seconds / max(self.n_control, 1),
        }

    def save_log(self, path):
        np.savez(path, goal=self.goal, seed=self.seed, predictor=self.predictor_name,
                 scenario=self.scenario, patterns=np.array([o.pattern for o in self.obstacles]),
                 **self.arrays())


def summarize(rows):
    """One line per (predictor, safety)."""
    out = []
    for key in dict.fromkeys((r["predictor"], r["safety"]) for r in rows):
        sub = [r for r in rows if (r["predictor"], r["safety"]) == key]
        reached = [r["time_to_goal"] for r in sub if r["reached_goal"]]
        mean = lambda k: float(np.nanmean([r[k] for r in sub])) if sub else float("nan")  # noqa: E731
        out.append({
            "controller": f"{key[0]}{'' if key[1] else ' (no safety)'}",
            "episodes": len(sub),
            "collision_rate": mean("collided"),
            "near_miss_rate": mean("near_miss"),
            "success_rate": mean("reached_goal"),
            "mean_time_to_goal": float(np.mean(reached)) if reached else float("nan"),
            "mean_min_clearance": mean("min_clearance"),
            "stops": mean("stops"),
            "unnecessary_stops": mean("unnecessary_stops"),
            "pred_err_10": mean("pred_err_10"),
            "predict_ms": mean("predict_ms"),
            "plan_ms": mean("plan_ms"),
        })
    return out


def print_summary(summary):
    print(f"{'controller':<28}{'n':>4}{'collide':>8}{'near':>7}{'reached':>8}{'time s':>8}"
          f"{'min gap':>9}{'stops':>7}{'unnec.':>8}{'err@1s':>8}{'pred ms':>9}{'plan ms':>9}")
    for s in summary:
        print(f"{s['controller']:<28}{s['episodes']:>4}{s['collision_rate']:>8.0%}{s['near_miss_rate']:>7.0%}"
              f"{s['success_rate']:>8.0%}{s['mean_time_to_goal']:>8.2f}{s['mean_min_clearance']:>9.3f}"
              f"{s['stops']:>7.2f}{s['unnecessary_stops']:>8.2f}{s['pred_err_10']:>8.3f}"
              f"{s['predict_ms']:>9.2f}{s['plan_ms']:>9.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="warehouse", choices=list(SCENARIOS))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--predictors", nargs="+", default=ALL_PREDICTORS, choices=ALL_PREDICTORS)
    parser.add_argument("--no-safety", action="store_true", help="disable the safety filter")
    parser.add_argument("--save-logs", action="store_true")
    parser.add_argument("--out", type=Path, help="per-episode CSV (default data/metrics_<scenario>.csv)")
    args = parser.parse_args()

    rows = []
    t0 = time.perf_counter()
    for predictor in args.predictors:
        for seed in range(args.first_seed, args.first_seed + args.episodes):
            ep = Episode(seed, predictor, safety=not args.no_safety, scenario=args.scenario)
            rows.append(ep.run())
            if args.save_logs:
                (DATA / "episodes").mkdir(parents=True, exist_ok=True)
                ep.save_log(DATA / "episodes" / f"{args.scenario}_{predictor}_{seed}.npz")
    print(f"{len(rows)} episodes in {time.perf_counter() - t0:.1f} s wall-clock\n")
    print_summary(summarize(rows))

    DATA.mkdir(exist_ok=True)
    out = args.out or DATA / f"metrics_{args.scenario}{'_nosafety' if args.no_safety else ''}.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-episode metrics: {out}")


if __name__ == "__main__":
    main()
