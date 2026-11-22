"""TabPFN collision safety filter: the A1 walks to a goal through boxes it never saw.

Every STEP_TIME seconds the robot proposes a grid of walking commands
(forward x turn), TabPFN predicts P(collision within 0.8 s if this command is
held) for each one from the current lidar scan, and the robot executes the
first 0.2 s of the command closest to
"head straight for the goal" whose risk is below --risk-limit. If nothing is
safe enough it takes the least risky command.

There is no map and no hand-written avoidance rule: the only obstacle knowledge
is the collision model learned from datasets/collision_env_A.npz + _B.npz. The
test scenes (a1_scenarios.py: new static layouts and moving boxes) differ from both.

    python a1_safety_filter.py                          # viewer, random layout, seed 0
    python a1_safety_filter.py --scenario crossing      # moving boxes
    python a1_safety_filter.py --seed 3 --headless
    python a1_safety_filter.py --compare 3 --scenario all   # every scene x 3 seeds x 3 controllers
    python a1_safety_filter.py --scenario wall --planner    # steer to A* waypoints (path in green)

Note: TabPFN on CPU takes ~0.3 s per decision, so the viewer runs slower than
real time (physics simply waits while the model thinks).
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from a1_collision import CollisionRisk, features, load_table
from a1_lidar import (GOAL, MAX_TURN, SMOOTHING, STAND, Navigator, gait, read_lidar,
                      wrap, yaw)
from a1_planner import Planner
from a1_scenarios import DYNAMIC, SCENARIOS, make_scene

warnings.filterwarnings("ignore")

STEP_TIME = 0.2                      # decision period, same as the training data
FORWARDS = np.array([0.0, 0.5, 1.0])   # 0 = trot in place / turn on the spot
TURNS = np.linspace(-MAX_TURN, MAX_TURN, 7)
CANDIDATES = np.array([(f, t) for f in FORWARDS for t in TURNS])
GOAL_RADIUS = 0.3
TIME_LIMIT = 60.0                    # simulated seconds per run
FALL_HEIGHT = 0.15


def train_risk_model() -> CollisionRisk:
    tables = [load_table(env) for env in ("env_A", "env_B")]
    X = np.vstack([t[0] for t in tables])
    y = np.concatenate([t[1] for t in tables])
    t = time.time()
    risk = CollisionRisk().fit(X, y)
    print(f"TabPFN collision model ready ({time.time() - t:.0f}s, context from "
          f"{len(y)} recorded samples, collision rate {y.mean():.1%})")
    return risk


def goal_command(data, target=GOAL):
    """What a robot without any obstacle sensing would do: turn to the target and go."""
    to_goal = target - data.qpos[:2]
    goal_dir = wrap(np.arctan2(to_goal[1], to_goal[0]) - yaw(data))
    return np.array([max(0.3, np.cos(goal_dir)), np.clip(1.5 * goal_dir, -MAX_TURN, MAX_TURN)])


def nearest_goal_candidate(want):
    return int(np.argmin(np.abs(CANDIDATES - want).sum(1)))


FORWARD_WEIGHT = 1.0   # cost of going slower than wanted; > turn cost so it never just stands
DETOUR_STEPS = 10      # once blocked, keep turning to the same side for 2 s
DETOUR_PENALTY = 0.6   # extra cost for turning towards the other side during a detour


class SafetyFilter:
    """Pick the safe candidate closest to the goal command.

    Without memory the robot dithers left/right in front of a box (each side
    looks equally good every 0.2 s), so when the goal command is unsafe it
    commits to one side for DETOUR_STEPS decisions.
    """

    def __init__(self, risk: CollisionRisk, risk_limit: float):
        self.risk, self.limit = risk, risk_limit
        self.prev = np.zeros(2)
        self.last_risk = None
        self.side, self.detour_left = 0.0, 0

    def __call__(self, data, ranges, target=GOAL):
        want = goal_command(data, target)
        p = self.risk.predict(features(ranges, CANDIDATES, self.prev))
        cost = (np.abs(CANDIDATES[:, 1] - want[1])
                + FORWARD_WEIGHT * np.abs(CANDIDATES[:, 0] - want[0]))
        if self.detour_left > 0:
            cost += DETOUR_PENALTY * (np.sign(CANDIDATES[:, 1]) == -self.side)
            self.detour_left -= 1
        safe = p < self.limit
        i = np.argmin(np.where(safe, cost, np.inf)) if safe.any() else np.argmin(p)
        j = nearest_goal_candidate(want)
        if p[j] >= self.limit and self.detour_left == 0 and CANDIDATES[i, 1] != 0:
            self.side, self.detour_left = np.sign(CANDIDATES[i, 1]), DETOUR_STEPS
        self.last_risk = (p[j], p[i], safe.any())
        self.prev = CANDIDATES[i]
        return CANDIDATES[i]


def touching_box(model, data, obstacles) -> bool:
    """Robot in contact with a static box/wall or a moving box."""
    for c in data.contact[:data.ncon]:
        for g, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            if g in obstacles and other not in obstacles and model.geom_bodyid[other] != 0:
                return True
    return False


def draw_path(v, path):
    """Planned path as small green spheres in the viewer."""
    pts = path[::3][:v.user_scn.maxgeom]
    for k, (x, y) in enumerate(pts):
        mujoco.mjv_initGeom(v.user_scn.geoms[k], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.04, 0, 0]), np.array([x, y, 0.05]),
                            np.eye(3).flatten(), np.array([0.2, 0.9, 0.3, 0.8], np.float32))
    v.user_scn.ngeom = len(pts)


def run(seed, controller, risk=None, risk_limit=0.15, viewer=False, verbose=True,
        scenario="random", planner=False):
    """One episode. controller in {"tabpfn", "none", "gap"}; planner=True steers
    every controller towards A* waypoints instead of the goal. Returns a result dict."""
    scene = make_scene(scenario, seed)
    model = scene.model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    scene.update(data)
    mujoco.mj_forward(model, data)
    boxes = scene.obstacle_geoms()
    substeps = int(round(STEP_TIME / model.opt.timestep))
    alpha = model.opt.timestep / SMOOTHING

    policy = (SafetyFilter(risk, risk_limit) if controller == "tabpfn" else
              Navigator(model.opt.timestep) if controller == "gap" else None)
    planner_ = Planner() if planner else None
    cmd = np.zeros(2)
    res = dict(scenario=scenario, seed=seed, controller=controller + ("+plan" if planner else ""),
               reached=False, fell=False, collision_steps=0, decisions=0, time=0.0)

    def decide():
        ranges = read_lidar(data)
        target = planner_.waypoint(data, ranges) if planner_ else GOAL
        if controller == "tabpfn":
            return policy(data, ranges, target)
        if controller == "none":
            return goal_command(data, target)
        policy.goal = target
        return None  # gap navigator decides every physics step instead

    def control_step(v=None):
        nonlocal cmd
        action = decide()
        if v is not None and planner_ is not None:
            with v.lock():
                draw_path(v, planner_.path)
        hit = False
        for _ in range(substeps):
            if controller == "gap":
                data.ctrl[:] = gait(data.time, *policy(data, read_lidar(data)))
            else:
                cmd += alpha * (action - cmd)
                data.ctrl[:] = gait(data.time, *cmd)
            scene.update(data)
            mujoco.mj_step(model, data)
            hit |= touching_box(model, data, boxes)
            if v is not None:
                v.sync()
        res["decisions"] += 1
        res["collision_steps"] += hit
        if verbose and (hit or res["decisions"] % 10 == 0):
            extra = ""
            if controller == "tabpfn":
                naive, chosen, any_safe = policy.last_risk
                extra = (f" risk(goal-cmd)={naive:.2f} risk(chosen)={chosen:.2f}"
                         f"{'' if any_safe else ' [no safe command]'}")
            print(f"t={data.time:5.1f}s pos=({data.qpos[0]:.2f},{data.qpos[1]:.2f})"
                  f"{' COLLISION' if hit else ''}{extra}")

    def finished():
        if np.linalg.norm(GOAL - data.qpos[:2]) < GOAL_RADIUS:
            res["reached"] = True
        if data.qpos[2] < FALL_HEIGHT:
            res["fell"] = True
        return res["reached"] or res["fell"] or data.time > TIME_LIMIT

    if viewer:
        with mujoco.viewer.launch_passive(model, data) as v:
            with v.lock():
                v.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True
                v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                v.cam.trackbodyid = model.body("trunk").id
                v.cam.distance, v.cam.elevation = 3.0, -35
            while v.is_running() and not finished():
                control_step(v)
            data.ctrl[:] = np.tile(STAND, 4)
            while v.is_running():   # keep the window open at the end
                scene.update(data)
                mujoco.mj_step(model, data)
                v.sync()
                time.sleep(model.opt.timestep)
    else:
        while not finished():
            control_step()

    res["time"] = round(data.time, 1)
    print(f"[{scenario} {res['controller']} seed {seed}] reached goal: {res['reached']}, "
          f"fell: {res['fell']}, collision steps: {res['collision_steps']}/{res['decisions']}, "
          f"t={res['time']}s", flush=True)
    return res


CONTROLLERS = ("none", "tabpfn", "gap")


def summarise(results, scenarios):
    print(f"\n{'scenario':10s}{'controller':11s}{'reached':>9s}{'fell':>6s}"
          f"{'collision steps':>17s}{'mean time':>11s}")
    controllers = list(dict.fromkeys(r["controller"] for r in results))
    for sc in scenarios:
        for c in controllers:
            rs = [r for r in results if r["scenario"] == sc and r["controller"] == c]
            if not rs:
                continue
            tag = " (dynamic)" if sc in DYNAMIC and c == controllers[0] else ""
            print(f"{sc:10s}{c:11s}{sum(r['reached'] for r in rs):>6d}/{len(rs)}"
                  f"{sum(r['fell'] for r in rs):>6d}{sum(r['collision_steps'] for r in rs):>17d}"
                  f"{np.mean([r['time'] for r in rs]):>10.1f}s{tag}")
    print("collision steps = 0.2 s steps with the robot touching any box, wall or mover")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenario", choices=SCENARIOS + ["all"], default="random",
                    help="test scene, see a1_scenarios.py ('all' only with --compare)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--controller", choices=CONTROLLERS, default="tabpfn")
    ap.add_argument("--risk-limit", type=float, default=0.15)
    ap.add_argument("--planner", action="store_true",
                    help="steer towards A* waypoints from a lidar map (a1_planner.py)")
    ap.add_argument("--compare", type=int, default=0,
                    help="run seeds 0..N-1 headless with every controller and summarise")
    ap.add_argument("--controllers", nargs="+", choices=CONTROLLERS, default=list(CONTROLLERS),
                    help="controllers to include in --compare")
    ap.add_argument("--save", help="with --compare: write all results to this JSON file")
    args = ap.parse_args()

    if args.compare:
        scenarios = SCENARIOS if args.scenario == "all" else [args.scenario]
        risk = train_risk_model() if "tabpfn" in args.controllers else None
        results = [run(s, c, risk, args.risk_limit, verbose=False, scenario=sc,
                       planner=args.planner)
                   for sc in scenarios for s in range(args.compare) for c in args.controllers]
        summarise(results, scenarios)
        if args.save:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save).write_text(json.dumps(results, indent=1, default=float))
        return

    if args.scenario == "all":
        ap.error("--scenario all needs --compare N")
    risk = train_risk_model() if args.controller == "tabpfn" else None
    run(args.seed, args.controller, risk, args.risk_limit, viewer=not args.headless,
        scenario=args.scenario, planner=args.planner)


if __name__ == "__main__":
    main()
