"""Signature 3D demo: a Unitree A1 learns to manoeuvre an unknown crate through a doorway.

Same learning/planning stack as the planar lab (pushlab): body-frame push-outcome models,
SE(2) cost-to-go field around the walls, depth-2 lookahead. Pushes are executed with the
robot's real body (MuJoCo contacts), so outcomes include gait compliance, bumper geometry,
foot slip etc. that no analytical model describes exactly.

    python -m a1push.episode --model tabpfn_hybrid --seed 0 --video
    python -m a1push.episode --model physics_sysid --seed 0 --video
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import mujoco
import numpy as np

from pushlab.actions import ActionSampler, query_from_actions, concat
from pushlab.learners import make_model
from pushlab.planner import CostField, Planner
from pushlab.shapes import rect, l_shape
from pushlab.sim import body_delta, compose, wrap

from .env import Crate, Scene, build, crate_pose
from .pusher import Navigator, execute_push, ROBOT_R, STANDOFF
from .robot import A1, NOSE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "a1"

DOOR_X = 2.0


def make_scene(seed: int, crate_kind="rect"):
    rng = np.random.default_rng(seed)
    if crate_kind == "rect":
        shape = rect(0.7, 0.3)
    else:
        shape = l_shape(0.7, 0.3, 0.15)  # 0.3 m wide like the rect crate, so it fits the door
    mass = float(rng.uniform(3.0, 6.0))
    mu = float(rng.uniform(0.25, 0.5))
    minx, miny, maxx, maxy = shape.polygon.bounds
    com = (float(rng.uniform(0.15, 0.3) * maxx * rng.choice([-1, 1])), float(rng.uniform(-0.2, 0.2) * maxy))
    crate = Crate([tuple(b) for b in shape.boxes], mass=mass, mu=mu, com=com)
    door = 0.52
    L = 2.0
    t = 0.06
    walls = [(DOOR_X, (door / 2 + L) / 2, t / 2, (L - door / 2) / 2),
             (DOOR_X, -(door / 2 + L) / 2, t / 2, (L - door / 2) / 2),
             # side walls of the room
             (0.9, 2.0, 2.6, t / 2), (0.9, -2.0, 2.6, t / 2)]
    start = (0.45, float(rng.uniform(-0.3, 0.3)), np.pi / 2 + float(rng.uniform(-0.2, 0.2)))
    goal = (3.0, 0.0, 0.0)
    sc = Scene(crate, walls, goal=goal[:2], robot_start=(-0.6, 0.0, 0.0), crate_start=start)
    hidden = dict(mass=mass, mu=mu, com=com)
    return sc, shape, goal, hidden


class Recorder:
    def __init__(self, model, data, fps=30, speed=4.0, size=(640, 360)):
        self.m, self.d = model, data
        self.dt = speed / fps
        self.next_t = 0.0
        w, h = size
        self.r = mujoco.Renderer(model, h, w)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = model.body("trunk").id
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 2.6, -28, 145
        self.top = mujoco.MjvCamera()
        self.top.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.top.lookat[:] = [1.3, 0.0, 0.0]
        self.top.distance, self.top.elevation, self.top.azimuth = 5.2, -89.9, 90
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[3] = 1   # show the (hidden) centre of mass marker for the audience
        self.frames = []
        self.text = ""

    def __call__(self, model, data):
        if data.time < self.next_t:
            return
        self.next_t = data.time + self.dt
        self.r.update_scene(data, self.top, self.opt)
        a = self.r.render().copy()
        self.r.update_scene(data, self.cam, self.opt)
        b = self.r.render().copy()
        self.frames.append((np.concatenate([a, b], axis=1), self.text))

    def save(self, path, title=""):
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw, ImageFont
        try:
            font = ImageFont.truetype("arial.ttf", 20)
            small = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            font = small = ImageFont.load_default()
        w = imageio.get_writer(str(path), fps=30, codec="libx264", quality=7, macro_block_size=8)
        for fr, txt in self.frames:
            im = Image.fromarray(fr)
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, im.width, 30], fill=(15, 20, 30))
            dr.text((10, 5), title, fill=(255, 255, 255), font=font)
            dr.text((10, im.height - 26), txt, fill=(255, 255, 255), font=small, stroke_width=2, stroke_fill=(0, 0, 0))
            w.append_data(np.asarray(im))
        w.close()


def run(model_name: str, seed: int = 0, max_pushes: int = 18, video: bool = False, n_cand: int = 64,
        crate_kind="rect", prior=None, tag=""):
    sc, shape, goal, hidden = make_scene(seed, crate_kind)
    m, d = build(sc)
    robot = A1(m, d)
    bounds = (-0.4, 3.6, -1.7, 1.7)
    nav = Navigator(sc.walls, (-1.0, 4.0, -1.95, 1.95), res=0.025)
    field = CostField(shape, sc.walls, bounds, goal, pos_tol=0.12, ang_tol=None, res=0.05, n_theta=24,
                      margin=0.10, w_theta=0.15)
    planner = Planner(field, walls=sc.walls, lam_u=0.5, w_theta=0.15, step_cost=0.02, depth=2,
                      start_offset=NOSE + STANDOFF, start_margin=ROBOT_R)
    model = make_model(model_name, source=prior)
    desc = shape.descriptor()
    model.reset(shape, desc)
    sampler = ActionSampler(shape, seed=seed)
    rec = Recorder(m, d) if video else None
    gid = m.body("ghost").mocapid[0]
    Qs, Ys, log = [], [], []
    success = False
    falls = 0
    t_wall = time.time()
    for k in range(max_pushes):
        cp = crate_pose(m, d)
        if np.hypot(cp[0] - goal[0], cp[1] - goal[1]) < 0.15:
            success = True
            break
        A = sampler.sample(n_cand, d_range=(0.08, 0.4))
        Q = query_from_actions(shape, A, desc)
        t0 = time.perf_counter()
        mean, std = model.predict(Q)
        # only consider pushes whose pre-push stance the robot can actually walk to
        from .pusher import push_geometry, crate_world_poly
        unreachable = np.zeros(len(A), bool)
        for _ in range(12):
            i, info = planner.choose(cp, Q, mean, std, len(Ys), extra_blocked=unreachable)
            pre, _, _ = push_geometry(shape, cp, A[i])
            if nav.plan(robot.pose, pre, crate_world_poly(shape, cp)) is not None:
                break
            unreachable[i] = True
        t_plan = time.perf_counter() - t0
        pred = compose(cp, mean[i])
        d.mocap_pos[gid] = [pred[0], pred[1], 0.001]
        d.mocap_quat[gid] = [np.cos(pred[2] / 2), 0, 0, np.sin(pred[2] / 2)]
        if rec:
            rec.text = (f"push {k + 1}  |  model: {model_name}  |  experience: {len(Ys)} pushes  |  "
                        f"blue ghost = predicted outcome (+/-{1000 * np.linalg.norm(std[i, :2]):.0f} mm)")
        res = execute_push(m, d, robot, shape, nav, A[i], recorder=rec)
        if not res["ok"]:
            log.append(dict(k=k, ok=False, reason=res.get("reason")))
            print(f"  push {k + 1}: failed ({res.get('reason')})", flush=True)
            if res.get("reason") == "fell":
                falls += 1
                robot.recover()
                if falls > 4:
                    break
            continue
        y = body_delta(res["crate_before"], res["crate_after"])
        Qs.append({k_: v[i:i + 1] for k_, v in Q.items() if isinstance(v, np.ndarray)})
        Ys.append(y)
        model.fit(concat(Qs), np.array(Ys))
        err = np.hypot(*(res["crate_after"][:2] - pred[:2]))
        log.append(dict(k=k, ok=True, action=A[i].tolist(), before=res["crate_before"].tolist(),
                        after=res["crate_after"].tolist(), pred=pred.tolist(), std=std[i].tolist(),
                        pred_err_m=float(err), plan_s=t_plan, sim_t=float(d.time)))
        print(f"  push {k + 1}: crate {np.round(res['crate_after'], 2)}  pred err {1000 * err:.0f} mm  "
              f"ctg {field(res['crate_after'][None])[0]:.2f}", flush=True)
    cp = crate_pose(m, d)
    success = success or np.hypot(cp[0] - goal[0], cp[1] - goal[1]) < 0.15
    out = dict(model=model_name, seed=seed, crate=crate_kind, success=bool(success),
               pushes=len([l for l in log if l.get("ok")]), attempts=len(log), falls=falls,
               final_dist_m=float(np.hypot(cp[0] - goal[0], cp[1] - goal[1])), sim_time_s=float(d.time),
               wall_s=time.time() - t_wall, hidden=hidden,
               mean_pred_err_mm=float(1000 * np.mean([l["pred_err_m"] for l in log if l.get("ok")] or [np.nan])))
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"a1_door_{crate_kind}_{model_name}_s{seed}{tag}"
    with open(OUT / f"{stem}.pkl", "wb") as f:
        pickle.dump(dict(out=out, log=log, walls=sc.walls, goal=goal, shape_boxes=shape.boxes), f)
    if rec:
        title = (f"Unitree A1 learns an unknown crate  |  {model_name}  |  "
                 f"{'SUCCESS' if success else 'not finished'} after {out['pushes']} pushes")
        rec.save(OUT / f"{stem}.mp4", title)
    return out


def load_prior():
    """Prior experience: A1 pushes on 6 other crates (a1push/gen_data.py)."""
    from pushlab.actions import query_from_actions
    with open(ROOT / "data" / "a1_prior.pkl", "rb") as f:
        data = pickle.load(f)
    return [(o["shape"], o["shape"].descriptor(), query_from_actions(o["shape"], o["actions"]), o["Y"]) for o in data]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tabpfn_hybrid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crate", default="rect", choices=["rect", "L"])
    ap.add_argument("--max-pushes", type=int, default=18)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    prior = load_prior() if "transfer" in args.model else None
    out = run(args.model, args.seed, args.max_pushes, args.video, crate_kind=args.crate, prior=prior)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
