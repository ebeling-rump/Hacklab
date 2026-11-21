"""Evaluate (or watch) the scripted expert closed loop on random seeds.

    python -m tabpfn_policy.run_expert --episodes 10 --seed-start 0 [--noise 0.3]
    python -m tabpfn_policy.run_expert --episodes 3 --seed-start 10000 --viewer [--speed 0.5] [--camera wrist]
"""
from __future__ import annotations

import argparse
import time

from .expert import Phase
from .runner import run_expert_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--viewer", action="store_true", help="open the live MuJoCo viewer window")
    ap.add_argument("--speed", type=float, default=1.0, help="viewer: sim seconds per wall second (0 = unpaced)")
    ap.add_argument("--camera", choices=["free", "wrist"], default="free", help="viewer camera")
    ap.add_argument("--contacts", action="store_true", help="viewer: draw contact points/forces")
    ap.add_argument("--hold", type=float, default=1.0, help="viewer: keep the window open this long after each episode")
    a = ap.parse_args()
    viewer = None
    if a.viewer:
        from pickplace.viewer import ViewerOptions
        viewer = ViewerOptions(speed=a.speed, camera=a.camera, contacts=a.contacts, hold=a.hold)
    ok = 0
    for seed in range(a.seed_start, a.seed_start + a.episodes):
        t0 = time.time()
        res, st = run_expert_episode(seed, noise=a.noise, viewer=viewer)
        ok += st["success"]
        print(f"seed {seed:4d}  {'OK  ' if st['success'] else 'FAIL'}  steps={st['steps']:4d} "
              f"sim={st['sim_time']:5.1f}s wall={time.time() - t0:5.1f}s final={Phase(st['final_phase']).name:8s} "
              f"ik_fail={st['ik_failures']} coll={st['collisions']} phases={st['phases']}  {res.per_cube}", flush=True)
    print(f"success {ok}/{a.episodes} = {ok / a.episodes:.0%}")


if __name__ == "__main__":
    main()
