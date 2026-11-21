"""Run the closed-loop TabPFN student (and the teacher on the same seeds for comparison).

    python -m tabpfn_policy.run_policy --seed-start 10000 --episodes 20
    python -m tabpfn_policy.run_policy --seed-start 10000 --episodes 1 --viewer --no-teacher

Per control step (20 Hz, the simulation waits for the policy):
  camera frame -> CubeTracker belief -> perceived state -> phase_perc = Expert.next_phase(perceived, phase_perc)
  -> features.feature_vector -> TabPFNStudent.act -> SimEpisode.apply (IK -> joint targets)
The student never sees privileged state. For DIAGNOSTICS ONLY, the logger also runs the teacher on the true
state each step (true phase + the teacher's action as a label); nothing from it reaches the student.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .expert import Expert, Phase
from .features import CubeTracker, feature_vector, perceived_state
from .runner import SimEpisode, make_scene, run_expert_episode

ROOT = Path(__file__).resolve().parents[1]
HELD = {int(Phase.LIFT), int(Phase.CARRY), int(Phase.LOWER)}


def count_drops(phases: list[int]) -> int:
    """Times the phase sequence went from holding the cube straight back to APPROACH."""
    return sum(1 for a, b in zip(phases, phases[1:]) if a in HELD and b == int(Phase.APPROACH))


def compress(seq):
    out = []
    for p in seq:
        if not out or out[-1] != p:
            out.append(int(p))
    return out


def run_student_episode(student, seed: int, max_steps: int, viewer=None, log: list | None = None):
    expert = Expert()                    # used for its phase rules on the PERCEIVED state, and for diagnostics
    ep = SimEpisode(make_scene(seed), viewer=viewer)
    tracker = CubeTracker(ep.robot, ep.cfg.cubes[0].color)
    phase_p, phase_true = Phase.APPROACH, Phase.APPROACH
    seq_p, seq_true, lat = [], [], []
    step = 0
    try:
        for step in range(max_steps):
            p, _ = ep.tcp()
            belief = tracker.update(ep.robot.get_observation(), p)
            s_perc = perceived_state(ep, belief)
            phase_p = expert.next_phase(s_perc, phase_p)                     # student-side phase (no privilege)
            label, phase_true = expert.act(ep.true_state(), phase_true)      # diagnostics only
            seq_p.append(int(phase_p))
            seq_true.append(int(phase_true))
            if phase_p == Phase.DONE:
                break
            feats = feature_vector(s_perc, belief)
            a, ms = student.act(feats, int(phase_p))
            lat.append(ms)
            info = ep.apply(a)
            if log is not None:
                log.append(dict(seed=seed, step=step, t=step * ep.dt, phase_perc=int(phase_p),
                                phase_true=int(phase_true), **feats,
                                a_dx=a[0] * 1e3, a_dy=a[1] * 1e3, a_dz=a[2] * 1e3, a_dyaw=a[3], a_grip=a[4] * 1e3,
                                teacher_dx=label[0] * 1e3, teacher_dy=label[1] * 1e3, teacher_dz=label[2] * 1e3,
                                teacher_dyaw=label[3], teacher_grip=label[4] * 1e3, latency_ms=ms, ik_ok=info.ik_ok))
    finally:
        res = ep.finish()
    return dict(seed=seed, agent="student", success=res.n_placed == res.n_targets, steps=step,
                sim_s=step * ep.dt, final_phase=Phase(seq_p[-1]).name, final_phase_true=Phase(seq_true[-1]).name,
                phases=compress(seq_p), drops=count_drops(compress(seq_true)), ik_failures=ep.ik_failures,
                collisions=res.collision_steps, lat_mean_ms=float(np.mean(lat)) if lat else np.nan,
                lat_p95_ms=float(np.percentile(lat, 95)) if lat else np.nan, per_cube=res.per_cube)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "teacher.parquet"))
    ap.add_argument("--seed-start", type=int, default=10000)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--m", type=int, default=256, help="context states per phase")
    ap.add_argument("--model-version", default="v3.5-fast")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-teacher", action="store_true", help="skip the teacher comparison run")
    ap.add_argument("--viewer", action="store_true", help="watch the student episodes in the MuJoCo viewer")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--camera", choices=["free", "wrist"], default="free")
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()

    import os
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    import torch
    torch.set_num_threads(a.threads)
    from .student_policy import TabPFNStudent

    viewer = None
    if a.viewer:
        from pickplace.viewer import ViewerOptions
        viewer = ViewerOptions(speed=a.speed, camera=a.camera, hold=1.0)

    train = pd.read_parquet(a.data)
    held_out = set(range(a.seed_start, a.seed_start + a.episodes))
    assert not held_out & set(train.seed.unique()), "evaluation seeds overlap the training data"
    t0 = time.time()
    student = TabPFNStudent(train, m=a.m, version=a.model_version)
    print(f"student: {a.model_version}, {len(student.reg)} phase contexts "
          f"({', '.join(f'{Phase(p).name}:{n}' for p, n in student.context_size.items())}), "
          f"gripper classifier in {[Phase(p).name for p in student.grip_model] or 'no'} phases, "
          f"fit {student.fit_seconds:.1f}s", flush=True)

    rows, log = [], []
    for seed in sorted(held_out):
        t = time.time()
        r = run_student_episode(student, seed, a.max_steps, viewer, log)
        rows.append(r)
        print(f"student seed {seed}  {'OK  ' if r['success'] else 'FAIL'}  {r['sim_s']:5.1f}s sim  {time.time() - t:5.1f}s wall  "
              f"final {r['final_phase']:8s} drops {r['drops']} ikf {r['ik_failures']} coll {r['collisions']} "
              f"lat {r['lat_mean_ms']:.0f}/{r['lat_p95_ms']:.0f} ms  phases {r['phases']}", flush=True)
        if not a.no_teacher:
            res, st = run_expert_episode(seed)
            rows.append(dict(seed=seed, agent="teacher", success=st["success"], steps=st["steps"], sim_s=st["sim_time"],
                             final_phase=Phase(st["final_phase"]).name, final_phase_true=Phase(st["final_phase"]).name,
                             phases=st["phases"], drops=count_drops(st["phases"]), ik_failures=st["ik_failures"],
                             collisions=st["collisions"], per_cube=res.per_cube))
            print(f"teacher seed {seed}  {'OK  ' if st['success'] else 'FAIL'}  {st['sim_time']:5.1f}s sim  "
                  f"final {Phase(st['final_phase']).name:8s} drops {rows[-1]['drops']} ikf {st['ik_failures']}", flush=True)

    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    df.to_csv(out / f"policy_eval_{stamp}.csv", index=False)
    pd.DataFrame(log).to_csv(out / f"policy_steps_{stamp}.csv", index=False)

    print(f"\n=== held-out seeds {min(held_out)}-{max(held_out)} ({time.time() - t0:.0f}s wall) ===")
    for agent, g in df.groupby("agent", sort=False):
        ok = g[g.success]
        print(f"{agent:8s} success {g.success.sum()}/{len(g)}  duration of successes: mean {ok.sim_s.mean():.1f}s "
              f"median {ok.sim_s.median():.1f}s  drops {int(g.drops.sum())}  IK failures {int(g.ik_failures.sum())}  "
              f"collision steps {int(g.collisions.sum())}")
        fails = g[~g.success]
        if len(fails):
            print(f"         failures end in phase: {fails.final_phase.value_counts().to_dict()} "
                  f"(true-state phase: {fails.final_phase_true.value_counts().to_dict()})")
    st = df[df.agent == "student"]
    if st.lat_mean_ms.notna().any():
        steps = pd.DataFrame(log)
        print(f"student latency per step: mean {steps.latency_ms.mean():.0f} ms, p95 {steps.latency_ms.quantile(.95):.0f} ms, "
              f"max {steps.latency_ms.max():.0f} ms")
        agree = (steps.phase_perc == steps.phase_true).mean()
        print(f"student perceived phase == true phase on {agree:.0%} of steps")
    print(f"saved {out / f'policy_eval_{stamp}.csv'} and {out / f'policy_steps_{stamp}.csv'}")


if __name__ == "__main__":
    main()
