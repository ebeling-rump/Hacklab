"""MuJoCo bottle-cap simulation: hidden bottle properties, the low-level safety controller and
one *twist stroke* as the robot's unit action.

A stroke (grip force F [N], wrist torque tau [Nm]) is executed by the low-level controller:
  1. close the fingers, ramping the grip force to F             (0.08 s ramp + 0.04 s settle)
  2. ramp the wrist torque to tau and hold                       (0.10 s ramp, up to 0.35 s total)
     until the wrist has turned STROKE_ANGLE (90 deg) or the safety layer stops it
  3. remove the torque, open the fingers, let the bottle unwind  (0.10 s)
  4. regrasp: wrist back to 0 with the fingers open              (kinematic, costs REGRASP_TIME)

The safety layer (SafetyController) is identical for every high-level controller and runs at the
physics rate. It knows only generic limits, never the bottle's hidden properties.

Observations after a stroke are what the robot could sense: cap progress relative to the bottle
(vision), slip between wrist and cap, peak grip force and wall deflection (finger encoders),
peak bottle twist (vision), and a damage flag (audible crack / visible deformation).
"""
from dataclasses import dataclass, asdict
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).parent
XML = HERE / "scene.xml"

CAP_R = 0.016                 # m
OPEN_TURN = 1.25 * 2 * np.pi  # cap rotation (relative to the bottle) at which the cap comes off
STROKE_ANGLE = np.deg2rad(90)
REGRASP_TIME = 0.15           # s
F_MAX, TAU_MAX = 100.0, 2.0   # generic hardware / safety limits
SEAL_BREAK_SPEED = 0.4        # rad/s of cap relative motion = seal has broken ...
SEAL_BREAK_ANGLE = np.deg2rad(1.5)   # ... or this much cap motion within one stroke
PRESHAPE = 0.0017             # m finger travel before a grasp: pads 0.3 mm from the cap

# ---------------------------------------------------------------- bottles


@dataclass
class Bottle:
    mu: float          # cap-pad friction coefficient
    t_break: float     # Nm, breakaway (seal) torque
    t_run: float       # Nm, running torque once the seal has broken
    k_wall: float      # N/mm, radial stiffness of cap/neck under the fingers
    d_lim: float       # mm, wall deflection at which the cap is damaged (crush)
    k_tors: float      # Nm/rad, torsional stiffness of the bottle body

    @property
    def f_crush(self):
        return self.k_wall * self.d_lim

    @property
    def f_need(self):
        """Grip force at which the fingers can just transmit the breakaway torque."""
        return self.t_break / (2 * self.mu * CAP_R)


def sample_bottle(seed):
    """Random but always openable bottle (enough margin between needed and crushing grip force).

    Seeds in OOD_SEEDS give *out-of-distribution* bottles: slippery caps (mu 0.22-0.4, below the
    training range) with lower breakaway torques (0.25-0.9 Nm), so the grip needed is high
    relative to the crush limit and the prior from training bottles is systematically wrong.
    """
    rng = np.random.default_rng(seed)
    ood = seed in OOD_SEEDS
    while True:
        mu = rng.uniform(0.22, 0.4) if ood else rng.uniform(0.4, 1.1)
        t_break = rng.uniform(0.25, 0.9) if ood else rng.uniform(0.3, 1.6)
        t_run = t_break * rng.uniform(0.15, 0.45)
        k_wall = rng.uniform(15, 60)
        # thinner walls are softer *and* give way earlier, with scatter
        d_lim = float(np.clip(0.8 + 0.025 * k_wall + rng.normal(0, 0.2), 0.9, 2.6))
        k_tors = rng.uniform(2, 12)
        b = Bottle(mu, t_break, t_run, k_wall, d_lim, k_tors)
        if b.f_need * 1.2 < 0.85 * b.f_crush and b.f_need * 1.2 < F_MAX and t_break * 1.1 < TAU_MAX:
            return b


TRAIN_SEEDS = range(0, 150)
TEST_SEEDS = range(10_000, 10_040)    # unseen bottles, never used for prior data
OOD_SEEDS = range(20_000, 20_020)     # unseen *and* outside the training distribution

# ---------------------------------------------------------------- safety layer


class SafetyController:
    """Generic low-level protection, the same for every controller. Runs every physics step.

    - clamps commands to the hardware limits F_MAX / TAU_MAX and rate-limits them (ramps)
    - slip watchdog: if the wrist spins > 4 rad/s faster than the cap for 20 ms, cut the torque
    - twist watchdog: bottle twisted > 50 deg -> cut the torque (protects bottle and fixture)
    - deflection watchdog at a *generic* 2.6 mm (largest crush limit of any bottle design):
      release the grip. It bounds the severity of a crush but cannot prevent bottle-specific
      damage below that, which is the high-level controller's job.
    """
    SLIP_SPEED, SLIP_TIME = 4.0, 0.02
    TWIST_LIMIT = np.deg2rad(50)
    DEFLECTION_LIMIT = 2.6

    def __init__(self):
        self.slip_timer = 0.0
        self.torque_cut = False
        self.grip_cut = False
        self.events = set()

    def filter(self, F, tau, slip_speed, twist, deflection, dt):
        F = float(np.clip(F, 0.0, F_MAX))
        tau = float(np.clip(tau, 0.0, TAU_MAX))
        self.slip_timer = self.slip_timer + dt if slip_speed > self.SLIP_SPEED else 0.0
        if self.slip_timer > self.SLIP_TIME:
            self.torque_cut = True
            self.events.add("slip_stop")
        if abs(twist) > self.TWIST_LIMIT:
            self.torque_cut = True
            self.events.add("twist_stop")
        if deflection > self.DEFLECTION_LIMIT:
            self.grip_cut = self.torque_cut = True
            self.events.add("deflection_stop")
        return (0.0 if self.grip_cut else F), (0.0 if self.torque_cut else tau)

# ---------------------------------------------------------------- simulation


class BottleSim:
    def __init__(self, bottle: Bottle, seed=0, record=False):
        self.m = mujoco.MjModel.from_xml_path(str(XML))
        self.d = mujoco.MjData(self.m)
        self.b = bottle
        self.rng = np.random.default_rng(seed + 7919)
        self.record = record
        self.frames = []          # qpos snapshots for replay (if record)
        j = lambda n: self.m.joint(n)
        self.q_bottle, self.q_cap, self.q_wrist = (j(n).qposadr[0] for n in ("bottle_twist", "cap_turn", "wrist"))
        self.v_bottle, self.v_cap, self.v_wrist = (j(n).dofadr[0] for n in ("bottle_twist", "cap_turn", "wrist"))
        self.q_lift, self.q_gz = j("cap_lift").qposadr[0], j("grip_z").qposadr[0]
        self.q_fl, self.q_fr = j("finger_l").qposadr[0], j("finger_r").qposadr[0]
        self.v_gz = j("grip_z").dofadr[0]
        self.cap_geom = self.m.geom("cap").id
        self.pads = {self.m.geom("pad_l").id, self.m.geom("pad_r").id}
        # per-bottle physics
        self.m.geom_friction[self.cap_geom, 0] = bottle.mu
        self.m.jnt_stiffness[j("bottle_twist").id] = bottle.k_tors
        self.seal_broken = False
        self.m.dof_frictionloss[self.v_cap] = bottle.t_break
        self.time = 0.0
        self.damaged = False
        self.d.qpos[self.q_fl] = self.d.qpos[self.q_fr] = PRESHAPE
        self.stroke_cap0 = 0.0
        mujoco.mj_forward(self.m, self.d)

    # ---- sensing
    def grip_normal_force(self):
        """Mean normal force of the two pads on the cap (finger force sensing)."""
        f6 = np.zeros(6)
        per_pad = {}
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g = {c.geom1, c.geom2}
            if self.cap_geom in g and g & self.pads:
                mujoco.mj_contactForce(self.m, self.d, i, f6)
                pad = (g & self.pads).pop()
                per_pad[pad] = per_pad.get(pad, 0.0) + f6[0]
        return float(np.mean(list(per_pad.values()))) if len(per_pad) == 2 else 0.0

    def cap_rel(self):
        return float(self.d.qpos[self.q_cap])

    @property
    def opened(self):
        return self.cap_rel() >= OPEN_TURN

    def _step(self, F, tau, safety, peaks):
        d = self.d
        fn = self.grip_normal_force()
        defl = fn / self.b.k_wall                                    # mm (analytic wall model)
        slip_speed = d.qvel[self.v_wrist] - (d.qvel[self.v_cap] + d.qvel[self.v_bottle])
        twist = d.qpos[self.q_bottle]
        F, tau = safety.filter(F, tau, slip_speed, twist, defl, self.m.opt.timestep)
        d.ctrl[:] = [F, F, tau]
        mujoco.mj_step(self.m, d)
        self.time += self.m.opt.timestep
        # seal: dry friction drops from breakaway to running torque once the cap moves
        if not self.seal_broken and (abs(d.qvel[self.v_cap]) > SEAL_BREAK_SPEED or
                                     abs(d.qpos[self.q_cap] - self.stroke_cap0) > SEAL_BREAK_ANGLE):
            self.seal_broken = True
            self.m.dof_frictionloss[self.v_cap] = self.b.t_run
        peaks["fn"] = max(peaks["fn"], fn)
        peaks["twist"] = max(peaks["twist"], abs(twist))
        if defl > self.b.d_lim:
            self.damaged = True
            peaks["damage"] = 1
        if self.record and int(round(self.time / self.m.opt.timestep)) % 20 == 0:
            self.frames.append(d.qpos.copy())

    def stroke(self, F, tau):
        """Execute one twist stroke. Returns the observation dict."""
        dt = self.m.opt.timestep
        d = self.d
        safety = SafetyController()
        peaks = {"fn": 0.0, "twist": 0.0, "damage": 0}
        cap0, wrist0 = self.cap_rel(), d.qpos[self.q_wrist]
        capw0 = d.qpos[self.q_cap] + d.qpos[self.q_bottle]
        seal0 = self.seal_broken
        t0 = self.time
        self.stroke_cap0 = self.cap_rel()
        # 1. close
        for k in range(int(0.04 / dt)):          # approach with a light contact force
            self._step(min(F, 3.0), 0.0, safety, peaks)
        for k in range(int(0.08 / dt)):          # ramp to the commanded grip force
            self._step(min(F, 3.0) + max(F - 3.0, 0.0) * min(1.0, k * dt / 0.07), 0.0, safety, peaks)
        # 2. twist
        n_twist = int(0.35 / dt)
        for k in range(n_twist):
            self._step(F, tau * min(1.0, k * dt / 0.10), safety, peaks)
            if d.qpos[self.q_wrist] - wrist0 >= STROKE_ANGLE or self.opened:
                break
        dwrist = d.qpos[self.q_wrist] - wrist0
        dcapw = d.qpos[self.q_cap] + d.qpos[self.q_bottle] - capw0
        # 3. release torque then grip, let the bottle unwind
        for k in range(int(0.04 / dt)):
            self._step(F, 0.0, safety, peaks)
        for k in range(int(0.06 / dt)):
            self._step(-5.0, 0.0, safety, peaks)    # small opening force
        # 4. regrasp (kinematic): wrist back to 0, fingers open, gripper follows the cap height
        d.qpos[self.q_wrist] = 0.0
        d.qpos[self.q_fl] = d.qpos[self.q_fr] = PRESHAPE
        d.qpos[self.q_gz] = d.qpos[self.q_lift]
        d.qvel[self.v_wrist] = d.qvel[self.v_gz] = 0.0
        d.qvel[self.m.joint("finger_l").dofadr[0]] = d.qvel[self.m.joint("finger_r").dofadr[0]] = 0.0
        mujoco.mj_forward(self.m, d)
        self.time += REGRASP_TIME
        noise = lambda s: float(self.rng.normal(0, s))
        progress = self.cap_rel() - cap0
        return {
            "F": float(F), "tau": float(tau),
            "progress": np.rad2deg(progress) + noise(0.5),          # deg, vision
            "slip": np.rad2deg(max(dwrist - dcapw, 0.0)) + noise(0.5),
            "wrist": np.rad2deg(dwrist),
            "fn": peaks["fn"] + noise(0.5),
            "deflection": peaks["fn"] / self.b.k_wall + noise(0.03),   # mm
            "twist": np.rad2deg(peaks["twist"]) + noise(0.5),
            "damage": int(peaks["damage"]),
            "seal_broke": int(self.seal_broken and not seal0),
            "safety": sorted(safety.events),
            "duration": self.time - t0,
            "opened": bool(self.opened),
        }


MAX_STROKES = 40


def run_episode(bottle, controller, seed=0, max_strokes=MAX_STROKES, record=False, verbose=False):
    """Closed loop: controller.act(history) -> (F, tau) -> stroke -> observation -> history."""
    sim = BottleSim(bottle, seed, record=record)
    controller.reset()
    history = []
    for _ in range(max_strokes):
        F, tau = controller.act(history)
        obs = sim.stroke(F, tau)
        history.append(obs)
        if verbose:
            print(f"  F={F:6.1f} tau={tau:4.2f} -> prog {obs['progress']:6.1f}  slip {obs['slip']:5.1f}  "
                  f"defl {obs['deflection']:4.2f}  twist {obs['twist']:5.1f}  dmg {obs['damage']} {obs['safety']}")
        if obs["opened"]:
            break
    return {"opened": sim.opened, "time": sim.time, "history": history, "bottle": asdict(bottle),
            "frames": sim.frames if record else None}
