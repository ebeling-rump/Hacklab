"""Interactive episode explorer: pick bottle, bottle set and controller with the keyboard *inside*
the MuJoCo viewer and watch the evaluated episode.

    python explorer.py                      # starts on bottle 20002 with TabPFN closed loop
    python explorer.py --seed 10000 --controller classical

Keys (with the viewer window focused; they are also listed in the window):
    Up / Down        next / previous bottle in the current set
    Insert           switch set: in-distribution (40) <-> out-of-distribution slippery caps (20)
    Home / End       previous / next controller
    Enter            pause / resume
    Page Up / Down   faster / slower
    Delete           restart the episode

Episodes are re-simulated from the actions cached in data/results/ (run evaluate.py first).
Physics and sensor noise are seeded, so this reproduces the evaluated episodes exactly and needs
no TabPFN calls. Regrasps (wrist back to 0 deg, fingers open) appear as instant jumps.
"""
import argparse
import json
import threading
import time

import mujoco
import numpy as np
from mujoco import viewer as mj_viewer

from evaluate import RES
from sim import OOD_SEEDS, OPEN_TURN, TEST_SEEDS, BottleSim, sample_bottle

CONTROLLERS = ["classical", "tabpfn_prior_only", "tabpfn", "oracle"]
NAMES = {"classical": "classical adaptive", "tabpfn_prior_only": "TabPFN prior only",
         "tabpfn": "TabPFN closed loop", "oracle": "oracle (knows the bottle)"}
SETS = {"in-distribution": list(TEST_SEEDS), "out-of-distribution (slippery caps)": list(OOD_SEEDS)}
SPEEDS = [0.1, 0.2, 0.33, 0.5, 1.0, 2.0]
FRAME_DT = 0.01                       # sim seconds between recorded frames

# GLFW key codes
K_UP, K_DOWN, K_HOME, K_END, K_INSERT, K_DELETE, K_ENTER, K_KP_ENTER, K_PGUP, K_PGDN = \
    265, 264, 268, 269, 260, 261, 257, 335, 266, 267


def simulate(controller, seed):
    """Re-simulate a cached episode; returns frames, per-frame stroke index and the strokes."""
    cached = json.loads((RES / f"{controller}_{seed}.json").read_text())
    sim = BottleSim(sample_bottle(seed), seed, record=True)
    stroke_of_frame, strokes = [], []
    for o in cached["history"]:
        start = len(sim.frames)
        obs = sim.stroke(o["F"], o["tau"])
        stroke_of_frame += [len(strokes)] * (len(sim.frames) - start)
        strokes.append(obs)
        if obs["opened"]:
            break
    return {"frames": sim.frames, "stroke_of_frame": stroke_of_frame, "strokes": strokes,
            "opened": sim.opened, "time": sim.time, "q_cap": sim.q_cap, "q_bottle": sim.q_bottle,
            "bottle": sample_bottle(seed)}


# Guided tour (--tour): the most instructive episodes, mostly as pairs on the same bottle.
TOUR = [
    (20016, "classical", "Biggest gap (1/2): classical escalates torque step by step, 14 strokes"),
    (20016, "tabpfn", "Biggest gap (2/2): TabPFN on the same bottle, 5 strokes"),
    (20002, "tabpfn_prior_only", "Prior only gets stuck: cap more slippery than any training bottle, slips 37x"),
    (20002, "tabpfn", "Closed loop on the same bottle opens in 5 (but the blind 1st grasp crushes it)"),
    (10019, "tabpfn", "Excessive force: blind first grasp (33 N) above this soft cap's 30 N crush limit"),
    (10025, "classical", "Excessive force, classical: grip escalation overshoots the crush limit on stroke 3"),
    (10037, "classical", "Soft bottle (2.1 Nm/rad): watch the red label wind up before the seal breaks"),
    (20017, "tabpfn", "TabPFN probes: slips at 33 N, works at 83 N, tries 25 N and slips, settles at 75 N"),
]


class Explorer:
    def __init__(self, seed, controller, tour=False):
        self.tour_i = 0 if tour else None
        if tour:
            seed, controller, _ = TOUR[0]
        self.speed = SPEEDS.index(0.33)
        self.paused = False
        self.pending = None            # command from the key thread
        self.lock = threading.Lock()
        self.select(seed, controller)

    def select(self, seed, controller):
        self.set_name = next(n for n, s in SETS.items() if seed in s)
        self.idx = SETS[self.set_name].index(seed)
        self.ctrl = CONTROLLERS.index(controller)
        self.load()

    def caption(self):
        if self.tour_i is None:
            return None
        seed, ctrl, text = TOUR[self.tour_i]
        if (seed, ctrl) != (self.seed, CONTROLLERS[self.ctrl]):
            return None                # user wandered off the tour
        return f"Tour {self.tour_i + 1}/{len(TOUR)}: {text}"

    def tour_step(self, delta):
        self.tour_i = (self.tour_i + delta) % len(TOUR)
        seed, ctrl, _ = TOUR[self.tour_i]
        self.select(seed, ctrl)

    @property
    def seed(self):
        return SETS[self.set_name][self.idx]

    def load(self):
        self.ep = simulate(CONTROLLERS[self.ctrl], self.seed)
        self.frame = 0
        self.hold = 1.0                # seconds to show the first frame

    def key(self, code):
        with self.lock:
            self.pending = code

    def handle(self):
        with self.lock:
            code, self.pending = self.pending, None
        if code is None:
            return
        names = list(SETS)
        if self.tour_i is not None and code in (K_UP, K_DOWN):
            self.tour_step(1 if code == K_UP else -1)
            return
        if code == K_UP:
            self.idx = (self.idx + 1) % len(SETS[self.set_name])
        elif code == K_DOWN:
            self.idx = (self.idx - 1) % len(SETS[self.set_name])
        elif code == K_INSERT:
            self.set_name = names[(names.index(self.set_name) + 1) % len(names)]
            self.idx = min(self.idx, len(SETS[self.set_name]) - 1)
        elif code == K_END:
            self.ctrl = (self.ctrl + 1) % len(CONTROLLERS)
        elif code == K_HOME:
            self.ctrl = (self.ctrl - 1) % len(CONTROLLERS)
        elif code in (K_ENTER, K_KP_ENTER):
            self.paused = not self.paused
            return
        elif code == K_PGUP:
            self.speed = min(self.speed + 1, len(SPEEDS) - 1)
            return
        elif code == K_PGDN:
            self.speed = max(self.speed - 1, 0)
            return
        elif code != K_DELETE:
            return
        self.load()

    def texts(self):
        ep, b = self.ep, self.ep["bottle"]
        q = ep["frames"][self.frame]
        cap = np.rad2deg(q[ep["q_cap"]])
        twist = np.rad2deg(q[ep["q_bottle"]])
        k = ep["stroke_of_frame"][self.frame]
        s = ep["strokes"][k]
        finished = self.frame >= len(ep["frames"]) - 1
        head = (f"Bottle {self.seed}  [{self.idx + 1}/{len(SETS[self.set_name])}]  {self.set_name}\n"
                f"Controller: {NAMES[CONTROLLERS[self.ctrl]]}\n"
                f"Hidden: mu {b.mu:.2f}  breakaway {b.t_break:.2f} Nm  running {b.t_run:.2f} Nm\n"
                f"        wall {b.k_wall:.0f} N/mm  crush at {b.f_crush:.0f} N  torsion {b.k_tors:.1f} Nm/rad\n"
                f"        grip needed at breakaway {b.f_need:.0f} N")
        events = []
        if s["slip"] > 5:
            events.append("SLIP")
        if s["damage"]:
            events.append("DAMAGE (grip over crush limit)")
        if s["seal_broke"]:
            events.append("seal broke")
        live = (f"Stroke {k + 1}/{len(ep['strokes'])}:  grip {s['F']:.0f} N   torque {s['tau']:.2f} Nm\n"
                f"  -> progress {s['progress']:.0f} deg   slip {max(s['slip'], 0):.0f} deg   "
                f"{'  '.join(events)}\n"
                f"Cap {cap:.0f} / {np.rad2deg(OPEN_TURN):.0f} deg   bottle twist {twist:.1f} deg")
        if finished:
            dmg = sum(o["damage"] for o in ep["strokes"])
            live += (f"\n{'OPENED' if ep['opened'] else 'NOT OPENED'} after {len(ep['strokes'])} strokes, "
                     f"{ep['time']:.1f} s simulated, {dmg} damaging stroke(s)")
        state = f"speed x{SPEEDS[self.speed]:g}" + ("   PAUSED" if self.paused else "")
        keys = ("Up/Down  bottle\nInsert  in-dist / slippery set\nHome/End  controller\n"
                "Enter  pause\nPgUp/PgDn  speed\nDelete  restart")
        F = mujoco.mjtFontScale.mjFONTSCALE_150
        G = mujoco.mjtGridPos
        cap_text = self.caption()
        if cap_text:
            head = cap_text + "\n\n" + head
        if self.tour_i is not None:
            keys = keys.replace("Up/Down  bottle", "Up/Down  next / prev tour case")
        return [(F, G.mjGRID_TOPLEFT, head, None),
                (F, G.mjGRID_BOTTOMLEFT, live, None),
                (F, G.mjGRID_TOPRIGHT, state, None),
                (F, G.mjGRID_BOTTOMRIGHT, keys, None)]

    def run(self):
        sim = BottleSim(sample_bottle(self.seed))
        m, d = sim.m, sim.d
        with mj_viewer.launch_passive(m, d, key_callback=self.key, show_left_ui=False) as v:
            v.cam.lookat[:] = [0, 0, 0.22]; v.cam.distance = 0.35; v.cam.elevation = -20; v.cam.azimuth = 120
            last = time.perf_counter()
            acc = 0.0
            while v.is_running():
                self.handle()
                now = time.perf_counter()
                dt, last = now - last, now
                last_frame = len(self.ep["frames"]) - 1
                if self.paused:
                    pass
                elif self.hold > 0:
                    self.hold -= dt
                    if self.hold <= 0 and self.frame >= last_frame:   # result shown: loop
                        if self.caption():                        # tour: next case
                            self.tour_step(1)
                        else:
                            self.frame, self.hold = 0, 1.0
                else:
                    acc += dt * SPEEDS[self.speed]
                    steps, acc = int(acc / FRAME_DT), acc % FRAME_DT
                    self.frame = min(self.frame + steps, last_frame)
                    if self.frame >= last_frame:
                        self.hold = 3.0                               # show the result
                with v.lock():
                    d.qpos[:] = self.ep["frames"][self.frame]
                    mujoco.mj_forward(m, d)
                v.set_texts(self.texts())
                v.sync()
                time.sleep(1 / 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20002)
    ap.add_argument("--controller", default="tabpfn", choices=CONTROLLERS)
    ap.add_argument("--tour", action="store_true", help="guided tour of the most instructive episodes")
    a = ap.parse_args()
    Explorer(a.seed, a.controller, tour=a.tour).run()


if __name__ == "__main__":
    main()
