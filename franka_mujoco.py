"""Franka Emika Panda plays tic-tac-toe in MuJoCo.

The Panda model comes from MuJoCo Menagerie (cloned automatically on first run).
A table with a 3x3 board and two rows of pieces (red X, blue O) is added to the
scene. The robot picks its piece from its supply row and places it on the board
using inverse kinematics; its moves come from a minimax search (it never loses).

    pip install mujoco
    python franka_mujoco.py                     # you (X) vs robot (O), viewer
    python franka_mujoco.py --robot-plays X     # robot opens the game
    python franka_mujoco.py --mode selfplay     # robot plays both sides
    python franka_mujoco.py --headless          # self-play, no window

In human mode, type a cell number 1-9 in the terminal (board as seen from the
initial camera, i.e. standing opposite the robot), 'r' to restart, 'q' to quit.
"""

import argparse
import queue
import random
import subprocess
import sys
import threading
import time
from collections import deque
from functools import lru_cache
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

MENAGERIE_DIR = Path(__file__).parent / "mujoco_menagerie"
SCENE_XML = MENAGERIE_DIR / "franka_emika_panda" / "scene.xml"

# --- Scene geometry (metres) -------------------------------------------------
TABLE_TOP = 0.20                      # height of the table surface
BOARD_CENTER = np.array([0.50, 0.0])  # board centre (x, y) in front of the robot
CELL = 0.09                           # cell size
SUPPLY_Y = {"X": 0.27, "O": -0.27}    # y of each player's row of spare pieces
PUCK_R, PUCK_H = 0.022, 0.02          # piece = puck with a symbol on top
HOVER = 0.12                          # approach height above a piece / cell
GRIP_OPEN, GRIP_HOLD = 255.0, 135.0   # gripper ctrl (0 closed .. 255 open)
N_PIECES = 5

COLORS = {"X": [0.85, 0.15, 0.15, 1], "O": [0.15, 0.35, 0.9, 1]}


def ensure_model() -> Path:
    """Download MuJoCo Menagerie if the Panda model is not present."""
    if not SCENE_XML.exists():
        print("Cloning MuJoCo Menagerie (one-time download)...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/google-deepmind/mujoco_menagerie.git",
             str(MENAGERIE_DIR)],
            check=True,
        )
    return SCENE_XML


# --- Board coordinates -------------------------------------------------------
# Cell k (0..8) -> row r = k // 3, col c = k % 3, as seen by a viewer standing
# at +x looking back at the robot: row 0 is the far row (nearest the robot),
# col 0 is the viewer's left (-y).
def cell_xy(k: int) -> np.ndarray:
    r, c = divmod(k, 3)
    return BOARD_CENTER + np.array([(r - 1) * CELL, (c - 1) * CELL])


def supply_xy(symbol: str, i: int) -> np.ndarray:
    x = BOARD_CENTER[0] + (i - (N_PIECES - 1) / 2) * 0.075
    return np.array([x, SUPPLY_Y[symbol]])


def piece_z() -> float:
    return TABLE_TOP + PUCK_H / 2


# --- Scene construction ------------------------------------------------------
def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(ensure_model()))
    spec.body("hand").add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.004] * 3,
                               rgba=[0, 0, 0, 0])
    world = spec.worldbody
    box, cyl = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
    visual = dict(contype=0, conaffinity=0)  # decorative geoms: no collisions

    # Table and board.
    world.add_geom(type=box, size=[0.25, 0.42, TABLE_TOP / 2],
                   pos=[BOARD_CENTER[0], 0, TABLE_TOP / 2],
                   rgba=[0.55, 0.4, 0.25, 1], **visual)
    world.add_geom(type=box, size=[1.5 * CELL, 1.5 * CELL, 0.001],
                   pos=[*BOARD_CENTER, TABLE_TOP + 0.001],
                   rgba=[0.95, 0.95, 0.9, 1], **visual)
    for off in (-0.5 * CELL, 0.5 * CELL):
        for size, dxy in (([1.5 * CELL, 0.003, 0.0015], [0, off]),
                          ([0.003, 1.5 * CELL, 0.0015], [off, 0])):
            world.add_geom(type=box, size=size,
                           pos=[*(BOARD_CENTER + dxy), TABLE_TOP + 0.002],
                           rgba=[0.1, 0.1, 0.1, 1], **visual)

    # Pieces are mocap bodies: we move them directly (attached to the gripper
    # while carried), which keeps the grasp simple and robust.
    white = [1, 1, 1, 1]
    for sym in "XO":
        for i in range(N_PIECES):
            b = world.add_body(name=f"{sym}{i}", mocap=True,
                               pos=[*supply_xy(sym, i), piece_z()])
            b.add_geom(type=cyl, size=[PUCK_R, PUCK_H / 2, 0],
                       rgba=COLORS[sym], **visual)
            top = PUCK_H / 2 + 0.001
            if sym == "X":
                for qz in (0.3826834, -0.3826834):  # +/-45 deg about z
                    b.add_geom(type=box, size=[0.016, 0.003, 0.001],
                               pos=[0, 0, top], quat=[0.9238795, 0, 0, qz],
                               rgba=white, **visual)
            else:
                b.add_geom(type=cyl, size=[0.013, 0.001, 0], pos=[0, 0, top],
                           rgba=white, **visual)
                b.add_geom(type=cyl, size=[0.009, 0.0015, 0], pos=[0, 0, top],
                           rgba=COLORS[sym], **visual)
    return spec.compile()


# --- Inverse kinematics ------------------------------------------------------
class IK:
    """Damped least-squares IK for the TCP site (position + orientation)."""

    def __init__(self, model: mujoco.MjModel, q_home: np.ndarray):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site = model.site("tcp").id
        self.q_home = q_home.copy()
        self.lo = model.jnt_range[:7, 0]
        self.hi = model.jnt_range[:7, 1]
        # Keep the gripper orientation from the home pose (pointing down).
        self.data.qpos[:7] = q_home
        mujoco.mj_kinematics(model, self.data)
        self.quat = np.zeros(4)
        mujoco.mju_mat2Quat(self.quat, self.data.site_xmat[self.site])

    def solve(self, pos, q_init, iters=300, damping=1e-2) -> np.ndarray:
        m, d = self.model, self.data
        d.qpos[:7] = q_init
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        cur, err_rot = np.zeros(4), np.zeros(3)
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err_pos = pos - d.site_xpos[self.site]
            mujoco.mju_mat2Quat(cur, d.site_xmat[self.site])
            mujoco.mju_subQuat(err_rot, self.quat, cur)  # in the local TCP frame
            err_rot_world = d.site_xmat[self.site].reshape(3, 3) @ err_rot
            err = np.concatenate([err_pos, err_rot_world])
            if np.linalg.norm(err) < 1e-5:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site)
            J = np.vstack([jacp[:, :7], jacr[:, :7]])
            JJt = J @ J.T + damping**2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            # Null-space pull towards the home posture keeps the arm tidy.
            N = np.eye(7) - J.T @ np.linalg.solve(JJt, J)
            dq += N @ (0.05 * (self.q_home - d.qpos[:7]))
            d.qpos[:7] = np.clip(d.qpos[:7] + dq, self.lo, self.hi)
        return d.qpos[:7].copy()


# --- Tic-tac-toe logic -------------------------------------------------------
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
         (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def winner(board) -> str | None:
    for a, b, c in LINES:
        if board[a] != " " and board[a] == board[b] == board[c]:
            return board[a]
    return "draw" if " " not in board else None


@lru_cache(maxsize=None)
def _score(board: tuple, player: str) -> int:
    """Minimax value of `board` with `player` to move, from X's perspective."""
    w = winner(board)
    if w == "X":
        return 1
    if w == "O":
        return -1
    if w == "draw":
        return 0
    other = "O" if player == "X" else "X"
    scores = [_score(board[:k] + (player,) + board[k + 1:], other)
              for k in range(9) if board[k] == " "]
    return max(scores) if player == "X" else min(scores)


def best_move(board, player: str) -> int:
    other = "O" if player == "X" else "X"
    sign = 1 if player == "X" else -1
    moves = [k for k in range(9) if board[k] == " "]
    values = {k: sign * _score(tuple(board[:k] + [player] + board[k + 1:]), other)
              for k in moves}
    best = max(values.values())
    return random.choice([k for k in moves if values[k] == best])


def board_str(board) -> str:
    cells = [board[k] if board[k] != " " else str(k + 1) for k in range(9)]
    rows = [" " + " | ".join(cells[r * 3:r * 3 + 3]) for r in range(3)]
    return "\n---+---+---\n".join(rows)


# --- Game + motion controller ------------------------------------------------
class TicTacToe:
    def __init__(self, model, data, robot_symbols: set[str], human_input=None):
        self.m, self.d = model, data
        self.robot_symbols = robot_symbols
        self.inputs = human_input
        self.tcp = model.site("tcp").id
        key = model.key("home").id
        self.q_home = model.key_qpos[key][:7].copy()
        self.ik = IK(model, self.q_home)
        self.q_ready = self.ik.solve(np.array([*BOARD_CENTER, TABLE_TOP + 0.30]),
                                     self.q_home)
        self.segments: deque = deque()
        self.seg = None
        self.carried = None   # mocap id of the piece in the gripper
        self.quit = False
        self.restart_at = None
        self.reset()

    # -- game state --
    def reset(self):
        self.board = [" "] * 9
        self.turn = "X"
        self.used = {"X": 0, "O": 0}
        self.prompted = False
        for sym in "XO":
            for i in range(N_PIECES):
                self.d.mocap_pos[self._mocap(sym, i)] = [*supply_xy(sym, i), piece_z()]
        print("\n=== New game ===\n" + board_str(self.board))
        self.go_to(self.q_ready, GRIP_OPEN, 1.5)

    def _mocap(self, sym, i):
        return self.m.body_mocapid[self.m.body(f"{sym}{i}").id]

    def _commit(self, cell):
        self.board[cell] = self.turn
        print(f"\n{self.turn} -> cell {cell + 1}\n" + board_str(self.board))
        self.turn = "O" if self.turn == "X" else "X"
        self.prompted = False
        w = winner(self.board)
        if w:
            print("\nResult: " + ("draw!" if w == "draw" else f"{w} wins!"))
            if self.inputs is None:
                self.restart_at = self.d.time + 3.0
            else:
                print("Type 'r' to play again or 'q' to quit.")

    # -- motion primitives --
    def go_to(self, q, grip, dur, on_end=None):
        self.segments.append(dict(q=q, grip=grip, dur=dur, on_end=on_end))

    def plan_robot_move(self, cell):
        sym = self.turn
        i = self.used[sym]
        self.used[sym] += 1
        mocap = self._mocap(sym, i)
        src, dst = supply_xy(sym, i), cell_xy(cell)
        z = piece_z()

        q = self.q_ready
        wp = {}
        for name, xy, h in (("src_up", src, HOVER), ("src", src, 0.0),
                            ("dst_up", dst, HOVER), ("dst", dst, 0.0)):
            q = self.ik.solve(np.array([*xy, z + h]), q)
            wp[name] = q

        def grasp():
            self.carried = mocap

        def release():
            self.carried = None
            self.d.mocap_pos[mocap] = [*dst, z]
            self._commit(cell)

        self.go_to(wp["src_up"], GRIP_OPEN, 1.2)
        self.go_to(wp["src"], GRIP_OPEN, 0.6)
        self.go_to(wp["src"], GRIP_HOLD, 0.4, on_end=grasp)
        self.go_to(wp["src_up"], GRIP_HOLD, 0.6)
        self.go_to(wp["dst_up"], GRIP_HOLD, 1.2)
        self.go_to(wp["dst"], GRIP_HOLD, 0.6)
        self.go_to(wp["dst"], GRIP_OPEN, 0.4, on_end=release)
        self.go_to(wp["dst_up"], GRIP_OPEN, 0.5)
        self.go_to(self.q_ready, GRIP_OPEN, 1.0)

    def place_human_piece(self, cell):
        sym = self.turn
        i = self.used[sym]
        self.used[sym] += 1
        self.d.mocap_pos[self._mocap(sym, i)] = [*cell_xy(cell), piece_z()]
        self._commit(cell)

    # -- called every physics step --
    def update(self):
        d = self.d
        if self.seg is None and self.segments:
            self.seg = self.segments.popleft()
            self.seg["t0"] = d.time
            self.seg["q0"] = d.ctrl[:7].copy()
            self.seg["g0"] = d.ctrl[7]

        if self.seg is not None:
            s = self.seg
            a = min(1.0, (d.time - s["t0"]) / s["dur"])
            a = a * a * (3 - 2 * a)  # smoothstep
            d.ctrl[:7] = s["q0"] + a * (s["q"] - s["q0"])
            d.ctrl[7] = s["g0"] + a * (s["grip"] - s["g0"])
            if d.time - s["t0"] >= s["dur"] + 0.15:  # small settle time
                self.seg = None
                if s["on_end"]:
                    s["on_end"]()
        else:
            self._decide()

        if self.carried is not None:
            d.mocap_pos[self.carried] = d.site_xpos[self.tcp]

    def _decide(self):
        cmd = None
        if self.inputs is not None:
            try:
                cmd = self.inputs.get_nowait().strip().lower()
            except queue.Empty:
                pass
        if cmd == "q":
            self.quit = True
            return
        if cmd == "r":
            self.reset()
            return

        if winner(self.board):
            if self.restart_at is not None and self.d.time >= self.restart_at:
                self.restart_at = None
                self.reset()
            return

        if self.turn in self.robot_symbols:
            self.plan_robot_move(best_move(self.board, self.turn))
            return

        if not self.prompted:
            print(f"\nYour move ({self.turn}) - enter cell 1-9: ", end="", flush=True)
            self.prompted = True
        if cmd:
            if cmd.isdigit() and 1 <= int(cmd) <= 9 and self.board[int(cmd) - 1] == " ":
                self.place_human_piece(int(cmd) - 1)
            else:
                print("Invalid move, enter a free cell 1-9: ", end="", flush=True)


def stdin_reader(q: queue.Queue):
    for line in sys.stdin:
        q.put(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["human", "selfplay"], default="human")
    parser.add_argument("--robot-plays", choices=["X", "O"], default="O",
                        help="robot's symbol in human mode (X moves first)")
    parser.add_argument("--headless", action="store_true",
                        help="self-play one game without a viewer")
    args = parser.parse_args()

    model = build_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    if args.headless or args.mode == "selfplay":
        game = TicTacToe(model, data, robot_symbols={"X", "O"})
    else:
        inputs = queue.Queue()
        threading.Thread(target=stdin_reader, args=(inputs,), daemon=True).start()
        game = TicTacToe(model, data, robot_symbols={args.robot_plays}, human_input=inputs)

    if args.headless:
        while not winner(game.board) and data.time < 120:
            game.update()
            mujoco.mj_step(model, data)
        print(f"\nSimulated {data.time:.1f}s")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [*BOARD_CENTER, TABLE_TOP]
            viewer.cam.distance = 1.6
            viewer.cam.azimuth = 180   # stand opposite the robot
            viewer.cam.elevation = -45
        while viewer.is_running() and not game.quit:
            step_start = time.time()
            game.update()
            mujoco.mj_step(model, data)
            viewer.sync()
            sleep = model.opt.timestep - (time.time() - step_start)
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    main()
