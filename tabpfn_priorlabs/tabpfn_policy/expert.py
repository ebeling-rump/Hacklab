"""Reactive task-space scripted expert (the teacher).

`Expert.act(state, phase)` maps ANY world state + the current phase to
  (action, new_phase), action = [dx, dy, dz, dyaw, grip]
where (dx, dy, dz) [m] and dyaw [rad] are a small, clipped TCP move for the next control period and
`grip` is the commanded finger gap [m] (0 = close, 0.08 = open).

It is a phase machine with proportional control toward the phase target. Phase transitions are
distance / width thresholds on the current state only (no timers, no memory beyond `phase`), so the
expert can relabel arbitrary states visited by a student (DAgger) and it re-plans by itself when the
world changes (cube nudged -> keeps tracking it; cube dropped -> back to APPROACH).

The expert may be fed the TRUE cube pose (labeling only). The student never sees it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .geometry import clip_norm, to_bin_frame, wrap90


class Phase(IntEnum):
    APPROACH = 0   # fly to a hover point above the cube, align yaw, open gripper
    DESCEND = 1    # go straight down to grasp height (keeps tracking cube xy)
    GRASP = 2      # hold still, close the fingers
    LIFT = 3       # straight up to carry height
    CARRY = 4      # fly above the bin centre, align cube with bin walls
    LOWER = 5      # go down to release height inside the bin
    RELEASE = 6    # open fingers
    RETREAT = 7    # straight up out of the bin
    DONE = 8


N_PHASES = len(Phase)
ACTION_NAMES = ("dx", "dy", "dz", "dyaw", "grip")


@dataclass
class ExpertState:
    """Everything the expert needs. Base frame, metres / radians."""
    tcp_pos: np.ndarray      # (3,)
    tcp_yaw: float
    width: float             # measured finger gap
    cube_pos: np.ndarray     # (3,) cube centre
    cube_yaw: float
    bin_xy: np.ndarray       # (2,) bin centre
    bin_yaw: float
    width_vel: float = 0.0   # d(width)/dt [m/s], finite difference
    tcp_vel: np.ndarray | None = None  # (3,) measured TCP velocity [m/s] (None = no accel limit)
    bin_inner: tuple = (0.24, 0.24)
    floor_top: float = 0.01
    cube_size: float = 0.04


@dataclass
class ExpertConfig:
    gain: float = 0.5            # fraction of the remaining position error commanded per step
    yaw_gain: float = 0.5
    max_step: float = 0.010      # m per control period (clip on |dxyz|)
    max_dyaw: float = 0.08       # rad per control period
    max_dyaw_held: float = 0.025 # slower wrist rotation while holding a cube (torsional slip in the weak grip)
    # While holding (the model's gripper squeezes only ~2-4 N): step <= ramp_gain * measured step + ramp_add.
    # ramp_gain > 1 keeps the ramp growing even though the lagging servo moves less than commanded.
    ramp_gain: float = 1.5
    ramp_add: float = 0.0025
    dt: float = 0.05             # control period [s], to turn tcp_vel into a per-step move
    yaw_reach: float = 0.10      # rad beyond +-45 deg the wrist may go (hysteresis) before we pick the canonical yaw
    hover: float = 0.10          # TCP height above the cube centre during APPROACH
    slow_zone: float = 0.03      # DESCEND: within this of grasp height, cap the step at slow_step (no overshoot)
    slow_step: float = 0.004
    carry_yaw_slow: float = 0.15 # CARRY: while |yaw error| > this, move sideways at most slow_carry_step (cube slips
    slow_carry_step: float = 0.003  #        out of the weak grip when the wrist turns during a fast transfer)
    carry_z: float = 0.22        # TCP height while lifting / carrying / retreating
    drop_clearance: float = 0.02 # cube bottom above the bin floor at release
    open_w: float = 0.08
    close_w: float = 0.0
    # thresholds
    xy_tol_approach: float = 0.006
    z_tol_approach: float = 0.015
    yaw_tol: float = 0.05
    xy_tol_descend: float = 0.005   # stop descending while xy error exceeds this
    xy_abort: float = 0.02          # cube moved away -> back to APPROACH
    z_tol_grasp: float = 0.004
    held_tol: float = 0.008         # width >= cube_size - this -> fingers did not close past the cube
    held_max: float = 0.060         # width <= this (cube diagonal 0.057 + margin) -> fingers still on the cube
    settle_wvel: float = 0.002      # |width_vel| below this [m/s] -> squeeze has settled, ok to lift
    held_dist: float = 0.045        # cube centre within this of the TCP -> cube is in the hand (cubes slip ~2 cm in the weak grip)
    xy_tol_carry: float = 0.008
    z_tol_lower: float = 0.006
    open_done_w: float = 0.065
    in_bin_margin: float = 0.02


class Expert:
    def __init__(self, cfg: ExpertConfig | None = None):
        self.cfg = cfg or ExpertConfig()

    # ---- predicates on the state -------------------------------------------------------------------
    def fingers_on_cube(self, s: ExpertState) -> bool:
        return s.cube_size - self.cfg.held_tol <= s.width <= self.cfg.held_max

    def cube_in_hand(self, s: ExpertState) -> bool:
        return self.fingers_on_cube(s) and float(np.linalg.norm(s.cube_pos - s.tcp_pos)) <= self.cfg.held_dist

    def cube_in_bin(self, s: ExpertState) -> bool:
        b = to_bin_frame(s.cube_pos, s.bin_xy, s.bin_yaw)
        m = self.cfg.in_bin_margin
        inside = abs(b[0]) <= s.bin_inner[0] / 2 - m and abs(b[1]) <= s.bin_inner[1] / 2 - m
        return bool(inside and s.cube_pos[2] < s.floor_top + 1.5 * s.cube_size)

    def cube_over_bin(self, s: ExpertState) -> bool:
        """Cube footprint inside the bin walls (any height)."""
        b = to_bin_frame(s.cube_pos, s.bin_xy, s.bin_yaw)
        m = self.cfg.in_bin_margin
        return bool(abs(b[0]) <= s.bin_inner[0] / 2 - m and abs(b[1]) <= s.bin_inner[1] / 2 - m)

    def release_z(self, s: ExpertState) -> float:
        return s.floor_top + self.cfg.drop_clearance + s.cube_size / 2

    # ---- phase machine -----------------------------------------------------------------------------
    def next_phase(self, s: ExpertState, phase: Phase) -> Phase:
        c = self.cfg
        P = Phase
        cube_xy_err = float(np.linalg.norm(s.cube_pos[:2] - s.tcp_pos[:2]))
        hover_z = s.cube_pos[2] + c.hover
        yaw_err = abs(self.yaw_error(s.cube_yaw, s.tcp_yaw))
        bin_xy_err = float(np.linalg.norm(s.bin_xy - s.tcp_pos[:2]))
        bin_yaw_err = abs(self.yaw_error(s.bin_yaw, s.tcp_yaw))

        # Global reactions, independent of the current phase.
        if phase in (P.APPROACH, P.DESCEND, P.GRASP) and self.cube_in_bin(s):
            return P.RETREAT if s.tcp_pos[2] < c.carry_z - 0.015 else P.DONE
        if phase in (P.LIFT, P.CARRY, P.LOWER) and not self.cube_in_hand(s):
            if phase == P.LOWER and self.cube_over_bin(s):
                return P.RELEASE                                 # cube touched down early inside the bin
            return P.APPROACH                                    # dropped / slipped: start over

        if phase == P.APPROACH:
            if cube_xy_err < c.xy_tol_approach and abs(s.tcp_pos[2] - hover_z) < c.z_tol_approach \
                    and yaw_err < c.yaw_tol and s.width > c.open_done_w:
                return P.DESCEND
        elif phase == P.DESCEND:
            if cube_xy_err > c.xy_abort:
                return P.APPROACH
            if abs(s.tcp_pos[2] - s.cube_pos[2]) < c.z_tol_grasp and cube_xy_err < c.xy_tol_descend:
                return P.GRASP
        elif phase == P.GRASP:
            if cube_xy_err > c.xy_abort:
                return P.APPROACH
            if s.width < s.cube_size - c.held_tol:               # closed on nothing
                return P.APPROACH
            if s.width <= s.cube_size + 0.0005 and abs(s.width_vel) < c.settle_wvel:   # squeezed and settled
                return P.LIFT
        elif phase == P.LIFT:
            if s.tcp_pos[2] > c.carry_z - 0.015:
                return P.CARRY
        elif phase == P.CARRY:
            if bin_xy_err < c.xy_tol_carry and bin_yaw_err < c.yaw_tol:
                return P.LOWER
        elif phase == P.LOWER:
            if s.tcp_pos[2] < self.release_z(s) + c.z_tol_lower:
                return P.RELEASE
        elif phase == P.RELEASE:
            if s.width > c.open_done_w:
                return P.RETREAT
        elif phase == P.RETREAT:
            if s.tcp_pos[2] > c.carry_z - 0.015:
                return P.DONE
        return phase

    # ---- control law -------------------------------------------------------------------------------
    def yaw_error(self, target: float, tcp_yaw: float) -> float:
        """Yaw to turn by. Cube, bin and gripper are 90-deg symmetric, so the nearest equivalent is used, unless that
        would take the wrist further than 45 deg + yaw_reach from home (then joint 7 runs into its limit once the base
        joint turns toward the bin): then turn to the canonical target in (-45, 45] deg instead."""
        e = wrap90(target - tcp_yaw)
        if abs(tcp_yaw + e) > np.pi / 4 + self.cfg.yaw_reach:
            e = wrap90(target) - tcp_yaw
        return float(e)

    def control(self, s: ExpertState, phase: Phase) -> np.ndarray:
        c = self.cfg
        P = Phase
        p = s.tcp_pos
        target, yaw_target, grip = p.copy(), s.tcp_yaw, c.open_w
        if phase == P.APPROACH:
            target = np.r_[s.cube_pos[:2], s.cube_pos[2] + c.hover]
            # do not drag the hand sideways through the table: climb first if low and far
            if p[2] < s.cube_pos[2] + 0.5 * c.hover and np.linalg.norm(target[:2] - p[:2]) > c.xy_abort:
                target = np.r_[p[:2], target[2]]
            yaw_target = s.cube_yaw
        elif phase == P.DESCEND:
            z = s.cube_pos[2]
            if np.linalg.norm(s.cube_pos[:2] - p[:2]) > c.xy_tol_descend:
                z = p[2]                                         # fix xy first
            target = np.r_[s.cube_pos[:2], z]
            yaw_target = s.cube_yaw
        elif phase == P.GRASP:
            grip = c.close_w                                     # hold still: the closing fingers centre the cube
        elif phase == P.LIFT:
            target, yaw_target, grip = np.r_[p[:2], c.carry_z], s.bin_yaw, c.close_w   # pre-rotate while lifting
        elif phase == P.CARRY:
            target, yaw_target, grip = np.r_[s.bin_xy, c.carry_z], s.bin_yaw, c.close_w
        elif phase == P.LOWER:
            target, yaw_target, grip = np.r_[s.bin_xy, self.release_z(s)], s.bin_yaw, c.close_w
        elif phase == P.RELEASE:
            target, yaw_target, grip = np.r_[s.bin_xy, p[2]], s.bin_yaw, c.open_w
        elif phase == P.RETREAT:
            target = np.r_[p[:2], c.carry_z]
        elif phase == P.DONE:
            pass
        holding = phase in (P.LIFT, P.CARRY, P.LOWER)
        max_step = c.max_step
        if holding and s.tcp_vel is not None:
            max_step = min(max_step, c.ramp_gain * float(np.linalg.norm(s.tcp_vel)) * c.dt + c.ramp_add)
        if phase == P.DESCEND and p[2] - s.cube_pos[2] < c.slow_zone:
            max_step = min(max_step, c.slow_step)
        if phase == P.CARRY and abs(self.yaw_error(yaw_target, s.tcp_yaw)) > c.carry_yaw_slow:
            max_step = min(max_step, c.slow_carry_step)
        dxyz = clip_norm(c.gain * (target - p), max_step)
        max_dyaw = c.max_dyaw_held if holding else c.max_dyaw
        dyaw = float(np.clip(c.yaw_gain * self.yaw_error(yaw_target, s.tcp_yaw), -max_dyaw, max_dyaw))
        return np.r_[dxyz, dyaw, grip]

    def act(self, s: ExpertState, phase: Phase) -> tuple[np.ndarray, Phase]:
        """One reactive step: update the phase from the state, then return the action for that phase."""
        new_phase = self.next_phase(s, Phase(phase))
        return self.control(s, new_phase), new_phase
