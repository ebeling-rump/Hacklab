"""T7 - Failure detection and recovery policy."""
from __future__ import annotations

from enum import Enum


class GraspStatus(Enum):
    HELD = "held"               # fingers stopped on something cube-sized
    EMPTY = "empty"             # fingers (almost) closed: nothing between them
    MISALIGNED = "misaligned"   # something else (cube grabbed on a corner / wrong width)


class Action(Enum):
    CONTINUE = "continue"
    RESCAN_RETRY = "rescan_retry"   # open gripper, go back up, scan again, plan again
    GIVE_UP = "give_up"             # skip this colour


def classify_grasp(width: float, cube_size: float, tol: float = 0.012) -> GraspStatus:
    """HELD if the gap matches the cube edge, EMPTY if the fingers (almost) closed, else MISALIGNED."""
    if abs(width - cube_size) <= tol:
        return GraspStatus.HELD
    if width <= tol:
        return GraspStatus.EMPTY
    return GraspStatus.MISALIGNED

def still_holding(width: float, cube_size: float, tol: float = 0.015) -> bool:
    """True while the finger gap is still consistent with a held cube."""
    return abs(width - cube_size) <= tol

class RecoveryPolicy:
    """Bounded retries per colour. TODO(T7.3): implement next_action.

    - HELD -> CONTINUE (and reset the failure counter of that colour).
    - otherwise count a failure for `color`; RESCAN_RETRY while failures < max_attempts, else GIVE_UP.
    - after GIVE_UP the colour is in `given_up`.
    """

    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts
        self.failures: dict[str, int] = {}
        self.given_up: set[str] = set()

    def next_action(self, color: str, status: GraspStatus) -> Action:
        if status == GraspStatus.HELD:
            self.failures[color] = 0
            return Action.CONTINUE
        self.failures[color] = self.failures.get(color, 0) + 1
        if self.failures[color] < self.max_attempts:
            return Action.RESCAN_RETRY
        self.given_up.add(color)
        return Action.GIVE_UP
