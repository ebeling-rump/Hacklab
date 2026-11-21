"""Scenarios: everything that defines one episode besides the controller.

    open       - Phase 1: open arena, one constant-velocity and one random-turn obstacle
    warehouse  - aisles and shelves, 4 rule-following AGVs, robot route through intersections
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

import warehouse
from controller import R_ROBOT
from obstacles import make_obstacles

HERE = Path(__file__).parent


@dataclass
class Scenario:
    name: str
    xml: str                       # MJCF string
    start: np.ndarray
    goal: np.ndarray
    waypoints: list                # intermediate targets, last one is the goal
    obstacles: list
    free_space: Callable = None    # points (..., 2) -> bool mask; None = whole arena
    max_time: float = 20.0
    shelves: list = None           # (xmin, xmax, ymin, ymax) boxes, for plotting


def open_arena(seed):
    rng = np.random.default_rng(seed)
    start = np.array([-4.0, rng.uniform(-1, 1)])
    goal = np.array([4.0, rng.uniform(-1, 1)])
    return Scenario("open", (HERE / "scene.xml").read_text(), start, goal, [goal], make_obstacles(rng))


def warehouse_intersections(seed):
    rng = np.random.default_rng(seed)
    start, goal, waypoints = warehouse.make_route(rng)
    agvs = warehouse.make_agvs(rng, start)
    return Scenario("warehouse", warehouse.scene_xml(len(agvs)), start, goal, waypoints, agvs,
                    free_space=lambda p: warehouse.in_aisles(p, margin=R_ROBOT),
                    max_time=40.0, shelves=warehouse.shelf_boxes())


SCENARIOS = {"open": open_arena, "warehouse": warehouse_intersections}
