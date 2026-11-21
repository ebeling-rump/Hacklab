"""Warehouse intersection scenario: aisles, shelves and rule-following AGVs.

Layout (10 m x 10 m): three horizontal and three vertical aisles (1.8 m wide) at
-3, 0, +3 m; shelves fill the blocks in between. The nine aisle crossings are
intersections. Short dead-end stubs run from the outer intersections to the walls;
the robot starts and ends in those stubs.

AGV behaviour is deterministic and depends only on things an observer could infer
from position, heading and surroundings:
  - drive along the aisle centreline at a per-AGV cruise speed (accel-limited),
  - aisles are one-way for AGVs (perimeter counter-clockwise, middle aisles west/north),
  - at each intersection pick a direction with a fixed, location-based routing rule
    (see `route_choice`), never into a dead-end stub,
  - slow down before turning, pause at pick stations,
  - stop if the robot is close ahead in its path, or an AGV it must yield to
    (same direction, or coming from the right).
Nothing is random after spawning. The per-AGV parameters (cruise speed, routing table)
are hidden: predictors must not read them.
"""
import numpy as np

from obstacles import Obstacle

AISLES = np.array([-3.0, 0.0, 3.0])   # centrelines of both horizontal and vertical aisles
AISLE_HALF = 1.0                       # half-width (m): room for the robot to pass a stopped AGV
ARENA = 5.0
R_AGV = 0.3

STATIONS = np.array([[-1.5, 0.0], [1.5, 3.0], [0.0, -1.5], [-3.0, 1.5], [3.0, -1.5]])
STATION_PAUSE = 1.5      # s
TURN_SPEED = 0.35        # m/s through a turning intersection
SLOW_ZONE = 1.2          # m before an intersection/station where slowing starts
ACCEL = 1.5              # m/s^2, max acceleration and braking
APPROACH = 1.7           # m from an intersection centre where the stop-line rule applies
YIELD_AHEAD = 1.4        # m: stop if something is this close ahead...
YIELD_LATERAL = 0.75     # ...and within this lateral offset of the AGV's path (other AGVs)
YIELD_LATERAL_ROBOT = 0.55   # robot only blocks if the AGV would otherwise touch it
ROBOT_STOP_AHEAD = 0.7       # AGVs have right of way: they brake for the robot only this close
                             # (an emergency stop, like a safety scanner), not at YIELD_AHEAD

DIRS = {"E": np.array([1.0, 0.0]), "N": np.array([0.0, 1.0]),
        "W": np.array([-1.0, 0.0]), "S": np.array([0.0, -1.0])}
LEFT_OF = {"E": "N", "N": "W", "W": "S", "S": "E"}
RIGHT_OF = {v: k for k, v in LEFT_OF.items()}
BACK_OF = {"E": "W", "W": "E", "N": "S", "S": "N"}


def shelf_boxes():
    """Shelf rectangles (xmin, xmax, ymin, ymax): everything not inside an aisle band."""
    edges = [-ARENA] + [e for a in AISLES for e in (a - AISLE_HALF, a + AISLE_HALF)] + [ARENA]
    spans = [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2)]
    return [(x0, x1, y0, y1) for (x0, x1) in spans for (y0, y1) in spans]


def in_aisles(points, margin=0.0):
    """True where a point (..., 2) lies inside an aisle band, `margin` away from the shelves."""
    p = np.asarray(points)
    half = AISLE_HALF - margin
    in_x = (np.abs(p[..., 0, None] - AISLES) <= half).any(-1)   # inside a vertical aisle
    in_y = (np.abs(p[..., 1, None] - AISLES) <= half).any(-1)   # inside a horizontal aisle
    inside = (np.abs(p) <= ARENA - margin).all(-1)
    return (in_x | in_y) & inside


# One-way aisles (AGV traffic only; the robot may drive either way).
# Perimeter runs counter-clockwise, the middle aisles run west and north.
H_DIR = ["E", "W", "W"]   # horizontal aisle j (y = AISLES[j])
V_DIR = ["S", "N", "N"]   # vertical aisle i (x = AISLES[i])


def _step_node(node, d):
    return node[0] + int(DIRS[d][0]), node[1] + int(DIRS[d][1])


def exits(node):
    """Allowed directions out of intersection `node` (i, j) that stay on the 3x3 grid."""
    i, j = node
    return [d for d in (H_DIR[j], V_DIR[i]) if 0 <= _step_node(node, d)[0] < 3 and 0 <= _step_node(node, d)[1] < 3]


def route_choice(node, heading, turn_parity=1):
    """Routing table: the direction an AGV takes when it reaches intersection `node`.

    Location-based rule: where (i + j) % 2 == turn_parity the AGV turns if it can,
    elsewhere it goes straight if it can. So the decision is a function of (position,
    heading) only. turn_parity=1 is the standard fleet; 0 is the "unseen" flipped routing.
    """
    options = exits(node)
    if len(options) == 1:
        return options[0]
    prefer_turn = (node[0] + node[1]) % 2 == turn_parity
    straight = heading in options
    if straight and not prefer_turn:
        return heading
    turns = [d for d in options if d != heading]
    return turns[0] if turns else heading


class AGV(Obstacle):
    """Automated guided vehicle following the aisle network with the rules above."""
    pattern = "warehouse_agv"

    def __init__(self, node, heading, progress, cruise, turn_parity=1):
        """Start on the segment leaving intersection `node` in direction `heading`, `progress` m along."""
        self.turn_parity = turn_parity
        self.heading = heading
        self.next_node = _step_node(node, heading)
        self.cruise = cruise
        self.speed = cruise
        self.mode = "cruise"
        self.pause_left = 0.0
        self.visited_station = None
        start = np.array([AISLES[node[0]], AISLES[node[1]]]) + DIRS[heading] * progress
        super().__init__(start, DIRS[heading] * cruise, radius=R_AGV)

    def _node_pos(self, node):
        return np.array([AISLES[node[0]], AISLES[node[1]]])

    def _next_station(self):
        """Distance to a station ahead on the current segment, or None."""
        ahead = STATIONS - self.pos
        along = ahead @ DIRS[self.heading]
        lateral = np.abs(ahead @ DIRS[LEFT_OF[self.heading]])
        ok = (along > -0.01) & (lateral < 0.05) & (along < np.linalg.norm(self._node_pos(self.next_node) - self.pos))
        for k in np.flatnonzero(ok):
            if k != self.visited_station:
                return along[k], k
        return None

    def _blocked(self, robot_pos, agvs):
        """Distance to the nearest thing ahead in the path corridor that this AGV must yield to.

        Always yields to the robot. Yields to another AGV only if it drives the same way
        (following) or comes from the right (priority to the right at crossings).
        """
        fwd, left = DIRS[self.heading], DIRS[LEFT_OF[self.heading]]
        candidates = [] if robot_pos is None else [(robot_pos, YIELD_LATERAL_ROBOT, ROBOT_STOP_AHEAD)]
        candidates += [(a.pos, YIELD_LATERAL, YIELD_AHEAD) for a in agvs
                       if a is not self and a.heading in (self.heading, LEFT_OF[self.heading])]
        best = np.inf
        for p, max_lateral, max_ahead in candidates:
            rel = p - self.pos
            along, lateral = rel @ fwd, abs(rel @ left)
            if 0 < along < max_ahead and lateral < max_lateral:
                best = min(best, along)
        return best

    def _wait_at_intersection(self, agvs):
        """Stop-line rule while approaching an intersection.

        Wait if another AGV is already inside the intersection box, or if one with priority
        (coming from the right) is approaching the same intersection.
        """
        node = self._node_pos(self.next_node)
        my_dist = np.linalg.norm(node - self.pos)
        if not (AISLE_HALF + 0.05 < my_dist < APPROACH):
            return False                       # not approaching, or already past the stop line
        for a in agvs:
            if a is self:
                continue
            inside = (np.abs(a.pos - node) < AISLE_HALF).all()
            from_right = (a.heading == LEFT_OF[self.heading] and a.next_node == self.next_node
                          and np.linalg.norm(node - a.pos) < APPROACH and a.speed > 0.05)
            if inside or from_right:
                return True
        return False

    def step(self, dt, robot_pos=None, agvs=()):
        if self.pause_left > 0:                      # standing at a station
            self.pause_left -= dt
            self.speed = 0.0
            self.vel = np.zeros(2)
            self.mode = "station_pause"
            return

        node_pos = self._node_pos(self.next_node)
        dist_node = np.linalg.norm(node_pos - self.pos)
        turning = route_choice(self.next_node, self.heading, self.turn_parity) != self.heading

        # target speed from the rules; `mode` is a hidden label kept for analysis only
        target, self.mode = self.cruise, "cruise"
        if turning and dist_node < SLOW_ZONE:
            target = min(target, TURN_SPEED + (self.cruise - TURN_SPEED) * dist_node / SLOW_ZONE)
            self.mode = "turn"
        station = self._next_station()
        if station is not None and station[0] < SLOW_ZONE:
            target = min(target, 0.15 + (self.cruise - 0.15) * station[0] / SLOW_ZONE)
            self.mode = "station_approach"
        if self._blocked(robot_pos, agvs) < np.inf or self._wait_at_intersection(agvs):
            target = 0.0
            self.mode = "yield"

        self.speed += np.clip(target - self.speed, -ACCEL * dt, ACCEL * dt)
        move = self.speed * dt

        if station is not None and station[0] <= move:   # arrive at station: stop and pause
            self.pos = STATIONS[station[1]].copy()
            self.visited_station = station[1]
            self.pause_left = STATION_PAUSE
            self.speed = 0.0
        elif move >= dist_node:                          # pass through the intersection
            new_heading = route_choice(self.next_node, self.heading, self.turn_parity)
            node = self.next_node
            self.pos = node_pos + DIRS[new_heading] * (move - dist_node)
            self.heading = new_heading
            self.next_node = _step_node(node, new_heading)
            self.visited_station = None
        else:
            self.pos = self.pos + DIRS[self.heading] * move
        self.vel = DIRS[self.heading] * self.speed


def make_agvs(rng, robot_start, n=4, cruise=(0.7, 1.1), turn_parity=1):
    """Spawn `n` AGVs on random grid segments, away from the robot start and from each other.

    `cruise` (speed range) and `turn_parity` (routing table) define the fleet's motion pattern;
    the defaults are the standard fleet, other values create unseen patterns for testing.
    """
    agvs = []
    while len(agvs) < n:
        node = (int(rng.integers(3)), int(rng.integers(3)))
        heading = str(rng.choice(exits(node)))
        agv = AGV(node, heading, progress=rng.uniform(0.3, 2.7), cruise=rng.uniform(*cruise),
                  turn_parity=turn_parity)
        if np.linalg.norm(agv.pos - robot_start) < 2.5:
            continue
        if any(np.linalg.norm(agv.pos - a.pos) < 1.5 for a in agvs):
            continue
        agvs.append(agv)
    return agvs


def scene_xml(n_agvs):
    """MJCF for the warehouse: floor, walls, shelves (solid for the robot), robot, AGVs, goal."""
    shelves = "\n".join(
        f'    <geom type="box" pos="{(x0 + x1) / 2:g} {(y0 + y1) / 2:g} 0.6" '
        f'size="{(x1 - x0) / 2:g} {(y1 - y0) / 2:g} 0.6" rgba="0.55 0.45 0.33 1" contype="1" conaffinity="1"/>'
        for x0, x1, y0, y1 in shelf_boxes())
    stations = "\n".join(
        f'    <site type="box" pos="{x:g} {y:g} 0.003" size="0.3 0.3 0.003" rgba="0.98 0.7 0.1 0.6"/>'
        for x, y in STATIONS)
    colors = ["0.92 0.41 0.2", "0.38 0.31 0.84", "0.11 0.69 0.48", "0.91 0.48 0.64", "0.89 0.29 0.28"]
    agvs = "\n".join(
        f'    <body name="obs_{i}" mocap="true" pos="0 0 0.2">\n'
        f'      <geom type="cylinder" size="{R_AGV} 0.2" rgba="{colors[i % len(colors)]} 1"/>\n'
        f'      <geom type="box" pos="0 0 0.21" size="0.12 0.12 0.01" rgba="0.1 0.1 0.1 1"/>\n'
        f'    </body>' for i in range(n_agvs))
    return f"""<mujoco model="warehouse">
  <option timestep="0.002"/>
  <visual>
    <headlight ambient="0.3 0.3 0.3" diffuse="0.3 0.3 0.3" specular="0 0 0"/>
    <global offwidth="720" offheight="720"/>
  </visual>
  <default><geom contype="0" conaffinity="0"/></default>
  <worldbody>
    <light pos="0 0 10" dir="0 0 -1" directional="true" diffuse="0.35 0.35 0.35" specular="0 0 0"/>
    <camera name="top" pos="0 0 13" xyaxes="1 0 0 0 1 0" fovy="45"/>
    <geom type="plane" size="5 5 0.1" rgba="0.6 0.62 0.64 1"/>
{shelves}
{stations}
    <site name="goal" pos="4.5 0 0.005" type="cylinder" size="0.2 0.005" rgba="0.05 0.64 0.05 0.9"/>
    <body name="robot" pos="0 0 0.15">
      <joint name="robot_x" type="slide" axis="1 0 0"/>
      <joint name="robot_y" type="slide" axis="0 1 0"/>
      <geom name="robot" type="cylinder" size="0.25 0.15" mass="10" rgba="0.16 0.47 0.84 1"
            contype="1" conaffinity="1"/>
      <geom type="box" pos="0 0 0.16" size="0.05 0.05 0.01" rgba="1 1 1 1"/>
    </body>
{agvs}
  </worldbody>
  <actuator>
    <velocity name="vx" joint="robot_x" kv="200" ctrlrange="-1 1" ctrllimited="true"/>
    <velocity name="vy" joint="robot_y" kv="200" ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


def make_route(rng):
    """Robot start (west stub), goal (east stub) and intersection waypoints between them."""
    ys, yg = rng.choice(AISLES), rng.choice(AISLES)
    start, goal = np.array([-4.5, ys]), np.array([4.5, yg])
    if ys == yg:
        return start, goal, [goal]
    xt = rng.choice(AISLES)                  # vertical aisle used to change rows
    return start, goal, [np.array([xt, ys]), np.array([xt, yg]), goal]
