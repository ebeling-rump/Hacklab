"""Obstacle motion patterns.

Each obstacle owns its *hidden* motion parameters (velocity, turn schedule, random
generator). Controllers and learning code must only use what an observer could see:
the positions over time (see controller.ObstacleTracker). Never read these attributes
as features.
"""
import numpy as np

ARENA = 5.0      # half-size of the arena (m); walls at +-ARENA
R_OBS = 0.3      # obstacle radius (m)


class Obstacle:
    pattern = "base"

    def __init__(self, pos, vel, radius=R_OBS, arena=ARENA):
        self.pos = np.asarray(pos, dtype=float).copy()
        self.vel = np.asarray(vel, dtype=float).copy()
        self.radius = radius
        self.limit = arena - radius

    def _reflect(self):
        """Bounce off the arena walls (mirror position, flip that velocity component)."""
        for i in range(2):
            if self.pos[i] > self.limit:
                self.pos[i] = 2 * self.limit - self.pos[i]
                self.vel[i] = -abs(self.vel[i])
            elif self.pos[i] < -self.limit:
                self.pos[i] = -2 * self.limit - self.pos[i]
                self.vel[i] = abs(self.vel[i])

    def step(self, dt):
        raise NotImplementedError


class ConstantVelocity(Obstacle):
    """Straight line at constant velocity; only the walls change its direction."""
    pattern = "constant_velocity"

    def step(self, dt):
        self.pos += self.vel * dt
        self._reflect()


class RandomTurns(Obstacle):
    """Constant speed, but every 0.5-1.5 s it turns abruptly to a new random heading."""
    pattern = "random_turns"

    def __init__(self, pos, speed, rng, interval=(0.5, 1.5), **kw):
        self.rng = rng
        self.speed = speed
        self.interval = interval
        super().__init__(pos, self._random_velocity(), **kw)
        self.time_to_turn = rng.uniform(*interval)

    def _random_velocity(self):
        heading = self.rng.uniform(0, 2 * np.pi)
        return self.speed * np.array([np.cos(heading), np.sin(heading)])

    def step(self, dt):
        self.time_to_turn -= dt
        if self.time_to_turn <= 0:
            self.vel = self._random_velocity()
            self.time_to_turn = self.rng.uniform(*self.interval)
        self.pos += self.vel * dt
        self._reflect()


def make_obstacles(rng):
    """The Phase 1 pair: one constant-velocity crosser, one random-turner near the middle.

    The constant-velocity obstacle starts above or below the robot's path and heads
    across it, so encounters are likely. The random-turner starts near the centre.
    """
    side = rng.choice([-1.0, 1.0])
    crosser = ConstantVelocity(
        pos=[rng.uniform(-1.5, 1.5), side * rng.uniform(2.5, 4.0)],
        vel=[rng.uniform(-0.3, 0.3), -side * rng.uniform(0.5, 0.9)],
    )
    turner = RandomTurns(pos=rng.uniform(-2.0, 2.0, size=2), speed=rng.uniform(0.5, 0.9), rng=rng)
    return [crosser, turner]
