"""Procedural outdoor sites for a wheeled robot with a 3D lidar.

Each site is a 60 m x 60 m grid (0.5 m cells) with seven terrain types, a height
map, a moisture field and hard obstacles (trees, boulders). Site-level hidden
parameters (wetness, clay content, vegetation season, lidar intensity gain)
change how terrain behaves, so the mapping "lidar features -> slip" differs
from site to site. That is the setting where a model has to be learned anew
from a few hundred self-labelled cells at every deployment.

Ground truth per cell:
    slip      latent wheel slip in [0, 1] when the robot drives over the cell
    label     1 if the measured slip (slip + sensor noise) exceeds SLIP_THRESHOLD
    p_stuck   probability of getting stuck when entering the cell

Observed per cell (what the lidar gives you, range dependent, with NaNs):
    FEATURES  see list below
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

N = 120            # cells per side
CELL = 0.5         # metres per cell
SLIP_THRESHOLD = 0.45

TERRAIN = ["track", "short grass", "tall grass", "bush", "mud", "sand", "rubble"]
T_TRACK, T_SGRASS, T_TGRASS, T_BUSH, T_MUD, T_SAND, T_RUBBLE = range(7)
OBSTACLE = 7       # hard obstacle id in the terrain map (trees, boulders)

FEATURES = [
    "range_m",       # distance from the sensor when the cell was observed
    "n_points",      # lidar returns falling into the cell
    "h_max",         # max height above the local ground estimate [m]
    "h_std",         # std of point heights [m]
    "roughness",     # residual of a plane fit [m]
    "slope_deg",     # slope of the fitted plane [deg]
    "multi_echo",    # share of beams with more than one echo
    "ground_frac",   # share of last returns within 5 cm of the ground
    "int_mean",      # mean lidar intensity (uncalibrated, sensor gain applies)
    "int_std",       # std of lidar intensity
]


@dataclass
class SiteParams:
    wetness: float   # 0 dry .. 1 after heavy rain
    clay: float      # 0 sandy soil .. 1 clay soil
    season: float    # 0 sparse/short vegetation .. 1 dense/tall vegetation
    gain: float      # lidar intensity gain of this robot unit (sensor, firmware, sun)
    h_bias: float    # height offset of this unit's lidar mount calibration [m]
    stiff: float     # vegetation species: 0 soft grass .. 1 woody, tangling stems
    loose: float     # rubble: 0 locked, angular rock .. 1 loose, rolling stones
    mix: np.ndarray  # terrain proportions


@dataclass
class Site:
    seed: int
    params: SiteParams
    terrain: np.ndarray   # (N, N) int, OBSTACLE for hard obstacles
    height: np.ndarray
    slope: np.ndarray     # deg
    moisture: np.ndarray
    veg_h: np.ndarray
    veg_d: np.ndarray
    rock: np.ndarray
    slip: np.ndarray
    p_stuck: np.ndarray


def _smooth(rng, sigma):
    g = gaussian_filter(rng.normal(size=(N, N)), sigma, mode="wrap")
    return (g - g.mean()) / (g.std() + 1e-9)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def sample_params(rng) -> SiteParams:
    return SiteParams(
        wetness=float(rng.beta(2, 2)),
        clay=float(rng.uniform(0, 1)),
        season=float(rng.uniform(0, 1)),
        gain=float(rng.lognormal(0, 0.4)),
        h_bias=float(rng.normal(0, 0.05)),
        stiff=float(rng.uniform(0, 1)),
        loose=float(rng.uniform(0, 1)),
        mix=rng.dirichlet(2.5 * np.array([3, 3, 2, 0.8, 1.6, 1.6, 1.0])),
    )


def make_site(seed: int) -> Site:
    rng = np.random.default_rng(seed)
    p = sample_params(rng)

    # Terrain relief: scale so the 90th percentile slope is about 12 degrees.
    h = _smooth(rng, 14) + 0.25 * _smooth(rng, 4)
    gy, gx = np.gradient(h, CELL)
    gmag = np.hypot(gx, gy)
    scale = np.tan(np.radians(12)) / np.percentile(gmag, 90)
    h = h * scale
    slope = np.degrees(np.arctan(gmag * scale))
    hn = (h - h.mean()) / (h.std() + 1e-9)

    moisture = np.clip(p.wetness + 0.2 * _smooth(rng, 8) - 0.15 * hn, 0, 1)

    # Terrain types: argmax of smooth random fields, biased by proportions,
    # moisture (mud in wet low ground, sand where dry) and slope (rubble).
    fields = np.stack([_smooth(rng, 6) for _ in range(7)])
    fields += 0.8 * np.log(p.mix * 7)[:, None, None]
    fields[T_MUD] += 2.0 * (moisture - 0.5)
    fields[T_SAND] += 1.2 * (0.5 - moisture)
    fields[T_RUBBLE] += 0.06 * (slope - 6)
    fields[T_TRACK] = -9  # tracks are drawn explicitly below
    terrain = fields.argmax(0)

    # One or two winding dirt tracks.
    for _ in range(rng.integers(1, 3)):
        t = np.linspace(0, 1, 400)
        horizontal = rng.random() < 0.5
        a = rng.uniform(15, N - 15)
        wiggle = np.cumsum(rng.normal(0, 0.8, size=t.size))
        wiggle = gaussian_filter(wiggle, 25)
        line = np.clip(a + wiggle - wiggle.mean(), 3, N - 4)
        along = t * (N - 1)
        for u, v in zip(along, line):
            r, c = (int(v), int(u)) if horizontal else (int(u), int(v))
            terrain[max(r - 2, 0): r + 2, max(c - 2, 0): c + 2] = T_TRACK

    # Vegetation and rocks.
    veg_h = np.clip(0.3 + 0.6 * p.season + 0.15 * _smooth(rng, 5), 0.2, 1.3)
    veg_d = np.clip(0.25 + 0.6 * p.season + 0.2 * _smooth(rng, 5), 0, 1)
    bush_h = np.clip(0.9 + 0.25 * _smooth(rng, 3), 0.5, 1.6)
    veg_h = np.where(terrain == T_BUSH, bush_h, veg_h)
    rock = np.clip(0.17 + 0.09 * _smooth(rng, 4), 0.03, 0.45)

    # Latent slip per terrain type. Site parameters enter through moisture,
    # clay and vegetation: the same lidar signature can mean different slip.
    s = np.zeros((N, N))
    m, c = moisture, p.clay
    s = np.where(terrain == T_TRACK, 0.05 + 0.6 * m**2 * c + 0.01 * slope, s)
    s = np.where(terrain == T_SGRASS, 0.06 + 0.55 * m * c * slope / 12 + 0.15 * m * slope / 20, s)
    s = np.where(terrain == T_TGRASS, 0.08 + 0.75 * _sig((veg_d * veg_h * (0.4 + 1.2 * p.stiff) - 0.42) / 0.05) + 0.1 * m, s)
    s = np.where(terrain == T_BUSH, 0.88 + 0.05 * _smooth(rng, 2), s)
    s = np.where(terrain == T_MUD, 0.12 + 0.8 * _sig((m * (0.3 + c) - 0.45) / 0.07), s)
    s = np.where(terrain == T_SAND, 0.08 + 0.7 * _sig(((1 - m) * (1 - 0.6 * c) * slope - 4.5) / 1.2), s)
    s = np.where(terrain == T_RUBBLE, 0.08 + 0.75 * _sig((rock - (0.26 - 0.14 * p.loose)) / 0.035) + 0.008 * slope, s)
    slip = np.clip(s, 0, 1)

    # Hard obstacles: trees and boulders, never on the track.
    n_obs = rng.integers(60, 110)
    yy, xx = np.mgrid[0:N, 0:N]
    for _ in range(n_obs):
        cy, cx = rng.uniform(0, N, 2)
        rad = rng.uniform(0.6, 1.6) / CELL
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad**2
        mask &= terrain != T_TRACK
        terrain = np.where(mask, OBSTACLE, terrain)

    p_stuck = 0.25 * _sig((slip - 0.6) / 0.05)
    p_stuck = np.where(terrain == OBSTACLE, 1.0, p_stuck)

    return Site(seed, p, terrain, h, slope, moisture, veg_h, veg_d, rock, slip, p_stuck)


def observe(site: Site, rows: np.ndarray, cols: np.ndarray, ranges: np.ndarray, rng) -> np.ndarray:
    """Simulated lidar statistics for the given cells, observed at the given ranges."""
    p = site.params
    t = site.terrain[rows, cols]
    m = site.moisture[rows, cols]
    vh = site.veg_h[rows, cols]
    vd = site.veg_d[rows, cols]
    rk = site.rock[rows, cols]
    sl = site.slope[rows, cols]
    k = len(rows)

    n = rng.poisson(900.0 / ranges**2)
    sig_h = 0.01 + 0.003 * ranges

    base_h = np.select(
        [t == T_TRACK, t == T_SGRASS, t == T_TGRASS, t == T_BUSH, t == T_MUD, t == T_SAND, t == T_RUBBLE],
        [0.02, 0.05 + 0.05 * p.season, vh, vh, 0.02, 0.03, rk], 0.02)
    seen = np.clip(0.55 + 0.08 * n, 0, 1)  # sparse scans miss vegetation tips
    h_max = np.maximum(base_h * seen + np.abs(rng.normal(0, sig_h, k)) + p.h_bias, 0)

    std_base = np.select(
        [t == T_TGRASS, t == T_BUSH, t == T_RUBBLE, t == T_SGRASS], [0.28 * vh, 0.3 * vh, 0.45 * rk, 0.02], 0.008)
    h_std = std_base * rng.lognormal(0, 0.15, k) + np.abs(rng.normal(0, sig_h / 2, k))

    rough_base = np.select(
        [t == T_RUBBLE, t == T_TGRASS, t == T_BUSH, t == T_SAND, t == T_SGRASS, t == T_TRACK],
        [0.32 * rk, 0.12 * vh, 0.22 * vh, 0.012 + 0.002 * sl, 0.018, 0.01], 0.007)
    roughness = rough_base * rng.lognormal(0, 0.2, k) + np.abs(rng.normal(0, sig_h / 2, k))

    veg = (t == T_TGRASS) | (t == T_BUSH)
    slope_obs = sl + rng.normal(0, 1.0 + 8 / np.sqrt(np.maximum(n, 1)), k) + veg * rng.normal(0, 3, k)
    slope_obs = np.abs(slope_obs)

    p_me = np.select([t == T_TGRASS, t == T_BUSH, t == T_SGRASS], [0.5 + 0.2 * (1 - vd), 0.33, 0.08], 0.02)
    p_gr = np.select([t == T_TGRASS, t == T_BUSH, t == T_SGRASS], [1 - 0.85 * vd, 0.1, 0.85], 0.97)
    nn = np.maximum(n, 1)
    multi_echo = rng.binomial(nn, p_me) / nn
    ground_frac = rng.binomial(nn, p_gr) / nn

    refl = np.select(
        [t == T_TRACK, t == T_SGRASS, t == T_TGRASS, t == T_BUSH, t == T_MUD, t == T_SAND, t == T_RUBBLE],
        [0.32, 0.55, 0.62 - 0.15 * p.season, 0.5, 0.2, 0.5 + 0.12 * (1 - p.clay), 0.4], 0.3)
    soil = ~veg
    refl = refl * np.where(soil, 1 - 0.45 * m, 1 - 0.1 * m)
    int_mean = p.gain * refl * (1 - 0.012 * ranges) + rng.normal(0, 0.02 + 0.08 / np.sqrt(nn), k)
    tex = np.select([veg, t == T_RUBBLE], [0.12, 0.09], 0.04)
    int_std = p.gain * tex * rng.lognormal(0, 0.2, k)

    X = np.column_stack([ranges, n, h_max, h_std, roughness, slope_obs, multi_echo, ground_frac, int_mean, int_std])
    X[n < 3, 3] = np.nan            # h_std
    X[n < 4, 4] = np.nan            # roughness
    X[n < 4, 5] = np.nan            # slope
    X[n < 1, 2:] = np.nan           # nothing seen
    return X.astype(np.float64)


def measure_labels(site: Site, rows, cols, rng):
    """Proprioceptive label after driving over the cell: measured slip above threshold."""
    measured = site.slip[rows, cols] + rng.normal(0, 0.05, len(rows))
    return (measured > SLIP_THRESHOLD).astype(int)


def sample_cells(site: Site, rng, n: int, region: str):
    """Random drivable (non-obstacle) cells from the calibration (left) or test (right) half."""
    cols_lo, cols_hi = (0, N // 2) if region == "calib" else (N // 2, N)
    rr, cc = np.nonzero(site.terrain[:, cols_lo:cols_hi] != OBSTACLE)
    cc = cc + cols_lo
    idx = rng.choice(len(rr), size=min(n, len(rr)), replace=False)
    return rr[idx], cc[idx]


def make_dataset(site: Site, n: int, region: str, rng, range_lo=2.0, range_hi=20.0):
    r, c = sample_cells(site, rng, n, region)
    ranges = rng.uniform(range_lo, range_hi, len(r))
    X = observe(site, r, c, ranges, rng)
    y = measure_labels(site, r, c, rng)
    return X, y, r, c
