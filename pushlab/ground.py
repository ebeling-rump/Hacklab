"""Observable ground features: where the visible ice lies under the object.

The robot can SEE the ice patches (shiny floor), but not how slippery they are or how the object
will react. Per push it gets three numbers, computed from the object's footprint and the patch map:

    frac   share of the footprint on ice (0..1)
    cx, cy centroid of the iced part of the footprint, in the object's body frame (m; 0 if none)

The ice friction itself stays hidden, like mass and the floor friction.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import Point

GROUND_SCALE = np.array([1.0, 0.1, 0.1])
_GRID: dict = {}


def footprint_grid(shape, spacing=0.008):
    """Body-frame sample points inside the object footprint (cached per shape)."""
    key = id(shape)
    if key not in _GRID:
        minx, miny, maxx, maxy = shape.polygon.bounds
        xs = np.arange(minx + spacing / 2, maxx, spacing)
        ys = np.arange(miny + spacing / 2, maxy, spacing)
        P = np.array([(x, y) for x in xs for y in ys if shape.polygon.contains(Point(x, y))])
        _GRID[key] = P
    return _GRID[key]


def on_ice(xy, ice):
    m = np.zeros(len(xy), bool)
    for cx, cy, hx, hy in ice:
        m |= (np.abs(xy[:, 0] - cx) <= hx) & (np.abs(xy[:, 1] - cy) <= hy)
    return m


def ground_features(shape, pose, ice):
    P = footprint_grid(shape)
    c, s = np.cos(pose[2]), np.sin(pose[2])
    W = np.column_stack([pose[0] + c * P[:, 0] - s * P[:, 1], pose[1] + s * P[:, 0] + c * P[:, 1]])
    m = on_ice(W, ice)
    frac = m.mean()
    cx, cy = (P[m].mean(0) if m.any() else (0.0, 0.0))
    return np.array([frac, cx, cy])


def ground_extra(shape, pose, ice, n):
    """(n, 3) scaled feature block for n candidate pushes from the same pose."""
    g = ground_features(shape, pose, ice) / GROUND_SCALE
    return np.repeat(g[None], n, axis=0)
