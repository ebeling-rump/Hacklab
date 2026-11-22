"""Planar object shapes built from boxes (and discs).

Every shape is described in its *body frame*, whose origin is the geometric (area) centroid
of the footprint. The geometric centroid is observable from vision; the centre of mass is
NOT (it is a hidden physical parameter, see objects.py).

A shape provides:
  * `boxes`  - list of (cx, cy, hx, hy) rectangles (body frame) used to build MuJoCo geoms
  * `discs`  - list of (cx, cy, r)
  * `polygon` - shapely footprint (union of the parts)
  * perimeter sampling: point + outward normal for a perimeter parameter s in [0, 1)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from shapely import affinity


@dataclass
class Shape:
    name: str
    boxes: list = field(default_factory=list)   # (cx, cy, hx, hy)
    discs: list = field(default_factory=list)   # (cx, cy, r)

    def __post_init__(self):
        parts = [box(cx - hx, cy - hy, cx + hx, cy + hy) for cx, cy, hx, hy in self.boxes]
        parts += [Point(cx, cy).buffer(r, quad_segs=16) for cx, cy, r in self.discs]
        poly = unary_union(parts)
        if poly.geom_type != "Polygon":  # parts that only touch: fuse them
            poly = poly.buffer(1e-5, join_style=2).buffer(-1e-5, join_style=2)
        c = poly.centroid
        # Shift everything so the geometric centroid is the origin.
        self.boxes = [(cx - c.x, cy - c.y, hx, hy) for cx, cy, hx, hy in self.boxes]
        self.discs = [(cx - c.x, cy - c.y, r) for cx, cy, r in self.discs]
        self.polygon: Polygon = affinity.translate(poly, -c.x, -c.y).simplify(1e-6)
        self._build_perimeter()

    # ------------------------------------------------------------------ geometry
    def _build_perimeter(self):
        ring = np.asarray(self.polygon.exterior.coords)  # closed ring
        # Ensure counter-clockwise orientation so the outward normal is (dy, -dx).
        if Polygon(ring).exterior.is_ccw is False:
            ring = ring[::-1]
        seg = np.diff(ring, axis=0)
        length = np.linalg.norm(seg, axis=1)
        keep = length > 1e-9
        self._p0 = ring[:-1][keep]
        self._seg = seg[keep]
        self._len = length[keep]
        self._cum = np.concatenate([[0.0], np.cumsum(self._len)])
        self.perimeter = float(self._cum[-1])
        t = self._seg / self._len[:, None]
        self._tangent = t
        self._normal = np.stack([t[:, 1], -t[:, 0]], axis=1)  # outward for CCW ring

    def perimeter_point(self, s):
        """Point and outward normal at perimeter parameter s in [0,1). Vectorised."""
        s = np.atleast_1d(np.asarray(s, dtype=float)) % 1.0
        arc = s * self.perimeter
        idx = np.clip(np.searchsorted(self._cum, arc, side="right") - 1, 0, len(self._len) - 1)
        frac = (arc - self._cum[idx])
        pts = self._p0[idx] + self._tangent[idx] * frac[:, None]
        return pts, self._normal[idx], idx

    def edge_info(self, idx):
        return self._p0[idx], self._tangent[idx], self._normal[idx], self._len[idx]

    @property
    def area(self) -> float:
        return float(self.polygon.area)

    @property
    def extents(self):
        minx, miny, maxx, maxy = self.polygon.bounds
        return maxx - minx, maxy - miny

    def uniform_inertia_c(self) -> float:
        """Radius of gyration (sqrt(I_z / m)) for uniform density about the centroid."""
        # Exact polygon second moment via Green's theorem.
        if getattr(self, "_rg", None) is None:
            p = np.asarray(self.polygon.exterior.coords)
            x0, y0, x1, y1 = p[:-1, 0], p[:-1, 1], p[1:, 0], p[1:, 1]
            cr = x0 * y1 - x1 * y0
            A = cr.sum() / 2
            Iz = (cr * (x0 ** 2 + x0 * x1 + x1 ** 2 + y0 ** 2 + y0 * y1 + y1 ** 2)).sum() / 12
            self._rg = float(np.sqrt(abs(Iz / A)))
        return self._rg

    def descriptor(self) -> np.ndarray:
        """Observable geometric descriptor (what a camera could measure).

        [width, height, area, perimeter, radius_of_gyration, convexity(area/hull_area)]
        """
        w, h = self.extents
        hull = self.polygon.convex_hull.area
        return np.array([w, h, self.area, self.perimeter, self.uniform_inertia_c(),
                         self.area / hull])

    def world_polygon(self, pose):
        x, y, th = pose
        return affinity.translate(affinity.rotate(self.polygon, th, origin=(0, 0), use_radians=True), x, y)


# ---------------------------------------------------------------------- library
def rect(w, h, name=None):
    return Shape(name or f"rect_{w:.2f}x{h:.2f}", boxes=[(0, 0, w / 2, h / 2)])


def l_shape(a, b, t, name=None):
    """L with arm lengths a (along x) and b (along y), thickness t."""
    return Shape(name or f"L_{a:.2f}_{b:.2f}_{t:.2f}",
                 boxes=[(a / 2, t / 2, a / 2, t / 2), (t / 2, t + (b - t) / 2, t / 2, (b - t) / 2)])


def t_shape(a, b, t, name=None):
    """T: top bar length a, stem length b, thickness t."""
    return Shape(name or f"T_{a:.2f}_{b:.2f}_{t:.2f}",
                 boxes=[(0, b - t / 2, a / 2, t / 2), (0, (b - t) / 2, t / 2, (b - t) / 2)])


def u_shape(a, b, t, name=None):
    return Shape(name or f"U_{a:.2f}_{b:.2f}_{t:.2f}",
                 boxes=[(0, t / 2, a / 2, t / 2),
                        (-a / 2 + t / 2, t + (b - t) / 2, t / 2, (b - t) / 2),
                        (a / 2 - t / 2, t + (b - t) / 2, t / 2, (b - t) / 2)])


def plus_shape(a, t, name=None):
    return Shape(name or f"plus_{a:.2f}_{t:.2f}",
                 boxes=[(0, 0, a / 2, t / 2), (0, 0, t / 2, a / 2)])


def disc(r, name=None):
    return Shape(name or f"disc_{r:.2f}", discs=[(0, 0, r)])


def z_shape(a, t, name=None):
    return Shape(name or f"Z_{a:.2f}_{t:.2f}",
                 boxes=[(0, 0, t / 2, a / 2), (a / 2 - t / 2, a / 2 - t / 2, a / 2, t / 2),
                        (-a / 2 + t / 2, -a / 2 + t / 2, a / 2, t / 2)])


def random_shape(rng: np.random.Generator, family: str, scale: float = 1.0) -> Shape:
    s = scale
    if family == "rect":
        w = rng.uniform(0.08, 0.20) * s
        return rect(w, w * rng.uniform(0.35, 1.0))
    if family == "square":
        w = rng.uniform(0.08, 0.16) * s
        return rect(w, w, name=f"square_{w:.2f}")
    if family == "L":
        return l_shape(rng.uniform(0.12, 0.20) * s, rng.uniform(0.10, 0.18) * s, rng.uniform(0.04, 0.06) * s)
    if family == "T":
        return t_shape(rng.uniform(0.12, 0.20) * s, rng.uniform(0.10, 0.18) * s, rng.uniform(0.04, 0.06) * s)
    if family == "U":
        return u_shape(rng.uniform(0.14, 0.20) * s, rng.uniform(0.10, 0.16) * s, rng.uniform(0.04, 0.05) * s)
    if family == "plus":
        return plus_shape(rng.uniform(0.12, 0.18) * s, rng.uniform(0.04, 0.06) * s)
    if family == "disc":
        return disc(rng.uniform(0.05, 0.09) * s)
    if family == "Z":
        return z_shape(rng.uniform(0.12, 0.18) * s, rng.uniform(0.04, 0.05) * s)
    raise ValueError(family)


FAMILIES = ["rect", "square", "L", "T", "U", "plus", "disc", "Z"]
