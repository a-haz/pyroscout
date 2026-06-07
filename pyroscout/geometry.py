"""Geometric primitives shared across the simulator.

The single most important routine here is :func:`cast_rays`, a vectorised
ray-vs-segment intersection test.  Both the LIDAR (geometry sensing) and the
thermal sensor (line-of-sight / occlusion checks) are built on top of it, so it
is worth getting right and fast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pose:
    """A 2D rigid-body pose: position ``(x, y)`` and heading ``theta`` (radians)."""

    x: float
    y: float
    theta: float

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


def wrap_to_pi(angle):
    """Wrap an angle (or array of angles) to the half-open interval (-pi, pi].

    Heading errors must be wrapped before they are fed to a controller,
    otherwise a robot pointing at -179 deg and wanting +179 deg would think it
    needs to spin almost all the way around instead of nudging 2 deg.
    """
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def cast_rays(origin, angles, seg_p1, seg_p2, max_range):
    """Cast rays from ``origin`` and return the distance to the nearest segment.

    Solves, for every (ray, segment) pair, the linear system that places a point
    both on the ray ``O + t*d`` (``t >= 0``) and on the segment
    ``p1 + u*(p2 - p1)`` (``0 <= u <= 1``).

    Parameters
    ----------
    origin : array-like, shape (2,)
        Common origin of all rays (the sensor location).
    angles : array-like, shape (B,)
        Absolute ray directions in radians.
    seg_p1, seg_p2 : array-like, shape (N, 2)
        Segment endpoints.
    max_range : float
        Rays that hit nothing closer than this return exactly ``max_range``.

    Returns
    -------
    np.ndarray, shape (B,)
        Distance along each ray to the first segment it strikes, clipped to
        ``max_range``.
    """
    origin = np.asarray(origin, dtype=float)
    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    seg_p1 = np.asarray(seg_p1, dtype=float).reshape(-1, 2)
    seg_p2 = np.asarray(seg_p2, dtype=float).reshape(-1, 2)

    if seg_p1.shape[0] == 0:
        return np.full(angles.shape, float(max_range))

    ox, oy = origin
    dx = np.cos(angles)[:, None]  # (B, 1)
    dy = np.sin(angles)[:, None]  # (B, 1)

    x1 = seg_p1[:, 0][None, :]  # (1, N)
    y1 = seg_p1[:, 1][None, :]
    ex = (seg_p2[:, 0] - seg_p1[:, 0])[None, :]  # (1, N)
    ey = (seg_p2[:, 1] - seg_p1[:, 1])[None, :]

    fx = x1 - ox  # (1, N)
    fy = y1 - oy

    # Cramer's rule on [[dx, -ex], [dy, -ey]] @ [t, u] = [fx, fy].
    det = ex * dy - ey * dx  # (B, N)

    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ex * fy - ey * fx) / det  # (B, N)
        u = (dx * fy - fx * dy) / det  # (B, N)

    eps = 1e-9
    valid = (
        (np.abs(det) > eps)
        & (t >= 0.0)
        & (t <= max_range)
        & (u >= -eps)
        & (u <= 1.0 + eps)
    )

    t = np.where(valid, t, np.inf)
    nearest = np.min(t, axis=1)
    return np.where(np.isfinite(nearest), nearest, float(max_range))


def bearing_to(origin_xy, target_xy) -> float:
    """Absolute bearing (radians) from ``origin_xy`` to ``target_xy``."""
    dx = target_xy[0] - origin_xy[0]
    dy = target_xy[1] - origin_xy[1]
    return math.atan2(dy, dx)
