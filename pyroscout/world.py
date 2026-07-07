"""The simulated environment: walls, obstacles and heat sources.

Obstacles are axis-aligned rectangles, which keeps both ray casting (four
segments each) and collision checking (point-in-inflated-rectangle) cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Rectangle:
    """Axis-aligned rectangle with its lower-left corner at ``(x, y)``."""

    x: float
    y: float
    w: float
    h: float

    def edges(self):
        """Return the four boundary segments as ``(p1, p2)`` tuples."""
        x0, y0, x1, y1 = self.x, self.y, self.x + self.w, self.y + self.h
        c = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        return [(c[i], c[(i + 1) % 4]) for i in range(4)]

    def contains(self, px: float, py: float, margin: float = 0.0) -> bool:
        return (
            self.x - margin <= px <= self.x + self.w + margin
            and self.y - margin <= py <= self.y + self.h + margin
        )


@dataclass(frozen=True)
class HeatSource:
    """A point heat source (e.g. a victim's body heat or a fire).

    ``intensity`` is the notional power at the source; the thermal sensor sees it
    attenuated by distance (see :mod:`pyroscout.sensors.thermal`).
    """

    x: float
    y: float
    intensity: float = 100.0

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


@dataclass
class World:
    """A rectangular arena populated with obstacles and heat sources."""

    width: float
    height: float
    obstacles: list[Rectangle] = field(default_factory=list)
    heat_sources: list[HeatSource] = field(default_factory=list)

    _seg_cache: tuple[np.ndarray, np.ndarray] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def segments(self) -> tuple[np.ndarray, np.ndarray]:
        """All wall segments (arena boundary + obstacle edges).

        Returns two ``(N, 2)`` arrays ``(p1, p2)`` suitable for
        :func:`pyroscout.geometry.cast_rays`.  The result is cached because the
        geometry is static and the sensors query it every single time step.
        """
        if self._seg_cache is not None:
            return self._seg_cache

        segs: list[tuple[tuple[float, float], tuple[float, float]]] = []

        # Arena boundary.
        w, h = self.width, self.height
        boundary = Rectangle(0.0, 0.0, w, h)
        segs.extend(boundary.edges())

        for obs in self.obstacles:
            segs.extend(obs.edges())

        p1 = np.array([s[0] for s in segs], dtype=float)
        p2 = np.array([s[1] for s in segs], dtype=float)
        self._seg_cache = (p1, p2)
        return self._seg_cache

    def in_collision(self, x: float, y: float, radius: float = 0.0) -> bool:
        """True if a disc of ``radius`` at ``(x, y)`` overlaps a wall."""
        if (
            x < radius
            or x > self.width - radius
            or y < radius
            or y > self.height - radius
        ):
            return True
        return any(obs.contains(x, y, margin=radius) for obs in self.obstacles)
