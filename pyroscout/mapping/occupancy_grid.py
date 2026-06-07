"""A log-odds occupancy grid built online from LIDAR scans.

The grid stores, per cell, the *log-odds* of that cell being occupied::

    l = log( p(occupied) / p(free) )

Log-odds are convenient because Bayesian updates become simple additions: every
beam that passes *through* a cell subtracts a "free" increment, and the cell a
beam *lands in* gets an "occupied" increment.  Starting from ``l = 0``
(``p = 0.5``, i.e. unknown), the map sharpens towards 0 or 1 as evidence builds.

On top of the raw map we provide two planning-oriented views:

* :meth:`costmap` — a boolean "is this cell blocked?" grid, with obstacles
  *inflated* by the robot radius so a point-robot planner produces body-safe
  paths.
* :meth:`find_frontiers` — free cells that border still-unknown space, the
  classic target set for autonomous exploration.
"""

from __future__ import annotations

import math

import numpy as np


def bresenham(x0: int, y0: int, x1: int, y1: int):
    """Integer cells on the line from ``(x0, y0)`` to ``(x1, y1)`` inclusive."""
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return cells


class OccupancyGrid:
    """A 2D probabilistic occupancy grid aligned with the world origin."""

    def __init__(
        self,
        width: float,
        height: float,
        resolution: float = 0.1,
        l_occ: float = 0.85,
        l_free: float = 0.4,
        l_clamp: float = 8.0,
    ):
        self.width = float(width)
        self.height = float(height)
        self.resolution = float(resolution)
        self.nx = int(math.ceil(self.width / self.resolution))
        self.ny = int(math.ceil(self.height / self.resolution))
        self.log_odds = np.zeros((self.ny, self.nx), dtype=float)

        self.l_occ = float(l_occ)
        self.l_free = float(l_free)
        self.l_clamp = float(l_clamp)

    # --- coordinate transforms -------------------------------------------------
    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        return int(x // self.resolution), int(y // self.resolution)

    def grid_to_world(self, cx: int, cy: int) -> tuple[float, float]:
        return (cx + 0.5) * self.resolution, (cy + 0.5) * self.resolution

    def in_bounds(self, cx: int, cy: int) -> bool:
        return 0 <= cx < self.nx and 0 <= cy < self.ny

    # --- belief views ----------------------------------------------------------
    @property
    def prob(self) -> np.ndarray:
        """Per-cell occupancy probability in ``[0, 1]``."""
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))

    @property
    def known_mask(self) -> np.ndarray:
        """True for cells that have received at least one observation."""
        return self.log_odds != 0.0

    # --- mapping ---------------------------------------------------------------
    def update(self, scan) -> None:
        """Fuse one :class:`~pyroscout.sensors.lidar.LidarScan` into the map."""
        rx, ry = self.world_to_grid(scan.pose.x, scan.pose.y)
        endpoints = scan.endpoints()
        hit_mask = scan.hit_mask

        for (ex, ey), hit in zip(endpoints, hit_mask, strict=True):
            cx, cy = self.world_to_grid(ex, ey)
            ray = bresenham(rx, ry, cx, cy)
            # Every cell the beam crosses before the end is evidence of free space.
            for px, py in ray[:-1]:
                if self.in_bounds(px, py):
                    self.log_odds[py, px] -= self.l_free
            # The final cell is occupied only if the beam actually hit something.
            lx, ly = ray[-1]
            if self.in_bounds(lx, ly):
                if hit:
                    self.log_odds[ly, lx] += self.l_occ
                else:
                    self.log_odds[ly, lx] -= self.l_free

        np.clip(self.log_odds, -self.l_clamp, self.l_clamp, out=self.log_odds)

    # --- planning views --------------------------------------------------------
    def costmap(self, occ_threshold: float = 0.65, inflate_radius: float = 0.0):
        """Boolean grid where ``True`` means "blocked / do not traverse".

        Unknown cells are treated as *free* (optimistic), which lets the planner
        route through unexplored space and replan as the map fills in.

        Parameters
        ----------
        occ_threshold : float
            Probability above which a cell counts as an obstacle.
        inflate_radius : float
            Obstacles are dilated by this many *metres* (typically the robot
            radius plus a safety margin) so planned paths keep the body clear.
        """
        blocked = self.prob > occ_threshold
        r_cells = int(round(inflate_radius / self.resolution))
        if r_cells <= 0:
            return blocked
        return self._dilate(blocked, r_cells)

    def clearance_m(
        self, occ_threshold: float = 0.65, max_cells: int = 12
    ) -> np.ndarray:
        """Approximate distance (in metres) from each cell to the nearest wall.

        Computed with an iterative Chamfer distance transform, capped at
        ``max_cells`` because the planner only cares about clearance up to a
        small "comfort" radius — beyond that, all cells are equally comfortable.
        A* uses this to prefer routes through the *middle* of corridors and
        doorways instead of scraping the corners.
        """
        blocked = self.prob > occ_threshold
        cap = float(max_cells + 1)
        d = np.where(blocked, 0.0, cap)
        a, b = 1.0, math.sqrt(2.0)
        for _ in range(max_cells):
            nd = d.copy()
            nd[1:, :] = np.minimum(nd[1:, :], d[:-1, :] + a)
            nd[:-1, :] = np.minimum(nd[:-1, :], d[1:, :] + a)
            nd[:, 1:] = np.minimum(nd[:, 1:], d[:, :-1] + a)
            nd[:, :-1] = np.minimum(nd[:, :-1], d[:, 1:] + a)
            nd[1:, 1:] = np.minimum(nd[1:, 1:], d[:-1, :-1] + b)
            nd[1:, :-1] = np.minimum(nd[1:, :-1], d[:-1, 1:] + b)
            nd[:-1, 1:] = np.minimum(nd[:-1, 1:], d[1:, :-1] + b)
            nd[:-1, :-1] = np.minimum(nd[:-1, :-1], d[1:, 1:] + b)
            if np.array_equal(nd, d):
                break
            d = nd
        return np.minimum(d, cap) * self.resolution

    @staticmethod
    def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
        """Binary dilation by a disc of radius ``r`` cells."""
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return mask.copy()
        out = mask.copy()
        h, w = mask.shape
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy > r * r:
                    continue
                ny = ys + dy
                nx = xs + dx
                ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
                out[ny[ok], nx[ok]] = True
        return out

    def find_frontiers(
        self, free_threshold: float = 0.35
    ) -> list[tuple[int, int]]:
        """Free cells that touch unknown space — targets for exploration."""
        prob = self.prob
        known = self.known_mask
        free = known & (prob < free_threshold)
        unknown = ~known

        # A cell is a frontier if it is free and any 4-neighbour is unknown.
        neighbour_unknown = np.zeros_like(unknown)
        neighbour_unknown[:-1, :] |= unknown[1:, :]
        neighbour_unknown[1:, :] |= unknown[:-1, :]
        neighbour_unknown[:, :-1] |= unknown[:, 1:]
        neighbour_unknown[:, 1:] |= unknown[:, :-1]

        frontier = free & neighbour_unknown
        ys, xs = np.nonzero(frontier)
        return list(zip(xs.tolist(), ys.tolist(), strict=True))

    def nearest_free_cell(
        self, cx: int, cy: int, occ_threshold: float = 0.65, max_radius: int = 40
    ) -> tuple[int, int] | None:
        """Spiral outwards from ``(cx, cy)`` to the closest non-blocked cell."""
        blocked = self.prob > occ_threshold
        for r in range(max_radius + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue  # only the ring at radius r
                    nx, ny = cx + dx, cy + dy
                    if self.in_bounds(nx, ny) and not blocked[ny, nx]:
                        return nx, ny
        return None
