"""A* search over an occupancy costmap.

A* finds the lowest-cost path on a graph by expanding nodes in order of
``f = g + h`` where ``g`` is the cost so far and ``h`` is an *admissible*
(never-overestimating) estimate of the cost to the goal.  On a grid with
8-connectivity, straight-line (Euclidean) distance is admissible, so the path
A* returns is guaranteed optimal.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

SQRT2 = math.sqrt(2.0)

# (dx, dy, step_cost)
_NEIGHBOURS_8 = [
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, SQRT2),
    (1, -1, SQRT2),
    (-1, 1, SQRT2),
    (-1, -1, SQRT2),
]
_NEIGHBOURS_4 = _NEIGHBOURS_8[:4]


def astar(
    blocked: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    allow_diagonal: bool = True,
    cost_grid: np.ndarray | None = None,
):
    """Plan a path of grid cells from ``start`` to ``goal``.

    Parameters
    ----------
    blocked : np.ndarray (bool), shape (ny, nx)
        ``True`` marks impassable cells.
    start, goal : (cx, cy)
        Start and goal cells in (column, row) order.
    allow_diagonal : bool
        Permit 8-connected moves.  Diagonal moves that would cut a wall corner
        are forbidden.
    cost_grid : np.ndarray | None, shape (ny, nx)
        Optional non-negative extra cost added for *entering* each cell — e.g. a
        clearance penalty that biases the path away from walls.  Because the
        added cost is non-negative, the Euclidean heuristic stays admissible.

    Returns
    -------
    list[tuple[int, int]] | None
        Cells from start to goal inclusive, or ``None`` if no path exists.
    """
    h_grid, w_grid = blocked.shape
    sx, sy = start
    gx, gy = goal

    def valid(x, y):
        return 0 <= x < w_grid and 0 <= y < h_grid

    if not (valid(sx, sy) and valid(gx, gy)):
        return None
    if blocked[sy, sx] or blocked[gy, gx]:
        return None
    if start == goal:
        return [start]

    neighbours = _NEIGHBOURS_8 if allow_diagonal else _NEIGHBOURS_4

    # Dense per-cell bookkeeping (g, closed, parent) instead of dicts and
    # sets: the algorithm is identical, but the hot loop skips the tuple
    # hashing, which roughly halves planning time on this grid size.
    g_score = np.full((h_grid, w_grid), np.inf)
    g_score[sy, sx] = 0.0
    closed = np.zeros((h_grid, w_grid), dtype=bool)
    parent_x = np.full((h_grid, w_grid), -1, dtype=np.int32)
    parent_y = np.full((h_grid, w_grid), -1, dtype=np.int32)

    open_heap: list[tuple[float, int, int, int]] = []
    counter = 0
    heapq.heappush(open_heap, (math.hypot(sx - gx, sy - gy), counter, sx, sy))

    while open_heap:
        _, _, cx, cy = heapq.heappop(open_heap)
        if closed[cy, cx]:
            continue
        if cx == gx and cy == gy:
            return _reconstruct(parent_x, parent_y, gx, gy)
        closed[cy, cx] = True
        g_current = g_score[cy, cx]

        for dx, dy, cost in neighbours:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w_grid and 0 <= ny < h_grid) or blocked[ny, nx]:
                continue
            # Forbid cutting across a wall corner on diagonal moves.
            if dx != 0 and dy != 0:
                if blocked[cy, nx] or blocked[ny, cx]:
                    continue
            if closed[ny, nx]:
                continue
            tentative = g_current + cost
            if cost_grid is not None:
                tentative += float(cost_grid[ny, nx])
            if tentative < g_score[ny, nx]:
                parent_x[ny, nx] = cx
                parent_y[ny, nx] = cy
                g_score[ny, nx] = tentative
                counter += 1
                f = tentative + math.hypot(nx - gx, ny - gy)
                heapq.heappush(open_heap, (f, counter, nx, ny))

    return None


def _reconstruct(parent_x, parent_y, x, y):
    path = [(x, y)]
    while parent_x[y, x] >= 0:
        x, y = int(parent_x[y, x]), int(parent_y[y, x])
        path.append((x, y))
    path.reverse()
    return path
