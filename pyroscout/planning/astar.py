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

    def heuristic(x, y):
        return math.hypot(x - gx, y - gy)

    open_heap: list[tuple[float, int, tuple[int, int]]] = []
    counter = 0
    heapq.heappush(open_heap, (heuristic(sx, sy), counter, start))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score = {start: 0.0}
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            return _reconstruct(came_from, current)
        closed.add(current)
        cx, cy = current

        for dx, dy, cost in neighbours:
            nx, ny = cx + dx, cy + dy
            if not valid(nx, ny) or blocked[ny, nx]:
                continue
            # Forbid cutting across a wall corner on diagonal moves.
            if dx != 0 and dy != 0:
                if blocked[cy, nx] or blocked[ny, cx]:
                    continue
            neighbour = (nx, ny)
            if neighbour in closed:
                continue
            tentative = g_score[current] + cost
            if cost_grid is not None:
                tentative += float(cost_grid[ny, nx])
            if tentative < g_score.get(neighbour, math.inf):
                came_from[neighbour] = current
                g_score[neighbour] = tentative
                counter += 1
                f = tentative + heuristic(nx, ny)
                heapq.heappush(open_heap, (f, counter, neighbour))

    return None


def _reconstruct(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
