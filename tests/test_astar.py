import math

import numpy as np
import pytest

from pyroscout.planning import astar

SQRT2 = math.sqrt(2.0)


def path_cost(path):
    """Sum of step costs: 1 for a straight step, sqrt(2) for a diagonal."""
    total = 0.0
    for (x0, y0), (x1, y1) in zip(path, path[1:], strict=False):
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        assert max(dx, dy) == 1, "consecutive cells must be neighbours"
        total += SQRT2 if dx and dy else 1.0
    return total


def assert_valid_path(path, blocked, start, goal):
    assert isinstance(path, list)
    assert path[0] == start
    assert path[-1] == goal
    assert all(isinstance(c, tuple) and len(c) == 2 for c in path)
    assert all(not blocked[y, x] for x, y in path)


def test_astar_diagonal_shortest_path_on_open_grid():
    blocked = np.zeros((5, 5), dtype=bool)
    path = astar(blocked, (0, 0), (4, 4))
    assert_valid_path(path, blocked, (0, 0), (4, 4))
    assert path == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    assert path_cost(path) == pytest.approx(4 * SQRT2)


def test_astar_mixed_moves_shortest_cost_8_connected():
    blocked = np.zeros((4, 6), dtype=bool)
    path = astar(blocked, (0, 0), (4, 2))
    assert_valid_path(path, blocked, (0, 0), (4, 2))
    # Two diagonal steps plus two straight steps.
    assert len(path) == 5
    assert path_cost(path) == pytest.approx(2 * SQRT2 + 2.0)


def test_astar_4_connected_uses_manhattan_steps():
    blocked = np.zeros((4, 6), dtype=bool)
    path = astar(blocked, (0, 0), (4, 2), allow_diagonal=False)
    assert_valid_path(path, blocked, (0, 0), (4, 2))
    assert len(path) == 7
    assert path_cost(path) == pytest.approx(6.0)
    for (x0, y0), (x1, y1) in zip(path, path[1:], strict=False):
        assert abs(x1 - x0) + abs(y1 - y0) == 1


def test_astar_cells_are_column_row_order():
    # Non-square grid: shape is (ny, nx) = (2, 5), cells are (cx, cy).
    blocked = np.zeros((2, 5), dtype=bool)
    path = astar(blocked, (0, 0), (4, 1))
    assert_valid_path(path, blocked, (0, 0), (4, 1))
    # (1, 4) would be out of bounds in (column, row) order.
    assert astar(blocked, (0, 0), (1, 4)) is None


def test_astar_start_equals_goal():
    blocked = np.zeros((3, 3), dtype=bool)
    assert astar(blocked, (1, 1), (1, 1)) == [(1, 1)]


def test_astar_routes_around_wall_8_connected():
    blocked = np.zeros((5, 5), dtype=bool)
    blocked[0:4, 2] = True  # wall at column 2, gap only at row 4
    start, goal = (0, 0), (4, 0)
    path = astar(blocked, start, goal)
    assert_valid_path(path, blocked, start, goal)
    wall_cells = {(2, y) for y in range(4)}
    assert not wall_cells & set(path)
    assert (2, 4) in path  # must pass through the gap
    straight = 4.0
    assert path_cost(path) > straight
    # No corner cutting into the gap, so it steps down to row 4 first:
    # one diagonal each end plus eight straight steps.
    assert path_cost(path) == pytest.approx(8.0 + 2 * SQRT2)


def test_astar_routes_around_wall_4_connected():
    blocked = np.zeros((5, 5), dtype=bool)
    blocked[0:4, 2] = True
    path = astar(blocked, (0, 0), (4, 0), allow_diagonal=False)
    assert_valid_path(path, blocked, (0, 0), (4, 0))
    assert (2, 4) in path
    assert path_cost(path) == pytest.approx(12.0)
    assert len(path) == 13


def test_astar_no_path_when_goal_enclosed():
    blocked = np.zeros((5, 5), dtype=bool)
    # Ring of walls around (3, 3); the goal cell itself is free.
    for x in range(2, 5):
        for y in range(2, 5):
            if (x, y) != (3, 3):
                blocked[y, x] = True
    assert not blocked[3, 3]
    assert astar(blocked, (0, 0), (3, 3)) is None
    assert astar(blocked, (0, 0), (3, 3), allow_diagonal=False) is None


def test_astar_no_path_when_goal_occupied():
    blocked = np.zeros((5, 5), dtype=bool)
    blocked[4, 4] = True  # goal cell itself is a wall
    assert astar(blocked, (0, 0), (4, 4)) is None


def test_astar_no_path_when_start_occupied():
    blocked = np.zeros((5, 5), dtype=bool)
    blocked[0, 0] = True
    assert astar(blocked, (0, 0), (4, 4)) is None


def test_astar_no_path_when_goal_out_of_bounds():
    blocked = np.zeros((5, 5), dtype=bool)
    assert astar(blocked, (0, 0), (5, 0)) is None
    assert astar(blocked, (0, 0), (-1, 0)) is None
