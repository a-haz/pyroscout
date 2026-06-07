
import numpy as np

from pyroscout.control import PurePursuit
from pyroscout.geometry import Pose
from pyroscout.planning import astar


def test_astar_simple_path():
    blocked = np.zeros((5, 5), dtype=bool)
    path = astar(blocked, (0, 0), (4, 4))
    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)


def test_astar_no_path_when_walled_off():
    blocked = np.zeros((5, 5), dtype=bool)
    blocked[:, 2] = True  # full vertical wall
    assert astar(blocked, (0, 0), (4, 0)) is None


def test_astar_routes_around_wall():
    blocked = np.zeros((5, 5), dtype=bool)
    blocked[0:4, 2] = True  # wall with a gap on the last row
    path = astar(blocked, (0, 0), (4, 0))
    assert path is not None
    assert all(not blocked[y, x] for x, y in path)


def test_astar_does_not_cut_corners():
    blocked = np.zeros((3, 3), dtype=bool)
    blocked[0, 1] = True  # (x=1, y=0)
    blocked[1, 0] = True  # (x=0, y=1)
    # The only way out of (0,0) would be a diagonal that clips the corner.
    assert astar(blocked, (0, 0), (1, 1)) is None


def test_astar_cost_grid_prefers_clearance():
    blocked = np.zeros((3, 7), dtype=bool)
    cost = np.zeros((3, 7))
    cost[1, :] = 10.0  # make the straight middle row expensive
    path = astar(blocked, (0, 1), (6, 1), cost_grid=cost)
    assert path is not None
    assert any(y != 1 for _, y in path)  # detours off the costly row


def test_pure_pursuit_drives_straight():
    pp = PurePursuit(lookahead=0.5, v_nominal=0.6)
    path = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    v, omega, done = pp.control(Pose(0, 0, 0.0), path)
    assert v > 0
    assert abs(omega) < 1e-6
    assert not done


def test_pure_pursuit_turns_left_and_right():
    pp = PurePursuit(lookahead=0.5, v_nominal=0.6)
    left = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
    _, omega_l, _ = pp.control(Pose(0, 0, 0.0), left)
    assert omega_l > 0
    right = np.array([[0, 0], [1, -1], [2, -2]], dtype=float)
    _, omega_r, _ = pp.control(Pose(0, 0, 0.0), right)
    assert omega_r < 0


def test_pure_pursuit_reports_done_at_goal():
    pp = PurePursuit(goal_tolerance=0.25)
    path = np.array([[0, 0], [1, 0]], dtype=float)
    v, omega, done = pp.control(Pose(0.95, 0.0, 0.0), path)
    assert done
    assert v == 0.0 and omega == 0.0
