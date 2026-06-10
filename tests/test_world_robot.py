import math

from pyroscout.robot import DiffDriveRobot
from pyroscout.world import HeatSource, Rectangle, World


def test_rectangle_edges_and_contains():
    r = Rectangle(0, 0, 2, 2)
    assert len(r.edges()) == 4
    assert r.contains(1, 1)
    assert not r.contains(3, 1)
    assert r.contains(2.1, 1, margin=0.2)


def test_world_segments_include_boundary_and_obstacles():
    w = World(10, 10, obstacles=[Rectangle(4, 4, 1, 1)])
    p1, p2 = w.segments()
    # 4 boundary edges + 4 obstacle edges.
    assert p1.shape == (8, 2)
    assert p2.shape == (8, 2)


def test_world_collision():
    w = World(10, 10, obstacles=[Rectangle(4, 4, 2, 2)])
    assert w.in_collision(5, 5, radius=0.2)  # inside obstacle
    assert w.in_collision(0.1, 5, radius=0.2)  # against the boundary
    assert not w.in_collision(2, 2, radius=0.2)  # open space


def test_rectangle_distance_to():
    r = Rectangle(1, 1, 2, 2)
    assert r.distance_to(2, 2) == 0.0  # inside
    assert r.distance_to(3, 2) == 0.0  # on the edge
    assert abs(r.distance_to(4, 2) - 1.0) < 1e-12  # beside an edge
    assert abs(r.distance_to(0, 0) - math.sqrt(2)) < 1e-12  # off a corner


def test_world_collision_is_exact_at_corners():
    w = World(10, 10, obstacles=[Rectangle(4, 4, 2, 2)])
    r = 0.2
    # Diagonally off the corner: inside the inflated *square* but the disc
    # does not actually touch the rectangle (corner distance ~0.25 > r).
    assert not w.in_collision(4 - 0.9 * r, 4 - 0.9 * r, radius=r)
    # Close enough diagonally that the disc really does overlap (~0.14 < r).
    assert w.in_collision(4 - 0.5 * r, 4 - 0.5 * r, radius=r)


def test_heat_source_xy():
    s = HeatSource(3, 4, 50)
    assert tuple(s.xy) == (3.0, 4.0)


def test_robot_straight_line_motion():
    r = DiffDriveRobot(0, 0, 0.0)
    r.step(1.0, 0.0, 1.0)
    assert abs(r.x - 1.0) < 1e-9
    assert abs(r.y) < 1e-9
    assert abs(r.theta) < 1e-9


def test_robot_pure_rotation():
    r = DiffDriveRobot(0, 0, 0.0)
    r.step(0.0, 1.0, 1.0)
    assert abs(r.x) < 1e-9 and abs(r.y) < 1e-9
    assert abs(r.theta - 1.0) < 1e-9


def test_robot_quarter_circle_arc():
    r = DiffDriveRobot(0, 0, 0.0, v_max=10, omega_max=10)
    r.step(1.0, 1.0, math.pi / 2)  # radius 1, quarter turn
    assert abs(r.x - 1.0) < 1e-6
    assert abs(r.y - 1.0) < 1e-6
    assert abs(r.theta - math.pi / 2) < 1e-6


def test_robot_clamps_velocity():
    r = DiffDriveRobot(0, 0, 0.0, v_max=1.0, omega_max=1.0)
    r.step(100.0, 0.0, 1.0)
    assert abs(r.x - 1.0) < 1e-9  # clamped to v_max
