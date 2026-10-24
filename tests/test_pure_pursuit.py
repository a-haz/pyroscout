"""Pure pursuit steering sign tests.

Frame convention used by the controller: standard right-handed world frame,
heading ``theta`` measured counter-clockwise from +x. The heading error is
``alpha = atan2(dy, dx) - theta`` and ``omega = v * 2 * sin(alpha) / L_d``,
so a carrot to the robot's left gives positive (counter-clockwise) omega and
a carrot to the right gives negative (clockwise) omega.
"""

import math

import numpy as np
import pytest

from pyroscout.control import PurePursuit
from pyroscout.geometry import Pose


def make_controller():
    return PurePursuit(lookahead=0.6, v_nominal=0.7, omega_max=2.5, slowdown_radius=0.8)


def test_goal_to_left_gives_positive_omega():
    pp = make_controller()
    path = np.array([[0.0, 0.0], [2.0, 1.0]])
    v, omega, done = pp.control(Pose(0.0, 0.0, 0.0), path)
    assert not done
    assert v > 0
    assert omega > 0


def test_goal_to_right_gives_negative_omega():
    pp = make_controller()
    path = np.array([[0.0, 0.0], [2.0, -1.0]])
    v, omega, done = pp.control(Pose(0.0, 0.0, 0.0), path)
    assert not done
    assert v > 0
    assert omega < 0


def test_left_and_right_are_mirror_images():
    pp = make_controller()
    _, omega_l, _ = pp.control(Pose(0.0, 0.0, 0.0), np.array([[0.0, 0.0], [2.0, 1.0]]))
    _, omega_r, _ = pp.control(Pose(0.0, 0.0, 0.0), np.array([[0.0, 0.0], [2.0, -1.0]]))
    assert omega_l == pytest.approx(-omega_r)


def test_sign_is_relative_to_robot_heading():
    # Robot faces +y. A goal at -x is on its left, a goal at +x is on its right.
    pp = make_controller()
    pose = Pose(0.0, 0.0, math.pi / 2.0)
    _, omega_l, _ = pp.control(pose, np.array([[0.0, 0.0], [-1.0, 2.0]]))
    _, omega_r, _ = pp.control(pose, np.array([[0.0, 0.0], [1.0, 2.0]]))
    assert omega_l > 0
    assert omega_r < 0


def test_omega_matches_pure_pursuit_curvature():
    pp = make_controller()
    path = np.array([[0.0, 0.0], [2.0, 1.0]])
    v, omega, _ = pp.control(Pose(0.0, 0.0, 0.0), path)
    alpha = math.atan2(1.0, 2.0)
    expected_v = 0.7 * math.cos(alpha)  # far from goal, so no slowdown
    assert v == pytest.approx(expected_v)
    assert omega == pytest.approx(expected_v * 2.0 * math.sin(alpha) / 0.6)


def test_straight_path_ahead_gives_near_zero_omega():
    pp = make_controller()
    path = np.array([[x, 0.0] for x in np.linspace(0.0, 5.0, 11)])
    v, omega, done = pp.control(Pose(0.0, 0.0, 0.0), path)
    assert not done
    assert v == pytest.approx(0.7)
    assert abs(omega) < 1e-9


def test_straight_path_ahead_when_heading_is_rotated():
    pp = make_controller()
    theta = math.radians(30.0)
    pts = [[r * math.cos(theta), r * math.sin(theta)] for r in np.linspace(0.0, 5.0, 11)]
    _, omega, done = pp.control(Pose(0.0, 0.0, theta), np.array(pts))
    assert not done
    assert abs(omega) < 1e-9
