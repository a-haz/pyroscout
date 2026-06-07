import math

import numpy as np

from pyroscout.geometry import bearing_to, cast_rays, wrap_to_pi


def test_wrap_to_pi_scalar():
    assert abs(float(wrap_to_pi(2 * math.pi + 0.3)) - 0.3) < 1e-9
    assert abs(float(wrap_to_pi(-2 * math.pi - 0.3)) + 0.3) < 1e-9
    assert abs(float(wrap_to_pi(2.5 * math.pi)) - math.pi / 2) < 1e-9


def test_wrap_to_pi_array():
    out = wrap_to_pi(np.array([0.0, 3 * math.pi / 2, -3 * math.pi / 2]))
    assert np.allclose(out, [0.0, -math.pi / 2, math.pi / 2])


def test_cast_rays_hits_segment_straight_ahead():
    p1 = np.array([[5.0, -1.0]])
    p2 = np.array([[5.0, 1.0]])
    r = cast_rays([0.0, 0.0], [0.0], p1, p2, 100.0)
    assert abs(r[0] - 5.0) < 1e-6


def test_cast_rays_miss_returns_max_range():
    p1 = np.array([[5.0, -1.0]])
    p2 = np.array([[5.0, 1.0]])
    r = cast_rays([0.0, 0.0], [math.pi], p1, p2, 50.0)  # pointing away
    assert r[0] == 50.0


def test_cast_rays_returns_nearest_of_many():
    p1 = np.array([[5.0, -1.0], [3.0, -1.0]])
    p2 = np.array([[5.0, 1.0], [3.0, 1.0]])
    r = cast_rays([0.0, 0.0], [0.0], p1, p2, 100.0)
    assert abs(r[0] - 3.0) < 1e-6


def test_cast_rays_vectorised_over_beams():
    # A unit box around the origin; from the centre every axis-aligned beam is 1 away.
    p1 = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    p2 = np.array([[1, -1], [1, 1], [-1, 1], [-1, -1]], dtype=float)
    angles = [0.0, math.pi / 2, math.pi, -math.pi / 2]
    r = cast_rays([0.0, 0.0], angles, p1, p2, 10.0)
    assert np.allclose(r, 1.0, atol=1e-6)


def test_bearing_to():
    assert abs(bearing_to((0, 0), (1, 1)) - math.pi / 4) < 1e-9
