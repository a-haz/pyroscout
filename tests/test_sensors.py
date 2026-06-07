import math

import numpy as np

from pyroscout.geometry import Pose
from pyroscout.sensors import Lidar2D, ThermalSensor
from pyroscout.world import HeatSource, Rectangle, World


def _lidar(**kw):
    kw.setdefault("noise_std", 0.0)
    return Lidar2D(num_beams=180, fov=2 * math.pi, max_range=10.0, **kw)


def test_lidar_measures_boundary_distance():
    world = World(10, 10)
    scan = _lidar().scan(Pose(5, 5, 0.0), world)
    assert scan.ranges.shape == (180,)
    # Beam index 90 points along +x (offset 0); the wall is 5 m away.
    assert abs(scan.ranges[90] - 5.0) < 0.05
    assert scan.hit_mask.all()


def test_lidar_sees_obstacle_closer_than_wall():
    world = World(10, 10, obstacles=[Rectangle(7.0, 4.0, 0.2, 2.0)])
    scan = _lidar().scan(Pose(5, 5, 0.0), world)
    assert scan.ranges[90] < 2.5  # blocked by the wall at x=7, ~2 m ahead


def test_lidar_endpoints_match_ranges():
    world = World(10, 10)
    scan = _lidar().scan(Pose(5, 5, 0.0), world)
    pts = scan.endpoints()
    d = np.hypot(pts[:, 0] - 5, pts[:, 1] - 5)
    assert np.allclose(d, scan.ranges, atol=1e-6)


def _thermal(**kw):
    kw.setdefault("intensity_noise", 0.0)
    kw.setdefault("bearing_noise", 0.0)
    return ThermalSensor(fov=math.pi, max_range=30.0, reference_intensity=100.0, **kw)


def test_thermal_detects_and_estimates_range():
    world = World(10, 10, heat_sources=[HeatSource(8, 5, intensity=100.0)])
    dets = _thermal().sense(Pose(5, 5, 0.0), world)
    assert len(dets) == 1
    # Source is 3 m ahead; inverting I0/(d^2+1) gives d=3 exactly when noiseless.
    assert abs(dets[0].est_range - 3.0) < 1e-6
    assert abs(dets[0].bearing) < 1e-6
    est = dets[0].estimated_position(Pose(5, 5, 0.0))
    assert np.allclose(est, [8.0, 5.0], atol=1e-6)


def test_thermal_blocked_by_wall():
    world = World(
        10, 10,
        obstacles=[Rectangle(6.5, 4.0, 0.2, 2.0)],
        heat_sources=[HeatSource(8, 5, intensity=100.0)],
    )
    dets = _thermal().sense(Pose(5, 5, 0.0), world)
    assert dets == []  # occluded


def test_thermal_outside_field_of_view():
    world = World(10, 10, heat_sources=[HeatSource(8, 5, intensity=100.0)])
    # Facing away (theta=pi): source is behind the robot.
    dets = _thermal().sense(Pose(5, 5, math.pi), world)
    assert dets == []


def test_thermal_field_decreases_with_distance():
    world = World(10, 10, heat_sources=[HeatSource(5, 5, intensity=100.0)])
    s = _thermal()
    near = s.field_at(np.array(5.5), np.array(5.0), world)
    far = s.field_at(np.array(9.0), np.array(5.0), world)
    assert near > far
