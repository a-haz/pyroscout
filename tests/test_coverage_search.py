"""Tests for the coverage-search fallback (frontier-exhausted, victim unseen).

With a short-range thermal sensor the robot can map the whole building without
ever getting a detection.  Plain frontier exploration then has nothing left to
do; the coverage-search fallback must sweep the sensor over the mapped free
space until the victim is found.
"""

import math

import numpy as np

from pyroscout.control import PurePursuit
from pyroscout.geometry import Pose
from pyroscout.mapping import OccupancyGrid
from pyroscout.navigator import Navigator, NavState
from pyroscout.robot import DiffDriveRobot
from pyroscout.scenarios import demo_navigator
from pyroscout.sensors import Lidar2D, ThermalSensor
from pyroscout.world import HeatSource, Rectangle, World


def test_short_range_thermal_reaches_victim_via_coverage_search():
    # 3 m range cannot see the victim from most of the building: frontier
    # exploration alone never finds it (see companion test below).
    nav, world, _ = demo_navigator(seed=7, thermal_range=3.0)
    result = nav.run(max_steps=1200)

    assert result.success
    assert result.state == NavState.REACHED
    err = float(np.hypot(*(result.goal_estimate - world.heat_sources[0].xy)))
    assert err < 0.3


def test_short_range_thermal_fails_without_coverage_search():
    # The pre-fallback behaviour this feature exists to fix: same episode,
    # fallback disabled -> the victim is never detected.
    nav, _, _ = demo_navigator(seed=7, thermal_range=3.0, coverage_search=False)
    result = nav.run(max_steps=1200)

    assert not result.success
    assert not nav.victim_seen


def test_sweep_marking_respects_walls_and_field_of_view():
    # A wall splits the arena; the robot faces +x from the left half.
    world = World(
        6.0, 4.0,
        obstacles=[Rectangle(2.9, 0.0, 0.2, 4.0)],
        heat_sources=[HeatSource(5.0, 2.0)],
    )
    robot = DiffDriveRobot(1.5, 2.0, 0.0, radius=0.25, v_max=0.8, omega_max=2.5)
    lidar = Lidar2D(num_beams=180, fov=2 * math.pi, max_range=8.0, noise_std=0.0, seed=0)
    thermal = ThermalSensor(fov=math.pi, max_range=3.0, seed=0)
    nav = Navigator(
        robot, world, lidar, thermal,
        grid=OccupancyGrid(world.width, world.height, resolution=0.1),
        controller=PurePursuit(), dt=0.1,
    )

    # Sense and mark once from a known pose.
    scan = lidar.scan(Pose(1.5, 2.0, 0.0), world)
    nav.grid.update(scan)
    nav._mark_swept()

    def bin_of(x, y):
        return int(x // nav._cov_res), int(y // nav._cov_res)

    # In front, same side of the wall, within range: swept.
    assert bin_of(2.5, 2.0) in nav._swept
    # Same distance but behind the wall: the sensor never saw it.
    assert bin_of(3.7, 2.0) not in nav._swept
    # Behind the robot (outside the forward FOV): not swept either.
    assert bin_of(0.6, 2.0) not in nav._swept
