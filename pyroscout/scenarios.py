"""Ready-made scenarios so the demo and the tests share one world.

``search_rescue_world`` lays out a small three-room "building": the robot starts
in the left room, the victim (heat source) is in the right room, and the two
interior walls have *offset* doorways so the robot has to wind through the middle
room — and cannot see the victim's heat until it rounds the corners.
"""

from __future__ import annotations

import math

from .control import PurePursuit
from .geometry import Pose
from .mapping import OccupancyGrid
from .navigator import Navigator
from .robot import DiffDriveRobot
from .sensors import Lidar2D, ThermalSensor
from .world import HeatSource, Rectangle, World


def search_rescue_world() -> tuple[World, Pose]:
    """Return the demo ``World`` and the robot's start ``Pose``."""
    walls = [
        # Wall A at x=5 with a doorway at y in [5.0, 7.0].
        Rectangle(4.9, 0.0, 0.2, 5.0),
        Rectangle(4.9, 7.0, 0.2, 5.0),
        # Wall B at x=10 with an offset doorway near the bottom, y in [2.0, 4.0].
        Rectangle(9.9, 0.0, 0.2, 2.0),
        Rectangle(9.9, 4.0, 0.2, 8.0),
        # A pillar in the middle room for good measure.
        Rectangle(7.0, 7.5, 1.0, 1.0),
    ]
    heat = [HeatSource(14.0, 10.5, intensity=100.0)]
    world = World(16.0, 12.0, obstacles=walls, heat_sources=heat)
    start = Pose(2.0, 6.0, 0.0)
    return world, start


def demo_navigator(seed: int = 0) -> tuple[Navigator, World, DiffDriveRobot]:
    """Fully wired navigator for the search-and-rescue demo."""
    world, start = search_rescue_world()
    robot = DiffDriveRobot(
        start.x, start.y, start.theta, radius=0.25, v_max=0.8, omega_max=2.5
    )
    lidar = Lidar2D(num_beams=140, fov=2 * math.pi, max_range=8.0, noise_std=0.02, seed=seed)
    thermal = ThermalSensor(
        fov=math.pi,
        max_range=30.0,
        reference_intensity=100.0,
        intensity_noise=0.05,
        bearing_noise=0.03,
        seed=seed,
    )
    grid = OccupancyGrid(world.width, world.height, resolution=0.1)
    controller = PurePursuit(lookahead=0.45, v_nominal=0.6, omega_max=2.5, goal_tolerance=0.25)
    nav = Navigator(
        robot, world, lidar, thermal,
        grid=grid, controller=controller, dt=0.1, reach_radius=0.5,
        replan_every=5, safety_margin=0.2,
    )
    return nav, world, robot
