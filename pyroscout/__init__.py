"""PyroScout — Thermal-LIDAR fusion for autonomous search-and-rescue navigation.

A compact, dependency-light 2D robotics simulator that demonstrates the full
mobile-robot autonomy stack:

    perception (LIDAR + thermal) -> mapping -> planning -> control

See the top-level README for the theory behind each layer.
"""

from .control import PurePursuit
from .geometry import Pose, cast_rays, wrap_to_pi
from .mapping import OccupancyGrid
from .navigator import Navigator, NavResult, NavState
from .planning import astar
from .robot import DiffDriveRobot
from .sensors import Lidar2D, ThermalSensor
from .world import HeatSource, Rectangle, World

__version__ = "0.1.0"

__all__ = [
    "Pose",
    "wrap_to_pi",
    "cast_rays",
    "World",
    "Rectangle",
    "HeatSource",
    "DiffDriveRobot",
    "Lidar2D",
    "ThermalSensor",
    "OccupancyGrid",
    "astar",
    "PurePursuit",
    "Navigator",
    "NavResult",
    "NavState",
]
