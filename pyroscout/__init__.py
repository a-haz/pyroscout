"""PyroScout: a 2D search-and-rescue simulator fusing LIDAR and thermal sensing.

See the README for how the layers (perception, mapping, planning, control)
fit together.
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
