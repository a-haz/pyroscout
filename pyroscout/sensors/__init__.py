"""Simulated sensors: LIDAR and thermal."""

from .lidar import Lidar2D, LidarScan
from .thermal import ThermalDetection, ThermalSensor

__all__ = ["Lidar2D", "LidarScan", "ThermalSensor", "ThermalDetection"]
