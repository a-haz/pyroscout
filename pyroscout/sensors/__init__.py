"""Exteroceptive sensors: LIDAR (geometry) and thermal (heat semantics)."""

from .lidar import Lidar2D, LidarScan
from .thermal import ThermalDetection, ThermalSensor

__all__ = ["Lidar2D", "LidarScan", "ThermalSensor", "ThermalDetection"]
