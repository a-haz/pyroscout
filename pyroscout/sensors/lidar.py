"""A simulated 2D LIDAR (planar laser range finder).

A real spinning LIDAR fires a laser at many angles and times the reflection to
estimate distance.  We reproduce that by casting one ray per beam against the
world geometry and adding a little Gaussian noise to mimic sensor error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..geometry import Pose, cast_rays, wrap_to_pi
from ..world import World


@dataclass
class LidarScan:
    """One LIDAR sweep, expressed relative to the pose it was taken from."""

    pose: Pose
    angles: np.ndarray  # beam offsets relative to robot heading (radians)
    ranges: np.ndarray  # measured distance per beam (metres)
    max_range: float

    @property
    def hit_mask(self) -> np.ndarray:
        """True for beams that actually struck something (vs. timed out)."""
        return self.ranges < self.max_range - 1e-6

    def absolute_angles(self) -> np.ndarray:
        return wrap_to_pi(self.pose.theta + self.angles)

    def endpoints(self) -> np.ndarray:
        """World-frame ``(N, 2)`` coordinates of each beam's end point."""
        a = self.pose.theta + self.angles
        x = self.pose.x + self.ranges * np.cos(a)
        y = self.pose.y + self.ranges * np.sin(a)
        return np.stack([x, y], axis=1)


class Lidar2D:
    """Planar LIDAR sensor.

    Parameters
    ----------
    num_beams : int
        Number of beams in one sweep.
    fov : float
        Angular field of view (radians).  Defaults to a full ``2*pi`` sweep,
        like a typical spinning mobile-robot LIDAR.
    max_range : float
        Maximum sensing distance (metres).
    noise_std : float
        Standard deviation of additive range noise (metres).
    seed : int | None
        Seed for the internal RNG, for reproducible runs.
    """

    def __init__(
        self,
        num_beams: int = 120,
        fov: float = 2.0 * math.pi,
        max_range: float = 8.0,
        noise_std: float = 0.02,
        seed: int | None = None,
    ):
        self.num_beams = int(num_beams)
        self.fov = float(fov)
        self.max_range = float(max_range)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)

        full_circle = self.fov >= 2.0 * math.pi - 1e-6
        self.offsets = np.linspace(
            -self.fov / 2.0,
            self.fov / 2.0,
            self.num_beams,
            endpoint=not full_circle,
        )

    def scan(self, pose: Pose, world: World) -> LidarScan:
        abs_angles = pose.theta + self.offsets
        p1, p2 = world.segments()
        ranges = cast_rays(pose.xy, abs_angles, p1, p2, self.max_range)

        if self.noise_std > 0.0:
            hit = ranges < self.max_range - 1e-6
            noise = self.rng.normal(0.0, self.noise_std, size=ranges.shape) * hit
            ranges = np.clip(ranges + noise, 0.0, self.max_range)

        return LidarScan(pose, self.offsets.copy(), ranges, self.max_range)
