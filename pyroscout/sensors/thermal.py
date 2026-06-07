"""A simulated thermal sensor.

This is the "semantic" sensor: where LIDAR reports *geometry* (distances to
surfaces), the thermal sensor reports *what matters* — the direction of, and a
rough distance to, any heat source it can see.

Physical model
--------------
Radiated power falls off with the square of distance, so we model the intensity
measured from a source of strength ``I0`` at distance ``d`` as::

    measured = I0 / (d**2 + 1)

(The ``+1`` keeps the value finite as ``d -> 0``.)  Because the relationship is
invertible, a calibrated sensor can turn an intensity reading back into a range
estimate::

    d_est = sqrt(I0 / measured - 1)

Two effects make this realistic — and make the navigation problem non-trivial:

* **Occlusion** — walls block infrared, so a source is only detected when the
  sensor has line of sight to it (checked with a LIDAR-style ray cast).
* **Noise** — both the intensity and the bearing readings are corrupted, so the
  resulting position estimate drifts and must be filtered over time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..geometry import Pose, cast_rays, wrap_to_pi
from ..world import World


@dataclass
class ThermalDetection:
    """A single heat detection, in the world frame."""

    bearing: float  # absolute bearing to the source (radians)
    intensity: float  # measured (attenuated) intensity
    est_range: float  # range estimate inverted from intensity

    def estimated_position(self, pose: Pose) -> np.ndarray:
        """Project the detection into world coordinates from ``pose``."""
        return np.array(
            [
                pose.x + self.est_range * math.cos(self.bearing),
                pose.y + self.est_range * math.sin(self.bearing),
            ]
        )


class ThermalSensor:
    """Forward-facing thermal sensor with occlusion and noise.

    Parameters
    ----------
    fov : float
        Field of view (radians); sources outside this cone are not seen.
    max_range : float
        Maximum detection distance (metres).
    reference_intensity : float
        The source strength ``I0`` the sensor is calibrated for, used to invert
        intensity into a range estimate.
    intensity_noise : float
        Relative (fractional) noise on the intensity reading.
    bearing_noise : float
        Standard deviation of bearing noise (radians).
    detect_threshold : float
        Minimum measured intensity to register a detection.
    seed : int | None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        fov: float = math.pi,
        max_range: float = 30.0,
        reference_intensity: float = 100.0,
        intensity_noise: float = 0.05,
        bearing_noise: float = 0.03,
        detect_threshold: float = 0.05,
        seed: int | None = None,
    ):
        self.fov = float(fov)
        self.max_range = float(max_range)
        self.reference_intensity = float(reference_intensity)
        self.intensity_noise = float(intensity_noise)
        self.bearing_noise = float(bearing_noise)
        self.detect_threshold = float(detect_threshold)
        self.rng = np.random.default_rng(seed)

    def _has_line_of_sight(self, pose: Pose, target_xy, world: World) -> bool:
        dx = target_xy[0] - pose.x
        dy = target_xy[1] - pose.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return True
        angle = math.atan2(dy, dx)
        p1, p2 = world.segments()
        hit = cast_rays(pose.xy, [angle], p1, p2, dist + 1.0)[0]
        # A wall in the way means the first hit is closer than the source.
        return hit >= dist - 1e-3

    def sense(self, pose: Pose, world: World) -> list[ThermalDetection]:
        """Return every heat source currently visible to the sensor."""
        detections: list[ThermalDetection] = []
        for src in world.heat_sources:
            dx = src.x - pose.x
            dy = src.y - pose.y
            dist = math.hypot(dx, dy)
            if dist > self.max_range:
                continue

            true_bearing = math.atan2(dy, dx)
            rel = wrap_to_pi(true_bearing - pose.theta)
            if abs(rel) > self.fov / 2.0:
                continue

            if not self._has_line_of_sight(pose, src.xy, world):
                continue

            true_intensity = src.intensity / (dist**2 + 1.0)
            measured = true_intensity * (
                1.0 + self.rng.normal(0.0, self.intensity_noise)
            )
            if measured < self.detect_threshold:
                continue

            # Invert the falloff model to estimate range from intensity.
            ratio = self.reference_intensity / max(measured, 1e-9) - 1.0
            est_range = math.sqrt(max(ratio, 0.0))
            meas_bearing = true_bearing + self.rng.normal(0.0, self.bearing_noise)

            detections.append(
                ThermalDetection(
                    bearing=wrap_to_pi(meas_bearing).item(),
                    intensity=float(measured),
                    est_range=float(est_range),
                )
            )
        return detections

    def field_at(self, xs: np.ndarray, ys: np.ndarray, world: World) -> np.ndarray:
        """Vectorised thermal field over a grid (for visualisation only).

        Ignores occlusion — this is eye-candy for the ground-truth panel, not a
        sensor reading.
        """
        field = np.zeros(np.broadcast(xs, ys).shape, dtype=float)
        for src in world.heat_sources:
            d2 = (xs - src.x) ** 2 + (ys - src.y) ** 2
            field = field + src.intensity / (d2 + 1.0)
        return field
