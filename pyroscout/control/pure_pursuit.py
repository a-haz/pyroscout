"""A pure-pursuit path-following controller.

Pure pursuit steers a robot towards a "carrot" point a fixed *lookahead*
distance ahead on the path.  The geometry of driving an arc to that point gives
a steering curvature::

    kappa = 2 * sin(alpha) / L_d

where ``alpha`` is the heading error to the carrot and ``L_d`` the lookahead.
Turn rate is then ``omega = v * kappa``.  A larger lookahead yields smoother but
looser tracking; a smaller one tracks tightly but can oscillate.

We add two practical guards:

* if the carrot is well off to the side (or behind), slow down and turn nearly
  in place rather than swinging wide;
* slow down on the approach to the goal and report ``done`` inside a tolerance.
"""

from __future__ import annotations

import math

import numpy as np

from ..geometry import Pose, wrap_to_pi


class PurePursuit:
    def __init__(
        self,
        lookahead: float = 0.6,
        v_nominal: float = 0.7,
        omega_max: float = 2.5,
        goal_tolerance: float = 0.25,
        slowdown_radius: float = 0.8,
    ):
        self.lookahead = float(lookahead)
        self.v_nominal = float(v_nominal)
        self.omega_max = float(omega_max)
        self.goal_tolerance = float(goal_tolerance)
        self.slowdown_radius = float(slowdown_radius)

    def control(self, pose: Pose, path) -> tuple[float, float, bool]:
        """Return ``(v, omega, done)`` to follow ``path`` from ``pose``.

        ``path`` is an ``(N, 2)`` array of world-frame waypoints.
        """
        path = np.asarray(path, dtype=float).reshape(-1, 2)
        if path.shape[0] == 0:
            return 0.0, 0.0, True

        pos = pose.xy
        goal = path[-1]
        dist_to_goal = float(np.hypot(*(goal - pos)))
        if dist_to_goal <= self.goal_tolerance:
            return 0.0, 0.0, True

        target = self._lookahead_point(pos, path)

        # Heading error to the carrot.
        alpha = wrap_to_pi(
            math.atan2(target[1] - pos[1], target[0] - pos[0]) - pose.theta
        ).item()

        if abs(alpha) > math.pi / 2.0:
            # Carrot is behind us: rotate towards it before driving.
            v = 0.05
            omega = math.copysign(self.omega_max, alpha)
            return v, float(np.clip(omega, -self.omega_max, self.omega_max)), False

        # Pure-pursuit curvature -> turn rate.
        v = self.v_nominal
        if dist_to_goal < self.slowdown_radius:
            v *= max(0.25, dist_to_goal / self.slowdown_radius)
        # Ease off the throttle while turning hard so we don't swing wide.
        v *= max(0.3, math.cos(alpha))

        curvature = 2.0 * math.sin(alpha) / self.lookahead
        omega = float(np.clip(v * curvature, -self.omega_max, self.omega_max))
        return v, omega, False

    def _lookahead_point(self, pos: np.ndarray, path: np.ndarray) -> np.ndarray:
        """Pick the carrot: first waypoint at least ``lookahead`` away.

        We start the search from the path point closest to the robot so that
        progress already made along the path is not re-followed.
        """
        dists = np.hypot(path[:, 0] - pos[0], path[:, 1] - pos[1])
        nearest = int(np.argmin(dists))
        for i in range(nearest, path.shape[0]):
            if dists[i] >= self.lookahead:
                return path[i]
        return path[-1]
