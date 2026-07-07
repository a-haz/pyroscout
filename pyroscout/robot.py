"""Differential-drive robot kinematics.

Commands are a forward velocity ``v`` and turn rate ``omega``; the pose
evolves as ``x' = v cos(theta)``, ``y' = v sin(theta)``, ``theta' = omega``.
Integrated in closed form per step, which is exact for piecewise-constant
commands (the robot traces true arcs, unlike Euler integration).
"""

from __future__ import annotations

import math

from .geometry import Pose, wrap_to_pi


class DiffDriveRobot:
    """A unicycle / differential-drive robot.

    Parameters
    ----------
    x, y, theta : float
        Initial pose.
    radius : float
        Body radius, used by the simulator for collision checking.
    v_max : float
        Maximum linear speed (m/s); commands are clamped to ``[-v_max, v_max]``.
    omega_max : float
        Maximum turn rate (rad/s); commands are clamped to
        ``[-omega_max, omega_max]``.
    """

    def __init__(
        self,
        x: float,
        y: float,
        theta: float = 0.0,
        radius: float = 0.25,
        v_max: float = 1.0,
        omega_max: float = 2.5,
    ):
        self.x = float(x)
        self.y = float(y)
        self.theta = wrap_to_pi(theta).item()
        self.radius = float(radius)
        self.v_max = float(v_max)
        self.omega_max = float(omega_max)

    @property
    def pose(self) -> Pose:
        return Pose(self.x, self.y, self.theta)

    def step(self, v: float, omega: float, dt: float) -> Pose:
        """Advance the pose by one control step and return the new pose."""
        v = max(-self.v_max, min(self.v_max, float(v)))
        omega = max(-self.omega_max, min(self.omega_max, float(omega)))

        if abs(omega) < 1e-6:
            # Straight-line motion (avoids a divide-by-zero in the arc form).
            self.x += v * math.cos(self.theta) * dt
            self.y += v * math.sin(self.theta) * dt
        else:
            # Exact integration of a constant-curvature arc.
            radius = v / omega
            dtheta = omega * dt
            self.x += radius * (math.sin(self.theta + dtheta) - math.sin(self.theta))
            self.y -= radius * (math.cos(self.theta + dtheta) - math.cos(self.theta))
            self.theta = wrap_to_pi(self.theta + dtheta).item()

        return self.pose
