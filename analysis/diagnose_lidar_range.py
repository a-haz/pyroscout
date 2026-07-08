#!/usr/bin/env python3
"""Behavioural diagnosis of the LIDAR-range failure cliff.

The sensitivity sweep shows a *non-monotonic* cliff: a 3 m LIDAR succeeds ~40%
of the time, but 4.5 m only ~5% (both far below the 100% at >= 6 m). This
script instruments the navigator's internal behaviour to show *what the robot
is doing differently* at each range — it tracks per-step:

* behaviour-state flips (SEARCHING <-> NAVIGATING churn),
* exploration-target churn (how often the frontier target is replaced),
* steps spent rotating in place (no path to follow),
* map coverage growth and distance driven.

Usage
-----
    python analysis/diagnose_lidar_range.py [--seeds 12]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from pyroscout.control import PurePursuit
from pyroscout.mapping import OccupancyGrid
from pyroscout.navigator import Navigator, NavState
from pyroscout.robot import DiffDriveRobot
from pyroscout.scenarios import search_rescue_world
from pyroscout.sensors import Lidar2D, ThermalSensor

MAX_STEPS = 700
DT = 0.1


def diagnose(lidar_range: float, seed: int) -> dict:
    world, start = search_rescue_world()
    robot = DiffDriveRobot(start.x, start.y, start.theta, radius=0.25, v_max=0.8, omega_max=2.5)
    lidar = Lidar2D(
        num_beams=140, fov=2 * math.pi, max_range=lidar_range, noise_std=0.02, seed=seed
    )
    thermal = ThermalSensor(
        fov=math.pi, max_range=30.0, reference_intensity=100.0,
        intensity_noise=0.05, bearing_noise=0.03, seed=seed,
    )
    controller = PurePursuit(lookahead=0.45, v_nominal=0.6, omega_max=2.5, goal_tolerance=0.25)
    nav = Navigator(
        robot, world, lidar, thermal,
        grid=OccupancyGrid(world.width, world.height, resolution=0.1),
        controller=controller, dt=DT, reach_radius=0.5, replan_every=5, safety_margin=0.2,
    )

    state_flips = 0
    target_churn = 0
    rotate_steps = 0
    dist = 0.0
    prev_state = nav.state
    prev_target: np.ndarray | None = None
    prev_xy = robot.pose.xy
    first_detect: int | None = None
    steps = 0

    for i in range(MAX_STEPS):
        rec = nav.step()
        steps = i + 1
        if first_detect is None and rec.detections:
            first_detect = steps
        if nav.state != prev_state:
            state_flips += 1
            prev_state = nav.state
        target = nav.explore_target
        if target is not None and (
            prev_target is None or float(np.hypot(*(target - prev_target))) > 0.3
        ):
            target_churn += 1
            prev_target = target.copy()
        if nav.current_path is None:
            rotate_steps += 1
        now = robot.pose.xy
        dist += float(np.hypot(*(now - prev_xy)))
        prev_xy = now
        if nav.state in (NavState.REACHED, NavState.FAILED):
            break

    return {
        "success": nav.state == NavState.REACHED,
        "steps": steps,
        "state_flips": state_flips,
        "target_churn": target_churn,
        "rotate_steps": rotate_steps,
        "dist": dist,
        "speed": dist / (steps * DT),
        "known": float(nav.grid.known_mask.mean()),
        "first_detect_t": first_detect * DT if first_detect else math.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=12)
    args = parser.parse_args()

    print(f"{'range':>6} {'succ':>5} {'flips':>6} {'churn':>6} {'rot%':>6} "
          f"{'speed':>6} {'known%':>7} {'detect':>7}")
    for lidar_range in (3.0, 4.5, 6.0, 8.0):
        rows = [diagnose(lidar_range, s) for s in range(args.seeds)]
        med = {k: float(np.nanmedian([r[k] for r in rows])) for k in rows[0] if k != "success"}
        succ = sum(r["success"] for r in rows)
        print(
            f"{lidar_range:>6.1f} {succ:>3}/{args.seeds:<2}"
            f"{med['state_flips']:>6.0f} {med['target_churn']:>6.0f} "
            f"{100 * med['rotate_steps'] / med['steps']:>5.0f}% "
            f"{med['speed']:>5.2f}m/s {100 * med['known']:>6.0f}% "
            f"{med['first_detect_t']:>6.1f}s"
        )


if __name__ == "__main__":
    main()
