#!/usr/bin/env python3
"""Run the thermal-LIDAR search-and-rescue demo end to end.

Builds the three-room world, runs the autonomous navigator until it reaches the
victim, prints run metrics, and writes a hero PNG and an animated GIF.

Usage
-----
    python examples/run_search_rescue.py                 # seed 0 -> assets/
    python examples/run_search_rescue.py --seed 3        # a different run
    python examples/run_search_rescue.py --no-gif        # skip the (slow) GIF
"""

from __future__ import annotations

import argparse
import os

from pyroscout import viz
from pyroscout.scenarios import demo_navigator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sensor noise")
    parser.add_argument("--max-steps", type=int, default=700, help="step budget")
    parser.add_argument("--out-dir", default="assets", help="output directory")
    parser.add_argument("--no-gif", action="store_true", help="skip GIF rendering")
    parser.add_argument("--fps", type=int, default=12, help="GIF frame rate")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    nav, world, robot = demo_navigator(seed=args.seed)

    print(f"Running thermal-LIDAR search-and-rescue (seed={args.seed}) ...")
    result = nav.run(max_steps=args.max_steps)

    print("\n=== Run report ===")
    print(f"  outcome      : {'REACHED victim' if result.success else result.state.value}")
    print(f"  steps        : {result.steps}  ({result.sim_time:.1f} s simulated)")
    print(f"  path length  : {result.path_length:.2f} m")
    print(f"  collisions   : {result.collisions}")
    if result.goal_estimate is not None:
        true_xy = world.heat_sources[0].xy
        err = float(((result.goal_estimate - true_xy) ** 2).sum() ** 0.5)
        print(f"  victim est.  : ({result.goal_estimate[0]:.2f}, {result.goal_estimate[1]:.2f})")
        print(f"  estimate err : {err * 100:.1f} cm")

    hero = viz.save_snapshot(
        result, world, os.path.join(args.out_dir, "hero.png"), robot_radius=robot.radius
    )
    print(f"\nWrote {hero}")
    if not args.no_gif:
        print("Rendering GIF (this takes a moment) ...")
        gif = viz.animate(
            result,
            world,
            os.path.join(args.out_dir, "demo.gif"),
            robot_radius=robot.radius,
            fps=args.fps,
        )
        print(f"Wrote {gif}")


if __name__ == "__main__":
    main()
