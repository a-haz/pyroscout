#!/usr/bin/env python3
"""Visualise a single 2D LIDAR sweep against a small world.

Writes ``assets/lidar_scan.png``.  A good sanity check that ray casting works
and a figure for the README's perception section.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from matplotlib.patches import Rectangle as MplRectangle

from pyroscout.geometry import Pose  # noqa: E402
from pyroscout.sensors import Lidar2D  # noqa: E402
from pyroscout.world import Rectangle, World  # noqa: E402


def main() -> None:
    os.makedirs("assets", exist_ok=True)
    world = World(
        12, 8,
        obstacles=[Rectangle(7.5, 4.5, 1.5, 1.5), Rectangle(3.0, 1.0, 0.4, 4.0),
                   Rectangle(8.5, 1.0, 2.0, 0.4)],
    )
    pose = Pose(4.5, 4.0, 0.3)
    scan = Lidar2D(num_beams=180, max_range=8.0, noise_std=0.03, seed=1).scan(pose, world)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.set_facecolor("#0d1b2a")
    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_aspect("equal")
    ax.set_title("2D LIDAR sweep: 180 beams ray-cast against the world", fontsize=11)

    for obs in world.obstacles:
        ax.add_patch(MplRectangle((obs.x, obs.y), obs.w, obs.h,
                                  facecolor="#415a77", edgecolor="#778da9"))

    ends = scan.endpoints()
    segs = [[(pose.x, pose.y), tuple(e)] for e in ends]
    ax.add_collection(LineCollection(segs, colors="#ffd166", linewidths=0.4, alpha=0.6))
    hit = scan.hit_mask
    ax.scatter(ends[hit, 0], ends[hit, 1], s=6, color="#ff5d3b", zorder=4, label="hits")
    ax.add_patch(Circle((pose.x, pose.y), 0.25, facecolor="#3b8df4",
                        edgecolor="white", zorder=5))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("assets/lidar_scan.png", dpi=130, bbox_inches="tight")
    print("Wrote assets/lidar_scan.png")


if __name__ == "__main__":
    main()
