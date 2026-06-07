#!/usr/bin/env python3
"""Compare a naive shortest A* path with the clearance-aware planner.

Rasterises the demo building into an occupancy grid, then plans from the start
to the victim twice: once minimising distance only (hugs corners) and once with
the clearance cost field (stays in the middle of doorways).  Writes
``assets/planning.png``.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pyroscout.mapping import OccupancyGrid  # noqa: E402
from pyroscout.navigator import _nearest_free  # noqa: E402
from pyroscout.planning import astar  # noqa: E402
from pyroscout.scenarios import search_rescue_world  # noqa: E402


def rasterise(world, resolution=0.1) -> OccupancyGrid:
    """Turn the ground-truth world into a fully-known occupancy grid."""
    g = OccupancyGrid(world.width, world.height, resolution)
    for cy in range(g.ny):
        for cx in range(g.nx):
            wx, wy = g.grid_to_world(cx, cy)
            g.log_odds[cy, cx] = g.l_clamp if world.in_collision(wx, wy) else -g.l_clamp
    return g


def plan(grid, start_xy, goal_xy, inflate, clearance_weight, comfort):
    blocked = grid.costmap(inflate_radius=inflate)
    start = _nearest_free(blocked, grid.world_to_grid(*start_xy))
    goal = _nearest_free(blocked, grid.world_to_grid(*goal_xy))
    cost = None
    if clearance_weight > 0:
        clr = grid.clearance_m()
        cost = clearance_weight * np.maximum(0.0, comfort - clr)
    cells = astar(blocked, start, goal, cost_grid=cost)
    return np.array([grid.grid_to_world(cx, cy) for cx, cy in cells])


def main() -> None:
    os.makedirs("assets", exist_ok=True)
    world, start = search_rescue_world()
    grid = rasterise(world)
    goal = world.heat_sources[0].xy
    inflate = 0.45

    naive = plan(grid, start.xy, goal, inflate, clearance_weight=0.0, comfort=0.7)
    safe = plan(grid, start.xy, goal, inflate, clearance_weight=5.0, comfort=0.7)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(grid.prob, origin="lower", extent=(0, world.width, 0, world.height),
              cmap="gray_r", vmin=0, vmax=1)
    ax.plot(naive[:, 0], naive[:, 1], color="#f4a23b", lw=2.5,
            label="shortest A* (hugs corners)")
    ax.plot(safe[:, 0], safe[:, 1], color="#36c46a", lw=2.5,
            label="clearance-aware A* (stays centred)")
    ax.scatter([start.x], [start.y], marker="o", s=90, color="#3b8df4",
               edgecolors="white", zorder=5, label="start")
    ax.scatter([goal[0]], [goal[1]], marker="*", s=260, color="#ff5d3b",
               edgecolors="white", zorder=5, label="victim")
    ax.set_title("Global planning: shortest vs clearance-aware A*", fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("assets/planning.png", dpi=130, bbox_inches="tight")
    print("Wrote assets/planning.png")


if __name__ == "__main__":
    main()
