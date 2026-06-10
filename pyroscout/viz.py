"""Matplotlib visualisation: side-by-side "ground truth" vs "robot belief".

Everything here is rendered with the non-interactive *Agg* backend so it runs
headless (CI, servers, this container) and writes GIF/PNG files directly.

The left panel shows the true world the robot can never fully see; the right
panel shows the occupancy grid it has inferred from LIDAR.  Watching the right
panel fill in — and the planned path snake through it towards the heat estimate
— is the whole story of the project in one image.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: must be set before pyplot is imported.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from matplotlib.patches import Rectangle as MplRectangle

from .navigator import NavResult, NavState  # noqa: E402
from .world import World  # noqa: E402

_STATE_COLOUR = {
    NavState.SEARCHING: "#f4a23b",
    NavState.NAVIGATING: "#3b8df4",
    NavState.REACHED: "#36c46a",
    NavState.FAILED: "#e23b3b",
}


def _draw_truth(ax, world: World, rec, robot_radius: float) -> None:
    ax.clear()
    ax.set_title("Ground truth (what's really there)", fontsize=10)
    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_aspect("equal")
    ax.set_facecolor("#0d1b2a")

    for obs in world.obstacles:
        ax.add_patch(
            MplRectangle((obs.x, obs.y), obs.w, obs.h, facecolor="#415a77", edgecolor="#778da9")
        )

    # Heat sources, with a soft infrared "glow".
    for src in world.heat_sources:
        for radius, alpha in ((1.4, 0.10), (0.9, 0.16), (0.5, 0.26)):
            ax.add_patch(Circle((src.x, src.y), radius, color="#ff7b3b", alpha=alpha, zorder=3))
        ax.scatter([src.x], [src.y], marker="*", s=320, color="#ff5d3b",
                   edgecolors="white", linewidths=0.6, zorder=5)

    # LIDAR beams.
    pose = rec.pose
    ang = pose.theta + rec.beam_angles
    ends = np.stack(
        [pose.x + rec.ranges * np.cos(ang), pose.y + rec.ranges * np.sin(ang)], axis=1
    )
    segs = [[(pose.x, pose.y), (ex, ey)] for ex, ey in ends]
    ax.add_collection(LineCollection(segs, colors="#ffd166", linewidths=0.3, alpha=0.5))

    _draw_target_line(ax, pose, rec.goal_estimate)
    _draw_path(ax, rec.planned_path, "#36c46a")
    _draw_goal(ax, rec.goal_estimate)
    _draw_robot(ax, pose, robot_radius, _STATE_COLOUR[rec.state])
    _legend(ax)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_belief(ax, world: World, rec, robot_radius: float) -> None:
    ax.clear()
    ax.set_title("Robot's belief (LIDAR map + plan)", fontsize=10)
    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_aspect("equal")

    ax.imshow(
        rec.grid_prob,
        origin="lower",
        extent=(0, world.width, 0, world.height),
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    _draw_target_line(ax, rec.pose, rec.goal_estimate)
    _draw_path(ax, rec.planned_path, "#1b9e4b")
    _draw_goal(ax, rec.goal_estimate)
    _draw_robot(ax, rec.pose, robot_radius, _STATE_COLOUR[rec.state])
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_robot(ax, pose, radius, colour) -> None:
    ax.add_patch(Circle((pose.x, pose.y), radius, facecolor=colour, edgecolor="white",
                        linewidth=1.0, zorder=6))
    ax.plot(
        [pose.x, pose.x + 1.6 * radius * np.cos(pose.theta)],
        [pose.y, pose.y + 1.6 * radius * np.sin(pose.theta)],
        color="white", linewidth=1.4, zorder=7,
    )


def _draw_path(ax, path, colour) -> None:
    if path is not None and len(path) > 1:
        ax.plot(path[:, 0], path[:, 1], color=colour, linewidth=2.0, zorder=4)


def _draw_goal(ax, goal) -> None:
    if goal is not None:
        ax.scatter([goal[0]], [goal[1]], marker="X", s=110, color="#ff2eea",
                   edgecolors="white", linewidths=0.6, zorder=5)


def _draw_target_line(ax, pose, goal) -> None:
    """Dashed line from the robot to the fused thermal estimate."""
    if goal is not None:
        ax.plot([pose.x, goal[0]], [pose.y, goal[1]], color="#ff2eea",
                linewidth=1.0, linestyle="--", alpha=0.7, zorder=3)


_LEGEND_HANDLES = [
    Line2D([], [], color="#3b8df4", marker="o", linestyle="", markersize=7,
           markeredgecolor="white", label="Robot"),
    Line2D([], [], color="#ff5d3b", marker="*", linestyle="", markersize=11,
           markeredgecolor="white", label="Victim (heat)"),
    Line2D([], [], color="#ffd166", linewidth=2, label="LIDAR"),
    Line2D([], [], color="#36c46a", linewidth=2, label="Planned path"),
    Line2D([], [], color="#ff2eea", marker="X", linestyle="--", markersize=8,
           label="Heat estimate"),
]


def _legend(ax) -> None:
    leg = ax.legend(handles=_LEGEND_HANDLES, loc="upper left", fontsize=7,
                    framealpha=0.85, facecolor="#1b263b", edgecolor="none")
    for text in leg.get_texts():
        text.set_color("white")


def _frames(result: NavResult, stride: int, max_frames: int):
    idx = list(range(0, len(result.history), max(1, stride)))
    if idx and idx[-1] != len(result.history) - 1:
        idx.append(len(result.history) - 1)  # always show the final frame
    if len(idx) > max_frames:
        keep = np.linspace(0, len(idx) - 1, max_frames).round().astype(int)
        idx = [idx[i] for i in keep]
    return idx


def animate(
    result: NavResult,
    world: World,
    out_path: str = "assets/demo.gif",
    robot_radius: float = 0.25,
    fps: int = 12,
    stride: int = 2,
    max_frames: int = 160,
) -> str:
    """Render the run to an animated GIF and return the output path."""
    idx = _frames(result, stride, max_frames)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.6))
    dt = result.sim_time / max(result.steps, 1)

    def update(frame_i):
        rec = result.history[frame_i]
        _draw_truth(axL, world, rec, robot_radius)
        _draw_belief(axR, world, rec, robot_radius)
        fig.suptitle(
            f"PyroScout — {rec.state.value.upper()}   "
            f"t = {frame_i * dt:4.1f}s   step {frame_i + 1}/{result.steps}",
            fontsize=12, fontweight="bold",
        )

    anim = FuncAnimation(fig, update, frames=idx, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def save_snapshot(
    result: NavResult,
    world: World,
    out_path: str = "assets/hero.png",
    robot_radius: float = 0.25,
    frame: int = -1,
    dpi: int = 130,
) -> str:
    """Render a single frame to a PNG (defaults to the final frame)."""
    rec = result.history[frame]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.2))
    _draw_truth(axL, world, rec, robot_radius)
    _draw_belief(axR, world, rec, robot_radius)
    outcome = "REACHED" if result.success else result.state.value.upper()
    fig.suptitle(
        f"PyroScout — victim {outcome} in {result.sim_time:.1f}s, "
        f"{result.path_length:.1f} m travelled, {result.collisions} collisions",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
