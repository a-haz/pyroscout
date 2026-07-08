"""The autonomy brain: sensor fusion + a behaviour state machine.

This module wires every other layer together into a closed perception-plan-act
loop and decides *what the robot should be doing* at each instant:

1. **Sense.**  Take a LIDAR scan (update the map) and a thermal reading.
2. **Fuse.**  LIDAR gives the *map*; the thermal sensor gives the *goal*.  A
   thermal detection is projected into the world and smoothed over time into a
   victim-position estimate.
3. **Decide.**
   * ``SEARCHING`` — the victim has not been seen yet, so drive towards the
     nearest reachable *frontier* (free space bordering the unknown) to expand
     the map.  If the map is complete and the victim was still never seen (a
     short-range or badly occluded thermal sensor), fall back to *coverage
     search*: sweep the sensor over all reachable free space, spinning in
     place at each waypoint so the forward-facing sensor eventually looks
     everywhere a victim could hide.
   * ``NAVIGATING`` — the victim has been seen, so plan an A* path to the
     estimate and follow it; if walls still block the route, keep exploring in
     the victim's direction until one opens up.
   * ``REACHED`` / ``FAILED`` — terminal states.
4. **Act.**  Convert the active path into velocity commands with pure pursuit,
   with a reactive safety brake as a backstop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .control import PurePursuit
from .geometry import Pose, wrap_to_pi
from .mapping import OccupancyGrid
from .mapping.occupancy_grid import bresenham
from .planning import astar
from .robot import DiffDriveRobot
from .sensors import Lidar2D, ThermalSensor
from .sensors.thermal import ThermalDetection
from .world import World


class NavState(Enum):
    SEARCHING = "searching"
    NAVIGATING = "navigating"
    REACHED = "reached"
    FAILED = "failed"


@dataclass
class StepRecord:
    """A snapshot of one control step, used for visualisation and metrics."""

    pose: Pose
    state: NavState
    ranges: np.ndarray
    beam_angles: np.ndarray
    grid_prob: np.ndarray
    planned_path: np.ndarray | None
    goal_estimate: np.ndarray | None
    detections: list[ThermalDetection]


@dataclass
class NavResult:
    success: bool
    state: NavState
    steps: int
    sim_time: float
    path_length: float
    collisions: int
    goal_estimate: np.ndarray | None
    history: list[StepRecord] = field(default_factory=list, repr=False)


def _nearest_free(blocked: np.ndarray, cell: tuple[int, int], max_radius: int = 40):
    """Closest non-blocked cell to ``cell`` (a ring-by-ring spiral search)."""
    cx, cy = cell
    h, w = blocked.shape
    if 0 <= cx < w and 0 <= cy < h and not blocked[cy, cx]:
        return cell
    for r in range(1, max_radius + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and not blocked[ny, nx]:
                    return nx, ny
    return None


class Navigator:
    """Closed-loop autonomous navigator for one robot in one world."""

    def __init__(
        self,
        robot: DiffDriveRobot,
        world: World,
        lidar: Lidar2D,
        thermal: ThermalSensor,
        grid: OccupancyGrid | None = None,
        controller: PurePursuit | None = None,
        dt: float = 0.1,
        resolution: float = 0.1,
        safety_margin: float = 0.1,
        replan_every: int = 5,
        reach_radius: float = 0.5,
        goal_filter_alpha: float = 0.3,
        comfort: float = 0.7,
        clearance_weight: float = 5.0,
        coverage_search: bool = True,
        coverage_res: float = 0.5,
    ):
        self.robot = robot
        self.world = world
        self.lidar = lidar
        self.thermal = thermal
        self.grid = grid or OccupancyGrid(world.width, world.height, resolution)
        self.controller = controller or PurePursuit(
            omega_max=robot.omega_max, v_nominal=min(0.7, robot.v_max)
        )
        self.dt = float(dt)
        self.inflate = robot.radius + safety_margin
        self.replan_every = int(replan_every)
        self.reach_radius = float(reach_radius)
        self.goal_filter_alpha = float(goal_filter_alpha)
        self.comfort = float(comfort)
        self.clearance_weight = float(clearance_weight)
        self.coverage_search = bool(coverage_search)
        self._cov_res = float(coverage_res)
        # How far a coverage sweep trusts the thermal sensor: slightly inside
        # its true range, and capped so the per-step bookkeeping stays cheap
        # when the range is large (that regime never needs coverage search).
        self.sweep_radius = max(0.8, min(3.0, 0.9 * self.thermal.max_range))

        self.state = NavState.SEARCHING
        self.victim_seen = False
        self.goal_estimate: np.ndarray | None = None
        self.current_path: np.ndarray | None = None
        self.explore_target: np.ndarray | None = None
        self._steps_since_plan = 0
        self._stuck_counter = 0
        self._visited_targets: list[np.ndarray] = []
        self._swept: set[tuple[int, int]] = set()
        self._coverage_target: np.ndarray | None = None
        self._periscope_remaining = 0.0
        self._spin_bin: tuple[int, int] | None = None

    # --- fusion ---------------------------------------------------------------
    def _update_goal_estimate(self, detections: list[ThermalDetection]) -> None:
        if not detections:
            return
        strongest = max(detections, key=lambda d: d.intensity)
        est = strongest.estimated_position(self.robot.pose)
        if self.goal_estimate is None:
            self.goal_estimate = est
        else:
            a = self.goal_filter_alpha
            self.goal_estimate = (1 - a) * self.goal_estimate + a * est
        self.victim_seen = True

    # --- planning helpers -----------------------------------------------------
    def _plan_to(self, goal_xy: np.ndarray) -> np.ndarray | None:
        blocked = self.grid.costmap(inflate_radius=self.inflate)
        start = _nearest_free(blocked, self.grid.world_to_grid(self.robot.x, self.robot.y))
        goal = _nearest_free(blocked, self.grid.world_to_grid(*goal_xy))
        if start is None or goal is None:
            return None
        # Clearance penalty: prefer staying away from walls.
        clearance = self.grid.clearance_m()
        cost_grid = self.clearance_weight * np.maximum(0.0, self.comfort - clearance)
        cells = astar(blocked, start, goal, cost_grid=cost_grid)
        if not cells:
            return None
        return np.array([self.grid.grid_to_world(cx, cy) for cx, cy in cells])

    def _frontier_clusters(self) -> tuple[np.ndarray, np.ndarray]:
        """Cluster raw frontier cells into candidate goal points and sizes.

        Frontier *cells* are noisy and numerous; binning them into ~0.6 m cells
        and taking centroids yields a few stable, meaningful targets and throws
        away single-cell speckle.  The cluster *size* (cell count) is a proxy
        for how much new space lies beyond it — a whole doorway into an unmapped
        room is a big cluster, a leftover pocket is a small one.
        """
        frontiers = self.grid.find_frontiers()
        if not frontiers:
            return np.empty((0, 2)), np.empty((0,))
        pts = np.array([self.grid.grid_to_world(cx, cy) for cx, cy in frontiers])

        bin_size = 0.6
        keys = np.floor(pts / bin_size).astype(int)
        buckets: dict[tuple[int, int], list[np.ndarray]] = {}
        for p, k in zip(pts, map(tuple, keys), strict=True):
            buckets.setdefault(k, []).append(p)

        groups = [ps for ps in buckets.values() if len(ps) >= 2]
        if not groups:  # fall back to raw points if everything is speckle
            groups = list(buckets.values())
        centroids = np.array([np.mean(ps, axis=0) for ps in groups])
        sizes = np.array([len(ps) for ps in groups], dtype=float)
        return centroids, sizes

    def _choose_frontier_target(self, bias_xy: np.ndarray | None) -> np.ndarray | None:
        """Pick the best reachable frontier cluster we haven't just visited.

        Utility trades off proximity against cluster size (bigger = more new
        space to gain).  ``bias_xy`` (the victim estimate, when navigating but
        blocked) replaces "proximity to the robot" with "proximity to the goal",
        so the robot maps *around* the wall in its way rather than wandering off.
        """
        centroids, sizes = self._frontier_clusters()
        if len(centroids) == 0:
            return None

        ref = bias_xy if bias_xy is not None else self.robot.pose.xy
        dist = np.hypot(centroids[:, 0] - ref[0], centroids[:, 1] - ref[1])
        utility = dist - 0.15 * sizes  # lower is better
        order = np.argsort(utility)

        fallback = None
        for idx in order:
            cand = centroids[idx]
            if self._plan_to(cand) is None:
                continue
            if fallback is None:
                fallback = cand  # first reachable, even if recently visited
            if any(np.hypot(*(cand - v)) < 0.7 for v in self._visited_targets):
                continue  # prefer somewhere new
            return cand
        return fallback

    def _target_explored(self, target: np.ndarray) -> bool:
        """True once no frontier remains near ``target`` (area is mapped)."""
        centroids, _ = self._frontier_clusters()
        if len(centroids) == 0:
            return True
        return bool(
            np.min(np.hypot(centroids[:, 0] - target[0], centroids[:, 1] - target[1]))
            > 0.8
        )

    def _plan_exploration(self) -> None:
        """Frontier exploration with commitment: keep a target until reached."""
        pose_xy = self.robot.pose.xy
        need_new = (
            self.explore_target is None
            or np.hypot(*(self.explore_target - pose_xy)) < 0.6
            or self._target_explored(self.explore_target)
        )
        if need_new:
            self.explore_target = self._choose_frontier_target(bias_xy=None)
            if self.explore_target is not None:
                self._visited_targets.append(self.explore_target)
                self._visited_targets = self._visited_targets[-25:]

        if self.explore_target is None:
            self.current_path = None
            return
        path = self._plan_to(self.explore_target)
        if path is None:
            self.explore_target = None
        self.current_path = path

    # --- coverage search (frontier-exhausted fallback) ------------------------
    def _bin_of(self, x: float, y: float) -> tuple[int, int]:
        return int(x // self._cov_res), int(y // self._cov_res)

    def _cell_visible(self, rx: int, ry: int, gx: int, gy: int) -> bool:
        """Line of sight between two *grid* cells on the believed map.

        Unknown cells block sight: the sensor cannot vouch for space the map
        has never observed, so it is not credited with covering it.
        """
        prob = self.grid.prob
        known = self.grid.known_mask
        for px, py in bresenham(rx, ry, gx, gy):
            if not self.grid.in_bounds(px, py):
                return False
            if not known[py, px] or prob[py, px] > 0.65:
                return False
        return True

    def _mark_swept(self) -> None:
        """Credit the coverage cells the thermal sensor genuinely covered.

        A coverage cell is marked swept only if a victim at its centre would
        have produced a detection from the current pose: within
        ``sweep_radius``, inside the forward field of view, and with line of
        sight on the believed map.  Cells whose centre the map shows as
        occupied are vacuously swept — a victim cannot be inside a wall.
        """
        pose = self.robot.pose
        r = self.sweep_radius
        res = self._cov_res
        half_fov = self.thermal.fov / 2.0
        rx, ry = self.grid.world_to_grid(pose.x, pose.y)
        prob = self.grid.prob
        known = self.grid.known_mask

        for bx in range(int((pose.x - r) // res), int((pose.x + r) // res) + 1):
            for by in range(int((pose.y - r) // res), int((pose.y + r) // res) + 1):
                if (bx, by) in self._swept:
                    continue
                wx, wy = (bx + 0.5) * res, (by + 0.5) * res
                dx, dy = wx - pose.x, wy - pose.y
                if dx * dx + dy * dy > r * r:
                    continue
                gx, gy = self.grid.world_to_grid(wx, wy)
                if not self.grid.in_bounds(gx, gy):
                    continue
                if known[gy, gx] and prob[gy, gx] > 0.65:
                    self._swept.add((bx, by))
                    continue
                rel = wrap_to_pi(math.atan2(dy, dx) - pose.theta).item()
                if abs(rel) > half_fov:
                    continue
                if self._cell_visible(rx, ry, gx, gy):
                    self._swept.add((bx, by))

    def _unswept_nearby(self) -> bool:
        """Is any known-free coverage cell within sweep radius still unswept?"""
        pose = self.robot.pose
        r = self.sweep_radius
        res = self._cov_res
        prob = self.grid.prob
        known = self.grid.known_mask
        for bx in range(int((pose.x - r) // res), int((pose.x + r) // res) + 1):
            for by in range(int((pose.y - r) // res), int((pose.y + r) // res) + 1):
                if (bx, by) in self._swept:
                    continue
                wx, wy = (bx + 0.5) * res, (by + 0.5) * res
                if (wx - pose.x) ** 2 + (wy - pose.y) ** 2 > r * r:
                    continue
                gx, gy = self.grid.world_to_grid(wx, wy)
                if self.grid.in_bounds(gx, gy) and known[gy, gx] and prob[gy, gx] < 0.35:
                    return True
        return False

    def _choose_coverage_target(self) -> np.ndarray | None:
        """Nearest reachable known-free spot the thermal sensor has not swept.

        Bins observed free space into ``coverage_res`` cells and returns the
        closest one not yet swept, so the robot lawn-mows the (short-range,
        forward-facing) sensor over the whole building instead of giving up the
        moment frontiers run out.
        """
        free = self.grid.known_mask & (self.grid.prob < 0.35)
        ys, xs = np.nonzero(free)
        if xs.size == 0:
            return None
        res = self._cov_res
        wx = (xs + 0.5) * self.grid.resolution
        wy = (ys + 0.5) * self.grid.resolution
        keys = np.stack([np.floor(wx / res), np.floor(wy / res)], axis=1).astype(int)
        uncovered = [tuple(b) for b in np.unique(keys, axis=0) if tuple(b) not in self._swept]
        if not uncovered:
            return None
        rx, ry = self.robot.pose.xy
        centers = np.array([[(cx + 0.5) * res, (cy + 0.5) * res] for cx, cy in uncovered])
        order = np.argsort(np.hypot(centers[:, 0] - rx, centers[:, 1] - ry))
        for idx in order[:40]:
            cand = centers[idx]
            if self._plan_to(cand) is not None:
                return cand
        return None

    def _plan_coverage(self) -> None:
        """Sweep the thermal sensor over all reachable free space.

        Waypoints are unswept coverage cells.  On arrival the robot performs a
        full in-place "periscope" spin (skipped when everything nearby is
        already swept) so the forward-only sensor faces every direction, then
        the waypoint is retired even if its centre never became visible
        (corner shadows) — guaranteeing the search always makes progress.
        """
        if self._periscope_remaining > 1e-3:
            self.current_path = None  # hold position while the spin finishes
            return
        if self._spin_bin is not None:
            # Spin finished: retire the waypoint no matter what it revealed.
            self._swept.add(self._spin_bin)
            self._spin_bin = None

        pose_xy = self.robot.pose.xy
        if (
            self._coverage_target is not None
            and np.hypot(*(self._coverage_target - pose_xy)) < 0.6
        ):
            # Arrived: sweep here if anything nearby still needs covering.
            self._spin_bin = self._bin_of(*self._coverage_target)
            if self._unswept_nearby():
                self._periscope_remaining = 2.0 * math.pi
            self._coverage_target = None
            self.current_path = None
            return
        if (
            self._coverage_target is None
            or self._bin_of(*self._coverage_target) in self._swept
        ):
            # No target, or the drive-by cone already swept it: pick a new one.
            self._coverage_target = self._choose_coverage_target()
        if self._coverage_target is None:
            self.current_path = None  # everything reachable swept -> give up
            return
        self.current_path = self._plan_to(self._coverage_target)
        if self.current_path is None:
            self._coverage_target = None

    # --- main loop ------------------------------------------------------------
    def step(self) -> StepRecord:
        pose = self.robot.pose

        # 1. Sense + map.
        scan = self.lidar.scan(pose, self.world)
        self.grid.update(scan)
        detections = self.thermal.sense(pose, self.world)
        self._update_goal_estimate(detections)
        if self.coverage_search:
            self._mark_swept()

        # 2. Terminal check: have we arrived at the victim estimate?
        if self.victim_seen and self.goal_estimate is not None:
            if np.hypot(*(self.goal_estimate - pose.xy)) < self.reach_radius:
                self.state = NavState.REACHED
                return self._record(pose, scan, detections)

        # 3. (Re)plan if needed.
        need_plan = (
            self.current_path is None
            or self._steps_since_plan >= self.replan_every
        )
        if need_plan:
            self._replan()

        # 4. Act.
        if self._periscope_remaining > 1e-3 and not self.victim_seen:
            # Bounded in-place sweep so the forward-only thermal FOV eventually
            # faces every direction at this coverage waypoint.
            self._stuck_counter = 0
            spin = min(self.robot.omega_max, self._periscope_remaining / self.dt)
            self.robot.step(0.0, spin, self.dt)
            self._periscope_remaining -= spin * self.dt
        elif self.current_path is None or len(self.current_path) == 0:
            # Nothing to follow: rotate in place to gather information.
            self._stuck_counter += 1
            if self._stuck_counter > 80:
                self.state = NavState.FAILED
            self.robot.step(0.0, 0.6 * self.robot.omega_max, self.dt)
        else:
            self._stuck_counter = 0
            v, omega, done = self.controller.control(pose, self.current_path)
            v, omega = self._apply_safety(scan, v, omega)
            self.robot.step(v, omega, self.dt)
            if done:
                # Reached the current sub-goal; force a fresh plan next step.
                self.current_path = None

        self._steps_since_plan += 1
        return self._record(self.robot.pose, scan, detections)

    def _replan(self) -> None:
        self._steps_since_plan = 0
        if self.victim_seen and self.goal_estimate is not None:
            self.state = NavState.NAVIGATING
            path = self._plan_to(self.goal_estimate)
            if path is None:
                # Goal not yet reachable on the known map: explore towards it.
                target = self._choose_frontier_target(bias_xy=self.goal_estimate)
                path = self._plan_to(target) if target is not None else None
            self.current_path = path
        else:
            self.state = NavState.SEARCHING
            self._plan_exploration()
            if self.current_path is None and self.coverage_search:
                # Frontiers exhausted but victim never seen: sweep free space
                # rather than spinning in place until the stuck-timer fails.
                self._plan_coverage()

    def _apply_safety(self, scan, v: float, omega: float) -> tuple[float, float]:
        """Reactive brake: don't drive forward into something very close ahead."""
        front = np.abs(scan.angles) < math.radians(25)
        if np.any(front):
            min_front = float(np.min(scan.ranges[front]))
            if min_front < self.robot.radius + 0.12 and v > 0:
                v = 0.0  # keep omega so we can rotate away
        return v, omega

    def _record(self, pose, scan, detections) -> StepRecord:
        return StepRecord(
            pose=pose,
            state=self.state,
            ranges=scan.ranges.copy(),
            beam_angles=scan.angles.copy(),
            grid_prob=self.grid.prob.astype(np.float32),
            planned_path=None if self.current_path is None else self.current_path.copy(),
            goal_estimate=None if self.goal_estimate is None else self.goal_estimate.copy(),
            detections=list(detections),
        )

    # --- driver ---------------------------------------------------------------
    def run(self, max_steps: int = 600) -> NavResult:
        """Run the closed loop until the victim is reached or we give up."""
        history: list[StepRecord] = []
        path_length = 0.0
        collisions = 0
        prev = self.robot.pose.xy

        for _ in range(max_steps):
            rec = self.step()
            history.append(rec)

            now = self.robot.pose.xy
            path_length += float(np.hypot(*(now - prev)))
            prev = now
            if self.world.in_collision(now[0], now[1], self.robot.radius):
                collisions += 1

            if self.state in (NavState.REACHED, NavState.FAILED):
                break

        return NavResult(
            success=self.state == NavState.REACHED,
            state=self.state,
            steps=len(history),
            sim_time=len(history) * self.dt,
            path_length=path_length,
            collisions=collisions,
            goal_estimate=self.goal_estimate,
            history=history,
        )
