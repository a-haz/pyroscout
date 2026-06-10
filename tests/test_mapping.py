import math

import numpy as np

from pyroscout.geometry import Pose
from pyroscout.mapping import OccupancyGrid
from pyroscout.mapping.occupancy_grid import bresenham, bresenham_batch
from pyroscout.sensors import Lidar2D
from pyroscout.world import Rectangle, World


def test_coordinate_roundtrip():
    g = OccupancyGrid(10, 10, 0.1)
    cx, cy = g.world_to_grid(3.14, 6.28)
    wx, wy = g.grid_to_world(cx, cy)
    assert abs(wx - 3.14) <= 0.1 and abs(wy - 6.28) <= 0.1
    assert g.in_bounds(0, 0) and not g.in_bounds(-1, 0)


def test_bresenham_endpoints():
    cells = bresenham(0, 0, 3, 1)
    assert cells[0] == (0, 0)
    assert cells[-1] == (3, 1)


def test_bresenham_batch_matches_scalar():
    rng = np.random.default_rng(7)
    targets = rng.integers(-25, 26, size=(60, 2))
    xs, ys, lengths = bresenham_batch(3, -2, targets)
    for i, (tx, ty) in enumerate(targets):
        expected = bresenham(3, -2, int(tx), int(ty))
        got = list(zip(xs[i, : lengths[i]].tolist(), ys[i, : lengths[i]].tolist(), strict=True))
        assert got == expected


def _scalar_reference_update(grid: OccupancyGrid, log_odds: np.ndarray, scan) -> None:
    """The original cell-at-a-time update, kept as the semantic reference."""
    rx, ry = grid.world_to_grid(scan.pose.x, scan.pose.y)
    for (ex, ey), hit in zip(scan.endpoints(), scan.hit_mask, strict=True):
        cx, cy = grid.world_to_grid(ex, ey)
        ray = bresenham(rx, ry, cx, cy)
        for px, py in ray[:-1]:
            if grid.in_bounds(px, py):
                log_odds[py, px] -= grid.l_free
        lx, ly = ray[-1]
        if grid.in_bounds(lx, ly):
            if hit:
                log_odds[ly, lx] += grid.l_occ
            else:
                log_odds[ly, lx] -= grid.l_free
    np.clip(log_odds, -grid.l_clamp, grid.l_clamp, out=log_odds)


def test_update_matches_scalar_reference():
    world = World(6, 6, obstacles=[Rectangle(2.5, 1.0, 1.0, 3.0)])
    # max_range > arena size so some beams time out; noise so endpoints can
    # land fractionally outside the grid — both paths must match the reference.
    lidar = Lidar2D(num_beams=90, fov=2 * math.pi, max_range=7.0, noise_std=0.05, seed=11)
    g = OccupancyGrid(6, 6, 0.1)
    reference = np.zeros_like(g.log_odds)
    for pose in (Pose(1.0, 1.0, 0.3), Pose(4.8, 5.2, -2.0), Pose(0.6, 5.4, 1.2)):
        scan = lidar.scan(pose, world)
        g.update(scan)
        _scalar_reference_update(g, reference, scan)
    np.testing.assert_allclose(g.log_odds, reference, atol=1e-9)


def test_update_marks_occupied_and_free():
    world = World(6, 6, obstacles=[Rectangle(3.9, 0.0, 0.2, 6.0)])
    lidar = Lidar2D(num_beams=180, fov=2 * math.pi, max_range=10.0, noise_std=0.0)
    g = OccupancyGrid(6, 6, 0.1)
    scan = lidar.scan(Pose(2.0, 3.0, 0.0), world)
    g.update(scan)

    # The +x beam (index 90) struck the wall; that endpoint cell is occupied.
    ex, ey = scan.endpoints()[90]
    wx, wy = g.world_to_grid(ex, ey)
    assert g.prob[wy, wx] > 0.6
    fx, fy = g.world_to_grid(3.0, 3.0)  # between robot and wall
    assert g.prob[fy, fx] < 0.4
    assert g.known_mask[fy, fx]


def test_clearance_transform():
    g = OccupancyGrid(2, 2, 0.1)  # 20x20 cells
    g.log_odds[10, 10] = 5.0  # one occupied cell
    clr = g.clearance_m()
    assert clr[10, 10] == 0.0
    assert clr[10, 15] > 0.4  # ~5 cells = 0.5 m away


def test_costmap_inflation():
    g = OccupancyGrid(2, 2, 0.1)
    g.log_odds[10, 10] = 5.0
    cm = g.costmap(occ_threshold=0.5, inflate_radius=0.25)  # ~2 cells
    assert cm[10, 10]
    assert cm[10, 12]
    assert not cm[10, 18]


def test_find_frontiers():
    g = OccupancyGrid(2, 2, 0.1)
    g.log_odds[:, :10] = -2.0  # left half observed-free, right half unknown
    xs = {c[0] for c in g.find_frontiers()}
    assert 9 in xs  # free cells bordering the unknown region
    assert 5 not in xs  # interior free cells are not frontiers


def test_nearest_free_cell():
    g = OccupancyGrid(2, 2, 0.1)
    g.log_odds[10, 10] = 5.0
    assert g.nearest_free_cell(10, 10) != (10, 10)
