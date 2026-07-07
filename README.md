# PyroScout

[![CI](https://github.com/a-haz/pyroscout/actions/workflows/ci.yml/badge.svg)](https://github.com/a-haz/pyroscout/actions/workflows/ci.yml)

A small 2D robotics simulator. A robot is dropped into a building it has never
seen and has to find a victim using two sensors: a LIDAR, which tells it where
the walls are, and a thermal sensor, which tells it where the heat is. Neither
is enough on its own — the LIDAR builds the map, the thermal sensor picks the
goal, and a behaviour state machine drives the robot from exploration to
arrival without touching a wall.

<p align="center">
  <img src="assets/demo.gif" width="100%" alt="PyroScout search-and-rescue demo">
</p>

The left panel is the ground truth the robot can't see: walls, a pillar, the
victim's heat signature (star), and the live LIDAR fan. The right panel is the
occupancy grid the robot infers from LIDAR, with its current plan drawn on top.
It starts almost entirely grey (unknown) and fills in as the robot explores.

There's a longer narrated version of all of this, with runnable code and
figures, in [`notebooks/pyroscout_writeup.ipynb`](notebooks/pyroscout_writeup.ipynb).

On the bundled scenario (seed 0) the robot starts blind in the left room with
the victim two offset doorways away, out of thermal line of sight. It reaches
the victim in about 47 simulated seconds with zero wall contacts, and the fused
thermal estimate lands about a centimetre from the true position. Other noise
seeds behave the same; the integration test in
[`tests/test_navigator.py`](tests/test_navigator.py) runs the full scenario as
part of the suite.

## How it works

The layout is the usual mobile-robot pipeline, one module per layer:
perception ([`sensors/`](pyroscout/sensors)) feeds mapping
([`mapping/occupancy_grid.py`](pyroscout/mapping/occupancy_grid.py)), which
feeds planning ([`planning/astar.py`](pyroscout/planning/astar.py)), which
feeds control ([`control/pure_pursuit.py`](pyroscout/control/pure_pursuit.py)),
with [`navigator.py`](pyroscout/navigator.py) making the decisions on top.

### Sensing

LIDAR is simulated by casting one ray per beam and finding the nearest wall
segment it strikes. All beams are tested against all walls in a single
vectorised linear-algebra solve (`cast_rays` in
[`geometry.py`](pyroscout/geometry.py)), with Gaussian noise added on top.

<p align="center"><img src="assets/lidar_scan.png" width="65%" alt="LIDAR scan"></p>

The thermal sensor is what makes this a search problem. Radiated power falls
off with the square of distance, so a source of strength `I0` at distance `d`
reads `I0 / (d**2 + 1)`. That's invertible, so a calibrated sensor can turn an
intensity reading back into a range estimate. Two effects make it non-trivial:
walls block infrared (checked with a ray cast), so the robot has to physically
get around corners before it can see the heat; and both bearing and intensity
are noisy, so the position estimate is smoothed over time with an exponential
filter.

### Mapping

The robot starts with no map. Each LIDAR scan updates a grid storing the
log-odds of occupancy per cell, which turns Bayesian updates into additions:
cells a beam passes through get "more free", the cell it lands in gets "more
occupied". From this map come two planning views — a costmap with obstacles
inflated by the robot radius (so a point planner yields body-safe paths), and a
clearance distance transform used to keep paths away from walls.

### Planning

A* finds the optimal path on the grid, with two details that matter for a
physical robot: diagonal moves that would clip a wall corner are forbidden, and
the search consumes a cost field that penalises cells close to walls, so it
prefers the middle of doorways. The difference is visible below — plain
shortest-path (orange) scrapes the corners, clearance-aware (green) stays
centred.

<p align="center"><img src="assets/planning.png" width="70%" alt="Clearance-aware planning"></p>

### Control

A pure-pursuit controller turns the path into `(v, omega)` commands by steering
toward a point a fixed lookahead distance ahead, rotating in place when badly
misaligned and slowing on the approach to the goal.

### Behaviour

The [`Navigator`](pyroscout/navigator.py) ties it together with a small state
machine. While SEARCHING it does frontier-based exploration: drive to the
boundary between mapped free space and the unknown. Frontier cells are
clustered to ignore speckle, and target selection trades off proximity against
cluster size (a doorway into a new room beats a leftover pocket), with
commitment and visited-memory to avoid oscillating. The moment the thermal
sensor gets line of sight it switches to NAVIGATING: plan to the fused victim
estimate and follow it, replanning as the map grows. If the victim isn't
reachable on the known map yet, it explores toward the estimate instead,
mapping around whatever is in the way. REACHED and FAILED (no frontiers left)
are terminal.

## Running it

```bash
python -m pip install -e .[dev]

# the search-and-rescue demo; writes assets/demo.gif and assets/hero.png
python examples/run_search_rescue.py            # or --seed 3, --no-gif

# the explainer figures
python examples/demo_lidar.py
python examples/demo_planning.py

pytest
ruff check .
```

As a library:

```python
from pyroscout.scenarios import demo_navigator

nav, world, robot = demo_navigator(seed=0)
result = nav.run(max_steps=700)
print(result.success, result.collisions, result.path_length)
```

## Layout

```
pyroscout/
├── geometry.py              # vectorised ray casting, angle math, poses
├── world.py                 # arena, rectangular obstacles, heat sources
├── robot.py                 # differential-drive kinematics
├── sensors/
│   ├── lidar.py             # 2D LIDAR ray-casting
│   └── thermal.py           # inverse-square heat model + occlusion + noise
├── mapping/
│   └── occupancy_grid.py    # log-odds map, frontiers, clearance, costmap
├── planning/
│   └── astar.py             # A* with corner-cut prevention + cost fields
├── control/
│   └── pure_pursuit.py      # path-following controller
├── navigator.py             # fusion + behaviour state machine + exploration
├── viz.py                   # matplotlib rendering / GIF export
└── scenarios.py             # the reproducible demo world
examples/                    # runnable demos that also produce the figures
tests/                       # unit + integration tests
```

## Notes

Everything here is classical robotics on purpose — each layer is a
well-understood algorithm you can read in an afternoon. Obvious extensions if
you want to take it further: a particle filter to drop the perfect-odometry
assumption (making it proper SLAM), a local planner like DWA for moving
obstacles, or feeding it real data (a 3D LIDAR projected to 2D plus a thermal
camera's detections maps onto the same architecture).

MIT licensed, see [LICENSE](LICENSE).
