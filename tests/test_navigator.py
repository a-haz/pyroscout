"""End-to-end integration tests for the full autonomy loop."""

import numpy as np
import pytest

from pyroscout.navigator import NavState
from pyroscout.scenarios import demo_navigator


def test_navigator_starts_searching():
    nav, _, _ = demo_navigator(seed=1)
    assert nav.state == NavState.SEARCHING
    assert not nav.victim_seen
    assert nav.goal_estimate is None


@pytest.mark.parametrize("seed", [0, 3])
def test_navigator_reaches_victim_without_collision(seed):
    nav, world, robot = demo_navigator(seed=seed)
    result = nav.run(max_steps=800)

    assert result.success
    assert result.state == NavState.REACHED
    assert result.collisions == 0

    # Thermal+LIDAR fusion should localise the victim to within 30 cm.
    err = float(((result.goal_estimate - world.heat_sources[0].xy) ** 2).sum() ** 0.5)
    assert err < 0.3

    # The robot should physically be next to the victim.
    final = robot.pose.xy
    assert np.hypot(*(final - world.heat_sources[0].xy)) < 0.8


def test_navigator_builds_a_map():
    nav, _, _ = demo_navigator(seed=0)
    nav.run(max_steps=800)
    # A meaningful fraction of the grid should have been observed.
    observed = nav.grid.known_mask.mean()
    assert observed > 0.3
