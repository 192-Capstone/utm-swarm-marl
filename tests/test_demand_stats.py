"""
test_demand_stats.py
--------------------
D4 demand (utm/demand.py) as a random generator: the fast versions of scripts/demand_stats.py
(same functions, fewer seeds). Seeds are fixed (from 100, never 10-14), so results don't flake;
each check needs p > 0.001, and the last test shows the same checks reject a wrong model.

  counts     mean and variance of flights per run vs rate x duration x area (Poisson)
  gaps       inter-departure gaps vs exponential (KS)
  peak       departure times vs the rush-hour curve (chi-square)
  operators  operator shares; hub_share (chi-square)
  spread     Region.sample per map quarter vs the region's share (chi-square); generated starts
             close to it

Run: python -m pytest tests/test_demand_stats.py -q          (full numbers: scripts/demand_stats.py)
"""

import os
import sys

import numpy as np
import pytest

pytest.importorskip('h3')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.demand_stats import (load_maps, stat_counts, stat_gaps, stat_hub_share,  # noqa: E402
                                  stat_operators, stat_peak, stat_spread)
from utm.demand import departure_times                                             # noqa: E402
from utm.demand_checks import chi_square, ks_exponential, safe_seeds              # noqa: E402

P_MIN = 0.001


@pytest.fixture(scope='module')
def maps():
    return load_maps(('empty', 'empty-wide', 'medium', 'crop'))


def test_flight_count_is_poisson_in_rate_duration_and_area(maps):
    rows = stat_counts(maps, safe_seeds(60), rates=(5, 40), duration=60.0, names=('empty', 'empty-wide'))
    for r in rows:
        assert r['p_mean'] > P_MIN and r['p_var'] > P_MIN, r
    # twice the area, twice the flights
    by = {(r['map'], r['rate']): r['expected'] for r in rows}
    assert by[('empty-wide', 40)] == pytest.approx(2 * by[('empty', 40)])


def test_gaps_are_exponential():
    g = stat_gaps(safe_seeds(30))
    assert g['p'] > P_MIN, g
    assert g['mean_gap'] == pytest.approx(g['expected_gap'], rel=0.02)


def test_departures_follow_the_peak(maps):
    pk = stat_peak(maps, safe_seeds(150))
    assert pk['p'] > P_MIN, (pk['observed'], pk['expected'].round(1))


def test_operator_and_hub_shares(maps):
    op = stat_operators(maps, safe_seeds(40))
    assert op['p'] > P_MIN, op
    hs = stat_hub_share(maps, safe_seeds(40))
    assert hs['p'] > P_MIN, hs


def test_pads_spread_evenly_over_the_free_region(maps):
    for r in stat_spread(maps, safe_seeds(15), draws=8000, names=('empty', 'medium', 'crop')):
        assert r['p_sample'] > P_MIN, r
        assert r['max_diff'] < 0.05, r                         # starts: close to, not exactly, uniform


def test_the_checks_reject_a_wrong_model():
    """Power: the same tests clearly reject what the generator does not do."""
    rng = np.random.RandomState(100)
    gaps = np.concatenate([np.diff(departure_times(rng, 20.0, 3600.0)) for _ in range(5)])
    assert ks_exponential(gaps, 20.0 / 60.0 * 1.1)[1] < 1e-6               # 10% wrong rate
    assert chi_square([520, 480], [600, 400])[2] < 1e-6                    # 0.52 vs 0.6
    even = rng.uniform(0, 120, 2000)
    assert chi_square(np.histogram(even, 12)[0], np.full(12, 2000 / 12) * np.linspace(0.7, 1.3, 12))[2] < 1e-6
